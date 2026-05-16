#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA-ANALYSIS] %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(_e));                                    \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#ifndef ANALYSIS_BLOCK_SIZE
#define ANALYSIS_BLOCK_SIZE 256
#endif

// ── Halo spin: angular momentum Lx/Ly/Lz reduction ────────────────────────
//
// Each thread computes m_i × (r_i - r_com) × (v_i - v_com) and accumulates
// via shared-memory reduction + global atomics.

__global__ void halo_spin_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    const float* __restrict__ mass,
    int n,
    float cx, float cy, float cz,
    float vcx, float vcy, float vcz,
    double* lx_global, double* ly_global, double* lz_global
) {
    __shared__ double slx[ANALYSIS_BLOCK_SIZE];
    __shared__ double sly[ANALYSIS_BLOCK_SIZE];
    __shared__ double slz[ANALYSIS_BLOCK_SIZE];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    double lx = 0.0, ly = 0.0, lz = 0.0;
    if (i < n) {
        double rx = (double)x[i] - (double)cx;
        double ry = (double)y[i] - (double)cy;
        double rz = (double)z[i] - (double)cz;
        double dvx = (double)vx[i] - (double)vcx;
        double dvy = (double)vy[i] - (double)vcy;
        double dvz = (double)vz[i] - (double)vcz;
        double m   = (double)mass[i];
        lx = m * (ry * dvz - rz * dvy);
        ly = m * (rz * dvx - rx * dvz);
        lz = m * (rx * dvy - ry * dvx);
    }
    slx[tid] = lx; sly[tid] = ly; slz[tid] = lz;
    __syncthreads();

    // Parallel reduction in shared memory.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            slx[tid] += slx[tid + s];
            sly[tid] += sly[tid + s];
            slz[tid] += slz[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(lx_global, slx[0]);
        atomicAdd(ly_global, sly[0]);
        atomicAdd(lz_global, slz[0]);
    }
}

// ── Galaxy luminosity: total L + weighted B-V + g-r ───────────────────────
//
// Simplified BC03 analytic SSP:
//   L = M × age^{-0.8} × max(1 + 2.5*log10(max(Z,4e-4)/0.02), 0.1)
//   B-V = 0.35 + 0.25*log10(age+0.01) + 0.10*log10(max(Z,1e-3))
//   g-r  = 0.24 + 0.18*log10(age+0.01) + 0.07*log10(max(Z,1e-3))
// Only Star particles (ptype == 2) contribute.

__global__ void galaxy_luminosity_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ mass,
    const float* __restrict__ age_gyr,
    const float* __restrict__ metallicity,
    int n,
    double* l_total_g,
    double* bv_weighted_g,
    double* gr_weighted_g,
    int*    n_stars_g
) {
    __shared__ double sl[ANALYSIS_BLOCK_SIZE];
    __shared__ double sbv[ANALYSIS_BLOCK_SIZE];
    __shared__ double sgr[ANALYSIS_BLOCK_SIZE];
    __shared__ int    sn[ANALYSIS_BLOCK_SIZE];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    double l = 0.0, bv_w = 0.0, gr_w = 0.0;
    int ns = 0;
    if (i < n && ptype[i] == 2) {
        float age = fmaxf(age_gyr[i], 1.0e-4f);
        float z   = metallicity[i];
        float m   = mass[i];
        float z_safe = fmaxf(z, 4.0e-4f);
        float f_z = 1.0f + 2.5f * log10f(z_safe / 0.02f);
        if (f_z < 0.1f) f_z = 0.1f;
        double l_i = (double)m * pow((double)age, -0.8) * (double)f_z;
        float log_age = log10f(fmaxf(age, 1.0e-3f) + 0.01f);
        float log_z   = log10f(fmaxf(z, 1.0e-3f));
        float bv = fminf(fmaxf(0.35f + 0.25f * log_age + 0.10f * log_z, -0.3f), 1.5f);
        float gr = fminf(fmaxf(0.24f + 0.18f * log_age + 0.07f * log_z, -0.2f), 1.2f);
        l = l_i;
        bv_w = l_i * (double)bv;
        gr_w = l_i * (double)gr;
        ns = 1;
    }
    sl[tid] = l; sbv[tid] = bv_w; sgr[tid] = gr_w; sn[tid] = ns;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sl[tid]  += sl[tid + s];
            sbv[tid] += sbv[tid + s];
            sgr[tid] += sgr[tid + s];
            sn[tid]  += sn[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(l_total_g, sl[0]);
        atomicAdd(bv_weighted_g, sbv[0]);
        atomicAdd(gr_weighted_g, sgr[0]);
        atomicAdd(n_stars_g, sn[0]);
    }
}

// ── X-ray luminosity: bremsstrahlung emissivity × mass ─────────────────────
//
// L_X = Σ Λ_0 × ρ² × √T × m_i  (gas only, T > T_min)
// ρ = m / h³ (smoothing-length proxy), T = u × (γ-1) / (k_B / (m_H μ))

static constexpr double KB_OVER_MH_MU = 8.254e-3 / 0.6;
static constexpr double LAMBDA_0_X    = 3.0e-27;
static constexpr double T_X_MIN_K     = 1.0e5;

__global__ void xray_luminosity_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ mass,
    const float* __restrict__ h_sml,
    const float* __restrict__ internal_energy,
    int n,
    float gamma,
    double* lx_global
) {
    __shared__ double slx[ANALYSIS_BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    double lx = 0.0;
    if (i < n && ptype[i] == 1) { // Gas
        float h = h_sml[i];
        float m = mass[i];
        if (h > 0.0f) {
            double rho = (double)m / ((double)h * (double)h * (double)h);
            double t_k = (double)internal_energy[i] * ((double)gamma - 1.0) / KB_OVER_MH_MU;
            if (t_k >= T_X_MIN_K && rho > 0.0) {
                lx = LAMBDA_0_X * rho * rho * sqrt(t_k) * (double)m;
            }
        }
    }
    slx[tid] = lx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) slx[tid] += slx[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(lx_global, slx[0]);
}

// ── Launchers ─────────────────────────────────────────────────────────────

template <typename T>
static int alloc_copy(T** d, const T* h, int n) {
    size_t bytes = (size_t)n * sizeof(T);
    CUDA_CHECK(cudaMalloc(d, bytes));
    CUDA_CHECK(cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int cuda_analysis_halo_spin(
    const float* x, const float* y, const float* z,
    const float* vx, const float* vy, const float* vz,
    const float* mass, int n,
    float cx, float cy, float cz,
    float vcx, float vcy, float vcz,
    double* lx_out, double* ly_out, double* lz_out
) {
    if (n <= 0) { *lx_out = *ly_out = *lz_out = 0.0; return 0; }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return -1;

    float *dx, *dy, *dz, *dvx, *dvy, *dvz, *dm;
    double *dlx, *dly, *dlz;
    if (alloc_copy(&dx, x, n) || alloc_copy(&dy, y, n) || alloc_copy(&dz, z, n) ||
        alloc_copy(&dvx, vx, n) || alloc_copy(&dvy, vy, n) || alloc_copy(&dvz, vz, n) ||
        alloc_copy(&dm, mass, n)) return -1;

    CUDA_CHECK(cudaMalloc(&dlx, sizeof(double))); CUDA_CHECK(cudaMemset(dlx, 0, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dly, sizeof(double))); CUDA_CHECK(cudaMemset(dly, 0, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dlz, sizeof(double))); CUDA_CHECK(cudaMemset(dlz, 0, sizeof(double)));

    int blocks = (n + ANALYSIS_BLOCK_SIZE - 1) / ANALYSIS_BLOCK_SIZE;
    halo_spin_kernel<<<blocks, ANALYSIS_BLOCK_SIZE>>>(
        dx, dy, dz, dvx, dvy, dvz, dm, n,
        cx, cy, cz, vcx, vcy, vcz,
        dlx, dly, dlz
    );
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(lx_out, dlx, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ly_out, dly, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lz_out, dlz, sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dm);
    cudaFree(dlx); cudaFree(dly); cudaFree(dlz);
    return 0;
}

extern "C" int cuda_analysis_luminosity(
    const unsigned char* ptype, const float* mass,
    const float* age_gyr, const float* metallicity, int n,
    double* l_total_out, double* bv_weighted_out, double* gr_weighted_out, int* n_stars_out
) {
    if (n <= 0) { *l_total_out = *bv_weighted_out = *gr_weighted_out = 0.0; *n_stars_out = 0; return 0; }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return -1;

    unsigned char* dpt;
    float *dm, *da, *dz;
    double *dl, *dbv, *dgr;
    int *dns;
    if (alloc_copy(&dpt, ptype, n) || alloc_copy(&dm, mass, n) ||
        alloc_copy(&da, age_gyr, n) || alloc_copy(&dz, metallicity, n)) return -1;

    CUDA_CHECK(cudaMalloc(&dl,  sizeof(double))); CUDA_CHECK(cudaMemset(dl,  0, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dbv, sizeof(double))); CUDA_CHECK(cudaMemset(dbv, 0, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dgr, sizeof(double))); CUDA_CHECK(cudaMemset(dgr, 0, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dns, sizeof(int)));    CUDA_CHECK(cudaMemset(dns, 0, sizeof(int)));

    int blocks = (n + ANALYSIS_BLOCK_SIZE - 1) / ANALYSIS_BLOCK_SIZE;
    galaxy_luminosity_kernel<<<blocks, ANALYSIS_BLOCK_SIZE>>>(
        dpt, dm, da, dz, n, dl, dbv, dgr, dns
    );
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(l_total_out,     dl,  sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bv_weighted_out, dbv, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gr_weighted_out, dgr, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_stars_out,     dns, sizeof(int),    cudaMemcpyDeviceToHost));

    cudaFree(dpt); cudaFree(dm); cudaFree(da); cudaFree(dz);
    cudaFree(dl); cudaFree(dbv); cudaFree(dgr); cudaFree(dns);
    return 0;
}

extern "C" int cuda_analysis_xray(
    const unsigned char* ptype, const float* mass,
    const float* h_sml, const float* internal_energy,
    int n, float gamma,
    double* lx_out
) {
    if (n <= 0) { *lx_out = 0.0; return 0; }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return -1;

    unsigned char* dpt;
    float *dm, *dh, *du;
    double *dlx;
    if (alloc_copy(&dpt, ptype, n) || alloc_copy(&dm, mass, n) ||
        alloc_copy(&dh, h_sml, n) || alloc_copy(&du, internal_energy, n)) return -1;

    CUDA_CHECK(cudaMalloc(&dlx, sizeof(double))); CUDA_CHECK(cudaMemset(dlx, 0, sizeof(double)));

    int blocks = (n + ANALYSIS_BLOCK_SIZE - 1) / ANALYSIS_BLOCK_SIZE;
    xray_luminosity_kernel<<<blocks, ANALYSIS_BLOCK_SIZE>>>(dpt, dm, dh, du, n, gamma, dlx);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(lx_out, dlx, sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dpt); cudaFree(dm); cudaFree(dh); cudaFree(du); cudaFree(dlx);
    return 0;
}
