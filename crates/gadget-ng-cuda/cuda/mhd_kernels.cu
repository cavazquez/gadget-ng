#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA-MHD] %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                    \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#ifndef MHD_BLOCK_SIZE
#define MHD_BLOCK_SIZE 256
#endif

static constexpr unsigned char PTYPE_GAS = 1;
static constexpr float MU0_MHD = 1.0f;

__global__ void mhd_flux_freeze_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ mass,
    const float* __restrict__ internal_energy,
    const float* __restrict__ h_sml,
    const float* __restrict__ bx_in,
    const float* __restrict__ by_in,
    const float* __restrict__ bz_in,
    float* __restrict__ bx_out,
    float* __restrict__ by_out,
    float* __restrict__ bz_out,
    int n, float gamma, float beta_freeze, float rho_ref
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float bx = bx_in[i], by = by_in[i], bz = bz_in[i];
    if (ptype[i] == PTYPE_GAS) {
        float b2 = bx * bx + by * by + bz * bz;
        if (b2 >= 1.0e-30f && rho_ref > 0.0f) {
            float h = fmaxf(h_sml[i], 1.0e-10f);
            float rho = fmaxf(mass[i] / (h * h * h), 1.0e-30f);
            float p_th = (gamma - 1.0f) * rho * internal_energy[i];
            float beta = 2.0f * MU0_MHD * p_th / b2;
            if (beta > beta_freeze) {
                float scale = powf(rho / rho_ref, 2.0f / 3.0f);
                bx *= scale; by *= scale; bz *= scale;
            }
        }
    }
    bx_out[i] = bx; by_out[i] = by; bz_out[i] = bz;
}

__global__ void mhd_density_contrib_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ mass,
    const float* __restrict__ h_sml,
    float* __restrict__ rho_out,
    float* __restrict__ count_out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ptype[i] == PTYPE_GAS) {
        float h = fmaxf(h_sml[i], 1.0e-10f);
        rho_out[i] = mass[i] / (h * h * h);
        count_out[i] = 1.0f;
    } else {
        rho_out[i] = 0.0f;
        count_out[i] = 0.0f;
    }
}

__global__ void mhd_b_stats_contrib_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ mass,
    const float* __restrict__ bx,
    const float* __restrict__ by,
    const float* __restrict__ bz,
    float* __restrict__ m_out,
    float* __restrict__ mb_out,
    float* __restrict__ mb2_out,
    float* __restrict__ bmag_out,
    float* __restrict__ emag_out,
    float* __restrict__ count_out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ptype[i] == PTYPE_GAS) {
        float b2 = bx[i] * bx[i] + by[i] * by[i] + bz[i] * bz[i];
        float bmag = sqrtf(b2);
        m_out[i] = mass[i];
        mb_out[i] = mass[i] * bmag;
        mb2_out[i] = mass[i] * b2;
        bmag_out[i] = bmag;
        emag_out[i] = mass[i] * b2 / (2.0f * MU0_MHD);
        count_out[i] = 1.0f;
    } else {
        m_out[i] = mb_out[i] = mb2_out[i] = bmag_out[i] = emag_out[i] = count_out[i] = 0.0f;
    }
}

template <typename T>
static int alloc_copy(T** d, const T* h, int n) {
    size_t bytes = (size_t)n * sizeof(T);
    CUDA_CHECK(cudaMalloc(d, bytes));
    CUDA_CHECK(cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice));
    return 0;
}

template <typename T>
static int alloc_zero(T** d, int n) {
    size_t bytes = (size_t)n * sizeof(T);
    CUDA_CHECK(cudaMalloc(d, bytes));
    CUDA_CHECK(cudaMemset(*d, 0, bytes));
    return 0;
}

extern "C" int cuda_mhd_flux_freeze(
    const unsigned char* ptype, const float* mass, const float* internal_energy,
    const float* h_sml, const float* bx_in, const float* by_in, const float* bz_in,
    float* bx_out, float* by_out, float* bz_out,
    int n, float gamma, float beta_freeze, float rho_ref
) {
    if (n <= 0) return 0;
    unsigned char* dptype;
    float *dmass, *du, *dh, *dbx, *dby, *dbz, *dobx, *doby, *dobz;
    if (alloc_copy(&dptype, ptype, n) || alloc_copy(&dmass, mass, n) ||
        alloc_copy(&du, internal_energy, n) || alloc_copy(&dh, h_sml, n) ||
        alloc_copy(&dbx, bx_in, n) || alloc_copy(&dby, by_in, n) || alloc_copy(&dbz, bz_in, n) ||
        alloc_zero(&dobx, n) || alloc_zero(&doby, n) || alloc_zero(&dobz, n)) return -1;
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_flux_freeze_kernel<<<blocks, MHD_BLOCK_SIZE>>>(dptype, dmass, du, dh, dbx, dby, dbz, dobx, doby, dobz, n, gamma, beta_freeze, rho_ref);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(bx_out, dobx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(by_out, doby, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bz_out, dobz, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dmass); cudaFree(du); cudaFree(dh); cudaFree(dbx); cudaFree(dby); cudaFree(dbz);
    cudaFree(dobx); cudaFree(doby); cudaFree(dobz);
    return 0;
}

extern "C" int cuda_mhd_density_contrib(
    const unsigned char* ptype, const float* mass, const float* h_sml,
    float* rho_out, float* count_out, int n
) {
    if (n <= 0) return 0;
    unsigned char* dptype;
    float *dmass, *dh, *drho, *dcount;
    if (alloc_copy(&dptype, ptype, n) || alloc_copy(&dmass, mass, n) ||
        alloc_copy(&dh, h_sml, n) || alloc_zero(&drho, n) || alloc_zero(&dcount, n)) return -1;
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_density_contrib_kernel<<<blocks, MHD_BLOCK_SIZE>>>(dptype, dmass, dh, drho, dcount, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(rho_out, drho, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(count_out, dcount, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dmass); cudaFree(dh); cudaFree(drho); cudaFree(dcount);
    return 0;
}

extern "C" int cuda_mhd_b_stats_contrib(
    const unsigned char* ptype, const float* mass,
    const float* bx, const float* by, const float* bz,
    float* m_out, float* mb_out, float* mb2_out, float* bmag_out, float* emag_out, float* count_out,
    int n
) {
    if (n <= 0) return 0;
    unsigned char* dptype;
    float *dmass, *dbx, *dby, *dbz, *dm, *dmb, *dmb2, *dbmag, *demag, *dcount;
    if (alloc_copy(&dptype, ptype, n) || alloc_copy(&dmass, mass, n) ||
        alloc_copy(&dbx, bx, n) || alloc_copy(&dby, by, n) || alloc_copy(&dbz, bz, n) ||
        alloc_zero(&dm, n) || alloc_zero(&dmb, n) || alloc_zero(&dmb2, n) ||
        alloc_zero(&dbmag, n) || alloc_zero(&demag, n) || alloc_zero(&dcount, n)) return -1;
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_b_stats_contrib_kernel<<<blocks, MHD_BLOCK_SIZE>>>(dptype, dmass, dbx, dby, dbz, dm, dmb, dmb2, dbmag, demag, dcount, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(m_out, dm, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mb_out, dmb, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mb2_out, dmb2, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bmag_out, dbmag, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(emag_out, demag, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(count_out, dcount, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dmass); cudaFree(dbx); cudaFree(dby); cudaFree(dbz);
    cudaFree(dm); cudaFree(dmb); cudaFree(dmb2); cudaFree(dbmag); cudaFree(demag); cudaFree(dcount);
    return 0;
}

__device__ float mhd_minimum_image(float dx, float box) {
    if (box > 0.0f) dx -= nearbyintf(dx / box) * box;
    return dx;
}

__global__ void mhd_induction_resistivity_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    const float* __restrict__ mass, const float* __restrict__ rho, const float* __restrict__ h_sml,
    const float* __restrict__ bx, const float* __restrict__ by, const float* __restrict__ bz,
    float* __restrict__ bx_out, float* __restrict__ by_out, float* __restrict__ bz_out,
    int n, float dt, float resistivity, float periodic_box
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float bxi = bx[i], byi = by[i], bzi = bz[i];
    if (ptype[i] == PTYPE_GAS) {
        float dbx = 0.0f, dby = 0.0f, dbz = 0.0f;
        float hi = fmaxf(h_sml[i], 1.0e-6f);
        float rhoi = fmaxf(rho[i], 1.0e-20f);
        for (int j = 0; j < n; ++j) {
            if (i == j || ptype[j] != PTYPE_GAS) continue;
            float dx = mhd_minimum_image(x[i] - x[j], periodic_box);
            float dy = mhd_minimum_image(y[i] - y[j], periodic_box);
            float dz = mhd_minimum_image(z[i] - z[j], periodic_box);
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= 4.0f * hi * hi || r2 <= 1.0e-20f) continue;
            float inv_r = rsqrtf(r2);
            float w = fmaxf(0.0f, 1.0f - 0.5f * sqrtf(r2) / hi);
            float dvx = vx[i] - vx[j], dvy = vy[i] - vy[j], dvz = vz[i] - vz[j];
            float bdotr = bxi*dx + byi*dy + bzi*dz;
            float vdotr = dvx*dx + dvy*dy + dvz*dz;
            float fac = mass[j] * w * inv_r / rhoi;
            dbx += fac * (bxi * vdotr - dvx * bdotr);
            dby += fac * (byi * vdotr - dvy * bdotr);
            dbz += fac * (bzi * vdotr - dvz * bdotr);
            dbx += resistivity * fac * (bx[j] - bxi);
            dby += resistivity * fac * (by[j] - byi);
            dbz += resistivity * fac * (bz[j] - bzi);
        }
        bxi += dt * dbx; byi += dt * dby; bzi += dt * dbz;
    }
    bx_out[i] = bxi; by_out[i] = byi; bz_out[i] = bzi;
}

__global__ void mhd_magnetic_forces_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ mass, const float* __restrict__ rho, const float* __restrict__ h_sml,
    const float* __restrict__ bx, const float* __restrict__ by, const float* __restrict__ bz,
    float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
    int n, float mu0, float periodic_box
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float axi = 0.0f, ayi = 0.0f, azi = 0.0f;
    if (ptype[i] == PTYPE_GAS) {
        float hi = fmaxf(h_sml[i], 1.0e-6f);
        float b2i = bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i];
        float pbi = b2i / (2.0f * fmaxf(mu0, 1.0e-20f));
        for (int j = 0; j < n; ++j) {
            if (i == j || ptype[j] != PTYPE_GAS) continue;
            float dx = mhd_minimum_image(x[i] - x[j], periodic_box);
            float dy = mhd_minimum_image(y[i] - y[j], periodic_box);
            float dz = mhd_minimum_image(z[i] - z[j], periodic_box);
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= 4.0f * hi * hi || r2 <= 1.0e-20f) continue;
            float r = sqrtf(r2);
            float wgrad = -fmaxf(0.0f, 1.0f - 0.5f * r / hi);
            float pbj = (bx[j]*bx[j] + by[j]*by[j] + bz[j]*bz[j]) / (2.0f * fmaxf(mu0, 1.0e-20f));
            float fac = -mass[j] * (pbi/(rho[i]*rho[i] + 1.0e-20f) + pbj/(rho[j]*rho[j] + 1.0e-20f)) * wgrad / r;
            axi += fac * dx; ayi += fac * dy; azi += fac * dz;
        }
    }
    ax[i] = axi; ay[i] = ayi; az[i] = azi;
}

__global__ void mhd_dedner_cleaning_kernel(
    const unsigned char* ptype, const float* div_b, const float* psi_in,
    const float* bx_in, const float* by_in, const float* bz_in,
    float* psi_out, float* bx_out, float* by_out, float* bz_out,
    int n, float dt, float ch, float cr
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float psi = psi_in[i], bx = bx_in[i], by = by_in[i], bz = bz_in[i];
    if (ptype[i] == PTYPE_GAS) {
        psi += -dt * ch * ch * div_b[i];
        psi *= expf(-dt * cr);
        float corr = dt * psi;
        bx -= corr * 0.3333333333f;
        by -= corr * 0.3333333333f;
        bz -= corr * 0.3333333333f;
    }
    psi_out[i] = psi; bx_out[i] = bx; by_out[i] = by; bz_out[i] = bz;
}

__global__ void mhd_scalar_diffusion_kernel(
    const unsigned char* ptype, const float* scalar_in,
    const float* bx, const float* by, const float* bz,
    float* scalar_out, int n, float dt, float kappa_par, float kappa_perp
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = scalar_in[i];
    if (ptype[i] == PTYPE_GAS) {
        float b2 = bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i];
        float kappa = (b2 > 1.0e-20f) ? kappa_par : kappa_perp;
        float avg = 0.0f; int cnt = 0;
        for (int j = 0; j < n; ++j) if (j != i && ptype[j] == PTYPE_GAS) { avg += scalar_in[j]; cnt++; }
        if (cnt > 0) s += dt * kappa * (avg / cnt - s);
    }
    scalar_out[i] = s;
}

__global__ void mhd_braginskii_kernel(
    const unsigned char* ptype,
    const float* vx, const float* vy, const float* vz,
    const float* bx, const float* by, const float* bz,
    float* ovx, float* ovy, float* ovz, int n, float dt, float eta
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vxi = vx[i], vyi = vy[i], vzi = vz[i];
    if (ptype[i] == PTYPE_GAS) {
        float bmag = sqrtf(bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i]);
        if (bmag > 1.0e-20f) {
            float bhx = bx[i]/bmag, bhy = by[i]/bmag, bhz = bz[i]/bmag;
            float vpar = vxi*bhx + vyi*bhy + vzi*bhz;
            vxi -= dt * eta * vpar * bhx;
            vyi -= dt * eta * vpar * bhy;
            vzi -= dt * eta * vpar * bhz;
        }
    }
    ovx[i] = vxi; ovy[i] = vyi; ovz[i] = vzi;
}

__global__ void mhd_reconnection_streaming_dynamo_kernel(
    const unsigned char* ptype,
    const float* cr_in, const float* bx_in, const float* by_in, const float* bz_in, const float* u_in,
    float* cr_out, float* bx_out, float* by_out, float* bz_out, float* u_out,
    int n, float dt, float stream_coeff, float reconnection_frac, float dynamo_alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float cr = cr_in[i], bx = bx_in[i], by = by_in[i], bz = bz_in[i], u = u_in[i];
    if (ptype[i] == PTYPE_GAS) {
        float bscale = fmaxf(0.0f, 1.0f + dt * dynamo_alpha - dt * reconnection_frac);
        float b2_before = bx*bx + by*by + bz*bz;
        bx *= bscale; by *= bscale; bz *= bscale;
        float b2_after = bx*bx + by*by + bz*bz;
        u += fmaxf(0.0f, 0.5f * (b2_before - b2_after));
        cr *= expf(-dt * fmaxf(0.0f, stream_coeff));
    }
    cr_out[i] = cr; bx_out[i] = bx; by_out[i] = by; bz_out[i] = bz; u_out[i] = u;
}

#define MHD_WRAP_ALLOC_COPY(name, ptr) float* name; if (alloc_copy(&name, ptr, n)) return -1
#define MHD_WRAP_ALLOC_ZERO(name) float* name; if (alloc_zero(&name, n)) return -1

extern "C" int cuda_mhd_induction_resistivity(
    const unsigned char* ptype, const float* x, const float* y, const float* z,
    const float* vx, const float* vy, const float* vz, const float* mass, const float* rho,
    const float* h_sml, const float* bx_in, const float* by_in, const float* bz_in,
    float* bx_out, float* by_out, float* bz_out, int n, float dt, float resistivity, float periodic_box
) {
    if (n <= 0) return 0;
    unsigned char* dptype; if (alloc_copy(&dptype, ptype, n)) return -1;
    MHD_WRAP_ALLOC_COPY(dx, x); MHD_WRAP_ALLOC_COPY(dy, y); MHD_WRAP_ALLOC_COPY(dz, z);
    MHD_WRAP_ALLOC_COPY(dvx, vx); MHD_WRAP_ALLOC_COPY(dvy, vy); MHD_WRAP_ALLOC_COPY(dvz, vz);
    MHD_WRAP_ALLOC_COPY(dmass, mass); MHD_WRAP_ALLOC_COPY(drho, rho); MHD_WRAP_ALLOC_COPY(dh, h_sml);
    MHD_WRAP_ALLOC_COPY(dbx, bx_in); MHD_WRAP_ALLOC_COPY(dby, by_in); MHD_WRAP_ALLOC_COPY(dbz, bz_in);
    MHD_WRAP_ALLOC_ZERO(obx); MHD_WRAP_ALLOC_ZERO(oby); MHD_WRAP_ALLOC_ZERO(obz);
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_induction_resistivity_kernel<<<blocks, MHD_BLOCK_SIZE>>>(dptype, dx, dy, dz, dvx, dvy, dvz, dmass, drho, dh, dbx, dby, dbz, obx, oby, obz, n, dt, resistivity, periodic_box);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(bx_out, obx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(by_out, oby, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bz_out, obz, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dmass); cudaFree(drho); cudaFree(dh); cudaFree(dbx); cudaFree(dby); cudaFree(dbz); cudaFree(obx); cudaFree(oby); cudaFree(obz);
    return 0;
}

extern "C" int cuda_mhd_magnetic_forces(
    const unsigned char* ptype, const float* x, const float* y, const float* z,
    const float* mass, const float* rho, const float* h_sml,
    const float* bx, const float* by, const float* bz,
    float* ax_out, float* ay_out, float* az_out, int n, float mu0, float periodic_box
) {
    if (n <= 0) return 0;
    unsigned char* dptype; if (alloc_copy(&dptype, ptype, n)) return -1;
    MHD_WRAP_ALLOC_COPY(dx, x); MHD_WRAP_ALLOC_COPY(dy, y); MHD_WRAP_ALLOC_COPY(dz, z);
    MHD_WRAP_ALLOC_COPY(dmass, mass); MHD_WRAP_ALLOC_COPY(drho, rho); MHD_WRAP_ALLOC_COPY(dh, h_sml);
    MHD_WRAP_ALLOC_COPY(dbx, bx); MHD_WRAP_ALLOC_COPY(dby, by); MHD_WRAP_ALLOC_COPY(dbz, bz);
    MHD_WRAP_ALLOC_ZERO(oax); MHD_WRAP_ALLOC_ZERO(oay); MHD_WRAP_ALLOC_ZERO(oaz);
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_magnetic_forces_kernel<<<blocks, MHD_BLOCK_SIZE>>>(dptype, dx, dy, dz, dmass, drho, dh, dbx, dby, dbz, oax, oay, oaz, n, mu0, periodic_box);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(ax_out, oax, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay_out, oay, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az_out, oaz, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dmass); cudaFree(drho); cudaFree(dh); cudaFree(dbx); cudaFree(dby); cudaFree(dbz); cudaFree(oax); cudaFree(oay); cudaFree(oaz);
    return 0;
}

extern "C" int cuda_mhd_dedner_cleaning(
    const unsigned char* ptype, const float* div_b, const float* psi_in,
    const float* bx_in, const float* by_in, const float* bz_in,
    float* psi_out, float* bx_out, float* by_out, float* bz_out,
    int n, float dt, float ch, float cr
) {
    if (n <= 0) return 0;
    unsigned char* dptype; if (alloc_copy(&dptype, ptype, n)) return -1;
    MHD_WRAP_ALLOC_COPY(ddiv, div_b); MHD_WRAP_ALLOC_COPY(dpsi, psi_in);
    MHD_WRAP_ALLOC_COPY(dbx, bx_in); MHD_WRAP_ALLOC_COPY(dby, by_in); MHD_WRAP_ALLOC_COPY(dbz, bz_in);
    MHD_WRAP_ALLOC_ZERO(opsi); MHD_WRAP_ALLOC_ZERO(obx); MHD_WRAP_ALLOC_ZERO(oby); MHD_WRAP_ALLOC_ZERO(obz);
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_dedner_cleaning_kernel<<<blocks, MHD_BLOCK_SIZE>>>(dptype, ddiv, dpsi, dbx, dby, dbz, opsi, obx, oby, obz, n, dt, ch, cr);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(psi_out, opsi, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bx_out, obx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(by_out, oby, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bz_out, obz, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(ddiv); cudaFree(dpsi); cudaFree(dbx); cudaFree(dby); cudaFree(dbz); cudaFree(opsi); cudaFree(obx); cudaFree(oby); cudaFree(obz);
    return 0;
}

extern "C" int cuda_mhd_scalar_diffusion(
    const unsigned char* ptype, const float* scalar_in, const float* bx, const float* by, const float* bz,
    float* scalar_out, int n, float dt, float kappa_par, float kappa_perp
) {
    if (n <= 0) return 0;
    unsigned char* dptype; if (alloc_copy(&dptype, ptype, n)) return -1;
    MHD_WRAP_ALLOC_COPY(ds, scalar_in); MHD_WRAP_ALLOC_COPY(dbx, bx); MHD_WRAP_ALLOC_COPY(dby, by); MHD_WRAP_ALLOC_COPY(dbz, bz); MHD_WRAP_ALLOC_ZERO(os);
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_scalar_diffusion_kernel<<<blocks, MHD_BLOCK_SIZE>>>(dptype, ds, dbx, dby, dbz, os, n, dt, kappa_par, kappa_perp);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(scalar_out, os, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(ds); cudaFree(dbx); cudaFree(dby); cudaFree(dbz); cudaFree(os);
    return 0;
}

extern "C" int cuda_mhd_braginskii_viscosity(
    const unsigned char* ptype, const float* vx_in, const float* vy_in, const float* vz_in,
    const float* bx, const float* by, const float* bz, float* vx_out, float* vy_out, float* vz_out,
    int n, float dt, float eta
) {
    if (n <= 0) return 0;
    unsigned char* dptype; if (alloc_copy(&dptype, ptype, n)) return -1;
    MHD_WRAP_ALLOC_COPY(dvx, vx_in); MHD_WRAP_ALLOC_COPY(dvy, vy_in); MHD_WRAP_ALLOC_COPY(dvz, vz_in);
    MHD_WRAP_ALLOC_COPY(dbx, bx); MHD_WRAP_ALLOC_COPY(dby, by); MHD_WRAP_ALLOC_COPY(dbz, bz);
    MHD_WRAP_ALLOC_ZERO(ovx); MHD_WRAP_ALLOC_ZERO(ovy); MHD_WRAP_ALLOC_ZERO(ovz);
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_braginskii_kernel<<<blocks, MHD_BLOCK_SIZE>>>(dptype, dvx, dvy, dvz, dbx, dby, dbz, ovx, ovy, ovz, n, dt, eta);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(vx_out, ovx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vy_out, ovy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vz_out, ovz, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dbx); cudaFree(dby); cudaFree(dbz); cudaFree(ovx); cudaFree(ovy); cudaFree(ovz);
    return 0;
}

extern "C" int cuda_mhd_reconnection_streaming_dynamo(
    const unsigned char* ptype, const float* cr_in, const float* bx_in, const float* by_in, const float* bz_in, const float* u_in,
    float* cr_out, float* bx_out, float* by_out, float* bz_out, float* u_out,
    int n, float dt, float stream_coeff, float reconnection_frac, float dynamo_alpha
) {
    if (n <= 0) return 0;
    unsigned char* dptype; if (alloc_copy(&dptype, ptype, n)) return -1;
    MHD_WRAP_ALLOC_COPY(dcr, cr_in); MHD_WRAP_ALLOC_COPY(dbx, bx_in); MHD_WRAP_ALLOC_COPY(dby, by_in); MHD_WRAP_ALLOC_COPY(dbz, bz_in); MHD_WRAP_ALLOC_COPY(du, u_in);
    MHD_WRAP_ALLOC_ZERO(ocr); MHD_WRAP_ALLOC_ZERO(obx); MHD_WRAP_ALLOC_ZERO(oby); MHD_WRAP_ALLOC_ZERO(obz); MHD_WRAP_ALLOC_ZERO(ou);
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_reconnection_streaming_dynamo_kernel<<<blocks, MHD_BLOCK_SIZE>>>(dptype, dcr, dbx, dby, dbz, du, ocr, obx, oby, obz, ou, n, dt, stream_coeff, reconnection_frac, dynamo_alpha);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(cr_out, ocr, bytes, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(bx_out, obx, bytes, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(by_out, oby, bytes, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(bz_out, obz, bytes, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(u_out, ou, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dcr); cudaFree(dbx); cudaFree(dby); cudaFree(dbz); cudaFree(du); cudaFree(ocr); cudaFree(obx); cudaFree(oby); cudaFree(obz); cudaFree(ou);
    return 0;
}

// ── CR streaming: actualización de cr_energy por partícula ───────────────────
// Modelo: per-partícula local (sin interacción entre partículas).
// cr_energy += (-compressional_loss - streaming_loss) * dt
// compressional_loss ≈ (1/3) * cr_energy * (suma SPH de divergencia)
// streaming_loss     = coeff * v_alfven * cr_energy / h_sml
// v_alfven           = |B| / sqrt(rho)   con rho ≈ mass/(4π/3 h³)

__device__ static inline float cr_density_approx(float mass, float h) {
    const float PI3 = 4.18879020f;  // 4π/3
    float h3 = h * h * h;
    return h3 > 0.0f ? mass / (PI3 * h3) : 1.0e-30f;
}

// ── Difusión ambipolar (AP-16) ────────────────────────────────────────────────

__global__ void mhd_ambipolar_kernel(
    const unsigned char* ptype,
    const float* bx_in, const float* by_in, const float* bz_in,
    const float* u_in, const float* mass, const float* dust_to_gas,
    float* bx_out, float* by_out, float* bz_out, float* u_out,
    int n, float eta_ad, float ion_floor, float dust_coupling, float heat_eff, float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float bx = bx_in[i], by = by_in[i], bz = bz_in[i], u = u_in[i];
    if (ptype[i] != PTYPE_GAS) {
        bx_out[i] = bx; by_out[i] = by; bz_out[i] = bz; u_out[i] = u; return;
    }
    float b2b = bx*bx + by*by + bz*bz;
    if (b2b <= 0.0f) {
        bx_out[i] = bx; by_out[i] = by; bz_out[i] = bz; u_out[i] = u; return;
    }
    float th  = fmaxf(u, 0.0f);
    float col = th / (th + 1.0f);
    float ds  = 1.0f / (1.0f + fmaxf(dust_coupling, 0.0f)
                                * fmaxf(dust_to_gas[i], 0.0f) * 100.0f);
    float xi  = fmaxf(fminf(col * ds, 1.0f), ion_floor);
    float rate   = fmaxf(eta_ad, 0.0f) * fmaxf(1.0f / xi - 1.0f, 0.0f);
    float damp   = fminf(expf(-rate * dt), 1.0f);
    bx *= damp; by *= damp; bz *= damp;
    float b2a   = bx*bx + by*by + bz*bz;
    float diss  = 0.5f * fmaxf(b2b - b2a, 0.0f);
    float m     = fmaxf(mass[i], 1.0e-30f);
    bx_out[i] = bx; by_out[i] = by; bz_out[i] = bz;
    u_out[i]  = fmaxf(u + heat_eff * diss / m, 0.0f);
}

extern "C" int cuda_mhd_ambipolar(
    const unsigned char* ptype,
    const float* bx_in, const float* by_in, const float* bz_in,
    const float* u_in, const float* mass, const float* dust_to_gas,
    float* bx_out, float* by_out, float* bz_out, float* u_out,
    int n, float eta_ad, float ion_floor, float dust_coupling, float heat_eff, float dt
) {
    if (n <= 0) return 0;
    unsigned char* dp;
    float *dbxi, *dbyi, *dbzi, *dui, *dm, *ddtg;
    float *dbxo, *dbyo, *dbzo, *duo;
    if (alloc_copy(&dp,   ptype,       n)) return -1;
    if (alloc_copy(&dbxi, bx_in,       n)) return -1;
    if (alloc_copy(&dbyi, by_in,       n)) return -1;
    if (alloc_copy(&dbzi, bz_in,       n)) return -1;
    if (alloc_copy(&dui,  u_in,        n)) return -1;
    if (alloc_copy(&dm,   mass,        n)) return -1;
    if (alloc_copy(&ddtg, dust_to_gas, n)) return -1;
    if (alloc_zero(&dbxo, n)) return -1;
    if (alloc_zero(&dbyo, n)) return -1;
    if (alloc_zero(&dbzo, n)) return -1;
    if (alloc_zero(&duo,  n)) return -1;
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_ambipolar_kernel<<<blocks, MHD_BLOCK_SIZE>>>(
        dp, dbxi, dbyi, dbzi, dui, dm, ddtg,
        dbxo, dbyo, dbzo, duo,
        n, eta_ad, ion_floor, dust_coupling, heat_eff, dt);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(bx_out, dbxo, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(by_out, dbyo, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bz_out, dbzo, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(u_out,  duo,  (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dp); cudaFree(dbxi); cudaFree(dbyi); cudaFree(dbzi);
    cudaFree(dui); cudaFree(dm); cudaFree(ddtg);
    cudaFree(dbxo); cudaFree(dbyo); cudaFree(dbzo); cudaFree(duo);
    return 0;
}

// ── Ohmic resistive diffusion (Phase 187) ────────────────────────────────────
// dB/dt|_Ohm = -eta_Ohm * B / h²   →   B_new = B_old * exp(-eta_Ohm * dt / h²)
// u_out[i] = heat_eff * (B²_before - B²_after) / (2 * mass)   (energy delta)

__global__ void mhd_ohmic_kernel(
    const unsigned char* ptype,
    const float* bx_in, const float* by_in, const float* bz_in,
    const float* h_sml, const float* mass,
    float* bx_out, float* by_out, float* bz_out, float* u_delta_out,
    int n, float eta_ohm, float heat_eff, float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bx_out[i] = bx_in[i]; by_out[i] = by_in[i]; bz_out[i] = bz_in[i];
    u_delta_out[i] = 0.0f;
    if (ptype[i] != PTYPE_GAS) return;
    float bx = bx_in[i], by = by_in[i], bz = bz_in[i];
    float b2_before = bx*bx + by*by + bz*bz;
    if (b2_before <= 0.0f) return;
    float h  = fmaxf(h_sml[i], 1.0e-10f);
    float h2 = fmaxf(h * h, 1.0e-60f);
    float rate = eta_ohm * dt / h2;
    float damp = expf(-rate);
    damp = fminf(fmaxf(damp, 0.0f), 1.0f);
    float nbx = bx * damp, nby = by * damp, nbz = bz * damp;
    float b2_after = nbx*nbx + nby*nby + nbz*nbz;
    float dissipated = 0.5f * fmaxf(b2_before - b2_after, 0.0f);
    float m = fmaxf(mass[i], 1.0e-30f);
    bx_out[i] = nbx; by_out[i] = nby; bz_out[i] = nbz;
    u_delta_out[i] = heat_eff * dissipated / m;
}

extern "C" int cuda_mhd_ohmic(
    const unsigned char* ptype,
    const float* bx_in, const float* by_in, const float* bz_in,
    const float* h_sml, const float* mass,
    float* bx_out, float* by_out, float* bz_out, float* u_out,
    int n, float eta_ohm, float heat_eff, float dt
) {
    if (n <= 0) return 0;
    unsigned char* dp;
    float *dbxi, *dbyi, *dbzi, *dh, *dm;
    float *dbxo, *dbyo, *dbzo, *duo;
    if (alloc_copy(&dp,   ptype,  n)) return -1;
    if (alloc_copy(&dbxi, bx_in,  n)) return -1;
    if (alloc_copy(&dbyi, by_in,  n)) return -1;
    if (alloc_copy(&dbzi, bz_in,  n)) return -1;
    if (alloc_copy(&dh,   h_sml,  n)) return -1;
    if (alloc_copy(&dm,   mass,   n)) return -1;
    if (alloc_zero(&dbxo, n)) return -1;
    if (alloc_zero(&dbyo, n)) return -1;
    if (alloc_zero(&dbzo, n)) return -1;
    if (alloc_zero(&duo,  n)) return -1;
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_ohmic_kernel<<<blocks, MHD_BLOCK_SIZE>>>(
        dp, dbxi, dbyi, dbzi, dh, dm,
        dbxo, dbyo, dbzo, duo,
        n, eta_ohm, heat_eff, dt);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(bx_out, dbxo, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(by_out, dbyo, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bz_out, dbzo, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(u_out,  duo,  (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dp); cudaFree(dbxi); cudaFree(dbyi); cudaFree(dbzi);
    cudaFree(dh); cudaFree(dm);
    cudaFree(dbxo); cudaFree(dbyo); cudaFree(dbzo); cudaFree(duo);
    return 0;
}

// ── Two-fluid: acoplamiento electrón-ión Coulomb (AP-16) ─────────────────────

__global__ void mhd_two_fluid_kernel(
    const unsigned char* ptype,
    const float* u_in, const float* h_sml, const float* mass, const float* te_in,
    float* te_out, int n, float nu_ei_coeff, float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float te = te_in[i];
    if (ptype[i] != PTYPE_GAS) { te_out[i] = te; return; }
    const float GAMMA_TF = 5.0f / 3.0f;
    float t_i  = (GAMMA_TF - 1.0f) * fmaxf(u_in[i], 0.0f);
    if (te <= 0.0f) { te_out[i] = t_i; return; }
    float h    = fmaxf(h_sml[i], 1.0e-10f);
    float rho  = fmaxf(mass[i] / (h*h*h), 1.0e-30f);
    float te2  = fmaxf(fabsf(te), 1.0e-30f);
    float nu   = nu_ei_coeff * rho / (te2 * sqrtf(te2));
    float fac  = 1.0f - expf(-nu * dt);
    te_out[i]  = fmaxf(te + (t_i - te) * fac, 0.0f);
}

extern "C" int cuda_mhd_two_fluid(
    const unsigned char* ptype,
    const float* u_in, const float* h_sml, const float* mass, const float* te_in,
    float* te_out, int n, float nu_ei_coeff, float dt
) {
    if (n <= 0) return 0;
    unsigned char* dp;
    float *dui, *dh, *dm, *dtei, *dteo;
    if (alloc_copy(&dp,   ptype, n)) return -1;
    if (alloc_copy(&dui,  u_in,  n)) return -1;
    if (alloc_copy(&dh,   h_sml, n)) return -1;
    if (alloc_copy(&dm,   mass,  n)) return -1;
    if (alloc_copy(&dtei, te_in, n)) return -1;
    if (alloc_zero(&dteo, n))        return -1;
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_two_fluid_kernel<<<blocks, MHD_BLOCK_SIZE>>>(
        dp, dui, dh, dm, dtei, dteo, n, nu_ei_coeff, dt);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(te_out, dteo, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dp); cudaFree(dui); cudaFree(dh); cudaFree(dm); cudaFree(dtei); cudaFree(dteo);
    return 0;
}

__global__ void mhd_cr_streaming_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ mass,
    const float* __restrict__ h_sml,
    const float* __restrict__ cr_energy_in,
    const float* __restrict__ bx, const float* __restrict__ by, const float* __restrict__ bz,
    float* __restrict__ cr_energy_out,
    int n, float dt, float streaming_coeff
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ecr = cr_energy_in[i];
    if (ptype[i] != PTYPE_GAS || ecr <= 0.0f) {
        cr_energy_out[i] = ecr;
        return;
    }
    float h_i   = fmaxf(h_sml[i], 1.0e-10f);
    float rho_i = cr_density_approx(mass[i], h_i);
    float b2    = bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i];
    float v_a   = sqrtf(b2 / fmaxf(rho_i, 1.0e-30f));
    float stream_loss = (streaming_coeff > 0.0f && v_a > 1.0e-30f)
                        ? streaming_coeff * v_a * ecr / h_i
                        : 0.0f;
    // Compressional term omitted here (requires neighbor sum → O(N²) version below).
    // This kernel handles the streaming-only contribution for per-particle updates.
    cr_energy_out[i] = fmaxf(ecr - stream_loss * dt, 0.0f);
}

// O(N²) kernel: 1 hilo por partícula i, suma vecinos j para div_v.
// Sólo para N pequeño (smoke/parity); en producción se usa un tree.
__global__ void mhd_cr_streaming_o2_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    const float* __restrict__ mass,
    const float* __restrict__ h_sml,
    const float* __restrict__ cr_energy_in,
    const float* __restrict__ bx, const float* __restrict__ by, const float* __restrict__ bz,
    float* __restrict__ cr_energy_out,
    int n, float dt, float streaming_coeff, float periodic_box
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ecr = cr_energy_in[i];
    if (ptype[i] != PTYPE_GAS || ecr <= 0.0f) {
        cr_energy_out[i] = ecr;
        return;
    }
    float h_i   = fmaxf(h_sml[i], 1.0e-10f);
    float rho_i = cr_density_approx(mass[i], h_i);
    float b2    = bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i];
    float v_a   = sqrtf(b2 / fmaxf(rho_i, 1.0e-30f));

    // div_v SPH estimate
    float div_v = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (j == i || ptype[j] != PTYPE_GAS) continue;
        float dx = px[j] - px[i];
        float dy = py[j] - py[i];
        float dz = pz[j] - pz[i];
        if (periodic_box > 0.0f) {
            dx -= periodic_box * rintf(dx / periodic_box);
            dy -= periodic_box * rintf(dy / periodic_box);
            dz -= periodic_box * rintf(dz / periodic_box);
        }
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        if (r < 1.0e-14f || r > 2.0f * h_i) continue;
        float q = r / h_i;
        float dwdr;
        if (q > 1.0f) {
            float t = 0.5f * (2.0f - q);
            dwdr = -(21.0f / (2.0f * 3.14159265f)) / (h_i*h_i*h_i) * 4.0f * t*t*t * (1.0f + q) * (-0.5f);
        } else {
            float t = 1.0f - 0.5f * q;
            dwdr = -(21.0f / (2.0f * 3.14159265f)) / (h_i*h_i*h_i) * 4.0f * t*t*t * (-1.5f - 2.0f * q) / h_i;
        }
        float h_j = fmaxf(h_sml[j], 1.0e-10f);
        // vol_j = mass / rho_j = 4/3π h_j³  (matches CPU: mass/vol_j_cpu = 4/3π h_j³)
        float vol_j = mass[j] / fmaxf(cr_density_approx(mass[j], h_j), 1.0e-30f);
        float vdotdr = ((vx[i]-vx[j])*(dx/r) + (vy[i]-vy[j])*(dy/r) + (vz[i]-vz[j])*(dz/r));
        div_v += vol_j * vdotdr * dwdr / h_i;
    }
    float compressional = -(1.0f / 3.0f) * ecr * div_v;
    // streaming_loss has same sign as CPU: total = compressional + stream_loss
    float stream_loss   = (streaming_coeff > 0.0f && v_a > 1.0e-30f)
                          ? streaming_coeff * v_a * ecr / h_i
                          : 0.0f;
    cr_energy_out[i] = fmaxf(ecr + (compressional + stream_loss) * dt, 0.0f);
}

// Backreaction: acelera gas usando gradiente de presión CR (O(N²) pares)
// P_CR = (γ_CR - 1) * ε_CR  con γ_CR ≈ 4/3
// F_i += - sum_j m_j * (P_i/rho_i² + P_j/rho_j²) * grad_W_ij
__global__ void mhd_cr_backreaction_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ mass,
    const float* __restrict__ h_sml,
    const float* __restrict__ cr_energy,
    float* __restrict__ ax_out, float* __restrict__ ay_out, float* __restrict__ az_out,
    int n, float periodic_box
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || ptype[i] != PTYPE_GAS) return;

    float h_i   = fmaxf(h_sml[i], 1.0e-10f);
    float rho_i = cr_density_approx(mass[i], h_i);
    // P_CR = (γ_CR - 1) * rho * ε_CR  (matches CPU: p_cr = (GAMMA_CR-1) * rho * cr_energy)
    float p_cri = (4.0f / 3.0f - 1.0f) * rho_i * fmaxf(cr_energy[i], 0.0f);

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (j == i || ptype[j] != PTYPE_GAS) continue;
        float dx = px[j] - px[i];
        float dy = py[j] - py[i];
        float dz = pz[j] - pz[i];
        if (periodic_box > 0.0f) {
            dx -= periodic_box * rintf(dx / periodic_box);
            dy -= periodic_box * rintf(dy / periodic_box);
            dz -= periodic_box * rintf(dz / periodic_box);
        }
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        if (r < 1.0e-14f || r > 2.0f * h_i) continue;
        float q = r / h_i;
        float dwdr;
        if (q > 1.0f) {
            float t = 0.5f * (2.0f - q);
            dwdr = -(21.0f / (2.0f * 3.14159265f)) / (h_i*h_i*h_i) * 4.0f * t*t*t * (1.0f + q) * (-0.5f);
        } else {
            float t = 1.0f - 0.5f * q;
            dwdr = -(21.0f / (2.0f * 3.14159265f)) / (h_i*h_i*h_i) * 4.0f * t*t*t * (-1.5f - 2.0f * q) / h_i;
        }
        float h_j   = fmaxf(h_sml[j], 1.0e-10f);
        float rho_j = cr_density_approx(mass[j], h_j);
        float p_crj = (4.0f / 3.0f - 1.0f) * rho_j * fmaxf(cr_energy[j], 0.0f);
        // p_avg = (p_cr_i + p_cr_j) / 2  (matches CPU)
        float p_avg = 0.5f * (p_cri + p_crj);
        float coeff = mass[j] * p_avg * dwdr;
        ax += coeff * dx / r;
        ay += coeff * dy / r;
        az += coeff * dz / r;
    }
    ax_out[i] = ax;
    ay_out[i] = ay;
    az_out[i] = az;
}

extern "C" int cuda_mhd_cr_streaming(
    const unsigned char* ptype,
    const float* px, const float* py, const float* pz,
    const float* vx, const float* vy, const float* vz,
    const float* mass, const float* h_sml,
    const float* cr_energy_in,
    const float* bx, const float* by, const float* bz,
    float* cr_energy_out,
    int n, float dt, float streaming_coeff, float periodic_box
) {
    if (n <= 0) return 0;
    unsigned char* dptype;
    float *dpx, *dpy, *dpz, *dvx, *dvy, *dvz;
    float *dmass, *dh, *dcr, *dbx, *dby, *dbz, *docr;
    if (alloc_copy(&dptype, ptype, n) ||
        alloc_copy(&dpx, px, n) || alloc_copy(&dpy, py, n) || alloc_copy(&dpz, pz, n) ||
        alloc_copy(&dvx, vx, n) || alloc_copy(&dvy, vy, n) || alloc_copy(&dvz, vz, n) ||
        alloc_copy(&dmass, mass, n) || alloc_copy(&dh, h_sml, n) ||
        alloc_copy(&dcr, cr_energy_in, n) ||
        alloc_copy(&dbx, bx, n) || alloc_copy(&dby, by, n) || alloc_copy(&dbz, bz, n) ||
        alloc_zero(&docr, n)) return -1;
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_cr_streaming_o2_kernel<<<blocks, MHD_BLOCK_SIZE>>>(
        dptype, dpx, dpy, dpz, dvx, dvy, dvz, dmass, dh, dcr, dbx, dby, dbz, docr,
        n, dt, streaming_coeff, periodic_box);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(cr_energy_out, docr, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dpx); cudaFree(dpy); cudaFree(dpz);
    cudaFree(dvx); cudaFree(dvy); cudaFree(dvz);
    cudaFree(dmass); cudaFree(dh); cudaFree(dcr);
    cudaFree(dbx); cudaFree(dby); cudaFree(dbz); cudaFree(docr);
    return 0;
}

extern "C" int cuda_mhd_cr_backreaction(
    const unsigned char* ptype,
    const float* px, const float* py, const float* pz,
    const float* mass, const float* h_sml,
    const float* cr_energy,
    float* ax_out, float* ay_out, float* az_out,
    int n, float periodic_box
) {
    if (n <= 0) return 0;
    unsigned char* dptype;
    float *dpx, *dpy, *dpz, *dmass, *dh, *dcr, *dax, *day, *daz;
    if (alloc_copy(&dptype, ptype, n) ||
        alloc_copy(&dpx, px, n) || alloc_copy(&dpy, py, n) || alloc_copy(&dpz, pz, n) ||
        alloc_copy(&dmass, mass, n) || alloc_copy(&dh, h_sml, n) ||
        alloc_copy(&dcr, cr_energy, n) ||
        alloc_zero(&dax, n) || alloc_zero(&day, n) || alloc_zero(&daz, n)) return -1;
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_cr_backreaction_kernel<<<blocks, MHD_BLOCK_SIZE>>>(
        dptype, dpx, dpy, dpz, dmass, dh, dcr, dax, day, daz, n, periodic_box);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(ax_out, dax, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay_out, day, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az_out, daz, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dpx); cudaFree(dpy); cudaFree(dpz);
    cudaFree(dmass); cudaFree(dh); cudaFree(dcr);
    cudaFree(dax); cudaFree(day); cudaFree(daz);
    return 0;
}

// ── Conducción anisótropa / CR diffusion O(N²) (AP-17) ───────────────────────────
//
// Kernel Wendland-C6 simétrico: h_eff = h_i + h_j (banda completa),
// kappa_eff = kappa_perp + (kappa_par - kappa_perp) * cos²θ,
// cos θ  = B̂_i · r̂_ij.
// delta_scalar_i += kappa_eff * (scalar_j - scalar_i) * W(r, h_eff) * dt
// Para conducción térmica: scalar = T = (gamma-1)*u.
// Para CR diffusion:       scalar = cr_energy.
__device__ inline float wendland_c6_3d(float r, float h) {
    if (h <= 0.0f) return 0.0f;
    float q = r / h;
    if (q >= 2.0f) return 0.0f;
    float t = 1.0f - 0.5f * q;
    return (21.0f / (2.0f * 3.14159265f * h * h * h)) * t * t * t * t * (1.0f + 2.0f * q);
}

__global__ void mhd_anisotropic_pair_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ mass,
    const float* __restrict__ h_sml,
    const float* __restrict__ scalar_in,  // u para térmica, cr_energy para CR
    const float* __restrict__ bx, const float* __restrict__ by, const float* __restrict__ bz,
    float* __restrict__ scalar_out,
    int n, float kappa_par, float kappa_perp, float gamma_m1, float dt, float periodic_box
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || ptype[i] != PTYPE_GAS) {
        if (i < n) scalar_out[i] = scalar_in[i];
        return;
    }
    float si = scalar_in[i];
    float h_i = fmaxf(h_sml[i], 1.0e-10f);
    float bx_i = bx[i], by_i = by[i], bz_i = bz[i];
    float b_mag = sqrtf(bx_i*bx_i + by_i*by_i + bz_i*bz_i);
    float inv_b = (b_mag > 1.0e-30f) ? 1.0f / b_mag : 0.0f;
    // T_i or e_cr_i depending on mode (gamma_m1 != 0 → thermal; == 0 → CR)
    float field_i = (gamma_m1 > 0.0f) ? (gamma_m1 * si) : si;

    float delta = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (j == i || ptype[j] != PTYPE_GAS) continue;
        float dx = px[j] - px[i];
        float dy = py[j] - py[i];
        float dz = pz[j] - pz[i];
        if (periodic_box > 0.0f) {
            dx -= periodic_box * rintf(dx / periodic_box);
            dy -= periodic_box * rintf(dy / periodic_box);
            dz -= periodic_box * rintf(dz / periodic_box);
        }
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        if (r < 1.0e-14f) continue;
        float h_j = fmaxf(h_sml[j], 1.0e-10f);
        float h_eff = h_i + h_j;  // banda completa (= 2*h_avg)
        float w = wendland_c6_3d(r, h_eff);
        if (w <= 0.0f) continue;

        // cos²θ = (B̂_i · r̂_ij)²
        float inv_r = 1.0f / r;
        float rhat_x = dx * inv_r, rhat_y = dy * inv_r, rhat_z = dz * inv_r;
        float cos_th = (bx_i*rhat_x + by_i*rhat_y + bz_i*rhat_z) * inv_b;
        float cos2 = cos_th * cos_th;
        float kappa_eff = kappa_perp + (kappa_par - kappa_perp) * cos2;

        float sj = scalar_in[j];
        float field_j = (gamma_m1 > 0.0f) ? (gamma_m1 * sj) : sj;
        delta += kappa_eff * (field_j - field_i) * w * dt;
    }

    // Traducir delta de "temperatura" o "e_cr" de vuelta al campo scalar
    float new_scalar;
    if (gamma_m1 > 0.0f) {
        // thermal: delta es ΔT = (γ-1) Δu  →  Δu = delta / (γ-1)
        new_scalar = fmaxf(si + delta / gamma_m1, 0.0f);
    } else {
        new_scalar = fmaxf(si + delta, 0.0f);
    }
    scalar_out[i] = new_scalar;
}

extern "C" int cuda_mhd_anisotropic_conduction(
    const unsigned char* ptype,
    const float* px, const float* py, const float* pz,
    const float* mass, const float* h_sml, const float* u_in,
    const float* bx, const float* by, const float* bz,
    float* u_out, int n,
    float kappa_par, float kappa_perp, float gamma, float dt, float periodic_box
) {
    if (n <= 0) return 0;
    unsigned char* aniso_dptype; if (alloc_copy(&aniso_dptype, ptype, n)) return -1;
    float *aniso_px, *aniso_py, *aniso_pz, *aniso_m, *aniso_h, *aniso_u;
    float *aniso_bx, *aniso_by, *aniso_bz, *aniso_out;
    if (alloc_copy(&aniso_px, px, n) || alloc_copy(&aniso_py, py, n) || alloc_copy(&aniso_pz, pz, n) ||
        alloc_copy(&aniso_m, mass, n) || alloc_copy(&aniso_h, h_sml, n) || alloc_copy(&aniso_u, u_in, n) ||
        alloc_copy(&aniso_bx, bx, n) || alloc_copy(&aniso_by, by, n) || alloc_copy(&aniso_bz, bz, n) ||
        alloc_zero(&aniso_out, n)) return -1;
    int aniso_blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    float gamma_m1 = gamma - 1.0f;
    mhd_anisotropic_pair_kernel<<<aniso_blocks, MHD_BLOCK_SIZE>>>(
        aniso_dptype, aniso_px, aniso_py, aniso_pz, aniso_m, aniso_h, aniso_u,
        aniso_bx, aniso_by, aniso_bz,
        aniso_out, n, kappa_par, kappa_perp, gamma_m1, dt, periodic_box);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(u_out, aniso_out, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(aniso_dptype); cudaFree(aniso_px); cudaFree(aniso_py); cudaFree(aniso_pz);
    cudaFree(aniso_m); cudaFree(aniso_h); cudaFree(aniso_u);
    cudaFree(aniso_bx); cudaFree(aniso_by); cudaFree(aniso_bz); cudaFree(aniso_out);
    return 0;
}

extern "C" int cuda_mhd_cr_diffusion_anisotropic(
    const unsigned char* ptype,
    const float* px, const float* py, const float* pz,
    const float* mass, const float* h_sml, const float* cr_energy_in,
    const float* bx, const float* by, const float* bz,
    float* cr_energy_out, int n,
    float kappa_cr, float dt, float periodic_box
) {
    if (n <= 0) return 0;
    unsigned char* crdiff_ptype; if (alloc_copy(&crdiff_ptype, ptype, n)) return -1;
    float *crdiff_px, *crdiff_py, *crdiff_pz, *crdiff_m, *crdiff_h, *crdiff_cr;
    float *crdiff_bx, *crdiff_by, *crdiff_bz, *crdiff_out;
    if (alloc_copy(&crdiff_px, px, n) || alloc_copy(&crdiff_py, py, n) || alloc_copy(&crdiff_pz, pz, n) ||
        alloc_copy(&crdiff_m, mass, n) || alloc_copy(&crdiff_h, h_sml, n) || alloc_copy(&crdiff_cr, cr_energy_in, n) ||
        alloc_copy(&crdiff_bx, bx, n) || alloc_copy(&crdiff_by, by, n) || alloc_copy(&crdiff_bz, bz, n) ||
        alloc_zero(&crdiff_out, n)) return -1;
    int crdiff_blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    // CR diffusion: gamma_m1 = 0.0 → scalar field es cr_energy directamente
    mhd_anisotropic_pair_kernel<<<crdiff_blocks, MHD_BLOCK_SIZE>>>(
        crdiff_ptype, crdiff_px, crdiff_py, crdiff_pz, crdiff_m, crdiff_h, crdiff_cr,
        crdiff_bx, crdiff_by, crdiff_bz,
        crdiff_out, n, kappa_cr, 0.0f, 0.0f, dt, periodic_box);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(cr_energy_out, crdiff_out, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(crdiff_ptype); cudaFree(crdiff_px); cudaFree(crdiff_py); cudaFree(crdiff_pz);
    cudaFree(crdiff_m); cudaFree(crdiff_h); cudaFree(crdiff_cr);
    cudaFree(crdiff_bx); cudaFree(crdiff_by); cudaFree(crdiff_bz); cudaFree(crdiff_out);
    return 0;
}

// ── Hall drift (AP-20 / Phase 186) ────────────────────────────────────────────
// Rota B alrededor del eje (v × B) usando la fórmula de Rodrigues.
// Conserva |B| exactamente; no modifica u (Hall no disipa energía).

__global__ void mhd_hall_drift_kernel(
    const unsigned char* ptype,
    const float* bx_in, const float* by_in, const float* bz_in,
    const float* vx, const float* vy, const float* vz,
    const float* mass, const float* h_sml,
    float* bx_out, float* by_out, float* bz_out,
    int n, float eta_hall, float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float bx = bx_in[i], by = by_in[i], bz = bz_in[i];
    if (ptype[i] != PTYPE_GAS) {
        bx_out[i] = bx; by_out[i] = by; bz_out[i] = bz; return;
    }
    float b2 = bx*bx + by*by + bz*bz;
    if (b2 <= 0.0f) { bx_out[i] = bx; by_out[i] = by; bz_out[i] = bz; return; }
    float b_norm = sqrtf(b2);
    float h   = h_sml[i];
    float h3  = h * h * h;
    float rho = (h > 0.0f) ? (mass[i] / fmaxf(h3, 1.0e-30f)) : fmaxf(mass[i], 1.0e-30f);
    float theta = eta_hall * b_norm / fmaxf(rho, 1.0e-30f) * dt;
    // Clamp rotation to [-PI, PI]
    if (theta >  3.14159265f) theta =  3.14159265f;
    if (theta < -3.14159265f) theta = -3.14159265f;
    if (fabsf(theta) < 1.0e-15f) { bx_out[i] = bx; by_out[i] = by; bz_out[i] = bz; return; }
    // Rotation axis: k = normalize(v × B)
    float ax = vy[i]*bz - vz[i]*by;
    float ay = vz[i]*bx - vx[i]*bz;
    float az = vx[i]*by - vy[i]*bx;
    float a2 = ax*ax + ay*ay + az*az;
    if (a2 < 1.0e-60f) { bx_out[i] = bx; by_out[i] = by; bz_out[i] = bz; return; }
    float inv_a = rsqrtf(a2);
    float kx = ax*inv_a, ky = ay*inv_a, kz = az*inv_a;
    // Rodrigues: B_new = B cos(θ) + (k × B) sin(θ) + k (k·B)(1 − cos(θ))
    float cos_t = cosf(theta), sin_t = sinf(theta);
    float k_dot_b = kx*bx + ky*by + kz*bz;
    float kcx = ky*bz - kz*by;
    float kcy = kz*bx - kx*bz;
    float kcz = kx*by - ky*bx;
    float one_mc = 1.0f - cos_t;
    bx_out[i] = bx*cos_t + kcx*sin_t + kx*k_dot_b*one_mc;
    by_out[i] = by*cos_t + kcy*sin_t + ky*k_dot_b*one_mc;
    bz_out[i] = bz*cos_t + kcz*sin_t + kz*k_dot_b*one_mc;
}

extern "C" int cuda_mhd_hall_drift(
    const unsigned char* ptype,
    const float* bx_in, const float* by_in, const float* bz_in,
    const float* vx,    const float* vy,    const float* vz,
    const float* mass,  const float* h_sml,
    float* bx_out, float* by_out, float* bz_out,
    int n, float eta_hall, float dt
) {
    if (n <= 0) return 0;
    unsigned char* dp;
    float *dbxi, *dbyi, *dbzi, *dvx, *dvy, *dvz, *dm, *dh;
    float *dbxo, *dbyo, *dbzo;
    if (alloc_copy(&dp,   ptype,  n)) return -1;
    if (alloc_copy(&dbxi, bx_in,  n)) return -1;
    if (alloc_copy(&dbyi, by_in,  n)) return -1;
    if (alloc_copy(&dbzi, bz_in,  n)) return -1;
    if (alloc_copy(&dvx,  vx,     n)) return -1;
    if (alloc_copy(&dvy,  vy,     n)) return -1;
    if (alloc_copy(&dvz,  vz,     n)) return -1;
    if (alloc_copy(&dm,   mass,   n)) return -1;
    if (alloc_copy(&dh,   h_sml,  n)) return -1;
    if (alloc_zero(&dbxo, n)) return -1;
    if (alloc_zero(&dbyo, n)) return -1;
    if (alloc_zero(&dbzo, n)) return -1;
    int blocks = (n + MHD_BLOCK_SIZE - 1) / MHD_BLOCK_SIZE;
    mhd_hall_drift_kernel<<<blocks, MHD_BLOCK_SIZE>>>(
        dp, dbxi, dbyi, dbzi, dvx, dvy, dvz, dm, dh,
        dbxo, dbyo, dbzo, n, eta_hall, dt);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(bx_out, dbxo, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(by_out, dbyo, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bz_out, dbzo, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dp);   cudaFree(dbxi); cudaFree(dbyi); cudaFree(dbzi);
    cudaFree(dvx);  cudaFree(dvy);  cudaFree(dvz);
    cudaFree(dm);   cudaFree(dh);
    cudaFree(dbxo); cudaFree(dbyo); cudaFree(dbzo);
    return 0;
}
