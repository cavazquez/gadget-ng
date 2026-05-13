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
