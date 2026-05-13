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
