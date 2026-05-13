#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA-RT] %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                    \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#ifndef RT_BLOCK_SIZE
#define RT_BLOCK_SIZE 256
#endif

static constexpr float SIGMA_HI = 6.3e-18f;
static constexpr float H_NU_0_ERG = 2.179e-11f;
static constexpr float U_CODE_TO_ERG_G = 1.0e10f;
static constexpr unsigned char PTYPE_GAS = 1;

__global__ void rt_energy_xi_photoion_kernel(
    const float* __restrict__ energy,
    const float* __restrict__ flux_x,
    const float* __restrict__ flux_y,
    const float* __restrict__ flux_z,
    float* __restrict__ energy_contrib_out,
    float* __restrict__ xi_out,
    float* __restrict__ gamma_out,
    int n, float dv, float c_red_code, float c_red_cgs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float e = energy[i];
    energy_contrib_out[i] = e * dv;
    if (e < 1.0e-30f) {
        xi_out[i] = 0.0f;
    } else {
        float fmag = sqrtf(flux_x[i] * flux_x[i] + flux_y[i] * flux_y[i] + flux_z[i] * flux_z[i]);
        xi_out[i] = fminf(fmaxf(fmag / (c_red_code * e), 0.0f), 1.0f);
    }
    gamma_out[i] = SIGMA_HI * c_red_cgs * fmaxf(e, 0.0f) / H_NU_0_ERG;
}

__global__ void rt_photoheating_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    const float* __restrict__ internal_energy_in,
    const float* __restrict__ gamma_hi,
    float* __restrict__ internal_energy_out,
    int n_particles, int nx, int ny, int nz, float box_size, float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    float u = internal_energy_in[i];
    if (ptype[i] == PTYPE_GAS) {
        int ix = (int)floorf(px[i] / box_size * (float)nx);
        int iy = (int)floorf(py[i] / box_size * (float)ny);
        int iz = (int)floorf(pz[i] / box_size * (float)nz);
        ix = min(max(ix, 0), nx - 1);
        iy = min(max(iy, 0), ny - 1);
        iz = min(max(iz, 0), nz - 1);
        int cell = iz * ny * nx + iy * nx + ix;
        float gamma = gamma_hi[cell];
        if (gamma >= 1.0e-30f) {
            float delta_u = gamma * dt / U_CODE_TO_ERG_G;
            u += fminf(delta_u, u * 10.0f);
        }
    }
    internal_energy_out[i] = u;
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

extern "C" int cuda_rt_energy_xi_photoion(
    const float* energy, const float* flux_x, const float* flux_y, const float* flux_z,
    float* energy_contrib_out, float* xi_out, float* gamma_out,
    int n, float dv, float c_red_code, float c_red_cgs
) {
    if (n <= 0) return 0;
    float *de, *dfx, *dfy, *dfz, *deo, *dxi, *dgo;
    if (alloc_copy(&de, energy, n) || alloc_copy(&dfx, flux_x, n) ||
        alloc_copy(&dfy, flux_y, n) || alloc_copy(&dfz, flux_z, n) ||
        alloc_zero(&deo, n) || alloc_zero(&dxi, n) || alloc_zero(&dgo, n)) return -1;
    int blocks = (n + RT_BLOCK_SIZE - 1) / RT_BLOCK_SIZE;
    rt_energy_xi_photoion_kernel<<<blocks, RT_BLOCK_SIZE>>>(de, dfx, dfy, dfz, deo, dxi, dgo, n, dv, c_red_code, c_red_cgs);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(energy_contrib_out, deo, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xi_out, dxi, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gamma_out, dgo, bytes, cudaMemcpyDeviceToHost));
    cudaFree(de); cudaFree(dfx); cudaFree(dfy); cudaFree(dfz); cudaFree(deo); cudaFree(dxi); cudaFree(dgo);
    return 0;
}

extern "C" int cuda_rt_photoheating(
    const unsigned char* ptype, const float* px, const float* py, const float* pz,
    const float* internal_energy_in, const float* gamma_hi,
    float* internal_energy_out,
    int n_particles, int nx, int ny, int nz, float box_size, float dt
) {
    if (n_particles <= 0) return 0;
    int n_cells = nx * ny * nz;
    unsigned char* dptype;
    float *dpx, *dpy, *dpz, *du, *dgamma, *duout;
    if (alloc_copy(&dptype, ptype, n_particles) || alloc_copy(&dpx, px, n_particles) ||
        alloc_copy(&dpy, py, n_particles) || alloc_copy(&dpz, pz, n_particles) ||
        alloc_copy(&du, internal_energy_in, n_particles) || alloc_copy(&dgamma, gamma_hi, n_cells) ||
        alloc_zero(&duout, n_particles)) return -1;
    int blocks = (n_particles + RT_BLOCK_SIZE - 1) / RT_BLOCK_SIZE;
    rt_photoheating_kernel<<<blocks, RT_BLOCK_SIZE>>>(dptype, dpx, dpy, dpz, du, dgamma, duout, n_particles, nx, ny, nz, box_size, dt);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(internal_energy_out, duout, (size_t)n_particles * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dpx); cudaFree(dpy); cudaFree(dpz); cudaFree(du); cudaFree(dgamma); cudaFree(duout);
    return 0;
}
