#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA-DUST] %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_e));                                    \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#ifndef DUST_BLOCK_SIZE
#define DUST_BLOCK_SIZE 256
#endif

static constexpr unsigned char PTYPE_GAS = 1;
static constexpr float KB_OVER_MH_MU = 8.254e-3f / 0.6f;

__device__ inline float u_to_temperature_device(float u, float gamma) {
    return u * (gamma - 1.0f) / KB_OVER_MH_MU;
}

__global__ void dust_update_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ mass,
    const float* __restrict__ smoothing_length,
    const float* __restrict__ internal_energy,
    float* __restrict__ dust_to_gas,
    const float* __restrict__ metallicity,
    int n, float gamma, float dt,
    float d_to_g_max, float tau_grow, float t_destroy_k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ptype[i] != PTYPE_GAS) return;

    float dg = dust_to_gas[i];
    float u = fmaxf(internal_energy[i], 0.0f);
    float t = u_to_temperature_device(u, gamma);
    float z = fminf(fmaxf(metallicity[i], 0.0f), 1.0f);

    if (t < t_destroy_k) {
        float d_target = d_to_g_max * z;
        float tau = fmaxf(tau_grow, 1.0e-10f);
        dg += z * (d_target - dg) * dt / tau;
    } else {
        float t_ratio = t_destroy_k / fmaxf(t, t_destroy_k);
        float tau_sputter = tau_grow * fmaxf(t_ratio * t_ratio, 1.0e-6f);
        dg *= expf(-dt / tau_sputter);
    }
    dust_to_gas[i] = fminf(fmaxf(dg, 0.0f), d_to_g_max);
}

__global__ void dust_radiation_pressure_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ mass,
    const float* __restrict__ smoothing_length,
    const float* __restrict__ dust_to_gas,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    float* __restrict__ pos_z,
    int n, float dt, float z_reference,
    float kappa, float j_uv, float box_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ptype[i] != PTYPE_GAS) return;
    if (dust_to_gas[i] <= 0.0f) return;

    const float PI = 3.14159265358979323846f;
    float h = fmaxf(smoothing_length[i], 1.0e-30f);
    float rho = mass[i] / fmaxf((4.0f / 3.0f) * PI * h * h * h, 1.0e-100f);
    float a_mag = kappa * dust_to_gas[i] * j_uv / fmaxf(rho, 1.0e-30f);
    float dir = (pos_z[i] >= z_reference) ? 1.0f : -1.0f;
    vz[i] += dir * a_mag * dt;
}

template <typename T>
static int alloc_copy(T** d, const T* h, int n) {
    size_t bytes = (size_t)n * sizeof(T);
    CUDA_CHECK(cudaMalloc(d, bytes));
    CUDA_CHECK(cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int cuda_dust_update(
    const unsigned char* ptype, const float* mass, const float* smoothing_length,
    const float* internal_energy, float* dust_to_gas, const float* metallicity,
    int n, float gamma, float dt,
    float d_to_g_max, float tau_grow, float t_destroy_k
) {
    if (n <= 0) return 0;
    unsigned char* dptype;
    float *dmass, *dh, *du, *ddg, *dmet;
    if (alloc_copy(&dptype, ptype, n) || alloc_copy(&dmass, mass, n) ||
        alloc_copy(&dh, smoothing_length, n) || alloc_copy(&du, internal_energy, n) ||
        alloc_copy(&ddg, dust_to_gas, n) || alloc_copy(&dmet, metallicity, n)) return -1;
    int blocks = (n + DUST_BLOCK_SIZE - 1) / DUST_BLOCK_SIZE;
    dust_update_kernel<<<blocks, DUST_BLOCK_SIZE>>>(
        dptype, dmass, dh, du, ddg, dmet, n, gamma, dt, d_to_g_max, tau_grow, t_destroy_k);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dust_to_gas, ddg, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dmass); cudaFree(dh); cudaFree(du); cudaFree(ddg); cudaFree(dmet);
    return 0;
}

extern "C" int cuda_dust_radiation_pressure(
    const unsigned char* ptype, const float* mass, const float* smoothing_length,
    const float* dust_to_gas, float* vx, float* vy, float* vz, float* pos_z,
    int n, float dt, float z_reference, float kappa, float j_uv, float box_size
) {
    if (n <= 0) return 0;
    unsigned char* dptype;
    float *dmass, *dh, *ddg, *dvx, *dvy, *dvz, *dpz;
    if (alloc_copy(&dptype, ptype, n) || alloc_copy(&dmass, mass, n) ||
        alloc_copy(&dh, smoothing_length, n) || alloc_copy(&ddg, dust_to_gas, n) ||
        alloc_copy(&dvx, vx, n) || alloc_copy(&dvy, vy, n) || alloc_copy(&dvz, vz, n) ||
        alloc_copy(&dpz, pos_z, n)) return -1;
    int blocks = (n + DUST_BLOCK_SIZE - 1) / DUST_BLOCK_SIZE;
    dust_radiation_pressure_kernel<<<blocks, DUST_BLOCK_SIZE>>>(
        dptype, dmass, dh, ddg, dvx, dvy, dvz, dpz, n, dt, z_reference, kappa, j_uv, box_size);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(vx, dvx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vy, dvy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vz, dvz, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dmass); cudaFree(dh); cudaFree(ddg);
    cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dpz);
    return 0;
}
