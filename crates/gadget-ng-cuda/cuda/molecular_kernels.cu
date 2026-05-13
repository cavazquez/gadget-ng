#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA-MOL] %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                    \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#ifndef MOL_BLOCK_SIZE
#define MOL_BLOCK_SIZE 256
#endif

static constexpr unsigned char PTYPE_GAS = 1;

__device__ inline float dust_uv_opacity_device(
    float kappa, float dust_to_gas, float rho, float h
) {
    return kappa * dust_to_gas * rho * h;
}

__device__ inline float effective_kappa_device(
    int species_model,
    float kappa_dust_uv, float kappa_silicate_uv, float kappa_graphite_uv,
    float silicate_fraction, float graphite_fraction
) {
    if (species_model == 0) { // Single
        return fmaxf(kappa_dust_uv, 0.0f);
    }
    // SilicateGraphite
    float sil = fmaxf(silicate_fraction, 0.0f);
    float gra = fmaxf(graphite_fraction, 0.0f);
    float sum = sil + gra;
    if (sum <= 0.0f) return fmaxf(kappa_dust_uv, 0.0f);
    return (sil / sum) * fmaxf(kappa_silicate_uv, 0.0f)
         + (gra / sum) * fmaxf(kappa_graphite_uv, 0.0f);
}

__device__ inline float h2_shielding_device(
    float dust_to_gas, float rho, float h,
    float boost, float eff_kappa
) {
    if (dust_to_gas <= 0.0f) return 1.0f;
    float tau = fmaxf(dust_uv_opacity_device(eff_kappa, dust_to_gas, rho, h), 0.0f);
    return 1.0f + fmaxf(boost, 0.0f) * (1.0f - expf(-tau));
}

__global__ void h2_update_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ mass,
    const float* __restrict__ smoothing_length,
    float* __restrict__ h2_fraction,
    const float* __restrict__ dust_to_gas,
    int n, float dt, float rho_h2_threshold, float t_dissoc,
    int dust_enabled, float h2_shielding_boost,
    float kappa_dust_uv, float kappa_silicate_uv, float kappa_graphite_uv,
    float silicate_fraction, float graphite_fraction, int species_model
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ptype[i] != PTYPE_GAS) return;

    float h = fmaxf(smoothing_length[i], 1.0e-10f);
    float rho = mass[i] / (h * h * h);

    float shielding;
    if (dust_enabled) {
        float eff_kappa = effective_kappa_device(
            species_model, kappa_dust_uv, kappa_silicate_uv, kappa_graphite_uv,
            silicate_fraction, graphite_fraction);
        shielding = h2_shielding_device(
            dust_to_gas[i], rho, h, h2_shielding_boost, eff_kappa);
    } else {
        shielding = 1.0f;
    }

    float h2 = h2_fraction[i];
    if (rho > rho_h2_threshold) {
        float h2_eq = fminf((rho / rho_h2_threshold) * shielding, 1.0f);
        float tau = fminf(dt / t_dissoc, 1.0f);
        h2 += tau * (h2_eq - h2);
    } else {
        float t_dissoc_eff = t_dissoc * fmaxf(shielding, 1.0f);
        h2 *= expf(-dt / t_dissoc_eff);
    }
    h2_fraction[i] = fminf(fmaxf(h2, 0.0f), 1.0f);
}

template <typename T>
static int alloc_copy(T** d, const T* h, int n) {
    size_t bytes = (size_t)n * sizeof(T);
    CUDA_CHECK(cudaMalloc(d, bytes));
    CUDA_CHECK(cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int cuda_h2_update(
    const unsigned char* ptype, const float* mass, const float* smoothing_length,
    float* h2_fraction, const float* dust_to_gas,
    int n, float dt, float rho_h2_threshold, float t_dissoc,
    int dust_enabled, float h2_shielding_boost,
    float kappa_dust_uv, float kappa_silicate_uv, float kappa_graphite_uv,
    float silicate_fraction, float graphite_fraction, int species_model
) {
    if (n <= 0) return 0;
    unsigned char* dptype;
    float *dmass, *dh, *dh2, *ddg;
    if (alloc_copy(&dptype, ptype, n) || alloc_copy(&dmass, mass, n) ||
        alloc_copy(&dh, smoothing_length, n) || alloc_copy(&dh2, h2_fraction, n) ||
        alloc_copy(&ddg, dust_to_gas, n)) return -1;
    int blocks = (n + MOL_BLOCK_SIZE - 1) / MOL_BLOCK_SIZE;
    h2_update_kernel<<<blocks, MOL_BLOCK_SIZE>>>(
        dptype, dmass, dh, dh2, ddg,
        n, dt, rho_h2_threshold, t_dissoc,
        dust_enabled, h2_shielding_boost,
        kappa_dust_uv, kappa_silicate_uv, kappa_graphite_uv,
        silicate_fraction, graphite_fraction, species_model);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h2_fraction, dh2, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dmass); cudaFree(dh); cudaFree(dh2); cudaFree(ddg);
    return 0;
}
