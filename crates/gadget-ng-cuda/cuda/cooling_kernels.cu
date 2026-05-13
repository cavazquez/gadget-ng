#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA-COOL] %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_e));                                    \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#ifndef COOL_BLOCK_SIZE
#define COOL_BLOCK_SIZE 256
#endif

static constexpr unsigned char PTYPE_GAS = 1;

static constexpr float KB_OVER_MH_MU = 8.254e-3f / 0.6f;
static constexpr float LAMBDA_0 = 2.0e-5f;
static constexpr float BETA_COOL = 0.7f;
static constexpr float T_REF = 1.0e4f;
static constexpr float X_H = 0.76f;
static constexpr float Z_SUN = 0.0127f;
static constexpr float LAMBDA_M0 = 3.0e-5f;
static constexpr float LAMBDA_M1 = 1.0e-5f;
static constexpr float UV_NORM = 4.0e-5f;

// Tabla de metalicidades Z/Z_sun (7 bins)
__device__ static const float COOL_TABLE_Z[7] = {0.0f, 1e-4f, 1e-3f, 0.01f, 0.1f, 1.0f, 2.0f};

// Tabla de log10(T/K) (20 bins, 4.0–8.75)
__device__ static const float COOL_TABLE_LOG_T[20] = {
    4.0f, 4.25f, 4.5f, 4.75f, 5.0f, 5.25f, 5.5f, 5.75f,
    6.0f, 6.25f, 6.5f, 6.75f, 7.0f, 7.25f, 7.5f, 7.75f,
    8.0f, 8.25f, 8.5f, 8.75f
};

// Tabla de tasas de enfriamiento (7 bins Z x 20 bins log T)
__device__ static const float COOL_TABLE[7][20] = {
    {0.0f,0.0f,1e-7f,5e-7f,3e-6f,1e-5f,2e-5f,3e-5f,2.5e-5f,2e-5f,1.5e-5f,1e-5f,8e-6f,6e-6f,5e-6f,4.5e-6f,4e-6f,4e-6f,4e-6f,4e-6f},
    {0.0f,0.0f,1.1e-7f,5.2e-7f,3.1e-6f,1.05e-5f,2.1e-5f,3.1e-5f,2.6e-5f,2.1e-5f,1.6e-5f,1.1e-5f,8.2e-6f,6.2e-6f,5.2e-6f,4.6e-6f,4.1e-6f,4.1e-6f,4.1e-6f,4.1e-6f},
    {0.0f,0.0f,2e-7f,8e-7f,4e-6f,1.3e-5f,2.5e-5f,3.5e-5f,3e-5f,2.4e-5f,1.8e-5f,1.3e-5f,1e-5f,8e-6f,6.5e-6f,5.5e-6f,5e-6f,5e-6f,5e-6f,5e-6f},
    {0.0f,0.0f,5e-7f,2e-6f,8e-6f,2.5e-5f,4e-5f,5e-5f,4.5e-5f,3.5e-5f,2.5e-5f,1.8e-5f,1.4e-5f,1.1e-5f,9e-6f,8e-6f,7e-6f,7e-6f,7e-6f,7e-6f},
    {0.0f,0.0f,2e-6f,8e-6f,3e-5f,8e-5f,1.2e-4f,1.4e-4f,1.2e-4f,9e-5f,6.5e-5f,4.5e-5f,3.5e-5f,2.8e-5f,2.3e-5f,2e-5f,1.8e-5f,1.8e-5f,1.8e-5f,1.8e-5f},
    {0.0f,0.0f,1.5e-5f,6e-5f,2e-4f,5e-4f,7e-4f,8e-4f,7e-4f,5e-4f,3.5e-4f,2.5e-4f,1.8e-4f,1.4e-4f,1.1e-4f,9e-5f,8e-5f,8e-5f,8e-5f,8e-5f},
    {0.0f,0.0f,2.5e-5f,1e-4f,3.5e-4f,8e-4f,1.1e-3f,1.2e-3f,1.05e-3f,7.5e-4f,5e-4f,3.5e-4f,2.5e-4f,1.9e-4f,1.5e-4f,1.25e-4f,1.1e-4f,1.1e-4f,1.1e-4f,1.1e-4f}
};

__device__ inline float u_to_temperature_device(float u, float gamma) {
    return u * (gamma - 1.0f) / KB_OVER_MH_MU;
}

__device__ inline float temperature_to_u_device(float tk, float gamma) {
    return tk * KB_OVER_MH_MU / (gamma - 1.0f);
}

__device__ inline float cooling_rate_atomic_device(float u, float gamma, float t_floor_k) {
    float t = u_to_temperature_device(u, gamma);
    if (t <= t_floor_k) return 0.0f;
    return LAMBDA_0 * powf(t / T_REF, BETA_COOL);
}

__device__ inline float cooling_rate_metal_device(float u, float metallicity, float gamma, float t_floor_k) {
    float lambda_hhe = cooling_rate_atomic_device(u, gamma, t_floor_k);
    float t = u_to_temperature_device(u, gamma);
    if (t <= t_floor_k) return 0.0f;
    float z_ratio = fmaxf(metallicity / Z_SUN, 0.0f);
    float lambda_metal = (t < 1.0e7f)
        ? LAMBDA_M0 * powf(t / 1.0e5f, 0.7f)
        : LAMBDA_M1 * powf(t / 1.0e7f, 0.5f);
    return lambda_hhe + z_ratio * lambda_metal;
}

__device__ inline float cooling_rate_tabular_device(float u, float metallicity, float gamma, float t_floor_k) {
    float t = u_to_temperature_device(u, gamma);
    if (t <= t_floor_k || t <= 0.0f) return 0.0f;
    float log_t = log10f(t);
    float log_t_min = COOL_TABLE_LOG_T[0];
    float log_t_max = COOL_TABLE_LOG_T[19];
    float log_t_cl = fminf(fmaxf(log_t, log_t_min), log_t_max);

    int i_t = 0;
    for (int k = 0; k < 19; ++k) {
        if (log_t_cl >= COOL_TABLE_LOG_T[k] && log_t_cl <= COOL_TABLE_LOG_T[k + 1]) {
            i_t = k;
            break;
        }
    }
    float dt_bin = COOL_TABLE_LOG_T[i_t + 1] - COOL_TABLE_LOG_T[i_t];
    float ft = (dt_bin > 0.0f) ? (log_t_cl - COOL_TABLE_LOG_T[i_t]) / dt_bin : 0.0f;

    float z_over_zsun = fmaxf(metallicity / Z_SUN, 0.0f);
    float z_cl = fminf(fmaxf(z_over_zsun, COOL_TABLE_Z[0]), COOL_TABLE_Z[6]);
    int i_z = 0;
    for (int k = 0; k < 6; ++k) {
        if (z_cl >= COOL_TABLE_Z[k] && z_cl <= COOL_TABLE_Z[k + 1]) {
            i_z = k;
            break;
        }
    }
    float dz_bin = COOL_TABLE_Z[i_z + 1] - COOL_TABLE_Z[i_z];
    float fz = (dz_bin > 0.0f) ? (z_cl - COOL_TABLE_Z[i_z]) / dz_bin : 0.0f;

    float l00 = COOL_TABLE[i_z][i_t];
    float l10 = COOL_TABLE[i_z + 1][i_t];
    float l01 = COOL_TABLE[i_z][i_t + 1];
    float l11 = COOL_TABLE[i_z + 1][i_t + 1];
    return fmaxf(l00 * (1.0f - fz) * (1.0f - ft) + l10 * fz * (1.0f - ft)
               + l01 * (1.0f - fz) * ft + l11 * fz * ft, 0.0f);
}

__device__ inline float reionization_switch_device(float redshift, float z_reion) {
    float width = 0.8f;
    return 1.0f / (1.0f + expf((redshift - z_reion) / width));
}

__device__ inline float photo_heating_rate_uvb_device(
    float rho_local, int uv_background_model, float redshift,
    float reionization_redshift, float self_shielding_nh
) {
    if (uv_background_model == 0) return 0.0f;
    float n_h_proxy = fmaxf(X_H * rho_local, 0.0f);
    float shielding = 1.0f / (1.0f + powf(n_h_proxy / fmaxf(self_shielding_nh, 1e-12f), 2.0f));
    float rei = reionization_switch_device(redshift, reionization_redshift);
    return UV_NORM * rei * shielding;
}

__device__ inline float cooling_rate_uvb_device(
    float u, float rho_local, float metallicity, float gamma, float t_floor_k,
    float redshift, float reionization_redshift, int uv_background_model,
    float self_shielding_nh
) {
    float lambda_cool = cooling_rate_tabular_device(u, metallicity, gamma, t_floor_k);
    float gamma_photo = photo_heating_rate_uvb_device(
        rho_local, uv_background_model, redshift, reionization_redshift, self_shielding_nh);
    return lambda_cool - gamma_photo;
}

__global__ void cooling_apply_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ mass,
    const float* __restrict__ smoothing_length,
    float* __restrict__ internal_energy,
    const float* __restrict__ metallicity,
    const float* __restrict__ bx,
    const float* __restrict__ by,
    const float* __restrict__ bz,
    int n, float dt, float gamma, float t_floor_k, float redshift,
    int cooling_kind, float f_mag, float reionization_redshift,
    int uv_background_model, float self_shielding_nh
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ptype[i] != PTYPE_GAS) return;

    float u = internal_energy[i];
    float u_floor = temperature_to_u_device(t_floor_k, gamma);
    if (u <= u_floor) return;

    const float PI = 3.14159265358979323846f;
    float h = fmaxf(smoothing_length[i], 1.0e-10f);
    float rho_local = mass[i] / ((4.0f / 3.0f) * PI * h * h * h);

    float lambda;
    switch (cooling_kind) {
        case 1: // AtomicHHe
            lambda = cooling_rate_atomic_device(u, gamma, t_floor_k);
            break;
        case 2: // MetalCooling
            lambda = cooling_rate_metal_device(u, metallicity[i], gamma, t_floor_k);
            break;
        case 3: // MetalTabular
            lambda = cooling_rate_tabular_device(u, metallicity[i], gamma, t_floor_k);
            break;
        case 4: // UvBackground
            lambda = cooling_rate_uvb_device(u, rho_local, metallicity[i], gamma, t_floor_k,
                                             redshift, reionization_redshift,
                                             uv_background_model, self_shielding_nh);
            break;
        default:
            return;
    }
    if (lambda == 0.0f) return;

    float mag_suppression = 1.0f;
    if (f_mag > 0.0f) {
        float b2 = bx[i] * bx[i] + by[i] * by[i] + bz[i] * bz[i];
        if (b2 > 1.0e-60f) {
            float p_th = (gamma - 1.0f) * rho_local * u;
            float beta = 2.0f * p_th / b2;
            mag_suppression = 1.0f / (1.0f + f_mag / fmaxf(beta, 1.0e-10f));
        }
    }

    float du_dt = -lambda * X_H * X_H * rho_local * mag_suppression;
    float dt_eff;
    if (du_dt < 0.0f) {
        float dt_cool = u / fmaxf(-du_dt, 1.0e-30f);
        dt_eff = fminf(dt, dt_cool);
    } else {
        dt_eff = dt;
    }
    internal_energy[i] = fmaxf(u + du_dt * dt_eff, u_floor);
}

template <typename T>
static int alloc_copy(T** d, const T* h, int n) {
    size_t bytes = (size_t)n * sizeof(T);
    CUDA_CHECK(cudaMalloc(d, bytes));
    CUDA_CHECK(cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int cuda_cooling_apply(
    const unsigned char* ptype, const float* mass, const float* smoothing_length,
    float* internal_energy, const float* metallicity,
    const float* bx, const float* by, const float* bz,
    int n, float dt, float gamma, float t_floor_k, float redshift,
    int cooling_kind, float f_mag, float reionization_redshift,
    int uv_background_model, float self_shielding_nh
) {
    if (n <= 0) return 0;
    unsigned char* dptype;
    float *dmass, *dh, *du, *dmet, *dbx, *dby, *dbz;
    if (alloc_copy(&dptype, ptype, n) || alloc_copy(&dmass, mass, n) ||
        alloc_copy(&dh, smoothing_length, n) || alloc_copy(&du, internal_energy, n) ||
        alloc_copy(&dmet, metallicity, n) ||
        alloc_copy(&dbx, bx, n) || alloc_copy(&dby, by, n) || alloc_copy(&dbz, bz, n)) return -1;
    int blocks = (n + COOL_BLOCK_SIZE - 1) / COOL_BLOCK_SIZE;
    cooling_apply_kernel<<<blocks, COOL_BLOCK_SIZE>>>(
        dptype, dmass, dh, du, dmet, dbx, dby, dbz,
        n, dt, gamma, t_floor_k, redshift,
        cooling_kind, f_mag, reionization_redshift,
        uv_background_model, self_shielding_nh);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(internal_energy, du, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dmass); cudaFree(dh); cudaFree(du); cudaFree(dmet);
    cudaFree(dbx); cudaFree(dby); cudaFree(dbz);
    return 0;
}
