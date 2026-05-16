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

// ── Kernel M1 advección HLL completa ─────────────────────────────────────────

__device__ inline float m1_eddington(float xi) {
    float xi2 = xi * xi;
    float d = 5.0f + 2.0f * sqrtf(fmaxf(4.0f - 3.0f * xi2, 0.0f));
    return (3.0f + 4.0f * xi2) / d;
}

// Flujo HLL en la interfaz entre celdas L y R a lo largo de la dirección x.
// Devuelve (flux_E, flux_Fx).
__device__ inline void hll_flux(
    float el, float fxl,
    float er, float fxr,
    float c_red,
    float* fe_out, float* ff_out
) {
    float xi_l = fminf(fmaxf(fabsf(fxl) / fmaxf(c_red * el, 1.0e-30f), 0.0f), 1.0f);
    float xi_r = fminf(fmaxf(fabsf(fxr) / fmaxf(c_red * er, 1.0e-30f), 0.0f), 1.0f);
    float f_l  = m1_eddington(xi_l);
    float f_r  = m1_eddington(xi_r);
    // HLL wave speeds: ±c_red
    float s_l = -c_red;
    float s_r =  c_red;
    float flux_e_l  = fxl;
    float flux_e_r  = fxr;
    float flux_fx_l = c_red * c_red * f_l * el;
    float flux_fx_r = c_red * c_red * f_r * er;
    float denom = s_r - s_l;
    if (fabsf(denom) < 1.0e-30f) {
        *fe_out = 0.5f * (flux_e_l + flux_e_r);
        *ff_out = 0.5f * (flux_fx_l + flux_fx_r);
    } else {
        *fe_out = (s_r * flux_e_l  - s_l * flux_e_r  + s_r * s_l * (er  - el))  / denom;
        *ff_out = (s_r * flux_fx_l - s_l * flux_fx_r + s_r * s_l * (fxr - fxl)) / denom;
    }
}

// Kernel principal: un sub-paso M1 Godunov de primer orden.
// Cada thread procesa una celda i = ix + nx*(iy + ny*iz).
// Los arrays *_out deben ser distintos de *_in (no in-place).
__global__ void m1_substep_kernel(
    const float* __restrict__ e_in,
    const float* __restrict__ fx_in,
    const float* __restrict__ fy_in,
    const float* __restrict__ fz_in,
    float* __restrict__ e_out,
    float* __restrict__ fx_out,
    float* __restrict__ fy_out,
    float* __restrict__ fz_out,
    int nx, int ny, int nz, float dtdx, float c_red, float decay
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n3 = nx * ny * nz;
    if (i >= n3) return;

    int ix = i % nx;
    int iy = (i / nx) % ny;
    int iz = i / (nx * ny);

    int ix_l = (ix + nx - 1) % nx;
    int ix_r = (ix + 1) % nx;
    int iy_l = (iy + ny - 1) % ny;
    int iy_r = (iy + 1) % ny;
    int iz_l = (iz + nz - 1) % nz;
    int iz_r = (iz + 1) % nz;

    // ── Direction X ──────────────────────────────────────────────────────
    int ilx = iz * ny * nx + iy * nx + ix_l;
    int irx = iz * ny * nx + iy * nx + ix_r;
    float fe_xl, ffx_xl;
    float fe_xr, ffx_xr;
    hll_flux(e_in[ilx], fx_in[ilx], e_in[i], fx_in[i], c_red, &fe_xl, &ffx_xl);
    hll_flux(e_in[i],   fx_in[i],   e_in[irx], fx_in[irx], c_red, &fe_xr, &ffx_xr);

    // ── Direction Y (swap fx↔fy to reuse same HLL routine) ───────────────
    int ily = iz * ny * nx + iy_l * nx + ix;
    int iry = iz * ny * nx + iy_r * nx + ix;
    float fe_yl, ffy_yl;
    float fe_yr, ffy_yr;
    hll_flux(e_in[ily], fy_in[ily], e_in[i], fy_in[i], c_red, &fe_yl, &ffy_yl);
    hll_flux(e_in[i],   fy_in[i],   e_in[iry], fy_in[iry], c_red, &fe_yr, &ffy_yr);

    // ── Direction Z ──────────────────────────────────────────────────────
    int ilz = iz_l * ny * nx + iy * nx + ix;
    int irz = iz_r * ny * nx + iy * nx + ix;
    float fe_zl, ffz_zl;
    float fe_zr, ffz_zr;
    hll_flux(e_in[ilz], fz_in[ilz], e_in[i], fz_in[i], c_red, &fe_zl, &ffz_zl);
    hll_flux(e_in[i],   fz_in[i],   e_in[irz], fz_in[irz], c_red, &fe_zr, &ffz_zr);

    // ── Godunov update ────────────────────────────────────────────────────
    float de  = -dtdx * ((fe_xr  - fe_xl)  + (fe_yr  - fe_yl)  + (fe_zr  - fe_zl));
    float dfx = -dtdx *  (ffx_xr - ffx_xl);
    float dfy = -dtdx *  (ffy_yr - ffy_yl);
    float dfz = -dtdx *  (ffz_zr - ffz_zl);

    e_out[i]  = fmaxf(e_in[i]  + de,  0.0f) * decay;
    fx_out[i] = (fx_in[i] + dfx) * decay;
    fy_out[i] = (fy_in[i] + dfy) * decay;
    fz_out[i] = (fz_in[i] + dfz) * decay;
}

// Lanzador que gestiona sus propios buffers (sin pool persistente).
// Parámetros:
//   e/fx/fy/fz: arrays host de entrada/salida, modificados in-place.
//   nx, ny, nz: dimensiones de la malla.
//   dx: espaciado de celda.
//   dt: paso de tiempo total.
//   c_red: velocidad de luz reducida.
//   kappa: opacidad total (abs + scat).
//   n_substeps: número de sub-pasos (pre-calculado por el caller).
extern "C" int cuda_rt_m1_substep(
    float* e_host, float* fx_host, float* fy_host, float* fz_host,
    int nx, int ny, int nz,
    float dx, float dt_sub, float c_red, float kappa
) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return -1;
    int n3 = nx * ny * nz;
    if (n3 <= 0) return 0;

    float *d_e0, *d_fx0, *d_fy0, *d_fz0;
    float *d_e1, *d_fx1, *d_fy1, *d_fz1;
    size_t bytes = (size_t)n3 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_e0,  bytes)); CUDA_CHECK(cudaMalloc(&d_fx0, bytes));
    CUDA_CHECK(cudaMalloc(&d_fy0, bytes)); CUDA_CHECK(cudaMalloc(&d_fz0, bytes));
    CUDA_CHECK(cudaMalloc(&d_e1,  bytes)); CUDA_CHECK(cudaMalloc(&d_fx1, bytes));
    CUDA_CHECK(cudaMalloc(&d_fy1, bytes)); CUDA_CHECK(cudaMalloc(&d_fz1, bytes));

    CUDA_CHECK(cudaMemcpy(d_e0,  e_host,  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fx0, fx_host, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fy0, fy_host, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fz0, fz_host, bytes, cudaMemcpyHostToDevice));

    float dtdx  = dt_sub / dx;
    float decay = expf(-c_red * kappa * dt_sub);
    int blocks  = (n3 + RT_BLOCK_SIZE - 1) / RT_BLOCK_SIZE;

    m1_substep_kernel<<<blocks, RT_BLOCK_SIZE>>>(
        d_e0, d_fx0, d_fy0, d_fz0,
        d_e1, d_fx1, d_fy1, d_fz1,
        nx, ny, nz, dtdx, c_red, decay
    );
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(e_host,  d_e1,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fx_host, d_fx1, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fy_host, d_fy1, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fz_host, d_fz1, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_e0); cudaFree(d_fx0); cudaFree(d_fy0); cudaFree(d_fz0);
    cudaFree(d_e1); cudaFree(d_fx1); cudaFree(d_fy1); cudaFree(d_fz1);
    return 0;
}

// ── RT chemistry rates: Γ_HI por partícula (NGP lookup) ──────────────────────

static constexpr float SIGMA_HI_CHEM  = 6.3e-18f;
static constexpr float H_NU_0_ERG_CHEM = 2.179e-11f;

// 1 hilo por partícula gas; busca índice NGP en el campo E y calcula Γ_HI.
__global__ void rt_chemistry_rates_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    const float* __restrict__ energy_density,  // grilla nx*ny*nz
    float* __restrict__ gamma_hi_out,
    int n_particles, int nx, int ny, int nz,
    float box_size, float c_red_cgs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (ptype[i] != PTYPE_GAS) {
        gamma_hi_out[i] = 0.0f;
        return;
    }
    int ix = (int)floorf(px[i] / box_size * (float)nx);
    int iy = (int)floorf(py[i] / box_size * (float)ny);
    int iz = (int)floorf(pz[i] / box_size * (float)nz);
    ix = min(max(ix, 0), nx - 1);
    iy = min(max(iy, 0), ny - 1);
    iz = min(max(iz, 0), nz - 1);
    float e = fmaxf(energy_density[iz * ny * nx + iy * nx + ix], 0.0f);
    gamma_hi_out[i] = SIGMA_HI_CHEM * c_red_cgs * e / H_NU_0_ERG_CHEM;
}

// 1 hilo por partícula gas; aplica cooling_rate_approx y reduce internal_energy.
// cooling_rate [erg/cm³/s] → delta_u = rate * dt / (n_h_ref² * U_CODE_TO_ERG_G)
__global__ void rt_cooling_apply_kernel(
    const unsigned char* __restrict__ ptype,
    float* __restrict__ u_inout,
    const float* __restrict__ x_e,
    float gamma_eos, float n_h_ref, float dt, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || ptype[i] != PTYPE_GAS) return;
    float u = u_inout[i];
    // Temperatura (en K) desde energía interna: T = (γ-1) × u × U_CODE
    float t_k = fmaxf((gamma_eos - 1.0f) * u * U_CODE_TO_ERG_G / 1.38065e-16f, 1.0f);
    float xe = fmaxf(x_e[i], 0.0f);
    // Bremsstrahlung + Lyα
    float brems = 1.42e-27f * sqrtf(t_k) * n_h_ref * n_h_ref * xe;
    float lya = 7.5e-19f * expf(-118348.0f / t_k)
                / (1.0f + sqrtf(t_k / 1.0e5f)) * n_h_ref * n_h_ref * xe;
    float rate = brems + lya;  // erg/cm³/s
    float delta_u = rate * dt / (n_h_ref * n_h_ref * U_CODE_TO_ERG_G);
    u_inout[i] = fmaxf(u - delta_u, u * 0.01f);
}

extern "C" int cuda_rt_chemistry_rates(
    const unsigned char* ptype,
    const float* px, const float* py, const float* pz,
    const float* energy_density,
    float* gamma_hi_out,
    int n_particles, int n_cells,
    int nx, int ny, int nz,
    float box_size, float c_red_cgs
) {
    if (n_particles <= 0) return 0;
    unsigned char* dptype; float *dpx, *dpy, *dpz, *de, *dgout;
    if (alloc_copy(&dptype, ptype, n_particles) ||
        alloc_copy(&dpx, px, n_particles) || alloc_copy(&dpy, py, n_particles) ||
        alloc_copy(&dpz, pz, n_particles) || alloc_copy(&de, energy_density, n_cells) ||
        alloc_zero(&dgout, n_particles)) return -1;
    int blocks = (n_particles + RT_BLOCK_SIZE - 1) / RT_BLOCK_SIZE;
    rt_chemistry_rates_kernel<<<blocks, RT_BLOCK_SIZE>>>(
        dptype, dpx, dpy, dpz, de, dgout,
        n_particles, nx, ny, nz, box_size, c_red_cgs);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gamma_hi_out, dgout, (size_t)n_particles * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dpx); cudaFree(dpy); cudaFree(dpz); cudaFree(de); cudaFree(dgout);
    return 0;
}

extern "C" int cuda_rt_cooling_apply(
    const unsigned char* ptype, float* u_inout,
    const float* x_e,
    int n, float gamma_eos, float n_h_ref, float dt
) {
    if (n <= 0) return 0;
    unsigned char* dptype; float *du, *dxe;
    if (alloc_copy(&dptype, ptype, n) || alloc_copy(&du, u_inout, n) ||
        alloc_copy(&dxe, x_e, n)) return -1;
    int blocks = (n + RT_BLOCK_SIZE - 1) / RT_BLOCK_SIZE;
    rt_cooling_apply_kernel<<<blocks, RT_BLOCK_SIZE>>>(dptype, du, dxe, gamma_eos, n_h_ref, dt, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(u_inout, du, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(du); cudaFree(dxe);
    return 0;
}

// ── RT chemistry stiff solver: subciclo implícito por partícula ───────────────

static constexpr float F_HE_F  = 0.0789f;
static constexpr float F_D_F   = 2.5e-5f;
static constexpr float T_HI_F  = 157809.1f;
static constexpr float T_HEI_F = 285335.4f;
static constexpr float T_HEII_F = 631515.0f;
static constexpr int   CHEM_MAX_SUB = 2000;

__device__ static inline float chem_alpha_hii(float t) {
    float a = 315614.0f / t;
    return 2.753e-14f * powf(a, 1.5f)
           / powf(1.0f + powf(115188.0f / t, 0.407f), 2.242f);
}
__device__ static inline float chem_alpha_heii(float t) {
    return 1.26e-14f * powf(470000.0f / t, 0.75f);
}
__device__ static inline float chem_alpha_heiii(float t) {
    float a = 1263030.0f / t;
    return 4.0f * 2.753e-14f * powf(a, 1.5f)
           / powf(1.0f + powf(460751.0f / t, 0.407f), 2.242f);
}
__device__ static inline float chem_beta_hi(float t) {
    return 5.85e-11f * sqrtf(t) * expf(-T_HI_F / t) / (1.0f + sqrtf(t / 1.0e5f));
}
__device__ static inline float chem_beta_hei(float t) {
    return 2.38e-11f * sqrtf(t) * expf(-T_HEI_F / t) / (1.0f + sqrtf(t / 1.0e5f));
}
__device__ static inline float chem_beta_heii(float t) {
    return 5.68e-12f * sqrtf(t) * expf(-T_HEII_F / t) / (1.0f + sqrtf(t / 1.0e5f));
}
__device__ static inline float chem_k_hm_form(float t) {
    return 1.4e-18f * powf(t, 0.928f) * expf(-t / 16200.0f);
}
__device__ static inline float chem_k_h2_collisional(float t) {
    return 5.6e-11f * expf(-52000.0f / t);
}
__device__ static inline float chem_k_d_ionex(float t) {
    return 3.7e-10f * expf(-43.0f / t);
}
__device__ static inline float chem_k_d_recex(float t) {
    return fminf(3.7e-10f * expf(43.0f / t), 3.7e-9f);
}
__device__ static inline float chem_k_hd_dest(float t) {
    return 1.0e-9f * expf(-464.0f / t);
}

// 1 hilo por partícula gas; corre el bucle subcíclico completo
__global__ void rt_chemistry_stiff_kernel(
    const unsigned char* __restrict__ ptype,
    float* __restrict__ x_hi,   float* __restrict__ x_hii,
    float* __restrict__ x_hei,  float* __restrict__ x_heii,
    float* __restrict__ x_heiii,float* __restrict__ x_e,
    float* __restrict__ x_hm,   float* __restrict__ x_h2,
    float* __restrict__ x_h2p,  float* __restrict__ x_d,
    float* __restrict__ x_dp,   float* __restrict__ x_hd,
    const float* __restrict__ gamma_hi,
    const float* __restrict__ temperature,
    float dt, float n_h_ref, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || ptype[i] != PTYPE_GAS) return;

    float ghi = fmaxf(gamma_hi[i], 0.0f);
    float t   = fmaxf(temperature[i], 1.0f);

    float shi   = x_hi[i];
    float shii  = x_hii[i];
    float shei  = x_hei[i];
    float sheii = x_heii[i];
    float sheiii= x_heiii[i];
    float se    = x_e[i];
    float shm   = x_hm[i];
    float sh2   = x_h2[i];
    float sh2p  = x_h2p[i];
    float sd    = x_d[i];
    float sdp   = x_dp[i];
    float shd   = x_hd[i];

    // Tasas a temperatura fija
    float a_hii   = chem_alpha_hii(t);
    float a_heii  = chem_alpha_heii(t);
    float a_heiii = chem_alpha_heiii(t);
    float b_hi    = chem_beta_hi(t);
    float b_hei   = chem_beta_hei(t);
    float b_heii  = chem_beta_heii(t);
    float k_hm    = chem_k_hm_form(t);
    float k_h2_d  = chem_k_h2_collisional(t);
    float k_dion  = chem_k_d_ionex(t);
    float k_drec  = chem_k_d_recex(t);
    float k_hd_d  = chem_k_hd_dest(t);
    float k_h2_hm = 1.3e-9f;
    float k_h2p_f = 1.85e-23f * powf(fminf(t, 3.16e7f), 1.8f);
    float k_h2_h2p= 6.4e-10f;
    float k_hd_f  = 1.0e-9f;

    float t_elapsed = 0.0f;
    for (int sub = 0; sub < CHEM_MAX_SUB && t_elapsed < dt; ++sub) {
        float xe = fmaxf(se, 1.0e-20f);
        float rate_hii = fabsf((ghi + b_hi * xe) * shi - a_hii * xe * shii);
        float rate_hei = fabsf((b_hei * xe) * shei - a_heii * xe * sheii
                               - b_heii * xe * sheii + a_heiii * xe * sheiii);
        float h2_form  = k_hm * shi * xe + k_h2p_f * shi * fmaxf(shii, 0.0f);
        float h2_dest  = k_h2_d * sh2 * fmaxf(shi, 0.0f);
        float hd_form  = k_hd_f * sdp * sh2;
        float hd_dest  = k_hd_d * shd * fmaxf(shii, 0.0f);
        float max_rate = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(rate_hii, rate_hei),
                                  h2_form), h2_dest), hd_form), hd_dest);
        max_rate = fmaxf(max_rate, 1.0e-30f);
        float dt_sub = fminf(0.1f / max_rate, dt - t_elapsed);
        dt_sub = fmaxf(dt_sub, 1.0e-30f);

        // HII implícito
        float i_hi  = (ghi + b_hi * xe) * shi;
        shii = (shii + dt_sub * i_hi) / (1.0f + dt_sub * a_hii * xe);
        // HeII/HeIII implícito
        float i_hei = (b_hei * xe) * shei;
        float d_heii = dt_sub * (a_heii * xe + b_heii * xe);
        sheii = (sheii + dt_sub * i_hei + dt_sub * a_heiii * xe * sheiii) / (1.0f + d_heii);
        sheiii= (sheiii + dt_sub * b_heii * xe * sheii) / (1.0f + dt_sub * a_heiii * xe);
        // H2 implícito
        float i_h2 = h2_form + k_h2_h2p * sh2p;
        sh2 = (sh2 + dt_sub * i_h2) / (1.0f + dt_sub * (k_h2_d * fmaxf(shi, 0.0f)));
        // H2+ simples
        sh2p = fmaxf(sh2p + dt_sub * k_h2p_f * shi * fmaxf(shii, 0.0f)
                     - dt_sub * k_h2_h2p * sh2p, 0.0f);
        // H- simple
        shm  = fmaxf(shm + dt_sub * k_hm * shi * xe
                     - dt_sub * k_h2_hm * shm, 0.0f);
        // D/D+ charge exchange
        sdp  = fmaxf((sdp + dt_sub * k_dion * sd * fmaxf(shii, 0.0f))
                     / (1.0f + dt_sub * k_drec * fmaxf(shi, 0.0f)), 0.0f);
        sd   = fmaxf(F_D_F - sdp - shd, 0.0f);
        // HD implícito
        shd  = (shd + dt_sub * k_hd_f * sdp * sh2) / (1.0f + dt_sub * k_hd_d * fmaxf(shii, 0.0f));
        // Reconstruir conservados
        shi  = fmaxf(1.0f - shii - shm - 2.0f * sh2 - sh2p, 0.0f);
        shei = fmaxf(F_HE_F - sheii - sheiii, 0.0f);
        se   = shii + sheii + 2.0f * sheiii + sdp;
        // Clamp
        shii  = fmaxf(fminf(shii,  1.0f), 0.0f);
        sheii = fmaxf(fminf(sheii, F_HE_F), 0.0f);
        sheiii= fmaxf(fminf(sheiii,F_HE_F), 0.0f);
        sh2   = fmaxf(fminf(sh2,  0.5f), 0.0f);
        shd   = fmaxf(fminf(shd,  F_D_F), 0.0f);
        se    = fmaxf(se, 0.0f);

        t_elapsed += dt_sub;
    }

    x_hi[i]   = shi;   x_hii[i]  = shii;
    x_hei[i]  = shei;  x_heii[i] = sheii;
    x_heiii[i]= sheiii; x_e[i]   = se;
    x_hm[i]  = shm;   x_h2[i]   = sh2;
    x_h2p[i] = sh2p;  x_d[i]    = sd;
    x_dp[i]  = sdp;   x_hd[i]   = shd;
}

extern "C" int cuda_rt_chemistry_stiff(
    const unsigned char* ptype,
    float* x_hi, float* x_hii, float* x_hei, float* x_heii, float* x_heiii, float* x_e,
    float* x_hm, float* x_h2, float* x_h2p, float* x_d, float* x_dp, float* x_hd,
    const float* gamma_hi, const float* temperature,
    int n, float dt, float n_h_ref
) {
    if (n <= 0) return 0;
    unsigned char* dptype;
    float *dxhi, *dxhii, *dxhei, *dxheii, *dxheiii, *dxe;
    float *dxhm, *dxh2, *dxh2p, *dxd, *dxdp, *dxhd;
    float *dghi, *dtemp;

#define ALLOC_CHEM(ptr, src) if (alloc_copy(&(ptr), (src), n)) return -1
    ALLOC_CHEM(dptype,  ptype);
    ALLOC_CHEM(dxhi,    x_hi);   ALLOC_CHEM(dxhii,    x_hii);
    ALLOC_CHEM(dxhei,   x_hei);  ALLOC_CHEM(dxheii,   x_heii);
    ALLOC_CHEM(dxheiii, x_heiii); ALLOC_CHEM(dxe,      x_e);
    ALLOC_CHEM(dxhm,    x_hm);   ALLOC_CHEM(dxh2,     x_h2);
    ALLOC_CHEM(dxh2p,   x_h2p);  ALLOC_CHEM(dxd,      x_d);
    ALLOC_CHEM(dxdp,    x_dp);   ALLOC_CHEM(dxhd,     x_hd);
    ALLOC_CHEM(dghi,    gamma_hi); ALLOC_CHEM(dtemp,   temperature);
#undef ALLOC_CHEM

    int blocks = (n + RT_BLOCK_SIZE - 1) / RT_BLOCK_SIZE;
    rt_chemistry_stiff_kernel<<<blocks, RT_BLOCK_SIZE>>>(
        dptype,
        dxhi, dxhii, dxhei, dxheii, dxheiii, dxe,
        dxhm, dxh2, dxh2p, dxd, dxdp, dxhd,
        dghi, dtemp, dt, n_h_ref, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

#define DL_CHEM(dst, ptr) CUDA_CHECK(cudaMemcpy((dst), (ptr), (size_t)n * sizeof(float), cudaMemcpyDeviceToHost))
    DL_CHEM(x_hi,    dxhi);   DL_CHEM(x_hii,    dxhii);
    DL_CHEM(x_hei,   dxhei);  DL_CHEM(x_heii,   dxheii);
    DL_CHEM(x_heiii, dxheiii); DL_CHEM(x_e,      dxe);
    DL_CHEM(x_hm,    dxhm);   DL_CHEM(x_h2,     dxh2);
    DL_CHEM(x_h2p,   dxh2p);  DL_CHEM(x_d,      dxd);
    DL_CHEM(x_dp,    dxdp);   DL_CHEM(x_hd,     dxhd);
#undef DL_CHEM

    cudaFree(dptype);
    cudaFree(dxhi); cudaFree(dxhii); cudaFree(dxhei); cudaFree(dxheii); cudaFree(dxheiii); cudaFree(dxe);
    cudaFree(dxhm); cudaFree(dxh2);  cudaFree(dxh2p); cudaFree(dxd);   cudaFree(dxdp); cudaFree(dxhd);
    cudaFree(dghi); cudaFree(dtemp);
    return 0;
}

// ── RT reionization stats: reducción paralela sobre x_hii ────────────────────

__global__ void rt_reionization_stats_kernel(
    const float* __restrict__ x_hii,
    int n,
    double* __restrict__ sum_xhii_out,
    double* __restrict__ sum_sq_out,
    int* __restrict__ ionized_count_out
) {
    extern __shared__ char smem[];
    double* s_sum   = (double*)smem;
    double* s_sq    = s_sum + blockDim.x;
    int*    s_count = (int*)(s_sq + blockDim.x);

    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    s_sum[tid] = 0.0; s_sq[tid] = 0.0; s_count[tid] = 0;

    if (i < n) {
        float v = x_hii[i];
        s_sum[tid] = (double)v;
        s_sq[tid]  = (double)v * (double)v;
        s_count[tid] = (v > 0.5f) ? 1 : 0;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]   += s_sum[tid + s];
            s_sq[tid]    += s_sq[tid + s];
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(sum_xhii_out,    s_sum[0]);
        atomicAdd(sum_sq_out,      s_sq[0]);
        atomicAdd(ionized_count_out, s_count[0]);
    }
}

// Map: δT_b = 27 × x_HI × overdensity × sqrt((1+z)/10)   [mK]
__global__ void rt_cm21_field_kernel(
    const float* __restrict__ x_hii,
    const float* __restrict__ overdensity,
    float z,
    float* __restrict__ delta_tb_out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x_hi = fmaxf(1.0f - x_hii[i], 0.0f);
    float factor = sqrtf((1.0f + z) / 10.0f);
    delta_tb_out[i] = 27.0f * x_hi * overdensity[i] * factor;
}

extern "C" int cuda_rt_reionization_stats(
    const float* x_hii, int n,
    double* sum_xhii_out, double* sum_sq_out, int* ionized_count_out
) {
    if (n <= 0) { *sum_xhii_out = 0.0; *sum_sq_out = 0.0; *ionized_count_out = 0; return 0; }
    float *dxhii;
    double *ds_sum, *ds_sq;
    int *dcnt;
    if (alloc_copy(&dxhii, x_hii, n)) return -1;
    CUDA_CHECK(cudaMalloc(&ds_sum, sizeof(double))); CUDA_CHECK(cudaMemset(ds_sum, 0, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ds_sq,  sizeof(double))); CUDA_CHECK(cudaMemset(ds_sq,  0, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dcnt,   sizeof(int)));    CUDA_CHECK(cudaMemset(dcnt,   0, sizeof(int)));

    int block = RT_BLOCK_SIZE;
    int blocks = (n + block - 1) / block;
    size_t smem_bytes = block * (2 * sizeof(double) + sizeof(int));
    rt_reionization_stats_kernel<<<blocks, block, smem_bytes>>>(dxhii, n, ds_sum, ds_sq, dcnt);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(sum_xhii_out,    ds_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sum_sq_out,      ds_sq,  sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ionized_count_out, dcnt, sizeof(int),    cudaMemcpyDeviceToHost));
    cudaFree(dxhii); cudaFree(ds_sum); cudaFree(ds_sq); cudaFree(dcnt);
    return 0;
}

extern "C" int cuda_rt_cm21_field(
    const float* x_hii, const float* overdensity,
    float z, float* delta_tb_out, int n
) {
    if (n <= 0) return 0;
    float *dxhii, *dod, *dtb;
    if (alloc_copy(&dxhii, x_hii, n) || alloc_copy(&dod, overdensity, n) ||
        alloc_zero(&dtb, n)) return -1;
    int blocks = (n + RT_BLOCK_SIZE - 1) / RT_BLOCK_SIZE;
    rt_cm21_field_kernel<<<blocks, RT_BLOCK_SIZE>>>(dxhii, dod, z, dtb, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(delta_tb_out, dtb, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dxhii); cudaFree(dod); cudaFree(dtb);
    return 0;
}
