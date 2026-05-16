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
