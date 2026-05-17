/**
 * pm_gravity.cu — Solver PM (Particle-Mesh) GPU con CUDA + cuFFT.
 *
 * Pipeline:
 *   1. cic_assign_kernel   — masas → grilla de densidad (CIC, periódico, atomicAdd)
 *   2. FFT 3D Z2Z (f64)    — tres pasadas 1D en el mismo orden que `fft_poisson::fft3d_inplace`
 *   3. poisson_kernel      — Φ(k) = −4πG·ρ(k)/k²; F_α(k) = −ik_α·Φ(k)  (grilla N³ compleja)
 *   4. IFFT 3D Z2Z × 3     — con filtro: F̂_y,F̂_z×(−i) en k; luego Re·(1/N⁴) (paridad con `fft_poisson`)
 *   5. cic_interp_kernel   — grilla de fuerza → aceleraciones en partículas
 *
 * Precisión: ρ y grids de fuerza en f32; FFT/Poisson espectral en Z2Z f64. Posiciones en [0, box_size).
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <new>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

#include "pm_gravity.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA] %s:%d: %s\n",                              \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult _r = (call);                                                \
        if (_r != CUFFT_SUCCESS) {                                              \
            fprintf(stderr, "[cuFFT] %s:%d: error %d\n",                       \
                    __FILE__, __LINE__, (int)_r);                               \
            return -1;                                                          \
        }                                                                       \
    } while (0)

struct CudaPmState {
    int   N;
    float box_size;

    float*            d_rho;
    cuDoubleComplex*  d_rho_k;
    cuDoubleComplex*  d_fx_k;
    cuDoubleComplex*  d_fy_k;
    cuDoubleComplex*  d_fz_k;
    float*        d_fx;
    float*        d_fy;
    float*        d_fz;

    float* d_x;
    float* d_y;
    float* d_z;
    float* d_mass;
    float* d_ax;
    float* d_ay;
    float* d_az;
    int    d_cap;

    cufftHandle plan_x; /* batch N², eje x (stride 1, idist N) */
    cufftHandle plan_y; /* batch N  por subbloque, eje y (stride N, idist 1) */
    cufftHandle plan_z; /* batch N  por subbloque, eje z (stride N², idist 1) */
};

__device__ __forceinline__ int wrap_idx(int i, int N)
{
    int r = i % N;
    return (r < 0) ? r + N : r;
}

__global__ void cic_assign_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ mass,
    float* __restrict__ rho,
    int N, float inv_box, int n)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n) return;

    float px = fmodf(x[pid] * inv_box * (float)N, (float)N);
    if (px < 0.0f) px += (float)N;
    float py = fmodf(y[pid] * inv_box * (float)N, (float)N);
    if (py < 0.0f) py += (float)N;
    float pz = fmodf(z[pid] * inv_box * (float)N, (float)N);
    if (pz < 0.0f) pz += (float)N;
    float m  = mass[pid];

    int ix = (int)floorf(px);
    int iy = (int)floorf(py);
    int iz = (int)floorf(pz);

    float dx = px - (float)ix;
    float dy = py - (float)iy;
    float dz = pz - (float)iz;
    float tx = 1.0f - dx;
    float ty = 1.0f - dy;
    float tz = 1.0f - dz;

#define CELL(di,dj,dk) (wrap_idx(ix+(di), N) + wrap_idx(iy+(dj), N)*N + wrap_idx(iz+(dk), N)*N*N)
    atomicAdd(&rho[CELL(0,0,0)], m * tx * ty * tz);
    atomicAdd(&rho[CELL(1,0,0)], m * dx * ty * tz);
    atomicAdd(&rho[CELL(0,1,0)], m * tx * dy * tz);
    atomicAdd(&rho[CELL(1,1,0)], m * dx * dy * tz);
    atomicAdd(&rho[CELL(0,0,1)], m * tx * ty * dz);
    atomicAdd(&rho[CELL(1,0,1)], m * dx * ty * dz);
    atomicAdd(&rho[CELL(0,1,1)], m * tx * dy * dz);
    atomicAdd(&rho[CELL(1,1,1)], m * dx * dy * dz);
#undef CELL
}

__global__ void scale_density_kernel(float* rho, long long n3, float rho_vol_scale)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n3) return;
    rho[i] *= rho_vol_scale;
}

/** ρ real → complejo (parte imaginaria 0), mismo orden que `fft_poisson`. */
__global__ void float_to_complex_kernel(
    const float* __restrict__ rho,
    cuDoubleComplex* __restrict__ c,
    long long n3)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n3) return;
    double r = (double)rho[i];
    c[i].x = r;
    c[i].y = 0.0;
}

/** Convención DFT `freq_index` alineada con `fft_poisson::freq_index` (doble precisión k). */
__device__ __forceinline__ static double pm_wavenum_d(int j, int N, double dk)
{
    double n = (double)((j <= N / 2) ? j : (j - N));
    return n * dk;
}

/** Igual que `fft_poisson`: flat = ix + N*iy + N²*iz → freq_index en cada eje.
 *  FFT Z2Z en doble precisión + Poisson en f64 para paridad con rustfft + TreePM filtrado.
 */
__global__ void poisson_kernel(
    const cuDoubleComplex* __restrict__ rho_k,
    cuDoubleComplex* __restrict__ fx_k,
    cuDoubleComplex* __restrict__ fy_k,
    cuDoubleComplex* __restrict__ fz_k,
    float two_pi_over_box,
    float four_pi_G,
    float r_split,
    int N)
{
    long long N3 = (long long)N * N * N;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N3) return;

    int nm2 = N * N;
    int iz = (int)(idx / nm2);
    int iy = (int)((idx / N) % N);
    int ix = (int)(idx % N);

    double dk = (double)two_pi_over_box;
    double kx = pm_wavenum_d(ix, N, dk);
    double ky = pm_wavenum_d(iy, N, dk);
    double kz = pm_wavenum_d(iz, N, dk);

    double k2 = kx * kx + ky * ky + kz * kz;

    cuDoubleComplex rho = rho_k[idx];

    /* Coherente con fft_poisson (f64): k² < 1e-30 */
    if (k2 < 1e-30) {
        fx_k[idx] = {0.0, 0.0};
        fy_k[idx] = {0.0, 0.0};
        fz_k[idx] = {0.0, 0.0};
        return;
    }

    double W = 1.0;
    if (r_split > 0.0f) {
        double rs = (double)r_split;
        W = exp(-0.5 * k2 * rs * rs);
    }
    double factor = (-(double)four_pi_G / k2) * W;
    cuDoubleComplex phi;
    phi.x = factor * rho.x;
    phi.y = factor * rho.y;

    fx_k[idx] = {kx * phi.y, -kx * phi.x};
    fy_k[idx] = {ky * phi.y, -ky * phi.x};
    fz_k[idx] = {kz * phi.y, -kz * phi.x};
}

/** (a+ib) ↦ (b−ia) = −i·(a+ib). Alinea Re(IFFT(·)) con la referencia que solo toma .re tras IFFT. */
__global__ void kspace_mul_minus_i_kernel(cuDoubleComplex* d, long long n3)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n3) return;
    double a = d[i].x;
    double b = d[i].y;
    d[i].x = b;
    d[i].y = -a;
}

__global__ void real_part_scale_kernel(
    float* __restrict__ dst,
    const cuDoubleComplex* __restrict__ src,
    long long n3,
    float scale)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n3) return;
    dst[i] = (float)(src[i].x * (double)scale);
}

__global__ void cic_interp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ fx_grid,
    const float* __restrict__ fy_grid,
    const float* __restrict__ fz_grid,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az,
    int N, float inv_box, int n)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n) return;

    float px = fmodf(x[pid] * inv_box * (float)N, (float)N);
    if (px < 0.0f) px += (float)N;
    float py = fmodf(y[pid] * inv_box * (float)N, (float)N);
    if (py < 0.0f) py += (float)N;
    float pz = fmodf(z[pid] * inv_box * (float)N, (float)N);
    if (pz < 0.0f) pz += (float)N;

    int ix = (int)floorf(px);
    int iy = (int)floorf(py);
    int iz = (int)floorf(pz);

    float dx = px - (float)ix;
    float dy = py - (float)iy;
    float dz = pz - (float)iz;
    float tx = 1.0f - dx;
    float ty = 1.0f - dy;
    float tz = 1.0f - dz;

#define CELL(di,dj,dk) (wrap_idx(ix+(di), N) + wrap_idx(iy+(dj), N)*N + wrap_idx(iz+(dk), N)*N*N)
#define INTERP(grid) (                                         \
    grid[CELL(0,0,0)] * tx*ty*tz +                             \
    grid[CELL(1,0,0)] * dx*ty*tz +                             \
    grid[CELL(0,1,0)] * tx*dy*tz +                             \
    grid[CELL(1,1,0)] * dx*dy*tz +                             \
    grid[CELL(0,0,1)] * tx*ty*dz +                             \
    grid[CELL(1,0,1)] * dx*ty*dz +                             \
    grid[CELL(0,1,1)] * tx*dy*dz +                             \
    grid[CELL(1,1,1)] * dx*dy*dz)

    ax[pid] = INTERP(fx_grid);
    ay[pid] = INTERP(fy_grid);
    az[pid] = INTERP(fz_grid);
#undef INTERP
#undef CELL
}

/** Adelante: X, luego Y por planos iz, luego Z por “filas” iy — como `fft3d_inplace`. */
static int pm_fft3d_forward(CudaPmState* s, cuDoubleComplex* d)
{
    int N = s->N;

    CUFFT_CHECK(cufftExecZ2Z(s->plan_x, d, d, CUFFT_FORWARD));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int iz = 0; iz < N; ++iz) {
        cuDoubleComplex* base = d + (ptrdiff_t)iz * N * N;
        CUFFT_CHECK(cufftExecZ2Z(s->plan_y, base, base, CUFFT_FORWARD));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int iy = 0; iy < N; ++iy) {
        cuDoubleComplex* base = d + (ptrdiff_t)iy * N;
        CUFFT_CHECK(cufftExecZ2Z(s->plan_z, base, base, CUFFT_FORWARD));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/** Inversa: Z, Y, X — mismo orden que `ifft3d_inplace`. cuFFT Z2Z inverso no divide por N. */
static int pm_fft3d_inverse(CudaPmState* s, cuDoubleComplex* d)
{
    int N = s->N;

    for (int iy = 0; iy < N; ++iy) {
        cuDoubleComplex* base = d + (ptrdiff_t)iy * N;
        CUFFT_CHECK(cufftExecZ2Z(s->plan_z, base, base, CUFFT_INVERSE));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int iz = 0; iz < N; ++iz) {
        cuDoubleComplex* base = d + (ptrdiff_t)iz * N * N;
        CUFFT_CHECK(cufftExecZ2Z(s->plan_y, base, base, CUFFT_INVERSE));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUFFT_CHECK(cufftExecZ2Z(s->plan_x, d, d, CUFFT_INVERSE));
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

extern "C" {

cuda_pm_handle_t cuda_pm_create(int grid_size, float box_size)
{
    CudaPmState* s = new (std::nothrow) CudaPmState();
    if (!s) return nullptr;

    s->N = grid_size;
    s->box_size = box_size;
    s->d_cap = 0;
    s->d_x = s->d_y = s->d_z = s->d_mass = nullptr;
    s->d_ax = s->d_ay = s->d_az = nullptr;
    s->plan_x = s->plan_y = s->plan_z = 0;

    int N = grid_size;
    long long N3 = (long long)N * N * N;

    if (cudaMalloc(&s->d_rho,   N3 * sizeof(float))            != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_rho_k, N3 * sizeof(cuDoubleComplex)) != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fx_k,  N3 * sizeof(cuDoubleComplex)) != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fy_k,  N3 * sizeof(cuDoubleComplex)) != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fz_k,  N3 * sizeof(cuDoubleComplex)) != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fx,    N3 * sizeof(float))        != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fy,    N3 * sizeof(float))        != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fz,    N3 * sizeof(float))        != cudaSuccess) goto fail;

    {
        int n[1] = { N };
        if (cufftPlanMany(&s->plan_x, 1, n,
                          NULL, 1, N,
                          NULL, 1, N,
                          CUFFT_Z2Z, N * N) != CUFFT_SUCCESS)
            goto fail;
        if (cufftPlanMany(&s->plan_y, 1, n,
                          NULL, N, 1,
                          NULL, N, 1,
                          CUFFT_Z2Z, N) != CUFFT_SUCCESS)
            goto fail;
        if (cufftPlanMany(&s->plan_z, 1, n,
                          NULL, N * N, 1,
                          NULL, N * N, 1,
                          CUFFT_Z2Z, N) != CUFFT_SUCCESS)
            goto fail;
    }

    return (cuda_pm_handle_t)s;

fail:
    cuda_pm_destroy((cuda_pm_handle_t)s);
    return nullptr;
}

void cuda_pm_destroy(cuda_pm_handle_t h)
{
    if (!h) return;
    CudaPmState* s = (CudaPmState*)h;
    if (s->d_rho)   cudaFree(s->d_rho);
    if (s->d_rho_k) cudaFree(s->d_rho_k);
    if (s->d_fx_k)  cudaFree(s->d_fx_k);
    if (s->d_fy_k)  cudaFree(s->d_fy_k);
    if (s->d_fz_k)  cudaFree(s->d_fz_k);
    if (s->d_fx)    cudaFree(s->d_fx);
    if (s->d_fy)    cudaFree(s->d_fy);
    if (s->d_fz)    cudaFree(s->d_fz);
    if (s->d_x)     cudaFree(s->d_x);
    if (s->d_y)     cudaFree(s->d_y);
    if (s->d_z)     cudaFree(s->d_z);
    if (s->d_mass)  cudaFree(s->d_mass);
    if (s->d_ax)    cudaFree(s->d_ax);
    if (s->d_ay)    cudaFree(s->d_ay);
    if (s->d_az)    cudaFree(s->d_az);
    if (s->plan_x) cufftDestroy(s->plan_x);
    if (s->plan_y) cufftDestroy(s->plan_y);
    if (s->plan_z) cufftDestroy(s->plan_z);
    delete s;
}

int cuda_pm_solve(
    cuda_pm_handle_t h,
    const float* x, const float* y, const float* z,
    const float* mass,
    float* ax, float* ay, float* az,
    int n, float /*eps2*/, float g, float r_split)
{
    if (!h || n <= 0) return 0;
    CudaPmState* s = (CudaPmState*)h;
    int N   = s->N;
    long long N3 = (long long)N * N * N;

    if (n > s->d_cap) {
        if (s->d_x) cudaFree(s->d_x);
        if (s->d_y) cudaFree(s->d_y);
        if (s->d_z) cudaFree(s->d_z);
        if (s->d_mass) cudaFree(s->d_mass);
        if (s->d_ax) cudaFree(s->d_ax);
        if (s->d_ay) cudaFree(s->d_ay);
        if (s->d_az) cudaFree(s->d_az);
        CUDA_CHECK(cudaMalloc(&s->d_x,    n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_y,    n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_z,    n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_mass, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_ax,   n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_ay,   n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_az,   n * sizeof(float)));
        s->d_cap = n;
    }

    CUDA_CHECK(cudaMemcpy(s->d_x,    x,    n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_y,    y,    n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_z,    z,    n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_mass, mass, n*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(s->d_rho, 0, N3 * sizeof(float)));

    {
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        float inv_box = 1.0f / s->box_size;
        cic_assign_kernel<<<blocks, threads>>>(
            s->d_x, s->d_y, s->d_z, s->d_mass,
            s->d_rho, N, inv_box, n);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        float rho_vol_scale =
            (float)N * (float)N * (float)N
            / (s->box_size * s->box_size * s->box_size);
        int threads = 256;
        int blocks = (int)(((N3 + threads - 1) / threads));
        scale_density_kernel<<<blocks, threads>>>(s->d_rho, N3, rho_vol_scale);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        int threads = 256;
        int blocks = (int)(((N3 + threads - 1) / threads));
        float_to_complex_kernel<<<blocks, threads>>>(s->d_rho, s->d_rho_k, N3);
        CUDA_CHECK(cudaGetLastError());
    }

    if (pm_fft3d_forward(s, s->d_rho_k) != 0) return -1;

    {
        int threads = 256;
        int blocks = (int)(((N3 + threads - 1) / threads));
        float two_pi_over_box = 2.0f * (float)M_PI / s->box_size;
        float four_pi_G = 4.0f * (float)M_PI * g;
        poisson_kernel<<<blocks, threads>>>(
            s->d_rho_k,
            s->d_fx_k, s->d_fy_k, s->d_fz_k,
            two_pi_over_box, four_pi_G, r_split, N);
        CUDA_CHECK(cudaGetLastError());
        if (r_split > 0.0f) {
            kspace_mul_minus_i_kernel<<<blocks, threads>>>(s->d_fy_k, N3);
            CUDA_CHECK(cudaGetLastError());
            kspace_mul_minus_i_kernel<<<blocks, threads>>>(s->d_fz_k, N3);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    int threads = 256;
    int blocks  = (int)((N3 + (long long)threads - 1) / (long long)threads);
    /* cuFFT Z2Z inverso no divide por eje. Factor 1/N⁴ alinea con `fft_poisson` en smoke. */
    float norm_real =
        1.0f / ((float)N * (float)N * (float)N * (float)N);

    if (pm_fft3d_inverse(s, s->d_fx_k) != 0) return -1;
    real_part_scale_kernel<<<blocks, threads>>>(
        s->d_fx, s->d_fx_k, N3, norm_real);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (pm_fft3d_inverse(s, s->d_fy_k) != 0) return -1;
    real_part_scale_kernel<<<blocks, threads>>>(
        s->d_fy, s->d_fy_k, N3, norm_real);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (pm_fft3d_inverse(s, s->d_fz_k) != 0) return -1;
    real_part_scale_kernel<<<blocks, threads>>>(
        s->d_fz, s->d_fz_k, N3, norm_real);
    CUDA_CHECK(cudaGetLastError());

    {
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        float inv_box = 1.0f / s->box_size;
        cic_interp_kernel<<<blocks, threads>>>(
            s->d_x, s->d_y, s->d_z,
            s->d_fx, s->d_fy, s->d_fz,
            s->d_ax, s->d_ay, s->d_az,
            N, inv_box, n);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(ax, s->d_ax, n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay, s->d_ay, n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az, s->d_az, n*sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

} /* extern "C" */

// ── f(R) chameleon screening (AP-20) ─────────────────────────────────────────
// Implementa `fr_screening_field` de gadget-ng-pm en GPU.
// Paso 1: por celda, calcula S = fifth_force_factor(chameleon_field(delta)).
// Paso 2 (iterativo): suavizado Jacobi periódico, `iterations` veces.

#ifndef FR_BLOCK_SIZE
#define FR_BLOCK_SIZE 256
#endif

__device__ static float chameleon_field_f32(float delta, float f_r0, float n) {
    if (delta <= -1.0f) return fabsf(f_r0);
    float rho_ratio = 1.0f + delta;
    return fabsf(f_r0) * powf(rho_ratio, -(n + 1.0f));
}

__device__ static float fifth_force_factor_f32(float fr_local, float f_r0) {
    if (fabsf(f_r0) <= 0.0f) return 0.0f;
    return fminf(fabsf(fr_local) / fabsf(f_r0), 1.0f);
}

__global__ void fr_screening_per_cell_kernel(
    const float* density, float* screen, int nm3,
    float f_r0, float n_fr, float rho_bar
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nm3) return;
    float delta = (rho_bar > 0.0f) ? ((density[i] - rho_bar) / rho_bar) : 0.0f;
    float fr = chameleon_field_f32(delta, f_r0, n_fr);
    screen[i] = fifth_force_factor_f32(fr, f_r0);
}

__global__ void fr_screening_jacobi_kernel(
    const float* screen_old, float* screen_new, int nm, float mix
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nm3 = nm * nm * nm;
    if (i >= nm3) return;
    int iz = i / (nm * nm);
    int iy = (i / nm) % nm;
    int ix = i % nm;
    int izm = (iz + nm - 1) % nm; int izp = (iz + 1) % nm;
    int iym = (iy + nm - 1) % nm; int iyp = (iy + 1) % nm;
    int ixm = (ix + nm - 1) % nm; int ixp = (ix + 1) % nm;
    float neigh = screen_old[iz*nm*nm + iy*nm + ixm]
                + screen_old[iz*nm*nm + iy*nm + ixp]
                + screen_old[iz*nm*nm + iym*nm + ix]
                + screen_old[iz*nm*nm + iyp*nm + ix]
                + screen_old[izm*nm*nm + iy*nm + ix]
                + screen_old[izp*nm*nm + iy*nm + ix];
    float val = (1.0f - mix) * screen_old[i] + mix * neigh / 6.0f;
    screen_new[i] = fminf(fmaxf(val, 0.0f), 1.0f);
}

extern "C" int cuda_fr_screening_field(
    const float* density, float* screen_out, int nm,
    float f_r0, float n_fr, float smoothing, int iterations
) {
    int nm3 = nm * nm * nm;
    if (nm3 <= 0) return 0;
    // Compute rho_bar on host side (small array, fast)
    float rho_sum = 0.0f;
    for (int i = 0; i < nm3; i++) rho_sum += density[i];
    float rho_bar = rho_sum / (float)nm3;

    float *dden, *dscr, *dscr2;
    if (cudaMalloc(&dden,  (size_t)nm3 * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&dscr,  (size_t)nm3 * sizeof(float)) != cudaSuccess) { cudaFree(dden); return -1; }
    if (cudaMalloc(&dscr2, (size_t)nm3 * sizeof(float)) != cudaSuccess) { cudaFree(dden); cudaFree(dscr); return -1; }
    if (cudaMemcpy(dden, density, (size_t)nm3 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dden); cudaFree(dscr); cudaFree(dscr2); return -1;
    }
    int blocks = (nm3 + FR_BLOCK_SIZE - 1) / FR_BLOCK_SIZE;
    fr_screening_per_cell_kernel<<<blocks, FR_BLOCK_SIZE>>>(dden, dscr, nm3, f_r0, n_fr, rho_bar);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(dden); cudaFree(dscr); cudaFree(dscr2); return -1;
    }
    float mix = smoothing;
    if (mix < 0.0f) mix = 0.0f;
    if (mix > 1.0f) mix = 1.0f;
    for (int it = 0; it < iterations; it++) {
        fr_screening_jacobi_kernel<<<blocks, FR_BLOCK_SIZE>>>(dscr, dscr2, nm, mix);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(dden); cudaFree(dscr); cudaFree(dscr2); return -1;
        }
        float* tmp = dscr; dscr = dscr2; dscr2 = tmp;
    }
    if (cudaMemcpy(screen_out, dscr, (size_t)nm3 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(dden); cudaFree(dscr); cudaFree(dscr2); return -1;
    }
    cudaFree(dden); cudaFree(dscr); cudaFree(dscr2);
    return 0;
}
