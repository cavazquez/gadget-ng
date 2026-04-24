/**
 * pm_gravity.cu — Solver PM (Particle-Mesh) GPU con CUDA + cuFFT.
 *
 * Pipeline:
 *   1. cic_assign_kernel   — masas → grilla de densidad (CIC, periódico, atomicAdd)
 *   2. cuFFT 3D R→C        — ρ(x) → ρ(k)
 *   3. poisson_kernel      — Φ(k) = −4πG·ρ(k)/k²; F_α(k) = −ik_α·Φ(k)
 *   4. cuFFT 3D C→R × 3   — F_x(k),F_y(k),F_z(k) → F_x(x),F_y(x),F_z(x)
 *   5. cic_interp_kernel   — grilla de fuerza → aceleraciones en posición de partículas
 *
 * Precisión: f32 en device; las posiciones deben estar en [0, box_size).
 * Condiciones de contorno: periódicas (mediante la FFT y kernel de Poisson espectral).
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>
#include <cuda_runtime.h>
#include <cufft.h>

#include "pm_gravity.h"

// ── Macro de comprobación de errores ─────────────────────────────────────────

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

// ── Estado interno ────────────────────────────────────────────────────────────

struct CudaPmState {
    int   N;           /* lado de la grilla */
    float box_size;    /* tamaño de la caja física */
    int   nc;          /* N/2 + 1 (tamaño en frecuencias para FFT R2C) */

    /* Buffers device */
    float*        d_rho;       /* grilla de densidad real [N³] */
    cufftComplex* d_rho_k;    /* densidad compleja [N² × nc] */
    cufftComplex* d_fx_k;     /* campo de fuerza x en k-space */
    cufftComplex* d_fy_k;
    cufftComplex* d_fz_k;
    float*        d_fx;       /* campo de fuerza x en espacio real [N³] */
    float*        d_fy;
    float*        d_fz;

    /* Buffers device para partículas (realloc dinámico) */
    float* d_x;
    float* d_y;
    float* d_z;
    float* d_mass;
    float* d_ax;
    float* d_ay;
    float* d_az;
    int    d_cap;  /* capacidad actual (número de partículas) */

    /* Planes cuFFT */
    cufftHandle plan_r2c;  /* N×N×N real→complex */
    cufftHandle plan_c2r;  /* N×N×N complex→real (reutilizado 3 veces) */
};

// ── Kernel 1: CIC assign ──────────────────────────────────────────────────────

/**
 * Asigna las masas de n partículas a la grilla de densidad rho[] de tamaño N³
 * usando Cloud-In-Cell (interpolación trilineal a las 8 celdas vecinas).
 * Usa atomicAdd para thread-safety (las partículas pueden caer en celdas solapadas).
 *
 * Índice del array: i + j*N + k*N*N  (orden C, x rápido)
 */
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

    float px = x[pid] * inv_box * (float)N;
    float py = y[pid] * inv_box * (float)N;
    float pz = z[pid] * inv_box * (float)N;
    float m  = mass[pid];

    /* Celda base (esquina inferior izquierda del cubo CIC) */
    int ix = (int)floorf(px);
    int iy = (int)floorf(py);
    int iz = (int)floorf(pz);

    /* Distancia fraccionaria dentro de la celda */
    float dx = px - (float)ix;
    float dy = py - (float)iy;
    float dz = pz - (float)iz;
    float tx = 1.0f - dx;
    float ty = 1.0f - dy;
    float tz = 1.0f - dz;

    /* Contribuir a las 8 celdas con peso trilineal */
#define CELL(di,dj,dk) (((ix+(di)+N)%N) + ((iy+(dj)+N)%N)*N + ((iz+(dk)+N)%N)*N*N)
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

// ── Kernel 2: Poisson en k-space + diferenciación espectral ──────────────────

/**
 * Para cada modo (kx,ky,kz) de la grilla de Fourier N×N×nc:
 *   Φ(k) = −4πG · ρ(k) / k²       (k² = kx²+ky²+kz², k=0 → Φ=0)
 *   F_x(k) = −i·kx·Φ(k)
 *   F_y(k) = −i·ky·Φ(k)
 *   F_z(k) = −i·kz·Φ(k)
 *
 * Las frecuencias siguen la convención FFT estándar:
 *   kx ∈ [0..N/2, -(N/2-1)..-1]  (para kx >= N/2, kx_phys = kx - N)
 *   Similar para ky. kz ∈ [0..nc-1] siempre positivo (output R2C).
 *
 * El factor de normalización de cuFFT (1/N³) se aplica aquí.
 */
__global__ void poisson_kernel(
    const cufftComplex* __restrict__ rho_k,
    cufftComplex* __restrict__ fx_k,
    cufftComplex* __restrict__ fy_k,
    cufftComplex* __restrict__ fz_k,
    int N, float two_pi_over_box, float four_pi_G, float norm)
{
    /* Índice plano en el array R2C [N × N × nc] */
    int nc = N / 2 + 1;
    int total = N * N * nc;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    /* Recuperar (ix, iy, iz) */
    int iz = idx % nc;
    int tmp = idx / nc;
    int iy  = tmp % N;
    int ix  = tmp / N;

    /* Frecuencias físicas */
    float kx = (ix <= N/2) ? (float)ix : (float)(ix - N);
    float ky = (iy <= N/2) ? (float)iy : (float)(iy - N);
    float kz = (float)iz;

    kx *= two_pi_over_box;
    ky *= two_pi_over_box;
    kz *= two_pi_over_box;

    float k2 = kx*kx + ky*ky + kz*kz;

    cufftComplex rho = rho_k[idx];
    /* Normalización inversa cuFFT */
    rho.x *= norm;
    rho.y *= norm;

    if (k2 == 0.0f) {
        /* Modo DC: fuerza neta cero (universo periódico) */
        fx_k[idx] = {0.0f, 0.0f};
        fy_k[idx] = {0.0f, 0.0f};
        fz_k[idx] = {0.0f, 0.0f};
        return;
    }

    /* Φ(k) = −4πG·ρ(k) / k² */
    float factor = -four_pi_G / k2;
    cufftComplex phi;
    phi.x = factor * rho.x;
    phi.y = factor * rho.y;

    /* F_α(k) = −i·k_α·Φ(k)  →  Re[F] = k_α·Im[Φ],  Im[F] = −k_α·Re[Φ] */
    fx_k[idx] = { kx * phi.y,  -kx * phi.x };
    fy_k[idx] = { ky * phi.y,  -ky * phi.x };
    fz_k[idx] = { kz * phi.y,  -kz * phi.x };
}

// ── Kernel 3: CIC interpolation ───────────────────────────────────────────────

/**
 * Para cada partícula, interpola trilinealmente los campos de fuerza fx/fy/fz
 * en la posición de la partícula → aceleración.
 */
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

    float px = x[pid] * inv_box * (float)N;
    float py = y[pid] * inv_box * (float)N;
    float pz = z[pid] * inv_box * (float)N;

    int ix = (int)floorf(px);
    int iy = (int)floorf(py);
    int iz = (int)floorf(pz);

    float dx = px - (float)ix;
    float dy = py - (float)iy;
    float dz = pz - (float)iz;
    float tx = 1.0f - dx;
    float ty = 1.0f - dy;
    float tz = 1.0f - dz;

#define CELL(di,dj,dk) (((ix+(di)+N)%N) + ((iy+(dj)+N)%N)*N + ((iz+(dk)+N)%N)*N*N)
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

// ── API C ─────────────────────────────────────────────────────────────────────

extern "C" {

cuda_pm_handle_t cuda_pm_create(int grid_size, float box_size)
{
    CudaPmState* s = new (std::nothrow) CudaPmState();
    if (!s) return nullptr;

    s->N        = grid_size;
    s->box_size = box_size;
    s->nc       = grid_size / 2 + 1;
    s->d_cap    = 0;
    s->d_x = s->d_y = s->d_z = s->d_mass = nullptr;
    s->d_ax = s->d_ay = s->d_az = nullptr;

    long long N3  = (long long)grid_size * grid_size * grid_size;
    long long Nnc = (long long)grid_size * grid_size * s->nc;

    /* Allocar grillas device */
    if (cudaMalloc(&s->d_rho,   N3  * sizeof(float))         != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_rho_k, Nnc * sizeof(cufftComplex))  != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fx_k,  Nnc * sizeof(cufftComplex))  != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fy_k,  Nnc * sizeof(cufftComplex))  != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fz_k,  Nnc * sizeof(cufftComplex))  != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fx,    N3  * sizeof(float))         != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fy,    N3  * sizeof(float))         != cudaSuccess) goto fail;
    if (cudaMalloc(&s->d_fz,    N3  * sizeof(float))         != cudaSuccess) goto fail;

    /* Crear planes cuFFT */
    if (cufftPlan3d(&s->plan_r2c, grid_size, grid_size, grid_size,
                    CUFFT_R2C) != CUFFT_SUCCESS) goto fail;
    if (cufftPlan3d(&s->plan_c2r, grid_size, grid_size, grid_size,
                    CUFFT_C2R) != CUFFT_SUCCESS) goto fail;

    return (cuda_pm_handle_t)s;

fail:
    /* Limpiar lo que se haya allocado */
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
    cufftDestroy(s->plan_r2c);
    cufftDestroy(s->plan_c2r);
    delete s;
}

int cuda_pm_solve(
    cuda_pm_handle_t h,
    const float* x, const float* y, const float* z,
    const float* mass,
    float* ax, float* ay, float* az,
    int n, float /*eps2*/, float g)
{
    if (!h || n <= 0) return 0;
    CudaPmState* s = (CudaPmState*)h;
    int N   = s->N;
    long long N3  = (long long)N * N * N;

    /* ── Reasignar buffers de partículas si es necesario ─────────────────── */
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

    /* ── Copiar partículas host → device ─────────────────────────────────── */
    CUDA_CHECK(cudaMemcpy(s->d_x,    x,    n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_y,    y,    n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_z,    z,    n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_mass, mass, n*sizeof(float), cudaMemcpyHostToDevice));

    /* ── Paso 1: limpiar grilla de densidad ──────────────────────────────── */
    CUDA_CHECK(cudaMemset(s->d_rho, 0, N3 * sizeof(float)));

    /* ── Paso 2: CIC assign ──────────────────────────────────────────────── */
    {
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        float inv_box = 1.0f / s->box_size;
        cic_assign_kernel<<<blocks, threads>>>(
            s->d_x, s->d_y, s->d_z, s->d_mass,
            s->d_rho, N, inv_box, n);
        CUDA_CHECK(cudaGetLastError());
    }

    /* ── Paso 3: FFT R→C ─────────────────────────────────────────────────── */
    CUFFT_CHECK(cufftExecR2C(s->plan_r2c, s->d_rho, s->d_rho_k));

    /* ── Paso 4: Poisson kernel ──────────────────────────────────────────── */
    {
        int total   = N * N * s->nc;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        float two_pi_over_box = 2.0f * (float)M_PI / s->box_size;
        float four_pi_G = 4.0f * (float)M_PI * g;
        float norm = 1.0f / (float)N3;   /* factor de normalización cuFFT */
        poisson_kernel<<<blocks, threads>>>(
            s->d_rho_k,
            s->d_fx_k, s->d_fy_k, s->d_fz_k,
            N, two_pi_over_box, four_pi_G, norm);
        CUDA_CHECK(cudaGetLastError());
    }

    /* ── Paso 5: FFT inversa C→R para cada componente de fuerza ─────────── */
    CUFFT_CHECK(cufftExecC2R(s->plan_c2r, s->d_fx_k, s->d_fx));
    CUFFT_CHECK(cufftExecC2R(s->plan_c2r, s->d_fy_k, s->d_fy));
    CUFFT_CHECK(cufftExecC2R(s->plan_c2r, s->d_fz_k, s->d_fz));

    /* ── Paso 6: CIC interpolation ───────────────────────────────────────── */
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

    /* ── Copiar resultados device → host ─────────────────────────────────── */
    CUDA_CHECK(cudaMemcpy(ax, s->d_ax, n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay, s->d_ay, n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az, s->d_az, n*sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

} /* extern "C" */
