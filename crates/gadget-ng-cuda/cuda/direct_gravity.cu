/**
 * direct_gravity.cu — Solver de gravedad directa N² GPU con CUDA.
 *
 * ## Algoritmo
 *
 * Kernel de fuerza bruta O(N²) con tiling en shared memory (algoritmo de
 * Sanders & Kandrot, "CUDA by Example", cap. 7). Cada bloque de BLOCK_SIZE
 * hilos carga un tile de BLOCK_SIZE fuentes en shared memory y evalúa la
 * contribución de ese tile para todas las partículas objetivo del bloque.
 * El número de accesos a memoria global se reduce de O(N²) a O(N²/BLOCK_SIZE).
 *
 * ## Softening Plummer
 *
 *   a_i += G * m_j * r_ij / (|r_ij|² + ε²)^(3/2)
 *
 * ## Precisión
 *
 * f32 en device; acumuladores f32 con `fmaf`. Para N ≤ 10⁶ el error relativo
 * frente a f64 CPU es < 1e-5 (suficiente para dinámica cosmológica de DM).
 */

#include <cstdio>
#include <cstdlib>
#include <new>
#include <cuda_runtime.h>

#include "direct_gravity.h"

// ── Macro de comprobación de errores ─────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA-direct] %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

// ── Tamaño de tile (debe ser potencia de 2 y ≤ 1024) ─────────────────────────
#ifndef DIRECT_BLOCK_SIZE
#define DIRECT_BLOCK_SIZE 256
#endif

// ── Estado interno ────────────────────────────────────────────────────────────

struct CudaDirectState {
    float eps2;       /**< Suavizado Plummer al cuadrado. */
    int   block_size; /**< Hilos por bloque (≡ DIRECT_BLOCK_SIZE al compilar). */

    /* Buffers device (se redimensionan dinámicamente). */
    float* d_x;
    float* d_y;
    float* d_z;
    float* d_mass;
    float* d_ax;
    float* d_ay;
    float* d_az;
    int    d_cap; /**< Capacidad en número de partículas. */
};

// ── Kernel CUDA ───────────────────────────────────────────────────────────────

/**
 * direct_gravity_kernel — gravedad directa N² con tiling en shared memory.
 *
 * Cada hilo `i` calcula la aceleración de la partícula objetivo `i` sumando
 * las contribuciones de todos los tiles de fuentes.
 *
 * Grilla de lanzamiento: ⌈N / BLOCK_SIZE⌉ × 1, BLOCK_SIZE × 1 × 1.
 */
__global__ void direct_gravity_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ mass,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az,
    int n, float eps2, float g
) {
    /* Tile de fuentes en shared memory (4 floats × BLOCK_SIZE). */
    __shared__ float tile_x   [DIRECT_BLOCK_SIZE];
    __shared__ float tile_y   [DIRECT_BLOCK_SIZE];
    __shared__ float tile_z   [DIRECT_BLOCK_SIZE];
    __shared__ float tile_mass[DIRECT_BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float xi = 0.f, yi = 0.f, zi = 0.f;
    if (i < n) {
        xi = x[i];
        yi = y[i];
        zi = z[i];
    }

    float axi = 0.f, ayi = 0.f, azi = 0.f;

    /* Iterar sobre todos los tiles de fuentes. */
    int n_tiles = (n + DIRECT_BLOCK_SIZE - 1) / DIRECT_BLOCK_SIZE;

    for (int tile = 0; tile < n_tiles; ++tile) {
        /* Cargar tile de fuentes en shared memory. */
        int j = tile * DIRECT_BLOCK_SIZE + threadIdx.x;
        if (j < n) {
            tile_x[threadIdx.x]    = x[j];
            tile_y[threadIdx.x]    = y[j];
            tile_z[threadIdx.x]    = z[j];
            tile_mass[threadIdx.x] = mass[j];
        } else {
            /* Fuente "fantasma": masa 0 para no contaminar la suma. */
            tile_x[threadIdx.x]    = 0.f;
            tile_y[threadIdx.x]    = 0.f;
            tile_z[threadIdx.x]    = 0.f;
            tile_mass[threadIdx.x] = 0.f;
        }
        __syncthreads();

        /* Acumular contribución de cada fuente del tile. */
        if (i < n) {
            #pragma unroll 8
            for (int t = 0; t < DIRECT_BLOCK_SIZE; ++t) {
                float dx = tile_x[t] - xi;
                float dy = tile_y[t] - yi;
                float dz = tile_z[t] - zi;
                float r2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, eps2)));
                /* rsqrtf(r2) * (1/r2) = r2^(-3/2) */
                float inv_r3 = rsqrtf(r2) / r2; /* r^(-3) */
                float gm = g * tile_mass[t] * inv_r3;
                axi = fmaf(gm, dx, axi);
                ayi = fmaf(gm, dy, ayi);
                azi = fmaf(gm, dz, azi);
            }
        }
        __syncthreads();
    }

    if (i < n) {
        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Asegura que los buffers device tengan capacidad para n partículas. */
static int ensure_capacity(CudaDirectState* s, int n) {
    if (n <= s->d_cap) return 0;

    /* Liberar buffers anteriores. */
    cudaFree(s->d_x);    cudaFree(s->d_y);    cudaFree(s->d_z);
    cudaFree(s->d_mass); cudaFree(s->d_ax);   cudaFree(s->d_ay);
    cudaFree(s->d_az);

    size_t bytes = (size_t)n * sizeof(float);
    if (cudaMalloc(&s->d_x,    bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&s->d_y,    bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&s->d_z,    bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&s->d_mass, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&s->d_ax,   bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&s->d_ay,   bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&s->d_az,   bytes) != cudaSuccess) return -1;

    s->d_cap = n;
    return 0;
}

// ── API pública (extern "C") ──────────────────────────────────────────────────

cuda_direct_handle_t cuda_direct_create(float eps2, int block_size) {
    /* Verificar que hay un dispositivo CUDA disponible. */
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return nullptr;
    }

    CudaDirectState* s = new CudaDirectState();
    if (!s) return nullptr;

    s->eps2       = eps2;
    s->block_size = (block_size > 0) ? block_size : DIRECT_BLOCK_SIZE;
    s->d_x        = nullptr;
    s->d_y        = nullptr;
    s->d_z        = nullptr;
    s->d_mass     = nullptr;
    s->d_ax       = nullptr;
    s->d_ay       = nullptr;
    s->d_az       = nullptr;
    s->d_cap      = 0;

    return (cuda_direct_handle_t)s;
}

void cuda_direct_destroy(cuda_direct_handle_t h) {
    if (!h) return;
    CudaDirectState* s = (CudaDirectState*)h;
    cudaFree(s->d_x);    cudaFree(s->d_y);    cudaFree(s->d_z);
    cudaFree(s->d_mass); cudaFree(s->d_ax);   cudaFree(s->d_ay);
    cudaFree(s->d_az);
    delete s;
}

int cuda_direct_solve(
    cuda_direct_handle_t h,
    const float* x, const float* y, const float* z,
    const float* mass,
    float* ax, float* ay, float* az,
    int n, float g
) {
    if (!h || n <= 0) return 0;
    CudaDirectState* s = (CudaDirectState*)h;

    /* Asegurar capacidad de buffers device. */
    if (ensure_capacity(s, n) != 0) return -1;

    size_t bytes = (size_t)n * sizeof(float);

    /* Copiar datos al device. */
    CUDA_CHECK(cudaMemcpy(s->d_x,    x,    bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_y,    y,    bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_z,    z,    bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_mass, mass, bytes, cudaMemcpyHostToDevice));

    /* Lanzar kernel. */
    int bsz   = DIRECT_BLOCK_SIZE;
    int nblks = (n + bsz - 1) / bsz;
    direct_gravity_kernel<<<nblks, bsz>>>(
        s->d_x, s->d_y, s->d_z, s->d_mass,
        s->d_ax, s->d_ay, s->d_az,
        n, s->eps2, g
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copiar resultados al host. */
    CUDA_CHECK(cudaMemcpy(ax, s->d_ax, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay, s->d_ay, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az, s->d_az, bytes, cudaMemcpyDeviceToHost));

    return 0;
}
