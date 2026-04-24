/**
 * direct_gravity.hip — Solver de gravedad directa N² GPU con HIP/ROCm.
 *
 * ## Algoritmo
 *
 * Kernel de fuerza bruta O(N²) con tiling en LDS (Local Data Store — equivalente
 * HIP de la shared memory CUDA). Idéntico al algoritmo de direct_gravity.cu,
 * adaptado a las APIs HIP:
 *   - hipMalloc / hipMemcpy en lugar de cudaMalloc / cudaMemcpy
 *   - hipGetDeviceCount / hipGetLastError en lugar de sus equivalentes CUDA
 *   - __shared__ y __syncthreads() son portables: mismo código en HIP que en CUDA
 *   - rsqrtf() y fmaf() también son portables
 *
 * ## Softening Plummer
 *
 *   a_i += G * m_j * r_ij / (|r_ij|² + ε²)^(3/2)
 */

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#include "direct_gravity.h"

// ── Macro de comprobación de errores ─────────────────────────────────────────

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t _e = (call);                                                 \
        if (_e != hipSuccess) {                                                 \
            fprintf(stderr, "[HIP-direct] %s:%d: %s\n",                        \
                    __FILE__, __LINE__, hipGetErrorString(_e));                  \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#ifndef DIRECT_BLOCK_SIZE
#define DIRECT_BLOCK_SIZE 256
#endif

// ── Estado interno ────────────────────────────────────────────────────────────

struct HipDirectState {
    float eps2;
    int   block_size;
    float* d_x;
    float* d_y;
    float* d_z;
    float* d_mass;
    float* d_ax;
    float* d_ay;
    float* d_az;
    int    d_cap;
};

// ── Kernel HIP ────────────────────────────────────────────────────────────────

__global__ void hip_direct_gravity_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ mass,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az,
    int n, float eps2, float g
) {
    __shared__ float tile_x   [DIRECT_BLOCK_SIZE];
    __shared__ float tile_y   [DIRECT_BLOCK_SIZE];
    __shared__ float tile_z   [DIRECT_BLOCK_SIZE];
    __shared__ float tile_mass[DIRECT_BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float xi = 0.f, yi = 0.f, zi = 0.f;
    if (i < n) {
        xi = x[i]; yi = y[i]; zi = z[i];
    }

    float axi = 0.f, ayi = 0.f, azi = 0.f;

    int n_tiles = (n + DIRECT_BLOCK_SIZE - 1) / DIRECT_BLOCK_SIZE;
    for (int tile = 0; tile < n_tiles; ++tile) {
        int j = tile * DIRECT_BLOCK_SIZE + threadIdx.x;
        if (j < n) {
            tile_x[threadIdx.x]    = x[j];
            tile_y[threadIdx.x]    = y[j];
            tile_z[threadIdx.x]    = z[j];
            tile_mass[threadIdx.x] = mass[j];
        } else {
            tile_x[threadIdx.x] = tile_y[threadIdx.x] = tile_z[threadIdx.x] = 0.f;
            tile_mass[threadIdx.x] = 0.f;
        }
        __syncthreads();

        if (i < n) {
            #pragma unroll 8
            for (int t = 0; t < DIRECT_BLOCK_SIZE; ++t) {
                float dx = tile_x[t] - xi;
                float dy = tile_y[t] - yi;
                float dz = tile_z[t] - zi;
                float r2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, eps2)));
                float inv_r3 = rsqrtf(r2) / r2;
                float gm = g * tile_mass[t] * inv_r3;
                axi = fmaf(gm, dx, axi);
                ayi = fmaf(gm, dy, ayi);
                azi = fmaf(gm, dz, azi);
            }
        }
        __syncthreads();
    }

    if (i < n) {
        ax[i] = axi; ay[i] = ayi; az[i] = azi;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static int hip_ensure_capacity(HipDirectState* s, int n) {
    if (n <= s->d_cap) return 0;
    hipFree(s->d_x); hipFree(s->d_y); hipFree(s->d_z);
    hipFree(s->d_mass); hipFree(s->d_ax); hipFree(s->d_ay); hipFree(s->d_az);
    size_t bytes = (size_t)n * sizeof(float);
    if (hipMalloc(&s->d_x,    bytes) != hipSuccess) return -1;
    if (hipMalloc(&s->d_y,    bytes) != hipSuccess) return -1;
    if (hipMalloc(&s->d_z,    bytes) != hipSuccess) return -1;
    if (hipMalloc(&s->d_mass, bytes) != hipSuccess) return -1;
    if (hipMalloc(&s->d_ax,   bytes) != hipSuccess) return -1;
    if (hipMalloc(&s->d_ay,   bytes) != hipSuccess) return -1;
    if (hipMalloc(&s->d_az,   bytes) != hipSuccess) return -1;
    s->d_cap = n;
    return 0;
}

// ── API pública (extern "C") ──────────────────────────────────────────────────

hip_direct_handle_t hip_direct_create(float eps2, int block_size) {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        return nullptr;
    }
    HipDirectState* s = new HipDirectState();
    if (!s) return nullptr;
    s->eps2       = eps2;
    s->block_size = (block_size > 0) ? block_size : DIRECT_BLOCK_SIZE;
    s->d_x = s->d_y = s->d_z = s->d_mass = s->d_ax = s->d_ay = s->d_az = nullptr;
    s->d_cap = 0;
    return (hip_direct_handle_t)s;
}

void hip_direct_destroy(hip_direct_handle_t h) {
    if (!h) return;
    HipDirectState* s = (HipDirectState*)h;
    hipFree(s->d_x); hipFree(s->d_y); hipFree(s->d_z);
    hipFree(s->d_mass); hipFree(s->d_ax); hipFree(s->d_ay); hipFree(s->d_az);
    delete s;
}

int hip_direct_solve(
    hip_direct_handle_t h,
    const float* x, const float* y, const float* z,
    const float* mass,
    float* ax, float* ay, float* az,
    int n, float g
) {
    if (!h || n <= 0) return 0;
    HipDirectState* s = (HipDirectState*)h;

    if (hip_ensure_capacity(s, n) != 0) return -1;

    size_t bytes = (size_t)n * sizeof(float);
    HIP_CHECK(hipMemcpy(s->d_x,    x,    bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(s->d_y,    y,    bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(s->d_z,    z,    bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(s->d_mass, mass, bytes, hipMemcpyHostToDevice));

    int bsz   = DIRECT_BLOCK_SIZE;
    int nblks = (n + bsz - 1) / bsz;
    hipLaunchKernelGGL(
        hip_direct_gravity_kernel,
        dim3(nblks), dim3(bsz), 0, 0,
        s->d_x, s->d_y, s->d_z, s->d_mass,
        s->d_ax, s->d_ay, s->d_az,
        n, s->eps2, g
    );
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(ax, s->d_ax, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ay, s->d_ay, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(az, s->d_az, bytes, hipMemcpyDeviceToHost));

    return 0;
}
