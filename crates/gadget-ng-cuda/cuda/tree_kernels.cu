#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA-TREE] %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                    \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#ifndef TREE_BLOCK_SIZE
#define TREE_BLOCK_SIZE 256
#endif

static constexpr unsigned char PTYPE_DM = 0;

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

__global__ void tree_walk_monopole_kernel(
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ mass,
    float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
    int n, float g, float eps2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float axi = 0.0f, ayi = 0.0f, azi = 0.0f;
    float xi = x[i], yi = y[i], zi = z[i];
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        float rx = xi - x[j];
        float ry = yi - y[j];
        float rz = zi - z[j];
        float r2 = rx * rx + ry * ry + rz * rz + eps2;
        float rinv = rsqrtf(r2);
        float fac = -g * mass[j] * rinv * rinv * rinv;
        axi += fac * rx;
        ayi += fac * ry;
        azi += fac * rz;
    }
    ax[i] = axi; ay[i] = ayi; az[i] = azi;
}

__global__ void tree_sidm_scatter_kernel(
    const unsigned char* __restrict__ ptype,
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    const float* __restrict__ mass,
    float* __restrict__ ovx, float* __restrict__ ovy, float* __restrict__ ovz,
    int n, float dt, float sigma_over_m, float h
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vxi = vx[i], vyi = vy[i], vzi = vz[i];
    if (ptype[i] == PTYPE_DM && sigma_over_m > 0.0f && h > 0.0f) {
        float kickx = 0.0f, kicky = 0.0f, kickz = 0.0f;
        float h2 = h * h;
        for (int j = 0; j < n; ++j) {
            if (i == j || ptype[j] != PTYPE_DM) continue;
            float dx = x[i] - x[j], dy = y[i] - y[j], dz = z[i] - z[j];
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > h2) continue;
            float dvx = vx[j] - vxi, dvy = vy[j] - vyi, dvz = vz[j] - vzi;
            float weight = dt * sigma_over_m * mass[j] * (1.0f - sqrtf(r2) / h);
            kickx += weight * dvx; kicky += weight * dvy; kickz += weight * dvz;
        }
        vxi += kickx; vyi += kicky; vzi += kickz;
    }
    ovx[i] = vxi; ovy[i] = vyi; ovz[i] = vzi;
}

extern "C" int cuda_tree_walk_monopole(
    const float* x, const float* y, const float* z, const float* mass,
    float* ax_out, float* ay_out, float* az_out, int n, float g, float eps2
) {
    if (n <= 0) return 0;
    float *dx, *dy, *dz, *dm, *dax, *day, *daz;
    if (alloc_copy(&dx, x, n) || alloc_copy(&dy, y, n) || alloc_copy(&dz, z, n) ||
        alloc_copy(&dm, mass, n) || alloc_zero(&dax, n) || alloc_zero(&day, n) ||
        alloc_zero(&daz, n)) return -1;
    int blocks = (n + TREE_BLOCK_SIZE - 1) / TREE_BLOCK_SIZE;
    tree_walk_monopole_kernel<<<blocks, TREE_BLOCK_SIZE>>>(dx, dy, dz, dm, dax, day, daz, n, g, eps2);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(ax_out, dax, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay_out, day, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az_out, daz, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dm); cudaFree(dax); cudaFree(day); cudaFree(daz);
    return 0;
}

extern "C" int cuda_tree_sidm_scatter(
    const unsigned char* ptype, const float* x, const float* y, const float* z,
    const float* vx_in, const float* vy_in, const float* vz_in, const float* mass,
    float* vx_out, float* vy_out, float* vz_out,
    int n, float dt, float sigma_over_m, float h
) {
    if (n <= 0) return 0;
    unsigned char* dptype; float *dx, *dy, *dz, *dvx, *dvy, *dvz, *dm, *ovx, *ovy, *ovz;
    if (alloc_copy(&dptype, ptype, n) || alloc_copy(&dx, x, n) || alloc_copy(&dy, y, n) ||
        alloc_copy(&dz, z, n) || alloc_copy(&dvx, vx_in, n) || alloc_copy(&dvy, vy_in, n) ||
        alloc_copy(&dvz, vz_in, n) || alloc_copy(&dm, mass, n) ||
        alloc_zero(&ovx, n) || alloc_zero(&ovy, n) || alloc_zero(&ovz, n)) return -1;
    int blocks = (n + TREE_BLOCK_SIZE - 1) / TREE_BLOCK_SIZE;
    tree_sidm_scatter_kernel<<<blocks, TREE_BLOCK_SIZE>>>(dptype, dx, dy, dz, dvx, dvy, dvz, dm, ovx, ovy, ovz, n, dt, sigma_over_m, h);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(vx_out, ovx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vy_out, ovy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vz_out, ovz, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dptype); cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dm); cudaFree(ovx); cudaFree(ovy); cudaFree(ovz);
    return 0;
}
