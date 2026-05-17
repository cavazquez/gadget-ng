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

// ── LET traversal: mono + quad + oct per particle vs all LET nodes ─────────
//
// Replicates accel_soa_scalar() but in f32 on GPU, skipping hexadecapole.
// Each thread computes the acceleration for one particle from n_nodes LET nodes.

__global__ void tree_let_mono_quad_oct_kernel(
    // Particles
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    int n_particles,
    // LET nodes (SoA, f32)
    const float* __restrict__ cx,
    const float* __restrict__ cy,
    const float* __restrict__ cz,
    const float* __restrict__ node_mass,
    // Quadrupole: qxx, qxy, qxz, qyy, qyz, qzz
    const float* __restrict__ q0, const float* __restrict__ q1,
    const float* __restrict__ q2, const float* __restrict__ q3,
    const float* __restrict__ q4, const float* __restrict__ q5,
    // Octupole: o_xxx, o_xxy, o_xxz, o_xyy, o_xyz, o_yyy, o_yzz
    const float* __restrict__ o0, const float* __restrict__ o1,
    const float* __restrict__ o2, const float* __restrict__ o3,
    const float* __restrict__ o4, const float* __restrict__ o5,
    const float* __restrict__ o6,
    int n_nodes,
    float g, float eps2,
    float* __restrict__ ax_out,
    float* __restrict__ ay_out,
    float* __restrict__ az_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    float xi = px[i], yi = py[i], zi = pz[i];
    double ax = 0.0, ay = 0.0, az = 0.0;

    for (int j = 0; j < n_nodes; ++j) {
        float mj = node_mass[j];
        if (mj == 0.0f) continue;

        float rx = xi - cx[j];
        float ry = yi - cy[j];
        float rz = zi - cz[j];
        float r2 = rx*rx + ry*ry + rz*rz + eps2;
        if (r2 < 1.0e-30f) continue;
        float r_inv = rsqrtf(r2);

        // Monopole
        float r3_inv = r_inv * r_inv * r_inv;
        float mono = -g * mj * r3_inv;
        ax += (double)(mono * rx);
        ay += (double)(mono * ry);
        az += (double)(mono * rz);

        // Quadrupole
        float qxx=q0[j], qxy=q1[j], qxz=q2[j], qyy=q3[j], qyz=q4[j], qzz=q5[j];
        float r5_inv = r3_inv * r_inv * r_inv;
        float r7_inv = r5_inv * r_inv * r_inv;
        float qrx = qxx*rx + qxy*ry + qxz*rz;
        float qry = qxy*rx + qyy*ry + qyz*rz;
        float qrz = qxz*rx + qyz*ry + qzz*rz;
        float rqr = qrx*rx + qry*ry + qrz*rz;
        float c1 = g * r5_inv;
        float c2 = g * 2.5f * rqr * r7_inv;
        ax += (double)(c1*qrx - c2*rx);
        ay += (double)(c1*qry - c2*ry);
        az += (double)(c1*qrz - c2*rz);

        // Octupole
        float oxxx=o0[j], oxxy=o1[j], oxxz=o2[j], oxyy=o3[j];
        float oxyz=o4[j], oyyy=o5[j], oyzz=o6[j];
        float oxzz = -(oxxx + oxyy);
        float oyyz = -(oxxy + oyyy);
        float ozzz = -(oxxz - oxxy - oyyy);
        float orrx = oxxx*rx*rx + 2.0f*oxxy*rx*ry + 2.0f*oxxz*rx*rz
                   + oxyy*ry*ry + 2.0f*oxyz*ry*rz + oxzz*rz*rz;
        float orry = oxxy*rx*rx + 2.0f*oxyy*rx*ry + 2.0f*oxyz*rx*rz
                   + oyyy*ry*ry + 2.0f*oyyz*ry*rz + oyzz*rz*rz;
        float orrz = oxxz*rx*rx + 2.0f*oxyz*rx*ry + 2.0f*oxzz*rx*rz
                   + oyyz*ry*ry + 2.0f*oyzz*ry*rz + ozzz*rz*rz;
        float orrr = oxxx*rx*rx*rx + 3.0f*oxxy*rx*rx*ry + 3.0f*oxxz*rx*rx*rz
                   + 3.0f*oxyy*rx*ry*ry + 6.0f*oxyz*rx*ry*rz + 3.0f*oxzz*rx*rz*rz
                   + oyyy*ry*ry*ry + 3.0f*oyyz*ry*ry*rz + 3.0f*oyzz*ry*rz*rz
                   + ozzz*rz*rz*rz;
        float r9_inv = r7_inv * r_inv * r_inv;
        float co1 = -g * 0.5f * r7_inv;
        float co2 = g * (7.0f / 6.0f) * orrr * r9_inv;
        ax += (double)(co1*orrx + co2*rx);
        ay += (double)(co1*orry + co2*ry);
        az += (double)(co1*orrz + co2*rz);
    }

    ax_out[i] = (float)ax;
    ay_out[i] = (float)ay;
    az_out[i] = (float)az;
}

extern "C" int cuda_tree_let_accel(
    // Particles
    const float* px, const float* py, const float* pz, int n_particles,
    // LET nodes SoA (f32 already downcast by caller)
    const float* cx, const float* cy, const float* cz, const float* node_mass,
    const float* q0, const float* q1, const float* q2,
    const float* q3, const float* q4, const float* q5,
    const float* o0, const float* o1, const float* o2,
    const float* o3, const float* o4, const float* o5, const float* o6,
    int n_nodes,
    float g, float eps2,
    float* ax_out, float* ay_out, float* az_out
) {
    if (n_particles <= 0 || n_nodes <= 0) return 0;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return -1;

    // Upload particles
    float *dpx, *dpy, *dpz;
    if (alloc_copy(&dpx, px, n_particles) || alloc_copy(&dpy, py, n_particles) ||
        alloc_copy(&dpz, pz, n_particles)) return -1;

    // Upload LET nodes
    float *dcx, *dcy, *dcz, *dm;
    float *dq0, *dq1, *dq2, *dq3, *dq4, *dq5;
    float *do0, *do1, *do2, *do3, *do4, *do5, *do6;
    if (alloc_copy(&dcx, cx, n_nodes) || alloc_copy(&dcy, cy, n_nodes) ||
        alloc_copy(&dcz, cz, n_nodes) || alloc_copy(&dm, node_mass, n_nodes) ||
        alloc_copy(&dq0, q0, n_nodes) || alloc_copy(&dq1, q1, n_nodes) ||
        alloc_copy(&dq2, q2, n_nodes) || alloc_copy(&dq3, q3, n_nodes) ||
        alloc_copy(&dq4, q4, n_nodes) || alloc_copy(&dq5, q5, n_nodes) ||
        alloc_copy(&do0, o0, n_nodes) || alloc_copy(&do1, o1, n_nodes) ||
        alloc_copy(&do2, o2, n_nodes) || alloc_copy(&do3, o3, n_nodes) ||
        alloc_copy(&do4, o4, n_nodes) || alloc_copy(&do5, o5, n_nodes) ||
        alloc_copy(&do6, o6, n_nodes)) return -1;

    float *dax, *day, *daz;
    if (alloc_zero(&dax, n_particles) || alloc_zero(&day, n_particles) ||
        alloc_zero(&daz, n_particles)) return -1;

    int blocks = (n_particles + TREE_BLOCK_SIZE - 1) / TREE_BLOCK_SIZE;
    tree_let_mono_quad_oct_kernel<<<blocks, TREE_BLOCK_SIZE>>>(
        dpx, dpy, dpz, n_particles,
        dcx, dcy, dcz, dm,
        dq0, dq1, dq2, dq3, dq4, dq5,
        do0, do1, do2, do3, do4, do5, do6,
        n_nodes, g, eps2,
        dax, day, daz
    );
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    size_t bytes_p = (size_t)n_particles * sizeof(float);
    CUDA_CHECK(cudaMemcpy(ax_out, dax, bytes_p, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay_out, day, bytes_p, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az_out, daz, bytes_p, cudaMemcpyDeviceToHost));

    cudaFree(dpx); cudaFree(dpy); cudaFree(dpz);
    cudaFree(dcx); cudaFree(dcy); cudaFree(dcz); cudaFree(dm);
    cudaFree(dq0); cudaFree(dq1); cudaFree(dq2); cudaFree(dq3); cudaFree(dq4); cudaFree(dq5);
    cudaFree(do0); cudaFree(do1); cudaFree(do2); cudaFree(do3); cudaFree(do4); cudaFree(do5); cudaFree(do6);
    cudaFree(dax); cudaFree(day); cudaFree(daz);
    return 0;
}
// ── TreePM short-range erfc kernel (AP-20) ────────────────────────────────────
// O(N²) con guard r² < r_cut² y mínima imagen periódica.
// Usa la misma aproximación erfc (Abramowitz & Stegun §7.1.26) que el CPU.

__device__ static float erfc_approx_f32(float x) {
    if (x < 0.0f) return 2.0f - erfc_approx_f32(-x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float poly = t * (0.254829592f
               + t * (-0.284496736f
               + t * (1.421413741f
               + t * (-1.453152027f
               + t * 1.061405429f))));
    return poly * expf(-x * x);
}

__global__ void treepm_sr_erfc_kernel(
    const float* px, const float* py, const float* pz,
    const float* mass,
    float* ax, float* ay, float* az,
    int n, float r_split, float r_cut2, float eps2, float g, float box_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = px[i], yi = py[i], zi = pz[i];
    float axi = 0.0f, ayi = 0.0f, azi = 0.0f;
    float sqrt2_rsplit = 1.41421356f * r_split;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float dx = px[j] - xi;
        float dy = py[j] - yi;
        float dz = pz[j] - zi;
        // Mínima imagen
        if (box_size > 0.0f) {
            dx -= box_size * rintf(dx / box_size);
            dy -= box_size * rintf(dy / box_size);
            dz -= box_size * rintf(dz / box_size);
        }
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= r_cut2) continue;
        float r = sqrtf(r2);
        float w = erfc_approx_f32(r / sqrt2_rsplit);
        float denom = (r2 + eps2) * sqrtf(r2 + eps2);
        float fac = g * mass[j] * w / denom;
        axi += fac * dx;
        ayi += fac * dy;
        azi += fac * dz;
    }
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}

extern "C" int cuda_treepm_short_range(
    const float* x, const float* y, const float* z,
    const float* mass,
    float* ax_out, float* ay_out, float* az_out,
    int n, float r_split, float r_cut2, float eps2, float g, float box_size
) {
    if (n <= 0) return 0;
    float *dx, *dy, *dz, *dm, *dax, *day, *daz;
    if (alloc_copy(&dx, x, n) || alloc_copy(&dy, y, n) || alloc_copy(&dz, z, n) ||
        alloc_copy(&dm, mass, n) ||
        alloc_zero(&dax, n) || alloc_zero(&day, n) || alloc_zero(&daz, n)) return -1;
    int blocks = (n + TREE_BLOCK_SIZE - 1) / TREE_BLOCK_SIZE;
    treepm_sr_erfc_kernel<<<blocks, TREE_BLOCK_SIZE>>>(
        dx, dy, dz, dm, dax, day, daz,
        n, r_split, r_cut2, eps2, g, box_size);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ax_out, dax, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay_out, day, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az_out, daz, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dm);
    cudaFree(dax); cudaFree(day); cudaFree(daz);
    return 0;
}

// ── Barnes-Hut local GPU walk monopole (AP-20) ───────────────────────────────
// Recorre el octree BH con pila por hilo (stack[32]). Usa MAC de apertura
// theta2 y omite la auto-interacción con particle_idx.
// Los nodos se pasan como buffer binario coincidente con BhMonopoleGpuNode (repr C, 96 bytes).

#define BH_NO_CHILD    0xFFFFFFFFu
#define BH_NO_PARTICLE 0xFFFFFFFFu
#define BH_STACK_DEPTH 32

struct GpuBhNode {
    float com[3];
    float mass;
    float center[3];
    float half;
    unsigned int children[8];
    unsigned int particle_idx;
    unsigned int _reserved[7];
};

__global__ void bh_walk_monopole_kernel(
    const GpuBhNode* nodes, int n_nodes, unsigned int root_idx,
    const float* qx, const float* qy, const float* qz,
    const unsigned int* target_idx,
    float* ax, float* ay, float* az,
    int n_targets, float theta2, float g, float eps2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_targets) return;
    float xi = qx[i], yi = qy[i], zi = qz[i];
    unsigned int self_idx = target_idx[i];
    float axi = 0.0f, ayi = 0.0f, azi = 0.0f;

    unsigned int stack[BH_STACK_DEPTH];
    int top = 0;
    stack[top++] = root_idx;

    while (top > 0) {
        unsigned int nidx = stack[--top];
        if (nidx == BH_NO_CHILD || nidx >= (unsigned int)n_nodes) continue;
        const GpuBhNode& nd = nodes[nidx];
        float dx = nd.com[0] - xi;
        float dy = nd.com[1] - yi;
        float dz = nd.com[2] - zi;
        float d2 = dx*dx + dy*dy + dz*dz;
        float h  = nd.half;
        // MAC: node.half² < theta2 * d2 → usar monopolo
        if (h * h < theta2 * d2) {
            // Leaf self-skip
            if (nd.particle_idx != BH_NO_PARTICLE && nd.particle_idx == self_idx) continue;
            if (d2 < 1.0e-30f) continue;
            float denom = (d2 + eps2) * sqrtf(d2 + eps2);
            float fac = g * nd.mass / denom;
            axi += fac * dx;
            ayi += fac * dy;
            azi += fac * dz;
        } else {
            // Internal node: push children
            bool any_child = false;
            for (int c = 0; c < 8; c++) {
                unsigned int ch = nd.children[c];
                if (ch != BH_NO_CHILD && top < BH_STACK_DEPTH) {
                    stack[top++] = ch;
                    any_child = true;
                }
            }
            // Leaf (no children) — apply monopole directly
            if (!any_child) {
                if (nd.particle_idx == self_idx) continue;
                if (d2 < 1.0e-30f) continue;
                float denom = (d2 + eps2) * sqrtf(d2 + eps2);
                float fac = g * nd.mass / denom;
                axi += fac * dx;
                ayi += fac * dy;
                azi += fac * dz;
            }
        }
    }
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}

extern "C" int cuda_bh_walk_monopole(
    const void* nodes_raw, int n_nodes, unsigned int root_idx,
    const float* qx, const float* qy, const float* qz,
    const unsigned int* target_idx_h,
    float* ax_out, float* ay_out, float* az_out,
    int n_targets, float theta2, float g, float eps2
) {
    if (n_targets <= 0 || n_nodes <= 0) return 0;
    const GpuBhNode* nodes_h = (const GpuBhNode*)nodes_raw;
    GpuBhNode* d_nodes;
    float *dqx, *dqy, *dqz, *dax, *day, *daz;
    unsigned int* d_tidx;
    size_t node_bytes = (size_t)n_nodes * sizeof(GpuBhNode);
    if (cudaMalloc(&d_nodes, node_bytes) != cudaSuccess) return -1;
    if (cudaMemcpy(d_nodes, nodes_h, node_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_nodes); return -1;
    }
    if (alloc_copy(&dqx, qx, n_targets) || alloc_copy(&dqy, qy, n_targets) ||
        alloc_copy(&dqz, qz, n_targets) || alloc_copy(&d_tidx, target_idx_h, n_targets) ||
        alloc_zero(&dax, n_targets) || alloc_zero(&day, n_targets) || alloc_zero(&daz, n_targets)) {
        cudaFree(d_nodes); return -1;
    }
    int blocks = (n_targets + TREE_BLOCK_SIZE - 1) / TREE_BLOCK_SIZE;
    bh_walk_monopole_kernel<<<blocks, TREE_BLOCK_SIZE>>>(
        d_nodes, n_nodes, root_idx,
        dqx, dqy, dqz, d_tidx,
        dax, day, daz,
        n_targets, theta2, g, eps2);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ax_out, dax, (size_t)n_targets*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay_out, day, (size_t)n_targets*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az_out, daz, (size_t)n_targets*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_nodes); cudaFree(dqx); cudaFree(dqy); cudaFree(dqz);
    cudaFree(d_tidx); cudaFree(dax); cudaFree(day); cudaFree(daz);
    return 0;
}
