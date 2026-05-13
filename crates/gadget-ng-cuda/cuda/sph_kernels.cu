#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA-SPH] %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                    \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#ifndef SPH_BLOCK_SIZE
#define SPH_BLOCK_SIZE 128
#endif

static constexpr float SIGMA3 = 21.0f / (16.0f * 3.14159265358979323846f);
static constexpr float GAMMA_SPH = 5.0f / 3.0f;
static constexpr float N_NEIGH = 32.0f;
static constexpr int MAX_ITER = 30;
static constexpr float EPS_VISC = 0.01f;
static constexpr float EPS_BAL = 0.0001f;
static constexpr float ALPHA_VISC = 1.0f;

__device__ inline float clampf_sph(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ inline float3 periodic_delta(float ax, float ay, float az,
                                        float bx, float by, float bz,
                                        float box) {
    float3 d = make_float3(ax - bx, ay - by, az - bz);
    if (box > 0.0f) {
        d.x -= roundf(d.x / box) * box;
        d.y -= roundf(d.y / box) * box;
        d.z -= roundf(d.z / box) * box;
    }
    return d;
}

__device__ inline float w_kernel(float r, float h) {
    float q = r / h;
    if (q >= 2.0f) return 0.0f;
    float t = 1.0f - 0.5f * q;
    return SIGMA3 / (h * h * h) * t * t * t * t * (2.0f * q + 1.0f);
}

__device__ inline float grad_w_kernel(float r, float h) {
    float q = r / h;
    if (q >= 2.0f || r < 1.0e-30f) return 0.0f;
    float t = 1.0f - 0.5f * q;
    return SIGMA3 / (h * h * h * h) * (-5.0f * q * t * t * t);
}

__device__ inline void rho_and_deriv_device(
    const float* x, const float* y, const float* z, const float* mass,
    int n, float xi, float yi, float zi, float h, float box,
    float* rho_out, float* drho_out
) {
    float rho = 0.0f;
    float drho = 0.0f;
    for (int j = 0; j < n; ++j) {
        float3 rvec = periodic_delta(xi, yi, zi, x[j], y[j], z[j], box);
        float r = sqrtf(rvec.x * rvec.x + rvec.y * rvec.y + rvec.z * rvec.z);
        float w = w_kernel(r, h);
        float gw = grad_w_kernel(r, h);
        rho += mass[j] * w;
        drho += mass[j] * (-1.0f / h) * (3.0f * w + r * gw);
    }
    *rho_out = rho;
    *drho_out = drho;
}

__global__ void sph_density_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ mass,
    const unsigned char* __restrict__ is_gas,
    const float* __restrict__ u,
    const float* __restrict__ h_in,
    float* __restrict__ h_out,
    float* __restrict__ rho_out,
    float* __restrict__ pressure_out,
    float* __restrict__ entropy_out,
    int n, float box
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h_out[i] = h_in[i];
    rho_out[i] = 0.0f;
    pressure_out[i] = 0.0f;
    entropy_out[i] = 0.0f;
    if (!is_gas[i]) return;

    float h = fmaxf(h_in[i], 1.0e-10f);
    float xi = x[i], yi = y[i], zi = z[i];
    float mi = mass[i];

    for (int it = 0; it < MAX_ITER; ++it) {
        float rho, drho;
        rho_and_deriv_device(x, y, z, mass, n, xi, yi, zi, h, box, &rho, &drho);
        float n_eff = (4.0f * 3.14159265358979323846f / 3.0f) * powf(2.0f * h, 3.0f) * rho / mi;
        if (fabsf(n_eff - N_NEIGH) < 1.0e-2f) break;
        float dn_dh = (4.0f * 3.14159265358979323846f / 3.0f)
            * (24.0f * h * h * rho + powf(2.0f * h, 3.0f) * drho) / mi;
        float dh = -(n_eff - N_NEIGH) / (dn_dh + 1.0e-30f);
        h = fmaxf(h + clampf_sph(dh, -0.5f * h, 0.5f * h), 1.0e-10f);
    }

    float rho = 0.0f, unused = 0.0f;
    rho_and_deriv_device(x, y, z, mass, n, xi, yi, zi, h, box, &rho, &unused);
    float pressure = (GAMMA_SPH - 1.0f) * rho * u[i];
    float entropy = (rho > 0.0f) ? (GAMMA_SPH - 1.0f) * u[i] / powf(rho, GAMMA_SPH - 1.0f) : 0.0f;
    h_out[i] = h;
    rho_out[i] = rho;
    pressure_out[i] = pressure;
    entropy_out[i] = entropy;
}

__device__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__global__ void sph_balsara_kernel(
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    const float* __restrict__ mass, const unsigned char* __restrict__ is_gas,
    const float* __restrict__ rho, const float* __restrict__ pressure,
    const float* __restrict__ h_sml, float* __restrict__ balsara_out,
    int n, float box
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    balsara_out[i] = 1.0f;
    if (!is_gas[i]) return;
    if (rho[i] < 1.0e-20f) return;

    float div_v = 0.0f;
    float3 curl_v = make_float3(0.0f, 0.0f, 0.0f);
    float hi = h_sml[i];
    for (int j = 0; j < n; ++j) {
        if (j == i || !is_gas[j] || rho[j] < 1.0e-20f) continue;
        float3 rvec = periodic_delta(x[j], y[j], z[j], x[i], y[i], z[i], box);
        float r = sqrtf(rvec.x * rvec.x + rvec.y * rvec.y + rvec.z * rvec.z);
        float gw = grad_w_kernel(r, hi);
        if (gw == 0.0f) continue;
        float inv_r = (r > 1.0e-30f) ? 1.0f / r : 0.0f;
        float3 rhat = make_float3(rvec.x * inv_r, rvec.y * inv_r, rvec.z * inv_r);
        float3 nabla = make_float3(rhat.x * gw / hi, rhat.y * gw / hi, rhat.z * gw / hi);
        float3 dv = make_float3(vx[j] - vx[i], vy[j] - vy[i], vz[j] - vz[i]);
        div_v += mass[j] * (dv.x * nabla.x + dv.y * nabla.y + dv.z * nabla.z);
        float3 c = cross3(nabla, dv);
        curl_v.x += mass[j] * c.x;
        curl_v.y += mass[j] * c.y;
        curl_v.z += mass[j] * c.z;
    }
    float inv_rho = 1.0f / rho[i];
    div_v *= inv_rho;
    curl_v.x *= inv_rho; curl_v.y *= inv_rho; curl_v.z *= inv_rho;
    float abs_div = fabsf(div_v);
    float abs_curl = sqrtf(curl_v.x * curl_v.x + curl_v.y * curl_v.y + curl_v.z * curl_v.z);
    float cs = sqrtf(fmaxf(GAMMA_SPH * pressure[i] / rho[i], 0.0f));
    balsara_out[i] = abs_div / (abs_div + abs_curl + EPS_BAL * cs / hi);
}

__global__ void sph_forces_kernel(
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    const float* __restrict__ mass, const unsigned char* __restrict__ is_gas,
    const float* __restrict__ rho, const float* __restrict__ pressure,
    const float* __restrict__ h_sml,
    float* __restrict__ ax_out, float* __restrict__ ay_out, float* __restrict__ az_out,
    float* __restrict__ du_dt_out,
    int n, float box
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ax_out[i] = ay_out[i] = az_out[i] = du_dt_out[i] = 0.0f;
    if (!is_gas[i] || rho[i] < 1.0e-20f) return;

    float pi_rho2 = pressure[i] / (rho[i] * rho[i]);
    float cs_i = sqrtf(fmaxf(GAMMA_SPH * pressure[i] / rho[i], 0.0f));
    float ax = 0.0f, ay = 0.0f, az = 0.0f, dudt = 0.0f;

    for (int j = 0; j < n; ++j) {
        if (j == i || !is_gas[j] || rho[j] < 1.0e-20f) continue;
        float3 rij = periodic_delta(x[j], y[j], z[j], x[i], y[i], z[i], box);
        float r = sqrtf(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
        float hi = h_sml[i], hj = h_sml[j];
        float gwi = grad_w_kernel(r, hi);
        float gwj = grad_w_kernel(r, hj);
        if (gwi == 0.0f && gwj == 0.0f) continue;
        float inv_r = (r > 1.0e-30f) ? 1.0f / r : 0.0f;
        float3 rhat = make_float3(rij.x * inv_r, rij.y * inv_r, rij.z * inv_r);
        float3 nwi = make_float3(rhat.x * gwi / hi, rhat.y * gwi / hi, rhat.z * gwi / hi);
        float3 nwj = make_float3(rhat.x * gwj / hj, rhat.y * gwj / hj, rhat.z * gwj / hj);
        float3 vij = make_float3(vx[i] - vx[j], vy[i] - vy[j], vz[i] - vz[j]);
        float vdotr = vij.x * rij.x + vij.y * rij.y + vij.z * rij.z;
        float hbar = 0.5f * (hi + hj);
        float mu = 0.0f;
        if (vdotr < 0.0f) {
            float cs_j = sqrtf(fmaxf(GAMMA_SPH * pressure[j] / rho[j], 0.0f));
            float cs_bar = 0.5f * (cs_i + cs_j);
            float rho_bar = 0.5f * (rho[i] + rho[j]);
            float mu_raw = hbar * vdotr / (r * r + EPS_VISC * hbar * hbar);
            mu = -ALPHA_VISC * cs_bar * mu_raw / rho_bar;
        }
        float pj_rho2 = pressure[j] / (rho[j] * rho[j]);
        float ci = pi_rho2 + 0.5f * mu;
        float cj = pj_rho2 + 0.5f * mu;
        ax -= mass[j] * (nwi.x * ci + nwj.x * cj);
        ay -= mass[j] * (nwi.y * ci + nwj.y * cj);
        az -= mass[j] * (nwi.z * ci + nwj.z * cj);
        dudt += mass[j] * pi_rho2 * (vij.x * nwi.x + vij.y * nwi.y + vij.z * nwi.z);
    }
    ax_out[i] = ax; ay_out[i] = ay; az_out[i] = az; du_dt_out[i] = dudt;
}

__global__ void sph_gadget2_forces_kernel(
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    const float* __restrict__ mass, const unsigned char* __restrict__ is_gas,
    const float* __restrict__ rho, const float* __restrict__ pressure,
    const float* __restrict__ h_sml, const float* __restrict__ balsara,
    float* __restrict__ ax_out, float* __restrict__ ay_out, float* __restrict__ az_out,
    float* __restrict__ da_dt_out, float* __restrict__ du_dt_out,
    float* __restrict__ max_vsig_out,
    int n, float box
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ax_out[i] = ay_out[i] = az_out[i] = da_dt_out[i] = du_dt_out[i] = max_vsig_out[i] = 0.0f;
    if (!is_gas[i] || rho[i] < 1.0e-20f) return;

    float pi_rho2 = pressure[i] / (rho[i] * rho[i]);
    float cs_i = sqrtf(fmaxf(GAMMA_SPH * pressure[i] / rho[i], 0.0f));
    float fi = balsara[i];
    float ax = 0.0f, ay = 0.0f, az = 0.0f, da_raw = 0.0f, max_vsig = 0.0f;

    for (int j = 0; j < n; ++j) {
        if (j == i || !is_gas[j] || rho[j] < 1.0e-20f) continue;
        float3 rij = periodic_delta(x[j], y[j], z[j], x[i], y[i], z[i], box);
        float r = sqrtf(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
        float hi = h_sml[i], hj = h_sml[j];
        float gwi = grad_w_kernel(r, hi);
        float gwj = grad_w_kernel(r, hj);
        if (gwi == 0.0f && gwj == 0.0f) continue;
        float inv_r = (r > 1.0e-30f) ? 1.0f / r : 0.0f;
        float3 rhat = make_float3(rij.x * inv_r, rij.y * inv_r, rij.z * inv_r);
        float3 nwi = make_float3(rhat.x * gwi / hi, rhat.y * gwi / hi, rhat.z * gwi / hi);
        float3 nwj = make_float3(rhat.x * gwj / hj, rhat.y * gwj / hj, rhat.z * gwj / hj);
        float3 nwbar = make_float3(0.5f * (nwi.x + nwj.x), 0.5f * (nwi.y + nwj.y), 0.5f * (nwi.z + nwj.z));
        float3 vij = make_float3(vx[i] - vx[j], vy[i] - vy[j], vz[i] - vz[j]);
        float wij = vij.x * rhat.x + vij.y * rhat.y + vij.z * rhat.z;
        float pi_visc = 0.0f;
        if (wij < 0.0f) {
            float cs_j = sqrtf(fmaxf(GAMMA_SPH * pressure[j] / rho[j], 0.0f));
            float vsig = ALPHA_VISC * (cs_i + cs_j - 3.0f * wij) * 0.5f;
            float rho_bar = 0.5f * (rho[i] + rho[j]);
            float fij = 0.5f * (fi + balsara[j]);
            pi_visc = -fij * vsig * wij / rho_bar;
            max_vsig = fmaxf(max_vsig, vsig);
        }
        float pj_rho2 = pressure[j] / (rho[j] * rho[j]);
        float ci = pi_rho2 + 0.5f * pi_visc;
        float cj = pj_rho2 + 0.5f * pi_visc;
        ax -= mass[j] * (nwi.x * ci + nwj.x * cj);
        ay -= mass[j] * (nwi.y * ci + nwj.y * cj);
        az -= mass[j] * (nwi.z * ci + nwj.z * cj);
        da_raw += mass[j] * pi_visc * (vij.x * nwbar.x + vij.y * nwbar.y + vij.z * nwbar.z);
    }
    float da_factor = (rho[i] > 0.0f) ? (GAMMA_SPH - 1.0f) * 0.5f / powf(rho[i], GAMMA_SPH - 1.0f) : 0.0f;
    ax_out[i] = ax; ay_out[i] = ay; az_out[i] = az;
    da_dt_out[i] = da_factor * da_raw;
    du_dt_out[i] = pi_rho2 * da_raw;
    max_vsig_out[i] = max_vsig;
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

extern "C" int cuda_sph_density(
    const float* x, const float* y, const float* z, const float* mass,
    const unsigned char* is_gas, const float* u, const float* h_in,
    float* h_out, float* rho_out, float* pressure_out, float* entropy_out,
    int n, float periodic_box
) {
    if (n <= 0) return 0;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return -1;
    float *dx, *dy, *dz, *dm, *du, *dh, *dhout, *drho, *dp, *de;
    unsigned char* dgas;
    if (alloc_copy(&dx, x, n) || alloc_copy(&dy, y, n) || alloc_copy(&dz, z, n) ||
        alloc_copy(&dm, mass, n) || alloc_copy(&dgas, is_gas, n) ||
        alloc_copy(&du, u, n) || alloc_copy(&dh, h_in, n) ||
        alloc_zero(&dhout, n) || alloc_zero(&drho, n) || alloc_zero(&dp, n) || alloc_zero(&de, n)) return -1;
    int blocks = (n + SPH_BLOCK_SIZE - 1) / SPH_BLOCK_SIZE;
    sph_density_kernel<<<blocks, SPH_BLOCK_SIZE>>>(dx, dy, dz, dm, dgas, du, dh, dhout, drho, dp, de, n, periodic_box);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_out, dhout, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rho_out, drho, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pressure_out, dp, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(entropy_out, de, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dm); cudaFree(dgas); cudaFree(du); cudaFree(dh);
    cudaFree(dhout); cudaFree(drho); cudaFree(dp); cudaFree(de);
    return 0;
}

extern "C" int cuda_sph_balsara(
    const float* x, const float* y, const float* z,
    const float* vx, const float* vy, const float* vz,
    const float* mass, const unsigned char* is_gas,
    const float* rho, const float* pressure, const float* h_sml,
    float* balsara_out, int n, float periodic_box
) {
    if (n <= 0) return 0;
    float *dx, *dy, *dz, *dvx, *dvy, *dvz, *dm, *drho, *dp, *dh, *dbo;
    unsigned char* dgas;
    if (alloc_copy(&dx, x, n) || alloc_copy(&dy, y, n) || alloc_copy(&dz, z, n) ||
        alloc_copy(&dvx, vx, n) || alloc_copy(&dvy, vy, n) || alloc_copy(&dvz, vz, n) ||
        alloc_copy(&dm, mass, n) || alloc_copy(&dgas, is_gas, n) ||
        alloc_copy(&drho, rho, n) || alloc_copy(&dp, pressure, n) || alloc_copy(&dh, h_sml, n) ||
        alloc_zero(&dbo, n)) return -1;
    int blocks = (n + SPH_BLOCK_SIZE - 1) / SPH_BLOCK_SIZE;
    sph_balsara_kernel<<<blocks, SPH_BLOCK_SIZE>>>(dx, dy, dz, dvx, dvy, dvz, dm, dgas, drho, dp, dh, dbo, n, periodic_box);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(balsara_out, dbo, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dm); cudaFree(dgas);
    cudaFree(drho); cudaFree(dp); cudaFree(dh); cudaFree(dbo);
    return 0;
}

extern "C" int cuda_sph_forces(
    const float* x, const float* y, const float* z,
    const float* vx, const float* vy, const float* vz,
    const float* mass, const unsigned char* is_gas,
    const float* rho, const float* pressure, const float* h_sml,
    float* ax_out, float* ay_out, float* az_out, float* du_dt_out,
    int n, float periodic_box
) {
    if (n <= 0) return 0;
    float *dx, *dy, *dz, *dvx, *dvy, *dvz, *dm, *drho, *dp, *dh, *dax, *day, *daz, *ddu;
    unsigned char* dgas;
    if (alloc_copy(&dx, x, n) || alloc_copy(&dy, y, n) || alloc_copy(&dz, z, n) ||
        alloc_copy(&dvx, vx, n) || alloc_copy(&dvy, vy, n) || alloc_copy(&dvz, vz, n) ||
        alloc_copy(&dm, mass, n) || alloc_copy(&dgas, is_gas, n) ||
        alloc_copy(&drho, rho, n) || alloc_copy(&dp, pressure, n) || alloc_copy(&dh, h_sml, n) ||
        alloc_zero(&dax, n) || alloc_zero(&day, n) || alloc_zero(&daz, n) || alloc_zero(&ddu, n)) return -1;
    int blocks = (n + SPH_BLOCK_SIZE - 1) / SPH_BLOCK_SIZE;
    sph_forces_kernel<<<blocks, SPH_BLOCK_SIZE>>>(dx, dy, dz, dvx, dvy, dvz, dm, dgas, drho, dp, dh, dax, day, daz, ddu, n, periodic_box);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(ax_out, dax, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay_out, day, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az_out, daz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(du_dt_out, ddu, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dm); cudaFree(dgas);
    cudaFree(drho); cudaFree(dp); cudaFree(dh); cudaFree(dax); cudaFree(day); cudaFree(daz); cudaFree(ddu);
    return 0;
}

extern "C" int cuda_sph_gadget2_forces(
    const float* x, const float* y, const float* z,
    const float* vx, const float* vy, const float* vz,
    const float* mass, const unsigned char* is_gas,
    const float* rho, const float* pressure, const float* h_sml, const float* balsara,
    float* ax_out, float* ay_out, float* az_out,
    float* da_dt_out, float* du_dt_out, float* max_vsig_out,
    int n, float periodic_box
) {
    if (n <= 0) return 0;
    float *dx, *dy, *dz, *dvx, *dvy, *dvz, *dm, *drho, *dp, *dh, *db, *dax, *day, *daz, *dda, *ddu, *dmv;
    unsigned char* dgas;
    if (alloc_copy(&dx, x, n) || alloc_copy(&dy, y, n) || alloc_copy(&dz, z, n) ||
        alloc_copy(&dvx, vx, n) || alloc_copy(&dvy, vy, n) || alloc_copy(&dvz, vz, n) ||
        alloc_copy(&dm, mass, n) || alloc_copy(&dgas, is_gas, n) ||
        alloc_copy(&drho, rho, n) || alloc_copy(&dp, pressure, n) || alloc_copy(&dh, h_sml, n) ||
        alloc_copy(&db, balsara, n) || alloc_zero(&dax, n) || alloc_zero(&day, n) ||
        alloc_zero(&daz, n) || alloc_zero(&dda, n) || alloc_zero(&ddu, n) || alloc_zero(&dmv, n)) return -1;
    int blocks = (n + SPH_BLOCK_SIZE - 1) / SPH_BLOCK_SIZE;
    sph_gadget2_forces_kernel<<<blocks, SPH_BLOCK_SIZE>>>(dx, dy, dz, dvx, dvy, dvz, dm, dgas, drho, dp, dh, db, dax, day, daz, dda, ddu, dmv, n, periodic_box);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(ax_out, dax, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay_out, day, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az_out, daz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(da_dt_out, dda, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(du_dt_out, ddu, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(max_vsig_out, dmv, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dm); cudaFree(dgas);
    cudaFree(drho); cudaFree(dp); cudaFree(dh); cudaFree(db); cudaFree(dax); cudaFree(day); cudaFree(daz); cudaFree(dda); cudaFree(ddu); cudaFree(dmv);
    return 0;
}
