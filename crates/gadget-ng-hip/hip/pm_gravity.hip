/**
 * pm_gravity.hip — Solver PM GPU con HIP + rocFFT.
 *
 * ~95% idéntico al equivalente CUDA (pm_gravity.cu); las diferencias son:
 *   - `#include <hip/hip_runtime.h>` en lugar de `<cuda_runtime.h>`
 *   - `hipMalloc` / `hipFree` / `hipMemcpy` / `hipMemset` / `hipDeviceSynchronize`
 *   - API rocFFT en lugar de cuFFT (ver sección "Diferencia clave rocFFT vs cuFFT")
 *   - Tipos rocfft_complex en lugar de cufftComplex
 *   - Macros HIP_CHECK / ROCFFT_CHECK
 *
 * Pipeline:
 *   1. cic_assign_kernel  — masas → grilla de densidad (CIC, periódico, atomicAdd)
 *   2. rocfft R→C         — ρ(x) → ρ(k)
 *   3. poisson_kernel     — Φ(k) = −4πG·ρ(k)/k²; F_α(k) = −ik_α·Φ(k)
 *   4. rocfft C→R × 3    — F_x/y/z(k) → F_x/y/z(x)
 *   5. cic_interp_kernel  — grilla de fuerza → aceleraciones en posición de partículas
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>

#include "pm_gravity.h"

// ── Macros de comprobación de errores ─────────────────────────────────────────

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t _e = (call);                                                 \
        if (_e != hipSuccess) {                                                 \
            fprintf(stderr, "[HIP] %s:%d: %s\n",                               \
                    __FILE__, __LINE__, hipGetErrorString(_e));                 \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

#define ROCFFT_CHECK(call)                                                      \
    do {                                                                        \
        rocfft_status _r = (call);                                              \
        if (_r != rocfft_status_success) {                                      \
            fprintf(stderr, "[rocFFT] %s:%d: error %d\n",                      \
                    __FILE__, __LINE__, (int)_r);                               \
            return -1;                                                          \
        }                                                                       \
    } while (0)

/* rocfft_complex es un struct { float x, y; } */
typedef rocfft_complex HipComplex;

// ── Estado interno ────────────────────────────────────────────────────────────

struct HipPmState {
    int   N;
    float box_size;
    int   nc;

    /* Buffers device */
    float*      d_rho;
    HipComplex* d_rho_k;
    HipComplex* d_fx_k;
    HipComplex* d_fy_k;
    HipComplex* d_fz_k;
    float*      d_fx;
    float*      d_fy;
    float*      d_fz;

    /* Buffers de partículas (capacidad dinámica) */
    float* d_x;
    float* d_y;
    float* d_z;
    float* d_mass;
    float* d_ax;
    float* d_ay;
    float* d_az;
    int    d_cap;

    /* Planes rocFFT */
    rocfft_plan plan_r2c;
    rocfft_plan plan_c2r;
    rocfft_execution_info exec_info;
    void*  work_buf;
    size_t work_buf_size;
};

// ── Kernel 1: CIC assign ──────────────────────────────────────────────────────

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

// ── Kernel 2: Poisson en k-space ─────────────────────────────────────────────

__global__ void poisson_kernel(
    const HipComplex* __restrict__ rho_k,
    HipComplex* __restrict__ fx_k,
    HipComplex* __restrict__ fy_k,
    HipComplex* __restrict__ fz_k,
    int N, float two_pi_over_box, float four_pi_G, float norm)
{
    int nc    = N / 2 + 1;
    int total = N * N * nc;
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int iz  = idx % nc;
    int tmp = idx / nc;
    int iy  = tmp % N;
    int ix  = tmp / N;

    float kx = (ix <= N/2) ? (float)ix : (float)(ix - N);
    float ky = (iy <= N/2) ? (float)iy : (float)(iy - N);
    float kz = (float)iz;

    kx *= two_pi_over_box;
    ky *= two_pi_over_box;
    kz *= two_pi_over_box;

    float k2 = kx*kx + ky*ky + kz*kz;

    HipComplex rho;
    rho.x = rho_k[idx].x * norm;
    rho.y = rho_k[idx].y * norm;

    if (k2 == 0.0f) {
        fx_k[idx] = {0.0f, 0.0f};
        fy_k[idx] = {0.0f, 0.0f};
        fz_k[idx] = {0.0f, 0.0f};
        return;
    }

    float factor = -four_pi_G / k2;
    HipComplex phi;
    phi.x = factor * rho.x;
    phi.y = factor * rho.y;

    fx_k[idx] = { kx * phi.y,  -kx * phi.x };
    fy_k[idx] = { ky * phi.y,  -ky * phi.x };
    fz_k[idx] = { kz * phi.y,  -kz * phi.x };
}

// ── Kernel 3: CIC interpolation ───────────────────────────────────────────────

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

// ── Helpers rocFFT ────────────────────────────────────────────────────────────

/**
 * Crea un plan rocFFT 3D in-place R→C o C→R.
 * rocFFT requiere especificar el tamaño en orden [lento, ..., rápido].
 */
static rocfft_plan create_rocfft_plan(
    int N, rocfft_transform_type type,
    size_t* work_size_out)
{
    size_t lengths[3] = { (size_t)N, (size_t)N, (size_t)N };
    rocfft_plan plan = nullptr;
    rocfft_plan_description desc = nullptr;
    rocfft_plan_description_create(&desc);

    rocfft_plan_create(
        &plan,
        rocfft_placement_notinplace,   /* in-place no funciona bien para R2C */
        type,
        rocfft_precision_single,
        3,
        lengths,
        1,     /* número de transformadas */
        desc
    );
    rocfft_plan_description_destroy(desc);

    if (plan && work_size_out) {
        rocfft_plan_get_work_buffer_size(plan, work_size_out);
    }
    return plan;
}

// ── API C ─────────────────────────────────────────────────────────────────────

extern "C" {

hip_pm_handle_t hip_pm_create(int grid_size, float box_size)
{
    /* Inicializar rocFFT (idempotente si ya se llamó) */
    if (rocfft_setup() != rocfft_status_success) return nullptr;

    HipPmState* s = new (std::nothrow) HipPmState();
    if (!s) return nullptr;

    s->N        = grid_size;
    s->box_size = box_size;
    s->nc       = grid_size / 2 + 1;
    s->d_cap    = 0;
    s->d_x = s->d_y = s->d_z = s->d_mass = nullptr;
    s->d_ax = s->d_ay = s->d_az = nullptr;
    s->work_buf = nullptr;
    s->work_buf_size = 0;

    long long N3  = (long long)grid_size * grid_size * grid_size;
    long long Nnc = (long long)grid_size * grid_size * s->nc;

    if (hipMalloc(&s->d_rho,   N3  * sizeof(float))      != hipSuccess) goto fail;
    if (hipMalloc(&s->d_rho_k, Nnc * sizeof(HipComplex)) != hipSuccess) goto fail;
    if (hipMalloc(&s->d_fx_k,  Nnc * sizeof(HipComplex)) != hipSuccess) goto fail;
    if (hipMalloc(&s->d_fy_k,  Nnc * sizeof(HipComplex)) != hipSuccess) goto fail;
    if (hipMalloc(&s->d_fz_k,  Nnc * sizeof(HipComplex)) != hipSuccess) goto fail;
    if (hipMalloc(&s->d_fx,    N3  * sizeof(float))      != hipSuccess) goto fail;
    if (hipMalloc(&s->d_fy,    N3  * sizeof(float))      != hipSuccess) goto fail;
    if (hipMalloc(&s->d_fz,    N3  * sizeof(float))      != hipSuccess) goto fail;

    /* Planes rocFFT */
    {
        size_t ws_r2c = 0, ws_c2r = 0;
        s->plan_r2c = create_rocfft_plan(
            grid_size, rocfft_transform_type_real_forward, &ws_r2c);
        s->plan_c2r = create_rocfft_plan(
            grid_size, rocfft_transform_type_real_inverse, &ws_c2r);
        if (!s->plan_r2c || !s->plan_c2r) goto fail;

        /* Work buffer compartido (tomar el máximo) */
        s->work_buf_size = (ws_r2c > ws_c2r) ? ws_r2c : ws_c2r;
        if (s->work_buf_size > 0) {
            if (hipMalloc(&s->work_buf, s->work_buf_size) != hipSuccess) goto fail;
        }

        rocfft_execution_info_create(&s->exec_info);
        if (s->work_buf_size > 0) {
            rocfft_execution_info_set_work_buffer(
                s->exec_info, s->work_buf, s->work_buf_size);
        }
    }

    return (hip_pm_handle_t)s;

fail:
    hip_pm_destroy((hip_pm_handle_t)s);
    return nullptr;
}

void hip_pm_destroy(hip_pm_handle_t h)
{
    if (!h) return;
    HipPmState* s = (HipPmState*)h;
    if (s->d_rho)   hipFree(s->d_rho);
    if (s->d_rho_k) hipFree(s->d_rho_k);
    if (s->d_fx_k)  hipFree(s->d_fx_k);
    if (s->d_fy_k)  hipFree(s->d_fy_k);
    if (s->d_fz_k)  hipFree(s->d_fz_k);
    if (s->d_fx)    hipFree(s->d_fx);
    if (s->d_fy)    hipFree(s->d_fy);
    if (s->d_fz)    hipFree(s->d_fz);
    if (s->d_x)     hipFree(s->d_x);
    if (s->d_y)     hipFree(s->d_y);
    if (s->d_z)     hipFree(s->d_z);
    if (s->d_mass)  hipFree(s->d_mass);
    if (s->d_ax)    hipFree(s->d_ax);
    if (s->d_ay)    hipFree(s->d_ay);
    if (s->d_az)    hipFree(s->d_az);
    if (s->work_buf)  hipFree(s->work_buf);
    if (s->plan_r2c)  rocfft_plan_destroy(s->plan_r2c);
    if (s->plan_c2r)  rocfft_plan_destroy(s->plan_c2r);
    if (s->exec_info) rocfft_execution_info_destroy(s->exec_info);
    delete s;
    rocfft_cleanup();
}

int hip_pm_solve(
    hip_pm_handle_t h,
    const float* x, const float* y, const float* z,
    const float* mass,
    float* ax, float* ay, float* az,
    int n, float /*eps2*/, float g)
{
    if (!h || n <= 0) return 0;
    HipPmState* s = (HipPmState*)h;
    int N    = s->N;
    long long N3 = (long long)N * N * N;

    /* Reasignar buffers de partículas si es necesario */
    if (n > s->d_cap) {
        if (s->d_x)    hipFree(s->d_x);
        if (s->d_y)    hipFree(s->d_y);
        if (s->d_z)    hipFree(s->d_z);
        if (s->d_mass) hipFree(s->d_mass);
        if (s->d_ax)   hipFree(s->d_ax);
        if (s->d_ay)   hipFree(s->d_ay);
        if (s->d_az)   hipFree(s->d_az);
        HIP_CHECK(hipMalloc(&s->d_x,    n * sizeof(float)));
        HIP_CHECK(hipMalloc(&s->d_y,    n * sizeof(float)));
        HIP_CHECK(hipMalloc(&s->d_z,    n * sizeof(float)));
        HIP_CHECK(hipMalloc(&s->d_mass, n * sizeof(float)));
        HIP_CHECK(hipMalloc(&s->d_ax,   n * sizeof(float)));
        HIP_CHECK(hipMalloc(&s->d_ay,   n * sizeof(float)));
        HIP_CHECK(hipMalloc(&s->d_az,   n * sizeof(float)));
        s->d_cap = n;
    }

    /* Copiar partículas host → device */
    HIP_CHECK(hipMemcpy(s->d_x,    x,    n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(s->d_y,    y,    n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(s->d_z,    z,    n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(s->d_mass, mass, n*sizeof(float), hipMemcpyHostToDevice));

    /* Paso 1: limpiar grilla */
    HIP_CHECK(hipMemset(s->d_rho, 0, N3 * sizeof(float)));

    /* Paso 2: CIC assign */
    {
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        float inv_box = 1.0f / s->box_size;
        hipLaunchKernelGGL(cic_assign_kernel, blocks, threads, 0, 0,
            s->d_x, s->d_y, s->d_z, s->d_mass,
            s->d_rho, N, inv_box, n);
        HIP_CHECK(hipGetLastError());
    }

    /* Paso 3: rocFFT R→C */
    {
        void* in_ptr  = (void*)s->d_rho;
        void* out_ptr = (void*)s->d_rho_k;
        ROCFFT_CHECK(rocfft_execute(s->plan_r2c,
            (void**)&in_ptr, (void**)&out_ptr, s->exec_info));
    }

    /* Paso 4: Poisson kernel */
    {
        int total   = N * N * s->nc;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        float two_pi_over_box = 2.0f * (float)M_PI / s->box_size;
        float four_pi_G = 4.0f * (float)M_PI * g;
        float norm = 1.0f / (float)N3;
        hipLaunchKernelGGL(poisson_kernel, blocks, threads, 0, 0,
            s->d_rho_k,
            s->d_fx_k, s->d_fy_k, s->d_fz_k,
            N, two_pi_over_box, four_pi_G, norm);
        HIP_CHECK(hipGetLastError());
    }

    /* Paso 5: rocFFT C→R para cada componente */
    {
        void* fxk = (void*)s->d_fx_k;  void* fx = (void*)s->d_fx;
        ROCFFT_CHECK(rocfft_execute(s->plan_c2r,
            (void**)&fxk, (void**)&fx, s->exec_info));

        void* fyk = (void*)s->d_fy_k;  void* fy = (void*)s->d_fy;
        ROCFFT_CHECK(rocfft_execute(s->plan_c2r,
            (void**)&fyk, (void**)&fy, s->exec_info));

        void* fzk = (void*)s->d_fz_k;  void* fz = (void*)s->d_fz;
        ROCFFT_CHECK(rocfft_execute(s->plan_c2r,
            (void**)&fzk, (void**)&fz, s->exec_info));
    }

    /* Paso 6: CIC interpolation */
    {
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        float inv_box = 1.0f / s->box_size;
        hipLaunchKernelGGL(cic_interp_kernel, blocks, threads, 0, 0,
            s->d_x, s->d_y, s->d_z,
            s->d_fx, s->d_fy, s->d_fz,
            s->d_ax, s->d_ay, s->d_az,
            N, inv_box, n);
        HIP_CHECK(hipGetLastError());
    }

    /* Copiar resultados device → host */
    HIP_CHECK(hipMemcpy(ax, s->d_ax, n*sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ay, s->d_ay, n*sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(az, s->d_az, n*sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}

} /* extern "C" */
