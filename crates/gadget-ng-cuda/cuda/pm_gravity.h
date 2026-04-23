/**
 * pm_gravity.h — Interfaz C pura del solver PM CUDA.
 *
 * Diseñada para ser callable directamente desde Rust via FFI (extern "C").
 * Toda la gestión de memoria CUDA (device + host) está encapsulada en el handle.
 */

#ifndef GADGET_NG_PM_GRAVITY_H
#define GADGET_NG_PM_GRAVITY_H

#ifdef __cplusplus
extern "C" {
#endif

/** Handle opaco al solver PM; internamente es un puntero a CudaPmState. */
typedef void* cuda_pm_handle_t;

/**
 * Crea el solver PM para una grilla de `grid_size³` celdas y caja periódica
 * de lado `box_size` (en las mismas unidades que las posiciones de partículas).
 *
 * Aloca memoria device (density grid, campos de fuerza, planes cuFFT).
 *
 * @return Handle válido, o NULL si la inicialización falla (sin CUDA, sin VRAM, …).
 */
cuda_pm_handle_t cuda_pm_create(int grid_size, float box_size);

/**
 * Libera toda la memoria device y host asociada al handle.
 * Después de llamar a esta función el handle queda inválido.
 */
void cuda_pm_destroy(cuda_pm_handle_t h);

/**
 * Ejecuta el pipeline PM completo:
 *   1. CIC assign  — asignar masas a la grilla de densidad
 *   2. FFT forward — cuFFT 3D R→C
 *   3. Poisson     — Φ(k) = −4πG·ρ(k)/k² + diferenciación espectral → F_α(k)
 *   4. FFT inverse — 3× cuFFT 3D C→R (uno por componente de fuerza)
 *   5. CIC interp  — interpolar fuerza en posiciones de partículas
 *
 * Las posiciones deben estar en [0, box_size).
 *
 * @param h      Handle creado con cuda_pm_create.
 * @param x,y,z  Posiciones de las n partículas (longitud n cada una).
 * @param mass   Masas de las n partículas (longitud n).
 * @param ax,ay,az  Buffers de salida: aceleraciones (longitud n).
 * @param n      Número de partículas.
 * @param eps2   Suavizado Plummer² (reservado; no usado en PM puro).
 * @param g      Constante gravitatoria.
 *
 * @return 0 si OK; código de error CUDA (cudaError_t) en caso contrario.
 */
int cuda_pm_solve(
    cuda_pm_handle_t h,
    const float* x, const float* y, const float* z,
    const float* mass,
    float* ax, float* ay, float* az,
    int n, float eps2, float g
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GADGET_NG_PM_GRAVITY_H */
