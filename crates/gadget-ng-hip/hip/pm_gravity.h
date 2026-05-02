/**
 * pm_gravity.h — Interfaz C pura del solver PM HIP/ROCm.
 *
 * Misma interfaz que la versión CUDA para facilitar el intercambio en el motor.
 * Callable directamente desde Rust via FFI (extern "C").
 */

#ifndef GADGET_NG_PM_GRAVITY_HIP_H
#define GADGET_NG_PM_GRAVITY_HIP_H

#ifdef __cplusplus
extern "C" {
#endif

/** Handle opaco al solver PM; internamente es un puntero a HipPmState. */
typedef void* hip_pm_handle_t;

/**
 * Crea el solver PM para una grilla de `grid_size³` celdas y caja periódica
 * de lado `box_size`.
 *
 * @return Handle válido, o NULL si la inicialización falla.
 */
hip_pm_handle_t hip_pm_create(int grid_size, float box_size);

/**
 * Libera todos los recursos device y host asociados al handle.
 */
void hip_pm_destroy(hip_pm_handle_t h);

/**
 * Ejecuta el pipeline PM completo con ROCm/rocFFT.
 *
 * @param h       Handle creado con hip_pm_create.
 * @param x,y,z   Posiciones de las n partículas en [0, box_size).
 * @param mass    Masas (longitud n).
 * @param ax,ay,az  Buffers de salida: aceleraciones (longitud n).
 * @param n       Número de partículas.
 * @param eps2    Suavizado² (reservado).
 * @param g       Constante gravitatoria.
 * @param r_split Radio Gaussiano TreePM (>0 filtra en k-space); ≤0 sin filtro.
 *
 * @return 0 si OK; código de error HIP en caso contrario.
 */
int hip_pm_solve(
    hip_pm_handle_t h,
    const float* x, const float* y, const float* z,
    const float* mass,
    float* ax, float* ay, float* az,
    int n, float eps2, float g, float r_split
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GADGET_NG_PM_GRAVITY_HIP_H */
