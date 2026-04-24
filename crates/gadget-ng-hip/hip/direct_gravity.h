/**
 * direct_gravity.h — Interfaz C pura del solver de gravedad directa N² HIP/ROCm.
 *
 * Diseñada para ser callable directamente desde Rust via FFI (extern "C").
 * Toda la gestión de memoria HIP está encapsulada en el handle.
 *
 * API análoga a la versión CUDA (cuda/direct_gravity.h) pero usando
 * hipMalloc/hipMemcpy/hipLaunchKernelGGL en lugar de sus equivalentes CUDA.
 */

#ifndef GADGET_NG_HIP_DIRECT_GRAVITY_H
#define GADGET_NG_HIP_DIRECT_GRAVITY_H

#ifdef __cplusplus
extern "C" {
#endif

/** Handle opaco al estado del solver directo; internamente HipDirectState*. */
typedef void* hip_direct_handle_t;

/**
 * Crea el solver de gravedad directa HIP.
 *
 * @param eps2       Suavizado Plummer al cuadrado (ε²) en unidades internas.
 * @param block_size Número de hilos por workgroup (potencia de 2; e.g. 256).
 *
 * @return Handle válido, o NULL si la inicialización falla.
 */
hip_direct_handle_t hip_direct_create(float eps2, int block_size);

/**
 * Libera toda la memoria device y host asociada al handle.
 */
void hip_direct_destroy(hip_direct_handle_t h);

/**
 * Calcula aceleraciones gravitacionales directas para n partículas.
 *
 * @param h          Handle creado con hip_direct_create.
 * @param x,y,z      Posiciones de las n partículas (arrays de longitud n, f32).
 * @param mass       Masas de las n partículas (array de longitud n, f32).
 * @param ax,ay,az   Buffers de salida: aceleraciones (longitud n, f32).
 * @param n          Número de partículas.
 * @param g          Constante gravitatoria.
 *
 * @return 0 si OK; código de error HIP (hipError_t) en caso contrario.
 */
int hip_direct_solve(
    hip_direct_handle_t h,
    const float* x, const float* y, const float* z,
    const float* mass,
    float* ax, float* ay, float* az,
    int n, float g
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GADGET_NG_HIP_DIRECT_GRAVITY_H */
