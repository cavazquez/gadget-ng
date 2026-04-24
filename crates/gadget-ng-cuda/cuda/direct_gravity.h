/**
 * direct_gravity.h — Interfaz C pura del solver de gravedad directa N² CUDA.
 *
 * Diseñada para ser callable directamente desde Rust via FFI (extern "C").
 * Toda la gestión de memoria CUDA está encapsulada en el handle.
 *
 * ## Algoritmo
 *
 * Kernel de gravedad directa O(N²) con tiling en shared memory:
 *   - Cada bloque de BLOCK_SIZE hilos carga un tile de BLOCK_SIZE fuentes en
 *     shared memory y evalúa la contribución de ese tile para cada partícula
 *     objetivo del bloque.
 *   - Softening Plummer: a_i += G * m_j * r_ij / (|r_ij|² + ε²)^(3/2)
 *
 * ## Precisión
 *
 * f32 en device. Para N ≤ 10⁶ el error relativo frente a f64 es < 1e-5.
 */

#ifndef GADGET_NG_DIRECT_GRAVITY_H
#define GADGET_NG_DIRECT_GRAVITY_H

#ifdef __cplusplus
extern "C" {
#endif

/** Handle opaco al estado del solver directo; internamente CudaDirectState*. */
typedef void* cuda_direct_handle_t;

/**
 * Crea el solver de gravedad directa.
 *
 * @param eps2       Suavizado Plummer al cuadrado (ε²) en unidades internas.
 * @param block_size Número de hilos por bloque (potencia de 2; e.g. 256).
 *
 * @return Handle válido, o NULL si la inicialización falla (sin CUDA, etc.).
 */
cuda_direct_handle_t cuda_direct_create(float eps2, int block_size);

/**
 * Libera toda la memoria device y host asociada al handle.
 */
void cuda_direct_destroy(cuda_direct_handle_t h);

/**
 * Calcula aceleraciones gravitacionales directas para n partículas.
 *
 * Lanza el kernel directo O(N²/P) y copia los resultados de vuelta al host.
 *
 * @param h          Handle creado con cuda_direct_create.
 * @param x,y,z      Posiciones de las n partículas (arrays de longitud n, f32).
 * @param mass       Masas de las n partículas (array de longitud n, f32).
 * @param ax,ay,az   Buffers de salida: aceleraciones (longitud n, f32).
 * @param n          Número de partículas.
 * @param g          Constante gravitatoria.
 *
 * @return 0 si OK; código de error CUDA (cudaError_t) en caso contrario.
 */
int cuda_direct_solve(
    cuda_direct_handle_t h,
    const float* x, const float* y, const float* z,
    const float* mass,
    float* ax, float* ay, float* az,
    int n, float g
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GADGET_NG_DIRECT_GRAVITY_H */
