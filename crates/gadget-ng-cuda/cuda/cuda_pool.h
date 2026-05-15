/**
 * cuda_pool.h — Device buffer pool reutilizable para kernels CUDA.
 *
 * ## Motivación
 *
 * Los kernels stateless (SPH, MHD, Tree, RT, Cooling, Dust, Molecular)
 * hacen cudaMalloc/cudaFree en *cada* llamada, lo que agrega latencia
 * significativa (~50-100 µs por alloc) que domina el costo para N < ~10⁵.
 *
 * Pool reutiliza los buffers device entre llamadas, redimensionando solo
 * cuando el número de partículas excede la capacidad actual (doblamiento).
 *
 * ## Compatibilidad
 *
 * Solo usa CUDA Runtime API disponible desde CUDA 8.0 (sm_60).
 * No usa cudaMemPool (CUDA 11.2+) ni otras APIs recientes.
 *
 * ## Uso
 *
 * ```c
 * CudaPool* pool = cuda_pool_create(0);  // capacidad inicial 0
 * // ... en cada llamada:
 * float* d_x = cuda_pool_upload(pool, h_x, n, sizeof(float));
 * float* d_out = cuda_pool_alloc(pool, n, sizeof(float));
 * // ... lanza kernel con d_x, d_out ...
 * cuda_pool_download(pool, h_out, d_out, n * sizeof(float));
 * cuda_pool_reset(pool);  // marca todos los buffers como reutilizables
 * // ... al final:
 * cuda_pool_destroy(pool);
 * ```
 */

#ifndef GADGET_NG_CUDA_POOL_H
#define GADGET_NG_CUDA_POOL_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/** Handle opaco al pool de buffers device. */
typedef void* cuda_pool_t;

/**
 * Crea un pool vacío con capacidad inicial `initial_n` partículas
 * (0 = sin pre-asignar).
 *
 * Devuelve NULL si CUDA no está disponible o si cudaMalloc falla.
 */
cuda_pool_t cuda_pool_create(int initial_n);

/**
 * Libera todos los buffers device y el pool.
 */
void cuda_pool_destroy(cuda_pool_t pool);

/**
 * Asegura que el pool tenga capacidad para al menos `n` partículas
 * en cada buffer registrado. Si `n > capacity`, redimensiona todos
 * los buffers (la nueva capacidad será max(n, 2 * capacity_old)).
 *
 * Devuelve 0 si OK, código de error CUDA en caso contrario.
 */
int cuda_pool_ensure(cuda_pool_t pool, int n);

/**
 * Resetea el pool para la próxima serie de uploads: marca todos
 * los buffers como disponibles para reescritura (no libera memoria).
 * Debe llamarse antes de cada serie de uploads en un paso de simulación.
 */
void cuda_pool_reset(cuda_pool_t pool);

/**
 * Sube datos al device usando el slot `slot_index` del pool.
 * Si el slot no existe o es demasiado chico, se redimensiona.
 *
 * Devuelve el puntero device. Si falla, devuelve NULL.
 */
float* cuda_pool_upload_f32(cuda_pool_t pool, int slot_index,
                             const float* host_data, int n);

/**
 * Sube datos u8 al device usando el slot `slot_index` del pool.
 */
unsigned char* cuda_pool_upload_u8(cuda_pool_t pool, int slot_index,
                                    const unsigned char* host_data, int n);

/**
 * Aloca un buffer f32 de salida en el pool (cero-inicializado).
 * Usa el slot `slot_index`.
 */
float* cuda_pool_alloc_f32(cuda_pool_t pool, int slot_index, int n);

/**
 * Descarga datos f32 del device al host.
 * `n` es el número de elementos float, no bytes.
 */
int cuda_pool_download_f32(cuda_pool_t pool,
                            float* host_data, const float* device_data, int n);

/** Devuelve la capacidad actual del pool (en partículas). */
int cuda_pool_capacity(cuda_pool_t pool);

/** Devuelve el número de slots actualmente alojados. */
int cuda_pool_num_slots(cuda_pool_t pool);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GADGET_NG_CUDA_POOL_H */