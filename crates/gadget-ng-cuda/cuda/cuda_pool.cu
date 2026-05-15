/**
 * cuda_pool.cu — Implementación del pool de buffers device reutilizable.
 *
 * Diseño:
 *   - Un arreglo dinámico de device pointers con capacidad por slot.
 *   - `reset` marca los slots como reutilizables sin liberar memoria.
 *   - `ensure` redimensiona todos los slots cuando n > capacity.
 *   - `upload` copia host→device (o aloca+zero para allocs de salida).
 *
 * Solo usa cudaMalloc, cudaMemcpy, cudaMemset, cudaFree — API disponible
 * desde CUDA 8.0 (sm_60). No usa cudaMemPool (CUDA 11.2+).
 */

#include "cuda_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define POOL_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA-POOL] %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            return (int)_e;                                                     \
        }                                                                       \
    } while (0)

/** Un slot individual del pool: puntero device + capacidad en elementos. */
struct CudaPoolSlot {
    void*  ptr;       /**< Puntero device (NULL si no alojado). */
    int    capacity;  /**< Capacidad en elementos (float o u8). */
    int    in_use;     /**< 1 si ocupado esta ronda, 0 si reusable. */
    int    elem_size; /**< sizeof del tipo de elemento (4 para f32, 1 para u8). */
};

struct CudaPoolState {
    CudaPoolSlot* slots;  /**< Arreglo dinámico de slots. */
    int           n_slots; /**< Número de slots alojados. */
    int           n_alloc; /**< Capacidad del arreglo slots. */
    int           capacity_n; /**< Capacidad en partículas (todos los slots >= esta). */
};

// ── Helpers internos ──────────────────────────────────────────────────────

static CudaPoolState* to_state(cuda_pool_t p) {
    return static_cast<CudaPoolState*>(p);
}

static int pool_grow_slots(CudaPoolState* s, int needed) {
    if (needed <= s->n_alloc) return 0;
    int new_alloc = needed < 16 ? 16 : needed * 2;
    auto* new_slots = static_cast<CudaPoolSlot*>(
        std::realloc(s->slots, static_cast<size_t>(new_alloc) * sizeof(CudaPoolSlot)));
    if (!new_slots) return -1;
    // Zero-fill new slots
    std::memset(new_slots + s->n_alloc, 0,
                static_cast<size_t>(new_alloc - s->n_alloc) * sizeof(CudaPoolSlot));
    s->slots = new_slots;
    s->n_alloc = new_alloc;
    return 0;
}

static int pool_ensure_slot(CudaPoolState* s, int idx, int n, int elem_size) {
    if (idx >= s->n_alloc) {
        if (pool_grow_slots(s, idx + 1) != 0) return -1;
    }
    if (idx >= s->n_slots) {
        // Inicializar slots nuevos
        for (int i = s->n_slots; i <= idx; ++i) {
            s->slots[i].ptr = nullptr;
            s->slots[i].capacity = 0;
            s->slots[i].in_use = 0;
            s->slots[i].elem_size = elem_size;
        }
        s->n_slots = idx + 1;
    }

    CudaPoolSlot& slot = s->slots[idx];
    if (slot.capacity >= n && slot.elem_size == elem_size) {
        slot.in_use = 1;
        return 0; // ya tiene capacidad suficiente
    }

    // Necesitamos (re)alojar
    size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(elem_size);
    if (slot.ptr) {
        cudaFree(slot.ptr);
        slot.ptr = nullptr;
    }
    cudaError_t err = cudaMalloc(&slot.ptr, bytes);
    if (err != cudaSuccess) {
        slot.capacity = 0;
        slot.ptr = nullptr;
        return static_cast<int>(err);
    }
    slot.capacity = n;
    slot.elem_size = elem_size;
    slot.in_use = 1;
    return 0;
}

// ── API pública ───────────────────────────────────────────────────────────

extern "C" cuda_pool_t cuda_pool_create(int initial_n) {
    auto* s = static_cast<CudaPoolState*>(std::malloc(sizeof(CudaPoolState)));
    if (!s) return nullptr;
    std::memset(s, 0, sizeof(CudaPoolState));
    s->slots = nullptr;
    s->n_slots = 0;
    s->n_alloc = 0;
    s->capacity_n = 0;

    if (initial_n > 0) {
        // Pre-asignar un slot genérico con initial_n floats
        if (pool_ensure_slot(s, 0, initial_n, static_cast<int>(sizeof(float))) != 0) {
            std::free(s);
            return nullptr;
        }
        s->capacity_n = initial_n;
    }
    return static_cast<cuda_pool_t>(s);
}

extern "C" void cuda_pool_destroy(cuda_pool_t pool) {
    if (!pool) return;
    CudaPoolState* s = to_state(pool);
    for (int i = 0; i < s->n_slots; ++i) {
        if (s->slots[i].ptr) {
            cudaFree(s->slots[i].ptr);
        }
    }
    std::free(s->slots);
    std::free(s);
}

extern "C" int cuda_pool_ensure(cuda_pool_t pool, int n) {
    if (!pool || n <= 0) return -1;
    CudaPoolState* s = to_state(pool);
    if (n <= s->capacity_n) return 0;

    int new_cap = s->capacity_n < 1 ? n : (n > 2 * s->capacity_n ? n : 2 * s->capacity_n);

    // Redimensionar todos los slots existentes
    for (int i = 0; i < s->n_slots; ++i) {
        if (s->slots[i].capacity < new_cap) {
            size_t bytes = static_cast<size_t>(new_cap) * static_cast<size_t>(s->slots[i].elem_size);
            if (s->slots[i].ptr) cudaFree(s->slots[i].ptr);
            s->slots[i].ptr = nullptr;
            cudaError_t err = cudaMalloc(&s->slots[i].ptr, bytes);
            if (err != cudaSuccess) {
                s->slots[i].capacity = 0;
                return static_cast<int>(err);
            }
            s->slots[i].capacity = new_cap;
        }
    }
    s->capacity_n = new_cap;
    return 0;
}

extern "C" void cuda_pool_reset(cuda_pool_t pool) {
    if (!pool) return;
    CudaPoolState* s = to_state(pool);
    for (int i = 0; i < s->n_slots; ++i) {
        s->slots[i].in_use = 0;
    }
}

extern "C" float* cuda_pool_upload_f32(cuda_pool_t pool, int slot_index,
                                         const float* host_data, int n) {
    if (!pool || !host_data || n <= 0) return nullptr;
    CudaPoolState* s = to_state(pool);

    if (pool_ensure_slot(s, slot_index, n, static_cast<int>(sizeof(float))) != 0) {
        return nullptr;
    }

    CudaPoolSlot& slot = s->slots[slot_index];
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    cudaMemcpy(slot.ptr, host_data, bytes, cudaMemcpyHostToDevice);
    return static_cast<float*>(slot.ptr);
}

extern "C" unsigned char* cuda_pool_upload_u8(cuda_pool_t pool, int slot_index,
                                                const unsigned char* host_data, int n) {
    if (!pool || !host_data || n <= 0) return nullptr;
    CudaPoolState* s = to_state(pool);

    if (pool_ensure_slot(s, slot_index, n, 1) != 0) {
        return nullptr;
    }

    CudaPoolSlot& slot = s->slots[slot_index];
    size_t bytes = static_cast<size_t>(n);
    cudaMemcpy(slot.ptr, host_data, bytes, cudaMemcpyHostToDevice);
    return static_cast<unsigned char*>(slot.ptr);
}

extern "C" float* cuda_pool_alloc_f32(cuda_pool_t pool, int slot_index, int n) {
    if (!pool || n <= 0) return nullptr;
    CudaPoolState* s = to_state(pool);

    if (pool_ensure_slot(s, slot_index, n, static_cast<int>(sizeof(float))) != 0) {
        return nullptr;
    }

    CudaPoolSlot& slot = s->slots[slot_index];
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    cudaMemset(slot.ptr, 0, bytes);
    return static_cast<float*>(slot.ptr);
}

extern "C" int cuda_pool_download_f32(cuda_pool_t pool,
                                        float* host_data, const float* device_data, int n) {
    (void)pool;  // No necesitamos el pool para download directo
    if (!host_data || !device_data || n <= 0) return -1;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    cudaMemcpy(host_data, device_data, bytes, cudaMemcpyDeviceToHost);
    return 0;
}

extern "C" int cuda_pool_capacity(cuda_pool_t pool) {
    if (!pool) return 0;
    return to_state(pool)->capacity_n;
}

extern "C" int cuda_pool_num_slots(cuda_pool_t pool) {
    if (!pool) return 0;
    return to_state(pool)->n_slots;
}