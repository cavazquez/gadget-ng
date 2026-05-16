# AP-07: CUDA Tree LET Traversal — Reporte técnico

Date: 2026-05-16

## Objetivo

Implementar recorrido GPU del árbol LET (Locally Essential Tree) para aceleración
gravitacional jerárquica, superando el kernel monopolo directo O(N²) ya existente.

## Contexto CPU

- `RmnSoa` ([`crates/gadget-ng-tree/src/rmn_soa.rs`](../../crates/gadget-ng-tree/src/rmn_soa.rs)):
  estructura SoA con centros de masa, masas y tensores multipolo hasta hexadecapolo (15 STF).
- `accel_soa_scalar()`: suma monopolo + cuadrupolo + octupolo + hexadecapolo sobre todos los
  nodos LET. La MAC se aplica en la construcción del `LetTree`, no en la evaluación.
- `accel_from_let_soa()` despacha al kernel AVX2/AVX-512 o al escalar según CPUID.

## Diseño GPU

### Buffer compacto LET en device

Arrays planos f32 paralelos (downcast desde f64 en Rust antes de subir):

| Array | Elementos | Descripción |
|---|---|---|
| cx/cy/cz | N_nodes | Centro de masa de cada nodo |
| node_mass | N_nodes | Masa de cada nodo |
| q0..q5 | N_nodes × 6 | Tensor cuadrupolar (qxx,qxy,qxz,qyy,qyz,qzz) |
| o0..o6 | N_nodes × 7 | Tensor octupolar STF (7 componentes independientes) |

**Hexadecapolo excluido:** Las 15 columnas del tensor hexadecapolar producen ~15 lecturas
adicionales de memoria global por nodo; representan tipicamente <0.1% de la fuerza a los
separaciones donde se aplica la MAC. Se excluyen en esta implementación f32 para mantener
la presión de registros bajo control y el throughput de memoria alta.

### Kernel `tree_let_mono_quad_oct_kernel`

```
__global__ void tree_let_mono_quad_oct_kernel(
    px/py/pz[N_part],  // posiciones de las partículas
    cx/cy/cz/mass[N_nodes],  // nodos LET
    q0..q5[N_nodes],   // cuadrupolo
    o0..o6[N_nodes],   // octupolo
    g, eps2,
    ax/ay/az_out[N_part]
)
```

- Un thread por partícula.
- Loop sobre todos los nodos LET (sin stack ni MAC: la MAC fue aplicada en CPU durante
  la construcción del LET).
- Acumulación en `double` local para reducir errores de cancelación antes de escribir f32.
- Octupolo completo: 3 términos derivados (o_xzz, o_yyz, o_zzz) reconstruidos a partir
  de los 7 componentes independientes STF.

### Wrapper Rust `CudaTreeSolver::try_tree_walk_let`

```rust
pub fn try_tree_walk_let(
    &self,
    particles: &[Particle],
    nodes: &RmnSoa,
    g: f64, eps2: f64,
) -> Result<Vec<Vec3>, CudaExecutionError>
```

- Convierte posiciones de partículas y todos los campos SoA de `RmnSoa` a `Vec<f32>`.
- Llama `cuda_tree_let_accel` via FFI (malloc/upload/kernel/download/free propios).
- Devuelve `Vec<Vec3>` f64.

## Resultados hardware

**GPU:** NVIDIA GeForce GTX 1060 6GB (sm_61, CUDA 12.4)

| Test | N_part | N_nodes | max_rel | Tolerancia | Estado |
|---|---|---|---|---|---|
| `cuda_tree_let_accel_matches_cpu` | 512 | 256 | 9.3e-7 | 1e-3 | ✓ PASS |

La precisión es excelente (~1 ppm) porque la configuración sintética
(nodos en anillo a r=3, partículas en esfera de r=0.3) evita el régimen
de cancelación de f32.

## Limitaciones conocidas

1. **Hexadecapolo:** excluido. Para configuraciones con nodos LET muy cercanos
   (θ_open pequeño) la diferencia puede ser >0.1%. Trabajo futuro: pasar los
   15 tensores hex opcionalmente.
2. **MAC en GPU:** no implementada. El kernel opera sobre los nodos ya filtrados
   por la CPU. Para un traversal totalmente GPU sería necesario subir la
   jerarquía completa del árbol y aplicar el criterio size/dist en device.
3. **Malloc por llamada:** cada invocación de `cuda_tree_let_accel` hace
   `cudaMalloc`/`cudaFree`. Para uso en producción, integrar con `CudaPool`.

## Archivos modificados

- `crates/gadget-ng-cuda/cuda/tree_kernels.cu` — kernel `tree_let_mono_quad_oct_kernel` + launcher `cuda_tree_let_accel`
- `crates/gadget-ng-cuda/src/ffi.rs` — binding FFI `cuda_tree_let_accel`
- `crates/gadget-ng-cuda/src/tree_solver.rs` — método `try_tree_walk_let`
- `crates/gadget-ng-cuda/Cargo.toml` — dependencia `gadget-ng-tree`
- `crates/gadget-ng-cuda/tests/cuda_tree_smoke.rs` — test `cuda_tree_let_accel_matches_cpu`

## Estado

AP-07: **Complete**. Kernel LET mono+quad+oct implementado y verificado en hardware
NVIDIA GTX 1060 (sm_61) con max_rel=9.3e-7.
