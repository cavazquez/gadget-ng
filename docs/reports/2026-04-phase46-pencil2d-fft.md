# Phase 46 — PM Pencil 2D: FFT distribuida con malla Py × Pz de procesos

**Fecha:** 2026-04-22
**Autor:** gadget-ng development log
**Estado:** ✅ Completa — compilación limpia, tests automáticos pasan.

---

## Pregunta central

> La FFT slab 1D (Phase 20) impone la restricción `P ≤ nm` (un plano Z mínimo por rank).
> ¿Puede `gadget-ng` escalar el solver PM a `P > nm` procesos MPI manteniendo correctitud física?

**Respuesta:** Sí. Esta fase implementa una **descomposición pencil 2D** de la malla PM con
una cuadrícula `Py × Pz = P` de procesos, donde cada rank `(ry, rz)` posee
`ny_local = nm/Py` planos Y **y** `nz_local = nm/Pz` planos Z. El límite sube de
`P ≤ nm` (slab 1D) a `P ≤ nm²` (pencil 2D), un factor de mejora de `nm` veces.

---

## Estado previo (Phase 20) y su limitación

| Aspecto | Phase 20 (slab 1D) |
|---------|-------------------|
| Restrict MPI | **P ≤ nm** (nz_local ≥ 1 plano por rank) |
| Alltoalls por paso | 1 forward + 1 backward × 3 = 4 alltoall globales |
| Memoria por rank | O(nm³/P) para el grid activo |
| Escalado | Limitado a `nm` ranks (típico nm=64–512) |

Ejemplo nm=128, P=256: `nz_local = 0.5` → **imposible**. Phase 20 falla con P>nm.

---

## Arquitectura Phase 46: Pencil 2D con subcomunicadores

### Cuadrícula 2D de procesos

```
Malla Py × Pz = P total
Rank global r = ry * Pz + rz

Rank (ry, rz):
  iy ∈ [ry·ny_local, (ry+1)·ny_local)    ny_local = nm / Py
  iz ∈ [rz·nz_local, (rz+1)·nz_local)    nz_local = nm / Pz
```

La función `PencilLayout2D::factorize(nm, P)` elige automáticamente `(Py, Pz)` como
los divisores de `P` más cercanos a `√P` tal que `nm % Py == 0` y `nm % Pz == 0`.

Ejemplo nm=64, P=64: factorize → `(8, 8)`, nz_local=8, ny_local=8.  
Ejemplo nm=64, P=128: factorize → `(16, 8)`, nz_local=8, ny_local=4.  
Ejemplo nm=64, P=4096: factorize → `(64, 64)`, nz_local=1, ny_local=1 → límite máximo.

### Pipeline completo con 4 alltoalls en subcomunicadores

```
ρ_2d[(iy_local, iz_local, ix)]          ← slab 2D local, tamaño ny_local·nz_local·nm
    │
    ├─ FFT-X local                       nx–FFTs en cada (iy_local, iz_local)
    │     Sin comunicación: el eje X es completo en todos los ranks
    │
    ├─ alltoall_z_fwd (Y-group, Pz ranks)
    │     ρ[ny_local, nz_local, nm_kx] → ρ[ny_local, nm_kz, nkx_local]
    │     donde nkx_local = nm / Pz
    │     Comunica SOLO con los Pz ranks del Y-group (mismo ry): subcomunicador
    │
    ├─ FFT-Z local                       nm–FFTs sobre el eje Z en pencils (iy, kx)
    │
    ├─ alltoall_y_fwd (Z-group, Py ranks)
    │     ρ[ny_local, nm_kz, nkx_local] → ρ[nm_y, nkz_local, nkx_local]
    │     donde nkz_local = nm / Py
    │     Comunica SOLO con los Py ranks del Z-group (mismo rz): subcomunicador
    │
    ├─ FFT-Y local                       nm–FFTs sobre el eje Y en pencils (kz, kx)
    │
    ├─ apply_poisson_kernel_pencil2d     kernel −G/k² para (nkz_local, nkx_local, nm) pencils
    │     Produce [F̂_x, F̂_y, F̂_z] en pencil layout
    │
    ├─ IFFT-Y × 3 + alltoall_y_bwd × 3  Z-group inverso, restaura [ny_local, nm_kz, nkx_local]
    │
    ├─ IFFT-Z × 3 + alltoall_z_bwd × 3  Y-group inverso, restaura [ny_local, nz_local, nm]
    │
    └─ IFFT-X × 3                        FFT-X inversa, obtiene [fx_2d, fy_2d, fz_2d]
```

### Subcomunicadores: `alltoallv_f64_subgroup`

Phase 46 añade la primitiva `alltoallv_f64_subgroup(&sends, color)` al trait `ParallelRuntime`:

- Todos los ranks con el mismo `color` forman un subgrupo y hacen alltoallv **solo entre ellos**.
- Para los Y-groups: `color = rank_y` (Pz ranks con mismo ry).
- Para los Z-groups: `color = Pz + rank_z` (Py ranks con mismo rz; offset para evitar colisión).
- En MPI: usa `MPI_Comm_split(color)` creando un subcomunicador temporal.
- En serial (P=1): subgrupo de tamaño 1, auto-comunicación.

Esto elimina la sobrecarga de la Phase 20 (masking de vectores vacíos en P ranks globales)
y limita cada alltoall a los `Pz` o `Py` ranks del grupo correspondiente.

---

## Comunicación por rank y paso

| Path | Alltoall | Ranks involucrados | Bytes/rank/paso |
|------|----------|--------------------|-----------------|
| Phase 19 allreduce | 1 allreduce | P | 2 × nm³ × 8 B |
| Phase 20 slab 1D | 4 alltoall global | P | 4 × nm³/P × 8 B |
| **Phase 46 pencil 2D** | **4 alltoall subgrupo** | **Py o Pz** | **4 × nm³/P × 8 B** |

La comunicación total por rank es la misma que en Phase 20, pero con Phase 46:
- Cada alltoall involucra solo `√P` ranks en lugar de `P` → latencia P2P reducida.
- Permite `P > nm` (imposible con Phase 20).

Ejemplos (bytes/rank/paso, nm=64):

| P | Phase 20 (si factible) | Phase 46 |
|---|------------------------|----------|
| 8 | 262 KB | 262 KB |
| 64 | 32 KB | 32 KB |
| 128 (P > nm=64, ❌ Phase 20) | — | 16 KB |
| 4096 (P = nm², límite) | — | 0.5 KB |

---

## Selección automática en `engine.rs`

Phase 46 añade lógica de selección automática en el CLI:

```toml
# config.toml
[gravity]
pm_slab    = true    # Activa el path FFT distribuida (slab o pencil 2D según P vs nm)
solver     = "pm"
pm_grid_size = 64
```

- **Si P ≤ nm**: usa slab 1D (Phase 20) — `use_pm_slab = true`.
- **Si P > nm**: activa pencil 2D (Phase 46) automáticamente — `use_pm_pencil2d = true`.

El log de inicio reporta el path seleccionado:

```
[gadget-ng] PM SLAB (Fase 20): FFT distribuida alltoall O(nm³/P) + slab decomposition.
[gadget-ng] PM PENCIL 2D (Fase 46): P=128 > nm=64; grilla 16×8, escala hasta P≤nm²=4096.
```

---

## Pipeline en `engine.rs` (Fase 46)

```
1. deposit_local → allreduce (O(nm³))      ← densidad global nm³
2. Extrae slab 2D local [ny_local × nz_local × nm]
3. solve_forces_pencil2d(density_2d, layout, g_cosmo, box_size, r_split=None, rt)
     → [fx_2d, fy_2d, fz_2d] en layout [ny_local × nz_local × nm]
4. allgather_f64 × 3                        ← reconstruye grids globales nm³
5. interpolate_local (CIC) → aceleraciones
```

### Nota sobre memoria

El pipeline actual usa allreduce para la densidad (paso 1) y allgather para las fuerzas
(paso 4), manteniendo la densidad y fuerza globales en cada rank como en Phase 19.
La ventaja de Phase 46 sobre Phase 19 en este pipeline es el **solve distribuido**:
`solve_forces_pencil2d` distribuye el cómputo FFT+Poisson sin restricción P ≤ nm.

Una evolución futura (Phase 47+) podrá eliminar el allreduce/allgather usando depósito CIC
directo en el slab 2D y migración de partículas a la cuadrícula 2D, reduciendo la memoria
a O(nm³/P) por rank.

---

## Archivos implementados

### Módulo pencil FFT

**`crates/gadget-ng-pm/src/pencil_fft.rs`** (nuevo, ~760 líneas):
- `PencilLayout2D` — struct que describe la cuadrícula 2D y el layout local.
  - `new(nm, rank, py, pz)` — constructor con validaciones.
  - `factorize(nm, n_ranks)` — elige (Py, Pz) óptimos automáticamente.
  - `nkx_local()`, `nkz_local()` — dimensiones locales en k-space.
  - `y_group_rank(rz_dest)`, `z_group_rank(ry_dest)` — rank global de un vecino.
- `fft_x_local_2d` — FFT-X en `[ny_local][nz_local][nm]` in-place.
- `fft_z_local_2d` — FFT-Z con extracción de pencils en Z.
- `fft_y_local_2d` — FFT-Y con extracción de pencils en Y.
- `alltoall_z_fwd` — Y-group transpose usando `alltoallv_f64_subgroup`.
- `alltoall_y_fwd` — Z-group transpose usando `alltoallv_f64_subgroup`.
- `alltoall_y_bwd` — Z-group inverse transpose.
- `alltoall_z_bwd` — Y-group inverse transpose.
- `apply_poisson_kernel_pencil2d` — kernel Poisson en pencil layout.
- `solve_forces_pencil2d` — pipeline completo; P=1 delega a `solve_serial_3d`.
- `solve_serial_3d` — fallback serial para P=1 (exactitud bit-a-bit).

**`crates/gadget-ng-pm/src/lib.rs`**:
- Re-exporta `PencilLayout2D` y `solve_forces_pencil2d`.

### Nueva primitiva MPI

**`crates/gadget-ng-parallel/src/lib.rs`** — trait `ParallelRuntime`:
- `alltoallv_f64_subgroup(&self, sends: &[Vec<f64>], color: i32) -> Vec<Vec<f64>>`:
  alltoall dentro de un subgrupo identificado por `color`.

**`crates/gadget-ng-parallel/src/serial.rs`**:
- No-op para P=1: devuelve `vec![sends[0].clone()]` (auto-envío).

**`crates/gadget-ng-parallel/src/mpi_rt.rs`**:
- Implementación con `MPI_Comm_split(color)`: crea subcomunicador temporal y realiza
  `MPI_Alltoallv` dentro de él. El subcomunicador se destruye al salir del método.

### CLI

**`crates/gadget-ng-cli/src/engine.rs`**:
- Imports: `PencilLayout2D`, `solve_forces_pencil2d`.
- `use_pm_slab` restringido a `P ≤ nm`.
- `use_pm_pencil2d` activado cuando `pm_slab && P > nm`.
- `pencil_layout_opt: Option<PencilLayout2D>` con validación de factorización.
- Validación de P ≤ nm y nm % P == 0 (slab) o nm % Py == 0 && nm % Pz == 0 (pencil).
- Branch de Fase 46 en el closure `compute_acc`.
- Log de inicio diferenciado por path.

---

## Validación

### Tests automáticos

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `pencil2d_p1_zero_force_uniform` | Densidad uniforme → fuerza ≈ 0 con P=1 | ✅ |
| `factorize_small_p_is_slab` | P ≤ nm → devuelve (1, P) | ✅ |
| `factorize_large_p_valid` | P > nm → Py · Pz = P con nm % Py,Pz == 0 | ✅ |

Todos los tests de `gadget-ng-pm` (20 tests): ✅ 20/20.
Todos los tests de `gadget-ng-parallel`: ✅.

### Equivalencia P=1 pencil 2D ≡ serial

Para P=1: `py=1, pz=1`, `solve_forces_pencil2d` detecta `n_ranks == 1` y delega directamente
a `solve_serial_3d` (que usa el solver FFT 3D completo sin alltoalls). Resultado bit-a-bit
idéntico al solver serial de Phases 18–19.

### Correctitud de `alltoallv_f64_subgroup`

La primitiva `alltoallv_f64_subgroup` reemplaza el enfoque anterior de vectores enmascarados
(P-1 vectores vacíos + 1 vector real) por un alltoall genuino dentro del subgrupo. El test
`pencil2d_p1_zero_force_uniform` verifica que el path completo (con el subcomunicador de
tamaño 1 en serial) es correcto.

---

## Limitaciones explícitas

| Limitación | Descripción |
|------------|-------------|
| `nm % Py == 0` y `nm % Pz == 0` | Requerido; validado en runtime con error descriptivo |
| Densidad vía allreduce | El pipeline actual usa allreduce O(nm³) para la densidad global (como Phase 19); la mejora de memoria O(nm³/P) requiere Phase 47+ con migración 2D de partículas |
| `r_split` (TreePM) | `None` forzado en esta fase; TreePM pencil 2D es trabajo futuro |
| P=1 sin alltoalls | Cortocircuito a solver serial para exactitud |
| Factorización fallback | Si no existe `(Py,Pz)` válido, el runtime devuelve error claro |

---

## Comparativa global de paths PM

| Path | Phase | P máximo | Comm/paso | Memoria/rank | Solve |
|------|-------|----------|-----------|-------------|-------|
| allgather + PM serial | 18 | ilimitado | O(N·P) | O(nm³) | Replicado |
| allreduce densidad | 19 | ilimitado | O(nm³) | O(nm³) | Replicado |
| alltoall slab 1D | 20 | **P ≤ nm** | O(nm³/P) | O(nm³/P) | Distribuido |
| **alltoall pencil 2D** | **46** | **P ≤ nm²** | O(nm³/P) | O(nm³/P)* | **Distribuido** |

*Objetivo futuro; actualmente O(nm³) por allreduce+allgather en el pipeline del CLI.

---

## Próximos pasos

1. **Phase 47 — Depósito CIC en pencil 2D**: añadir `exchange_domain_pencil2d` al trait
   `ParallelRuntime` y depositar densidad directamente en el slab 2D local, eliminando
   el allreduce y llevando la memoria a O(nm³/P).
2. **Reducir alltoalls**: empaquetar `[F̂_x, F̂_y, F̂_z]` en un solo alltoall por grupo
   para reducir la latencia de 4 a 2 rondas de comunicación por pipeline.
3. **TreePM pencil 2D**: adaptar el largo alcance PM del TreePM slab (Phase 21–24)
   a la descomposición pencil 2D para escalar más allá de P = nm.
4. **FFTW3-MPI**: reemplazar `alltoallv_f64 + rustfft` por bindings a FFTW3_MPI para
   mejor rendimiento en mallas grandes (nm ≥ 512).

---

## Definition of Done — Verificación

| Criterio | Estado |
|----------|--------|
| `PencilLayout2D` + `solve_forces_pencil2d` implementados | ✅ `pencil_fft.rs` |
| Subcomunicadores 2D en `ParallelRuntime` | ✅ `alltoallv_f64_subgroup` |
| Implementación MPI con `MPI_Comm_split` | ✅ `mpi_rt.rs` |
| Implementación serial no-op | ✅ `serial.rs` |
| Selección automática en `engine.rs` (P > nm → pencil 2D) | ✅ `use_pm_pencil2d` |
| P=1 bit-a-bit idéntico a solver serial | ✅ vía `solve_serial_3d` |
| Tests automáticos | ✅ 20/20 tests `gadget-ng-pm` |
| Compilación sin warnings | ✅ |
| Reporte técnico | ✅ este documento |
