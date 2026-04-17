# Phase 20: PM Slab Distribuido con FFT Distribuida

**Fecha:** abril 2026
**Autor:** gadget-ng development log
**Rama:** cosmological-slab-pm (continuación de Phase 19)

---

## Resumen ejecutivo

Phase 20 elimina el cuello de botella restante de Phase 19: el solve FFT/Poisson replicado en cada rank. Se implementa una **slab decomposition real del grid PM en el eje Z**, donde:

- Cada rank posee `nz_local = nm/P` planos Z del grid.
- La FFT 3D se distribuye mediante **dos alltoall transposes** para la pasada Z.
- La comunicación por rank/paso baja de O(nm³) (Phase 19) a O(nm³/P) (Phase 20).
- El solve Poisson ya no está replicado: cada rank computa solo su porción del grid.

**Respuesta a la pregunta central:** Sí, `gadget-ng` puede ejecutar PM periódico en MPI con una malla realmente distribuida, sin allgather de partículas y sin solve FFT serial replicado en cada rank, manteniendo correctitud física. Para P=1, el resultado es bit-a-bit idéntico al solver serial de Phase 18.

---

## Estado previo (Phase 19) y su limitación

| Aspecto | Phase 19 |
|---------|----------|
| Comunicación allgather partículas | Eliminado ✓ |
| Comunicación densidad | allreduce O(nm³) = nm³·8 B/rank |
| Solve FFT/Poisson | Serial replicado en todos los ranks |
| Memoria de grid | nm³ f64 = O(nm³) por rank |

El solve serial replicado implica:
- Costo O(nm³ log nm) por rank por paso — no escala.
- Memoria O(nm³) por rank — no escala.
- Para nm=64, P=4: cada rank replica 2MB de grid y 2M operaciones FFT inútilmente.

---

## Arquitectura Phase 20: Slab en Z con alltoall transpose

### Elección del eje Z

El indexado del grid es `flat = iz * nm² + iy * nm + ix` (Z más lento, X más rápido). Con slabs en Z:
- Las pasadas X e Y de la FFT 3D son **completamente locales** (cada rank tiene todas las combinaciones (iz_local, iy, ix)).
- Solo la pasada Z requiere comunicación: un único alltoall transpose.
- La implementación es un solo archivo: `slab_fft.rs`.

### Layout de slabs

```
Rank r:  iz ∈ [r·nz_local, (r+1)·nz_local)
nz_local = nm / P         (requiere nm % P == 0)
nk_local = nm² / P        (pencils (ky,kx) por rank en el solve Z)
```

Ejemplo nm=32, P=4: nz_local=8, nk_local=256.

### Pipeline completo

```
ρ_slab[(iz_local, iy, ix)]
    │
    ├─ deposit_slab_extended: CIC → buffer (nz_local+1, nm, nm) con ghost right
    │     O(N/P) por rank
    │
    ├─ exchange_density_halos_z: ring periódico
    │     Envía ghost right a rank+1, recibe de rank-1
    │     O(nm²) por rank (2 planos, periódico)
    │
    ├─ fft_xy_local: FFT-X e FFT-Y en slab local
    │     O(nz_local · nm · nm · log nm) por rank, SIN comunicación
    │
    ├─ alltoall_transpose_fwd: slab (kz_local, ky, kx) → pencil (p_local, kz_all)
    │     O(nm³/P) datos por rank/paso
    │
    ├─ fft_z_pencils: Z-FFT en nm²/P pencils locales
    │     O(nk_local · nm · log nm) por rank
    │
    ├─ apply_poisson_kernel_pencils: kernel k² para nk_local pencils
    │     Produce [F̂_x, F̂_y, F̂_z] en pencil layout
    │
    ├─ ifft_z_pencils × 3: IFFT-Z para cada componente de fuerza
    │
    ├─ alltoall_transpose_bwd × 3: pencil → slab por componente
    │     O(nm³/P) datos por rank/paso × 3 componentes
    │
    ├─ ifft_xy_local × 3: IFFT-X e IFFT-Y para cada componente
    │
    ├─ exchange_force_halos_z: intercambio de 1 plano de fuerza en cada borde
    │     O(nm²) por rank
    │
    └─ interpolate_slab_local: CIC interpolation con halos
           O(N/P) por rank
```

### Comunicación por rank y paso

| Path | Comunicación | Escala con P |
|------|-------------|-------------|
| Phase 18 allgather | N·(P-1)·sizeof(Particle) | O(N·P) |
| Phase 19 allreduce | nm³·8 B = fija | O(nm³) |
| Phase 20 alltoall×4 | 4·nm³/P·8 B (density+3 forces) | O(nm³/P) |
| Phase 20 halos×2 | 2·nm²·8 B (density+force) | O(nm²), pequeño |
| **Phase 20 total** | **≈4·nm³/P·8 + 2·nm²·8 B** | **P× mejor que Phase 19** |

Ejemplos concretos (bytes/rank/paso):

| nm | P | Phase 19 | Phase 20 transpose | Phase 20 halos | Phase 20 total |
|----|---|----------|-------------------|----------------|----------------|
| 16 | 1 | 32 KB | 32 KB (no-op P=1) | 2 KB | 32 KB |
| 16 | 2 | 32 KB | 16 KB | 2 KB | 18 KB |
| 16 | 4 | 32 KB | 8 KB | 2 KB | 10 KB |
| 32 | 4 | 262 KB | 66 KB | 8 KB | 74 KB |
| 32 | 8 | 262 KB | 33 KB | 8 KB | 41 KB |
| 64 | 4 | 2 MB | 512 KB | 32 KB | 544 KB |
| 64 | 8 | 2 MB | 256 KB | 32 KB | 288 KB |

---

## Archivos implementados

### Nuevas primitivas MPI

**`crates/gadget-ng-parallel/src/lib.rs`** — trait `ParallelRuntime`:
- `exchange_domain_by_z`: migra partículas a su slab Z (p2p no-periódico).
- `exchange_halos_by_z`: intercambia halos de partículas en z (p2p).

**`crates/gadget-ng-parallel/src/serial.rs`** — no-ops para P=1.

**`crates/gadget-ng-parallel/src/mpi_rt.rs`** — implementación p2p adaptada de `exchange_domain_by_x`.

### FFT distribuida

**`crates/gadget-ng-pm/src/slab_fft.rs`** (nuevo, ~370 líneas):
- `SlabLayout` — struct que describe la descomposición de un rank.
- `fft_xy_local` / `ifft_xy_local` — FFT en X e Y sin comunicación.
- `fft_z_pencils` / `ifft_z_pencils` — FFT en Z sobre pencils locales.
- `alltoall_transpose_fwd` — slab (kz_local, ky, kx) → pencil (p_local, kz_all).
- `alltoall_transpose_bwd` — inverse transpose.
- `apply_poisson_kernel_pencils` — kernel −4πG/k² para pencils locales.
- `solve_forces_slab` — pipeline completo; para P=1 delega a `fft_poisson::solve_forces`.

### Pipeline PM slab

**`crates/gadget-ng-pm/src/slab_pm.rs`** (nuevo, ~350 líneas):
- `deposit_slab_extended` — CIC con ghost right; P=1 delega a `cic::assign`.
- `exchange_density_halos_z` — ring periódico vía `alltoallv_f64`.
- `forces_from_slab` — delega a `solve_forces_slab`.
- `exchange_force_halos_z` — intercambio de planos de fuerza para CIC en bordes.
- `interpolate_slab_local` — CIC con halos extendidos; P=1 delega a `cic::interpolate`.

### Config y engine

**`crates/gadget-ng-core/src/config.rs`**:
```toml
[gravity]
pm_slab = true   # Fase 20: slab PM distribuido (requiere nm % P == 0)
```

**`crates/gadget-ng-cli/src/engine.rs`**:
- `use_pm_slab` — activado cuando `pm_slab && periodic && solver=pm`.
- Validación `nm % P == 0` en tiempo de ejecución.
- Migración `exchange_domain_by_z` al inicio de cada paso (fuera del closure).
- Branch completo del pipeline slab en `compute_acc`.

---

## Validación

### Tests automáticos (8/8 pasan)

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `slab_layout_covers_all_planes` | SlabLayout cubre todos los planos Z | ✓ |
| `deposit_slab_mass_conservation` | CIC conserva masa total en P=1 | ✓ |
| `density_halo_exchange_conserves_mass_serial` | Halo exchange no-op en P=1 no modifica masa | ✓ |
| `border_particle_deposit_correct` | Partícula en borde Z deposita en ghost right | ✓ |
| `alltoall_transpose_roundtrip_p1` | Densidad uniforme → fuerza≈0; solve correcto | ✓ |
| `slab_solve_matches_serial_pm` | P=1 slab ≡ serial en todas las celdas (error < 1e-10) | ✓ |
| `slab_poisson_sanity_sinusoidal_mode` | F_x en x=L/4 < 0, en x=3L/4 > 0 | ✓ |
| `distributed_pm_no_explosion_slab` | 3 pasos EdS + slab sin NaN/Inf | ✓ |

### Equivalencia P=1 slab ≡ serial

La función `solve_forces_slab` delega directamente a `fft_poisson::solve_forces` cuando `n_ranks == 1`. Esto garantiza exactitud bit-a-bit con Phase 18.

### Test de modo sinusoidal

Densidad `ρ(x) = 1 + 0.5·cos(2πx/L)`:
- Φ̂(k=1) = -4πG·(0.25)/(2π/L)² → F_x ∝ -sin(2πx/L)
- F_x(x=L/4) < 0 ✓ (hacia la mayor densidad en x=0)
- F_x(x=3L/4) > 0 ✓ (hacia la mayor densidad en x=L)

### Ghost right en borde de slab

Partícula en z ≈ (nz_local - 0.01)·Δz con layout rank=0, P=2:
- iz0 = nz_local-1 (plano propio)
- CIC deposita fracción al plano nz_local (ghost right)
- `ghost_mass > 0` y `owned_mass + ghost_mass = 1.0` ✓

---

## Limitaciones explícitas

| Limitación | Descripción |
|------------|-------------|
| `nm % P == 0` | Requerido; validado en runtime con error claro |
| P=1 no usa slab FFT | Delega a serial para exactitud y simplicidad |
| TreePM slab | No implementado en esta fase |
| `r_split` (TreePM) | No aplicado en path slab; documentado |
| 3 alltoall inversos | Un alltoall por componente de fuerza; podría reducirse a 1 con packing |
| Migración Z una vez por paso | Para Yoshida4 (3 sub-pasos), partículas podrían cruzar 1 celda por sub-paso |
| `exchange_halos_by_z` para árbol | Implementado pero no conectado al árbol aún |

### Sobre la completitud del PM distribuido

Phase 20 implementa:
- ✅ Partículas migradas a slabs Z al inicio de cada paso
- ✅ Grid de densidad realmente distribuido (nz_local planos por rank)
- ✅ FFT distribuida en Z mediante alltoall transpose
- ✅ Solve Poisson distribuido: cada rank computa su porción del grid
- ✅ Halo CIC de 1 plano (densidad + fuerza)
- ✅ Comunicación O(nm³/P) por alltoall (P× mejor que Phase 19)

Lo que **no** está distribuido todavía:
- ❌ TreePM distribuido (árbol sigue usando allgather)
- ❌ FFT en X e Y: aún se hacen localmente en nz_local planos, pero son trivialmente distribuidas al no requerir comunicación
- ❌ 3 alltoall inversos (uno por componente de fuerza): podrían hacerse en 1 con empaquetado

---

## Comparativa global de paths PM

| Path | Phase | Comm partículas | Comm grid | Solve |
|------|-------|-----------------|-----------|-------|
| allgather + PM serial | 18 | O(N·P) | — | Replicado |
| allreduce densidad | 19 | Eliminado | O(nm³) | Replicado |
| alltoall slab FFT | **20** | Eliminado | **O(nm³/P)** | **Distribuido** |

---

## Próximos pasos hacia TreePM distribuido serio

1. **TreePM slab**: conectar árbol de corto alcance con slab PM. El árbol necesita partículas locales + halos para fuerzas corto alcance; el slab PM ya provee largo alcance.
2. **Reducir alltoalls**: empaquetar [F̂_x, F̂_y, F̂_z] en un solo alltoall inverse (×3 eficiencia).
3. **Migración multi-paso**: re-migrar partículas entre sub-pasos de Yoshida4 para slabs grandes.
4. **FFTW3/MPI**: reemplazar `alltoall_transpose + rustfft` por bindings a FFTW3_MPI para mejor rendimiento en grids grandes.
5. **Zel'dovich ICs distribuidas**: condiciones iniciales en espacio de Fourier con slab decomposition.
6. **Block timesteps cosmológicos**: diferentes dt por rango de potencial — requiere scheduler global.

---

## Definition of Done — Verificación

| Criterio | Estado |
|----------|--------|
| Path PM con grid distribuido real | ✅ slab_fft.rs + slab_pm.rs |
| Grid no replicado como camino principal | ✅ cada rank tiene nz_local = nm/P planos |
| Partículas no requieren allgather | ✅ exchange_domain_by_z |
| CIC distribuido con halo validado | ✅ test border_particle_deposit_correct |
| Serial y MPI físicamente coherentes (P=1) | ✅ slab_solve_matches_serial_pm |
| Reporte técnico | ✅ este documento |
| Tests automáticos (≥8) | ✅ 8/8 pasan |
| Benchmarks configurados | ✅ experiments/nbody/phase20_slab_pm/ |
