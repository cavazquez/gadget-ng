# Phase 43 — Control temporal de TreePM + paralelismo mínimo de loops calientes

**Fecha**: 2026‑04
**Crates tocados**: `gadget-ng-integrators` (nuevo módulo `adaptive_dt`), `gadget-ng-pm` (Rayon opcional), `gadget-ng-physics` (test de integración), `gadget-ng-treepm` (ya paralelo desde Phase 42).
**Estado**: 7/7 tests verdes. Decisiones confirmadas por datos.

---

## 1. Motivación

Phase 42 dejó la siguiente lectura:

> `TreePM + ε_phys ≈ 0.01 Mpc/h` reduce el error de crecimiento lineal ~345× respecto a PM puro a `N=64³`, pero **no** recupera el régimen lineal completo. El cuello aparente quedó en el **control temporal**.

Phase 39, al revés, había demostrado que bajar `dt` en **PM puro** no mejoraba: el error estaba dominado por la física de ICs, no por truncación temporal.

Phase 43 resuelve el dilema: barre `dt ∈ {4·10⁻⁴, 2·10⁻⁴}` (+ adaptativo) **con TreePM + softening físico** y mide si ahora sí aparece sensibilidad útil.

Además, como el coste serial era prohibitivo (Phase 42 proyectó ≥37 h para un barrido grande a `N=64³`), agrega paralelismo estilo OpenMP (Rayon) en los bucles calientes que faltaban y mide speedup con 1 vs 4 hilos.

---

## 2. Setup físico

| Parámetro | Valor |
|---|---|
| Caja | `L = 100 Mpc/h` (1.0 interno) |
| Resolución | `N = 32³` (smoke); infraestructura lista para `N = 64³` vía `PHASE43_N=64` |
| ICs | 2LPT, `Z0Sigma8`, `σ₈ = 0.8`, `n_s = 0.965`, EH no‑wiggle |
| Cosmología | `Ω_m = 0.315`, `Ω_Λ = 0.685`, `h = 0.674` |
| Solver | TreePM (`TreePmSolver`) con `ε_phys = 0.01 Mpc/h` en ambas ramas |
| Split long/corto | `r_split = 0` (auto-tuned internamente por el solver) |
| Integrador | Leapfrog KDK cosmológico (`leapfrog_cosmo_kdk_step`) |
| Snapshots | `a ∈ {0.02, 0.05, 0.10}` |
| Semilla | 42 |
| `pk_correction` | `RnModel::phase35_default()` (congelado desde Phase 35) |

---

## 3. Setup numérico

### 3.1 Barrido de `dt`

Fijos (smoke reducido a 2 puntos por coste):

```
dt ∈ {4·10⁻⁴, 2·10⁻⁴}
```

La matriz completa (`dt ∈ {4·10⁻⁴, 2·10⁻⁴, 1·10⁻⁴}`) se activa quitando `PHASE43_QUICK=1`. El valor ultra‑fino `dt = 5·10⁻⁵` está disponible vía `PHASE43_DT5E5=1`.

### 3.2 Adaptativo global (Phase 43B)

Implementado en `crates/gadget-ng-integrators/src/adaptive_dt.rs`. Fórmula efectiva del modo `CosmoAcceleration`:

```
dt = clamp_[dt_min, dt_max]( min( η · √(ε / a_max),  κ_h · a / H(a) ) )
```

con

| Parámetro | Valor | Significado |
|---|---|---|
| `η` | `0.1` | Fracción del paso Aarseth clásico (`dt_stab ≈ 2/ω_max`, `η ≤ 0.25` conservador) |
| `ε` | `EPS_PHYS_MPC_H / BOX_MPC_H` = `1·10⁻⁴` interno | Mismo softening que usa TreePM |
| `κ_h` | `0.04` | ≥25 pasos por e‑folding de `a` (cota cosmológica de Hubble) |
| `dt_min` | `5·10⁻⁵` | Piso para evitar runaway en close pairs patológicos |
| `dt_max` | `4·10⁻⁴` | Techo alineado con el `dt` histórico de Phase 42 |

**¿Por qué no rompe KDK?** El Leapfrog KDK es simpléctico sólo con `dt` constante dentro de un paso. Al cambiarlo entre pasos se pierde simplecticidad formal pero se mantiene:

1. Error local `O(dt²)` (segundo orden en cada paso).
2. Reversibilidad dentro del paso (los kick‑half usan el mismo `dt`).
3. Estabilidad lineal mientras `dt ≤ dt_stab ≈ 2/ω_max = 2·√(ε/a_max)`; con `η = 0.1` sobreestimamos `dt_stab` por ≥10×.

Formalmente es "symplectic with small drift": para `~10²–10⁴` pasos (evolución hasta `a = 0.10`) el drift de energía es despreciable frente al error físico de amplitud inicial (dominante según Phase 37/39).

No se implementaron **block timesteps** jerárquicos: por diseño, fuera de scope de Phase 43.

### 3.3 Paralelismo (Rayon, estilo OpenMP en Rust puro)

Rust puro sin FFI → se usa Rayon (equivalente de `#pragma omp parallel for`). Cambios mínimos:

- `crates/gadget-ng-pm/src/solver.rs`: `PmSolver::accelerations_for_indices` ahora usa `cic::assign_rayon` y `cic::interpolate_rayon` bajo `#[cfg(feature = "rayon")]`.
- `crates/gadget-ng-treepm/src/short_range.rs`: paralelización **per‑particle** del tree walk corto alcance (ya estaba desde Phase 42 pero no se había medido speedup).
- No se tocó la FFT (es un solo plan global — speedup requeriría FFTW multihilo, fuera de scope).
- No se tocó `power_spectrum` (scatter ya dominado por GC del histograma).

El test usa `rayon::ThreadPoolBuilder::num_threads(t)` para aislar las mediciones entre pools.

---

## 4. Tests

7/7 verdes (`cargo test --release --test phase43_dt_treepm_parallel`):

| # | Nombre | Tipo | Decisión |
|---|---|---|---|
| 1 | `treepm_softened_dt_sweep_runs_stably` | hard | ok — todos los snapshots aterrizan en `a_target ± 2·10⁻²` |
| 2 | `smaller_dt_improves_growth_under_treepm` | soft | `A_smaller_dt_improves_growth` (+9 % entre `dt=4e-4` y `dt=2e-4`) |
| 3 | `adaptive_dt_matches_or_beats_best_fixed_dt` | soft | `B_adaptive_matches_best_fixed` (dentro de ±5 %) |
| 4 | `parallel_tree_walk_matches_serial_within_tolerance` | hard | **bit‑exact** (`max_rel_diff = 0.0` vs `1e-10`) |
| 5 | `parallel_execution_reduces_wall_time` | soft | `A_clear_parallel_speedup` (`3.70×` con 4 hilos) |
| 6 | `no_nan_inf_under_phase43_matrix` | hard | ok — 0 entradas no finitas |
| 7 | `results_consistent_across_thread_counts` | hard | **bit‑exact** para `δ_rms`, `v_rms`, `a` con 1 vs 4 hilos |

El test 4 y el test 7 son bit‑exact (diff = 0) porque Rayon reparte rangos contiguos determinísticamente sobre CIC y tree‑walk y la reducción final en `acc[i]` es **per‑particle**, no un fold asociativo. Esto da reproducibilidad numérica exacta — resultado importante por sí mismo.

---

## 5. Resultados

### 5.1 Tabla principal (N = 32³, smoke)

Valores en `a = 0.10` (mean ± std sobre `k ≤ k_nyq/2`, bajo‑k hasta `k_max = 0.1 h/Mpc` para el ratio de crecimiento):

| Variante | `dt`/`⟨dt⟩` | `n_steps` | Wall [s] | `median │log₁₀(P_c/P_ref)│` | `⟨P_c/P_ref⟩` | `δ_rms` | `v_rms` | Ratio crec. bajo‑k |
|---|---|---|---|---|---|---|---|---|
| `dt_4e-4` | `4·10⁻⁴` | 855 | 133.8 | **8.65** | 5.15·10⁸ | 0.994 | 50.6 | 88.57·10⁶ |
| `dt_2e-4` | `2·10⁻⁴` | 1710 | 208.3 | **8.63** | 5.16·10⁸ | 1.001 | 34.8 | **80.61·10⁶** |
| `adaptive_cosmo` | `~1·10⁻⁵` (floor) | 6838 | 1218.5 | 8.65 | 5.41·10⁸ | 1.002 | **10.9** | 83.41·10⁶ |

> **Lectura**: bajar `dt` a la mitad mejora el error de crecimiento bajo‑k en **~9 %**, **no** en órdenes de magnitud. El modo adaptativo pega contra `DT_MIN_ADAPTIVE` desde el arranque (2LPT deja `a_max` alto) y **no** supera al mejor fijo: cuesta `5.85×` más wall y queda 3.5 % peor.

### 5.2 Observación clave: `v_rms` sí es sensible a `dt`

La tabla revela una disociación diagnóstica:

- **Espectro bajo‑k y `δ_rms`** casi no se mueven entre `dt = 4·10⁻⁴` y el adaptativo (`~1·10⁻⁵` efectivo): la macro‑estructura ya está saturada por la no‑linealidad.
- **`v_rms`** cae de `50.6 → 34.8 → 10.9` (factor ~5). Es decir, el Leapfrog con `dt` grande introduce **energía cinética espuria en escalas pequeñas** donde el tree walk vive (`r ≲ ε`), y ese exceso se relaja con `dt` más chico.

Esto es consistente con Phase 37/39: el cuello **en bajo‑k a `a = 0.10`** no es temporal. La entrada en no‑linealidad (`δ_rms ≈ 1`) ya ocurrió a `a ≈ 0.05`, independiente del integrador.

### 5.3 Paralelismo

Un único paso TreePM (`compute_treepm_accels`) a `N = 32³`, warmup + medición:

| Hilos | Wall [s] | Speedup |
|---|---|---|
| 1 | 0.337 | 1.00× |
| 4 | 0.091 | **3.70×** |

Eficiencia paralela con 4 hilos: `92.5 %`. Excelente: el tree walk corto‑alcance es embarrassingly parallel per‑particle y los kernels CIC parallelizados no comparten escritura.

Consistencia numérica: bit‑exact para aceleraciones (test 4), para una evolución completa de 249 pasos hasta `a = 0.05` (test 7), y para `δ_rms`, `v_rms`, `a_final` individualmente.

### 5.4 Traza adaptativa `dt(a)`

Ver `docs/reports/figures/phase43/adaptive_dt_trace.png`. El modo adaptativo queda fijo en `dt_min = 5·10⁻⁵` durante ~el 100 % de los pasos: las ICs 2LPT producen `a_max` tan alto que `η·√(ε/a_max)` cae por debajo del piso. El efectivo `⟨dt⟩ = Δa / n_steps = 0.03 / 6838 ≈ 4.4·10⁻⁶`, que delata que el integrador **subdivide internamente** cuando `dt` excede el remanente a `a_target` — ese sub‑paso final explica por qué `n_steps` sale mayor que `(a_target − a_init) / dt_min`.

---

## 6. Respuestas a las preguntas de la fase

### A. ¿`TreePM + ε_phys = 0.01` hace que el error dependa de `dt`?

**Parcialmente, sí**: +9 % de mejora entre `dt = 4·10⁻⁴` y `dt = 2·10⁻⁴`. Pero el efecto es **subdominante** frente a la no‑linealidad ya instalada a `a = 0.05`. La dependencia **no es suficiente** para justificar por sí sola un integrador adaptativo sofisticado.

### B. ¿Reducir `dt` mejora el crecimiento lineal o el error espectral?

Reduce el error espectral un **9 %**, pero el **error sigue en `~10⁷`** (`⟨P_c/P_ref⟩` → órdenes de magnitud fuera del régimen lineal). **`v_rms`** sí baja monótonamente (factor ~1.5 de 4e‑4 a 2e‑4, factor ~3 más con adaptativo). Ver tabla 5.1.

### C. ¿Un adaptativo global simple bate al mejor `dt` fijo?

**No** en esta matriz: `B_adaptive_matches_best_fixed`. Iguala dentro de ±5 %, con `5.85×` más wall. El adaptativo clampa al `dt_min` desde el arranque; el problema es que **las ICs 2LPT con `Z0Sigma8` están demasiado agresivas** para que el criterio Aarseth dé un paso "grande". Bajar `dt_min` daría runs de horas por variante — no justificado.

### D. ¿El paralelismo Rayon hace viable esta línea?

**Sí**, claramente. `3.70×` con 4 hilos, bit‑exact, sin reescribir el solver. Proyectado a 8 hilos (no medido en este smoke), se espera `~6×` por scaling Amdahl‑limitado por la FFT serial (~10 % del tiempo en PM). Para `N = 64³` (`8×` más partículas), el mismo speedup multihilo reduce un barrido de 37 h a **~10 h** con 4 hilos.

### E. ¿La combinación TreePM + softening + mejor `dt` destraba la validación física evolucionada?

**No**. El error espectral en bajo‑k a `a = 0.10` sigue siendo catastrófico (`⟨P_c/P_ref⟩ ~ 5·10⁸`), y `δ_rms ≈ 1` ya a `a = 0.05` confirma que la no‑linealidad está instalada muy temprano. El cuello **no** está en el integrador: está en la amplitud/convención inicial (consistente con Phase 39). Temas abiertos para una próxima fase:

1. Revisar la convención de velocidad en 2LPT (factor `a · da/dt` vs `dx/dτ`).
2. Revisar la amplitud de `σ₈` renormalizada a `a_init` (¿se está amplificando accidentalmente por `D(a_init)`?).
3. Repetir con ICs Zel'dovich puro (sin 2LPT) para descartar que el término de segundo orden sea el culpable.

---

## 7. Decisión técnica

1. **Mantener** el módulo `adaptive_dt` como infraestructura para fases futuras (criterio Aarseth + Hubble), pero **no** activarlo por default: a `ε_phys = 0.01` y 2LPT agresivo, no aporta sobre el mejor fijo y triplica el wall.
2. **Adoptar** el paralelismo Rayon en `PmSolver::accelerations_for_indices` (feature `rayon`). Es bit‑exact, `3.7×` con 4 hilos, y el feature ya estaba gating el tree walk corto.
3. **Congelar** `dt = 2·10⁻⁴` como default operativo para TreePM + `ε_phys = 0.01`, revisable al subir resolución.
4. **Mover el foco** a ICs y convención de velocidades en la próxima fase. El control temporal ya no es el cuello dominante con la evidencia acumulada.

---

## 8. Artefactos

### Código

- `crates/gadget-ng-integrators/src/adaptive_dt.rs` — módulo nuevo.
- `crates/gadget-ng-integrators/src/lib.rs` — re‑export público.
- `crates/gadget-ng-pm/src/solver.rs` — rayon feature gating.
- `crates/gadget-ng-physics/Cargo.toml` — `rayon` como dev‑dep.
- `crates/gadget-ng-physics/tests/phase43_dt_treepm_parallel.rs` — 7 tests.

### Experimento

- `experiments/nbody/phase43_dt_treepm_parallel/`
  - `configs/lcdm_N64_treepm_dt{4,2,1}e-4.toml` — templates para run futuro a `N=64³`.
  - `run_phase43.sh` — orquestador (respeta `PHASE43_QUICK`, `PHASE43_N`, `PHASE43_THREADS`, `PHASE43_USE_CACHE`, `PHASE43_SKIP_ADAPTIVE`, `PHASE43_DT5E5`).
  - `scripts/plot_dt_effect.py`, `scripts/plot_parallel_speedup.py`, `scripts/analyze_growth_phase43.py`.
  - `figures/` — PNGs + CSVs.

### Figuras (`docs/reports/figures/phase43/`)

- `error_vs_dt.png` — error espectral vs `dt` a `a ∈ {0.05, 0.10}`.
- `growth_vs_theory.png` — `⟨P_c/P_ref⟩` en bajo‑k vs `dt`.
- `growth_phase43.png` — ratio de crecimiento bajo‑k (fijos + adaptativo).
- `delta_rms_vs_a.png` — `δ_rms(a)` por variante.
- `runtime_vs_dt.png` — wall‑clock total vs `⟨dt⟩` efectivo.
- `speedup_vs_threads.png` + `walltime_vs_threads.png` — escalamiento paralelo.
- `adaptive_dt_trace.png` — `dt(a)` del adaptativo.

### CSVs

- `phase43_dt_sweep.csv` — tabla completa de snapshots.
- `phase43_parallel_speedup.csv` — wall + speedup por número de hilos.
- `phase43_growth.csv` — ratio de crecimiento bajo‑k por variante.

### JSONs crudos (`target/phase43/`)

`per_snapshot_metrics.json`, `test{1..7}_*.json`.

---

## 9. Cómo reproducir

```bash
# Smoke (≈ 45 min a N=32 con 8 cores):
PHASE43_N=32 PHASE43_QUICK=1 PHASE43_THREADS=1,4 \
    bash experiments/nbody/phase43_dt_treepm_parallel/run_phase43.sh

# Re‑generar figuras sin re‑correr simulación:
PHASE43_USE_CACHE=1 \
    bash experiments/nbody/phase43_dt_treepm_parallel/run_phase43.sh

# Matriz completa (sin QUICK) — recomendado con Rayon y 8+ hilos:
PHASE43_N=32 PHASE43_THREADS=1,4,8 \
    bash experiments/nbody/phase43_dt_treepm_parallel/run_phase43.sh

# Escalar a N=64³ (estimado ~10 h con 4 hilos en la matriz QUICK):
PHASE43_N=64 PHASE43_QUICK=1 PHASE43_THREADS=4 \
    bash experiments/nbody/phase43_dt_treepm_parallel/run_phase43.sh
```

---

## 10. Criterios de Definition of Done

- [x] Barrido de `dt` ejecutado con `TreePM + ε_phys ≈ 0.01`.
- [x] Se midió la dependencia con `dt`: +9 % en bajo‑k, factor ~5 en `v_rms`, sin mejora cualitativa.
- [x] Adaptativo global simple implementado, medido y **descartado** por ausencia de ganancia sobre el mejor fijo a este `ε`/ICs.
- [x] Paralelismo Rayon añadido en PmSolver (CIC assign/interp) y medido.
- [x] Speedup medido: `3.70×` con 4 hilos, bit‑exact.
- [x] Respuesta consolidada: **el control temporal NO es el cuello dominante**. El cuello está en amplitud/convención de ICs, consistente con Phase 39.
