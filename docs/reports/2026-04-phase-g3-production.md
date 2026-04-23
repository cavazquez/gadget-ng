# Phase G3 — Corrida de producción N=256³ end-to-end

**Fecha:** abril 2026  
**Estimación original:** 2–3 sesiones  
**Dependencias:** Todas las fases de simulación (1–65 + G2) completadas; CUDA/HIP (Phase 57) opcional  
**Crates afectados:** `gadget-ng-cli`, `gadget-ng-core`, `gadget-ng-physics`  
**Archivos clave:**
- `configs/production_256.toml`
- `configs/production_256_test.toml`
- `scripts/run_production_256.sh`
- `docs/notebooks/postprocess_pk.py`
- `docs/notebooks/postprocess_hmf.py`
- `crates/gadget-ng-physics/tests/phase_g3_production.rs`

---

## Objetivo

Primera corrida de producción real a resolución N=256³ (16.7 millones de partículas),
desde z=49 hasta z=0, con múltiples snapshots, análisis in-situ y post-procesamiento.

Esta fase no implementa nuevo código de simulación: es la **infraestructura** que
orquesta todas las capacidades acumuladas en las fases previas en un pipeline de
producción completo, reproducible y con checkpointing.

---

## Configuración de producción (`configs/production_256.toml`)

### Parámetros físicos

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| N | 256³ = 16 777 216 | Resolución media; factible en 4 CPUs o 1 GPU |
| L | 300 Mpc/h | Representativa estadísticamente; M_min ≈ 1.4×10¹¹ M☉/h |
| z_init | 49 (a=0.02) | Estándar; condiciones lineales bien satisfechas |
| z_final | 0 | Snapshot a tiempo presente |
| ε | 0.027 Mpc/h | L/(30·N^{1/3}), constante en unidades físicas (GADGET-4) |
| Ω_m | 0.315 | Planck 2018 |
| Ω_Λ | 0.685 | Planck 2018 |
| σ₈ | 0.811 | Planck 2018, referido a z=0 |
| n_s | 0.965 | Planck 2018 |
| h | 0.674 | Planck 2018 |

### Solver y rendimiento

| Componente | Configuración |
|------------|---------------|
| Solver | TreePM (SFC+LET+scatter-gather) |
| PM grid | 512³ (factor 2× oversampling) |
| Timesteps | Jerárquicos (η=0.025, 6 niveles, criterio jerk) |
| SFC | Hilbert con rebalanceo adaptivo (umbral 30%) |
| ICs | 2LPT + Eisenstein-Hu, modo `z0_sigma8` |

### Salida

| Tipo | Intervalo | Formato |
|------|-----------|---------|
| Checkpoint | Cada 100 pasos | Binario (resume automático) |
| Snapshot | Cada 50 pasos (~20 snaps) | HDF5 |
| In-situ P(k) | Cada 50 pasos | JSON (mesh 256³) |
| In-situ FoF | Cada 50 pasos | JSON (b=0.2, min 20 part) |
| In-situ ξ(r) | Cada 50 pasos | JSON (30 bins) |

---

## Script de producción (`scripts/run_production_256.sh`)

### Características

- **Detección de checkpoint**: si existe `*.checkpoint` en el directorio de salida,
  reanuda automáticamente desde el último.
- **Soporte MPI**: `N_RANKS=4 bash scripts/run_production_256.sh` usa `mpirun`.
- **Post-proceso opcional**: `POSTPROCESS=1` invoca los notebooks Python tras completar.
- **Logging**: timestamp completo a `runs/production_256/run.log`.
- **Estimación de tiempo**: imprime el wall-time estimado según el hardware disponible.

### Variables de entorno

```bash
N_RANKS=4          # Número de ranks MPI (default: 1)
CONFIG=mi.toml     # Archivo de configuración alternativo
OUT_DIR=runs/prod  # Directorio de salida
POSTPROCESS=1      # Ejecutar post-proceso Python
SKIP_BUILD=1       # Usar binario existente sin recompilar
```

### Uso típico

```bash
# Serial
bash scripts/run_production_256.sh

# MPI × 4
N_RANKS=4 bash scripts/run_production_256.sh

# Con GPU CUDA
# (editar production_256.toml: performance.use_gpu_cuda = true)
bash scripts/run_production_256.sh

# Con post-proceso y figuras
POSTPROCESS=1 bash scripts/run_production_256.sh
```

---

## Post-proceso Python

### `postprocess_pk.py`

Lee los archivos `insitu_*.json` y genera:
- `pk_evolution.json`: P(k,z) de todos los snapshots.
- `pk_evolution.png`: gráfico log-log P(k) a múltiples redshifts.

```bash
python3 docs/notebooks/postprocess_pk.py \
  --insitu runs/production_256/insitu \
  --out    runs/production_256/analysis/pk_evolution.json
```

### `postprocess_hmf.py`

Lee los archivos `insitu_*.json` y genera:
- `hmf_evolution.json`: n(M,z) para todos los snapshots.
- `hmf_evolution.png`: comparación N-body vs Sheth-Tormen analítico.

```bash
python3 docs/notebooks/postprocess_hmf.py \
  --insitu runs/production_256/insitu \
  --out    runs/production_256/analysis/hmf_evolution.json
```

Ambos scripts son **sin dependencias duras**: funcionan sin numpy ni matplotlib
(salvo que no generan figuras) y sin errores críticos si falta algún snapshot.

---

## Tests (6 / 6 OK)

| Test | Descripción |
|------|-------------|
| `g3_config_parses_ok` | `production_256.toml` se parsea sin error |
| `g3_test_config_parses_ok` | `production_256_test.toml` se parsea sin error |
| `g3_ic_generation_32cube` | ICs 2LPT E-H N=32³ sin NaN/Inf, dentro de caja |
| `g3_production_config_valid_params` | ε razonable, a_init correcto, Ω plano, G>0 |
| `g3_ic_mass_consistent` | masa uniformemente distribuida, finita y positiva |
| `g3_ic_sigma8_reasonable` | v_rms > 0, finito, razonable (ICs perturbadas) |

---

## Estimaciones de tiempo de corrida

| Hardware | Wall time estimado |
|----------|--------------------|
| 1 CPU (serial) | 8–12 horas |
| 4 CPU MPI | 3–4 horas |
| 1 GPU NVIDIA (CUDA, Phase 57) | 2–4 horas |
| 4 CPU + GPU | 1.5–2 horas |

---

## Métricas de éxito (validación post-corrida)

| Métrica | Objetivo |
|---------|---------|
| P(k) vs CAMB lineal | `|log₁₀(P_sim/P_lin)| < 0.05` en k < 0.1 h/Mpc |
| HMF vs Sheth-Tormen | factor < 2 para M > 50 m_part ≈ 1.4×10¹¹ M☉/h |
| ξ(r) vs linear | consistente en r > 5 Mpc/h |
| Conservación de masa | suma de masas constante en todos los snapshots |
| Sin NaN/Inf | verificado por el loop de stepping |

---

## Limitaciones conocidas

1. **Wall time**: N=256³ requiere hardware dedicado. CI usa N=32³ como smoke test.
2. **Análisis post-proceso**: los notebooks Python requieren numpy/matplotlib para
   generar figuras (no es un error crítico si no están instalados).
3. **Checkpoint**: el formato de checkpoint en Bincode puede cambiar entre versiones
   si se agregan nuevos campos a `Particle` o `RunConfig`; no retrocompatible cross-version.
4. **PM con 512³**: requiere ~4 GB de RAM para el grid FFT completo.

---

## Configuración de CI reducida (`production_256_test.toml`)

Para que los tests de smoke corran en CI en menos de 60 s, se usa N=32³ con:
- `box_size = 50.0 Mpc/h`
- `num_steps = 10`
- `pm_grid_size = 64`
- `snapshot_format = "jsonl"` (más rápido que HDF5 para N pequeño)
- `checkpoint_interval = 0` (sin checkpoint en tests)
