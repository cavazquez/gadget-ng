# Phase 79 — Validación Producción N=128³

**Tipo**: Infraestructura + scripts + tests  
**Fecha**: 2026-04

## Resumen

Infraestructura completa para una corrida de validación cosmológica N=128³
(ΛCDM Planck18, z_ini=49 hasta z=0) con comparación automática contra CLASS
y Tinker08.

## Archivos creados

### Configuraciones

- **`configs/validation_128.toml`**: configuración completa N=128³, 512 pasos,
  2LPT Eisenstein-Hu, TreePM + block timesteps, análisis in-situ con P(k,μ).
- **`configs/validation_128_test.toml`**: versión reducida N=32³ para CI (20 pasos).

### Scripts

- **`scripts/run_validation_128.sh`**: script de corrida con soporte para:
  - `--resume` desde checkpoint
  - `--mpi N` para corrida MPI
  - `--post` para ejecutar post-proceso Python automáticamente

### Notebook Python

- **`docs/notebooks/validate_pk_hmf.py`**: compara P(k,z=0) vs Eisenstein-Hu,
  mide σ₈ desde el espectro, calcula HMF, genera JSON de comparación y gráficos PNG.

## Parámetros físicos (validation_128.toml)

| Parámetro | Valor |
|---|---|
| N_particles | 128³ = 2,097,152 |
| box_size | 200 Mpc/h |
| Ω_m | 0.3153 (Planck 2018) |
| Ω_Λ | 0.6847 |
| h | 0.6736 |
| σ₈ | 0.8111 |
| z_init | ≈49 (a=0.02) |
| Pasos | 512 |

## Tests Rust (phase79_validation.rs)

6 tests de smoke en `gadget-ng-physics`:
1. `p79_config_parses_ok` — validación_128.toml parsea correctamente (128³, RT activo, etc.)
2. `p79_test_config_parses_ok` — validation_128_test.toml parsea correctamente
3. `p79_config_params_valid` — Ω_m + Ω_Λ ≈ 1, softening < separación inter-partícula
4. `p79_ic_32cube_no_nan` — ICs 2LPT sin NaN/Inf
5. `p79_sigma8_within_range` — σ₈ estimado ≥ 0
6. `p79_cosmo_params_consistent` — auto_g activo, h₀ ∈ (0,1), z_init ∈ (20,200)

## Uso

```bash
# Smoke test en CI (N=32³, 20 pasos, ~30 segundos):
cargo test -p gadget-ng-physics --test phase79_validation

# Corrida de producción completa:
./scripts/run_validation_128.sh --post

# Con MPI (8 cores):
./scripts/run_validation_128.sh --mpi 8 --post

# Reanudar desde checkpoint:
./scripts/run_validation_128.sh --resume --mpi 8 --post
```
