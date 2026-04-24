# Phase 104 — Análisis post-proceso CLI extendido

**Fecha:** 2026-04-23  
**Estado:** ✅ Completado

## Objetivo

Extender el subcomando `gadget-ng analyze` con 4 nuevos flags de análisis que cubren los módulos implementados en Phases 94-99:

| Flag | Módulo | Output |
|------|--------|--------|
| `--cm21` | 21cm FFT (Phase 97) | `analyze/cm21_output.json` |
| `--igm-temp` | Temperatura IGM (Phase 90) | `analyze/igm_temp.json` |
| `--agn-stats` | Feedback AGN (Phase 96/99) | `analyze/agn_stats.json` |
| `--eor-state` | Reionización (Phase 95) | `analyze/eor_state.json` |

## Cambios implementados

### `crates/gadget-ng-cli/src/analyze_cmd.rs`
- `AnalyzeParams` extendido con `cm21`, `igm_temp`, `agn_stats`, `eor_state: bool`.
- Defaults: todos `false`.
- Los outputs van a `<out_dir>/analyze/` (no se crea el directorio si ningún flag está activo).

**Módulos:**
- `--cm21`: `gadget_ng_rt::compute_cm21_output()` sobre partículas de gas → `cm21_output.json`.
- `--igm-temp`: `gadget_ng_rt::compute_igm_temp_profile()` → `igm_temp.json`.
- `--agn-stats`: clasifica partículas calientes (u > 10⁴) como candidatos BH → `agn_stats.json`.
- `--eor-state`: estima `x_HII` por fracción de partículas calientes (u > 100) → `eor_state.json`.

### `crates/gadget-ng-cli/src/main.rs`
- `Commands::Analyze` extendido con los 4 nuevos `#[arg(long)]` flags.
- Handler actualizado para pasarlos a `AnalyzeParams`.

## Uso

```bash
gadget-ng analyze \
  --snapshot out/snapshot_final \
  --output results/analyze.json \
  --cm21 \
  --igm-temp \
  --agn-stats \
  --eor-state

# Outputs generados:
# results/analyze.json         (resultados principales: FoF, P(k), ξ(r))
# results/analyze/cm21_output.json
# results/analyze/igm_temp.json
# results/analyze/agn_stats.json
# results/analyze/eor_state.json
```

## Tests (internos en `analyze_cmd.rs`)

| Test | Descripción | Estado |
|------|-------------|--------|
| `analyze_params_phase104_defaults_false` | defaults correctos | ✅ |
| `analyze_params_phase104_can_be_enabled` | flags activables | ✅ |
| `analyze_no_flags_no_analyze_dir` | sin flags no crea dir | ✅ |
| `analyze_agn_stats_no_gas_creates_empty_report` | 0 candidatos | ✅ |
| `analyze_eor_state_no_gas_x_hii_zero` | x_HII = 0 sin gas | ✅ |
| `analyze_eor_and_agn_flags_together` | múltiples flags | ✅ |

**Total: 6/6 tests pasan**
