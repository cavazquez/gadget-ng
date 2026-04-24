# Phase 132 — Benchmark MHD Criterion + CFL Unificado

**Fecha:** 2026-04-23  
**Estado:** ✅ COMPLETADA  
**Tests:** 6/6 pasados

## Objetivo

Agregar benchmarks Criterion para el stack MHD completo y formalizar el CFL unificado (ya implementado en Phase 127) como función documentada del motor.

## Benchmarks (`gadget-ng-mhd/benches/alfven_bench.rs`)

Cuatro grupos medidos sobre N = 100, 500, 1000 partículas:

| Benchmark | Función |
|---|---|
| `advance_induction/N` | Ecuación de inducción SPH (Phase 123) |
| `apply_magnetic_forces/N` | Tensión + presión magnética (Phase 124) |
| `dedner_cleaning/N` | Limpieza de div-B (Phase 125) |
| `full_mhd_step/N` | Inducción + fuerzas + cleaning (pipeline completo) |

### Ejecución

```bash
cargo bench -p gadget-ng-mhd
# Salida en target/criterion/mhd_stack/*/report/index.html
```

## CFL magnético unificado

El CFL de Alfvén (Phase 127) se consolida en `maybe_mhd!`:

```rust
macro_rules! maybe_mhd {
    () => {
        if cfg.mhd.enabled {
            let dt_alfven = gadget_ng_mhd::alfven_dt(&local, cfg.mhd.cfl_mhd);
            let dt_mhd = cfg.simulation.dt.min(dt_alfven);
            gadget_ng_mhd::advance_induction(&mut local, dt_mhd);
            gadget_ng_mhd::apply_magnetic_forces(&mut local, dt_mhd);
            gadget_ng_mhd::dedner_cleaning_step(&mut local, cfg.mhd.c_h, cfg.mhd.c_r, dt_mhd);
        }
    };
}
```

**Propiedad**: `dt_mhd ≤ dt_global` siempre. Con B=0, `dt_alfven = ∞` y `dt_mhd = dt_global` (sin overhead).

## Comportamiento del CFL

| Campo B | v_A | dt_alfven | dt_mhd |
|---|---|---|---|
| B = 0 | 0 | ∞ | dt_global |
| B débil (10⁻⁵) | ~ 10⁻⁵ | >> dt_global | dt_global |
| B típico (1) | ~ 1 | ~ h_min × cfl | min(...) |
| B fuerte (100) | ~ 100 | << dt_global | dt_alfven |

## Tests

| Test | Descripción |
|------|-------------|
| `alfven_dt_restricts_when_b_is_strong` | B fuerte → CFL activo |
| `alfven_dt_no_restrict_with_weak_b` | B débil → no restringe |
| `cfl_unified_is_minimum` | Siempre = mín(dt_g, dt_a) |
| `alfven_dt_scales_with_b` | B×2 → dt_A/2 |
| `full_mhd_step_with_cfl_produces_finite_result` | 10 pasos con CFL activo → finito |
| `mhd_bench_file_exists` | `alfven_bench.rs` existe en `benches/` |
