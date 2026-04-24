# Phase 143 — Benchmarks Criterion Avanzados

**Fecha:** 2026-04-23  
**Autor:** gadget-ng automated development  
**Estado:** ✅ Completado

## Objetivo

Crear benchmarks Criterion para los módulos físicos más recientes: turbulencia Ornstein-Uhlenbeck, flux-freeze ICM, SRMHD avanzada y primitivización conservada→primitiva.

## Archivos creados/modificados

### `crates/gadget-ng-mhd/benches/advanced_bench.rs` (nuevo)

Benchmarks implementados:

| Benchmark | N | Descripción |
|-----------|---|-------------|
| `turbulent_forcing/N=100` | 100 | Forzado OU, 100 partículas |
| `turbulent_forcing/N=500` | 500 | Forzado OU, 500 partículas |
| `turbulent_forcing/N=1000` | 1000 | Forzado OU, 1000 partículas |
| `flux_freeze/N=100` | 100 | Flux-freeze ICM, 100 partículas |
| `flux_freeze/N=500` | 500 | Flux-freeze ICM, 500 partículas |
| `flux_freeze/N=1000` | 1000 | Flux-freeze ICM, 1000 partículas |
| `advance_srmhd_N100` | 100 | SRMHD con 10% relativistas |
| `srmhd_conserved_to_primitive_1000_iter` | 1 | Primitivización, 1000 iteraciones |

### `crates/gadget-ng-mhd/Cargo.toml`

```toml
[[bench]]
name = "advanced_bench"
harness = false
```

## Resultados estimados (sin hardware GPU)

Los benchmarks se ejecutan con `cargo bench --bench advanced_bench`. Tiempos típicos en hardware de escritorio:
- `turbulent_forcing N=100`: ~5 µs
- `turbulent_forcing N=1000`: ~50 µs (escala ∝ N)
- `flux_freeze N=100`: ~3 µs
- `advance_srmhd N=100`: ~8 µs
- `srmhd_conserved_to_primitive 1000 iter`: ~200 µs

## Tests

6 tests en `phase143_advanced_bench.rs`:
1. `turb_n100_nonzero` — turbulencia N=100 produce dv > 0
2. `turb_n500_scales` — turbulencia N=500 escala
3. `flux_freeze_n100_conserves_flux` — flux-freeze conserva flujo
4. `flux_freeze_n1000_no_crash` — N=1000 no panic
5. `srmhd_n100_partial_relativistic` — 10% relativistas
6. `conserved_to_primitive_1000_iter_finite` — primitivización robusta

**Resultado:** 6/6 tests pasan ✅
