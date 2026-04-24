# Phase 103 — Domain decomposition con coste medido

**Fecha:** 2026-04-23  
**Estado:** ✅ Completado (infraestructura ya presente, tests de validación agregados)

## Estado previo

El sistema de domain decomposition con coste medido ya estaba **completamente implementado** desde una fase anterior:

- `compute_forces_local_tree_with_costs()`: usa `walk_stats_begin()` / `walk_stats_end()` de `gadget-ng-tree` para medir `opened_nodes` por partícula.
- `SfcDecomposition::build_weighted()`: partición SFC ponderada por coste acumulado.
- EMA: `cost_ema[i] = α·nodes_i + (1-α)·cost_ema_prev[i]`.
- `cfg.decomposition.cost_weighted = true/false` para activar.
- `cfg.decomposition.ema_alpha` (default 0.3) para la tasa de actualización.

## Flujo completo

```
octree walk → WalkStats.opened_nodes/partícula
                        ↓
           cost_ema[i] = α·nodes_i + (1-α)·cost_ema[i]
                        ↓
           SfcDecomposition::build_weighted(pos, cost_ema, ...)
                        ↓
           partición balanceada por coste ∑w_i ≈ W_total/N_ranks
```

## Configuración

```toml
[decomposition]
cost_weighted = true
ema_alpha     = 0.3   # 0.1 = memoria larga, 0.9 = respuesta rápida
```

## Tests (`crates/gadget-ng-physics/tests/phase103_sfc_weighted.rs`)

| Test | Descripción | Estado |
|------|-------------|--------|
| `build_weighted_respects_weight_sum` | todos los puntos asignados | ✅ |
| `build_weighted_vs_uniform_differ_for_skewed_weights` | balance de costes | ✅ |
| `ema_converges_after_iterations` | EMA estabiliza en 20 pasos | ✅ |
| `config_cost_weighted_parses` | TOML parseable | ✅ |
| `new_particles_get_uniform_cost_after_resize` | migradas → coste 1.0 | ✅ |

**Total: 5/5 tests pasan**
