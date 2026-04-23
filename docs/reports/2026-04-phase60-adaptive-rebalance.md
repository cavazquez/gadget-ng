# Phase 60 — Domain Decomposition Adaptativa (rebalanceo basado en costo)

**Fecha:** abril 2026  
**Crates:** `gadget-ng-core` (config), `gadget-ng-cli` (engine)  
**Archivo de test:** `crates/gadget-ng-physics/tests/phase60_adaptive_rebalance.rs`

---

## Contexto

Antes de Phase 60, el rebalanceo SFC solo se ejecutaba cada `sfc_rebalance_interval`
pasos (criterio fijo). El path SFC+LET BarnesHut ya tenía un criterio por costo
(`cost_rebalance_pending`) con threshold hardcodeado a 1.3, pero:

1. El threshold no era configurable.
2. Los paths jerárquico LET y cosmológico SFC no tenían criterio por costo.

---

## Cambios

### `rebalance_imbalance_threshold` en `PerformanceSection`

```toml
[performance]
# Umbral de desbalance de carga para rebalanceo inmediato.
# Si max(walk_ns)/min(walk_ns) > threshold → rebalanceo en el próximo paso.
# 0.0 = desactivado (solo por intervalo). Valores típicos: 1.3, 1.5, 2.0.
rebalance_imbalance_threshold = 1.3
```

### Helper `should_rebalance`

```rust
fn should_rebalance(step: u64, start_step: u64, interval: u64, cost_pending: bool) -> bool {
    if cost_pending { return true; }
    if interval == 0 { return true; }
    (step - start_step) % interval == 0
}
```

Centraliza la lógica de decisión; evita duplicación en los 4+ paths del motor.

### Paths actualizados

| Path | Antes | Después |
|------|-------|---------|
| SFC+LET BarnesHut | `cost_pending` con threshold 1.3 hardcoded | `should_rebalance` + threshold configurable |
| Jerárquico+LET | Solo por intervalo | + detección `this_grav max/min > threshold` |
| Cosmológico SFC+LET | Solo por intervalo | + detección `this_grav max/min > threshold` |

---

## Comportamiento del rebalanceo adaptativo

Ejemplo con `interval=20`, desbalance detectado en paso 5:

```
step=1  → rebalancear (inicio)
step=2..5 → no rebalancear
step=5  → detectar desbalance: cost_pending = true
step=6  → rebalancear (por costo, 15 pasos antes del intervalo)
step=7..20 → no rebalancear
step=21 → rebalancear (por intervalo)
```

---

## Tests

| Test | Descripción |
|------|-------------|
| `phase60_should_rebalance_interval` | Rebalanceo exactamente en múltiplos del intervalo |
| `phase60_should_rebalance_cost_override` | `cost_pending=true` fuerza rebalanceo anticipado |
| `phase60_should_rebalance_zero_interval` | `interval=0` rebalancea siempre |
| `phase60_threshold_config` | `rebalance_imbalance_threshold` default=0.0, configurable |
| `phase60_cost_triggers_early_rebalance` | Scenario completo: pasos de rebalanceo [1, 6, 21] |

Todos los tests pasan en modo serial (sin MPI).
