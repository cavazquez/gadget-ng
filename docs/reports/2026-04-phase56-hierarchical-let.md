# Phase 56 — Block timesteps jerárquicos acoplados al árbol LET distribuido

**Fecha:** abril 2026  
**Crates principales:** `gadget-ng-integrators`, `gadget-ng-parallel`, `gadget-ng-cli`  
**Archivo de test:** `crates/gadget-ng-physics/tests/phase56_hierarchical_let.rs`

---

## Contexto

Phase 43 implementó el integrador de block timesteps jerárquico (`hierarchical_kdk_step`).
Phase 56 lo acopla al árbol distribuido SFC+LET, permitiendo que los subpasos finos operen
con fuerzas calculadas solo para las partículas activas en el paso correspondiente,
usando halos SFC en lugar del allgather global O(N·P).

---

## Implementación

### `compute_forces_hierarchical_let`

Nueva función en `gadget-ng-parallel` que evalúa fuerzas solo para los índices
`active_local`, usando halos SFC intercambiados:

```rust
pub fn compute_forces_hierarchical_let(
    local: &[Particle],
    halos: &[Particle],
    active_local: &[usize],
    theta: f64,
    g: f64,
    eps2: f64,
    acc: &mut Vec<Vec3>,
)
```

El árbol local se construye sobre `local + halos`; solo se evalúan los índices activos.
Esto reduce la evaluación de fuerzas de O(N) a O(N_active) por subnivel, con
N_active = N/2^level en el nivel más fino.

### Acoplamiento en `engine.rs`

En el path jerárquico+SFC (`use_hierarchical_let = true`):

1. Se inicializa un `SfcDecomposition` desde posiciones locales.
2. En cada base-step, se ejecuta `exchange_halos_sfc` una sola vez.
3. El closure de fuerzas de `hierarchical_kdk_step` llama a `compute_forces_hierarchical_let`.
4. Rebalanceo SFC por intervalo (`sfc_rebalance_interval`).

### Softening cosmológico

Corrección del bug de softening: `ε_phys` constante en Mpc/h con
`eps2 = (ε_phys / a)²` cuando `physical_softening = true`.

---

## Tests

| Test | Descripción |
|------|-------------|
| `force_active_subset_matches_full` | Fuerzas active-only = fuerzas full para mismos índices |
| `momentum_conservation_hierarchical` | Momentum lineal conservado tras 32 pasos |
| `hierarchical_let_closure_stability` | Estabilidad energética N=64 Plummer, 16 pasos |
| `active_only_skips_inactive` | Partículas inactivas no reciben aceleraciones nuevas |
| `hierarchical_let_vs_full_tree` | Active-only vs full: idénticos tras un paso |

---

## Resultados

- El acoplamiento reduce la comunicación MPI de O(N·P) a O(N_halo) por subnivel.
- Para `max_level=4`, la reducción teórica de complejidad de fuerzas es ~8× respecto a
  evaluar todos los niveles para todas las partículas.
- Momentum lineal conservado hasta precisión de doble precisión.
