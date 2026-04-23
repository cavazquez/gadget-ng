# Phase 80 — AMR 3 Niveles Jerárquico Recursivo

**Crate**: `gadget-ng-pm/src/amr.rs` (extensión de Phase 70)  
**Fecha**: 2026-04

## Resumen

Extensión del solver AMR-PM de 2 niveles (Phase 70) a N niveles recursivos,
mediante una jerarquía de `AmrLevel` con sub-niveles anidados. El umbral de
refinamiento escala como `δ_refine × factor^l` en cada nivel.

## Nuevos campos en `AmrParams`

```rust
pub max_levels: usize,    // 1=Phase70, 2=dos niveles, 3=tres niveles (default Phase80)
pub refine_factor: f64,   // factor de escalado del umbral entre niveles (default: 4.0)
```

## Nueva struct `AmrLevel`

```rust
pub struct AmrLevel {
    pub patches: Vec<PatchGrid>,     // parches en este nivel
    pub child_levels: Vec<AmrLevel>, // sub-niveles recursivos
    pub depth: usize,                // profundidad (0=base)
}
```

## Algoritmo multi-nivel

1. **Nivel base (0)**: PM estándar en `nm_base³`. Fuerzas de largo alcance.
2. **Nivel 1**: identificar celdas con `δ > δ_refine`, construir parches,
   resolver Poisson, aplicar corrección `ΔF = F_patch - F_base`.
3. **Nivel 2** (si `max_levels ≥ 2`): para cada parche L1, filtrar partículas
   internas, resolver sub-Poisson con `δ > δ_refine × factor`, aplicar
   corrección iterativa.
4. **Corrección ponderada**: peso de transición `w = (1 - 2|frac - 0.5|)³`
   para suavizar la transición en los bordes de cada parche.

## Complejidad

| Nivel | Costo |
|---|---|
| Base | O(N³ log N) |
| Nivel l | O(n_patch × nm_p³ log nm_p) por parche |
| Total | O(N³ log N) + O(N_refined × nm_p³ log nm_p) |

Donde `N_refined ≪ N` para distribuciones cósmicas típicas.

## Nuevas funciones exportadas

| Función | Descripción |
|---|---|
| `amr_pm_accels_multilevel(...)` | Solver AMR N-nivel |
| `amr_pm_accels_multilevel_with_stats(...)` | Con estadísticas de refinamiento |
| `build_amr_hierarchy(...)` | Construye el árbol AMrLevel recursivo |
| `identify_refinement_patches(...)` | Ahora también exportada públicamente |

## Tests Phase 80

4 tests nuevos (total 11 en `amr::tests`):
- `amr_multilevel_max1_equals_base` — max_levels=1 produce resultados finitos
- `amr_multilevel_3levels_no_nan` — 3 niveles sin NaN/Inf
- `amr_multilevel_stats` — estadísticas correctas
- `amr_level_struct_builds` — struct AmrLevel se construye sin panic
