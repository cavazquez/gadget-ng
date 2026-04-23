# Phase 76 — Assembly Bias: Spin/c vs Entorno

**Crate**: `gadget-ng-analysis/src/assembly_bias.rs`  
**Fecha**: 2026-04

## Resumen

Implementación del análisis de assembly bias: correlación entre propiedades
internas de los halos (spin λ, concentración c) y la sobredensidad del entorno
de gran escala δ_env medida con un filtro top-hat esférico.

## Algoritmo

1. **Campo de densidad CIC** de todas las partículas en un grid de resolución `mesh`.
2. **Suavizado en k-space** con filtro top-hat esférico W(kR) de radio `R_smooth`.
3. **Interpolación trilineal** de δ_env en las posiciones de los halos.
4. **Correlación de Spearman** entre (λ, δ_env) y entre (c, δ_env).
5. **Sesgo por cuartiles**: dividir halos por cuartiles de λ o c, calcular
   `b_q = ⟨(1+δ_env)⟩_q / ⟨(1+δ_env)⟩_all − 1`.

## Física

El assembly bias (Gao, Springel & White 2005) predice que halos de la misma
masa pero formados antes (mayor concentración, menor spin) tienden a estar
en regiones más densas del campo de materia. Este efecto tiene implicaciones
directas para el bias de las galaxias.

## Filtro top-hat

```
W(x) = 3 [sin(x) - x·cos(x)] / x³   con x = k × R_smooth
```

## Nuevos tipos

```rust
pub struct AssemblyBiasResult {
    spearman_lambda:        f64,
    spearman_concentration: f64,
    bias_vs_lambda:         Vec<(f64, f64)>,
    bias_vs_concentration:  Vec<(f64, f64)>,
    n_halos: usize,
}
pub struct AssemblyBiasParams { smooth_radius, mesh, n_quartiles }
```

## Tests

9 tests en `assembly_bias::tests`:
- `assembly_bias_empty_halos`
- `assembly_bias_returns_finite`
- `assembly_bias_quartiles_count`
- `spearman_perfect_monotone` (ρ=1 para monotonía perfecta)
- `spearman_anti_monotone` (ρ=-1 para monotonía inversa)
- `spearman_independent_near_zero`
- `tophat_window_k0` (W(0)=1)
- `tophat_window_decays`
- `result_serializes`
