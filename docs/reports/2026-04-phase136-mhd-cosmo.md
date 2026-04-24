# Phase 136 — MHD Cosmológico End-to-End

**Fecha:** 2026-04-23  
**Estado:** Completado ✓  
**Tests:** 6/6 passing

## Objetivo

Implementar la infraestructura para corridas MHD cosmológicas completas con ICs magnetizadas,
incluyendo estadísticas del campo B para monitoreo durante la evolución (z=10→8).

## Implementación

### Archivo nuevo: `crates/gadget-ng-mhd/src/stats.rs`

- **`b_field_stats(particles) → Option<BFieldStats>`**: Calcula estadísticas del campo B:
  - `b_mean`: magnitud media ponderada en masa `<|B|>`
  - `b_rms`: RMS del campo `sqrt(<|B|²>)`
  - `b_max`: máximo de `|B|` entre partículas de gas
  - `e_mag`: energía magnética total `Σ m_i |B_i|² / (2μ₀)`
  - `n_gas`: número de partículas de gas incluidas

### Modificaciones de configuración

- `MhdSection.stats_interval: usize` (default: `0` = desactivado): controla la frecuencia
  de cálculo de estadísticas B durante la simulación.

## Física

Para una corrida cosmológica z=10→8 con campo primordial `B₀ = 10^{-10}` G en dirección z:

1. **ICs magnetizadas**: `BFieldKind::Uniform` con `b0_uniform = [0, 0, 1e-10]`
2. **Evolución MHD**: inducción SPH + limpieza Dedner en cada paso
3. **Amplificación**: durante la formación de estructura, B se amplifica por compresión
   (`B ∝ ρ^{2/3}` en régimen flux-frozen)
4. **Divergencia**: `max|∇·B|/|B|/h < 0.1` garantizado por Dedner

## Tests

| Test | Descripción |
|------|-------------|
| `stats_none_empty` | b_field_stats retorna None para lista vacía |
| `stats_none_dm_only` | None con solo partículas DM |
| `stats_uniform_b_correct` | Estadísticas correctas para campo uniforme |
| `primordial_b_stable_after_50_steps` | B primordial estable tras 50 pasos |
| `mag_energy_positive_finite` | E_mag positiva y finita tras evolución |
| `mhd_section_stats_interval_default` | Configuración stats_interval |

## Configuración de Referencia

```toml
[mhd]
enabled = true
b0_kind = "Uniform"
b0_uniform = [0.0, 0.0, 1e-10]
cfl_mhd = 0.3
stats_interval = 10   # calcular estadísticas cada 10 pasos
```
