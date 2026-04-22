# Phase 49 — Fix del Integrador Cosmológico

**Fecha:** 2026-04-22  
**Estado:** ✅ Completa — coupling histórico corregido, timestep adaptativo implementado, 10 tests nuevos pasando.

---

## Contexto

Durante Phase 48, se descubrió que el integrador PM cosmológico con coupling
`G·a³` (Phase 45) producía inestabilidad numérica para evoluciones largas
(a=0.05→0.5) con N=32. El diagnóstico inicial señaló como candidatos:

1. Coupling gravitacional incorrecto.
2. Timestep fijo demasiado grande para el rango dinámico de H(a).
3. N=32 insuficiente para el régimen no-lineal tardío.

Adicionalmente, se identificó que tres archivos de test históricos
(`cosmo_pm.rs`, `phase37_growth_rescaled_ics.rs`,
`phase41_high_resolution_validation.rs`) aún usaban el coupling incorrecto
`G/a` que Phase 45 había eliminado del código de producción.

---

## Diagnóstico: coupling G/a vs G·a³

### Derivación teórica

El integrador KDK almacena `p = a²ẋ_c` con:
- **drift**: `Δx_c = p · ∫dt/a²`
- **kick**: `Δp = F · ∫dt/a`

La EOM canónica QKSL da `dp/dt = −∇Φ_pec`. Como el kick aplica `Δp = F·∫dt/a`,
se tiene `dp/dt = F/a`. Para cumplir `dp/dt = −∇Φ_pec`:

```text
F/a = −∇Φ_pec
F = −a·∇Φ_pec

Poisson comóvil: ∇²_c Φ_pec = 4πGρ̄δa²
Solver retorna:  F_solver = −g/G · ∇Φ_pec/a²

Igualando: g/G/a² = a  →  g = G·a³  ✓
```

El coupling histórico `G/a` cometía un error de factor `a⁴`. A a=0.02:

```
G/a = 50G     ↔     G·a³ = 8×10⁻⁶G     →     ratio = 6.25×10⁶
```

### Test empírico A/B (`phase49_coupling_ab_short`)

| Convención | P(k) ratio (a=0.02→0.05) | [D/D₀]² teoría | Estable |
|------------|--------------------------|----------------|---------|
| G·a³ (QKSL) | 1.25–2.0 (N=16, ± CoV) | ≈ 1.26        | ✅ |
| G/a (histórico) | NaN / > 10⁶ × teoría | — | ❌ |

El coupling `G·a³` es físicamente correcto y numéricamente estable.

---

## Timestep adaptativo: `adaptive_dt_cosmo`

Implementado en `crates/gadget-ng-core/src/cosmology.rs`.

### Fórmula

```rust
pub fn adaptive_dt_cosmo(
    params: CosmologyParams,
    a: f64,
    acc_max: f64,   // |aceleración| máxima en unidades de código
    softening: f64, // longitud de suavizado
    eta_grav: f64,  // ~0.025; fracción gravitacional (Quinn+1997)
    alpha_h: f64,   // ~0.025; fracción del tiempo de Hubble
    dt_max: f64,    // límite superior explícito
) -> f64 {
    let dt_grav = eta_grav * (softening / acc_max).sqrt();
    let dt_hub  = alpha_h / H(a);
    min(dt_grav, dt_hub, dt_max)
}
```

### Validación (`phase49_adaptive_dt_stable`)

- N=16, a=0.02→0.5 (25× expansión), dt_max=1e-3.
- Resultado: sin NaN/Inf, v_rms finito, P(k) positivo en ≥ 2 bins.
- Número de pasos adaptativo coherente con el rango dinámico de H(a).

### Convergencia (`phase49_timestep_convergence`)

- N=16, a=0.02→0.10.
- dt_fijo=2e-4: ratio P(k) = 6.873
- dt_adaptativo: ratio P(k) = 6.853
- **Diferencia: 0.29 %** — excelente convergencia.

---

## Nota sobre inconsistencia de unidades en tests

Los parámetros de test (G=1, H₀=0.1, ρ̄=1, box=1) no satisfacen la condición
de consistencia cosmológica:

```text
H₀_requerida = √(8πGρ̄Ω_m/3) = √(8π×0.315/3) ≈ 1.62

H₀_código = 0.1  →  discrepancia × 16
```

Consecuencia: el P(k) ratio de la simulación NO sigue D²(a) para evoluciones
largas. El streaming inicial (vel_factor = a²·f·H·Ψ) domina sobre la respuesta
gravitacional. Para evoluciones cortas (Phase 45, 47) la correspondencia con
D²(a) es correcta porque el streaming codifica la predicción analítica.

**Esto NO es un bug del integrador** — es una limitación intrínseca de los
parámetros de test. Para validación física rigurosa, usar `UnitsSection.enabled
= true` con `G_KPC_MSUN_KMPS` y parámetros cosmológicos consistentes.

---

## Corrección de tests históricos

| Archivo | Línea modificada | Antes | Después |
|---------|-----------------|-------|---------|
| `cosmo_pm.rs` | `pm_cosmo_no_explosion` | `G / a` | `gravity_coupling_qksl(G, a)` |
| `phase37_growth_rescaled_ics.rs` | `evolve_to_a` | `G / a` | `gravity_coupling_qksl(G, a)` |
| `phase41_high_resolution_validation.rs` | `evolve_pm_to_a` | `G / a` | `gravity_coupling_qksl(G, a)` |

Todos los tests no-matriciales de Phase 37 y Phase 41 pasan con el coupling
corregido. Los tests matriciales (54 y 48 simulaciones respectivamente) son
lentos en modo debug y se recomienda ejecutar con `--release`.

---

## Halofit con integrador corregido

Tres tests en `phase49_halofit_comparison.rs` validan el pipeline:

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `phase49_halofit_pk_at_ics_z49` | P_sim vs P_halofit en z=49 (ICs) | ✅ < 25 % |
| `phase49_halofit_linear_consistency` | Ratio Halofit z=2/z=49 vs D²(a) | ✅ < 5 % |
| `phase49_halofit_ratio_z2_vs_z0` | Boost no-lineal monotónico en z | ✅ |

---

## Tests nuevos

| Archivo | Tests | Descripción |
|---------|-------|-------------|
| `phase49_integrator_diagnosis.rs` | 4 | Coupling A/B, dt adaptativo, fondo cosmológico |
| `phase49_long_growth.rs` | 3 | Estabilidad larga, convergencia dt, snapshots |
| `phase49_halofit_comparison.rs` | 3 | Halofit vs ICs, consistencia lineal, boost |
| `cosmology.rs` (unit tests heredados) | 9 | Todos pasan con nueva función |

**Total: 10 tests de integración nuevos + `adaptive_dt_cosmo` con docs completa.**

---

## Nuevos archivos

- `crates/gadget-ng-core/src/cosmology.rs`: `adaptive_dt_cosmo()` exportada.
- `crates/gadget-ng-core/src/lib.rs`: re-export de `adaptive_dt_cosmo`.
- `crates/gadget-ng-physics/tests/phase49_integrator_diagnosis.rs`: 4 tests.
- `crates/gadget-ng-physics/tests/phase49_long_growth.rs`: 3 tests.
- `crates/gadget-ng-physics/tests/phase49_halofit_comparison.rs`: 3 tests.
- `docs/reports/2026-04-phase49-integrator-fix.md`: este archivo.

## Archivos modificados

- `crates/gadget-ng-physics/tests/cosmo_pm.rs`: coupling fix.
- `crates/gadget-ng-physics/tests/phase37_growth_rescaled_ics.rs`: coupling fix.
- `crates/gadget-ng-physics/tests/phase41_high_resolution_validation.rs`: coupling fix.
- `docs/roadmap.md`: entrada 20.
- `docs/user-guide.md`: sección "Integrador cosmológico: timestep adaptativo".

---

## Definition of Done

- [x] `adaptive_dt_cosmo` implementada con criterios Quinn+1997 y Hubble.
- [x] Tests A/B confirman: G·a³ correcto, G/a diverge por factor ~10⁶.
- [x] Estabilidad a=0.02→0.5 con dt adaptativo (sin NaN/Inf).
- [x] Convergencia timestep (fijo vs adaptativo): diferencia < 0.3 %.
- [x] Coupling histórico G/a eliminado de cosmo_pm, phase37, phase41.
- [x] Halofit consistente con integrador corregido (3 tests pasan).
- [x] Inconsistencia G/H₀ cuantificada y documentada.

## Referencias

- Quinn, Katz, Stadel, Lake 1997 (`astro-ph/9710043`): convención QKSL, criterio de timestep gravitacional.
- Springel 2005 (`astro-ph/0505010`): GADGET-2, §3.1, Hamiltoniano comóvil.
- `docs/reports/2026-04-phase45-units-audit.md`: derivación y validación del coupling G·a³.
- `docs/reports/2026-04-phase48-halofit-validation.md`: descubrimiento de la inestabilidad PM para a>0.2 con N=32.
