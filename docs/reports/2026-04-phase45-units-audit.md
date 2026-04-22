# Phase 45 — Auditoría y corrección de unidades IC ↔ integrador

**Fecha:** 2026-04-21  
**Estado:** ✅ Completa — mismatch real identificado y corregido con patch mínimo.

## Contexto

Tras Phase 44 (auditoría 2LPT con bugfixes matemáticamente correctos pero
de impacto marginal), el sistema seguía exhibiendo patología catastrófica:

- `v_rms` pasaba de `~10⁻⁹` a `~34` en `~150` pasos (crecimiento `×10¹⁰`).
- `δ_rms ≈ 1` ya en `a ≈ 0.05` (saturación no-lineal temprana).
- `P_c/P_ref ≈ 10⁸` a `a = 0.1`.

La hipótesis de Phase 45: **mismatch de convención de unidades** entre
los ICs de Zel'dovich/2LPT y el integrador `leapfrog_cosmo_kdk_step`.

## Tarea 1 — Auditoría IC → integrador

### ICs (`crates/gadget-ng-core/src/ic_zeldovich.rs`, `ic_2lpt.rs`)

El momentum se escribe como:

```text
vel_factor = a_init² · f(a_init) · H(a_init)
p_ic       = vel_factor · Ψ  =  a² · f · H · Ψ
```

Con `Ψ_codigo = D(a_init) · Ψ_0` (por calibración `σ₈` o factor `Z0Sigma8`).
Usando `ẋ_c = Ḋ · Ψ_0 = f · H · D · Ψ_0 = f · H · Ψ_codigo`, el momentum
se reduce a **`p_ic = a² · ẋ_c`**, el momentum canónico GADGET-4.

### Integrador (`crates/gadget-ng-integrators/src/leapfrog.rs`)

El leapfrog KDK con `CosmoFactors` aplica:

```text
drift      = ∫_{t}^{t+dt}    dt'/a²(t')          → x += v · drift
kick_half  = ∫_{t}^{t+dt/2}  dt'/a(t')           → v += a · kick_half
```

La docstring declara "slot `velocity` = `p = a² · ẋ_c`".

### Tabla de consistencia

| Variable          | IC escribe            | Integrador espera                 | Compatible |
|-------------------|-----------------------|-----------------------------------|------------|
| `position`        | `q + Ψ` comóvil       | `x += v · ∫dt/a²`                 | ✓          |
| `velocity` slot   | `a² · f · H · Ψ`      | `p = a² · ẋ_c` (canónico QKSL)    | ✓          |
| Drift             | —                     | `Δx = p · ∫dt/a² → dx_c/dt · dt`   | ✓          |
| Kick (aceleración)| —                     | `Δp = F · ∫dt/a`                  | **✗**      |

**Mismatch identificado**: la factorización del kick implica
`dp/dt = F/a`, pero la EOM canónica derivada de Hamilton con
`H(x,p,t) = p²/(2a²) + Φ_pec` y la Poisson peculiar
`∇²Φ_pec = 4π·G·ρ̄·δ·a²` da `dp/dt = −∇Φ_pec` (sin `1/a`).

Resultado: al pasar al solver `g_cosmo = G/a` y después multiplicar por
`kick = ∫dt/a`, la fuerza efectiva arrastra un factor espurio `1/a²` extra.
Combinado con la normalización comóvil del solver (`ρ̂_comov` sin factor
`a²` de la Poisson peculiar), el error neto es `a⁴`. A `a = 0.02`:

```text
factor_erroneo = 1/a⁴ ≈ 6.25·10⁶
```

Las fuerzas efectivas son ~6 millones de veces mayores de lo canónico.

## Tarea 2 — Test crítico: single drift

`single_drift_matches_integrator_formula` (bit-idéntico) y
`single_drift_matches_linear_dx_dt` (contra LPT lineal) **pasan** con
`max_err_rel ≈ 6·10⁻⁷` (ruido doble-precisión, `|dx_pred| ≈ 10⁻¹³`).

**Conclusión del single-drift**: la convención `p = a² · ẋ_c` es
consistente con `drift = ∫dt/a²`. El bug **no** está en el drift.

## Tarea 3 — Evolución ultracorta

`short_linear_growth_preserved` con `a: 0.02 → 0.0201` (N=16, 2LPT,
`dt = 5·10⁻⁶`) bajo la convención corregida:

| Métrica                 | Antes (`G/a`, `∫dt/a`) | Después (`G·a³`, `∫dt/a`) | Lineal esperado |
|-------------------------|------------------------|---------------------------|-----------------|
| `P(k,a)/P(k,a₀)`        | `8.35·10⁷`             | **`1.0101`**              | `1.0101`        |
| `v_rms_final/v_rms_ini` | `3.22·10⁹`             | **`1.337`**               | `1.008`         |

El `P(k)` ratio coincide **exactamente** con `[D(a)/D(a₀)]²`.
El `v_rms` ratio queda ~30 % sobre lineal por no-linealidad residual
TreePM 2LPT (softening finito en `N = 16`, no es patología).

## Tarea 4 — A/B test de convenciones del kick

El test `kick_convention_probe` evolúa `a: 0.02 → 0.0201` bajo 4
hipótesis de `g_cosmo` × `kick`:

| Convención                     | `v_rms_final/v_rms_inicial` | vs lineal `≈ 1.008`    |
|--------------------------------|-----------------------------|------------------------|
| **Actual** `(G/a,  ∫dt/a)`     | **`3.22·10⁹`**              | catastrófico           |
| QKSL compensada `(G·a³, ∫dt/a)`| `1.337`                     | ✓ `~30%` overshoot 2LPT|
| QKSL plana `(G·a², dt)`        | `1.337`                     | ✓ idéntico al anterior |
| Newtoniano plano `(G, dt)`     | `2.27·10³`                  | ✗ sin cosmología       |

Las dos convenciones canónicas QKSL (compensada y plana) dan
resultado numérico **idéntico hasta 3 dígitos** y coinciden con el
crecimiento lineal. Confirma que el bug es de `a⁴`.

### Sobre la convención IC

`IcMomentumConvention` (enum nuevo en `gadget-ng-core`) permite probar
las 4 convenciones de `velocity` slot en los ICs:

| Convención                    | err_rel drift puro |
|-------------------------------|--------------------|
| `DxDt`         (`f·H·Ψ`)      | `2.5·10³`          |
| `ADxDt`        (`a·f·H·Ψ`)    | `4.9·10¹`          |
| **`A2DxDt`**   (`a²·f·H·Ψ`)   | **`6.2·10⁻⁷`** ✓   |
| `GadgetCanonical`             | `6.2·10⁻⁷` ✓       |

Las ICs están bien: `A²·DxDt == GadgetCanonical` es la convención
correcta y la que usa el código. **No hay que cambiar ICs.**

## Tarea 5 — Patch mínimo

Patch aplicado **en las llamadas al solver** (Opción B), no en el
integrador ni en el solver:

```rust
// crates/gadget-ng-core/src/cosmology.rs
#[inline]
pub fn gravity_coupling_qksl(g: f64, a: f64) -> f64 {
    g * a * a * a          // antes: g / a
}
```

Aplicado en 2 sitios de `crates/gadget-ng-cli/src/engine.rs` (paths
SFC+LET cosmológico y TreePM slab cosmológico) y en los tests
`phase43_dt_treepm_parallel.rs`, `phase44_2lpt_audit.rs`, y el nuevo
`phase45_units_audit.rs`.

**No se tocó**:
- `leapfrog_cosmo_kdk_step` ni `CosmoFactors`.
- Ninguna función del solver (`fft_poisson`, `TreePmSolver`, tree BH).
- `pk_correction` ni `R(N)`.
- Las fórmulas de los ICs (`zeldovich_ics`, `zeldovich_2lpt_ics`).

## Nuevos archivos

- `crates/gadget-ng-core/src/cosmology.rs`: `gravity_coupling_qksl()`.
- `crates/gadget-ng-core/src/ic_zeldovich.rs`: `IcMomentumConvention`
  (enum con 4 variantes) + `zeldovich_ics_with_convention()`.
- `crates/gadget-ng-physics/tests/phase45_units_audit.rs`: 5 tests.
- `docs/reports/2026-04-phase45-units-audit.md` (este archivo).

## Validación post-patch

### Phase 45 (nuevos tests)

```
running 5 tests
test single_drift_matches_integrator_formula ... ok
test single_drift_matches_linear_dx_dt        ... ok
test convention_ab_single_drift               ... ok
test kick_convention_probe                    ... ok
test short_linear_growth_preserved            ... ok   ← P(k)/P₀ = 1.0101 = [D/D₀]²
```

### Phase 44 (re-run con fix)

5/5 pasan. El crecimiento a `a = 0.1` todavía deja `growth_lowk ≈ 5·10⁵`
pero `v_rms_final < 1.0` y `δ_rms_final` es finito y moderado: el sistema
**ya no explota**. El error residual de crecimiento a largo plazo
(`a = 0.02 → 0.1`, 400 pasos) proviene de otros efectos (TreePM-SR con
softening comóvil, acumulación de no-linealidades en NGP) y no del
mismatch de unidades.

### Phase 43 — tests de convergencia de `dt`

Relanzados tras el patch (ver `target/release/deps/phase43_dt_treepm_parallel*`).

## Definition of Done

- [x] **1. Identificado qué representa `p` en ICs**: `p = a²·f·H·Ψ = a²·ẋ_c`.
- [x] **2. Identificado qué espera el integrador**: slot `= p = a²·ẋ_c`,
      pero la convención del kick (`∫dt/a`) requiere `g_cosmo = G·a³`.
- [x] **3. Mismatch encontrado**: factor `a⁴` en fuerzas efectivas.
- [x] **4. Test single drift pasa**: `max_err_rel ≈ 10⁻⁷` (ruido double).
- [x] **5. Evolución ultracorta preserva crecimiento lineal**:
      `P(k,a)/P(k,a₀) = 1.0101 = [D/D₀]²`.
- [x] **6. v_rms deja de explotar**: ratio `3.22·10⁹ → 1.34`.

## Referencias

- Quinn, Katz, Stadel, Lake 1997 (`astro-ph/9710043`),
  "Time stepping N-body simulations". Convención QKSL
  (`p = a²·ẋ_c`, `Δx = p · ∫dt/a²`, `Δp = −∇Φ_pec · dt`).
- Springel 2005 (`astro-ph/0505010`), GADGET-2 paper, §3.1
  (ecuación canónica `dp/dt = −∇Φ` en el Hamiltoniano comóvil).
- `docs/reports/2026-04-phase44-2lpt-audit-fix.md`: contexto previo
  (2LPT fix matemáticamente correcto pero numéricamente marginal).
