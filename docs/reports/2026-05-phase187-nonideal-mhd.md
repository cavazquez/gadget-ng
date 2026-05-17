# Phase 187: Non-ideal MHD — Ohmic Diffusion + Chemistry-Coupled Ambipolar

**Date:** 2026-05-17
**Status:** Completado
**Crates:** `gadget-ng-mhd`, `gadget-ng-core`, `gadget-ng-cli`

---

## Motivación

Phase 186 introdujo el término Hall (rotación de B sin disipación). La ecuación de
inducción no-ideal completa tiene tres términos:

```
dB/dt = ∇×(v×B)          [MHD ideal, Phase 123]
       − η_H ∇×(J×B)/ρ   [Hall, Phase 186]
       − η_Ohm B/h²       [Óhmica, Phase 187 ← nuevo]
       − η_AD/(ρ x_i) …  [Ambipolar, Phase 194]
```

Además, la difusión ambipolar usaba hasta ahora un proxy térmico `x_i = u/(u+1)`
para estimar la fracción ionizada. Phase 187 acopla correctamente la ambipolar al
solver de química (Phase 86/87), usando `x_e` real de `ChemState`.

---

## Implementación

### 1. Difusión óhmica resistiva (`apply_ohmic_diffusion`)

**Modelo local:** `dB/dt|_Ohm = −η_Ohm B / h²`

Amortiguamiento por paso: `B_new = B_old × exp(−η_Ohm dt / h²)`

Energía disipada → calor: `Δu = (γ−1) × ΔB²/(2m)`

**Rutas de ejecución** (mismo patrón que ambipolar y Hall):
- Rayon `par_iter_mut` cuando `feature = "rayon"`
- AVX-512F (lotes 8) → AVX2+FMA (lotes 4) → scalar en `#[not(rayon)]`

### 2. Ambipolar acoplada a química (`apply_ambipolar_diffusion_with_chem`)

Firma: `(particles, ion_fracs: &[f64], eta_ad, ion_floor, gamma, dt)`

`ion_fracs[i]` es extraído por el engine como `chem_states[i].x_e` (electrones
por átomo de H del solver de química). La función usa exactamente la misma
física que `apply_ambipolar_diffusion` pero con `x_i = ion_fracs[i].clamp(floor, 1.0)`
en lugar del proxy térmico.

**Nota de diseño:** La función acepta `&[f64]` en lugar de `&[ChemState]` para
preservar las fronteras de crates (`gadget-ng-mhd` no depende de `gadget-ng-rt`).

### 3. Config TOML

Nuevos campos en `[mhd]`:

```toml
[mhd]
# Difusión óhmica (Phase 187)
ohmic_enabled = false
ohmic_eta     = 0.0        # [L²/T en unidades code], e.g. 0.01

# Acoplamiento química para difusión ambipolar (Phase 187)
ambipolar_use_chem_ionization = false
# (requiere rt.enabled = true y stack de química activo)
```

### 4. Engine wiring (`context.rs`)

- `apply_ohmic_diffusion` se llama dentro de `step_mhd` cuando `ohmic_enabled && ohmic_eta > 0`.
- Nueva función `step_mhd_chem_ambipolar` se llama después de `step_mhd` en todos los
  loops de integración (7 call sites), usando el `sph_chem_states` disponible en scope.

---

## Tests (7 nuevos, todos en `nonideal::tests`)

| Test | Qué verifica |
|------|-------------|
| `ohmic_diffusion_reduces_b` | `|B|` disminuye con `η_Ohm > 0` |
| `ohmic_diffusion_heats_gas` | energía interna aumenta por disipación |
| `ohmic_diffusion_energy_conservation` | conservación total (magnética + térmica), tol 1e-12 |
| `ohmic_diffusion_no_effect_with_zero_eta` | `η_Ohm = 0` → B invariante |
| `ohmic_diffusion_no_effect_on_dm` | DM sin efecto |
| `chem_coupled_ambipolar_high_ionization_gives_less_damping` | `x_e = 0.001` amortigua más que `x_e = 0.99` |
| `chem_coupled_ambipolar_matches_proxy_at_full_ionization` | `x_e = 1` → tasa = 0 → B inalterado |

---

## Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-mhd/src/nonideal.rs` | `apply_ohmic_diffusion` (AVX2+512+Rayon), `apply_ambipolar_diffusion_with_chem`, 7 tests |
| `crates/gadget-ng-mhd/src/lib.rs` | re-exporta `apply_ohmic_diffusion`, `apply_ambipolar_diffusion_with_chem` |
| `crates/gadget-ng-core/src/config/sections/mhd.rs` | `ohmic_enabled`, `ohmic_eta`, `ambipolar_use_chem_ionization` |
| `crates/gadget-ng-core/src/config/mod.rs` | validación `ohmic_eta >= 0` |
| `crates/gadget-ng-cli/src/engine/stepping/context.rs` | wiring ohmic en `step_mhd`, nueva `step_mhd_chem_ambipolar` |
| `crates/gadget-ng-cli/src/engine/stepping/mod.rs` | 7 call sites: `step_mhd_chem_ambipolar` tras `step_mhd` |

---

## Relación con fases anteriores

| Phase | Término | Estado |
|-------|---------|--------|
| 123 | MHD ideal `dB/dt = ∇×(v×B)` | ✅ |
| 125 | Limpieza Dedner div-B | ✅ |
| 186 | Hall drift `η_H J×B/ρ` | ✅ |
| 194 | Ambipolar (proxy térmico) | ✅ |
| **187** | **Ohmic `η_Ohm B/h²` + ambipolar acoplada a química** | **✅** |
