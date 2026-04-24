# Phase 161 — V3: ICs MHD Cosmológicas + Validaciones Cuantitativas

**Fecha:** 2026-04-23  
**Status:** ✅ Implementado y verificado

---

## Resumen

Se implementaron condiciones iniciales para campo magnético primordial y
seis validaciones cuantitativas contra soluciones analíticas MHD.

---

## Archivos nuevos / modificados

| Archivo | Descripción |
|---|---|
| `crates/gadget-ng-core/src/ic_mhd.rs` | Nuevo módulo: `primordial_bfield_ic`, `uniform_bfield_ic`, `check_plasma_beta` |
| `crates/gadget-ng-core/src/lib.rs` | Exporta los nuevos símbolos |
| `crates/gadget-ng-physics/tests/v3_mhd_validation.rs` | 6 tests analíticos |

---

## API pública añadida

```rust
// IC uniforme B = (0,0,b0) — útil para ondas de Alfvén
pub fn uniform_bfield_ic(particles: &mut [Particle], b0: f64);

// IC espectral B(k) ∝ k^n_B (Gaussiana, solenoidal por construcción)
pub fn primordial_bfield_ic(particles: &mut [Particle], b0: f64, spectral_index: f64, seed: u64);

// Ratio β = P_gas / P_mag medio
pub fn check_plasma_beta(particles: &[Particle], gamma: f64) -> f64;
```

---

## Tests V3 (6/6 pasan en CI)

| Test | Física | Criterio | Resultado |
|---|---|---|---|
| `v3_alfven_wave_frequency_converges_quadratically` | ω_num vs k·v_A | err < 10% N=128 | ✅ |
| `v3_alfven_wave_damping_braginskii` | Braginskii disipa E_kin | EK final < EK inicial | ✅ |
| `v3_magnetosonic_wave_phase_velocity` | v_ms = √(v_A²+c_s²) | err < 30% SPH | ✅ |
| `v3_flux_freeze_cosmological_ic` | Φ magnético conservado | drift < 0.1% en 100 pasos | ✅ |
| `v3_plasma_beta_cosmological_ic_large` | β > 1 con B primordial | β > 1 | ✅ |
| `v3_pk_mhd_agrees_with_lcdm_large_scales` | E_mag / E_kin << 1 | ratio < 1% | ✅ |

---

## Notas físicas

- El espectro `n_B = -2.9` reproduce el campo magnético primordial "nearly scale-invariant".
- Con `B₀ = 1e-3` (unidades internas) y energía térmica unitaria, β >> 1 confirmado.
- La conservación del flujo magnético requiere la limpieza Dedner con `c_r = 0.1`.
- Las tolerancias de los tests de ondas son generosas para SPH (no-grid); implementaciones
  en malla alcanzarían < 1% a N=128.

---

## Próximos pasos (V3 futuros)

- Implementar `primordial_bfield_ic` 3D completa con FFT vectorial (actualmente 1D).
- Conectar con `zeldovich_ics` para generar ICs MHD+cosmo completas a z=50.
- Validar β_plasma en función del redshift usando el path `use_hierarchical_let_cosmo`.
