# Phase 120 — Engine Integration: nuevos módulos bariónico en engine.rs

**Fecha**: 2026-04-23  
**Estado**: ✅ Completada  
**Tiempo estimado**: 1 sesión

## Objetivo

Conectar los módulos físicos de Phases 114–119 (ISM multifase, rayos cósmicos, vientos estelares,
AGN bimodal) al loop principal de simulación en `engine.rs`.

## Cambios implementados

### `crates/gadget-ng-cli/src/engine.rs`

Macro `maybe_sph!` extendida con:

1. **Phase 114 — ISM multifase** (antes del feedback):
   ```rust
   if cfg.sph.ism.enabled {
       update_ism_phases(&mut local, &sfr_ism, rho_sf, &cfg.sph.ism, cfg.simulation.dt);
   }
   ```

2. **Phase 115 — Vientos estelares** (dentro del bloque feedback):
   ```rust
   if cfg.sph.feedback.stellar_wind_enabled {
       apply_stellar_wind_feedback(&mut local, &sfr, &cfg.sph.feedback, dt, &mut wind_seed);
   }
   ```

3. **Phase 117 — Rayos cósmicos** (inyección + difusión post-SN):
   ```rust
   if cfg.sph.cr.enabled {
       inject_cr_from_sn(&mut local, &sfr, cfg.sph.cr.cr_fraction, dt);
       diffuse_cr(&mut local, cfg.sph.cr.kappa_cr, dt);
   }
   ```

4. **Phase 113 — Avance de edad estelar + SN Ia**:
   ```rust
   let dt_gyr = cfg.simulation.dt * 1e-3;
   advance_stellar_ages(&mut local, dt_gyr);
   apply_snia_feedback(&mut local, dt_gyr, &mut ia_seed, &cfg.sph.feedback);
   ```

Macro `maybe_agn!` actualizada:
- Reemplaza `apply_agn_feedback` por `apply_agn_feedback_bimodal` (Phase 116)
- Usa `f_edd_threshold`, `r_bubble`, `eps_radio` desde `cfg.sph.agn`

### Benchmark Criterion

Nuevo archivo `crates/gadget-ng-sph/benches/baryonic_stack.rs`:
- `update_ism_phases_1000`: overhead ISM sobre 1000 partículas
- `inject_cr_from_sn_1000`: overhead inyección CR
- `diffuse_cr_100`: overhead difusión CR (O(N²), N=100)

## Tests — `phase120_engine_integration.rs`

| Test | Resultado |
|------|-----------|
| `ism_and_cr_independent` | ✅ ISM conserva energía total; CR no altera u térmica |
| `stellar_winds_and_cr_together` | ✅ Sin panic en secuencia conjunta |
| `agn_bimodal_low_edd_radio_mode` | ✅ Modo radio no modifica u |
| `disabled_modules_no_effect` | ✅ ISM/CR desactivados = no-op |
| `advance_ages_increments_stellar_age` | ✅ Edad estelar incrementa correctamente |
| `full_baryonic_stack_no_panic` | ✅ 50 partículas, ningún NaN |

**Total: 6/6**
