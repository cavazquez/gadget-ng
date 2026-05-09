# Phase 176 — Lyman-α Forest: Optical Depth, Transmitted Flux, P(k)_F

**Fecha:** 2026-05-09

## Objetivo

Implementar el análisis del bosque Lyman-α: calcular la profundidad óptica Gunn-Peterson τ_GP y el flujo transmitido F = exp(−τ) a lo largo de líneas de visión (sightlines) a través de partículas de gas, y derivar el power spectrum 1D del campo de contraste de flujo δ_F.

## Modelo físico

**Gunn-Peterson optical depth:**

$$\tau_{GP} \approx \tau_0 \, x_{\text{HI}} \, (1+\delta) \, \left(\frac{1+z}{4}\right)^{3/2} \, \left(\frac{\Omega_b h^2}{0.02}\right)$$

donde τ₀ ≈ 5.2 × 10⁻³ es el prefactor GPI para Ly-α a z ~ 3.

El flujo transmitido F = exp(−τ) se calcula por celda de posición comóvil, con deposición SPH gaussiana para cada partícula de gas.

**Power spectrum 1D del flujo:**

$$\delta_F = F/\langle F \rangle - 1, \quad P_F(k) = \langle |\tilde{\delta}_F(k)|^2 \rangle$$

promediado sobre N sightlines.

## API

Crate: `gadget-ng-analysis`, módulo `lya_forest`.

### Tipos principales

| Tipo | Descripción |
|------|-------------|
| `LyaParams` | n_sightlines, n_velocity_cells, z_source, dv_kms, t_igm_kelvin |
| `LyaCosmoParams` | h0, omega_m, omega_lambda |
| `LyaChemState` | x_hi (fracción neutral de H) |
| `LyaSightline` | flux[·], tau[·], velocity[·], mean_flux, tau_effective |
| `LyaForestResult` | n_sightlines, pk_flux, mean_flux, tau_effective |
| `LyaPkBin` | k, pk, n_modes |

### Funciones principales

| Función | Descripción |
|---------|-------------|
| `compute_tau_along_sightline(...)` | τ(v) y F(v) para una sightline |
| `compute_lya_pk_1d(sightlines, n_k_bins)` | P(k)_F desde múltiples sightlines |
| `analyze_lya_forest(particles, box_size, params, cosmo, dir, chem)` | Pipeline completo |
| `generate_impact_positions(n, box_size)` | Grilla regular de impactos |
| `hubble_z(z, h0, Ω_m, Ω_Λ)` | H(z) para ΛCDM plano |

### Config TOML

```toml
[insitu_analysis]
enabled = true
interval = 20
lya_enabled = true
lya_n_sightlines = 256
```

### Output JSON (insitu)

```json
{
  "lya": {
    "n_sightlines": 256,
    "mean_flux": 0.82,
    "tau_effective": 0.20,
    "n_pk_bins": 128
  }
}
```

## Tests (6)

1. `empty_particles_zero_flux` — Sin gas → F = 1 (τ = 0)
2. `neutral_gas_produces_absorption` — Gas neutral → min(F) < 1
3. `ionized_gas_transparent` — Gas ionizado → ⟨F⟩ > 0.99
4. `generate_impact_positions_count` — N posiciones correctas
5. `pk_1d_empty_gives_empty` — Sin sightlines → resultado vacío
6. `hubble_z_increases_with_redshift` — H(3) > H(0)

## Limitaciones

- Prefactor GPI simplificado (τ₀ = 5.2e-3 × (1+z)^{3/2}), no incluye Voigt profiles
- Template térmico fijo (T_IGM = 10⁴ K); no incluye temperatura-density relation T(ρ)
- Sightlines en grilla regular (no random)
- Sin acoplamiento directo con reionización/chemistry; usa LyaChemState simplificado
- P(k) 1D (no 3D); no estima fluctuating Gunn-Peterson approximation completa

## Archivos

| Archivo | Acción |
|---------|--------|
| `crates/gadget-ng-analysis/src/lya_forest.rs` | NUEVO (~600 líneas) |
| `crates/gadget-ng-analysis/src/lib.rs` | Editar (pub mod + re-exports) |
| `crates/gadget-ng-core/src/config/sections/analysis.rs` | Editar (lya_enabled, lya_n_sightlines) |
| `crates/gadget-ng-cli/src/insitu.rs` | Editar (LyaForestOut, cálculo Ly-α) |