# Phase 137 — Polvo + RT: Absorción UV por Polvo

**Fecha:** 2026-04-23  
**Estado:** Completado ✓  
**Tests:** 6/6 passing

## Objetivo

Acoplar la evolución del polvo intersticial (Phase 130) con el solver de transferencia radiativa
M1 (Phase 81), incluyendo la atenuación del flujo UV por opacidad del polvo.

## Física

El polvo atenúa el flujo ultravioleta que llega a cada región del ISM:

```
J_UV_eff = J_UV × exp(−τ_dust)
```

La profundidad óptica local del polvo (aproximación optically thin):

```
τ_dust = κ_dust × (D/G) × ρ × h
```

donde:
- `κ_dust = 1000 cm²/g` (UV, ISM estándar)
- `D/G`: relación polvo-gas de la partícula (Phase 130)
- `ρ`: densidad local
- `h`: longitud de suavizado

## Implementación

### Modificaciones: `crates/gadget-ng-sph/src/dust.rs`

Nueva función `dust_uv_opacity(kappa_dust_uv, dust_to_gas, rho, h) → f64`.

### Nuevo: `crates/gadget-ng-rt/src/coupling.rs`

Nueva función `radiation_gas_coupling_step_with_dust(particles, rad, params, kappa_dust_uv, dt, box_size)`:
- Calcula τ_dust para cada partícula de gas
- Aplica atenuación `exp(−τ_dust)` al fotocalentamiento local
- Con `kappa_dust_uv = 0` o `D/G = 0` → idéntico a la versión sin polvo

### Modificaciones de configuración

- `DustSection.kappa_dust_uv: f64` (default: `1000.0`)
- `M1Params.sigma_dust: f64` (default: `0.1`)

## Tests

| Test | Descripción |
|------|-------------|
| `tau_zero_with_no_dust` | τ=0 con D/G=0 |
| `tau_grows_with_dust_to_gas` | τ ∝ D/G |
| `tau_grows_with_kappa` | τ ∝ κ_dust (ratio exactamente 10) |
| `attenuation_in_range` | exp(−τ) ∈ [0,1] para todos los D/G |
| `dust_section_kappa_default` | κ_dust_uv default = 1000 cm²/g |
| `coupling_with_dust_runs` | Función completa sin crash |

## Relevancia Física

- **Regiones HII**: τ_dust ~ 1-10 en UV → regiones H2 protegidas por polvo (self-shielding)
- **Disco galáctico**: D/G ~ 0.01, τ_UV ~ 10 → ISM opaco en UV
- **Galaxias altas en z**: D/G evoluciona con Z, τ crece durante enriquecimiento
- **AGN obscurecidos (Seyfert 2)**: τ >> 1 → absorción total UV/X por torus de polvo
