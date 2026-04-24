# Phase 110 — Enriquecimiento Químico SPH

**Fecha**: 2026-04-23
**Estado**: ✅ Completado

## Objetivo

Implementar la distribución de metales desde eventos de supernova y estrellas AGB a las
partículas de gas vecinas usando el kernel SPH (Wendland C2 3D).

## Física implementada

### Algoritmo SN II

Para cada partícula de gas con `sfr[i] > 0`:

1. Masa metálica eyectada: `ΔZ = cfg.yield_snii × sfr[i] × dt`
2. Se buscan vecinas de gas dentro de `r < 2 × h_i`
3. Distribución ponderada por kernel:
   `ΔZ_j = (w_ij / Σ_k w_ik) × ΔZ / m_j`

### Algoritmo AGB

Para cada partícula estelar (`ParticleType::Star`):

- Rata gradual: `ΔZ = cfg.yield_agb × m_star × dt`
- Distribuida a vecinas de gas dentro de `2 × h_i`

### Kernel

Wendland C2 3D: `W(r, h) = (21/2π) h⁻³ (1 - r/2h)⁴ (1 + 2r/h)` para `r ≤ 2h`.

### Capping

La metalicidad de cualquier partícula se acota a `Z ≤ 1.0`.

## Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `gadget-ng-sph/src/enrichment.rs` | **Nuevo**: función `apply_enrichment` |
| `gadget-ng-sph/src/lib.rs` | Re-exporta `apply_enrichment` |

## Tests

`tests/phase110_enrichment.rs` — 6 tests, todos ✅:

| Test | Descripción |
|------|-------------|
| `enrichment_increases_neighbor_metallicity` | Gas donante enriquece vecino |
| `dm_particles_not_enriched` | DM no recibe metales |
| `zero_sfr_no_enrichment` | SFR=0 no inyecta metales |
| `disabled_enrichment_no_change` | Módulo desactivado = no-op |
| `higher_yield_gives_more_enrichment` | Yield mayor → mayor enriquecimiento |
| `metallicity_capped_at_one` | Z siempre ≤ 1 |

## Referencias

- Woosley & Weaver (1995) ApJS 101, 181 — yields SN II
- Portinari, Chiosi & Bressan (1998) A&A 334, 505 — yields AGB
