# Phase 152 — Líneas de emisión nebular (Hα, [OIII], [NII])

**Fecha:** 2026-04-23  
**Crate afectado:** `gadget-ng-analysis`  
**Archivo nuevo:** `crates/gadget-ng-analysis/src/emission_lines.rs`

## Resumen

Implementación de líneas de emisión nebular para clasificación espectroscópica de galaxias: Hα (recombinación case B), [OIII] 5007Å y [NII] 6583Å (excitación colisional). Incluye el diagrama BPT para separar galaxias SF de AGN.

## Física implementada

- **Hα 6563Å**: `j_Hα ∝ n_e² × T^{-0.9}` (Osterbrock & Ferland 2006)
- **[OIII] 5007Å**: `j_OIII ∝ n_e² × (Z/Z_sun) × exp(-29000/T)`
- **[NII] 6583Å**: `j_NII ∝ n_e² × (Z/Z_sun) × exp(-21500/T)`
- **BPT**: cocientes log([NII]/Hα) vs log([OIII]/Hβ) con Hβ = Hα/2.86

## API pública

| Función | Descripción |
|---------|-------------|
| `emissivity_halpha(rho, t_k)` | Emissividad Hα |
| `emissivity_oiii(rho, t_k, z)` | Emissividad [OIII] |
| `emissivity_nii(rho, t_k, z)` | Emissividad [NII] |
| `compute_emission_lines(particles, gamma)` | Líneas por partícula |
| `bpt_diagram(lines)` | Puntos del diagrama BPT |

## Tests (6/6 OK)

1–6: Hα finita, líneas = 0 frío, NII crece con Z, OIII crece con Z, BPT tiene puntos, N=100 sin panics.

## Referencias

- Osterbrock & Ferland (2006) "Astrophysics of Gaseous Nebulae"
- Baldwin, Phillips & Terlevich (1981) PASP 93, 5
