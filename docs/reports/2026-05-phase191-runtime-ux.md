# Phase 191 - Runtime UX + Experiment Polish

## Objetivo

Convertir la física ya integrada entre Phases 100-190 en una superficie
operable para barridos: flags CLI, validaciones de combinación, presets y una
matriz de experimentos cortos.

## Implementado

- Overrides en `gadget-ng stepping` para:
  - SPH, cooling, feedback, vientos,
  - AGN, radio mode, spin, mergers, PBH,
  - CR, difusión anisótropa y streaming,
  - MHD, campo inicial, turbulencia, two-fluid,
  - SIDM, f(R), RT, reionización,
  - WDM/FDM.
- Validación de combinaciones con dependencias físicas:
  - CR anisótropo/streaming requiere MHD,
  - turbulencia requiere MHD,
  - reionización/multifrecuencia requiere RT,
  - feedback y CR requieren SPH.
- Presets bajo `configs/experiments/`.
- Runner `scripts/run_phase191_experiments.sh`.
- Runbook `docs/runbooks/runtime-physics-overrides.md`.

## Principio

Los flags son overrides efímeros: no reescriben el TOML. Esto permite mantener
configs base versionadas y hacer barridos reproducibles desde comandos.
