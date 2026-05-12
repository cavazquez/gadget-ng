# Phase 194 - MHD no ideal: difusión ambipolar

## Objetivo

Agregar una primera pieza de MHD no ideal para gas parcialmente ionizado:
difusión ambipolar dependiente de ionización local.

## Implementación

- Nuevo módulo `gadget-ng-mhd::nonideal`.
- `ionization_fraction_proxy` estima una fracción ionizada acotada a partir de:
  - energía interna,
  - polvo local `dust_to_gas`,
  - piso `ambipolar_ion_floor`.
- `apply_ambipolar_diffusion` amortigua `B` en gas poco ionizado y deposita la
  energía magnética disipada como calor.
- Configuración nueva en `[mhd]`:
  - `ambipolar_diffusion_enabled`
  - `ambipolar_eta`
  - `ambipolar_ion_floor`
  - `ambipolar_dust_coupling`
- Overrides CLI:
  - `--ambipolar`
  - `--ambipolar-eta`
  - `--ambipolar-ion-floor`
  - `--ambipolar-dust-coupling`

## Alcance

Este es un modelo local y estable para barridos/diagnósticos. No resuelve aún
Hall MHD ni resistividad óhmica tensorial completa.

## Validación

Tests en `phase194_nonideal_mhd.rs` verifican que:

- gas más caliente tiene mayor ionización proxy,
- gas polvoriento/neutro difunde más campo magnético,
- la energía magnética disipada calienta el gas.
