# Phase 182 — Polvo IR / emisión térmica

## Objetivo

Completar el lado infrarrojo del modelo de polvo: además de atenuar UV y recibir
presión de radiación, el polvo ahora puede estimar una temperatura de equilibrio
y re-emitir energía en el grupo IR del RT multifrecuencia.

## Implementación

- `DustSection` suma knobs retrocompatibles:
  - `ir_emission_enabled`
  - `kappa_dust_ir`
  - `ir_emissivity`
  - `dust_temperature_floor_k`
  - `dust_temperature_cap_k`
- `gadget-ng-sph::dust_equilibrium_temperature` estima temperatura de granos con
  cuerpo gris modificado (`T^6`, beta=2).
- `gadget-ng-sph::dust_ir_luminosity` calcula luminosidad IR local para gas con
  `dust_to_gas > 0`.
- `gadget-ng-rt::deposit_dust_ir_emission` deposita esa energía en
  `PhotonGroup::Infrared` dentro de `MultiFrequencyField`.

## Validación

Nuevo test `phase182_dust_ir_emission.rs`:

- temperatura de polvo crece con el campo radiativo;
- luminosidad IR requiere gas y polvo;
- el depósito afecta solo al grupo IR;
- serde de los nuevos campos `[sph.dust]`.

## Limitaciones

La emisión IR queda expuesta como API local. La integración automática en el loop
principal queda para una fase posterior, cuando `step_rt` migre del campo M1 escalar
a `MultiFrequencyField` persistente.

