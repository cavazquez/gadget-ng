# Phase 181 — RT multifrecuencia + Lyman-Werner

## Objetivo

Agregar una capa espectral reducida sobre el solver M1 existente para distinguir
fotones ionizantes de HI/HeI/HeII, banda Lyman-Werner e IR. El primer acoplamiento
físico implementado es la fotodisociación LW de `H2` y `HD`, necesaria para Pop III.

## Implementación

- `gadget-ng-rt::multifrequency` define `PhotonGroup`, `MultiFrequencyField` y
  `MultiFrequencyRates`.
- Los grupos son `HiIonizing`, `HeiIonizing`, `HeiiIonizing`, `LymanWerner` e
  `Infrared`, con energías representativas y secciones eficaces reducidas.
- `apply_lw_photodissociation` aplica destrucción exponencial de `H2` y `HD` y
  re-normaliza `ChemState` para conservar núcleos de H/D.
- `[rt]` suma `multifrequency_enabled`, `lw_h2_factor` y `lw_hd_factor` como knobs
  de configuración retrocompatibles.

## Validación

Nuevo test `phase181_multifrequency_lw.rs`:

- estabilidad de índices/energías de grupos;
- separación espectral entre ionización HI y LW;
- destrucción LW de `H2`/`HD` con conservación de deuterio;
- serde de los nuevos campos `[rt]`.

## Limitaciones

La fase introduce la API y el acoplamiento químico local. El transporte cosmológico
de cada grupo en el loop principal queda preparado mediante `MultiFrequencyField`,
pero todavía no reemplaza el campo M1 escalar usado por `step_rt`.

