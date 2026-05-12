# Phase 184 — Warm / fuzzy dark matter

## Objetivo

Agregar materia oscura no fría como cutoff de pequeña escala en las condiciones
iniciales cosmológicas. La fase cubre WDM térmica y FDM ultraligera con una API
reducida y testeable.

## Implementación

- Nueva sección `[dark_matter]`:
  - `enabled`
  - `model = "cold" | "warm" | "fuzzy"`
  - `m_wdm_kev`
  - `m_fdm_22`
- Nuevo módulo `gadget-ng-core::dark_matter`:
  - `wdm_transfer_suppression`
  - `wdm_half_mode_k`
  - `fdm_transfer_suppression`
  - `fdm_half_mode_k`
  - `fdm_quantum_pressure_cs2`
  - `dark_matter_transfer_suppression`
- `zeldovich_ics`, `zeldovich_ics_with_convention` y `zeldovich_2lpt_ics` aplican
  el factor de transferencia WDM/FDM sobre la amplitud gaussiana de cada modo.

## Validación

Nuevo test `phase184_wdm_fdm.rs`:

- WDM tiende a unidad a bajo `k` y suprime alto `k`;
- WDM más masiva mueve el half-mode a `k` mayor;
- FDM suprime alto `k` y responde a la masa ultraligera;
- el proxy de presión cuántica escala como `k^4 / a^2`;
- serde de modelos warm/fuzzy.

## Limitaciones

La fase implementa el cutoff de ICs y un proxy de presión cuántica. No reemplaza
el solver gravitatorio por Schrödinger-Poisson ni agrega evolución hidrodinámica
completa del campo FDM.

