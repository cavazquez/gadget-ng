# Phase 106 — Restart con SPH State Completo

**Fecha**: 2026-04-23  
**Crate principal**: `gadget-ng-cli`  
**Archivos clave**: `crates/gadget-ng-cli/src/engine.rs`, `crates/gadget-ng-sph/src/agn.rs`, `crates/gadget-ng-rt/src/chemistry.rs`

## Problema

Aunque las partículas (con sus campos SPH gracias a Phase 105) se persistían en el checkpoint,
dos vectores de estado adicionales no se guardaban:

1. `agn_bhs: Vec<BlackHole>` — los agujeros negros supermasivos perdían su masa acumulada y tasa
   de acreción. Tras un resume, los BH se reinicializaban desde cero.
2. `sph_chem_states: Vec<ChemState>` — el estado de ionización (fracciones HI/HII/HeI/etc.)
   se reiniciaba en estado neutro, perdiendo el progreso de reionización.

## Solución

### Serde para tipos de estado

Se añadieron derives `Serialize`/`Deserialize` a:
- `BlackHole` en `crates/gadget-ng-sph/src/agn.rs`
- `ChemState` en `crates/gadget-ng-rt/src/chemistry.rs`

### CheckpointMeta ampliada

```rust
struct CheckpointMeta {
    // campos existentes ...
    #[serde(default)] has_agn_state: bool,
    #[serde(default)] has_chem_state: bool,
}
```

### save_checkpoint

- Si `agn_bhs` no está vacío → escribe `checkpoint/agn_bhs.json`.
- Si `sph_chem_states` no está vacío → escribe `checkpoint/chem_states.json`.

### load_checkpoint

- Lee `agn_bhs.json` si `has_agn_state = true`.
- Lee `chem_states.json` si `has_chem_state = true`.
- Retorna `Option<Vec<BlackHole>>` y `Option<Vec<ChemState>>` adicionales.

### Restauración en el motor

```rust
let mut agn_bhs = resume_agn_bhs.take().unwrap_or_default();
let mut sph_chem_states = if cfg.reionization.enabled {
    resume_chem_states.take().unwrap_or_else(|| local.iter().map(|_| neutral()).collect())
} else { Vec::new() };
```

## Tests

Archivo: `crates/gadget-ng-physics/tests/phase106_restart_sph.rs` (6 tests)

| Test | Descripción |
|------|-------------|
| `black_hole_serde_roundtrip` | BlackHole se serializa correctamente |
| `agn_bhs_checkpoint_roundtrip` | Vec de BHs se persiste y restaura |
| `chem_state_serde_roundtrip` | ChemState se serializa correctamente |
| `chem_states_checkpoint_roundtrip` | Vec de ChemStates con estado ionizado |
| `sph_fields_in_checkpoint_jsonl` | Integración Phase 105 + 106 |
| `full_checkpoint_all_state` | Checkpoint completo: partículas + BHs + chem |

## Impacto

- Las corridas largas con AGN y EoR ahora pueden reanudarse sin perder estado acumulado.
- La masa de los BH y las fracciones de ionización se preservan entre sesiones.
