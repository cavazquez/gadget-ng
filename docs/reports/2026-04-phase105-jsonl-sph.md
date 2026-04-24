# Phase 105 — JSONL con campos SPH

**Fecha**: 2026-04-23  
**Crate principal**: `gadget-ng-io`  
**Archivo clave**: `crates/gadget-ng-io/src/snapshot.rs`

## Problema

`ParticleRecord` no persistía `internal_energy`, `smoothing_length` ni `ptype`. Al escribir
snapshots JSONL de simulaciones con gas SPH, todo el estado termodinámico se perdía. Los
análisis post-proceso y los checkpoints dependían de estos campos para correcta reproducción.

## Solución

Se extendió `ParticleRecord` con tres campos opcionales usando `#[serde(default)]` para
garantizar compatibilidad retroactiva con archivos pre-Phase-105:

```rust
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ParticleRecord {
    // campos existentes ...
    #[serde(default)] pub internal_energy: f64,
    #[serde(default)] pub smoothing_length: f64,
    #[serde(default)] pub ptype: ParticleType,
}
```

- `From<&Particle>` mapeaestados termodinámicos completos.
- `into_particle()` restaura `internal_energy`, `smoothing_length` y `ptype`.
- JSONL antiguos sin los campos se leen con defaults `(0.0, 0.0, DarkMatter)`.

## Tests

Archivo: `crates/gadget-ng-physics/tests/phase105_jsonl_sph.rs` (6 tests)

| Test | Descripción |
|------|-------------|
| `gas_sph_fields_survive_roundtrip` | Gas roundtrip preserva u y h |
| `dm_particle_ptype_preserved` | DM mantiene ptype=DarkMatter y SPH=0 |
| `mixed_snapshot_roundtrip` | Snapshot mixto DM+gas |
| `legacy_jsonl_backward_compat` | JSONL viejo sin SPH fields → defaults |
| `particle_record_serializes_sph_fields` | JSON incluye campos SPH |
| `particle_record_into_particle_restores_sph` | Restauración correcta |

## Impacto

- Los checkpoints JSONL ahora son completos para simulaciones SPH.
- Habilitó Phase 106 (restart con SPH state completo).
- Compatible con todos los snapshots existentes (backward compat).
