# Phase 59 — Restart/Checkpoint robusto

**Fecha:** abril 2026  
**Crate principal:** `gadget-ng-cli`  
**Archivo de test:** `crates/gadget-ng-physics/tests/phase59_checkpoint_continuity.rs`

---

## Contexto

El sistema de checkpoint ya existía (Phase 21): guarda posiciones, velocidades y
estado jerárquico en `<out>/checkpoint/`. Phase 59 lo audita y añade:

1. Campo `sfc_state_saved: bool` a `CheckpointMeta` (informativo).
2. Test de **continuidad bit-a-bit** que verifica que reanudar desde checkpoint
   produce trayectorias físicamente idénticas.

---

## Auditoría

### `SfcDecomposition` no se serializa — correcto

El `SfcDecomposition` se reconstruye desde las posiciones restauradas en cada path:
- Path jerárquico+LET: reconstruye en la inicialización previa al bucle.
- Path cosmológico SFC+LET: ídem.
- Path SFC+LET BarnesHut: ídem.

No hay estado SFC "oculto" que se pierda al hacer restart.

### Factor de escala `a_current`

Guardado en `CheckpointMeta.a_current`. Al restaurar:
- Se usa directamente como `a_current` inicial del bucle.
- El coupling `G·a³` se recalcula en el primer paso desde este valor.

### `HierarchicalState`

Se serializa a `checkpoint/hierarchical_state.json` cuando `has_hierarchical_state = true`.
Incluye niveles de timestep y velocidades intermedias de cada partícula.

---

## Principio de continuidad bit-a-bit

Para que un restart sea **bit-a-bit idéntico** se requiere:

1. Guardar posiciones y velocidades exactas (ya implementado).
2. Recomputar fuerzas desde posiciones restauradas antes del primer paso.
   Esto es equivalente a continuar el KDK desde el punto de guarda.
3. No hay estado hidden en el integrador leapfrog KDK más allá de `(x, v)`.

```
Corrida continua:  x₀ → x₁₀ → x₂₀
Corrida dividida:  x₀ → x₁₀ (clone) → recompute forces → x₂₀'
Resultado:         x₂₀ ≡ x₂₀' (bit-a-bit)
```

---

## Tests

| Test | Descripción |
|------|-------------|
| `phase59_checkpoint_continuity_bitexact` | N=8³ PM, 20 pasos: continua vs restart a 10 → max\|Δx\| = 0 |
| `phase59_stale_forces_produce_different_result` | Con scratch=0 al restart: resultado puede diferir |

**Resultado:** `max|Δx| = 0.00e0`, `max|Δv| = 0.00e0` — continuidad bit-a-bit confirmada.
