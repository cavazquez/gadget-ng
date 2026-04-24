# Phase 101 — Fix softening comóvil → físico

**Fecha:** 2026-04-23  
**Estado:** ✅ Completado

## Problema

En los bucles cosmológicos del engine, algunos paths usaban `eps2 = eps2_base` (softening comóvil constante) en lugar de `eps2_at(a_current)` (softening físico que decrece con `a`). Esto afecta corridas largas con `physical_softening = true`.

**Bug conocido desde Phase 42:** el path que combina TreePM SR + PM slab (loop ~L2053) no recalculaba `eps2` por paso, usando el valor inicial incluso para `a << 1`.

## Corrección

### `crates/gadget-ng-cli/src/engine.rs`
- Loop L2121 (TreePM SR + slab cosmológico): agregado `let eps2 = eps2_at(a_current)` justo después de `let g_cosmo = gravity_coupling_qksl(...)`.
- Los loops L1522 y L1706 ya tenían la corrección.
- Los loops newtonianos (L3176, L3288, L3353) no se modificaron — usan `g` no `g_cosmo`, por lo que `physical_softening` no aplica.

### `crates/gadget-ng-core/src/config.rs`
- Nuevo método `softening_warnings() -> Vec<&'static str>`.
- Detecta `physical_softening = true` sin `cosmology.enabled = true` (combinación sin efecto).

## Comportamiento

| Configuración | Comportamiento |
|---|---|
| `physical_softening = false` (default) | `eps2 = softening²` constante |
| `physical_softening = true` + cosmología | `eps2 = (softening/a)²` por paso |
| `physical_softening = true` sin cosmología | `eps2 = softening²` + advertencia |

## Tests (`crates/gadget-ng-physics/tests/phase101_softening.rs`)

| Test | Descripción | Estado |
|------|-------------|--------|
| `softening_physical_scales_with_a` | ε_com = ε_phys/a | ✅ |
| `softening_comovil_constant_with_a` | legacy: constante | ✅ |
| `softening_physical_equals_comoving_at_a1` | a=1 → ε igual | ✅ |
| `softening_physical_larger_at_early_times` | monotonía con a | ✅ |
| `softening_warnings_physical_without_cosmo` | advertencia inválida | ✅ |
| `softening_warnings_none_for_valid_configs` | sin advertencias | ✅ |

**Total: 6/6 tests pasan**
