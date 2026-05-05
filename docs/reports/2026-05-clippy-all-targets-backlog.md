# Clippy all-targets backlog (2026-05)

Estado actual: `cargo clippy --workspace --all-targets -- -D warnings` sigue en modo advisory.

## Deuda detectada y reducida en esta iteración

- Corregidos inicializadores incompletos de `RunConfig` en benches de:
  - `gadget-ng-core`
  - `gadget-ng-pm`
  - `gadget-ng-tree`
  - `gadget-ng-treepm`
- Corregidos warnings deprecados/unused en benches MHD.
- Corregidos `approx_constant` (`TAU`) en benches/tests.
- Corregidos `needless_range_loop` en test CUDA PM.
- Corregido `field_reassign_with_default` en tests IO.
- Corregido `ptr_arg` y `unused variable` en tests physics.

## Deuda restante (snapshot)

- Benches/tests adicionales aún fallan intermitentemente bajo `--all-targets` en crates no bloqueantes.
- El pipeline bloqueante de CI se mantiene en:
  - `cargo clippy --workspace -- -D warnings`
- El pipeline advisory se mantiene en:
  - `cargo clippy --workspace --all-targets -- -D warnings`

## Criterio para graduar a bloqueante

1. `clippy-all-targets` pasa en `main` por al menos 2 semanas consecutivas.
2. Se migra el job advisory a bloqueante en `ci.yml`.
3. Se mantiene el workflow `physics-validation` para detectar regresiones lentas.
