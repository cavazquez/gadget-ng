# Oleada 13 — stepping smokes in-lib (TreePM, SPH, cosmo, checkpoint, in-situ)

**Fecha:** 2026-05  
**Línea base medida (post oleada 12):** **59,60%** (11 797 / 19 792 líneas)

## Objetivo

Subir cobertura de `engine/stepping/mod.rs` migrando los smokes de alto ROI desde `tests/lib_stepping_smokes.rs` a `#[cfg(test)]` in-lib.

| Smoke | Rama cubierta |
|-------|----------------|
| TreePM | solver híbrido |
| Plummer + Barnes–Hut | IC no retícula + árbol |
| Cosmo PM | integración cosmológica |
| Yoshida-4 / hierarchical | integradores alternativos |
| SPH / MHD / RT+reionización | hidrodinámica + radiación |
| SIDM / f(R) / WDM | extensiones de física |
| Checkpoint + resume | persistencia mid-run |
| snapshot_interval / final | I/O de frames |
| In-situ basic + extended + AGN | análisis on-the-fly |

## Verificación

```bash
cargo test -p gadget-ng-cli
cargo fmt --all
cargo clippy -p gadget-ng-cli -- -D warnings
cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5
```

## Meta

+2–4 pp sobre 59,60% → **~62–64%** global `--lib`; `stepping/mod.rs` de ~142/1840 hacia >500/1840.

## Medición (2026-05-22)

**62,48%** (12 367 / 19 792 líneas, **+2,87 pp** vs 59,60% post-oleada 12).

| Módulo | Antes → Después | Notas |
|--------|-----------------|-------|
| `engine/stepping/mod.rs` | 142/1840 → **345/1840** (+11 pp local) | 20 smokes in-lib |
| `engine/stepping/context.rs` | — → **172/344** | SPH/RT/cosmo paths |
| `engine/gravity.rs` | — → **44/114** | TreePM, BH, Plummer |
| `engine/diagnostics.rs` | — → **69/83** | rebalance + momentos |
| `engine/checkpoint.rs` | — → **58/63** | resume end-to-end |

Ramas aún sin cubrir en `stepping/mod.rs`: MPI/SFC-LET, GPU híbrido, dominios slab legacy, paths de error poco frecuentes.

## Fuera de alcance

- Eliminar smokes de integración en `tests/` (se mantienen como red de seguridad).
- MPI / GPU bajo tarpaulin.
