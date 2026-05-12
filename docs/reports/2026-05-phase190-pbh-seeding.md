# Phase 190 - PBH Seeding para SMBHs tempranos

## Objetivo

Agregar una ruta determinista para inicializar primordial black holes (PBHs)
masivos como semillas ligeras de SMBHs a alto redshift, motivada por escenarios
JWST con agujeros negros sobredimensionados en galaxias tempranas.

## Implementación

- Configuración en `[sph.agn]`:
  - `pbh_seeding_enabled`
  - `pbh_n_seeds`
  - `pbh_m_seed`
  - `pbh_min_host_mass`
  - `pbh_seed`
  - `pbh_host_kind`
- Nueva API `gadget_ng_sph::seed_primordial_black_holes`.
- Selección reproducible de hosts con hash estable de `(pbh_seed, global_id)`.
- Hosts restringidos a partículas no-gas, para no interferir con el estado SPH.
- Las PBHs se integran al vector existente de `BlackHole`, por lo que heredan:
  - acreción Bondi,
  - feedback quasar/radio,
  - spin Kerr,
  - mergers y recoil.

## Ejemplo TOML

```toml
[sph]
enabled = true

[sph.agn]
enabled = true
pbh_seeding_enabled = true
pbh_n_seeds = 8
pbh_m_seed = 1e3
pbh_min_host_mass = 0.0
pbh_seed = 190
pbh_host_kind = "dark_matter"
initial_spin = 0.0
```

## Overrides por CLI

`gadget-ng stepping` acepta overrides directos para explorar esta física sin
editar el TOML:

```bash
gadget-ng stepping \
  --config examples/pbh.toml \
  --out runs/pbh \
  --pbh-seeding \
  --pbh-n-seeds 8 \
  --pbh-m-seed 1e3 \
  --pbh-min-host-mass 0.0 \
  --pbh-seed 190 \
  --pbh-host-kind dark_matter
```

Valores de `--pbh-host-kind`: `dark_matter`, `star`, `collisionless`.

## Validación

Tests unitarios en `gadget-ng-sph` verifican:

- reproducibilidad bit-estable de la selección de hosts,
- exclusión de partículas de gas,
- no duplicación de PBHs al reanudar con BHs ya existentes.
