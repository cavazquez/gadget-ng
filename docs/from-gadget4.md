# gadget-ng para usuarios de GADGET-4

Este documento orienta a quien ya conoce el [manual y el paper de GADGET-4](https://wwwmpa.mpa-garching.mpg.de/gadget4/) (Springel et al., MNRAS 506, 2871, 2021). **gadget-ng no es compatible con los ficheros de parámetros de GADGET**; la configuración es **TOML** y variables de entorno con prefijo **`GADGET_NG_*`** (vía `figment`).

## Filosofía

| GADGET-4 | gadget-ng |
|----------|-----------|
| Fichero de parámetros estilo `param.txt` | `[secciones]` en TOML + env `GADGET_NG_*` |
| Binario único con árbol de opciones histórico | Workspace Rust multi-crate; features Cargo para HDF5, MPI, GPU, etc. |
| Referencia de facto para cosmología simulada | Inspiración conceptual y benchmarks de referencia, **sin paridad binaria** |

## Tabla de equivalencias (conceptuales)

| Idea GADGET / manual | Dónde en gadget-ng |
|---------------------|-------------------|
| `TreePM`, PM grid, softening | `[gravity]` → `solver = "tree_pm"` \| `"pm"`, `pm_grid_size`, `theta`, `r_split`, softening en `[simulation]` / `[gravity]` |
| Constante gravitatoria / cosmología | `[cosmology]` (`omega_m`, `omega_lambda`, `h0`, `a_init`, `auto_g`, …) |
| ICs Zel'dovich, 2LPT, Eisenstein–Hu | `[initial_conditions].kind = { zeldovich = { … } }` |
| Snapshots HDF5 Legados | `[output].snapshot_format = "hdf5"` (requiere feature `hdf5`) |
| MPI, dominio, árbol distribuido | `[performance]`, MPI en CLI; por defecto SFC+LET para BH multi-rank — ver [user-guide.md](user-guide.md#árbol-distribuido-mpi) |
| Block timesteps / jerárquico | `[timestep]` (`hierarchical`, `eta`, `max_level`, …) |

## Unidades

Las convenciones **comoving vs físico**, factores \(h\), y bloque opcional **`[units]`** están descritos en [user-guide.md](user-guide.md) (sección de unidades y sistema físico). Al migrar un experimento de GADGET, revisa **`box_size`**, **`softening`** y **`cosmology.h0`** / **`auto_g`** para que la interpretación de \(G\) y escalas sea coherente.

## Snapshots

- **HDF5** “estilo GADGET-4”: grupos `Header` / `PartType1`, datasets estándar; adecuado para **yt**, **pynbody**, etc.
- **JSONL** / **bincode** / **msgpack** / **NetCDF**: útiles para tests, pipelines ligeros o interoperabilidad; véase [architecture.md](architecture.md#io-de-snapshots).

Detalle de campos y provenance: crate `gadget-ng-io` y la enum `SnapshotFormat` en el código.

## Referencias y comparaciones cuantitativas

- Paper y sitio MPA: enlaces en el README del repo.
- Comparaciones de **P(k)**, \(\sigma_8\) u otras métricas frente a tablas tipo “referencia GADGET-4” son **orientativas**: dependen de resolución, ICs, tiempo de salida y modo MPI. Para un flujo reproducible local, usa el [runbook de validación](runbooks/validation-vs-gadget4-reference.md).

## Ejemplos TOML de partida

En [`examples/`](../examples/):

- [`gadget_like_treepm_lcdm.toml`](../examples/gadget_like_treepm_lcdm.toml) — caja periódica, TreePM, cosmología \(\Lambda\)CDM (plantilla corta).
- [`gadget_like_bh_lcdm.toml`](../examples/gadget_like_bh_lcdm.toml) — Barnes–Hut cosmológico **aperiódico** (conceptualmente “solo árbol”, sin PM periódico).
