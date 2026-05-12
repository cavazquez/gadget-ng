# Documentación de gadget-ng

Punto de entrada a las guías mantenidas en este directorio.

| Guía | Contenido |
|------|-----------|
| [getting-started.md](getting-started.md) | Instalación, primeros ejemplos, tabla por perfil |
| [user-guide.md](user-guide.md) | Opciones TOML, MPI, GPU, formatos de salida, unidades |
| [architecture.md](architecture.md) | Crates, solvers, MPI/SFC+LET, I/O |
| [from-gadget4.md](from-gadget4.md) | Migración conceptual desde GADGET-4 |
| [physics-roadmap.md](physics-roadmap.md) | Física implementada y línea temporal |
| [roadmap.md](roadmap.md) | Estado global de fases, actualmente Phases 1–185 |
| [roadmap-physics-extensions.md](roadmap-physics-extensions.md) | Cartera Physics Extensions cerrada en Phases 177–185 |
| [runbooks/validation-vs-gadget4-reference.md](runbooks/validation-vs-gadget4-reference.md) | Validación P(k) vs referencias |
| [runbooks/mpi-cluster.md](runbooks/mpi-cluster.md) | MPI en clúster |

Scripts útiles: [`scripts/validate_example_configs.sh`](../scripts/validate_example_configs.sh) ejecuta `gadget-ng config --config` sobre `examples/*.toml` y sobre configs listadas en el propio script (`validation_128*.toml`, `production_256*.toml`, `eor_test.toml`, …).

## Reportes recientes

| Phase | Reporte |
|-------|---------|
| 181 | [RT multifrecuencia + Lyman-Werner](reports/2026-05-phase181-rt-multifrequency-lw.md) |
| 182 | [Polvo IR / emisión térmica](reports/2026-05-phase182-dust-ir-thermal-emission.md) |
| 183 | [AGN spin + mergers](reports/2026-05-phase183-agn-spin-mergers.md) |
| 184 | [Warm / fuzzy dark matter](reports/2026-05-phase184-wdm-fdm.md) |
| 185 | [f(R) no lineal en malla](reports/2026-05-phase185-fr-nonlinear-mesh.md) |
