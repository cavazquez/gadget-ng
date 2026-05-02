# Documentación de gadget-ng

Punto de entrada a las guías mantenidas en este directorio.

| Guía | Contenido |
|------|-----------|
| [getting-started.md](getting-started.md) | Instalación, primeros ejemplos, tabla por perfil |
| [user-guide.md](user-guide.md) | Opciones TOML, MPI, GPU, formatos de salida, unidades |
| [architecture.md](architecture.md) | Crates, solvers, MPI/SFC+LET, I/O |
| [from-gadget4.md](from-gadget4.md) | Migración conceptual desde GADGET-4 |
| [physics-roadmap.md](physics-roadmap.md) | Física implementada y línea temporal |
| [runbooks/validation-vs-gadget4-reference.md](runbooks/validation-vs-gadget4-reference.md) | Validación P(k) vs referencias |
| [runbooks/mpi-cluster.md](runbooks/mpi-cluster.md) | MPI en clúster |

Scripts útiles: [`scripts/validate_example_configs.sh`](../scripts/validate_example_configs.sh) ejecuta `gadget-ng config --config` sobre `examples/*.toml` y sobre configs listadas en el propio script (`validation_128*.toml`, `production_256*.toml`, `eor_test.toml`, …).
