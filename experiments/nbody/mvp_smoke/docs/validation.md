# Validación del experimento `mvp_smoke`

## Objetivo

Demostrar reproducibilidad numérica básica y paridad **serial vs MPI** para el MVP de gravedad directa + leapfrog KDK.

## Tolerancias declaradas

| Prueba | Magnitud | Tolerancia | Justificación breve |
|--------|-----------|------------|----------------------|
| Momento lineal (red cúbica) | \(\|\sum_i m_i \mathbf{v}_i\|\) | `< 5e-11` tras 30 pasos | Fuerzas internas antisimétricas por pares + suma global idéntica; errores de redondeo `f64` acumulados. Test: `momentum_lattice.rs`. |
| Energía armónica (integrador) | máximo \(\|E(t)\|\) vs \(\|E(0)\|\) | `< 2|E(0)| + 1e-6` en 500 pasos | Esquema KDK simpléctico con \(\Delta t\) pequeño; cota conservadora frente a deriva numérica. Test: `harmonic_oscillator_energy.rs`. |
| Aceleración sub-bloque vs global | \(\|\mathbf{a}_{\text{split}}-\mathbf{a}_{\text{full}}\|\) por partícula | `< 1e-15` | Misma aritmética de suma ordenada \(j=0..N-1\). Test: `split_vs_full_accel.rs`. |
| Paridad serial / MPI (snapshot) | \(\max\) de \(\|q_{\text{serial}}-q_{\text{mpi}}\|\) sobre `px,py,pz,vx,vy,vz,mass` | `≤ 1e-12` | Mismo algoritmo y orden de evaluación; MPI solo replica estado global antes de la fuerza. Script: `scripts/validation/compare_serial_mpi.sh`. |
| Barnes–Hut vs directo (`theta = 0`) | \(\|\mathbf{a}_{\text{BH}}-\mathbf{a}_{\text{dir}}\|\) / \(\|\mathbf{a}_{\text{dir}}\|\) (red 27) | `≲ 1e-11` | Sin MAC: recorrido equivalente al directo salvo redondeo. Test: `gadget-ng-tree` `regression_vs_direct.rs`. |
| Barnes–Hut precisión (lattice 125) | error relativo medio en partículas con \(\|\mathbf{a}_{\text{dir}}\| > 0.01\,\max_j\|\mathbf{a}_{\text{dir},j}\|\) | `< 3 %` con `theta = 0.5`; `< 1 %` con `theta = 0.25` | Monopolo puro; en lattice casi simétrico muchas \(\mathbf{a}_{\text{dir}}\) casi nulas y el error relativo carece de sentido sin filtrar. Tests: `regression_vs_direct.rs`. |
| Barnes–Hut + stepping | energía cinética finita / acotada | `KE < 1e6` tras 50 pasos (lattice en reposo) | Comprueba que el integrador no explota con BH aproximado. Test: `bh_stepping_energy.rs`. |
| Paridad serial / MPI con BH | misma métrica que arriba sobre snapshot | `≤ 1e-12` | Cada rango reconstruye el mismo octree sobre el estado global allgathered. Script: `scripts/validation/compare_serial_mpi_bh.sh` y config `config/barnes_hut.toml`. |
| I/O JSONL | `meta.json` + `particles.jsonl` coherentes con partículas de prueba | (tests unitarios) | `gadget-ng-io` `snapshot::tests::jsonl_roundtrip_particles_match`. |
| I/O bincode | roundtrip `particles.bin` vs partículas | idéntico | Requiere `cargo test -p gadget-ng-io --features bincode`. |
| I/O HDF5 | `Header/NumPart_Total`, `PartType1/Coordinates`, etc. | coherente con 2 partículas de prueba | Requiere `cargo test -p gadget-ng-io --features hdf5` y `libhdf5-dev`. |

**Nota:** Los scripts `compare_serial_mpi*.sh` leen `particles.jsonl`; para paridad con HDF5 habría que extender el comparador Python o usar `snapshot_format = "jsonl"` en esas configs.

**Nota sobre Rayon (`[performance] deterministic = false`):** cuando se activa el modo paralelo, el orden de acumulación de fuerzas entre partículas puede variar entre ejecuciones y entre rangos MPI. En consecuencia, los tests de paridad serial/MPI (`compare_serial_mpi.sh`) **no son aplicables** en ese modo; solo son válidos con `deterministic = true` (default). Los benchmarks de rendimiento se ejecutan independientemente con `cargo bench`.

## Cómo reproducir

```bash
./scripts/check.sh
./scripts/validation/compare_serial_mpi.sh
./scripts/validation/compare_serial_mpi_bh.sh
```

## Limitaciones

- \(O(N^2)\): tamaños grandes solo para estudios de correctitud, no de escalado fuerte débil en HPC real.
- Sin comparación contra GADGET-4 binario (proyecto independiente).
