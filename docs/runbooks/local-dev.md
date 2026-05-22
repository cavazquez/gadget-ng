# Runbook: desarrollo local

## Requisitos

- Rust estable (`rustup`) con `rustfmt` y `clippy`.
- Para MPI: OpenMPI o MPICH (`libopenmpi-dev`, `openmpi-bin` en Debian/Ubuntu).
- Para `./scripts/check.sh` (usa `clippy`/`test` con **todas** las features): `libhdf5-dev` en Debian/Ubuntu (enlazado por el crate `hdf5`).

## Comandos

```bash
cargo build
cargo test
cargo run -p gadget-ng-cli -- config --config experiments/nbody/mvp_smoke/config/default.toml
cargo run -p gadget-ng-cli -- stepping --config experiments/nbody/mvp_smoke/config/default.toml --out experiments/nbody/mvp_smoke/runs/local --snapshot
```

## Calidad

```bash
./scripts/check.sh
```

## MPI local

```bash
cargo build --features mpi
./scripts/mpi/run_smoke.sh
./scripts/validation/compare_serial_mpi.sh
```

Variable opcional: `MPIRUN` (por defecto `mpiexec`).

## Cobertura (CI / Codecov)

El job de tarpaulin **no** corre en cada push a `main`; está en `.github/workflows/coverage.yml`.

**Local:**

```bash
cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5
```

**Publicar a Codecov (tag en el commit a medir):**

```bash
git tag coverage/2026-05-22
git push origin coverage/2026-05-22
```

También acepta tags `coverage-*` (p. ej. `coverage-v0.3.0`), el botón **Run workflow** en Actions, y un cron semanal (domingo 06:00 UTC).

Si `Coverage (tarpaulin)` figuraba como check obligatorio en branch protection, quitarlo de las reglas de `main`.
