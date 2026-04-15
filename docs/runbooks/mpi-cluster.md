# Runbook: clúster MPI

## Compilación

En el nodo de login o en un job de compilación:

```bash
module load openmpi   # ejemplo; usar el módulo del centro
cargo build --release --features mpi
```

El binario resultante es `target/release/gadget-ng`. Debe enlazar contra la misma familia MPI que usará `mpiexec`/`srun` en tiempo de ejecución.

## Ejemplo (Slurm, ilustrativo)

```bash
srun -n 128 ./target/release/gadget-ng stepping \
  --config /ruta/absoluta/a/experiments/nbody/mvp_smoke/config/default.toml \
  --out /ruta/absoluta/a/experiments/nbody/mvp_smoke/runs/cluster_001 \
  --snapshot
```

Ajustar partición, cuenta, y límites de tiempo según la política del centro. **No** se asumen rutas del host en el código: siempre pasar `--config` y `--out` explícitos (o variables `GADGET_NG_*` documentadas en figment).

## Salidas

- `diagnostics.jsonl`: una línea JSON por paso (energía cinética total reducida).
- `snapshot_final/`: solo en el rango 0 si se pasa `--snapshot` (gather global).
