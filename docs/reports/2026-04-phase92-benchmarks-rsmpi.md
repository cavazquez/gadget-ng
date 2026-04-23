# Phase 92 — Benchmarks formales MPI scaling + P(k) vs GADGET-4

**Fecha**: 2026-04-23  
**Scripts creados**: `scripts/bench_mpi_scaling.sh`, `docs/notebooks/bench_pk_vs_gadget4.py`

## Objetivo

Añadir infraestructura de benchmarking formal para:
1. Medir el scaling MPI (strong/weak) de gadget-ng con `mpirun`.
2. Comparar cuantitativamente el espectro de potencia P(k) contra valores de referencia de GADGET-4.
3. Verificar que `MpiRuntime` en `gadget-ng-parallel` está completamente implementado con `rsmpi` real.

## Nuevo script: `scripts/bench_mpi_scaling.sh`

Script bash para benchmarks de scaling MPI:

```bash
bash scripts/bench_mpi_scaling.sh [--weak | --strong] [--n-ranks "1 2 4 8"]
```

**Protocolo**:
1. Compila `gadget-ng-cli` en `--release` con feature `mpi` (o sin MPI como fallback).
2. Para cada número de ranks (default: 1, 2, 4, 8):
   - Crea un directorio de salida temporal.
   - Lanza la simulación con `mpirun --oversubscribe -n N`.
   - Mide el tiempo de pared con `date +%s%N`.
3. Genera `bench_results/scaling_<timestamp>.json` con resultados.
4. Opcional: imprime tabla de speedup y eficiencia paralela.

**Output JSON**:
```json
[
  { "n_ranks": 1, "scaling_mode": "strong", "elapsed_s": 12.34, "timestamp": "..." },
  { "n_ranks": 2, "scaling_mode": "strong", "elapsed_s": 6.78, "timestamp": "..." }
]
```

## Nuevo script: `docs/notebooks/bench_pk_vs_gadget4.py`

Script Python para comparación cuantitativa P(k):

```bash
python3 docs/notebooks/bench_pk_vs_gadget4.py \
    --insitu-dir runs/validation/insitu/ \
    --output bench_results/pk_comparison.json \
    --plot bench_results/pk_comparison.png
```

**Funcionalidades**:
- `eh_transfer_function(k, cosmo)`: función de transferencia analítica de Eisenstein & Hu (1998).
- `eh_power_spectrum(k_arr, cosmo)`: P(k) = A k^ns T(k)² sin normalizar.
- `compute_sigma8_from_pk(k_arr, pk_arr, r=8)`: integración numérica de sigma_8 con ventana top-hat.
- `compare_pk(snap, cosmo)`: métricas cuantitativas (ratio P(k), error en sigma_8).

**Valores de referencia GADGET-4** (Springel et al. 2021):
- `sigma_8(z=0) = 0.811 ± 0.010` para N=128³, L=100 Mpc/h, Planck18.
- `P(k=0.1 h/Mpc, z=0) ≈ 3000 (h/Mpc)³` (estimado).

**Output JSON**:
```json
{
  "metrics_z0": {
    "sigma8_sim": 0.798,
    "sigma8_gadget4_ref": 0.811,
    "sigma8_error_percent": 1.6,
    "pk_ratio_mean": 0.97,
    "pk_ratio_std": 0.08
  }
}
```

## Verificación: MpiRuntime rsmpi completo

Se verificó que `crates/gadget-ng-parallel/src/mpi_rt.rs` implementa el protocolo
`ParallelRuntime` completo con rsmpi real (sin stubs):

| Método | Implementación |
|--------|---------------|
| `allgatherv_state` | `MPI_Allgatherv` via `all_gather_varcount_into` |
| `root_gather_particles` | `MPI_Gatherv` via `gather_varcount_into_root` |
| `allreduce_sum_f64` | `MPI_Allreduce` via `all_reduce_into(SystemOperation::sum())` |
| `allreduce_min_f64` | `MPI_Allreduce` via `all_reduce_into(SystemOperation::min())` |
| `allreduce_max_f64` | `MPI_Allreduce` via `all_reduce_into(SystemOperation::max())` |
| `exchange_domain_by_x/z` | `MPI_Send`/`MPI_Recv` patrón odd-even |
| `exchange_halos_sfc` | `MPI_Alltoallv` + AABB allgather |
| `alltoallv_f64_overlap` | `Isend`/`Irecv` no-bloqueante con overlap de cómputo |
| `alltoallv_f64_subgroup` | `MPI_Alltoallv` en sub-comunicador |
| `exchange_halos_3d_periodic` | `MPI_Alltoallv` con distancia periódica 3D |

El único `eprintln!` restante en `root_eprintln` es intencional (logging desde rank 0).

## Directorio bench_results/

```
bench_results/
├── scaling_20260423_123456.json    # resultados de bench_mpi_scaling.sh
└── pk_comparison.json              # resultados de bench_pk_vs_gadget4.py
```

(Los archivos de resultados no se incluyen en el repositorio via `.gitignore`.)
