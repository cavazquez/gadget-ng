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

## Cómo reproducir

```bash
./scripts/check.sh
./scripts/validation/compare_serial_mpi.sh
```

## Limitaciones

- \(O(N^2)\): tamaños grandes solo para estudios de correctitud, no de escalado fuerte débil en HPC real.
- Sin comparación contra GADGET-4 binario (proyecto independiente).
