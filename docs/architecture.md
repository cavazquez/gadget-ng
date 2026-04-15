# Arquitectura de **gadget-ng**

Implementación nueva en Rust inspirada **conceptualmente** en GADGET-4 ([sitio oficial](https://wwwmpa.mpa-garching.mpg.de/gadget4/), paper: Springel et al., *Simulating cosmic structure formation with the GADGET-4 code*, MNRAS 506, 2871, 2021; manual PDF enlazado desde el sitio). **No se reutiliza ni copia código** de GADGET.

## Qué se toma de GADGET-4

- **Separación modular** entre estado de partículas, cálculo de fuerzas, integración temporal, I/O de snapshots y capa de comunicación (MPI), análoga a la organización descrita en el paper/manual.
- **N-body colisionless** con **suavizado Plummer** en el denominador de la fuerza pareada \((r^2+\varepsilon^2)^{3/2}\), práctica estándar en códigos cosmológicos.
- **Integración leapfrog** en forma **kick–drift–kick (KDK)** sincronizada con paso global \(\Delta t\) en el MVP (equivalente al núcleo simple del esquema colisionless; sin paso jerárquico local como en GADGET-4 avanzado).
- **Paralelismo MPI** con **descomposición por bloques contiguos de `global_id`** y **reunión global** de posiciones/masas (`MPI_Allgatherv` vía `mpi` crate) antes de evaluar la fuerza directa en cada subconjunto local — patrón conceptualmente alineado con “acumular estado global para el solver”, simplificado frente a árboles y halos de vecinos.

## Qué se simplifica

- **Gravedad**: solo **fuerza directa \(O(N^2)\)** con `DirectGravity`; sin Tree/TreePM/FMM (fase posterior; interfaz `GravitySolver` ya permite extender).
- **Dominio**: sin cosmología, SPH, ni I/O binario legacy; snapshots **JSONL** versionados con `provenance.json`.
- **MPI**: sin híbrido MPI+OpenMP del paper de GADGET-4; un solo hilo por rango en el MVP.
- **Configuración**: TOML + variables de entorno `GADGET_NG_*` (figment), por legibilidad y alineación con el ecosistema Rust.

## Qué se descarta (por ahora)

- Hidrodinámica, cosmología obligatoria en el core, árboles de fusión, y demás componentes de GADGET-4 no necesarios para un **MVP N-body** verificable.
- **GPU** y **NetCDF**: features `gpu` / `netcdf` con *stubs* documentados (ver `gadget-ng-io`).

## Crates

| Crate | Rol |
|--------|-----|
| `gadget-ng-core` | `Vec3`, `Particle`, `RunConfig`, IC sintéticas, `DirectGravity` / `GravitySolver` |
| `gadget-ng-integrators` | `leapfrog_kdk_step` (KDK, `FnMut` para aceleraciones) |
| `gadget-ng-parallel` | `ParallelRuntime`: `SerialRuntime`, `MpiRuntime` (`feature = "mpi"`) |
| `gadget-ng-io` | Snapshots + `Provenance` |
| `gadget-ng-cli` | Binario `gadget-ng` (`config`, `stepping`, `snapshot`) |

## Flujo de `stepping` (MPI)

```mermaid
sequenceDiagram
  participant R as CadaRango
  participant MPI as MPI_Allgatherv
  participant G as DirectGravity
  participant L as LeapfrogKDK
  R->>MPI: posiciones_masas_locales
  MPI->>R: estado_global
  R->>G: aceleraciones_indices_locales
  R->>L: kick_drift_kick
```

## Limitaciones del MVP

- Escalado asintótico \(O(N^2)\) no está pensado para producción masiva; el objetivo es **arquitectura limpia**, **MPI real** y **cadena de validación** reproducible.
- La paridad serial/MPI se valida numéricamente con tolerancia explícita en [experiments/nbody/mvp_smoke/docs/validation.md](../experiments/nbody/mvp_smoke/docs/validation.md).
