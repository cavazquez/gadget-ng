# Arquitectura de **gadget-ng**

Implementación nueva en Rust inspirada **conceptualmente** en GADGET-4 ([sitio oficial](https://wwwmpa.mpa-garching.mpg.de/gadget4/), paper: Springel et al., *Simulating cosmic structure formation with the GADGET-4 code*, MNRAS 506, 2871, 2021; manual PDF enlazado desde el sitio). **No se reutiliza ni copia código** de GADGET.

## Qué se toma de GADGET-4

- **Separación modular** entre estado de partículas, cálculo de fuerzas, integración temporal, I/O de snapshots y capa de comunicación (MPI), análoga a la organización descrita en el paper/manual.
- **N-body colisionless** con **suavizado Plummer** en el denominador de la fuerza pareada \((r^2+\varepsilon^2)^{3/2}\), práctica estándar en códigos cosmológicos.
- **Integración leapfrog** en forma **kick–drift–kick (KDK)** sincronizada con paso global \(\Delta t\) en el MVP (equivalente al núcleo simple del esquema colisionless; sin paso jerárquico local como en GADGET-4 avanzado).
- **Paralelismo MPI** con **descomposición por bloques contiguos de `global_id`** y **reunión global** de posiciones/masas (`MPI_Allgatherv` vía `mpi` crate) antes de evaluar el solver de gravedad en cada subconjunto local — patrón conceptualmente alineado con “acumular estado global para el solver”, simplificado frente a halos de vecinos y frente a un árbol MPI distribuido (fase futura).

## Qué se simplifica

- **Gravedad**: **fuerza directa \(O(N^2)\)** (`DirectGravity`), **Barnes–Hut monopolar** \(O(N\log N)\) (`BarnesHutGravity`), **Particle-Mesh periódico** \(O(N + N_M^3 \log N_M)\) (`PmSolver`), y **TreePM** (`TreePmSolver`, splitting Gaussiano en k-space: corto alcance con kernel erfc via octree + largo alcance con PM filtrado). Se elige vía TOML `[gravity].solver = "direct" | "barnes_hut" | "pm" | "tree_pm"`.
- **Dominio**: sin cosmología, SPH, ni I/O binario legacy; snapshots versionados con `provenance.json` y `meta.json` (véase I/O abajo).
- **MPI**: sin híbrido MPI+OpenMP del paper de GADGET-4; un solo hilo por rango en el MVP.
- **Configuración**: TOML + variables de entorno `GADGET_NG_*` (figment), por legibilidad y alineación con el ecosistema Rust.

## Qué se descarta (por ahora)

- Hidrodinámica, cosmología obligatoria en el core, árboles de fusión, y demás componentes de GADGET-4 no necesarios para un **MVP N-body** verificable.
- **GPU**: crate `gadget-ng-gpu` placeholder (sin kernels reales aún); `GpuDirectGravity` llama a `unimplemented!`. **HDF5 / bincode**: opcionales en `gadget-ng-io` (ver sección I/O).

## Crates

| Crate | Rol |
|--------|-----|
| `gadget-ng-core` | `Vec3`, `Particle`, `RunConfig`, IC sintéticas, `DirectGravity` / trait `GravitySolver`; bajo `feature = "gpu"`: `gpu_bridge` (impl `GravitySolver` para `GpuDirectGravity`, conversiones SoA↔Particle) |
| `gadget-ng-tree` | Octree + `BarnesHutGravity` (MAC `s/d` con `d` al COM; no MAC si la evaluación cae dentro de la celda del nodo) |
| `gadget-ng-pm` | Solver Particle-Mesh periódico 3D: `PmSolver` (CIC, FFT 3D `rustfft`, Poisson en k-space); `solve_forces_filtered` con filtro Gaussiano `exp(-k²·r_s²/2)` para uso desde TreePM |
| `gadget-ng-treepm` | `TreePmSolver`: PM filtrado (largo alcance, kernel erf) + octree erfc (corto alcance). `ShortRangeParams` agrupa los parámetros del kernel; `erfc_approx` (A&S 7.1.26) |
| `gadget-ng-integrators` | `leapfrog_kdk_step` (KDK global); `hierarchical_kdk_step` + `HierarchicalState` (serializable, `save`/`load` a JSON) + `aarseth_bin` (block timesteps al estilo GADGET-4) |
| `gadget-ng-parallel` | `ParallelRuntime`: `SerialRuntime`, `MpiRuntime` (`feature = "mpi"`) |
| `gadget-ng-io` | Snapshots (`SnapshotFormat`: JSONL / bincode / HDF5 / **msgpack**) + `Provenance` |
| `gadget-ng-gpu` | Layout SoA (`GpuParticlesSoA`: 8 arrays planos de f64/usize), `GpuDirectGravity` stub; sin dep sobre `gadget-ng-core` para evitar ciclo |
| `gadget-ng-cli` | Binario `gadget-ng` (`config`, `stepping`, `snapshot`) |

### I/O de snapshots

El TOML `[output] snapshot_format` usa el enum `SnapshotFormat` en [`config.rs`](../crates/gadget-ng-core/src/config.rs) (`jsonl` \| `bincode` \| `hdf5` \| `msgpack`). Siempre se escriben `meta.json` y `provenance.json` (incluyen `time`, `redshift`, `box_size` para cabeceras HDF5).

| Formato | Feature Cargo | Ficheros extra | Notas |
|--------|----------------|----------------|--------|
| JSONL | (default) | `particles.jsonl` | Una línea JSON por partícula; scripts de paridad actuales lo consumen. |
| bincode | `gadget-ng-io/bincode` | `particles.bin` | `Vec<ParticleRecord>` serializado; sin dependencias C. |
| HDF5 | `gadget-ng-io/hdf5` | `snapshot.hdf5` | Grupos `Header` / `PartType1` al estilo GADGET-4 (`Coordinates`, `Velocities`, `Masses`, `ParticleIDs`); dataset `Provenance/gadget_ng_json_utf8`. Requiere `libhdf5` en el sistema. |
| msgpack | `gadget-ng-io/msgpack` | `particles.msgpack` | `Vec<ParticleRecord>` en MessagePack (`rmp-serde`); puro Rust, compacto, interoperable con Python/R/Julia (`msgpack.unpackb`). |

**Escritura:** `gadget_ng_io::writer_for` + trait `SnapshotWriter`, o `write_snapshot_formatted` (usado por el CLI).

**Lectura:** API simétrica a la de escritura: trait `SnapshotReader` en `reader.rs` con `SnapshotData { particles, time, redshift, box_size }`.

| Lector | Struct | Fuente de datos |
|--------|--------|-----------------|
| JSONL | `JsonlReader` | `meta.json` + `particles.jsonl` |
| bincode | `BincodeReader` | `meta.json` + `particles.bin` |
| HDF5 | `Hdf5Reader` | `snapshot.hdf5` (atributos `Header/*` + datasets `PartType1/*`) |

Función de conveniencia: `read_snapshot_formatted(fmt, dir)` análoga a `write_snapshot_formatted`.

### Barnes–Hut vs FMM (decisión MVP)

Para salir de \(O(N^2)\) sin multiplicar la superficie de código, el MVP usa **monopolo por nodo** (masa total + COM) y el MAC clásico `s/d < \theta` con \(d\) la distancia al COM del nodo. **FMM** u órdenes multipolares superiores quedan reservados para cuando haga falta más precisión por celda con el mismo \(\theta\).

## Flujo de `stepping` (MPI)

```mermaid
sequenceDiagram
  participant R as CadaRango
  participant MPI as MPI_Allgatherv
  participant G as GravitySolver
  participant L as LeapfrogKDK
  R->>MPI: posiciones_masas_locales
  MPI->>R: estado_global
  R->>G: aceleraciones_indices_locales
  R->>L: kick_drift_kick
```

### Rendimiento / Paralelismo intra-rango

La sección `[performance]` del TOML controla el paralelismo dentro de cada rango MPI:

```toml
[performance]
deterministic = false   # true (default) = serial; false = Rayon activo
num_threads = 4         # opcional; None → número de CPUs lógicas
```

#### Jerarquía de solvers directos

| Solver | Condición | Kernel interno | Paridad serial/MPI |
|--------|-----------|----------------|--------------------|
| `DirectGravity` | `deterministic=true` | AoS escalar | **garantizada** |
| `SimdDirectGravity` | `simd` feature, uso directo | SoA + caché-blocking + AVX2 | determinista (no bit-idéntico a `DirectGravity`) |
| `RayonDirectGravity` | `simd` + `deterministic=false` | Rayon outer + SoA+blocking+AVX2 inner | **no garantizada** |
| `BarnesHutGravity` / `RayonBarnesHutGravity` | igual que directos | tree walk | ídem |

#### Kernel SIMD (`gravity_simd`)

El módulo `gadget_ng_core::gravity_simd` implementa:

- **SoA layout**: extrae `xs`, `ys`, `zs`, `masses` como `Vec<f64>` contiguos antes del bucle de partículas.
- **Caché-blocking** (`BLOCK_J = 64`): divide el bucle `j` en tiles de 64 elementos (~2 KB × 4 arrays × 8 B en L1).
- **Auto-vectorización AVX2+FMA**: la función `inner_blocked_avx2` lleva `#[target_feature(enable = "avx2", enable = "fma")]`; con datos SoA contiguos el compilador emite instrucciones `ymm` de 256 bits (4× f64 por ciclo).
- **Mask sin branch**: la condición `j == skip` se convierte en `0.0|1.0`, evitando saltos que impedirían SIMD.
- **Detección en runtime**: `is_x86_feature_detected!` con fallback escalar en CPUs sin AVX2.

`RayonDirectGravity` usa `accel_soa_blocked` en su bucle interno, combinando paralelismo Rayon en el eje `i` con SIMD+blocking en el eje `j`.

El paralelismo se aplica al **bucle externo** de partículas (cada partícula es independiente). El tree walk interno de BH permanece serial por partícula (los nodos del árbol son de solo lectura: `Sync` sin cambios en `Octree`).

Los benchmarks viven en `crates/gadget-ng-core/benches/direct_gravity.rs` y `crates/gadget-ng-tree/benches/bh_gravity.rs` (Criterion). Para ejecutarlos:

```bash
cargo bench -p gadget-ng-core --features gadget-ng-core/simd
cargo bench -p gadget-ng-tree --features gadget-ng-tree/simd
```

**Rayon en PM / TreePM** (`feature = "pm-rayon"` en el CLI):

| Función | Estrategia Rayon |
|---------|-----------------|
| `cic::assign_rayon` | `par_iter().fold()` con arrays locales por hilo + `reduce()` para sumar |
| `cic::interpolate_rayon` | `par_iter().map()` — lectura independiente del grid por partícula |
| `fft_poisson` (k-space) | `(0..nm³).into_par_iter().map().collect()` — cálculo `Φ̂(k)` y `F̂` independiente por celda |
| `short_range::short_range_accels` (TreePM) | `par_iter_mut().zip().for_each()` — árbol `&Octree` compartido (`Sync`) |

Activar con:
```toml
# Cargo.toml del binario:
features = ["pm-rayon"]
```

### Pasos temporales jerárquicos (block timesteps)

La sección `[timestep]` del TOML activa el esquema de block timesteps al estilo GADGET-4:

```toml
[timestep]
hierarchical = true   # false (default) = paso global uniforme
eta = 0.025           # parámetro Aarseth; dt_i = eta * sqrt(eps / |a_i|)
max_level = 6         # máx. subdivisiones; n_fine = 2^max_level sub-pasos
```

**Criterio de Aarseth** (`aarseth_bin`): el paso individual de cada partícula se
cuantiza a la potencia de 2 inmediatamente menor o igual a `dt_courant = eta * sqrt(eps / |a|)`:

```
nivel k  →  dt_i = dt_base / 2^k   (k ∈ [0, max_level])
```

**Algoritmo KDK con predictor** (`hierarchical_kdk_step`): para cada sub-paso fino `s`:

1. **START kick** para partículas que inician su paso en `t = s·fine_dt` (`s % stride(k) == 0`):
   `elapsed[i] = 0; v += a * (dt_i / 2)`
2. **Drift** de *todas* las partículas (primer orden): `x += v * fine_dt; elapsed[i] += 1`
3. **Predictor + END kick** para partículas que terminan su paso en `t = (s+1)·fine_dt`:
   - Antes de evaluar fuerzas, las posiciones de las partículas **inactivas** se mejoran temporalmente:
     `Δx_j = 0.5 * a_j * (elapsed[j] * fine_dt)²` (predictor de Störmer)
   - Se evalúan fuerzas con las posiciones predichas.
   - Se restauran las posiciones reales (`x_j -= Δx_j`).
   - `v += a_new * (dt_i / 2)`, `elapsed[i] = 0`, se reasigna el bin.

El predictor reduce el error de posición de las inactivas de O(Δt²) a O(Δt³) para la
evaluación de fuerzas, sin alterar la integración simpléctica de las activas.
`HierarchicalState` incluye `elapsed: Vec<u64>` para rastrear el tiempo desde el último kick.

`HierarchicalState` mantiene el vector de niveles fuera de `Particle` para no alterar
`PartialEq` y otros derives del struct de core.

**Snapshot de HierarchicalState:** para permitir reanudar simulaciones jerárquicas, `HierarchicalState` implementa `Serialize/Deserialize` (serde) y expone:

```rust
state.save(dir)           // escribe <dir>/hierarchical_state.json
HierarchicalState::load(dir) // carga desde <dir>/hierarchical_state.json
```

El engine guarda automáticamente el estado junto al snapshot final cuando `[timestep] hierarchical = true`.

### Solver TreePM

El crate `gadget-ng-treepm` implementa `TreePmSolver` que divide el campo gravitacional entre largo y corto alcance mediante un **splitting Gaussiano en k-space**, siguiendo el esquema estándar de los códigos cosmológicos (cf. GADGET-4):

```
F_total(r) = F_lr(r)  +  F_sr(r)

F_lr(r) = G·m/r² · erf(r / (√2·r_s))     ← PM con filtro exp(-k²·r_s²/2)
F_sr(r) = G·m/r² · erfc(r / (√2·r_s))    ← octree + kernel erfc, cutoff r_cut = 5·r_s
```

La partición `erf + erfc = 1` garantiza que `F_lr + F_sr = F_Newton`.

#### Parámetros

| Campo TOML | Descripción | Default |
|-----------|-------------|---------|
| `pm_grid_size` | Celdas por lado del grid PM | 64 |
| `r_split` | Radio de splitting (≤ 0 → auto 2.5 × cell_size) | 0.0 |

```toml
[gravity]
solver       = "tree_pm"
pm_grid_size = 64
r_split      = 0.0
```

#### Largo alcance (`gadget-ng-pm::fft_poisson::solve_forces_filtered`)

Igual que `solve_forces` pero multiplica el potencial en k-space por `exp(-k²·r_s²/2)`.
Esto corresponde a convolucionar la densidad con una Gaussiana de anchura `r_s` en espacio real.

#### Corto alcance (`gadget-ng-treepm::short_range`)

- Construye un octree con `Octree::build` del crate `gadget-ng-tree`.
- Para cada partícula activa, recorre el árbol con un **cutoff** `r_cut = 5·r_s` (fuera de r_cut, erfc < 1e-8 → fuerza nula).
- Para nodos lejanos pero dentro del cutoff: usa el **monopolo** si `half_size < 0.1·r_cut`.
- Para nodos cercanos: baja hasta las hojas (pares exactos).
- `erfc_approx(x)` implementa Abramowitz & Stegun §7.1.26 (error máx. 1.5×10⁻⁷, sin dependencias C).

### Solver Particle-Mesh (PM) periódico 3D

El crate `gadget-ng-pm` implementa `PmSolver` que resuelve la ecuación de Poisson gravitacional usando una malla 3D periódica y FFT pura en Rust (`rustfft`). El costo es **O(N + N_M³ log N_M)** por evaluación de fuerzas.

#### Algoritmo

```
Partículas (x_i, m_i)
  │  CIC assign
  ▼
ρ[NM³]           (masa/celda; NM = pm_grid_size)
  │  FFT 3D (3× pasadas 1D con rustfft)
  ▼
ρ̂(k)
  │  Poisson: Φ̂(k) = -4πG·ρ̂(k) / k²   (k=0 → 0)
  ▼
F̂_α(k) = -i·k_α·Φ̂(k)   (α = x, y, z)
  │  IFFT 3D × 3
  ▼
F_α[NM³]
  │  CIC interpolate
  ▼
a_i  (aceleraciones por partícula)
```

1. **CIC mass assignment** (`cic::assign`): cada partícula distribuye su masa a los 8 nodos vecinos con pesos trilineales. El grid es periódico (`% nm`).
2. **FFT 3D** (`fft_poisson`): tres pasadas de 1D FFTs (eje X → Y → Z) con `FftPlanner` de `rustfft`. Sin dependencias C.
3. **Poisson en k-space**: `Φ̂(k) = -4πG·ρ̂(k) / k²`; el modo DC (k=0) se pone a cero para eliminar la fuerza de fondo uniforme.
4. **Fuerzas** por componente: `F̂_α = -i·k_α·Φ̂`, IFFT 3D.
5. **CIC interpolation** (`cic::interpolate`): aceleración por partícula interpolada del grid con los mismos pesos trilineales.

#### Configuración TOML

```toml
[gravity]
solver = "pm"
pm_grid_size = 64   # grid NM³; potencia de 2 recomendada para eficiencia FFT
```

#### Notas de diseño

- La resolución de fuerza es la escala de celda `box_size / pm_grid_size`; para fuerzas de largo alcance se recomienda `pm_grid_size` ≈ N^(1/3) – 2×N^(1/3).
- El solver es **serial** por defecto; la paralelización con Rayon se puede añadir en el futuro.
- Las fuerzas son **periódicas** por construcción (incluyen todas las imágenes del sistema).
- No aplica softening explícito (`eps2` se ignora); el suavizado natural es la escala de celda.

## Limitaciones del MVP

- Con `solver = "direct"`, el escalado \(O(N^2)\) no está pensado para producción masiva; con **Barnes–Hut** el coste es \(O(N\log N)\) por paso pero sigue sin PM ni dominios cosmológicos grandes. El objetivo sigue siendo **arquitectura limpia**, **MPI real** y **cadena de validación** reproducible.
- La paridad serial/MPI se valida numéricamente con tolerancia explícita en [experiments/nbody/mvp_smoke/docs/validation.md](../experiments/nbody/mvp_smoke/docs/validation.md).
