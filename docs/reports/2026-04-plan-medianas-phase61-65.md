# Plan de Fases 61–65 — Tareas Medianas

**Fecha de redacción:** abril 2026  
**Estimación por fase:** 1–2 sesiones  
**Dependencias base:** Phases 1–60 completadas  
**Documento relacionado:** [`future-grandes.md`](2026-04-future-grandes.md)

---

## Visión general

```
Phase 61 — FoF paralelo MPI
     ↓
Phase 62 — Merger trees (ligero, single-pass)
     ↓
Phase 63 — Análisis in-situ (P(k) + ξ(r) en el loop stepping)
     ↓
Phase 64 — gadget-ng-vis: proyecciones adicionales y mapa de densidad
     ↓
Phase 65 — Snapshots HDF5 paralelos (MPI-IO + chunking)
```

Cada fase es independiente en implementación, pero el orden maximiza la
reutilización: FoF paralelo desbloquea merger trees y es prerequisito de
corridas de producción.

---

## Phase 61 — FoF paralelo MPI

**Crates:** `gadget-ng-analysis`, `gadget-ng-parallel`, `gadget-ng-physics`  
**Test:** `crates/gadget-ng-physics/tests/phase61_fof_parallel.rs`  
**Sesiones estimadas:** 1–2

### Problema

El FoF actual (`find_halos`) opera en el rank 0 sobre todas las partículas.
Para N > 10⁶ partículas distribuidas entre ranks MPI, el gather de posiciones
al rank 0 es O(N·P) en memoria y comunicación, y el FoF serial es un cuello
de botella.

### Diseño

#### Paso 1 — FoF local

Cada rank corre FoF sobre sus partículas locales. Se producen halos locales
parciales (grupos que cruzan el límite de dominio están incompletos).

#### Paso 2 — Intercambio de partículas en la frontera

Las partículas en una franja de ancho `b × l_mean` alrededor del límite de
cada dominio SFC se intercambian con el rank vecino (halo de FoF, similar a
`exchange_halos_sfc`). Las partículas de halo son *read-only* en el rank receptor.

```
rank k:  [---- dominio_k ----|=halo=]
rank k+1:[=halo=|---- dominio_k+1 ----]
```

#### Paso 3 — Unión de grupos cross-boundary

Con las partículas de halo, el FoF local puede extender sus grupos a través de
la frontera. Se producen `LinkRecord { local_group_id, halo_particle_id, remote_rank }`.

#### Paso 4 — Reducción global de grupos

`allreduce` de tabla de uniones → Union-Find global distribuido (algoritmo de
Meyerhenke et al. 2012). Cada rank obtiene el ID global canónico de sus grupos.

### API target

```rust
// En gadget-ng-parallel:
pub fn find_halos_parallel<R: ParallelRuntime>(
    local_particles: &[Particle],
    runtime: &R,
    decomp: &SfcDecomposition,
    b: f64,
    min_particles: usize,
    rho_crit: f64,
) -> Vec<FofHalo>;
```

### Tests

| Test | Descripción |
|------|-------------|
| `fof_parallel_vs_serial_n64` | Comparar halos MPI (P=1) vs serial sobre N=64³ |
| `fof_parallel_cross_boundary` | Halo construido intencionalmente a caballo de frontera |
| `fof_parallel_mass_conservation` | Σ masa halos = Σ masa total de partículas (ligadas) |
| `fof_parallel_scaling_p4` | Mismo catálogo con P=1, 2, 4 ranks |

### Complejidad estimada

Media-alta. La parte difícil es el Union-Find distribuido; puede simplificarse
con una versión no-óptima O(N_boundary × P) en una primera iteración.

---

## Phase 62 — Merger trees (single-pass, sin base de datos)

**Crates:** `gadget-ng-analysis`, `gadget-ng-cli`  
**Test:** `crates/gadget-ng-physics/tests/phase62_merger_trees.rs`  
**Sesiones estimadas:** 1

### Problema

No hay mecanismo para rastrear la historia de ensamble de halos entre snapshots.
Un merger tree es esencial para entender la formación de estructura y comparar
con simulaciones de referencia.

### Diseño (lightweight, single-pass)

No se requiere base de datos: el árbol se construye en una sola pasada
hacia atrás sobre catálogos de halos consecutivos.

```
Para snapshots S_N, S_{N-1}, ..., S_0:
  Mantener: Map<particle_id → halo_global_id> del snapshot anterior.
  Para cada snapshot S_i:
    Para cada halo H_j en S_i:
      Votar: para cada partícula p ∈ H_j, buscar halo en S_{i+1} donde p estaba.
      Progenitor principal = halo ganador por mayoría.
      Emitir: MergerTreeNode { snap_id, halo_id, mass, prog_main_id, mergers[] }.
```

### Estructuras de datos

```rust
pub struct MergerTreeNode {
    pub snapshot:       usize,
    pub halo_id:        u64,
    pub mass_msun_h:    f64,
    pub n_particles:    usize,
    pub x_com:          [f64; 3],
    pub prog_main_id:   Option<u64>,       // progenitor principal en snap+1
    pub merger_ids:     Vec<u64>,          // progenitores secundarios (mergers)
    pub merger_mass_ratio: Vec<f64>,       // masa relativa de cada merger
}

pub struct MergerForest {
    pub nodes: Vec<MergerTreeNode>,        // todos los halos de todos los snaps
    pub roots: Vec<u64>,                  // halos en el snapshot final (z=0)
}
```

### API target

```rust
pub fn build_merger_forest(
    catalogs: &[(Vec<FofHalo>, Vec<Particle>)],  // (halos, partículas) por snap
    min_shared_fraction: f64,                     // default: 0.5
) -> MergerForest;
```

### CLI

```bash
gadget-ng merge-tree \
  --snapshots "runs/cosmo/snap_*.jsonl" \
  --catalogs  "runs/cosmo/halos_*.jsonl" \
  --out runs/cosmo/merger_tree.json \
  --min-shared 0.5
```

### Tests

| Test | Descripción |
|------|-------------|
| `merger_tree_trivial_no_mergers` | Halo que crece suavemente sin mergers |
| `merger_tree_binary_merger` | Dos halos que se fusionan en un único paso |
| `merger_tree_mass_accretion_history` | MAH(z) = M_main(z)/M_z0 monótonamente no-decreciente |
| `merger_tree_roundtrip_json` | Serializar/deserializar `MergerForest` |

---

## Phase 63 — Análisis in-situ en el loop `stepping`

**Crates:** `gadget-ng-cli` (engine), `gadget-ng-analysis`  
**Test:** `crates/gadget-ng-physics/tests/phase63_insitu_analysis.rs`  
**Sesiones estimadas:** 1

### Problema

Actualmente P(k), FoF y ξ(r) solo se calculan en postproceso (`gadget-ng analyze`).
Para corridas largas es valioso tener series temporales de P(k)(z) y N_halos(z)
sin necesidad de guardar todos los snapshots completos.

### Diseño

Nueva sección `[analysis]` en el TOML de configuración:

```toml
[analysis]
enabled          = true
interval         = 20          # cada 20 pasos
pk_mesh          = 32          # resolución de la grilla P(k)
fof_b            = 0.2
fof_min_part     = 20
xi_bins          = 0           # 0 = desactivado (lento)
output_dir       = "runs/cosmo/insitu"
```

En cada paso múltiplo de `interval`, el engine llama a `run_insitu_analysis()`
que:
1. Calcula P(k) desde las posiciones locales (si MPI: allreduce de la grilla).
2. Corre FoF local (Phase 61 si MPI, serial si P=1).
3. Opcionalmente calcula ξ(r) via FFT del P(k) ya calculado.
4. Escribe `insitu_{step:06}.json` con los resultados.

### Estructura de salida

```json
{
  "step": 100,
  "a": 0.25,
  "z": 3.0,
  "power_spectrum": [...],
  "n_halos": 142,
  "m_total_halos_msun_h": 4.2e15,
  "xi_r": [...]
}
```

### API interna

```rust
pub struct InsituAnalysisConfig { /* espejo del [analysis] TOML */ }

pub fn run_insitu_analysis(
    particles: &[Particle],
    config: &InsituAnalysisConfig,
    cosmo: Option<&CosmologyParams>,
    a: f64,
    step: u64,
    out_dir: &Path,
) -> anyhow::Result<()>;
```

### Tests

| Test | Descripción |
|------|-------------|
| `insitu_pk_written_at_interval` | Verifica que el archivo existe cada `interval` pasos |
| `insitu_pk_monotone_growth` | P(k,a2)/P(k,a1) > 1 para a2 > a1 en runs de 50 pasos |
| `insitu_fof_halos_increase` | N_halos no decreciente con a en corrida estándar |
| `insitu_no_output_between_interval` | No se escribe en pasos intermedios |

---

## Phase 64 — gadget-ng-vis: proyecciones adicionales y mapa de densidad

**Crates:** `gadget-ng-vis`, `gadget-ng-cli`  
**Test:** `crates/gadget-ng-vis/tests/` (unitarios existentes extendidos)  
**Sesiones estimadas:** 1

### Problema

El módulo `gadget-ng-vis` solo tiene proyección XY y render PPM plano sin
información de densidad. Para diagnóstico visual de simulaciones es útil:
- Proyecciones XZ e YZ además de XY.
- Mapa de densidad proyectada (número de partículas por pixel → colormap).
- Soporte de múltiples formatos de salida (PPM ya existe; agregar PNG nativo).

### Diseño

#### Proyecciones adicionales

```rust
pub enum Projection { XY, XZ, YZ }

pub fn render_ppm_projection(
    positions: &[Vec3],
    box_size: f64,
    width: usize,
    height: usize,
    proj: Projection,
) -> Vec<u8>;
```

#### Mapa de densidad

En lugar de pintar cada partícula como un pixel blanco, acumular conteos
en la grilla `grid[i][j] += 1` y mapear a color con un colormap (Viridis,
que ya existe en `CpuCanvas`):

```rust
pub fn render_density_ppm(
    positions: &[Vec3],
    box_size: f64,
    width: usize,
    height: usize,
    proj: Projection,
    colormap: Colormap,  // Viridis | Hot | Grayscale
) -> Vec<u8>;
```

El mapa de densidad usa escala logarítmica: `log10(1 + count)` normalizado
al máximo del frame.

#### PNG nativo (sin dependencias externas)

Escribir un encoder PNG mínimo en pure Rust que soporte solo RGB8, usando
deflate via la crate `miniz_oxide` (ya en el árbol de dependencias transitivas).
Si `miniz_oxide` no está disponible, usar el encoder PPM existente como fallback.

```rust
pub fn write_png(path: &Path, pixels: &[u8], width: usize, height: usize) -> io::Result<()>;
```

#### CLI

```bash
# Render PPM densidad en proyección XZ:
gadget-ng stepping --config sim.toml --out out/ \
  --vis-snapshot 1 --vis-proj xz --vis-mode density --vis-colormap viridis

# Render PNG de tres proyecciones simultáneas:
gadget-ng stepping --config sim.toml --out out/ \
  --vis-snapshot 1 --vis-all-proj --vis-format png
```

### Tests

| Test | Descripción |
|------|-------------|
| `density_map_empty_is_black` | Sin partículas → todos los pixels negros |
| `density_map_uniform_is_flat` | Distribución uniforme → varianza de color < 5% |
| `density_map_concentrated_bright` | Clúster en el centro → pixel central más brillante |
| `projection_xz_uses_y_as_depth` | Partícula en (x,0,z) aparece en pixel (x,z) de XZ |
| `write_png_header` | Verificar cabecera PNG `\x89PNG` |

---

## Phase 65 — Snapshots HDF5 paralelos (MPI-IO)

**Crates:** `gadget-ng-io`, `gadget-ng-parallel`, `gadget-ng-cli`  
**Test:** `crates/gadget-ng-physics/tests/phase65_hdf5_parallel.rs`  
**Sesiones estimadas:** 2  
**Feature flag:** `hdf5-parallel` (requiere HDF5 compilado con `--enable-parallel` y MPI)

### Problema

El writer HDF5 actual (`gadget-ng-io`) escribe desde el rank 0 después de un
gather global. Para N > 10⁶ partículas esto es prohibitivo en memoria y tiempo.
HDF5 paralelo con MPI-IO permite que cada rank escriba su porción directamente.

### Diseño

#### Layout GADGET-4 estándar

```
/Header
  /NumPart_ThisFile  [i32 × 6]
  /NumPart_Total     [i32 × 6]
  /BoxSize           f64
  /Time              f64
  /Redshift          f64
/PartType1
  /Coordinates       [N × 3] f32
  /Velocities        [N × 3] f32
  /ParticleIDs       [N]     u64
  /Masses            [N]     f32   (si masas variables)
```

#### Escritura paralela

```rust
// Feature: hdf5-parallel
pub fn write_snapshot_hdf5_parallel<R: ParallelRuntime>(
    path: &Path,
    particles: &[Particle],
    env: &SnapshotEnv,
    runtime: &R,
) -> anyhow::Result<()>;
```

Internamente:
1. Cada rank calcula su offset: `offset_i = Σ_{j<i} N_j` (via allreduce de prefijos).
2. Abre el archivo HDF5 con `H5Pset_fapl_mpio` (MPI-IO colectivo).
3. Cada rank escribe su porción en el dataset con `H5Sselect_hyperslab`.
4. Llamada colectiva `H5Dwrite` — todos los ranks escriben simultáneamente.

#### Compresión y chunking

```toml
[output]
hdf5_chunk_size   = 65536    # partículas por chunk (ajustar según I/O local)
hdf5_compression  = 1        # nivel gzip (0=sin compresión, 9=máximo)
```

#### Lectura paralela

```rust
pub fn read_snapshot_hdf5_parallel<R: ParallelRuntime>(
    path: &Path,
    runtime: &R,
    decomp: &SfcDecomposition,
) -> anyhow::Result<Vec<Particle>>;
```

Cada rank lee solo las partículas que le pertenecen según la descomposición SFC
(lectura selectiva con hyperslab).

### Tests

| Test | Descripción |
|------|-------------|
| `hdf5_parallel_write_read_p1` | Escribir P=1 → leer → partículas idénticas |
| `hdf5_parallel_write_read_p4` | Escribir P=4 → leer con P=4 → idénticas |
| `hdf5_parallel_layout_gadget4` | Verificar grupos `/Header`, `/PartType1` y datasets |
| `hdf5_parallel_vs_serial_content` | Mismo snapshot escrito serial y paralelo → bit-idéntico |
| `hdf5_parallel_chunking` | Chunk size configurable no rompe integridad |

### Compatibilidad

El archivo resultante debe poder leerse con:
```python
import h5py
f = h5py.File("snapshot.hdf5", "r")
pos = f["/PartType1/Coordinates"][:]   # funciona sin modificación
```

---

## Tabla resumen

| Fase | Tarea | Crates | Sesiones | Prioridad |
|------|-------|--------|----------|-----------|
| **61** | ⚡ FoF paralelo MPI | analysis, parallel | 1–2 | Alta |
| **62** | 🌳 Merger trees (single-pass) | analysis, cli | 1 | Media |
| **63** | 📈 Análisis in-situ en stepping | cli (engine), analysis | 1 | Alta |
| **64** | 🖼️ Proyecciones adicionales + densidad | vis, cli | 1 | Baja |
| **65** | 💾 HDF5 paralelo (MPI-IO) | io, parallel, cli | 2 | Media |

**Total estimado:** 6–7 sesiones.

---

## Dependencias entre fases

```
61 (FoF paralelo)
    ├──→ 62 (Merger trees, necesita catálogos por rank)
    └──→ Grandes G1, G3, G5 (SUBFIND, producción, merger trees completos)

63 (In-situ)
    └──→ depende idealmente de 61 para in-situ MPI correcto

65 (HDF5 paralelo)
    └──→ Grandes G3 (corrida N=256³ requiere I/O eficiente)

64 (Vis)
    └──→ independiente, puede hacerse en cualquier momento
```

---

## Criterios de cierre por fase

| Fase | Criterio de cierre |
|------|-------------------|
| 61 | `cargo test phase61` pasa con P=1 y P=4; catálogo FoF idéntico al serial |
| 62 | MAH del halo más masivo en test sintético consistente con input conocido |
| 63 | Serie temporal P(k)(a) escrita cada `interval` pasos; tests sin flakiness |
| 64 | Proyecciones XY/XZ/YZ + densidad + PNG pasan tests; visualmente correctas |
| 65 | HDF5 escrito por P=4 legible con `h5py` y bit-idéntico al escrito por P=1 |
