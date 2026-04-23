# Phase 61 — FoF paralelo MPI (cross-boundary Union-Find)

**Fecha:** abril 2026  
**Crates:** `gadget-ng-analysis` (feature `parallel`), `gadget-ng-parallel`, `gadget-ng-physics`  
**Archivos nuevos:**  
- `crates/gadget-ng-analysis/src/fof_parallel.rs`  
- `crates/gadget-ng-analysis/src/fof.rs` (helper `find_halos_combined`)  
- `crates/gadget-ng-physics/tests/phase61_fof_parallel.rs`

---

## Contexto

`find_halos` (Phase anterior) operaba sobre todas las partículas en un único proceso.
Con N > 10⁶ partículas distribuidas en P ranks MPI, un gather al rank 0 es O(N·P),
lo que escala mal para producción. Phase 61 implementa un FoF distribuido que escala
a O(N/P + N_frontera) usando el intercambio de halos SFC ya existente.

---

## Algoritmo

```
1. Longitud de enlace global: ll = b × (V/N_total)^{1/3}
   donde N_total = allreduce_sum(N_local)

2. exchange_halos_sfc(local, decomp, ll)
   → partículas de vecinos dentro de una franja ll alrededor de las fronteras SFC

3. Ejecutar find_halos_combined(local + halos_recibidos)
   - Construye cell-linked-list sobre el conjunto combinado
   - Union-Find sobre todos los pares dentro de ll
   - Solo retiene grupos cuya raíz Union-Find es local (índice < N_local)

4. Cada rank emite su porción del catálogo sin duplicar halos
```

### Helper `find_halos_combined`

```rust
pub fn find_halos_combined(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    n_local: usize,     // Las primeras n_local entradas son "propias"
    box_size: f64,
    b: f64,
    min_particles: usize,
    rho_crit: f64,
) -> Vec<FofHalo>
```

La función ejecuta el Union-Find habitual pero filtra los grupos por raíz local:

```rust
// Solo groups cuya raíz Union-Find es local (índice < n_local).
if root < n_local {
    groups.entry(root).or_default().push(i);
}
```

Esto garantiza que cada halo se emite exactamente desde el rank cuya partícula
es la raíz del árbol Union-Find.

---

## Firma pública

```rust
// crates/gadget-ng-analysis/src/fof_parallel.rs
pub fn find_halos_parallel<R: ParallelRuntime>(
    local: &[Particle],
    runtime: &R,
    decomp: &SfcDecomposition,
    box_size: f64,
    b: f64,
    min_particles: usize,
    rho_crit: f64,
) -> Vec<FofHalo>
```

Con `SerialRuntime` (P=1) no hay intercambio y el resultado es idéntico al FoF serial.
Disponible con `gadget-ng-analysis = { features = ["parallel"] }`.

---

## Feature flag

```toml
# crates/gadget-ng-analysis/Cargo.toml
[features]
parallel = ["dep:gadget-ng-parallel"]
```

---

## Tests

| Test | Descripción |
|------|-------------|
| `phase61_vs_serial_p1` | Con `SerialRuntime` P=1, mismo número de halos y masa total idéntica que el FoF serial |
| `phase61_cross_boundary_halo_recovered` | Cluster dividido artificialmente entre local/halos: `find_halos_combined` lo recupera completo con ≥10 partículas |
| `phase61_mass_conservation` | `Σ masa(halos) ≤ masa total`; exactamente 2 halos en un sistema de 2 clusters + campo |

Todos los tests pasan en modo serial (sin MPI).

---

## Comportamiento escalar (correctitud sin MPI)

```
P=1: exchange_halos_sfc → [] (vacío) → find_halos_combined ≡ find_halos
P>1: halos cruzando fronteras SFC son recuperados en el rank cuya raíz es local
```

Para el path P>1 se requiere HDF5 paralela o análisis post-proceso sobre catálogos
parciales que se concatenen en el rank 0.
