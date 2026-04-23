# Phase G4 — AMR-PM: refinamiento adaptativo de la malla Particle-Mesh

**Fecha:** abril 2026  
**Estimación original:** 5+ sesiones (simplificado a 2 niveles: ~2 sesiones)  
**Dependencias:** PM slab (Phase 20), PM pencil 2D (Phase 46), TreePM (Phase 21–25)  
**Crates afectados:** `gadget-ng-pm`  
**Archivos clave:**
- `crates/gadget-ng-pm/src/amr.rs` (nuevo)
- `crates/gadget-ng-pm/src/lib.rs` (actualizado)
- `crates/gadget-ng-physics/tests/phase_g4_amr_pm.rs` (nuevo)

---

## Objetivo

Implementar un solver PM con refinamiento adaptativo de malla (AMR) en dos
niveles jerárquicos. El grid base resuelve fuerzas de largo alcance globalmente;
los parches de refinamiento agregan fuerzas de corto alcance a alta resolución
en regiones de alta densidad.

---

## Diseño e implementación

### Arquitectura AMR de 2 niveles

```
Nivel 0 (base):
  grid nm³ periódico, box_size global
  → fuerzas de fondo F_base(x)

Nivel 1 (parches):
  ≤ max_patches grids locales nm_patch³
  → F_patch(x) solo para partículas dentro del parche
  → corrección ΔF = F_patch - F_base con peso de transición
```

La fuerza total de cada partícula es:

```
F_total(i) = F_base(i) + Σ_{parche p ∋ i} w_p(i) · [F_patch_p(i) - F_base(i)]
```

donde `w_p(i) ∈ [0,1]` es un peso que vale 1 en el centro del parche y 0
en los bordes, suavizando la transición para evitar discontinuidades.

### Estructuras públicas

#### `AmrParams`

```rust
pub struct AmrParams {
    pub delta_refine: f64,       // δ_refine: umbral de sobredensidad (default: 10)
    pub patch_cells_base: usize, // ancho del parche en celdas base (default: 5)
    pub nm_patch: usize,         // resolución interna del parche (default: 32)
    pub max_patches: usize,      // máximo de parches por paso (default: 16)
    pub zero_pad: bool,          // convolución no periódica (default: true)
}
```

#### `PatchGrid`

```rust
pub struct PatchGrid {
    pub center: Vec3,          // centro del parche
    pub size: f64,             // lado físico
    pub nm: usize,             // resolución (celdas por lado)
    pub density: Vec<f64>,     // masa/celda (nm³)
    pub forces: [Vec<f64>; 3], // componentes de fuerza (nm³ cada una)
}
```

### Pipeline de una llamada a `amr_pm_accels`

```
1. CIC deposit → base_density (nm_base³)
2. FFT Poisson → [fx_base, fy_base, fz_base]
3. CIC interpolate → accels_base[i]  ← base grid forces
4. identify_refinement_patches(base_density, δ_refine)
   → lista de PatchGrid
5. Para cada parche:
   a. deposit_to_patch(positions, masses, patch)  ← CIC local
   b. solve_patch(patch, g, zero_pad)             ← Poisson local
   c. interpolate_patch_forces(patch, positions)  ← CIC local
   d. Aplicar corrección ponderada ΔF a partículas dentro del parche
6. Return accels
```

### Zero-padding para condiciones no periódicas

Para evitar que los parches asuman periodicidad (incorrecta para regiones locales):

```
Densidad parche nm³ → zero-pad → 2nm³
FFT Poisson en 2nm³ con box_size = 2×patch_size
Extraer fuerzas del octante [0, nm)³
```

Esto implementa la convolución no periódica (Green's function de espacio libre),
equivalente al método de Hockney & Eastwood (1988) para cálculos libres de
condiciones de contorno espurias.

### Identificación de parches

El algoritmo identifica las celdas del base grid donde:

```
ρ_celda > ρ̄ × (1 + δ_refine)
```

Las celdas calificadas se ordenan por densidad descendente. Se toman
las `max_patches` regiones de mayor densidad, descartando aquellas que
se solapan demasiado con parches ya seleccionados (distancia < 0.5×patch_size
en los 3 ejes).

---

## API pública

```rust
// Solver AMR completo
pub fn amr_pm_accels(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    nm_base: usize,
    g: f64,
    params: &AmrParams,
) -> Vec<Vec3>

// Con estadísticas de refinamiento
pub fn amr_pm_accels_with_stats(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    nm_base: usize,
    g: f64,
    params: &AmrParams,
) -> (Vec<Vec3>, AmrStats)

// Funciones de bajo nivel (públicas para testing)
pub fn identify_refinement_patches(...) -> Vec<PatchGrid>
pub fn deposit_to_patch(positions, masses, patch: &mut PatchGrid)
pub fn solve_patch(patch: &mut PatchGrid, g: f64, zero_pad: bool)
pub fn interpolate_patch_forces(patch: &PatchGrid, positions: &[Vec3]) -> Vec<Vec3>
```

---

## Complejidad computacional

| Componente | Tiempo | Memoria |
|-----------|--------|---------|
| Base grid | O(nm³ log nm) | O(nm³) |
| Identificación | O(nm³) | O(P) donde P=n_patches |
| Por parche (P parches) | O(P × nm_p³ log nm_p) | O(P × nm_p³) |
| Interpolación base | O(N) | O(N) |
| Interpolación parche | O(N_p) por parche | O(N_p) |

Con `nm_patch = 32`, `nm_base = 512`, `P = 16`:
- Base: O(512³ × 9) ≈ 10⁹ operaciones
- Parches: O(16 × 32³ × 5) ≈ 2.6×10⁶ operaciones (< 0.3% del base)

El overhead AMR es mínimo si los parches son mucho más pequeños que el base grid.

---

## Tests (7 / 7 OK)

| Test | Descripción |
|------|-------------|
| `g4_amr_force_no_nan` | AMR no produce NaN/Inf para distribución mixta |
| `g4_amr_refines_dense_region` | parches detectados cerca del cluster denso |
| `g4_amr_consistent_with_base` | sin parches, AMR ≡ PM base (error < 1e-12) |
| `g4_amr_mass_conservation` | masa depositada en parche = suma de masas dentro |
| `g4_patch_solve_produces_finite_forces` | solve_patch da fuerzas finitas |
| `g4_patch_zero_padding_vs_periodic` | ambos modos dan fuerzas finitas |
| `g4_amr_stats` | estadísticas n_patches, n_refined, max_overdensity coherentes |

Tests unitarios adicionales en `gadget-ng-pm --lib`:

| Test | Descripción |
|------|-------------|
| `amr_uniform_no_patches` | δ_refine=50 → sin parches en distribución uniforme |
| `amr_concentrated_cluster_creates_patch` | cluster de 100 partículas → parche |
| `amr_patch_contains_logic` | `PatchGrid::contains()` correcta |
| `amr_deposit_mass_conservation` | masa conservada en depósito CIC local |
| `amr_pm_accels_no_nan` | aceleraciones finitas (smoke test) |
| `amr_stats_reports_correct_n_patches` | `AmrStats` coherente |
| `amr_params_default` | valores por defecto correctos |

**Total: 7 + 7 = 14 tests OK**

---

## Limitaciones conocidas

1. **Un solo nivel de refinamiento:** la implementación actual soporta 2 niveles
   (base + parches). La jerarquía recursiva completa (nivel 0→1→2→…) queda para
   una futura iteración.

2. **Comunicación MPI no implementada:** los parches son locales al rank que los
   identifica. Para parches que cruzan fronteras MPI se requeriría una negociación
   de ownership, similar a la sincronización SFC↔slab del PM distribuido.

3. **Zero-padding requiere 8× más memoria:** para `nm_patch = 32`, el grid
   extendido es 64³ = 262 144 celdas por parche. Con 16 parches: ~33 MB adicionales.

4. **Criterio de umbral fijo:** `delta_refine` es un parámetro global. Una mejora
   futura sería adaptar el umbral según la resolución objetivo (convergencia de
   la fuerza relativa en la región).

---

## Referencia

- Kravtsov, Klypin & Khokhlov (1997), ApJS 111, 73 — ART.
- Knebe, Green & Binney (2001), MNRAS 325, 845 — AMR cosmológico.
- Hockney & Eastwood (1988), "Computer Simulation Using Particles" — zero-padding.
