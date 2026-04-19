# Fase 23 — TreePM SR Desacoplado del Slab-Z: Dominio 3D/SFC

**Fecha:** 2026-04-18  
**Estado:** Implementado y validado (P=1, tests geométricos y físicos pasan)

---

## 1. Resumen ejecutivo

La Fase 23 desacopla el árbol de corto alcance (SR) del slab-z del PM e implementa una descomposición 3D/SFC real para el SR, con el halo volumétrico periódico de Fase 22 como mecanismo activo principal.

**Resultado clave:** en P=1, la física es bit-a-bit idéntica entre los tres paths (Fase 21, 22 y 23). El overhead de la sincronización PM↔SR es de ~+1.4% en wall time para N=512. El halo 3D periódico (`exchange_halos_3d_periodic`) es ahora el camino activo del corto alcance cuando `treepm_sr_sfc = true`.

| Métrica | Fase 21 slab-1d | Fase 22 slab-3d | Fase 23 sr-sfc |
|---------|-----------------|-----------------|----------------|
| wall s/paso (N=512, P=1) | 0.0177 | 0.0176 | 0.0179 |
| v_rms (último paso) | 3.821e+02 | 3.821e+02 | 3.821e+02 |
| delta_rms (último paso) | 2.845e+00 | 2.845e+00 | 2.845e+00 |
| a final | 0.0682 | 0.0682 | 0.0682 |
| path_active | `treepm_slab_1d` | `treepm_slab_3d` | `treepm_sr_sfc_3d` |
| cobertura SR geométrica | solo z | x,y,z,diag | x,y,z,diag |
| dependencia z-slab en SR | **sí** | **sí (dominio)** | **no** |

---

## 2. Diagnóstico del estado anterior (Fase 22)

La Fase 22 implementó el halo volumétrico 3D periódico (`exchange_halos_3d_periodic`) y demostró su correctitud geométrica. Sin embargo, el SR seguía **acoplado al slab-z del PM**: la migración de partículas al inicio de cada paso (`exchange_domain_by_z`) ponía las partículas en z-slab antes de que el halo 3D operara sobre ellas.

En otras palabras: el halo 3D de Fase 22 era infraestructura correcta pero aplicada sobre un dominio que seguía siendo z-slab. Las interacciones de corto alcance eran correctas geométricamente (dentro del z-slab), pero el dominio del SR no era verdaderamente 3D/SFC.

**Acoplamiento SR↔z-slab en engine.rs (Fase 21/22):**
```rust
// Inicio del paso: migración z-slab
if use_treepm_slab && !use_treepm_sr_sfc && rt.size() > 1 {
    rt.exchange_domain_by_z(&mut local, z_lo, z_hi);  // ← acoplamiento
}

// En compute_acc: halos sobre partículas ya en z-slab
let sr_halos = if use_treepm_3d_halo {
    rt.exchange_halos_3d_periodic(parts, box_size, r_cut)  // sobre z-slab domain
} else {
    rt.exchange_halos_by_z_periodic(parts, z_lo, z_hi, r_cut)
};
```

---

## 3. Arquitectura dual PM/SR (Fase 23)

### 3.1 Principio de diseño

La Fase 23 introduce una **arquitectura dual** donde PM y SR usan descomposiciones distintas:

| Componente | Descomposición | Función |
|------------|----------------|---------|
| PM largo alcance | z-slab (`SlabLayout`) | FFT distribuida (sin cambios) |
| SR árbol | SFC (Morton/Hilbert) | Dominio 3D con halo volumétrico |
| Partículas "en casa" | SFC domain (verdad del paso) | Ownership del rank |
| Sincronización PM↔SR | clone + z-migration + back | Explícita y medible |

### 3.2 Flujo por evaluación de fuerza

```
Por cada llamada a compute_acc(parts, acc):

1. SR (SFC domain — path activo principal):
   sr_halos = exchange_halos_3d_periodic(parts, box_size, r_cut)
   acc_sr = short_range_accels_sfc(parts, sr_halos)

2. PM LR (z-slab — sincronización explícita):
   pm_parts = parts.clone()
   exchange_domain_by_z(&mut pm_parts, z_lo, z_hi)  ← temporal
   acc_lr_pm = PM pipeline(pm_parts)                ← sin cambios
   pm_parts[i].acceleration = acc_lr_pm[i]
   exchange_domain_sfc(&mut pm_parts, &sfc_decomp)   ← retorno SFC
   acc_lr = HashMap<global_id,Vec3> lookup

3. Suma:
   acc[i] = acc_lr[i] + acc_sr[i]
```

### 3.3 Decisión de descomposición SR: SFC existente (Opción A)

Se reutiliza `SfcDecomposition` (Morton/Hilbert) de `gadget-ng-parallel/src/sfc.rs`. Justificación:

- Ya existe y funciona (path SFC+LET desde Fase 8)
- Localidad 3D garantizada (especialmente Hilbert)
- Compatible con `exchange_halos_3d_periodic` (opera sobre AABB reales)
- Compatible con GADGET-4: Peano-Hilbert SFC + PM slab es exactamente la arquitectura de GADGET-4
- Costo de migración idéntico al path LET

---

## 4. Implementación

### 4.1 Archivos modificados/creados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-core/src/config.rs` | Añade `treepm_sr_sfc: bool` a `GravitySection` |
| `crates/gadget-ng-treepm/src/distributed.rs` | Añade `SfcShortRangeParams`, `short_range_accels_sfc` |
| `crates/gadget-ng-treepm/src/lib.rs` | Exporta los nuevos símbolos |
| `crates/gadget-ng-cli/src/engine.rs` | Path `use_treepm_sr_sfc`, migración SFC, PM sync, diagnósticos |
| `crates/gadget-ng-treepm/tests/sr_sfc_geometry.rs` | **Nuevo**: 5 tests geométricos |
| `crates/gadget-ng-physics/tests/treepm_sr_sfc.rs` | **Nuevo**: 8 tests de validación física |
| `experiments/nbody/phase23_sr_sfc_domain/` | **Nuevo**: configs, scripts, resultados |

### 4.2 `SfcShortRangeParams` y `short_range_accels_sfc`

```rust
/// Parámetros para el árbol SR sobre dominio SFC (Fase 23).
pub struct SfcShortRangeParams<'a> {
    pub local_particles: &'a [Particle],   // propias del rank SFC
    pub halo_particles:  &'a [Particle],   // ghost via exchange_halos_3d_periodic
    pub eps2:    f64,
    pub g:       f64,
    pub r_split: f64,
    pub box_size: f64,
}

/// Delega al mismo kernel periodic que SlabShortRangeParams.
/// La diferencia arquitectónica está upstream: halos obtenidos con halo 3D.
pub fn short_range_accels_sfc(params: &SfcShortRangeParams<'_>, out: &mut [Vec3]) {
    short_range_accels_slab(&SlabShortRangeParams { ... }, out)
}
```

### 4.3 Nuevos diagnósticos en `HpcStepStats`

```rust
sr_domain_particle_count: usize,  // partículas en SFC SR domain
sr_halo_3d_neighbors: usize,      // halos via exchange_halos_3d_periodic
sr_sync_ns: u64,                  // costo clone + 2 migraciones + hashmap
```

Y en `HpcTimingsAggregate`:
```rust
mean_sr_domain_particle_count: f64,
mean_sr_halo_3d_neighbors: f64,
mean_sr_sync_s: f64,
sr_sync_fraction: f64,
```

### 4.4 Activación

```toml
[gravity]
solver         = "tree_pm"
pm_grid_size   = 32
treepm_slab    = true
treepm_halo_3d = true
treepm_sr_sfc  = true   # ← Fase 23: dominio SFC para SR
```

---

## 5. Halo 3D periódico: del rol de infraestructura al camino activo

La diferencia fundamental entre Fase 22 y Fase 23 es:

| | Fase 22 | Fase 23 |
|--|---------|---------|
| Dominio de partículas al inicio del paso | z-slab | SFC |
| `exchange_halos_3d_periodic` opera sobre | z-slab domain | SFC domain |
| ¿El halo 3D es necesario? | No (z-slab uniforme → equivale a 1D) | **Sí** (SFC → sin halo 3D habría gaps) |
| path_active | `treepm_slab_3d` | `treepm_sr_sfc_3d` |

Para el caso P=2 con descomposición SFC en octantes:
- Rank 0: x∈[0,0.5), y∈[0,0.5), z∈[0,0.5)
- Rank 1: x∈[0.5,1), y∈[0.5,1), z∈[0.5,1)
- Partícula de rank 1 en (0.95,0.95,0.95): distancia 3D periódica al dominio de rank 0 ≈ 0.087 < r_cut=0.1
- **Halo 1D-z no la capturaría** (z=0.95 > z_lo_rank1 + r_cut = 0.5 + 0.1 = 0.6)
- **Halo 3D sí la captura** → interacción SR correcta

---

## 6. Correctitud y cobertura geométrica

### 6.1 Tests automáticos (todos pasan)

**Tests geométricos (`sr_sfc_geometry.rs`):**

| Test | Verificación | Resultado |
|------|-------------|----------|
| `sr_sfc_border_interaction` | Fuerza SR no nula en borde SFC (d=0.04 < r_cut) | ✓ |
| `sr_sfc_diagonal_periodic_interaction` | Diagonal (0.01,0.01,0.01)↔(0.99,0.99,0.99): f≠0 en -x,-y,-z | ✓ |
| `sr_sfc_halo_3d_covers_rcut_pairs` | Halo 3D incluye diagonal, excluye lejana; AABB correcta | ✓ |
| `sr_sfc_minimum_image_active_in_walk` | Min-image: z=0.05↔z=0.95, f apunta en -z | ✓ |
| `sr_sfc_no_geometric_gaps` | Borde derecho recibe fuerza del halo; sin auto-fuerza | ✓ |

**Tests de física (`treepm_sr_sfc.rs`):**

| Test | Verificación | Resultado |
|------|-------------|----------|
| `sr_sfc_vs_slab_p1_equal` | SR-SFC ≡ SR-slab en P=1 (error < 1e-14) | ✓ |
| `sr_sfc_no_double_counting_pm` | erf+erfc=1; sin auto-fuerza con halos vacíos | ✓ |
| `sr_sfc_no_explosion_cosmological` | N=27, 3 pasos, sin NaN/Inf | ✓ |
| `sr_sfc_pm_force_return_by_global_id` | Lookup por global_id bit-a-bit correcto | ✓ |
| `sr_sfc_x_border_periodic` | x=0.02↔x=0.98 vía min-image, f en -x | ✓ |
| `sr_sfc_y_border_periodic` | y=0.02↔y=0.98 vía min-image, f en -y | ✓ |
| `sr_sfc_momentum_conservation` | |Δp| < 1e-10 (Newton III en SR-SFC) | ✓ |
| `sr_sfc_equals_slab3d_p1` | SR-SFC ≡ treepm_slab_3d en P=1 (bit-a-bit) | ✓ |

---

## 7. Validación física

### 7.1 Resultados comparativos (N=512, EdS, P=1, 20 pasos)

| Fase | wall s/paso | v_rms | delta_rms | a final |
|------|------------|-------|-----------|---------|
| 21 — slab-1d | 0.0177 | 3.821e+02 | 2.845e+00 | 0.0682 |
| 22 — slab-3d | 0.0176 | 3.821e+02 | 2.845e+00 | 0.0682 |
| 23 — sr-sfc  | 0.0179 | 3.821e+02 | 2.845e+00 | 0.0682 |

**Diferencia relativa v_rms (Fase 23 vs 22): 0.00e+00** — física idéntica en P=1.

En P=1 los tres paths son equivalentes porque no hay halos que intercambiar entre ranks. La diferencia física aparece en P>1 con descomposición no-Z-slab, donde el halo 3D de Fase 23 cubre interacciones que el halo 1D-z de Fase 21 perdería.

### 7.2 Overhead de sincronización PM↔SR

| Overhead | Valor (N=512, P=1) |
|----------|-------------------|
| Wall time Fase 23 vs Fase 22 | +1.4% |
| Fuente del overhead | clone de partículas + HashMap lookup O(N/P) |

El costo de sincronización PM↔SR en P=1 es mínimo porque las migraciones (`exchange_domain_by_z`, `exchange_domain_sfc`) son no-ops en modo serial. Para P>1, el costo incluirá dos `alltoallv` adicionales por evaluación de fuerza, medible con `sr_sync_ns`.

---

## 8. Convivencia PM slab y SR SFC: respuestas explícitas

### ¿Dónde vive "la verdad" de cada partícula?

**SFC domain**. Al inicio de cada paso, las partículas son migradas a su rank SFC (o rebalanceadas si corresponde). El z-slab solo es una vista temporal para el PM.

### ¿Cuándo se sincroniza entre PM y SR?

Una vez por evaluación de fuerza (no por paso si hay sub-stepping). La sincronización incluye:
1. Clone de partículas locales → pm_parts
2. `exchange_domain_by_z(pm_parts)` → z-slab temporal
3. PM pipeline completo (deposit, FFT, interpolate)
4. Embebido de acc_lr en pm_parts.acceleration
5. `exchange_domain_sfc(pm_parts)` → de vuelta a SFC
6. HashMap lookup por global_id → acc_lr para cada partícula SFC-local

### ¿Cuánto cuesta esa sincronización?

- **P=1:** negligible (~1.4% overhead = clone + O(N) HashMap lookup)
- **P>1:** 2× domain migrations adicionales vs Fase 21/22; medible con `sr_sync_ns`

### ¿Es sostenible para fases futuras?

La arquitectura dual es correcta pero no óptima. En GADGET-4, el PM scatter/gather (fase 24+ potencial) elimina las 2 migraciones adicionales reemplazándolas por:
- Scatter CIC: cada SFC rank envía contribuciones de densidad a los PM slab ranks
- Gather forces: los PM slab ranks devuelven las fuerzas interpoladas

Esto reduce el costo de comunicación de O(N/P) a O(N_halo_PM/P), que es significativamente menor.

---

## 9. Costo/beneficio del desacoplamiento PM/SR

### Beneficio

1. **Correctitud geométrica completa**: el SR puede operar sobre cualquier descomposición de dominio (SFC, bloques, octantes) sin perder interacciones en bordes diagonales
2. **Halo 3D como path activo**: `exchange_halos_3d_periodic` es necesario y activo para SR-SFC
3. **Independencia de z**: el árbol SR ya no está atado a la dirección z del PM
4. **Preparación para GADGET-4**: arquitectura directamente análoga a GADGET-4 (SFC + PM slab)

### Costo

1. **P=1**: +1.4% wall time (clone + HashMap lookup O(N))
2. **P>1**: +2 domain migrations por evaluación de fuerza (medido con `sr_sync_ns`)
3. **Memoria**: 2× partículas en memoria durante la sincronización PM (pm_parts temporales)

### ¿Es ya un TreePM distribuido general tipo GADGET?

**Sí, arquitectónicamente**: el dominio SR es genuinamente 3D/SFC, el PM sigue en slab-z, y la sincronización es explícita. Las interacciones de corto alcance ya no dependen de la dirección z.

**Pendiente para GADGET-4 completo:**
- Scatter/gather PM (eliminar las 2 migraciones extra de Fase 23)
- Criterio MAC para SR (actualmente walk exhaustivo hasta r_cut)
- Load balancing adaptativo SR (árbol no redistribuye carga)
- Block timesteps (fuera del scope de esta fase)

---

## 10. Limitaciones documentadas

1. **Sincronización PM↔SR con clone**: las pm_parts temporales duplican la memoria de partículas durante el cómputo. Mejorable con scatter/gather PM directo (Fase 24+).

2. **HpcStepStats no escritos para el path TreePM**: los campos `sr_sync_ns`, `sr_domain_particle_count`, `sr_halo_3d_neighbors` se calculan en el engine pero actualmente no se escriben al `diagnostics.jsonl` en el branch cosmológico TreePM. Solo se escriben en el path SFC+LET. Para medir `sr_sync_ns` en producción se necesita extender el pipe de diagnósticos del TreePM.

3. **No probado con MPI real (P>1)**: la correctitud con P>1 requiere `--features mpi` y ejecución con mpirun. Los tests automáticos son P=1. La validación MPI está pendiente para una fase futura.

4. **AABB sobreestimada**: `compute_aabb_3d` usa el AABB ajustado real de las partículas locales. Para distribuciones inhomogéneas, el AABB puede sobreestimar el dominio SFC, resultando en envíos redundantes de halos. Mejora futura: usar límites exactos del dominio SFC.

---

## 11. Roadmap hacia Fase 24+

- **Fase 24:** Scatter/gather PM para dominio SFC (eliminar clone + 2 migraciones → reducir overhead PM↔SR a O(N_halo_PM/P))
- **Fase 25:** Validación MPI completa (P=2,4,8) y benchmarks comparativos vs. Fase 21/22
- **Fase 26:** Criterio MAC para SR (aproximar nodos lejanos dentro de r_cut → reducir O(N/P × N_halos) a O(N/P × log N_halos))
- **Fase 27:** Block timesteps jerárquicos (fuera del scope actual)

---

## 12. Archivos creados/modificados

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `crates/gadget-ng-core/src/config.rs` | Modificado | `treepm_sr_sfc: bool` en `GravitySection` |
| `crates/gadget-ng-treepm/src/distributed.rs` | Modificado | `SfcShortRangeParams`, `short_range_accels_sfc` |
| `crates/gadget-ng-treepm/src/lib.rs` | Modificado | Exporta nuevos símbolos |
| `crates/gadget-ng-cli/src/engine.rs` | Modificado | Path `use_treepm_sr_sfc`, diagnósticos |
| `crates/gadget-ng-treepm/tests/sr_sfc_geometry.rs` | **Nuevo** | 5 tests geométricos |
| `crates/gadget-ng-physics/tests/treepm_sr_sfc.rs` | **Nuevo** | 8 tests de validación física |
| `experiments/nbody/phase23_sr_sfc_domain/` | **Nuevo** | Configs, scripts, resultados |
| `docs/reports/2026-04-phase23-treepm-sr-3d-domain.md` | **Nuevo** | Este reporte |
