# Tareas Grandes — Trabajos futuros (3–5 sesiones c/u)

**Fecha de redacción:** abril 2026  
**Estado del proyecto al momento de redacción:** Phase 60 + Rápidas completadas.  
**Propósito:** Documento de referencia para planificación futura. No está comprometido a un sprint.

---

## Contexto

Con las Phases 1–60 completadas, el simulador cubre el pipeline completo de una
corrida cosmológica ΛCDM: ICs 2LPT/EH → integración leapfrog/Yoshida4 cosmológica
→ TreePM/PM distribuido → análisis (FoF, P(k), ξ(r), c(M), HMF) → checkpoint robusto
→ GPU PM (CUDA/HIP) → domain decomposition adaptativa.

Las siguientes son tareas de gran envergadura que requieren diseño cuidadoso,
posiblemente varios sprints, y en algunos casos decisiones de arquitectura no triviales.

---

## G1 — SUBFIND: subestructura dentro de halos

**Estimación:** 3–4 sesiones  
**Dependencias:** FoF (Phase 14), NFW (Phase 53), FoF paralelo MPI (Mediana 61)  
**Crates afectados:** `gadget-ng-analysis`, `gadget-ng-cli`

### Descripción

Implementar un algoritmo de búsqueda de subestructura similar a SUBFIND
(Springel et al. 2001). El objetivo es encontrar, dentro de cada halo FoF,
subgrupos gravitacionalmente ligados (subhalos).

### Algoritmo propuesto

```
Para cada halo FoF con N > N_min:
  1. Ordenar partículas por densidad local (estimada con SPH kernel o CIC).
  2. Walk descendiente de densidad: construcción del árbol de saddle points.
  3. Umbral de ligadura gravitacional: calcular E_pot + E_cin para cada candidato.
  4. Retener solo grupos con E_total < 0.
  5. Emitir SubhaloRecord { halo_id, subhalo_id, n_part, mass, x_com, v_com, v_disp }.
```

### API target

```rust
pub fn find_subhalos(
    halo: &FofHalo,
    particles: &[Particle],
    params: &SubfindParams,
) -> Vec<SubhaloRecord>;
```

### Salida

- Catálogo `subhalos.jsonl` por snapshot.
- Integración con `gadget-ng analyze` como opción `--subfind`.
- Estadísticas: función de masa de subhalos, perfil de subhalos dentro del halo host.

### Complejidad

Alta. La ligadura gravitacional requiere O(N²) o un árbol BH adicional por halo.
Para halos masivos (N > 10⁴) es necesario paralelizar con Rayon o delegar a GPU.

---

## G2 — SPH cosmológico integrado

**Estimación:** 4–5 sesiones  
**Dependencias:** SPH (Phase 16), cosmología serial (Phase 17a), TreePM (Phase 21–25)  
**Crates afectados:** `gadget-ng-sph`, `gadget-ng-integrators`, `gadget-ng-cli`, `gadget-ng-physics`

### Descripción

El crate `gadget-ng-sph` implementa el kernel SPH, densidad adaptativa y viscosidad
artificial, pero está desacoplado del pipeline cosmológico. Esta tarea lo integra:

1. **Acoplamiento cosmológico:** adaptar `sph_kdk_step` para usar
   `drift_kick_factors` cosmológicos (`∫dt'/a²`, `∫dt'/a`).
2. **Dos tipos de partículas:** soporte explícito para `ParticleType::DarkMatter`
   y `ParticleType::Gas` en el motor `engine.rs`; solo las partículas Gas
   participan en SPH.
3. **Cooling e ionización (simplificado):** función de enfriamiento atómico
   Λ(T) = Λ₀·T^β para H+He; temperatura de equilibrio en 10⁴ K (fotoionización UV).
4. **Checkpoint:** guardar `u` (energía interna) y `h_sml` en checkpoint.
5. **Tests:** reproducción de la prueba de Sedov-Taylor en caja periódica,
   temperatura virial de halo de gas.

### Configuración target

```toml
[sph]
enabled       = true
gamma         = 1.6667
alpha_visc    = 1.0
n_neigh       = 32
cooling       = "atomic_h_he"   # "none" | "atomic_h_he"
t_floor_k     = 1e4             # temperatura mínima en K
```

### Validación

- Prueba de Sedov-Taylor: perfil de energía radial vs solución analítica.
- Temperatura media del gas vs temperatura virial del halo DM subyacente.
- Conservación de energía total (DM + gas) a < 1% en 100 pasos.

---

## G3 — Corrida de producción N=256³ end-to-end

**Estimación:** 2–3 sesiones (infraestructura + corrida real)  
**Dependencias:** Todas las fases de simulación completadas; idealmente CUDA/HIP (Phase 57)  
**Crates afectados:** `gadget-ng-cli`, `gadget-ng-io`, `gadget-ng-physics`

### Descripción

Primera corrida de producción real a resolución N=256³ (16.7 millones de
partículas), desde z=49 hasta z=0, con múltiples snapshots y análisis completo.

### Requerimientos técnicos

| Parámetro | Valor objetivo |
|-----------|----------------|
| N | 256³ = 16,777,216 |
| BOX | 300 Mpc/h |
| z_init | 49 |
| z_final | 0 |
| Solver | TreePM con CUDA o MPI × 4 ranks |
| Snapshots | 20 (z = 49, 10, 5, 3, 2, 1.5, 1, 0.5, 0.3, 0.1, 0) |
| Análisis | FoF (b=0.2) + P(k) + ξ(r) + HMF cada snapshot |
| Wall time estimado | ~4–8h en 1 GPU NVIDIA o ~12h en 4 CPU ranks |

### Infraestructura necesaria

- Script `scripts/run_production_256.sh` con checkpointing automático cada 2h.
- I/O: snapshots en HDF5 comprimido (requiere G4 — HDF5 paralelo).
- Post-proceso: notebooks Python para figuras de paper: P(k)(z), HMF(z), ξ(r)(z=0).
- Comparación final contra CAMB + HMF Sheth-Tormen + ξ(r) lineal.

### Métricas de éxito

- `median|log10(P_c/P_CLASS)| < 0.05` en todos los snapshots.
- Masa mínima resuelta: `M_min = 50 × m_part ≈ 1.4×10¹¹ M_sun/h`.
- HMF dentro de factor 2 de ST a z=0 para M > M_min.

---

## G4 — AMR-PM: refinamiento adaptativo de la malla PM

**Estimación:** 5+ sesiones  
**Dependencias:** PM slab (Phase 20), PM pencil 2D (Phase 46), TreePM (Phase 21–25)  
**Crates afectados:** `gadget-ng-pm`, `gadget-ng-treepm`, `gadget-ng-core`

### Descripción

La grilla PM actual es uniforme. En regiones de alta densidad (halos), la resolución
de la malla PM limita la fuerza de corto alcance. AMR-PM subdivide la malla
jerárquicamente donde la densidad supera un umbral.

### Diseño propuesto

```
Nivel 0: grilla base nm³ (global, periódica)
Nivel 1: parches nm_patch³ en celdas con overdensidad > δ_refine
Nivel 2: sub-parches recursivos hasta nivel máximo L_max
```

Cada nivel usa su propia FFT periódica sobre el parche. La fuerza total es la
superposición de las soluciones de Poisson en cada nivel, con suavizado de
transición (tipo multigrid V-cycle).

### Complejidad

Muy alta. Requiere:
- Estructura de datos jerárquica de parches (tipo Chombo/AMReX simplificado).
- Protocolo de comunicación MPI para parches distribuidos.
- Integración con el integrador jerárquico (block timesteps por nivel AMR).

### Alternativa simplificada

Para una primera iteración: **TreePM adaptativo** — aumentar la resolución de la
grilla PM solo en la región del halo más masivo, sin jerarquía completa.
Estimación: 2 sesiones.

---

## G5 — Merger trees: historia de ensamble de masa

**Estimación:** 3 sesiones  
**Dependencias:** FoF paralelo (Mediana 61), múltiples snapshots  
**Crates afectados:** `gadget-ng-analysis`, `gadget-ng-cli`

### Descripción

Conectar catálogos FoF de snapshots consecutivos para rastrear la historia
de cada halo: identificar progenitores, mergers, pérdida de masa por stripping.

### Algoritmo

```
Para snapshots S_i y S_{i+1}:
  1. Crear diccionario: particle_id → halo_id en S_{i+1}.
  2. Para cada halo H en S_i:
     a. Contar cuántas de sus partículas aparecen en cada halo de S_{i+1}.
     b. Progenitor principal = halo de S_{i+1} con mayor fracción compartida.
     c. Registrar MergerEvent si múltiples progenitores aportan > f_min.
  3. Emitir árbol como grafo dirigido: { halo_id, snapshot, prog_id, mass }.
```

### Formato de salida

```json
{
  "trees": [
    {
      "root_halo_id": 0, "root_snapshot": 20,
      "nodes": [
        {"snap": 20, "halo_id": 0, "mass": 2.3e14, "prog_ids": [1, 3]},
        {"snap": 19, "halo_id": 1, "mass": 1.8e14, "prog_ids": [2]},
        ...
      ]
    }
  ]
}
```

### Uso desde CLI

```bash
gadget-ng merge-tree \
  --snapshots "runs/cosmo/snap_*.jsonl" \
  --catalogs  "runs/cosmo/halos_*.jsonl" \
  --out runs/cosmo/merger_tree.json
```

### Métricas de validación

- MAH (Mass Accretion History): M(z) del halo más masivo vs ajuste McBride+2009.
- Tasa de mergers: dN_merge/dz vs Fakhouri+2010 (simulaciones Millennium).

---

## Tabla resumen

| ID | Tarea | Sesiones | Prioridad | Dependencias |
|----|-------|----------|-----------|--------------|
| G1 | SUBFIND / subhalos | 3–4 | Media | FoF paralelo (M61) |
| G2 | SPH cosmológico integrado | 4–5 | Alta | SPH existente + cosmo |
| G3 | Corrida producción N=256³ | 2–3 | Alta | CUDA/HIP (P57) + HDF5‖ |
| G4 | AMR-PM | 5+ | Baja | PM avanzado |
| G5 | Merger trees | 3 | Media | FoF paralelo (M61) |

---

## Orden de ataque recomendado

```
G2 (SPH cosmo)  →  G3 (producción N=256³)  →  G5 (merger trees)  →  G1 (SUBFIND)  →  G4 (AMR)
    ↑
requiere antes:
  Phase 61–65 (Medianas)
  especialmente M61 (FoF MPI) y M65 (HDF5 paralelo)
```

**Próximo paso inmediato:** completar las Medianas (Phases 61–65) antes de abordar
cualquier Grande, ya que FoF paralelo y HDF5 paralelo son dependencias base de G1–G3.
