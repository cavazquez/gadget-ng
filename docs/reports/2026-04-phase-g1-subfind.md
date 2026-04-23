# Phase G1 — SUBFIND: subestructura dentro de halos FoF

**Fecha:** 2026-04-23  
**Crates afectados:** `gadget-ng-analysis`, `gadget-ng-cli`  
**Tests:** 6 / 6 ✅

---

## Resumen

Implementación del algoritmo SUBFIND simplificado para identificar subestructuras
gravitacionalmente ligadas (subhalos) dentro de los halos FoF encontrados por el
halo finder (Phase 7). El algoritmo combina estimación de densidad SPH local,
un walk de densidad descendente con union-find, y filtrado por energía de enlace.

---

## Implementación: `gadget-ng-analysis/src/subfind.rs`

### Tipos públicos

```rust
pub struct SubfindParams {
    pub k_neighbors: usize,           // default: 32
    pub min_subhalo_particles: usize, // default: 20
    pub saddle_density_factor: f64,   // default: 0.5
    pub use_tree_potential: bool,     // default: false
    pub pot_tree_threshold: usize,    // default: 1000
    pub gravitational_constant: f64,  // default: 1.0
}

pub struct SubhaloRecord {
    pub halo_id: usize,
    pub subhalo_id: usize,
    pub n_particles: usize,
    pub mass: f64,
    pub x_com: [f64; 3],
    pub v_com: [f64; 3],
    pub v_disp: f64,
    pub e_total: f64,   // < 0 → gravitacionalmente ligado
}
```

### API pública

```rust
pub fn local_density_sph(
    positions: &[Vec3],
    masses: &[f64],
    k_neighbors: usize,
) -> Vec<f64>

pub fn find_subhalos(
    halo: &FofHalo,
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    params: &SubfindParams,
) -> Vec<SubhaloRecord>
```

---

## Algoritmo

```
1. local_density_sph: para cada partícula i
   → distancias a todos los j ≠ i
   → h_i = 2 × dist_{k-ésimo vecino}
   → ρ_i = Σ_j m_j W(|r_ij|, h_i)   (kernel Wendland C2)

2. Walk de densidad descendente (union-find):
   → Ordenar partículas por ρ_i descendente
   → Para cada i: unir con el vecino más denso j con ρ_j > ρ_i más cercano

3. Agrupar por componentes del union-find

4. Para cada grupo con N ≥ min_subhalo_particles:
   → Calcular CoM, v_CoM, dispersión de velocidades
   → E_cin = Σ ½ m_i |v_i - v_CoM|²
   → E_pot = -G Σ_{i<j} m_i m_j / |r_ij|   (suma directa)
   → Retener si E_cin + E_pot < 0

5. Ordenar por masa descendente, re-asignar IDs
```

---

## Integración CLI: `gadget-ng analyze --subfind`

```bash
gadget-ng analyze \
  --snapshot out/snapshot_final \
  --output analysis/results.json \
  --subfind \
  --subfind-min-particles 30
```

El campo `subfind` en `results.json` contiene:
```json
{
  "subfind": [
    {
      "halo_id": 0,
      "n_subhalos": 2,
      "subhalos": [
        { "subhalo_id": 0, "n_particles": 45, "mass": 45.0,
          "x_com": [...], "v_com": [...], "v_disp": 0.01, "e_total": -12.3 },
        ...
      ]
    }
  ]
}
```

---

## Tests

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `subfind_params_defaults` | Valores default de `SubfindParams` | ✅ |
| `local_density_concentrated` | Densidad interna > densidad periférica | ✅ |
| `subfind_single_isolated_cluster` | Cluster compacto → ≥ 1 subhalo ligado | ✅ |
| `subfind_two_subclusters` | Dos clusters separados → subestructura detectada | ✅ |
| `subfind_mass_conservation` | Σ masa(subhalos) ≤ masa(halo host) | ✅ |
| `subfind_binding_energy_negative` | E_tot < 0 para todos los subhalos retornados | ✅ |

---

## Limitaciones conocidas y trabajo futuro

- **Complejidad O(N²)**: la estimación de densidad y E_pot son O(N²). Para halos
  con N > 5000 se necesita un kd-tree para densidad y árbol BH para E_pot
  (implementable con `use_tree_potential = true`).
- **Saddle points simplificados**: el filtrado de saddle points está simplificado;
  el SUBFIND original de Springel+2001 usa un criterio más sofisticado basado
  en el valle de densidad relativo entre grupos.
- **MPI**: la versión actual es serial. Para datos distribuidos se necesita
  recolectar los miembros del halo en un rank antes de llamar a `find_subhalos`.
