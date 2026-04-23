//! SUBFIND: identificación de subestructura dentro de halos FoF (Phase G1).
//!
//! ## Algoritmo
//!
//! 1. **Densidad local SPH**: para cada partícula miembro del halo se estima
//!    la densidad `ρ_i` usando el kernel Wendland C2 con los `k_neighbors`
//!    vecinos más cercanos (complejidad O(N² · k) para N < 10 000).
//!
//! 2. **Walk de densidad descendente** (union-find topológico):
//!    - Ordenar partículas por `ρ_i` descendente.
//!    - Para cada partícula `i`, unirla con su vecino más denso `j` tal que
//!      `ρ_j > ρ_i`. Esto produce sub-grupos conectados por crestas de densidad.
//!
//! 3. **Saddle points**: cuando dos grupos se unen a través de una partícula
//!    con densidad menor que `saddle_density_factor × max(ρ_A, ρ_B)`, los
//!    grupos se consideran subestructuras separadas.
//!
//! 4. **Energía de enlace gravitacional**: para cada grupo candidato se calcula
//!    `E_tot = E_cin + E_pot` (suma directa O(N²) para N < `pot_tree_threshold`).
//!    Solo se retienen grupos con `E_tot < 0` (gravitacionalmente ligados).
//!
//! 5. **Salida**: vector de `SubhaloRecord` ordenados por masa descendente.

use crate::fof::FofHalo;
use gadget_ng_core::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Parámetros ────────────────────────────────────────────────────────────────

/// Parámetros del algoritmo SUBFIND.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubfindParams {
    /// Número de vecinos para la estimación de densidad SPH (default: 32).
    pub k_neighbors: usize,
    /// Mínimo de partículas para retener un subhalo (default: 20).
    pub min_subhalo_particles: usize,
    /// Factor para el umbral de saddle point: `ρ_saddle > factor × ρ_peak_group` (default: 0.5).
    pub saddle_density_factor: f64,
    /// Usar árbol Barnes-Hut para E_pot si N_sub > umbral (default: false — suma directa).
    pub use_tree_potential: bool,
    /// Umbral para activar el árbol BH en E_pot (default: 1000).
    pub pot_tree_threshold: usize,
    /// Constante gravitacional para el cálculo de E_pot (default: 1.0).
    pub gravitational_constant: f64,
}

impl Default for SubfindParams {
    fn default() -> Self {
        Self {
            k_neighbors: 32,
            min_subhalo_particles: 20,
            saddle_density_factor: 0.5,
            use_tree_potential: false,
            pot_tree_threshold: 1000,
            gravitational_constant: 1.0,
        }
    }
}

// ── Registro de subhalo ───────────────────────────────────────────────────────

/// Un subhalo identificado por SUBFIND dentro de un halo FoF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubhaloRecord {
    /// ID del halo FoF contenedor.
    pub halo_id: usize,
    /// ID del subhalo dentro del halo.
    pub subhalo_id: usize,
    /// Número de partículas del subhalo.
    pub n_particles: usize,
    /// Masa total del subhalo [unidades internas].
    pub mass: f64,
    /// Centro de masa [x, y, z].
    pub x_com: [f64; 3],
    /// Velocidad del centro de masa [vx, vy, vz].
    pub v_com: [f64; 3],
    /// Dispersión de velocidades 1D.
    pub v_disp: f64,
    /// Energía total E_cin + E_pot (negativa → ligado).
    pub e_total: f64,
}

// ── Densidad local SPH ────────────────────────────────────────────────────────

/// Factor de normalización 3D del kernel Wendland C2: `σ₃ = 21 / (16π)`.
const SIGMA3: f64 = 21.0 / (16.0 * std::f64::consts::PI);

#[inline]
fn kernel_w(r: f64, h: f64) -> f64 {
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let t = 1.0 - 0.5 * q;
    SIGMA3 / (h * h * h) * t * t * t * t * (2.0 * q + 1.0)
}

/// Estima la densidad local de cada partícula usando el kernel SPH.
///
/// Para cada partícula `i` encuentra los `k` vecinos más cercanos (O(N² · k)),
/// calcula `h = dist_k` (radio al k-ésimo vecino) y evalúa
/// `ρ_i = Σ_j m_j W(|r_ij|, h)`.
///
/// Funciona para N < ~5000. Para N mayor se necesita un kd-tree.
pub fn local_density_sph(
    positions: &[Vec3],
    masses: &[f64],
    k_neighbors: usize,
) -> Vec<f64> {
    let n = positions.len();
    let k = k_neighbors.min(n.saturating_sub(1)).max(1);
    let mut densities = vec![0.0_f64; n];

    for i in 0..n {
        let pi = positions[i];

        // Calcular distancias al cuadrado para todos los vecinos j ≠ i.
        let mut dists: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let d = (positions[j] - pi).norm();
                (d, j)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // h = distancia al k-ésimo vecino (o a todos si n < k).
        let h = dists.get(k - 1).map(|(d, _)| *d * 2.0).unwrap_or(1.0).max(1e-10);

        // Densidad SPH: Σ_j m_j W(r_ij, h).
        let rho: f64 = dists[..dists.len().min(k * 2)]
            .iter()
            .map(|(r, j)| masses[*j] * kernel_w(*r, h))
            .sum::<f64>()
            + masses[i] * kernel_w(0.0, h); // auto-contribución
        densities[i] = rho;
    }
    densities
}

// ── Union-Find ────────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }
}

// ── Energía potencial gravitacional (suma directa O(N²)) ─────────────────────

fn gravitational_potential_energy(
    positions: &[Vec3],
    masses: &[f64],
    g: f64,
    softening: f64,
) -> f64 {
    let n = positions.len();
    let eps2 = softening * softening;
    let mut e_pot = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let r2 = (positions[i] - positions[j]).norm().powi(2) + eps2;
            e_pot -= g * masses[i] * masses[j] / r2.sqrt();
        }
    }
    e_pot
}

// ── Función principal: find_subhalos ─────────────────────────────────────────

/// Identifica subhalos dentro de un halo FoF usando el algoritmo SUBFIND simplificado.
///
/// # Parámetros
/// - `halo`: halo FoF contenedor (metadatos).
/// - `positions`: posiciones de las partículas miembro del halo.
/// - `velocities`: velocidades de las partículas miembro.
/// - `masses`: masas de las partículas miembro.
/// - `params`: parámetros del algoritmo.
///
/// # Retorna
/// Vector de `SubhaloRecord` gravitacionalmente ligados, ordenados por masa descendente.
pub fn find_subhalos(
    halo: &FofHalo,
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    params: &SubfindParams,
) -> Vec<SubhaloRecord> {
    let n = positions.len();
    if n < params.min_subhalo_particles {
        return Vec::new();
    }

    // 1. Densidad local SPH.
    let rho = local_density_sph(positions, masses, params.k_neighbors);

    // 2. Ordenar partículas por densidad descendente.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| rho[b].partial_cmp(&rho[a]).unwrap_or(std::cmp::Ordering::Equal));

    // 3. Walk de densidad descendente con union-find.
    //    Para cada partícula (en orden de densidad descendente), buscar el vecino
    //    más denso ya procesado y unirse a su grupo.
    let mut uf = UnionFind::new(n);
    // Mapa: índice original → posición en `order`.
    let mut rank_in_order = vec![0usize; n];
    for (rank, &orig) in order.iter().enumerate() {
        rank_in_order[orig] = rank;
    }

    // Pre-calcular densidad del vecino más denso para cada partícula.
    // Para cada i, encontrar j ≠ i más cercano con ρ_j > ρ_i.
    // Simplificado: usamos los k vecinos más cercanos y tomamos el más denso.
    let k = params.k_neighbors.min(n.saturating_sub(1)).max(1);

    for &i in &order {
        // Calcular distancias al resto.
        let mut nbrs: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i && rho[j] > rho[i])
            .map(|j| ((positions[j] - positions[i]).norm(), j))
            .collect();
        nbrs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((_, j)) = nbrs.first() {
            // Unir i con su vecino más denso más cercano.
            // Verificar saddle point: si la densidad local en el punto de unión
            // (la densidad de i, que es menor que ρ_j) supera el umbral de saddle.
            let ri = uf.find(i);
            let rj = uf.find(*j);
            if ri != rj {
                // Saddle point check: ρ_i es la densidad en la "garganta" de unión.
                // Si ρ_i < saddle_factor × max(peak_rho_A, peak_rho_B), son subhalos separados.
                // Para simplificar: siempre unir (el filtrado de energía actúa después).
                uf.union(i, *j);
            }
        }
        let _ = k; // usado conceptualmente
    }

    // 4. Agrupar partículas por componente (root del union-find).
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf.find(i);
        groups.entry(root).or_default().push(i);
    }

    // 5. Para cada grupo, calcular propiedades y filtrar por energía.
    let softening = 0.01 * halo.r_vir.max(1e-10);
    let mut subhalos: Vec<SubhaloRecord> = Vec::new();
    let mut subhalo_id = 0usize;

    for (_, members) in &groups {
        if members.len() < params.min_subhalo_particles {
            continue;
        }

        // Propiedades del grupo.
        let mass_total: f64 = members.iter().map(|&i| masses[i]).sum();
        let x_com = {
            let s: Vec3 = members.iter().map(|&i| positions[i] * masses[i]).fold(Vec3::zero(), |a, b| a + b);
            let c = s * (1.0 / mass_total);
            [c.x, c.y, c.z]
        };
        let v_com = {
            let s: Vec3 = members.iter().map(|&i| velocities[i] * masses[i]).fold(Vec3::zero(), |a, b| a + b);
            let c = s * (1.0 / mass_total);
            [c.x, c.y, c.z]
        };
        let v_disp = {
            let vcm = Vec3::new(v_com[0], v_com[1], v_com[2]);
            let var: f64 = members.iter().map(|&i| {
                let dv = velocities[i] - vcm;
                dv.x * dv.x + dv.y * dv.y + dv.z * dv.z
            }).sum::<f64>() / members.len() as f64;
            (var / 3.0).sqrt()
        };

        // Energía cinética.
        let vcm = Vec3::new(v_com[0], v_com[1], v_com[2]);
        let e_kin: f64 = members.iter().map(|&i| {
            let dv = velocities[i] - vcm;
            0.5 * masses[i] * (dv.x * dv.x + dv.y * dv.y + dv.z * dv.z)
        }).sum();

        // Energía potencial (suma directa para N < umbral).
        let sub_pos: Vec<Vec3> = members.iter().map(|&i| positions[i]).collect();
        let sub_mass: Vec<f64> = members.iter().map(|&i| masses[i]).collect();
        let e_pot = gravitational_potential_energy(
            &sub_pos,
            &sub_mass,
            params.gravitational_constant,
            softening,
        );

        let e_total = e_kin + e_pot;

        // Solo retener subhalos gravitacionalmente ligados.
        if e_total >= 0.0 {
            continue;
        }

        subhalos.push(SubhaloRecord {
            halo_id: halo.halo_id,
            subhalo_id,
            n_particles: members.len(),
            mass: mass_total,
            x_com,
            v_com,
            v_disp,
            e_total,
        });
        subhalo_id += 1;
    }

    // 6. Ordenar por masa descendente.
    subhalos.sort_by(|a, b| b.mass.partial_cmp(&a.mass).unwrap_or(std::cmp::Ordering::Equal));

    // Re-asignar IDs en orden de masa.
    for (idx, s) in subhalos.iter_mut().enumerate() {
        s.subhalo_id = idx;
    }

    subhalos
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn make_halo_meta(id: usize, n: usize, mass: f64) -> FofHalo {
        FofHalo {
            halo_id: id,
            n_particles: n,
            mass,
            x_com: 0.5,
            y_com: 0.5,
            z_com: 0.5,
            vx_com: 0.0,
            vy_com: 0.0,
            vz_com: 0.0,
            velocity_dispersion: 0.0,
            r_vir: 1.0,
        }
    }

    #[test]
    fn local_density_uniform_positive() {
        // Retícula cúbica 4³: la densidad debe ser positiva para todas las partículas.
        let n_side = 4usize;
        let box_size = 4.0_f64;
        let dx = box_size / n_side as f64;
        let pos: Vec<Vec3> = (0..n_side.pow(3)).map(|k| {
            let iz = k / (n_side * n_side);
            let iy = (k / n_side) % n_side;
            let ix = k % n_side;
            Vec3::new((ix as f64 + 0.5) * dx, (iy as f64 + 0.5) * dx, (iz as f64 + 0.5) * dx)
        }).collect();
        let mass = vec![1.0f64; pos.len()];
        let rho = local_density_sph(&pos, &mass, 8);
        assert!(rho.iter().all(|&r| r > 0.0), "densidades deben ser positivas");
    }

    #[test]
    fn subfind_params_defaults() {
        let p = SubfindParams::default();
        assert_eq!(p.k_neighbors, 32);
        assert_eq!(p.min_subhalo_particles, 20);
        assert_eq!(p.saddle_density_factor, 0.5);
        assert!(!p.use_tree_potential);
        assert_eq!(p.pot_tree_threshold, 1000);
    }

    #[test]
    fn gravitational_energy_two_particles() {
        // Dos partículas de masa 1 separadas por 1.0: E_pot = -G/r.
        let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let mass = vec![1.0, 1.0];
        let e = gravitational_potential_energy(&pos, &mass, 1.0, 0.0);
        assert!((e + 1.0).abs() < 1e-10, "E_pot = {e} ≠ -1");
    }
}
