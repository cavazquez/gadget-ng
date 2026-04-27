//! Friends-of-Friends (FoF) halo finder.
//!
//! Algoritmo:
//! 1. Construir una grilla de celdas (cell-linked-list) con paso `ll`.
//! 2. Para cada par de partículas en celdas vecinas, enlazar si `|r_ij| < ll`.
//! 3. Usar Union-Find (path-compression + union-by-rank) para agrupar.
//! 4. Calcular propiedades de cada grupo con N ≥ `min_particles`.
//!
//! La longitud de enlace estándar es `ll = b × l̄` con `b = 0.2` y
//! `l̄ = (V/N)^{1/3}` la separación media inter-partícula.

use gadget_ng_core::Vec3;
use std::collections::HashMap;

// ── Union-Find ────────────────────────────────────────────────────────────────

struct Uf {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl Uf {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path-compression
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

// ── Cell-linked-list ──────────────────────────────────────────────────────────

/// Índice de celda 3D → índice lineal.
#[inline]
fn cell_idx(ix: i32, iy: i32, iz: i32, nc: i32) -> usize {
    let ix = ix.rem_euclid(nc) as usize;
    let iy = iy.rem_euclid(nc) as usize;
    let iz = iz.rem_euclid(nc) as usize;
    ix * (nc as usize * nc as usize) + iy * nc as usize + iz
}

struct CellList {
    /// head[cell] = primer índice de partícula en esa celda, o `usize::MAX`.
    head: Vec<usize>,
    /// next[i] = siguiente partícula en la misma celda que i, o `usize::MAX`.
    next: Vec<usize>,
    /// Número de celdas por eje.
    nc: i32,
}

impl CellList {
    fn build(positions: &[Vec3], ll: f64, box_size: f64) -> Self {
        // Número de celdas: al menos 1 por eje, al más box_size/ll.
        let nc = ((box_size / ll).floor() as i32).max(1);
        let cell_size = box_size / nc as f64;
        let n_cells = (nc * nc * nc) as usize;
        let mut head = vec![usize::MAX; n_cells];
        let mut next = vec![usize::MAX; positions.len()];
        for (i, &pos) in positions.iter().enumerate() {
            let ix = (pos.x / cell_size).floor() as i32;
            let iy = (pos.y / cell_size).floor() as i32;
            let iz = (pos.z / cell_size).floor() as i32;
            let c = cell_idx(ix, iy, iz, nc);
            next[i] = head[c];
            head[c] = i;
        }
        Self { head, next, nc }
    }

    /// Itera sobre todos los índices en la celda (ix, iy, iz).
    fn iter_cell(&self, ix: i32, iy: i32, iz: i32) -> CellIter<'_> {
        let c = cell_idx(ix, iy, iz, self.nc);
        CellIter {
            next: &self.next,
            cur: self.head[c],
        }
    }
}

struct CellIter<'a> {
    next: &'a [usize],
    cur: usize,
}

impl Iterator for CellIter<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.cur == usize::MAX {
            return None;
        }
        let out = self.cur;
        self.cur = self.next[self.cur];
        Some(out)
    }
}

// ── Resultado FoF ─────────────────────────────────────────────────────────────

/// Propiedades de un halo FoF.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FofHalo {
    /// Identificador único del halo (entero creciente ordenado por masa desc).
    pub halo_id: usize,
    /// Número de partículas miembro.
    pub n_particles: usize,
    /// Masa total (suma de masas de las partículas miembro).
    pub mass: f64,
    /// Centro de masa — coordenada x.
    pub x_com: f64,
    /// Centro de masa — coordenada y.
    pub y_com: f64,
    /// Centro de masa — coordenada z.
    pub z_com: f64,
    /// Velocidad del centro de masa — vx.
    pub vx_com: f64,
    /// Velocidad del centro de masa — vy.
    pub vy_com: f64,
    /// Velocidad del centro de masa — vz.
    pub vz_com: f64,
    /// Dispersión de velocidades 1D (raíz de la varianza por eje, promediada).
    pub velocity_dispersion: f64,
    /// Radio de virial estimado: `r_vir = (3M / 4π·200·ρ_crit)^{1/3}`.
    /// Requiere pasar `rho_crit`; si `rho_crit = 0`, se devuelve `r_max`
    /// (la distancia máxima de una partícula al COM).
    pub r_vir: f64,
}

/// Ejecuta el FoF sobre las posiciones y velocidades dadas.
///
/// # Parámetros
/// - `positions`: posiciones de las partículas.
/// - `velocities`: velocidades de las partículas (mismo orden).
/// - `masses`: masas de las partículas.
/// - `box_size`: tamaño de la caja (se asume periodicidad).
/// - `b`: parámetro de enlace (típico 0.2).
/// - `min_particles`: número mínimo de partículas para ser un halo.
/// - `rho_crit`: densidad crítica en unidades internas (0 → `r_vir` = `r_max`).
pub fn find_halos(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    box_size: f64,
    b: f64,
    min_particles: usize,
    rho_crit: f64,
) -> Vec<FofHalo> {
    let n = positions.len();
    if n == 0 {
        return Vec::new();
    }

    // Longitud de enlace ll = b × l̄ = b × (V/N)^{1/3}
    let l_mean = (box_size * box_size * box_size / n as f64).cbrt();
    let ll = b * l_mean;
    let ll2 = ll * ll;

    // Construir cell-linked-list.
    let cll = CellList::build(positions, ll, box_size);
    let nc = cll.nc;

    // Union-Find para agrupar partículas.
    let mut uf = Uf::new(n);

    // Recorrer cada celda y sus 27 vecinos (incluyendo sí misma).
    for ix in 0..nc {
        for iy in 0..nc {
            for iz in 0..nc {
                for i in cll.iter_cell(ix, iy, iz) {
                    for dix in -1..=1i32 {
                        for diy in -1..=1i32 {
                            for diz in -1..=1i32 {
                                for j in cll.iter_cell(ix + dix, iy + diy, iz + diz) {
                                    if j <= i {
                                        continue;
                                    }
                                    let dx =
                                        periodic_diff(positions[i].x, positions[j].x, box_size);
                                    let dy =
                                        periodic_diff(positions[i].y, positions[j].y, box_size);
                                    let dz =
                                        periodic_diff(positions[i].z, positions[j].z, box_size);
                                    if dx * dx + dy * dy + dz * dz < ll2 {
                                        uf.union(i, j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Agrupar partículas por raíz.
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        groups.entry(uf.find(i)).or_default().push(i);
    }

    // Calcular propiedades de cada grupo y filtrar por min_particles.
    let mut halos: Vec<FofHalo> = groups
        .values()
        .filter(|g| g.len() >= min_particles)
        .enumerate()
        .map(|(halo_id, members)| {
            halo_props(
                halo_id, members, positions, velocities, masses, box_size, rho_crit,
            )
        })
        .collect();

    // Ordenar por masa descendente y reasignar IDs.
    halos.sort_by(|a, b| b.mass.partial_cmp(&a.mass).unwrap());
    for (i, h) in halos.iter_mut().enumerate() {
        h.halo_id = i;
    }
    halos
}

/// Como [`find_halos`] pero además devuelve la membresía por partícula.
///
/// Retorna `(halos, membership)` donde `membership[i] = Some(halo_idx)` si la
/// partícula `i` pertenece al halo `halo_idx`, o `None` si es campo.
/// El orden de `halos` es por masa descendente (mismo que `find_halos`).
///
/// # Phase 107
///
/// Esta función habilita la construcción de [`crate::merger_tree::ParticleSnapshot`]
/// con `halo_idx` real, resolviendo el bug donde todos los snapshots usaban `None`.
pub fn find_halos_with_membership(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    box_size: f64,
    b: f64,
    min_particles: usize,
    rho_crit: f64,
) -> (Vec<FofHalo>, Vec<Option<usize>>) {
    let n = positions.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let l_mean = (box_size * box_size * box_size / n as f64).cbrt();
    let ll = b * l_mean;
    let ll2 = ll * ll;

    let cll = CellList::build(positions, ll, box_size);
    let nc = cll.nc;
    let mut uf = Uf::new(n);

    for ix in 0..nc {
        for iy in 0..nc {
            for iz in 0..nc {
                for i in cll.iter_cell(ix, iy, iz) {
                    for dix in -1..=1i32 {
                        for diy in -1..=1i32 {
                            for diz in -1..=1i32 {
                                for j in cll.iter_cell(ix + dix, iy + diy, iz + diz) {
                                    if j <= i {
                                        continue;
                                    }
                                    let dx =
                                        periodic_diff(positions[i].x, positions[j].x, box_size);
                                    let dy =
                                        periodic_diff(positions[i].y, positions[j].y, box_size);
                                    let dz =
                                        periodic_diff(positions[i].z, positions[j].z, box_size);
                                    if dx * dx + dy * dy + dz * dz < ll2 {
                                        uf.union(i, j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Construir grupos (raíz → lista de índices).
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        groups.entry(uf.find(i)).or_default().push(i);
    }

    // Calcular propiedades y filtrar.
    let mut halos_with_members: Vec<(FofHalo, Vec<usize>)> = groups
        .values()
        .filter(|g| g.len() >= min_particles)
        .enumerate()
        .map(|(halo_id, members)| {
            let h = halo_props(
                halo_id, members, positions, velocities, masses, box_size, rho_crit,
            );
            (h, members.clone())
        })
        .collect();

    // Ordenar por masa descendente.
    halos_with_members.sort_by(|a, b| b.0.mass.partial_cmp(&a.0.mass).unwrap());

    // Construir membresía por partícula.
    let mut membership: Vec<Option<usize>> = vec![None; n];
    for (new_idx, (h, members)) in halos_with_members.iter_mut().enumerate() {
        h.halo_id = new_idx;
        for &part_idx in members.iter() {
            membership[part_idx] = Some(new_idx);
        }
    }

    let halos = halos_with_members.into_iter().map(|(h, _)| h).collect();
    (halos, membership)
}

/// Construye `ParticleSnapshot` para el merger tree asignando halo_idx desde
/// un catálogo de halos mediante proximidad al centro de masa.
///
/// Para cada partícula, asigna el halo más cercano si la distancia al COM del
/// halo es ≤ `r_vir`. Las partículas fuera de todo halo reciben `halo_idx = None`.
///
/// # Phase 107
///
/// Usado en [`super::merge_tree_cmd`] cuando solo se dispone del catálogo
/// de halos JSONL (sin re-ejecutar FoF).
pub fn particle_snapshots_from_catalog(
    positions: &[Vec3],
    global_ids: &[u64],
    halos: &[FofHalo],
    box_size: f64,
) -> Vec<super::merger_tree::ParticleSnapshot> {
    positions
        .iter()
        .zip(global_ids.iter())
        .map(|(pos, &id)| {
            let mut best_idx: Option<usize> = None;
            let mut best_d2 = f64::INFINITY;
            for (h_idx, h) in halos.iter().enumerate() {
                let dx = periodic_diff(pos.x, h.x_com, box_size);
                let dy = periodic_diff(pos.y, h.y_com, box_size);
                let dz = periodic_diff(pos.z, h.z_com, box_size);
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < h.r_vir * h.r_vir && d2 < best_d2 {
                    best_d2 = d2;
                    best_idx = Some(h_idx);
                }
            }
            super::merger_tree::ParticleSnapshot {
                id,
                halo_idx: best_idx,
            }
        })
        .collect()
}

/// Ejecuta FoF sobre el conjunto combinado `local + halos`, devolviendo solo los
/// grupos cuya raíz Union-Find corresponde a una partícula local (índice < `n_local`).
///
/// Esto permite que cada rank MPI calcule su porción del catálogo sin duplicar halos:
/// - `n_local`: número de partículas propias al inicio de `positions`.
/// - Las partículas con índice `[n_local..)` son "halos" recibidos de vecinos y no
///   generan halos propios (su raíz pertenece a otro rank).
///
/// # Uso
///
/// ```rust,no_run
/// use gadget_ng_analysis::fof::{find_halos_combined, find_halos};
/// use gadget_ng_core::Vec3;
/// // Para P=1 con halos vacíos, produce el mismo resultado que find_halos.
/// let pos = vec![Vec3::new(0.1, 0.1, 0.1), Vec3::new(0.9, 0.9, 0.9)];
/// let vel = vec![Vec3::zero(); 2];
/// let mass = vec![1.0f64; 2];
/// let h1 = find_halos(&pos, &vel, &mass, 1.0, 0.2, 2, 0.0);
/// let h2 = find_halos_combined(&pos, &vel, &mass, 2, 1.0, 0.2, 2, 0.0);
/// assert_eq!(h1.len(), h2.len());
/// ```
#[allow(clippy::too_many_arguments)]
pub fn find_halos_combined(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    n_local: usize,
    box_size: f64,
    b: f64,
    min_particles: usize,
    rho_crit: f64,
) -> Vec<FofHalo> {
    let n = positions.len();
    if n == 0 || n_local == 0 {
        return Vec::new();
    }

    let l_mean = (box_size * box_size * box_size / n as f64).cbrt();
    let ll = b * l_mean;
    let ll2 = ll * ll;

    let cll = CellList::build(positions, ll, box_size);
    let nc = cll.nc;

    let mut uf = Uf::new(n);

    for ix in 0..nc {
        for iy in 0..nc {
            for iz in 0..nc {
                for i in cll.iter_cell(ix, iy, iz) {
                    for dix in -1..=1i32 {
                        for diy in -1..=1i32 {
                            for diz in -1..=1i32 {
                                for j in cll.iter_cell(ix + dix, iy + diy, iz + diz) {
                                    if j <= i {
                                        continue;
                                    }
                                    let dx =
                                        periodic_diff(positions[i].x, positions[j].x, box_size);
                                    let dy =
                                        periodic_diff(positions[i].y, positions[j].y, box_size);
                                    let dz =
                                        periodic_diff(positions[i].z, positions[j].z, box_size);
                                    if dx * dx + dy * dy + dz * dz < ll2 {
                                        uf.union(i, j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Solo groups cuya raíz es local (índice < n_local).
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf.find(i);
        if root < n_local {
            groups.entry(root).or_default().push(i);
        }
    }

    let mut halos: Vec<FofHalo> = groups
        .values()
        .filter(|g| {
            // Al menos min_particles miembros locales.
            let n_local_in_group = g.iter().filter(|&&idx| idx < n_local).count();
            n_local_in_group >= min_particles
        })
        .enumerate()
        .map(|(halo_id, members)| {
            halo_props(
                halo_id, members, positions, velocities, masses, box_size, rho_crit,
            )
        })
        .collect();

    halos.sort_by(|a, b| b.mass.partial_cmp(&a.mass).unwrap());
    for (i, h) in halos.iter_mut().enumerate() {
        h.halo_id = i;
    }
    halos
}

/// Diferencia periódica en [-L/2, L/2].
#[inline]
fn periodic_diff(a: f64, b: f64, l: f64) -> f64 {
    let d = a - b;
    d - l * (d / l).round()
}

fn halo_props(
    halo_id: usize,
    members: &[usize],
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    box_size: f64,
    rho_crit: f64,
) -> FofHalo {
    let mass: f64 = members.iter().map(|&i| masses[i]).sum();
    // Centro de masa con corrección periódica respecto a la primera partícula.
    let ref_pos = positions[members[0]];
    let mut com = Vec3::zero();
    for &i in members {
        let dx = periodic_diff(positions[i].x, ref_pos.x, box_size);
        let dy = periodic_diff(positions[i].y, ref_pos.y, box_size);
        let dz = periodic_diff(positions[i].z, ref_pos.z, box_size);
        com.x += masses[i] * dx;
        com.y += masses[i] * dy;
        com.z += masses[i] * dz;
    }
    com = com * (1.0 / mass) + ref_pos;
    // Envolver al dominio [0, box_size).
    com.x = com.x.rem_euclid(box_size);
    com.y = com.y.rem_euclid(box_size);
    com.z = com.z.rem_euclid(box_size);

    // Velocidad del COM.
    let mut v_com = Vec3::zero();
    for &i in members {
        v_com += velocities[i] * masses[i];
    }
    v_com *= 1.0 / mass;

    // Dispersión de velocidades 1D (σ² promediado sobre x,y,z).
    let mut sig2 = 0.0f64;
    for &i in members {
        let dv = velocities[i] - v_com;
        sig2 += dv.x * dv.x + dv.y * dv.y + dv.z * dv.z;
    }
    let velocity_dispersion = (sig2 / (3.0 * members.len() as f64)).sqrt();

    // Radio de virial: r_vir = (3M / (4π 200 ρ_crit))^{1/3}, o r_max si ρ_crit=0.
    let r_vir = if rho_crit > 0.0 {
        let factor = 3.0 * mass / (4.0 * std::f64::consts::PI * 200.0 * rho_crit);
        factor.cbrt()
    } else {
        // r_max: mayor distancia de una partícula al COM.
        members
            .iter()
            .map(|&i| {
                let dx = periodic_diff(positions[i].x, com.x, box_size);
                let dy = periodic_diff(positions[i].y, com.y, box_size);
                let dz = periodic_diff(positions[i].z, com.z, box_size);
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(0.0f64, f64::max)
    };

    FofHalo {
        halo_id,
        n_particles: members.len(),
        mass,
        x_com: com.x,
        y_com: com.y,
        z_com: com.z,
        vx_com: v_com.x,
        vy_com: v_com.y,
        vz_com: v_com.z,
        velocity_dispersion,
        r_vir,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fof_two_isolated_clusters() {
        // 4 partículas en dos grupos bien separados.
        let pos = vec![
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(0.11, 0.1, 0.1),
            Vec3::new(0.9, 0.9, 0.9),
            Vec3::new(0.91, 0.9, 0.9),
        ];
        let vel = vec![Vec3::zero(); 4];
        let mass = vec![1.0f64; 4];
        let halos = find_halos(&pos, &vel, &mass, 1.0, 0.2, 2, 0.0);
        assert_eq!(halos.len(), 2, "debe encontrar exactamente 2 halos");
        assert_eq!(halos[0].n_particles, 2);
        assert_eq!(halos[1].n_particles, 2);
    }

    #[test]
    fn fof_single_cluster() {
        // 8 partículas muy juntas forman 1 halo.
        let pos: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    0.5 + 0.001 * (i % 2) as f64,
                    0.5 + 0.001 * ((i / 2) % 2) as f64,
                    0.5 + 0.001 * (i / 4) as f64,
                )
            })
            .collect();
        let vel = vec![Vec3::zero(); 8];
        let mass = vec![1.0f64; 8];
        let halos = find_halos(&pos, &vel, &mass, 1.0, 0.2, 2, 0.0);
        assert_eq!(halos.len(), 1);
        assert_eq!(halos[0].n_particles, 8);
    }

    #[test]
    fn fof_min_particles_filter() {
        // 2 partículas separadas; min_particles = 3 → ningún halo.
        let pos = vec![Vec3::new(0.1, 0.1, 0.1), Vec3::new(0.9, 0.9, 0.9)];
        let vel = vec![Vec3::zero(); 2];
        let mass = vec![1.0f64; 2];
        let halos = find_halos(&pos, &vel, &mass, 1.0, 0.2, 3, 0.0);
        assert!(halos.is_empty());
    }
}
