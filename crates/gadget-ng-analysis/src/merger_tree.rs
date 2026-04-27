//! Merger trees de halos FoF: historia de ensamble de masa (Phase 62).
//!
//! ## Algoritmo (single-pass)
//!
//! Dados catálogos FoF de snapshots consecutivos `S_0, S_1, ..., S_N` (del más
//! antiguo al más reciente), para cada par `(S_i, S_{i+1})`:
//!
//! 1. Construir `Map<particle_id → halo_id>` del snapshot más reciente `S_{i+1}`.
//! 2. Para cada halo `H` en `S_i`, votar: contar cuántas de sus partículas
//!    aparecen en cada halo de `S_{i+1}`.
//! 3. **Progenitor principal**: halo de `S_i` que aporta la mayor fracción de
//!    partículas a un halo de `S_{i+1}` (relación "descendente → progenitor").
//! 4. **Merger**: si un halo recibe ≥ `min_shared_fraction` de sus partículas
//!    de dos o más progenitores, los secundarios se registran como mergers.
//!
//! ## Formato de salida
//!
//! ```json
//! {
//!   "nodes": [
//!     {
//!       "snapshot": 1,
//!       "halo_id": 0,
//!       "mass_msun_h": 2.3e14,
//!       "n_particles": 512,
//!       "x_com": [50.0, 50.0, 50.0],
//!       "prog_main_id": 0,
//!       "merger_ids": [2],
//!       "merger_mass_ratio": [0.3]
//!     }
//!   ],
//!   "roots": [0]
//! }
//! ```

use serde::{Deserialize, Serialize};

use crate::fof::FofHalo;

/// Nodo del merger tree: un halo en un snapshot con sus progenitores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergerTreeNode {
    /// Índice del snapshot (0 = más antiguo).
    pub snapshot: usize,
    /// ID del halo dentro de ese snapshot.
    pub halo_id: u64,
    /// Masa en M_sun/h (campo de uso libre; se copia de `FofHalo::mass`).
    pub mass_msun_h: f64,
    /// Número de partículas miembro.
    pub n_particles: usize,
    /// Centro de masa `[x, y, z]`.
    pub x_com: [f64; 3],
    /// ID del progenitor principal en `snapshot + 1` (`None` si es el snapshot más reciente).
    pub prog_main_id: Option<u64>,
    /// IDs de progenitores secundarios (mergers relevantes).
    pub merger_ids: Vec<u64>,
    /// Fracción de masa relativa de cada merger secundario respecto al halo descendente.
    pub merger_mass_ratio: Vec<f64>,
}

/// Colección de todos los árboles de la simulación.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MergerForest {
    /// Todos los nodos de todos los snapshots (orden cronológico).
    pub nodes: Vec<MergerTreeNode>,
    /// Halos raíz: IDs en el snapshot más reciente (z=0).
    pub roots: Vec<u64>,
}

/// Construye el `MergerForest` a partir de catálogos FoF y sus conjuntos de partículas.
///
/// # Parámetros
/// - `catalogs`: lista `(halos, partículas)` ordenada del snapshot **más antiguo**
///   (índice 0, z_max) al **más reciente** (último índice, z=0).  
///   Las partículas deben tener `Particle::id` único para el seguimiento.
/// - `min_shared_fraction`: fracción mínima de partículas compartidas para registrar
///   un progenitor secundario como merger (típico 0.1–0.3).
///
/// # Retorna
/// `MergerForest` con todos los nodos poblados y la lista de halos raíz.
pub fn build_merger_forest(
    catalogs: &[(Vec<FofHalo>, Vec<ParticleSnapshot>)],
    min_shared_fraction: f64,
) -> MergerForest {
    if catalogs.is_empty() {
        return MergerForest::default();
    }

    let n_snaps = catalogs.len();
    let mut all_nodes: Vec<MergerTreeNode> = Vec::new();

    // Inicializar nodos para todos los halos de todos los snapshots.
    for (snap_idx, (halos, _)) in catalogs.iter().enumerate() {
        for h in halos {
            all_nodes.push(MergerTreeNode {
                snapshot: snap_idx,
                halo_id: h.halo_id as u64,
                mass_msun_h: h.mass,
                n_particles: h.n_particles,
                x_com: [h.x_com, h.y_com, h.z_com],
                prog_main_id: None,
                merger_ids: Vec::new(),
                merger_mass_ratio: Vec::new(),
            });
        }
    }

    // Para cada par de snapshots consecutivos, conectar progenitores.
    // Iteramos de más reciente a más antiguo: S_{i+1} → S_i.
    for snap_idx in 0..n_snaps.saturating_sub(1) {
        let next_snap = snap_idx + 1;
        let (next_halos, next_particles) = &catalogs[next_snap];
        let (curr_halos, curr_particles) = &catalogs[snap_idx];

        // Mapa: particle_id → halo_id en el snapshot siguiente.
        let mut pid_to_next_halo: std::collections::HashMap<u64, u64> =
            std::collections::HashMap::new();
        for (h_idx, h) in next_halos.iter().enumerate() {
            // Marcar todas las partículas de este halo.
            for p in next_particles.iter().filter(|p| p.halo_idx == Some(h_idx)) {
                pid_to_next_halo.insert(p.id, h.halo_id as u64);
            }
        }

        // Para cada halo en el snapshot actual, votar progenitores en el siguiente.
        for (h_idx, h) in curr_halos.iter().enumerate() {
            let mut votes: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
            let curr_part: Vec<&ParticleSnapshot> = curr_particles
                .iter()
                .filter(|p| p.halo_idx == Some(h_idx))
                .collect();
            let n_curr = curr_part.len().max(1);

            for p in &curr_part {
                if let Some(&next_halo_id) = pid_to_next_halo.get(&p.id) {
                    *votes.entry(next_halo_id).or_insert(0) += 1;
                }
            }

            if votes.is_empty() {
                continue;
            }

            // Ordenar candidatos por votos descendentes.
            let mut candidates: Vec<(u64, usize)> = votes.into_iter().collect();
            candidates.sort_by(|a, b| b.1.cmp(&a.1));

            // El progenitor principal del halo descendente (en next_snap) es este halo
            // actual si tiene la mayoría de votos. Aquí registramos la relación inversa:
            // "este halo actual" → "progenitor principal en next_snap".
            // Para el nodo del snapshot actual, anotamos el descendente principal.
            let (main_descendant_id, main_votes) = candidates[0];
            let frac_main = main_votes as f64 / n_curr as f64;

            // Actualizar el nodo actual con su descendente principal.
            if let Some(node) = all_nodes
                .iter_mut()
                .find(|n| n.snapshot == snap_idx && n.halo_id == h.halo_id as u64)
            {
                if frac_main >= min_shared_fraction {
                    node.prog_main_id = Some(main_descendant_id);
                }
                // Mergers secundarios.
                for &(sec_id, sec_votes) in candidates.iter().skip(1) {
                    let frac = sec_votes as f64 / n_curr as f64;
                    if frac >= min_shared_fraction {
                        node.merger_ids.push(sec_id);
                        let main_mass = next_halos
                            .iter()
                            .find(|h| h.halo_id as u64 == main_descendant_id)
                            .map(|h| h.mass)
                            .unwrap_or(1.0);
                        let sec_mass = next_halos
                            .iter()
                            .find(|hh| hh.halo_id as u64 == sec_id)
                            .map(|hh| hh.mass)
                            .unwrap_or(0.0);
                        node.merger_mass_ratio.push(sec_mass / main_mass.max(1e-30));
                    }
                }
            }
        }
    }

    // Las raíces son los halos del snapshot más reciente.
    let roots: Vec<u64> = if let Some((last_halos, _)) = catalogs.last() {
        last_halos.iter().map(|h| h.halo_id as u64).collect()
    } else {
        Vec::new()
    };

    MergerForest {
        nodes: all_nodes,
        roots,
    }
}

// ── Historia de Acreción de Masa (MAH) — Phase 67 ────────────────────────────

/// Historia de acreción de masa a lo largo de la rama principal de un halo.
///
/// La rama principal se define como la cadena de progenitores más masivos
/// que lleva desde el halo raíz (z=0) hasta el snapshot más antiguo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassAccretionHistory {
    /// ID del halo raíz (en el snapshot más reciente / último).
    pub halo_id: u64,
    /// Índices de snapshot (0 = más antiguo, N = más reciente).
    pub snapshots: Vec<usize>,
    /// Redshift correspondiente a cada snapshot (`len == snapshots.len()`).
    pub redshifts: Vec<f64>,
    /// Masa del progenitor principal en cada snapshot.
    pub masses: Vec<f64>,
}

/// Extrae la Historia de Acreción de Masa (MAH) a lo largo de la rama principal.
///
/// Recorre el árbol desde la raíz hacia atrás en el tiempo, siguiendo siempre
/// el progenitor principal (`prog_main_id`).
///
/// # Parámetros
/// - `forest`: bosque de merger trees construido con `build_merger_forest`.
/// - `root_halo_id`: ID del halo raíz en el snapshot más reciente.
/// - `redshifts`: `z[snap]` con el mismo orden que `catalogs` pasado a `build_merger_forest`.
///
/// # Retorna
/// `MassAccretionHistory` con los puntos `(z, M)` a lo largo de la rama principal,
/// ordenados del snapshot más reciente al más antiguo.
pub fn mah_main_branch(
    forest: &MergerForest,
    root_halo_id: u64,
    redshifts: &[f64],
) -> MassAccretionHistory {
    // Determinar el snapshot más reciente como punto de partida.
    let max_snap = forest.nodes.iter().map(|n| n.snapshot).max().unwrap_or(0);

    let mut snapshots = Vec::new();
    let mut zs = Vec::new();
    let mut masses = Vec::new();

    // Comenzar en la raíz (snapshot más reciente).
    let mut current_halo_id = root_halo_id;
    let mut current_snap = max_snap;

    loop {
        // Buscar el nodo actual.
        let node = forest
            .nodes
            .iter()
            .find(|n| n.snapshot == current_snap && n.halo_id == current_halo_id);
        let node = match node {
            Some(n) => n,
            None => break,
        };

        snapshots.push(current_snap);
        let z = redshifts.get(current_snap).copied().unwrap_or(0.0);
        zs.push(z);
        masses.push(node.mass_msun_h);

        // Buscar el progenitor principal en el snapshot anterior.
        // El progenitor es el nodo en snap `current_snap - 1` cuyo `prog_main_id == current_halo_id`.
        if current_snap == 0 {
            break;
        }
        let prev_snap = current_snap - 1;
        let progenitor = forest
            .nodes
            .iter()
            .find(|n| n.snapshot == prev_snap && n.prog_main_id == Some(current_halo_id));
        match progenitor {
            Some(prog) => {
                current_halo_id = prog.halo_id;
                current_snap = prev_snap;
            }
            None => break,
        }
    }

    MassAccretionHistory {
        halo_id: root_halo_id,
        snapshots,
        redshifts: zs,
        masses,
    }
}

/// Ajuste analítico de la MAH según McBride et al. (2009).
///
/// Modelo: `M(z) = M₀ · (1+z)^β · exp(−α · z)`
///
/// Parámetros Millennium (McBride+2009): `α ≈ 1.0`, `β ≈ 0.0`.
///
/// # Parámetros
/// - `m0`: masa actual en z=0 [M_sun/h].
/// - `z`: redshift al que evaluar M(z).
/// - `alpha`: parámetro exponencial (default 1.0).
/// - `beta`: exponente de potencia (default 0.0).
pub fn mah_mcbride2009(m0: f64, z: f64, alpha: f64, beta: f64) -> f64 {
    m0 * (1.0 + z).powf(beta) * (-alpha * z).exp()
}

/// Información de una partícula en un snapshot para el merger tree.
///
/// Versión simplificada que solo requiere el ID único de la partícula y a qué
/// halo pertenece (por índice en el vector de halos del snapshot).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleSnapshot {
    /// ID único de la partícula (se usa para seguimiento entre snapshots).
    pub id: u64,
    /// Índice del halo al que pertenece en el catálogo FoF (`None` si es campo).
    pub halo_idx: Option<usize>,
}

impl ParticleSnapshot {
    /// Construye desde una lista de halos FoF y las posiciones/IDs de las partículas.
    ///
    /// Asigna `halo_idx` a cada partícula comparando su GID con los miembros de cada
    /// halo. Para usar en tests donde se conocen los miembros directamente.
    pub fn from_halos_and_ids(
        halos: &[FofHalo],
        particle_ids: &[u64],
        halo_members: &[Vec<usize>],
    ) -> Vec<Self> {
        let mut result: Vec<ParticleSnapshot> = particle_ids
            .iter()
            .map(|&id| ParticleSnapshot { id, halo_idx: None })
            .collect();

        for (h_idx, members) in halo_members.iter().enumerate() {
            if h_idx >= halos.len() {
                break;
            }
            for &part_idx in members {
                if part_idx < result.len() {
                    result[part_idx].halo_idx = Some(h_idx);
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fof::FofHalo;

    fn make_halo(id: usize, n: usize, mass: f64) -> FofHalo {
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
            r_vir: 0.1,
        }
    }

    fn make_particles(n: usize, halo_idx: Option<usize>, id_offset: u64) -> Vec<ParticleSnapshot> {
        (0..n)
            .map(|i| ParticleSnapshot {
                id: id_offset + i as u64,
                halo_idx,
            })
            .collect()
    }

    #[test]
    fn merger_tree_empty_input() {
        let forest = build_merger_forest(&[], 0.5);
        assert!(forest.nodes.is_empty());
        assert!(forest.roots.is_empty());
    }

    #[test]
    fn merger_tree_single_snapshot() {
        let halos = vec![make_halo(0, 10, 1e14)];
        let parts = make_particles(10, Some(0), 0);
        let forest = build_merger_forest(&[(halos, parts)], 0.5);
        assert_eq!(forest.nodes.len(), 1);
        assert_eq!(forest.roots, vec![0]);
    }
}
