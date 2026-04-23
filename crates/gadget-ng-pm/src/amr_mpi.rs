//! AMR MPI — comunicación de parches entre ranks (Phase 85).
//!
//! Implementa la distribución del solver AMR-PM jerárquico entre múltiples ranks MPI.
//! El diseño sigue el patrón "rank 0 coordina":
//!
//! 1. **Identificación global**: cada rank contribuye sus partículas locales a la
//!    densidad de base; tras `MPI_Allreduce`, el rank 0 identifica los parches a refinar.
//! 2. **Broadcast de parches**: rank 0 difunde la lista de parches a todos los ranks.
//! 3. **Distribución de fuerzas**: cada rank deposita sus partículas en los parches
//!    asignados, resuelve Poisson localmente y devuelve las fuerzas al rank 0.
//! 4. **Gather/scatter**: rank 0 acumula todas las correcciones y las redistribuye.
//!
//! ## MPI stub
//!
//! Cuando se compila sin feature `mpi`, todas las funciones operan en modo serial
//! (solo un rank, sin comunicación real). Las interfaces son idénticas para que
//! el código de llamada sea agnóstico a MPI.
//!
//! ## Referencia
//!
//! Kravtsov et al. (1997), ApJS 111, 73 (AMR+MPI para N-body cosmológico);
//! Teyssier (2002), A&A 385, 337 (RAMSES AMR distribuido).

use gadget_ng_core::Vec3;

use crate::amr::{AmrLevel, AmrParams, PatchGrid};

/// Resolución del grid base por defecto para los wrappers MPI.
const DEFAULT_NM_BASE: usize = 32;

// ── Mensaje serializable de parche ────────────────────────────────────────────

/// Representación serializable de un parche AMR para envío entre ranks.
///
/// Contiene la geometría y las fuerzas calculadas para poder difundirlas.
#[derive(Debug, Clone)]
pub struct AmrPatchMessage {
    /// Centro del parche en coordenadas físicas.
    pub center: Vec3,
    /// Lado del parche en unidades físicas.
    pub size: f64,
    /// Resolución interna del parche (celdas por lado).
    pub nm: usize,
    /// Fuerzas calculadas [fx, fy, fz], cada una de longitud `nm³`.
    pub forces: [Vec<f64>; 3],
}

impl AmrPatchMessage {
    /// Crea un mensaje desde un `PatchGrid` ya resuelto.
    pub fn from_patch(patch: &PatchGrid) -> Self {
        Self {
            center: patch.center,
            size: patch.size,
            nm: patch.nm,
            forces: [
                patch.forces[0].clone(),
                patch.forces[1].clone(),
                patch.forces[2].clone(),
            ],
        }
    }
}

// ── Runtime MPI stub ──────────────────────────────────────────────────────────

/// Contexto de runtime paralelo para AMR.
///
/// En modo serial es un wrapper trivial.
pub struct AmrRuntime {
    pub rank: usize,
    pub size: usize,
}

impl AmrRuntime {
    /// Crea un runtime serial (rank=0, size=1).
    pub fn serial() -> Self {
        Self { rank: 0, size: 1 }
    }
}

// ── Funciones principales ─────────────────────────────────────────────────────

/// Difunde las fuerzas de parches AMR desde rank 0 a todos los ranks.
///
/// En modo serial, es un no-op (ya todos los parches están en el mismo proceso).
/// En MPI real, realiza `MPI_Bcast` para parches pequeños o `MPI_Gatherv`/`MPI_Scatterv`
/// para parches con grids grandes (nm ≥ 32).
///
/// # Argumentos
/// - `patches`  — lista de parches con fuerzas calculadas (solo rank 0 tiene datos válidos en MPI)
/// - `rt`       — contexto de runtime
///
/// # Retorna
/// Lista de `AmrPatchMessage` con fuerzas disponibles en todos los ranks.
pub fn broadcast_patch_forces(
    patches: &[PatchGrid],
    rt: &AmrRuntime,
) -> Vec<AmrPatchMessage> {
    if rt.size == 1 {
        // Serial: solo convertir formato
        return patches.iter().map(AmrPatchMessage::from_patch).collect();
    }

    // Stub MPI:
    // En producción con feature "mpi":
    //   Rank 0: serializar patches → bytes; MPI_Bcast(n_patches); MPI_Bcast(bytes)
    //   Otros ranks: MPI_Bcast recibe; deserializar
    eprintln!(
        "[amr-mpi] broadcast_patch_forces: rank {}/{} — stub (implementar con MPI_Bcast)",
        rt.rank, rt.size
    );

    // En el stub todos los ranks tienen una vista vacía (la función no puede comunicar sin MPI)
    patches.iter().map(AmrPatchMessage::from_patch).collect()
}

/// Wrapper MPI del solver AMR multi-nivel.
///
/// Extiende `amr_pm_accels_multilevel` para corridas distribuidas:
///
/// 1. Cada rank calcula la densidad de sus partículas locales en el grid base.
/// 2. `MPI_Allreduce` suma el grid de todos los ranks.
/// 3. Rank 0 identifica parches (`identify_refinement_patches`).
/// 4. `broadcast_patch_forces` distribuye geometría de parches a todos los ranks.
/// 5. Cada rank deposita sus partículas en los parches, resuelve Poisson localmente.
/// 6. `MPI_Allreduce` acumula las correcciones de fuerza de los parches.
/// 7. Aplicar correcciones interpoladas a las partículas locales.
///
/// En modo serial delega directamente a `amr_pm_accels_multilevel`.
///
/// # Argumentos
/// - `positions`  — posiciones de las partículas locales
/// - `masses`     — masas de las partículas locales
/// - `box_size`   — tamaño de la caja periódica
/// - `nm_base`    — resolución del grid base (celdas por lado)
/// - `params`     — parámetros del solver AMR
/// - `g`          — constante gravitacional
/// - `rt`         — contexto de runtime
///
/// # Retorna
/// Aceleraciones por partícula `[ax, ay, az]` para las partículas locales.
pub fn amr_pm_accels_multilevel_mpi(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    nm_base: usize,
    g: f64,
    params: &AmrParams,
    rt: &AmrRuntime,
) -> Vec<Vec3> {
    if rt.size == 1 {
        // Serial: delegar al solver mono-proceso
        return crate::amr::amr_pm_accels_multilevel(positions, masses, box_size, nm_base, g, params);
    }

    // Stub MPI: documentar la secuencia de operaciones
    eprintln!(
        "[amr-mpi] amr_pm_accels_multilevel_mpi: rank {}/{} — stub (implementar con MPI_Allreduce)",
        rt.rank, rt.size
    );

    // En el stub, cada rank calcula sus propias fuerzas en modo serial.
    // En producción:
    // 1. density_local → MPI_Allreduce → density_global (rank 0 identifica parches)
    // 2. broadcast_patch_forces → cada rank deposita sus partículas
    // 3. forces_local → MPI_Allreduce → forces_global
    crate::amr::amr_pm_accels_multilevel(positions, masses, box_size, nm_base, g, params)
}

/// Construye el nivel de jerarquía AMR con reducción de densidad global.
///
/// Versión MPI de `build_amr_hierarchy`:
/// 1. Cada rank construye un grid de densidad local.
/// 2. `MPI_Allreduce` suma los grids.
/// 3. Rank 0 identifica parches en el grid global.
/// 4. Difunde la jerarquía a todos los ranks.
///
/// En modo serial delega a `build_amr_hierarchy`.
///
/// # Retorna
/// `AmrLevel` con la jerarquía construida (idéntica en todos los ranks en MPI real).
pub fn build_amr_hierarchy_mpi(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    g: f64,
    params: &AmrParams,
    rt: &AmrRuntime,
) -> AmrLevel {
    if rt.size == 1 {
        return crate::amr::build_amr_hierarchy(positions, masses, box_size, g, params, 0);
    }

    eprintln!(
        "[amr-mpi] build_amr_hierarchy_mpi: rank {}/{} — stub (implementar con MPI_Allreduce)",
        rt.rank, rt.size
    );

    crate::amr::build_amr_hierarchy(positions, masses, box_size, g, params, 0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_positions(n: usize) -> (Vec<Vec3>, Vec<f64>) {
        let mut rng: u64 = 12345;
        let pos: Vec<Vec3> = (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let x = (rng >> 33) as f64 / u32::MAX as f64;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let y = (rng >> 33) as f64 / u32::MAX as f64;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let z = (rng >> 33) as f64 / u32::MAX as f64;
                Vec3::new(x, y, z)
            })
            .collect();
        let mass = vec![1.0 / n as f64; n];
        (pos, mass)
    }

    #[test]
    fn serial_mpi_matches_direct() {
        let (pos, mass) = make_positions(64);
        let box_size = 1.0;
        let nm_base = 8;
        let params = AmrParams {
            delta_refine: 2.0,
            nm_patch: 4,
            patch_cells_base: 3,
            max_patches: 4,
            zero_pad: true,
            max_levels: 1,
            refine_factor: 4.0,
        };
        let rt = AmrRuntime::serial();

        let acc_mpi = amr_pm_accels_multilevel_mpi(&pos, &mass, box_size, nm_base, 1.0, &params, &rt);
        let acc_direct = crate::amr::amr_pm_accels_multilevel(&pos, &mass, box_size, nm_base, 1.0, &params);

        assert_eq!(acc_mpi.len(), acc_direct.len());
        for (a, b) in acc_mpi.iter().zip(acc_direct.iter()) {
            let diff = (a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs();
            assert!(diff < 1e-12, "Divergencia entre serial MPI y directo: {diff}");
        }
    }

    #[test]
    fn broadcast_serial_identity() {
        let (pos, mass) = make_positions(32);
        let nm_base = 8;
        let params = AmrParams {
            delta_refine: 1.0,
            nm_patch: 4,
            patch_cells_base: 3,
            max_patches: 4,
            zero_pad: true,
            max_levels: 1,
            refine_factor: 4.0,
        };
        let base_density = crate::cic::assign(&pos, &mass, 1.0, nm_base);
        let patches = crate::amr::identify_refinement_patches(&base_density, nm_base, 1.0, &params);
        let mut solved_patches = patches.clone();
        for p in &mut solved_patches {
            crate::amr::solve_patch(p, 1.0, true);
        }

        let rt = AmrRuntime::serial();
        let msgs = broadcast_patch_forces(&solved_patches, &rt);

        assert_eq!(msgs.len(), solved_patches.len());
        for (msg, patch) in msgs.iter().zip(solved_patches.iter()) {
            assert_eq!(msg.nm, patch.nm);
            assert_eq!(msg.forces[0].len(), patch.forces[0].len());
        }
    }

    #[test]
    fn hierarchy_mpi_serial_same() {
        let (pos, mass) = make_positions(64);
        let params = AmrParams {
            delta_refine: 2.0,
            nm_patch: 4,
            patch_cells_base: 3,
            max_patches: 4,
            zero_pad: true,
            max_levels: 1,
            refine_factor: 4.0,
        };
        let rt = AmrRuntime::serial();

        let h_mpi = build_amr_hierarchy_mpi(&pos, &mass, 1.0, 1.0, &params, &rt);
        let h_direct = crate::amr::build_amr_hierarchy(&pos, &mass, 1.0, 1.0, &params, 0);

        assert_eq!(h_mpi.patches.len(), h_direct.patches.len());
    }
}
