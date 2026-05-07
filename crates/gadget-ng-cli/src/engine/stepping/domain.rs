//! Domain decomposition configuration for PM/TreePM cosmological paths.
//!
//! Encapsulates the boolean flags and layout precomputations (Fases 18–24, 46)
//! that control which solver path is active in the PM/TreePM branch.

use gadget_ng_core::{RunConfig, SolverKind};
use gadget_ng_parallel::SfcDecomposition;
use gadget_ng_pm::slab_fft::SlabLayout;
use gadget_ng_pm::PencilLayout2D;

use crate::error::CliError;

/// Domain decomposition configuration for the PM/TreePM cosmological branch.
#[derive(Debug)]
pub(crate) struct PmTreepmDomain {
    pub pm_nm: usize,
    /// PM distribuido sin slab.
    pub use_pm_dist: bool,
    /// PM slab 1D (Fase 20).
    pub use_pm_slab: bool,
    /// PM pencil 2D (Fase 46, P > nm).
    pub use_pm_pencil2d: bool,
    /// TreePM slab distribuido (Fase 21).
    pub use_treepm_slab: bool,
    /// Halo volumétrico 3D periódico (Fase 22).
    pub use_treepm_3d_halo: bool,
    /// SFC para SR desacoplado del slab-z (Fase 23).
    pub use_treepm_sr_sfc: bool,
    /// Scatter/gather PM mínimo (Fase 24).
    pub use_treepm_pm_scatter_gather: bool,
    /// Layout para PM slab.
    pub slab_layout_opt: Option<SlabLayout>,
    /// Layout para PM pencil 2D.
    pub pencil_layout_opt: Option<PencilLayout2D>,
    /// Layout para TreePM slab.
    pub treepm_slab_layout_opt: Option<SlabLayout>,
    /// Radio de splitting Gaussiano TreePM.
    pub treepm_r_split: f64,
    /// Cutoff de corto alcance: r_cut = 5·r_split.
    pub treepm_r_cut: f64,
    /// SfcDecomposition para el dominio SR (Fase 23).
    pub sr_sfc_decomp_opt: Option<SfcDecomposition>,
    /// Intervalo de rebalanceo SFC SR.
    pub sr_sfc_rebalance: u64,
    /// SfcKind para SR.
    pub sr_sfc_kind: gadget_ng_core::SfcKind,
}

impl PmTreepmDomain {
    /// Construye la configuración de dominio a partir del `RunConfig` y el runtime.
    ///
    /// Valida parámetros (grid size % ranks == 0, factorización pencil 2D, etc.)
    /// y precomputa layouts y descomposiciones SFC.
    pub fn from_cfg<R: gadget_ng_parallel::ParallelRuntime + ?Sized>(
        cfg: &RunConfig,
        rt: &R,
        local: &[gadget_ng_core::Particle],
    ) -> Result<Self, CliError> {
        let box_size = cfg.simulation.box_size;
        let pm_nm = cfg.gravity.pm_grid_size;

        let _cosmo_periodic = cfg.cosmology.periodic;

        // ── Flags de dominio ──────────────────────────────────────────────────
        let use_pm_dist = cfg.gravity.pm_distributed
            && !cfg.gravity.pm_slab
            && cfg.cosmology.periodic
            && cfg.gravity.solver == SolverKind::Pm;

        let use_pm_slab = cfg.gravity.pm_slab
            && cfg.cosmology.periodic
            && cfg.gravity.solver == SolverKind::Pm
            && (rt.size() as usize) <= pm_nm;

        let use_pm_pencil2d = cfg.gravity.pm_slab
            && cfg.cosmology.periodic
            && cfg.gravity.solver == SolverKind::Pm
            && (rt.size() as usize) > pm_nm;

        let use_treepm_slab = cfg.gravity.treepm_slab
            && cfg.cosmology.periodic
            && cfg.gravity.solver == SolverKind::TreePm;

        let use_treepm_3d_halo = cfg.gravity.treepm_halo_3d && use_treepm_slab;

        let use_treepm_sr_sfc = cfg.gravity.treepm_sr_sfc && use_treepm_slab;

        let use_treepm_pm_scatter_gather =
            cfg.gravity.treepm_pm_scatter_gather && use_treepm_sr_sfc;

        // ── Layouts ──────────────────────────────────────────────────────────
        let slab_layout_opt: Option<SlabLayout> = if use_pm_slab {
            Some(SlabLayout::new(
                pm_nm,
                rt.rank() as usize,
                rt.size() as usize,
            ))
        } else {
            None
        };

        let pencil_layout_opt: Option<PencilLayout2D> = if use_pm_pencil2d {
            let p = rt.size() as usize;
            let (py, pz) = PencilLayout2D::factorize(pm_nm, p);
            if py * pz != p || !pm_nm.is_multiple_of(py) || !pm_nm.is_multiple_of(pz) {
                return Err(CliError::InvalidConfig(format!(
                    "pencil_2d: no existe factorización válida para nm={pm_nm} y P={p}. \
                     Se requiere P ≤ nm² con nm % py == 0 y nm % pz == 0."
                )));
            }
            Some(PencilLayout2D::new(pm_nm, rt.rank() as usize, py, pz))
        } else {
            None
        };

        let treepm_slab_layout_opt: Option<SlabLayout> = if use_treepm_slab {
            let nm = pm_nm;
            let p = rt.size() as usize;
            if !nm.is_multiple_of(p) {
                return Err(CliError::InvalidConfig(format!(
                    "treepm_slab requiere pm_grid_size ({nm}) % n_ranks ({p}) == 0"
                )));
            }
            Some(SlabLayout::new(nm, rt.rank() as usize, p))
        } else {
            None
        };

        // ── Radio de splitting TreePM ────────────────────────────────────────
        let treepm_r_split = if use_treepm_slab {
            let r_s = cfg.gravity.r_split;
            if r_s > 0.0 {
                r_s
            } else {
                2.5 * box_size / pm_nm as f64
            }
        } else {
            0.0
        };
        let treepm_r_cut = 5.0 * treepm_r_split;

        // ── SfcDecomposition para SR (Fase 23) ───────────────────────────────
        let sr_sfc_kind = cfg.performance.sfc_kind;
        let sr_sfc_rebalance = cfg.performance.sfc_rebalance_interval;
        let sr_sfc_decomp_opt: Option<SfcDecomposition> =
            if use_treepm_sr_sfc && rt.size() > 1 {
                use gadget_ng_parallel::sfc::global_bbox;
                let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, local);
                let pos_loc: Vec<gadget_ng_core::Vec3> =
                    local.iter().map(|p| p.position).collect();
                Some(SfcDecomposition::build_with_bbox_and_kind(
                    &pos_loc,
                    gxlo, gxhi, gylo, gyhi, gzlo, gzhi,
                    rt.size(),
                    sr_sfc_kind,
                ))
            } else {
                None
            };

        Ok(Self {
            pm_nm,
            use_pm_dist,
            use_pm_slab,
            use_pm_pencil2d,
            use_treepm_slab,
            use_treepm_3d_halo,
            use_treepm_sr_sfc,
            use_treepm_pm_scatter_gather,
            slab_layout_opt,
            pencil_layout_opt,
            treepm_slab_layout_opt,
            treepm_r_split,
            treepm_r_cut,
            sr_sfc_decomp_opt,
            sr_sfc_rebalance,
            sr_sfc_kind,
        })
    }
}
