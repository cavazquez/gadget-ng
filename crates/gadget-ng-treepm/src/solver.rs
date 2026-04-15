//! `TreePmSolver`: hibridación Barnes-Hut + PM para fuerzas gravitacionales.
//!
//! ## Esquema de división de fuerzas
//!
//! ```text
//! F_total = F_lr  +  F_sr
//!
//! F_lr  → solver PM con filtro Gaussiano exp(-k²·r_s²/2) en k-space
//!           ↔ erf(r / (√2·r_s)) del par Newton en espacio real
//!
//! F_sr  → recorrido de octree con kernel erfc(r / (√2·r_s)),
//!           cortado en r_cut ≈ 5·r_s donde erfc < 10⁻¹¹
//! ```
//!
//! La suma `F_lr + F_sr` recupera el Newton exacto
//! (`erf(x) + erfc(x) = 1`).
//!
//! ## Parámetros clave
//!
//! | Parámetro | Descripción | Valor típico |
//! |-----------|-------------|--------------|
//! | `grid_size` | NM: celdas por lado del grid PM | 64–256 |
//! | `box_size` | longitud del cubo periódico | según simulación |
//! | `r_split` | radio de splitting Gaussiano | 2–3 × cell_size |
//! | `theta` | ángulo de apertura BH (largo alcance para PM suavizado) | 0.5 (no usado en SR) |
//!
//! ## Configuración TOML
//!
//! ```toml
//! [gravity]
//! solver    = "tree_pm"
//! pm_grid_size = 64
//! r_split   = 0.0    # 0 → auto: 2.5 × cell_size
//! ```

use gadget_ng_core::{GravitySolver, Vec3};
use gadget_ng_pm::cic;
use gadget_ng_pm::fft_poisson;

use crate::short_range::{self, ShortRangeParams};

/// Solver TreePM: largo alcance via PM filtrado + corto alcance via octree con kernel erfc.
#[derive(Debug, Clone)]
pub struct TreePmSolver {
    /// Número de celdas por lado del grid PM (NM). Potencia de 2 recomendada.
    pub grid_size: usize,
    /// Longitud del cubo periódico.
    pub box_size: f64,
    /// Radio de splitting Gaussiano. Si es ≤ 0 se elige automáticamente como
    /// `2.5 × (box_size / grid_size)` (2.5 celdas PM).
    pub r_split: f64,
}

impl TreePmSolver {
    /// Devuelve el radio de splitting efectivo (resuelve el valor automático si r_split ≤ 0).
    fn effective_r_split(&self) -> f64 {
        if self.r_split > 0.0 {
            self.r_split
        } else {
            2.5 * self.box_size / self.grid_size as f64
        }
    }
}

impl GravitySolver for TreePmSolver {
    fn accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
    ) {
        assert_eq!(global_indices.len(), out.len());

        let nm = self.grid_size;
        let r_s = self.effective_r_split();
        // Cutoff de corto alcance: erfc(r_cut / (√2·r_s)) ≈ erfc(5/√2) ≈ 1e-8 → fuerza despreciable.
        let r_cut = 5.0 * r_s;

        // ── 1. Largo alcance: PM con filtro Gaussiano ──────────────────────────
        let density = cic::assign(global_positions, global_masses, self.box_size, nm);
        let [fx_lr, fy_lr, fz_lr] =
            fft_poisson::solve_forces_filtered(&density, g, nm, self.box_size, r_s);
        let active_pos: Vec<Vec3> = global_indices
            .iter()
            .map(|&i| global_positions[i])
            .collect();
        let acc_lr = cic::interpolate(&fx_lr, &fy_lr, &fz_lr, &active_pos, self.box_size, nm);

        // ── 2. Corto alcance: octree con kernel erfc ───────────────────────────
        let sr_params = ShortRangeParams {
            positions: global_positions,
            masses: global_masses,
            eps2,
            g,
            r_split: r_s,
            r_cut2: r_cut * r_cut,
        };
        let mut acc_sr = vec![Vec3::zero(); global_indices.len()];
        short_range::short_range_accels(&sr_params, global_indices, &mut acc_sr);

        // ── 3. Suma: F_total = F_lr + F_sr ────────────────────────────────────
        for (k, a) in out.iter_mut().enumerate() {
            *a = acc_lr[k] + acc_sr[k];
        }
    }
}

unsafe impl Send for TreePmSolver {}
unsafe impl Sync for TreePmSolver {}
