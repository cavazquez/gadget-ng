//! Descomposición de dominio espacial por slabs en el eje x.
//!
//! Cada rango recibe una banda `[x_lo_r, x_hi_r)` del espacio de simulación.
//! La anchura del halo (`halo_width`) determina cuántas partículas de los
//! rangos vecinos se intercambian para calcular fuerzas cortas con el árbol local.

use gadget_ng_core::Vec3;

/// Descomposición 1D por slabs uniformes en x.
///
/// Los límites se calculan una vez a partir de las posiciones globales
/// (`x_lo_global`, `x_hi_global` recibidos del motor vía `allreduce_min/max`).
#[derive(Debug, Clone, Copy)]
pub struct SlabDecomposition {
    pub n_ranks: i32,
    pub x_lo:   f64,
    pub x_hi:   f64,
}

impl SlabDecomposition {
    /// Crea una descomposición uniforme sobre el intervalo `[x_lo, x_hi]`.
    pub fn new(x_lo: f64, x_hi: f64, n_ranks: i32) -> Self {
        let margin = (x_hi - x_lo) * 0.001 + 1e-12;
        Self { n_ranks, x_lo: x_lo - margin, x_hi: x_hi + margin }
    }

    pub fn slab_width(&self) -> f64 {
        (self.x_hi - self.x_lo) / self.n_ranks as f64
    }

    /// Rango responsable de la posición `x`.
    pub fn rank_for_x(&self, x: f64) -> i32 {
        let r = ((x - self.x_lo) / self.slab_width()) as i32;
        r.clamp(0, self.n_ranks - 1)
    }

    /// Límites `[lo, hi)` del slab del rango `rank`.
    pub fn bounds(&self, rank: i32) -> (f64, f64) {
        let w = self.slab_width();
        let lo = self.x_lo + rank as f64 * w;
        let hi = lo + w;
        (lo, hi)
    }

    /// Anchura recomendada de halo para un parámetro de apertura `theta`.
    /// Conservador: `theta * slab_width`.
    pub fn halo_width(&self, theta: f64) -> f64 {
        theta * self.slab_width()
    }
}

/// Filtra `positions` para estimar los límites globales x (usado en motor serial).
pub fn x_bounds_of(positions: &[Vec3]) -> (f64, f64) {
    let lo = positions.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let hi = positions.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    #[test]
    fn rank_assignment_and_bounds() {
        let d = SlabDecomposition::new(0.0, 4.0, 4);
        assert_eq!(d.rank_for_x(0.5), 0);
        assert_eq!(d.rank_for_x(1.5), 1);
        assert_eq!(d.rank_for_x(2.5), 2);
        assert_eq!(d.rank_for_x(3.5), 3);
        let (lo, hi) = d.bounds(1);
        assert!((lo - 1.0).abs() < 0.01);
        assert!((hi - 2.0).abs() < 0.01);
    }

    #[test]
    fn x_bounds_of_vec() {
        let positions = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)];
        let (lo, hi) = x_bounds_of(&positions);
        assert!((lo - 1.0).abs() < 1e-12);
        assert!((hi - 3.0).abs() < 1e-12);
    }
}
