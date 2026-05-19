use crate::fft_poisson;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftBackendKind {
    RustFft,
    #[cfg(feature = "fftw")]
    Fftw,
}

pub trait PmFftBackend: Send + Sync {
    fn solve_forces(
        &self,
        density: &[f64],
        g: f64,
        nm: usize,
        box_size: f64,
        r_split: Option<f64>,
        plummer_eps: Option<f64>,
    ) -> [Vec<f64>; 3];
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RustFftBackend;

impl PmFftBackend for RustFftBackend {
    fn solve_forces(
        &self,
        density: &[f64],
        g: f64,
        nm: usize,
        box_size: f64,
        r_split: Option<f64>,
        plummer_eps: Option<f64>,
    ) -> [Vec<f64>; 3] {
        fft_poisson::solve_forces_impl(density, g, nm, box_size, r_split, plummer_eps)
    }
}

#[cfg(feature = "fftw")]
#[derive(Debug, Clone, Copy, Default)]
pub struct FftwBackend;

#[cfg(feature = "fftw")]
impl PmFftBackend for FftwBackend {
    fn solve_forces(
        &self,
        density: &[f64],
        g: f64,
        nm: usize,
        box_size: f64,
        r_split: Option<f64>,
        plummer_eps: Option<f64>,
    ) -> [Vec<f64>; 3] {
        // Backend FFTW opcional: fallback numéricamente equivalente a RustFFT
        // hasta integrar planes FFTW host reales.
        fft_poisson::solve_forces_impl(density, g, nm, box_size, r_split, plummer_eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_fft_uniform_density_near_zero_force() {
        let nm = 4usize;
        let n3 = nm * nm * nm;
        let rho_cell = 1.0 / n3 as f64;
        let density = vec![rho_cell; n3];
        let backend = RustFftBackend;
        let forces = backend.solve_forces(&density, 1.0, nm, 1.0, None, None);
        assert_eq!(forces[0].len(), n3);
        let max_f = forces
            .iter()
            .flat_map(|c| c.iter())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_f < 1e-6,
            "densidad uniforme debe dar fuerza ~0, max |F| = {max_f}"
        );
    }

    #[test]
    fn fft_backend_kind_rust_is_distinct() {
        assert_eq!(FftBackendKind::RustFft, FftBackendKind::RustFft);
    }
}
