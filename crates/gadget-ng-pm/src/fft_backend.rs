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

