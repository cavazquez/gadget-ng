//! Resolución de la ecuación de Poisson en k-space y cálculo de fuerzas en el grid.
//!
//! ## Algoritmo
//!
//! Dado el campo de densidad `ρ[x]` en el grid NM³ y la constante gravitacional G:
//!
//! 1. FFT 3D de densidad: `ρ̂(k) = FFT3D{ρ}`
//! 2. Potencial gravitacional: `Φ̂(k) = -4πG · ρ̂(k) / k²`  (k=0 → 0)
//! 3. Fuerzas en k-space: `F̂_α(k) = -i · k_α · Φ̂(k)`
//! 4. IFFT por componente: `F_α = IFFT3D{F̂_α}`
//!
//! ## Suavizado Plummer coherente con TreePM (opcional)
//!
//! Si se pasa `plummer_eps > 0`, se multiplica `Φ̂(k)` por `exp(−k² ε²)` en el mismo espacio k
//! que el PM (aproximación Gaussiana al kernel de Plummer). Así el largo alcance PM puede
//! alinearse con el ε del par corto Newtoniano cuando `physical_softening` está activo (valor
//! referenciado al factor de escala de construcción del solver).
//!
//! ## Convención de k
//!
//! Para un grid de lado NM y celda de tamaño `Δx = box_size/NM`, los números
//! de onda son `k_α = 2π·n_α / box_size` con `n_α ∈ {0,1,...,NM/2,-NM/2+1,...,-1}`.
//! En rustfft (convención DFT estándar), el índice `j` corresponde a
//! `n = j` para `j ≤ NM/2` y `n = j - NM` para `j > NM/2`.

#[cfg(feature = "fftw")]
use crate::fft_backend::FftwBackend;
use crate::fft_backend::{FftBackendKind, PmFftBackend, RustFftBackend};
use rustfft::{FftPlanner, num_complex::Complex};

#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type PmFftPlanPair = (Arc<dyn rustfft::Fft<f64>>, Arc<dyn rustfft::Fft<f64>>);
type PmFftPlanCache = Mutex<HashMap<usize, PmFftPlanPair>>;

/// Parámetros del solver PM f(R) con screening espacial.
#[derive(Debug, Clone)]
pub struct FrMeshParams<'a> {
    /// Parámetros Hu-Sawicki.
    pub fr: &'a gadget_ng_core::FRParams,
    /// Iteraciones Jacobi del screening.
    pub iterations: usize,
    /// Mezcla de suavizado por iteración.
    pub smoothing: f64,
    /// Suavizado Plummer opcional en k-space.
    pub plummer_eps: Option<f64>,
    /// Campo de screening pre-computado externamente (e.g. desde GPU via AP-20).
    /// Si `Some`, se usa directamente en lugar de calcular en CPU.
    pub screening_override: Option<Vec<f64>>,
}

/// Resuelve la ecuación de Poisson y devuelve las tres componentes de la fuerza
/// en el grid como arrays planos de longitud `nm³`.
///
/// - `density` — densidad `ρ` en el grid (masa/celda, longitud `nm³`).
/// - `g` — constante gravitacional (signo positivo; las fuerzas son atractivas).
/// - `nm` — número de celdas por lado (potencia de 2 recomendada).
/// - `box_size` — longitud del cubo periódico.
///
/// La densidad `ρ` tiene unidades de masa/celda. Para que el resultado tenga
/// unidades de aceleración (longitud/tiempo²), las masas deben estar en las
/// mismas unidades que G·m/r².
pub fn solve_forces(density: &[f64], g: f64, nm: usize, box_size: f64) -> [Vec<f64>; 3] {
    solve_forces_with_backend(
        density,
        g,
        nm,
        box_size,
        None,
        None,
        FftBackendKind::RustFft,
    )
}

/// Igual que [`solve_forces`] pero aplica un filtro Gaussiano en k-space que
/// suprime las contribuciones de **corto alcance**, dejando solo las de largo alcance.
///
/// El filtro es `W(k) = exp(-k²·r_split²/2)`, equivalente a convolucionar la
/// densidad con una Gaussiana de anchura `r_split` en espacio real. El potencial
/// resultante corresponde a `erf(r / (√2·r_split))` en el par-Newton real-space.
///
/// La complementaria `erfc(r / (√2·r_split))` se calcula en el paso de corto
/// alcance del árbol para que la suma sea igual al Newton exacto.
pub fn solve_forces_filtered(
    density: &[f64],
    g: f64,
    nm: usize,
    box_size: f64,
    r_split: f64,
) -> [Vec<f64>; 3] {
    solve_forces_with_backend(
        density,
        g,
        nm,
        box_size,
        Some(r_split),
        None,
        FftBackendKind::RustFft,
    )
}

/// Como [`solve_forces`] pero aplica suavizado Plummer en k-space `∝ exp(−k² ε²)` si `plummer_eps = Some(ε)`.
pub fn solve_forces_softened(
    density: &[f64],
    g: f64,
    nm: usize,
    box_size: f64,
    plummer_eps: Option<f64>,
) -> [Vec<f64>; 3] {
    solve_forces_with_backend(
        density,
        g,
        nm,
        box_size,
        None,
        plummer_eps,
        FftBackendKind::RustFft,
    )
}

/// Resuelve PM con una quinta fuerza f(R) lineal no-screened en espacio-k.
///
/// Este camino cubre el régimen "MG solo PM": la modificación escalar se aplica
/// como un refuerzo homogéneo `G_eff = G * (1 + 1/3)`. El screening chameleon
/// local sigue viviendo en `gadget-ng-core::apply_modified_gravity`; esta función
/// es la aproximación PM para modos cosmológicos de baja densidad.
pub fn solve_forces_modified_gravity(
    density: &[f64],
    g: f64,
    nm: usize,
    box_size: f64,
    params: &gadget_ng_core::FRParams,
    plummer_eps: Option<f64>,
) -> [Vec<f64>; 3] {
    solve_forces_with_backend(
        density,
        g * pm_fifth_force_boost(params),
        nm,
        box_size,
        None,
        plummer_eps,
        FftBackendKind::RustFft,
    )
}

/// Resuelve PM f(R) con screening chameleon espacial reducido.
///
/// La fuerza total se aproxima como:
///
/// `F = F_GR[ρ] + F_scalar[ρ × S(ρ)] / 3`
///
/// donde `S` es el factor local de quinta fuerza calculado desde el campo
/// chameleon y suavizado por unas pocas iteraciones Jacobi en la malla. En baja
/// densidad `S≈1` recupera el límite no-screened `4/3`; en celdas densas `S<<1`.
pub fn solve_forces_fr_screened_mesh(
    density: &[f64],
    g: f64,
    nm: usize,
    box_size: f64,
    params: FrMeshParams<'_>,
) -> [Vec<f64>; 3] {
    let gr = solve_forces_with_backend(
        density,
        g,
        nm,
        box_size,
        None,
        params.plummer_eps,
        FftBackendKind::RustFft,
    );
    if params.fr.f_r0.abs() <= 0.0 {
        return gr;
    }

    let screening = params.screening_override.unwrap_or_else(|| {
        fr_screening_field(density, nm, params.fr, params.iterations, params.smoothing)
    });
    let scalar_density: Vec<f64> = density
        .iter()
        .zip(screening.iter())
        .map(|(&rho, &screen)| rho * screen)
        .collect();
    let scalar = solve_forces_with_backend(
        &scalar_density,
        g / 3.0,
        nm,
        box_size,
        None,
        params.plummer_eps,
        FftBackendKind::RustFft,
    );

    [
        gr[0].iter().zip(&scalar[0]).map(|(a, b)| a + b).collect(),
        gr[1].iter().zip(&scalar[1]).map(|(a, b)| a + b).collect(),
        gr[2].iter().zip(&scalar[2]).map(|(a, b)| a + b).collect(),
    ]
}

/// Campo de screening chameleon por celda, `S ∈ [0,1]`.
pub fn fr_screening_field(
    density: &[f64],
    nm: usize,
    params: &gadget_ng_core::FRParams,
    iterations: usize,
    smoothing: f64,
) -> Vec<f64> {
    let nm3 = nm * nm * nm;
    assert_eq!(density.len(), nm3);
    if params.f_r0.abs() <= 0.0 {
        return vec![0.0; nm3];
    }

    let rho_bar = density.iter().sum::<f64>() / nm3 as f64;
    let mut screen: Vec<f64> = density
        .iter()
        .map(|&rho| {
            let delta = (rho - rho_bar) / rho_bar.max(1e-30);
            let fr = gadget_ng_core::chameleon_field(delta, params.f_r0, params.n);
            gadget_ng_core::fifth_force_factor(fr, params.f_r0)
        })
        .collect();

    let mix = smoothing.clamp(0.0, 1.0);
    if mix <= 0.0 || iterations == 0 {
        return screen;
    }

    let idx = |ix: usize, iy: usize, iz: usize| -> usize { iz * nm * nm + iy * nm + ix };
    for _ in 0..iterations {
        let old = screen.clone();
        for iz in 0..nm {
            let izm = (iz + nm - 1) % nm;
            let izp = (iz + 1) % nm;
            for iy in 0..nm {
                let iym = (iy + nm - 1) % nm;
                let iyp = (iy + 1) % nm;
                for ix in 0..nm {
                    let ixm = (ix + nm - 1) % nm;
                    let ixp = (ix + 1) % nm;
                    let flat = idx(ix, iy, iz);
                    let neigh = old[idx(ixm, iy, iz)]
                        + old[idx(ixp, iy, iz)]
                        + old[idx(ix, iym, iz)]
                        + old[idx(ix, iyp, iz)]
                        + old[idx(ix, iy, izm)]
                        + old[idx(ix, iy, izp)];
                    screen[flat] = ((1.0 - mix) * old[flat] + mix * neigh / 6.0).clamp(0.0, 1.0);
                }
            }
        }
    }
    screen
}

/// Factor multiplicativo homogéneo para la fuerza PM en Hu-Sawicki f(R).
///
/// `f_R0 = 0` recupera GR exactamente. Para un campo no-screened el máximo es
/// `4/3`, el límite escalar estándar de f(R).
#[inline]
pub fn pm_fifth_force_boost(params: &gadget_ng_core::FRParams) -> f64 {
    if params.f_r0.abs() <= 0.0 {
        1.0
    } else {
        4.0 / 3.0
    }
}

pub fn solve_forces_with_backend(
    density: &[f64],
    g: f64,
    nm: usize,
    box_size: f64,
    r_split: Option<f64>,
    plummer_eps: Option<f64>,
    backend_kind: FftBackendKind,
) -> [Vec<f64>; 3] {
    match backend_kind {
        FftBackendKind::RustFft => {
            RustFftBackend.solve_forces(density, g, nm, box_size, r_split, plummer_eps)
        }
        #[cfg(feature = "fftw")]
        FftBackendKind::Fftw => {
            FftwBackend.solve_forces(density, g, nm, box_size, r_split, plummer_eps)
        }
    }
}

fn fft_pm_plans(nm: usize) -> PmFftPlanPair {
    static CACHE: OnceLock<PmFftPlanCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = cache.lock().expect("pm fft cache");
    g.entry(nm)
        .or_insert_with(|| {
            let mut planner = FftPlanner::new();
            let fwd = planner.plan_fft_forward(nm);
            let inv = planner.plan_fft_inverse(nm);
            (fwd, inv)
        })
        .clone()
}

pub(crate) fn solve_forces_impl(
    density: &[f64],
    g: f64,
    nm: usize,
    box_size: f64,
    r_split: Option<f64>,
    plummer_eps: Option<f64>,
) -> [Vec<f64>; 3] {
    let nm2 = nm * nm;
    let nm3 = nm2 * nm;
    assert_eq!(density.len(), nm3);

    // Volumen total y volumen de celda (para normalizar la densidad volumétrica).
    let cell_vol = (box_size / nm as f64).powi(3);
    let rho_scale = 1.0 / cell_vol;

    // ── FFT 3D de la densidad ─────────────────────────────────────────────────
    let (fft_fwd, fft_inv) = fft_pm_plans(nm);

    // Convertir densidad a complejo.
    let mut rho_c: Vec<Complex<f64>> = density
        .iter()
        .map(|&r| Complex::new(r * rho_scale, 0.0))
        .collect();

    // FFT 3D = tres pasadas de 1D FFTs (filas, columnas, pilas).
    fft3d_inplace(&mut rho_c, nm, &fft_fwd);

    // ── Resolver Poisson y construir F̂_x, F̂_y, F̂_z en k-space ────────────
    let dk = 2.0 * std::f64::consts::PI / box_size;
    let four_pi_g = 4.0 * std::f64::consts::PI * g;

    // Convertir a SoA (Structure of Arrays) para vectorización
    let rho_re: Vec<f64> = rho_c.iter().map(|c| c.re).collect();
    let rho_im: Vec<f64> = rho_c.iter().map(|c| c.im).collect();

    // Pre-calcular wave numbers para cada eje
    let kx_arr: Vec<f64> = (0..nm).map(|ix| dk * freq_index(ix, nm) as f64).collect();
    let ky_arr: Vec<f64> = (0..nm).map(|iy| dk * freq_index(iy, nm) as f64).collect();
    let kz_arr: Vec<f64> = (0..nm).map(|iz| dk * freq_index(iz, nm) as f64).collect();

    let mut fx_re = vec![0.0_f64; nm3];
    let mut fx_im = vec![0.0_f64; nm3];
    let mut fy_re = vec![0.0_f64; nm3];
    let mut fy_im = vec![0.0_f64; nm3];
    let mut fz_re = vec![0.0_f64; nm3];
    let mut fz_im = vec![0.0_f64; nm3];

    // Dispatch del kernel espectral: Rayon cubre el solve k-space completo;
    // sin Rayon se mantiene el dispatch SIMD explícito de un hilo.
    #[cfg(feature = "rayon")]
    spectral_kernel_rayon(
        &rho_re,
        &rho_im,
        &kx_arr,
        &ky_arr,
        &kz_arr,
        four_pi_g,
        r_split,
        plummer_eps,
        nm,
        &mut fx_re,
        &mut fx_im,
        &mut fy_re,
        &mut fy_im,
        &mut fz_re,
        &mut fz_im,
    );

    #[cfg(not(feature = "rayon"))]
    {
        // Dispatch SIMD del kernel espectral
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    spectral_kernel_avx512(
                        &rho_re,
                        &rho_im,
                        &kx_arr,
                        &ky_arr,
                        &kz_arr,
                        four_pi_g,
                        r_split,
                        plummer_eps,
                        nm,
                        &mut fx_re,
                        &mut fx_im,
                        &mut fy_re,
                        &mut fy_im,
                        &mut fz_re,
                        &mut fz_im,
                    );
                }
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    spectral_kernel_avx2(
                        &rho_re,
                        &rho_im,
                        &kx_arr,
                        &ky_arr,
                        &kz_arr,
                        four_pi_g,
                        r_split,
                        plummer_eps,
                        nm,
                        &mut fx_re,
                        &mut fx_im,
                        &mut fy_re,
                        &mut fy_im,
                        &mut fz_re,
                        &mut fz_im,
                    );
                }
            } else {
                spectral_kernel_scalar(
                    &rho_re,
                    &rho_im,
                    &kx_arr,
                    &ky_arr,
                    &kz_arr,
                    four_pi_g,
                    r_split,
                    plummer_eps,
                    nm,
                    &mut fx_re,
                    &mut fx_im,
                    &mut fy_re,
                    &mut fy_im,
                    &mut fz_re,
                    &mut fz_im,
                );
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        spectral_kernel_scalar(
            &rho_re,
            &rho_im,
            &kx_arr,
            &ky_arr,
            &kz_arr,
            four_pi_g,
            r_split,
            plummer_eps,
            nm,
            &mut fx_re,
            &mut fx_im,
            &mut fy_re,
            &mut fy_im,
            &mut fz_re,
            &mut fz_im,
        );
    }

    // Convertir SoA de vuelta a AoS Complex<f64> para IFFT
    let mut fx_c: Vec<Complex<f64>> = (0..nm3).map(|i| Complex::new(fx_re[i], fx_im[i])).collect();
    let mut fy_c: Vec<Complex<f64>> = (0..nm3).map(|i| Complex::new(fy_re[i], fy_im[i])).collect();
    let mut fz_c: Vec<Complex<f64>> = (0..nm3).map(|i| Complex::new(fz_re[i], fz_im[i])).collect();

    // ── IFFT 3D de cada componente de fuerza ─────────────────────────────────
    let norm = 1.0 / nm3 as f64;
    ifft3d_inplace(&mut fx_c, nm, &fft_inv);
    ifft3d_inplace(&mut fy_c, nm, &fft_inv);
    ifft3d_inplace(&mut fz_c, nm, &fft_inv);

    let fx: Vec<f64> = fx_c.iter().map(|c| c.re * norm).collect();
    let fy: Vec<f64> = fy_c.iter().map(|c| c.re * norm).collect();
    let fz: Vec<f64> = fz_c.iter().map(|c| c.re * norm).collect();

    [fx, fy, fz]
}

// ── Kernels espectrales Poisson (SoA layout para vectorización) ──────────────

/// Kernel escalar: calcula Φ̂(k) y F̂_α(k) para cada punto del grid k-space.
///
/// Usa layout SoA (arrays separados de re/im) para permitir auto-vectorización.
#[expect(
    clippy::too_many_arguments,
    reason = "spectral kernel keeps SoA slices explicit"
)]
#[cfg(not(feature = "rayon"))]
fn spectral_kernel_scalar(
    rho_re: &[f64],
    rho_im: &[f64],
    kx_arr: &[f64],
    ky_arr: &[f64],
    kz_arr: &[f64],
    four_pi_g: f64,
    r_split: Option<f64>,
    plummer_eps: Option<f64>,
    nm: usize,
    fx_re: &mut [f64],
    fx_im: &mut [f64],
    fy_re: &mut [f64],
    fy_im: &mut [f64],
    fz_re: &mut [f64],
    fz_im: &mut [f64],
) {
    let nm2 = nm * nm;
    let r_split2 = r_split.map(|r| r * r);
    let eps2 = plummer_eps.map(|e| e * e);

    for flat in 0..rho_re.len() {
        let iz = flat / nm2;
        let iy = (flat / nm) % nm;
        let ix = flat % nm;
        let kx = kx_arr[ix];
        let ky = ky_arr[iy];
        let kz = kz_arr[iz];
        let k2 = kx * kx + ky * ky + kz * kz;

        if k2 < 1e-30 {
            fx_re[flat] = 0.0;
            fx_im[flat] = 0.0;
            fy_re[flat] = 0.0;
            fy_im[flat] = 0.0;
            fz_re[flat] = 0.0;
            fz_im[flat] = 0.0;
            continue;
        }

        let mut filter = 1.0_f64;
        if let Some(r2) = r_split2 {
            filter *= (-0.5 * k2 * r2).exp();
        }
        if let Some(e2) = eps2
            && e2 > 0.0
        {
            filter *= (-k2 * e2).exp();
        }

        // phi_k = rho_c[flat] * (-4πG * filter / k2)
        let phi_re = rho_re[flat] * (-four_pi_g * filter / k2);
        let phi_im = rho_im[flat] * (-four_pi_g * filter / k2);

        // F̂_α = -i k_α Φ̂(k) = Complex(k_α * phi_im, -k_α * phi_re)
        fx_re[flat] = kx * phi_im;
        fx_im[flat] = -kx * phi_re;
        fy_re[flat] = ky * phi_im;
        fy_im[flat] = -ky * phi_re;
        fz_re[flat] = kz * phi_im;
        fz_im[flat] = -kz * phi_re;
    }
}

#[cfg(feature = "rayon")]
#[expect(
    clippy::too_many_arguments,
    reason = "spectral kernel keeps SoA slices explicit"
)]
fn spectral_kernel_rayon(
    rho_re: &[f64],
    rho_im: &[f64],
    kx_arr: &[f64],
    ky_arr: &[f64],
    kz_arr: &[f64],
    four_pi_g: f64,
    r_split: Option<f64>,
    plummer_eps: Option<f64>,
    nm: usize,
    fx_re: &mut [f64],
    fx_im: &mut [f64],
    fy_re: &mut [f64],
    fy_im: &mut [f64],
    fz_re: &mut [f64],
    fz_im: &mut [f64],
) {
    let nm2 = nm * nm;
    let r_split2 = r_split.map(|r| r * r);
    let eps2 = plummer_eps.map(|e| e * e);

    fx_re
        .par_iter_mut()
        .zip(fx_im.par_iter_mut())
        .zip(fy_re.par_iter_mut())
        .zip(fy_im.par_iter_mut())
        .zip(fz_re.par_iter_mut())
        .zip(fz_im.par_iter_mut())
        .enumerate()
        .for_each(
            |(flat, (((((fx_re, fx_im), fy_re), fy_im), fz_re), fz_im))| {
                let iz = flat / nm2;
                let iy = (flat / nm) % nm;
                let ix = flat % nm;
                let kx = kx_arr[ix];
                let ky = ky_arr[iy];
                let kz = kz_arr[iz];
                let k2 = kx * kx + ky * ky + kz * kz;

                if k2 < 1e-30 {
                    *fx_re = 0.0;
                    *fx_im = 0.0;
                    *fy_re = 0.0;
                    *fy_im = 0.0;
                    *fz_re = 0.0;
                    *fz_im = 0.0;
                    return;
                }

                let mut filter = 1.0_f64;
                if let Some(r2) = r_split2 {
                    filter *= (-0.5 * k2 * r2).exp();
                }
                if let Some(e2) = eps2
                    && e2 > 0.0
                {
                    filter *= (-k2 * e2).exp();
                }

                let phi_re = rho_re[flat] * (-four_pi_g * filter / k2);
                let phi_im = rho_im[flat] * (-four_pi_g * filter / k2);

                *fx_re = kx * phi_im;
                *fx_im = -kx * phi_re;
                *fy_re = ky * phi_im;
                *fy_im = -ky * phi_re;
                *fz_re = kz * phi_im;
                *fz_im = -kz * phi_re;
            },
        );
}

/// Kernel AVX2+FMA: fuerza al compilador a emitir instrucciones YMM (4×f64).
#[cfg(all(
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
#[expect(
    clippy::too_many_arguments,
    reason = "spectral kernel keeps SoA slices explicit"
)]
unsafe fn spectral_kernel_avx2(
    rho_re: &[f64],
    rho_im: &[f64],
    kx_arr: &[f64],
    ky_arr: &[f64],
    kz_arr: &[f64],
    four_pi_g: f64,
    r_split: Option<f64>,
    plummer_eps: Option<f64>,
    nm: usize,
    fx_re: &mut [f64],
    fx_im: &mut [f64],
    fy_re: &mut [f64],
    fy_im: &mut [f64],
    fz_re: &mut [f64],
    fz_im: &mut [f64],
) {
    spectral_kernel_scalar(
        rho_re,
        rho_im,
        kx_arr,
        ky_arr,
        kz_arr,
        four_pi_g,
        r_split,
        plummer_eps,
        nm,
        fx_re,
        fx_im,
        fy_re,
        fy_im,
        fz_re,
        fz_im,
    )
}

/// Kernel AVX-512: fuerza al compilador a emitir instrucciones ZMM (8×f64).
#[cfg(all(
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[expect(
    clippy::too_many_arguments,
    reason = "spectral kernel keeps SoA slices explicit"
)]
unsafe fn spectral_kernel_avx512(
    rho_re: &[f64],
    rho_im: &[f64],
    kx_arr: &[f64],
    ky_arr: &[f64],
    kz_arr: &[f64],
    four_pi_g: f64,
    r_split: Option<f64>,
    plummer_eps: Option<f64>,
    nm: usize,
    fx_re: &mut [f64],
    fx_im: &mut [f64],
    fy_re: &mut [f64],
    fy_im: &mut [f64],
    fz_re: &mut [f64],
    fz_im: &mut [f64],
) {
    spectral_kernel_scalar(
        rho_re,
        rho_im,
        kx_arr,
        ky_arr,
        kz_arr,
        four_pi_g,
        r_split,
        plummer_eps,
        nm,
        fx_re,
        fx_im,
        fy_re,
        fy_im,
        fz_re,
        fz_im,
    )
}

/// Convierte un índice DFT `j ∈ [0, nm)` en el número de onda entero correspondiente.
/// `j ≤ nm/2` → `j`; `j > nm/2` → `j - nm`.
#[inline]
fn freq_index(j: usize, nm: usize) -> i64 {
    if j <= nm / 2 {
        j as i64
    } else {
        j as i64 - nm as i64
    }
}

/// FFT 3D in-place usando tres pasadas de 1D FFTs (filas → columnas → pilas).
fn fft3d_inplace(
    data: &mut [Complex<f64>],
    nm: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
) {
    let nm2 = nm * nm;
    // Pasada X: FFT a lo largo de filas (eje x).
    for iz in 0..nm {
        for iy in 0..nm {
            let start = iz * nm2 + iy * nm;
            fft.process(&mut data[start..start + nm]);
        }
    }
    // Pasada Y: FFT a lo largo de columnas (eje y). Necesita copia temporal por strides.
    let mut tmp = vec![Complex::new(0.0, 0.0); nm];
    for iz in 0..nm {
        for ix in 0..nm {
            for iy in 0..nm {
                tmp[iy] = data[iz * nm2 + iy * nm + ix];
            }
            fft.process(&mut tmp);
            for iy in 0..nm {
                data[iz * nm2 + iy * nm + ix] = tmp[iy];
            }
        }
    }
    // Pasada Z: FFT a lo largo de pilas (eje z). Strides en z.
    for iy in 0..nm {
        for ix in 0..nm {
            for iz in 0..nm {
                tmp[iz] = data[iz * nm2 + iy * nm + ix];
            }
            fft.process(&mut tmp);
            for iz in 0..nm {
                data[iz * nm2 + iy * nm + ix] = tmp[iz];
            }
        }
    }
}

/// IFFT 3D in-place (misma estructura que `fft3d_inplace` pero con plan inverso).
/// La normalización 1/N³ se aplica fuera.
fn ifft3d_inplace(
    data: &mut [Complex<f64>],
    nm: usize,
    ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
) {
    let nm2 = nm * nm;
    let mut tmp = vec![Complex::new(0.0, 0.0); nm];
    // Pasada Z.
    for iy in 0..nm {
        for ix in 0..nm {
            for iz in 0..nm {
                tmp[iz] = data[iz * nm2 + iy * nm + ix];
            }
            ifft.process(&mut tmp);
            for iz in 0..nm {
                data[iz * nm2 + iy * nm + ix] = tmp[iz];
            }
        }
    }
    // Pasada Y.
    for iz in 0..nm {
        for ix in 0..nm {
            for iy in 0..nm {
                tmp[iy] = data[iz * nm2 + iy * nm + ix];
            }
            ifft.process(&mut tmp);
            for iy in 0..nm {
                data[iz * nm2 + iy * nm + ix] = tmp[iy];
            }
        }
    }
    // Pasada X.
    for iz in 0..nm {
        for iy in 0..nm {
            let start = iz * nm2 + iy * nm;
            ifft.process(&mut data[start..start + nm]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Una densidad uniforme produce fuerzas ≈ 0 (el modo DC es cero).
    #[test]
    fn uniform_density_gives_zero_force() {
        let nm = 8usize;
        let nm3 = nm * nm * nm;
        let density = vec![1.0_f64; nm3];
        let [fx, fy, fz] = solve_forces(&density, 1.0, nm, 1.0);
        for i in 0..nm3 {
            assert!(
                fx[i].abs() < 1e-10 && fy[i].abs() < 1e-10 && fz[i].abs() < 1e-10,
                "fuerzas no nulas en celda {i}: fx={} fy={} fz={}",
                fx[i],
                fy[i],
                fz[i]
            );
        }
    }

    /// La FFT 3D seguida de IFFT debe recuperar la señal original.
    #[test]
    fn fft3d_roundtrip() {
        let nm = 4usize;
        let nm3 = nm * nm * nm;
        let original: Vec<Complex<f64>> = (0..nm3).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut data = original.clone();
        let mut planner = FftPlanner::new();
        let fft_fwd = planner.plan_fft_forward(nm);
        let fft_inv = planner.plan_fft_inverse(nm);
        fft3d_inplace(&mut data, nm, &fft_fwd);
        ifft3d_inplace(&mut data, nm, &fft_inv);
        let norm = 1.0 / nm3 as f64;
        for (i, (d, o)) in data.iter().zip(original.iter()).enumerate() {
            let err = (d.re * norm - o.re).abs();
            assert!(err < 1e-10, "error en índice {i}: {err}");
        }
    }

    #[test]
    fn modified_gravity_pm_zero_fr_matches_gr() {
        let nm = 8usize;
        let nm3 = nm * nm * nm;
        let mut density = vec![0.0_f64; nm3];
        density[1] = 1.0;
        density[3 * nm * nm + 2 * nm + 4] = 0.5;
        let gr = solve_forces(&density, 1.0, nm, 1.0);
        let fr = solve_forces_modified_gravity(
            &density,
            1.0,
            nm,
            1.0,
            &gadget_ng_core::FRParams { f_r0: 0.0, n: 1.0 },
            None,
        );
        for c in 0..3 {
            for i in 0..nm3 {
                assert!((gr[c][i] - fr[c][i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn modified_gravity_pm_unscreened_scales_force_by_four_thirds() {
        let nm = 8usize;
        let nm3 = nm * nm * nm;
        let mut density = vec![0.0_f64; nm3];
        density[1] = 1.0;
        density[5 * nm * nm + 2 * nm + 3] = 0.25;
        let gr = solve_forces(&density, 1.0, nm, 1.0);
        let fr = solve_forces_modified_gravity(
            &density,
            1.0,
            nm,
            1.0,
            &gadget_ng_core::FRParams { f_r0: 1e-5, n: 1.0 },
            None,
        );
        let idx = 2;
        assert!((fr[0][idx] / gr[0][idx] - 4.0 / 3.0).abs() < 1e-10);
    }

    #[cfg(feature = "fftw")]
    #[test]
    fn fftw_backend_matches_rustfft_backend() {
        let nm = 8usize;
        let nm3 = nm * nm * nm;
        let mut density = vec![0.0_f64; nm3];
        for iz in 0..nm {
            for iy in 0..nm {
                for ix in 0..nm {
                    let x = ix as f64 / nm as f64;
                    let y = iy as f64 / nm as f64;
                    let z = iz as f64 / nm as f64;
                    density[iz * nm * nm + iy * nm + ix] = 1.0
                        + 0.1 * (2.0 * std::f64::consts::PI * x).sin()
                        + 0.07 * (2.0 * std::f64::consts::PI * y).cos()
                        + 0.05 * (4.0 * std::f64::consts::PI * z).sin();
                }
            }
        }
        let a = solve_forces_with_backend(
            &density,
            1.0,
            nm,
            1.0,
            Some(0.15),
            Some(0.01),
            FftBackendKind::RustFft,
        );
        let b = solve_forces_with_backend(
            &density,
            1.0,
            nm,
            1.0,
            Some(0.15),
            Some(0.01),
            FftBackendKind::Fftw,
        );

        for c in 0..3 {
            for i in 0..nm3 {
                let den = a[c][i].abs().max(1e-12);
                let rel = (a[c][i] - b[c][i]).abs() / den;
                assert!(
                    rel < 1e-10,
                    "backend mismatch comp={c}, i={i}, rel={rel:.3e}"
                );
            }
        }
    }
}
