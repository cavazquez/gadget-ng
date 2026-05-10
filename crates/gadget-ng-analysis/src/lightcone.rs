use gadget_ng_core::Vec3;
use rustfft::{FftPlanner, num_complex::Complex};

#[derive(Debug, Clone, Copy)]
pub struct LightconeConfig {
    pub observer: Vec3,
    pub r_min: f64,
    pub r_max: f64,
    pub pencil_beam_axis: Option<Vec3>,
    pub pencil_beam_cos_half_angle: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct LightconeHit {
    pub particle_index: usize,
    pub position: Vec3,
    pub distance: f64,
}

#[inline]
fn normalize(v: Vec3) -> Vec3 {
    let n = v.norm().max(1e-300);
    v / n
}

pub fn detect_lightcone_crossings(
    prev_positions: &[Vec3],
    curr_positions: &[Vec3],
    cfg: LightconeConfig,
) -> Vec<LightconeHit> {
    let mut out = Vec::new();
    let axis = cfg.pencil_beam_axis.map(normalize);
    for (i, (&x0, &x1)) in prev_positions.iter().zip(curr_positions.iter()).enumerate() {
        let r0v = x0 - cfg.observer;
        let r1v = x1 - cfg.observer;
        let r0 = r0v.norm();
        let r1 = r1v.norm();
        let crosses = (r0 < cfg.r_min && r1 >= cfg.r_min)
            || (r0 <= cfg.r_max && r1 > cfg.r_max)
            || (r0 > cfg.r_min && r0 < cfg.r_max);
        if !crosses {
            continue;
        }
        if let Some(a) = axis {
            let mu = (normalize(r1v)).dot(a);
            if mu < cfg.pencil_beam_cos_half_angle {
                continue;
            }
        }
        out.push(LightconeHit {
            particle_index: i,
            position: x1,
            distance: r1,
        });
    }
    out
}

#[derive(Debug, Clone)]
pub struct LensingMap {
    pub nside: usize,
    pub kappa: Vec<f64>,
    pub gamma1: Vec<f64>,
    pub gamma2: Vec<f64>,
}

impl LensingMap {
    pub fn new(nside: usize) -> Self {
        let n = nside * nside;
        Self {
            nside,
            kappa: vec![0.0; n],
            gamma1: vec![0.0; n],
            gamma2: vec![0.0; n],
        }
    }
}

/// Pipeline Born simple: acumula masas de hits en una malla angular cartesiana.
pub fn accumulate_born_lensing(
    hits: &[LightconeHit],
    masses: &[f64],
    observer: Vec3,
    nside: usize,
) -> LensingMap {
    let mut map = LensingMap::new(nside);
    for h in hits {
        if h.particle_index >= masses.len() {
            continue;
        }
        let r = h.position - observer;
        let rn = normalize(r);
        let u = ((rn.x + 1.0) * 0.5 * nside as f64).clamp(0.0, (nside - 1) as f64) as usize;
        let v = ((rn.y + 1.0) * 0.5 * nside as f64).clamp(0.0, (nside - 1) as f64) as usize;
        let pix = v * nside + u;
        let w = masses[h.particle_index] / h.distance.max(1e-6);
        map.kappa[pix] += w;
        map.gamma1[pix] += w * (rn.x * rn.x - rn.y * rn.y);
        map.gamma2[pix] += w * (2.0 * rn.x * rn.y);
    }
    map
}

/// Parámetros para la reconstrucción Kaiser-Squires.
#[derive(Debug, Clone)]
pub struct KsParams {
    /// Tamaño de la malla (n_pixels × n_pixels).
    pub n_pixels: usize,
    /// Campo de visión en radianes (lado).
    pub fov_rad: f64,
}

impl Default for KsParams {
    fn default() -> Self {
        Self {
            n_pixels: 256,
            fov_rad: 0.1,
        }
    }
}

/// Resultado de la reconstrucción Kaiser-Squires: κ reconstruido a partir de γ.
#[derive(Debug, Clone)]
pub struct KsResult {
    /// Convergencia reconstruida (n_pixels²).
    pub kappa: Vec<f64>,
    /// Número de píxeles por lado.
    pub n_pixels: usize,
    /// Campo de visión en radianes.
    pub fov_rad: f64,
}

/// Bin del espectro angular de potencia C_ℓ.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClBin {
    pub ell: f64,
    pub cl: f64,
}

/// Parámetros para tomografía de lente gravitacional débil.
#[derive(Debug, Clone)]
pub struct TomographyParams {
    /// Límites de redshift para los bins tomográficos (n_bins + 1 bordes).
    pub z_edges: Vec<f64>,
    /// Número de píxeles por lado del mapa angular.
    pub n_pixels: usize,
}

impl Default for TomographyParams {
    fn default() -> Self {
        Self {
            z_edges: vec![0.0, 0.5, 1.0, 1.5, 2.0],
            n_pixels: 256,
        }
    }
}

/// Mapa tomográfico de convergencia por bin de redshift.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TomographicLensingMap {
    /// Número de píxeles por lado.
    pub n_pixels: usize,
    /// Convergencia por bin tomográfico (índice: bin × n_pixels² + pixel).
    pub kappa_tomo: Vec<Vec<f64>>,
    /// Shear γ₁ por bin tomográfico.
    pub gamma1_tomo: Vec<Vec<f64>>,
    /// Shear γ₂ por bin tomográfico.
    pub gamma2_tomo: Vec<Vec<f64>>,
    /// Bordes de redshift usados.
    pub z_edges: Vec<f64>,
}

/// Reconstrucción Kaiser-Squires inversa: de shear (γ₁, γ₂) a convergencia (κ).
///
/// La relación en espacio de Fourier plano es:
///
/// ```text
/// κ̂(ℓ) = -(ℓ₁² - ℓ₂² + 2iℓ₁ℓ₂) / (ℓ₁² + ℓ₂²) × γ̂(ℓ)
/// ```
///
/// donde `γ̂ = γ̂₁ + iγ̂₂`. Para ℓ = 0 se asigna κ = 0 (modo nulo).
pub fn kaiser_squires_reconstruct(map: &LensingMap, fov_rad: f64) -> KsResult {
    let n = map.nside;
    let nn = n * n;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);
    let ifft = planner.plan_fft_forward(n);

    // Transformar cada fila de γ₁ y γ₂ con FFT 1D (columnas primero)
    let mut gamma1_freq_2d: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); nn];
    let mut gamma2_freq_2d: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); nn];

    // FFT 2D por filas + columnas
    for row in 0..n {
        let mut row1: Vec<Complex<f64>> = (0..n)
            .map(|col| Complex::new(map.gamma1[row * n + col], 0.0))
            .collect();
        let mut row2: Vec<Complex<f64>> = (0..n)
            .map(|col| Complex::new(map.gamma2[row * n + col], 0.0))
            .collect();
        fft.process(&mut row1);
        fft.process(&mut row2);
        for col in 0..n {
            gamma1_freq_2d[row * n + col] = row1[col];
            gamma2_freq_2d[row * n + col] = row2[col];
        }
    }
    // Columnas
    for col in 0..n {
        let mut col1: Vec<Complex<f64>> = (0..n).map(|row| gamma1_freq_2d[row * n + col]).collect();
        let mut col2: Vec<Complex<f64>> = (0..n).map(|row| gamma2_freq_2d[row * n + col]).collect();
        fft.process(&mut col1);
        fft.process(&mut col2);
        for row in 0..n {
            gamma1_freq_2d[row * n + col] = col1[row];
            gamma2_freq_2d[row * n + col] = col2[row];
        }
    }

    // Aplicar P_κ(ℓ) = -(ℓ₁² - ℓ₂² + 2iℓ₁ℓ₂)/(ℓ₁² + ℓ₂²) × γ̂
    let dl = 2.0 * std::f64::consts::PI / fov_rad;
    let mut kappa_freq: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); nn];
    for row in 0..n {
        let l2_row = if row <= n / 2 {
            row as f64
        } else {
            (row as f64) - n as f64
        };
        for col in 0..n {
            let l1_col = if col <= n / 2 {
                col as f64
            } else {
                (col as f64) - n as f64
            };
            let ell1 = l1_col * dl;
            let ell2 = l2_row * dl;
            let ell_sq = ell1 * ell1 + ell2 * ell2;
            if ell_sq < 1e-30 {
                continue;
            }
            let gamma_hat = gamma1_freq_2d[row * n + col]
                + Complex::new(0.0, 1.0) * gamma2_freq_2d[row * n + col];
            let kernel =
                (-(ell1 * ell1 - ell2 * ell2) + Complex::new(0.0, 2.0) * ell1 * ell2) / ell_sq;
            kappa_freq[row * n + col] = kernel * gamma_hat;
        }
    }

    // IFFT 2D inversa
    for row in 0..n {
        let mut row_k: Vec<Complex<f64>> = (0..n).map(|col| kappa_freq[row * n + col]).collect();
        ifft.process(&mut row_k);
        for col in 0..n {
            kappa_freq[row * n + col] = row_k[col];
        }
    }
    for col in 0..n {
        let mut col_k: Vec<Complex<f64>> = (0..n).map(|row| kappa_freq[row * n + col]).collect();
        ifft.process(&mut col_k);
        for row in 0..n {
            kappa_freq[row * n + col] = col_k[row];
        }
    }

    let norm = 1.0 / (n * n) as f64;
    let kappa: Vec<f64> = kappa_freq.iter().map(|c| c.re * norm).collect();

    KsResult {
        kappa,
        n_pixels: n,
        fov_rad,
    }
}

/// Calcula el espectro angular de potencia C_ℓ del campo de convergencia κ.
///
/// Usa FFT 2D sobre la malla angular y promedia en anillos de ℓ.
pub fn convergence_angular_cl(map: &LensingMap, fov_rad: f64, n_ell_bins: usize) -> Vec<ClBin> {
    let n = map.nside;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);

    // FFT 2D de κ
    let mut kappa_freq: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n * n];
    for row in 0..n {
        let mut row_k: Vec<Complex<f64>> = (0..n)
            .map(|col| Complex::new(map.kappa[row * n + col], 0.0))
            .collect();
        fft.process(&mut row_k);
        for col in 0..n {
            kappa_freq[row * n + col] = row_k[col];
        }
    }
    for col in 0..n {
        let mut col_k: Vec<Complex<f64>> = (0..n).map(|row| kappa_freq[row * n + col]).collect();
        fft.process(&mut col_k);
        for row in 0..n {
            kappa_freq[row * n + col] = col_k[row];
        }
    }

    let pixel_area = (fov_rad / n as f64).powi(2);
    let dl = 2.0 * std::f64::consts::PI / fov_rad;
    let omega_pixel = fov_rad * fov_rad / (n * n) as f64;

    // Calcular ℓ para cada (row, col)
    let ell_max = (n as f64 / 2.0) * dl;
    let d_ell = ell_max / n_ell_bins as f64;
    let mut cl_sum = vec![0.0f64; n_ell_bins];
    let mut cl_count = vec![0usize; n_ell_bins];

    for row in 0..n {
        let l2 = if row <= n / 2 {
            row as f64
        } else {
            (row as f64) - n as f64
        };
        for col in 0..n {
            let l1 = if col <= n / 2 {
                col as f64
            } else {
                (col as f64) - n as f64
            };
            let ell = (l1 * l1 + l2 * l2).sqrt() * dl;
            let power = kappa_freq[row * n + col].norm_sqr() * omega_pixel * omega_pixel;
            let bin = ((ell / d_ell) as usize).min(n_ell_bins - 1);
            cl_sum[bin] += power;
            cl_count[bin] += 1;
        }
    }

    let norm = pixel_area / (n * n) as f64;
    cl_sum
        .iter()
        .zip(cl_count.iter())
        .enumerate()
        .filter(|(_, (_, c))| **c > 0)
        .map(|(i, (&s, &c))| ClBin {
            ell: (i as f64 + 0.5) * d_ell,
            cl: s / c as f64 * norm,
        })
        .collect()
}

/// Acumulación tomográfica del lente débil: separa los hits por bins de redshift.
///
/// Cada partícula se asigna al bin tomográfico correspondiente según `z = 1/a - 1`
/// (calculado a partir de `distance` y `box_size/c` asumidos), y se proyecta
/// en un mapa angular de convergencia independiente por bin.
pub fn accumulate_tomographic_lensing(
    hits: &[LightconeHit],
    masses: &[f64],
    redshifts: &[f64],
    observer: Vec3,
    params: &TomographyParams,
) -> TomographicLensingMap {
    let n_bins = params.z_edges.len() - 1;
    let n = params.n_pixels;
    let nn = n * n;
    let mut kappa_tomo = vec![vec![0.0f64; nn]; n_bins];
    let mut gamma1_tomo = vec![vec![0.0f64; nn]; n_bins];
    let mut gamma2_tomo = vec![vec![0.0f64; nn]; n_bins];

    for h in hits {
        if h.particle_index >= masses.len() || h.particle_index >= redshifts.len() {
            continue;
        }
        let z = redshifts[h.particle_index];
        let bin_idx = match params
            .z_edges
            .windows(2)
            .position(|w| z >= w[0] && z < w[1])
        {
            Some(i) => i,
            None => continue,
        };
        let r = h.position - observer;
        let rn = normalize(r);
        let u = ((rn.x + 1.0) * 0.5 * n as f64).clamp(0.0, (n - 1) as f64) as usize;
        let v = ((rn.y + 1.0) * 0.5 * n as f64).clamp(0.0, (n - 1) as f64) as usize;
        let pix = v * n + u;
        let w = masses[h.particle_index] / h.distance.max(1e-6);
        kappa_tomo[bin_idx][pix] += w;
        gamma1_tomo[bin_idx][pix] += w * (rn.x * rn.x - rn.y * rn.y);
        gamma2_tomo[bin_idx][pix] += w * (2.0 * rn.x * rn.y);
    }

    TomographicLensingMap {
        n_pixels: n,
        kappa_tomo,
        gamma1_tomo,
        gamma2_tomo,
        z_edges: params.z_edges.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_shell_crossing() {
        let prev = vec![Vec3::new(0.1, 0.0, 0.0)];
        let curr = vec![Vec3::new(0.6, 0.0, 0.0)];
        let cfg = LightconeConfig {
            observer: Vec3::zero(),
            r_min: 0.2,
            r_max: 1.0,
            pencil_beam_axis: None,
            pencil_beam_cos_half_angle: -1.0,
        };
        let hits = detect_lightcone_crossings(&prev, &curr, cfg);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn ks_reconstruct_identity_zero_shear() {
        let map = LensingMap::new(16);
        let result = kaiser_squires_reconstruct(&map, 0.1);
        assert!(
            result.kappa.iter().all(|&k| k.abs() < 1e-10),
            "Zero shear should give zero convergence"
        );
    }

    #[test]
    fn ks_reconstruct_preserves_dimensions() {
        let n = 32;
        let mut map = LensingMap::new(n);
        map.gamma1[n / 2 * n + n / 2] = 1.0;
        let result = kaiser_squires_reconstruct(&map, 0.05);
        assert_eq!(result.kappa.len(), n * n);
        assert_eq!(result.n_pixels, n);
    }

    #[test]
    fn convergence_cl_returns_bins() {
        let mut map = LensingMap::new(16);
        map.kappa[8 * 16 + 8] = 1.0;
        let cl = convergence_angular_cl(&map, 0.1, 4);
        assert!(!cl.is_empty(), "C_ell should have bins");
        assert!(
            cl.iter().all(|b| b.ell > 0.0),
            "All ell values should be positive"
        );
    }

    #[test]
    fn tomographic_lensing_assigns_bins() {
        let observer = Vec3::zero();
        let hits = vec![
            LightconeHit {
                particle_index: 0,
                position: Vec3::new(0.5, 0.0, 0.5),
                distance: (0.25_f64 + 0.25_f64).sqrt(),
            },
            LightconeHit {
                particle_index: 1,
                position: Vec3::new(0.0, 0.5, 0.5),
                distance: (0.25_f64 + 0.25_f64).sqrt(),
            },
            LightconeHit {
                particle_index: 2,
                position: Vec3::new(0.3, 0.3, 0.3),
                distance: (0.09_f64 * 3.0).sqrt(),
            },
        ];
        let masses = vec![1.0, 2.0, 1.5];
        let redshifts = vec![0.1, 0.7, 1.3];
        let params = TomographyParams {
            z_edges: vec![0.0, 0.5, 1.0, 2.0],
            n_pixels: 16,
        };
        let tomo = accumulate_tomographic_lensing(&hits, &masses, &redshifts, observer, &params);
        assert_eq!(tomo.kappa_tomo.len(), 3, "Should have 3 redshift bins");
        assert_eq!(tomo.z_edges.len(), 4, "Should have 4 edges");
    }
}
