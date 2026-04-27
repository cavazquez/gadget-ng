//! Assembly bias: correlación entre propiedades internas de halos y entorno (Phase 76).
//!
//! El **assembly bias** es el fenómeno por el cual halos de la misma masa pero
//! formados en distintos tiempos (o con distinto spin/concentración) tienen
//! sesgos de clustering diferentes. Se mide correlacionando la propiedad interna
//! (λ, c, z_form) con el sobredensidad del entorno δ_env a escala R_smooth.
//!
//! ## Algoritmo
//!
//! 1. Construir el campo de densidad CIC en un grid de resolución `n_smooth`.
//! 2. Suavizar el campo con un filtro top-hat esférico de radio `R_smooth` en k-space.
//! 3. Para cada halo, interpolar δ_env en su posición.
//! 4. Dividir los halos en cuartiles de la propiedad interna (λ ó c).
//! 5. Calcular el sesgo lineal b = (1 + δ_halo) / (1 + δ_matter) para cada cuartil.
//! 6. Medir la correlación de Spearman entre la propiedad y δ_env.
//!
//! ## Referencia
//!
//! Gao, Springel & White (2005), MNRAS 363, L66;
//! Wechsler & Tinker (2018), ARA&A 56, 435.

use gadget_ng_core::Vec3;
use rustfft::{FftPlanner, num_complex::Complex};

// ── Structs públicos ──────────────────────────────────────────────────────

/// Resultado del análisis de assembly bias.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssemblyBiasResult {
    /// Radio de suavizado usado (en unidades de la simulación).
    pub smooth_radius: f64,
    /// Correlación de Spearman entre λ y δ_env.
    pub spearman_lambda: f64,
    /// Correlación de Spearman entre concentración c y δ_env.
    pub spearman_concentration: f64,
    /// Sesgo por cuartil de spin: [(λ_med, b_lambda)].
    pub bias_vs_lambda: Vec<(f64, f64)>,
    /// Sesgo por cuartil de concentración: [(c_med, b_c)].
    pub bias_vs_concentration: Vec<(f64, f64)>,
    /// Número de halos analizados.
    pub n_halos: usize,
}

/// Parámetros para el análisis de assembly bias.
#[derive(Debug, Clone)]
pub struct AssemblyBiasParams {
    /// Radio de suavizado del campo de densidad (en unidades de la simulación).
    pub smooth_radius: f64,
    /// Resolución del grid para el campo de densidad.
    pub mesh: usize,
    /// Número de cuartiles (4 por defecto).
    pub n_quartiles: usize,
}

impl Default for AssemblyBiasParams {
    fn default() -> Self {
        Self {
            smooth_radius: 5.0,
            mesh: 32,
            n_quartiles: 4,
        }
    }
}

// ── API principal ─────────────────────────────────────────────────────────

/// Calcula el assembly bias para un conjunto de halos.
///
/// # Parámetros
/// - `halo_positions`     — posiciones del centro de masa de cada halo.
/// - `halo_masses`        — masas de los halos.
/// - `halo_spins`         — parámetros de spin λ (Peebles) de cada halo.
/// - `halo_concentrations`— concentraciones c de cada halo (0.0 si no disponible).
/// - `all_positions`      — posiciones de TODAS las partículas (para el campo de fondo).
/// - `all_masses`         — masas de TODAS las partículas.
/// - `box_size`           — tamaño de la caja periódica.
/// - `params`             — parámetros de análisis.
#[allow(clippy::too_many_arguments)]
pub fn compute_assembly_bias(
    halo_positions: &[Vec3],
    halo_masses: &[f64],
    halo_spins: &[f64],
    halo_concentrations: &[f64],
    all_positions: &[Vec3],
    all_masses: &[f64],
    box_size: f64,
    params: &AssemblyBiasParams,
) -> AssemblyBiasResult {
    let n_halos = halo_positions.len();
    if n_halos == 0 {
        return AssemblyBiasResult {
            smooth_radius: params.smooth_radius,
            spearman_lambda: 0.0,
            spearman_concentration: 0.0,
            bias_vs_lambda: Vec::new(),
            bias_vs_concentration: Vec::new(),
            n_halos: 0,
        };
    }

    // ── 1. Campo de densidad suavizado δ_env ──────────────────────────────
    let delta_env = compute_smoothed_delta(all_positions, all_masses, box_size, params);

    // ── 2. Interpolar δ_env en las posiciones de los halos ────────────────
    let halo_delta: Vec<f64> = halo_positions
        .iter()
        .map(|&pos| interpolate_trilinear(&delta_env, pos, box_size, params.mesh))
        .collect();

    // ── 3. Correlación de Spearman ────────────────────────────────────────
    let has_spins = halo_spins.len() == n_halos;
    let has_conc =
        halo_concentrations.len() == n_halos && halo_concentrations.iter().any(|&c| c > 0.0);

    let spearman_lambda = if has_spins {
        spearman_correlation(halo_spins, &halo_delta)
    } else {
        0.0
    };

    let spearman_concentration = if has_conc {
        spearman_correlation(halo_concentrations, &halo_delta)
    } else {
        0.0
    };

    // ── 4. Sesgo por cuartiles ────────────────────────────────────────────
    let bias_vs_lambda = if has_spins && n_halos >= params.n_quartiles {
        bias_by_quartile(halo_spins, &halo_delta, halo_masses, params.n_quartiles)
    } else {
        Vec::new()
    };

    let bias_vs_concentration = if has_conc && n_halos >= params.n_quartiles {
        bias_by_quartile(
            halo_concentrations,
            &halo_delta,
            halo_masses,
            params.n_quartiles,
        )
    } else {
        Vec::new()
    };

    AssemblyBiasResult {
        smooth_radius: params.smooth_radius,
        spearman_lambda,
        spearman_concentration,
        bias_vs_lambda,
        bias_vs_concentration,
        n_halos,
    }
}

// ── Campo de densidad suavizado ───────────────────────────────────────────

/// Construye el campo δ suavizado con filtro top-hat esférico de radio R_smooth.
fn compute_smoothed_delta(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    params: &AssemblyBiasParams,
) -> Vec<f64> {
    let n = params.mesh;
    let n3 = n * n * n;
    let cell = box_size / n as f64;
    let total_mass: f64 = masses.iter().sum();
    let mean_rho = total_mass / box_size.powi(3);
    let vol_cell = cell.powi(3);

    // CIC deposit
    let mut rho = vec![0.0f64; n3];
    for (&pos, &m) in positions.iter().zip(masses.iter()) {
        cic_assign(&mut rho, pos, m, n, cell);
    }

    // Convertir a sobredensidad
    let mut buf: Vec<Complex<f64>> = rho
        .iter()
        .map(|&r| Complex::new(r / (mean_rho * vol_cell) - 1.0, 0.0))
        .collect();

    // FFT 3D
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    for row in buf.chunks_exact_mut(n) {
        fft.process(row);
    }
    fft_axis_y(&mut buf, n, &fft);
    fft_axis_x(&mut buf, n, &fft);

    // Aplicar filtro top-hat esférico W(kR) = 3[sin(kR) - kR·cos(kR)] / (kR)³
    let k_fund = 2.0 * std::f64::consts::PI / box_size;
    let r_smooth = params.smooth_radius;
    for ix in 0..n {
        let kx = freq(ix, n) as f64 * k_fund;
        for iy in 0..n {
            let ky = freq(iy, n) as f64 * k_fund;
            for iz in 0..n {
                let kz = freq(iz, n) as f64 * k_fund;
                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();
                let w = tophat_window(k_mag * r_smooth);
                let idx = ix * n * n + iy * n + iz;
                buf[idx] *= w;
            }
        }
    }

    // IFFT 3D
    for row in buf.chunks_exact_mut(n) {
        ifft.process(row);
    }
    fft_axis_y(&mut buf, n, &ifft);
    fft_axis_x(&mut buf, n, &ifft);

    // Normalizar
    let norm = 1.0 / n3 as f64;
    buf.iter().map(|c| c.re * norm).collect()
}

/// Filtro top-hat esférico W(x) = 3[sin(x) - x·cos(x)] / x³.
#[inline]
fn tophat_window(x: f64) -> f64 {
    if x < 1e-6 {
        1.0
    } else {
        3.0 * (x.sin() - x * x.cos()) / (x * x * x)
    }
}

/// Interpolación trilineal del campo en la posición pos.
fn interpolate_trilinear(field: &[f64], pos: Vec3, box_size: f64, n: usize) -> f64 {
    let cell = box_size / n as f64;
    let fx = (pos.x / cell).rem_euclid(n as f64);
    let fy = (pos.y / cell).rem_euclid(n as f64);
    let fz = (pos.z / cell).rem_euclid(n as f64);
    let ix = fx.floor() as usize;
    let iy = fy.floor() as usize;
    let iz = fz.floor() as usize;
    let tx = fx - ix as f64;
    let ty = fy - iy as f64;
    let tz = fz - iz as f64;

    let mut val = 0.0f64;
    for (ddx, wx) in [(0usize, 1.0 - tx), (1, tx)] {
        for (ddy, wy) in [(0usize, 1.0 - ty), (1, ty)] {
            for (ddz, wz) in [(0usize, 1.0 - tz), (1, tz)] {
                let jx = (ix + ddx) % n;
                let jy = (iy + ddy) % n;
                let jz = (iz + ddz) % n;
                val += field[jx * n * n + jy * n + jz] * wx * wy * wz;
            }
        }
    }
    val
}

// ── Estadística ───────────────────────────────────────────────────────────

/// Coeficiente de correlación de Spearman entre x e y.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 3 {
        return 0.0;
    }
    let rx = rank_vector(x);
    let ry = rank_vector(y);
    pearson_correlation(&rx, &ry)
}

/// Devuelve el vector de rangos (media de rangos para empates).
fn rank_vector(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Correlación de Pearson entre vectores x e y.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mx) * (yi - my))
        .sum::<f64>()
        / n;
    let sx = (x.iter().map(|&xi| (xi - mx).powi(2)).sum::<f64>() / n).sqrt();
    let sy = (y.iter().map(|&yi| (yi - my).powi(2)).sum::<f64>() / n).sqrt();
    if sx < 1e-30 || sy < 1e-30 {
        0.0
    } else {
        cov / (sx * sy)
    }
}

/// Calcula el sesgo lineal b en bins de la propiedad `prop`, en función de δ_env.
///
/// Para cada cuartil: b = ⟨(1+δ_env)⟩_quartile / ⟨(1+δ_env)⟩_all − 1.
fn bias_by_quartile(
    prop: &[f64],
    delta_env: &[f64],
    masses: &[f64],
    n_quartiles: usize,
) -> Vec<(f64, f64)> {
    let n = prop.len().min(delta_env.len());
    if n < n_quartiles {
        return Vec::new();
    }

    // Ordenar halos por prop
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        prop[a]
            .partial_cmp(&prop[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Media global ponderada por masa
    let total_m: f64 = if masses.len() >= n {
        (0..n).map(|i| masses[i]).sum()
    } else {
        n as f64
    };
    let delta_mean: f64 = if masses.len() >= n {
        (0..n).map(|i| delta_env[i] * masses[i]).sum::<f64>() / total_m
    } else {
        delta_env[..n].iter().sum::<f64>() / n as f64
    };

    let q_size = n / n_quartiles;
    let mut result = Vec::with_capacity(n_quartiles);

    for q in 0..n_quartiles {
        let start = q * q_size;
        let end = if q + 1 == n_quartiles {
            n
        } else {
            (q + 1) * q_size
        };
        let slice = &idx[start..end];

        let prop_med = prop[slice[slice.len() / 2]];

        let (delta_q, w_q) = slice.iter().fold((0.0f64, 0.0f64), |(d, w), &i| {
            let m = if masses.len() > i { masses[i] } else { 1.0 };
            (d + delta_env[i] * m, w + m)
        });
        let delta_q_mean = if w_q > 0.0 { delta_q / w_q } else { 0.0 };

        let bias = if delta_mean.abs() > 1e-8 {
            (1.0 + delta_q_mean) / (1.0 + delta_mean) - 1.0
        } else {
            delta_q_mean
        };

        result.push((prop_med, bias));
    }
    result
}

// ── Helpers FFT ───────────────────────────────────────────────────────────

#[inline]
fn freq(i: usize, n: usize) -> i64 {
    let h = (n / 2) as i64;
    let ii = i as i64;
    if ii <= h { ii } else { ii - n as i64 }
}

fn cic_assign(grid: &mut [f64], pos: Vec3, m: f64, n: usize, cell: f64) {
    let n2 = n * n;
    let fx = (pos.x / cell).rem_euclid(n as f64);
    let fy = (pos.y / cell).rem_euclid(n as f64);
    let fz = (pos.z / cell).rem_euclid(n as f64);
    let ix = fx.floor() as usize;
    let iy = fy.floor() as usize;
    let iz = fz.floor() as usize;
    let tx = fx - ix as f64;
    let ty = fy - iy as f64;
    let tz = fz - iz as f64;
    for (ddx, wx) in [(0usize, 1.0 - tx), (1, tx)] {
        for (ddy, wy) in [(0usize, 1.0 - ty), (1, ty)] {
            for (ddz, wz) in [(0usize, 1.0 - tz), (1, tz)] {
                let jx = (ix + ddx) % n;
                let jy = (iy + ddy) % n;
                let jz = (iz + ddz) % n;
                grid[jx * n2 + jy * n + jz] += m * wx * wy * wz;
            }
        }
    }
}

fn fft_axis_y(buf: &mut [Complex<f64>], n: usize, fft: &std::sync::Arc<dyn rustfft::Fft<f64>>) {
    let mut tmp = vec![Complex::default(); n];
    for ix in 0..n {
        for iz in 0..n {
            for iy in 0..n {
                tmp[iy] = buf[ix * n * n + iy * n + iz];
            }
            fft.process(&mut tmp);
            for iy in 0..n {
                buf[ix * n * n + iy * n + iz] = tmp[iy];
            }
        }
    }
}

fn fft_axis_x(buf: &mut [Complex<f64>], n: usize, fft: &std::sync::Arc<dyn rustfft::Fft<f64>>) {
    let mut tmp = vec![Complex::default(); n];
    for iy in 0..n {
        for iz in 0..n {
            for ix in 0..n {
                tmp[ix] = buf[ix * n * n + iy * n + iz];
            }
            fft.process(&mut tmp);
            for ix in 0..n {
                buf[ix * n * n + iy * n + iz] = tmp[ix];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn make_random_halos(n: usize, seed: u64) -> (Vec<Vec3>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut s = seed;
        let lcg = |ss: &mut u64| -> f64 {
            *ss = ss
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*ss >> 33) as f64) / (u32::MAX as f64)
        };
        let mut pos = Vec::new();
        let mut mass = Vec::new();
        let mut spins = Vec::new();
        let mut conc = Vec::new();
        for _ in 0..n {
            pos.push(Vec3::new(lcg(&mut s), lcg(&mut s), lcg(&mut s)));
            mass.push(1e10 * (1.0 + lcg(&mut s)));
            spins.push(lcg(&mut s) * 0.1 + 0.01);
            conc.push(5.0 + lcg(&mut s) * 10.0);
        }
        (pos, mass, spins, conc)
    }

    #[test]
    fn assembly_bias_empty_halos() {
        let params = AssemblyBiasParams::default();
        let result = compute_assembly_bias(&[], &[], &[], &[], &[], &[], 1.0, &params);
        assert_eq!(result.n_halos, 0);
        assert_eq!(result.bias_vs_lambda.len(), 0);
    }

    #[test]
    fn assembly_bias_returns_finite() {
        let (pos, mass, spins, conc) = make_random_halos(20, 42);
        let params = AssemblyBiasParams {
            smooth_radius: 0.1,
            mesh: 8,
            n_quartiles: 4,
        };
        let result = compute_assembly_bias(&pos, &mass, &spins, &conc, &pos, &mass, 1.0, &params);
        assert!(result.spearman_lambda.is_finite(), "ρ_λ no finito");
        assert!(result.spearman_concentration.is_finite(), "ρ_c no finito");
        assert_eq!(result.n_halos, 20);
    }

    #[test]
    fn assembly_bias_quartiles_count() {
        let (pos, mass, spins, conc) = make_random_halos(40, 7);
        let params = AssemblyBiasParams {
            smooth_radius: 0.1,
            mesh: 8,
            n_quartiles: 4,
        };
        let result = compute_assembly_bias(&pos, &mass, &spins, &conc, &pos, &mass, 1.0, &params);
        assert_eq!(result.bias_vs_lambda.len(), 4, "Debe haber 4 cuartiles");
        assert_eq!(result.bias_vs_concentration.len(), 4);
    }

    #[test]
    fn spearman_perfect_monotone() {
        // Correlación perfecta → ρ = 1.0
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..10).map(|i| i as f64 * 2.0 + 1.0).collect();
        let rho = spearman_correlation(&x, &y);
        assert!(
            (rho - 1.0).abs() < 1e-10,
            "ρ debe ser 1.0 para monotonía perfecta: {rho}"
        );
    }

    #[test]
    fn spearman_anti_monotone() {
        // Correlación inversa → ρ = -1.0
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..10).map(|i| -(i as f64)).collect();
        let rho = spearman_correlation(&x, &y);
        assert!(
            (rho + 1.0).abs() < 1e-10,
            "ρ debe ser -1.0 para monotonía inversa: {rho}"
        );
    }

    #[test]
    fn spearman_independent_near_zero() {
        // Variables independientes → ρ ≈ 0
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]; // pi decimals
        let rho = spearman_correlation(&x, &y);
        assert!(
            rho.abs() < 0.7,
            "ρ no debería ser extremo para datos no monotónos: {rho}"
        );
    }

    #[test]
    fn tophat_window_k0() {
        // W(0) = 1
        assert!((tophat_window(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tophat_window_decays() {
        // W debe decrecer para x > 0
        let w0 = tophat_window(0.0);
        let w1 = tophat_window(1.0);
        let w5 = tophat_window(5.0);
        assert!(w0 > w1, "W debe decrecer");
        assert!(w1.abs() > w5.abs() || w5.abs() < 0.5);
    }

    #[test]
    fn result_serializes() {
        let r = AssemblyBiasResult {
            smooth_radius: 5.0,
            spearman_lambda: 0.3,
            spearman_concentration: -0.2,
            bias_vs_lambda: vec![(0.02, 0.1), (0.05, -0.1)],
            bias_vs_concentration: vec![(8.0, 0.2)],
            n_halos: 100,
        };
        let s = serde_json::to_string(&r).unwrap();
        let r2: AssemblyBiasResult = serde_json::from_str(&s).unwrap();
        assert_eq!(r2.n_halos, 100);
        assert!((r2.spearman_lambda - 0.3).abs() < 1e-10);
    }
}
