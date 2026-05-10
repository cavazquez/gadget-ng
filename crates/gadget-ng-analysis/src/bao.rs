//! Detección del pico de Oscilaciones Acústicas de Bariones (BAO) en P(k).
//!
//! ## Modelo físico
//!
//! Las BAO dejan una huella característica en el espectro de potencias P(k)
//! en forma de un pico oscilatorio alrededor de k ~ 0.06–0.1 h/Mpc,
//! correspondiente a la escala del horizonte sonoro ~105 Mpc/h.
//!
//! El método de detección:
//! 1. Ajusta una ley de potencias *no-wiggle* P_nw(k) = A·k^n a los
//!    extremos de la ventana de ajuste (excluyendo la zona central del pico).
//! 2. Calcula el residuo fraccional δ(k) = (P(k) − P_nw(k)) / P_nw(k).
//! 3. Suaviza los residuos con promedio móvil para reducir ruido.
//! 4. Busca el máximo de δ(k) cerca de k_peak_expected.
//! 5. Estima la significancia como |δ_max| / σ(δ_edge).

use crate::power_spectrum::PkBin;

/// Parámetros para la detección del pico BAO.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaoParams {
    /// Número de onda esperado del pico BAO [h/Mpc].
    pub k_peak_expected: f64,
    /// Ancho de la ventana de ajuste alrededor de k_peak [h/Mpc].
    pub fit_width: f64,
}

impl Default for BaoParams {
    fn default() -> Self {
        Self {
            k_peak_expected: 0.1,
            fit_width: 0.03,
        }
    }
}

/// Resultado de la detección del pico BAO.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaoResult {
    /// Posición del pico detectado [h/Mpc].
    pub k_peak: f64,
    /// Amplitud fraccional del pico: δ(k_peak) = (P − P_nw) / P_nw.
    pub amplitude: f64,
    /// Significancia del pico (amplitud / σ_ruido).
    pub significance: f64,
    /// Se detectó un pico significativo (significance > 2σ).
    pub detected: bool,
    /// Bins de P(k) usados en el ajuste.
    pub fit_bins: Vec<PkBin>,
}

/// Plantilla Gaussiana BAO: G(k) = exp(−(k − k_peak)² / (2σ²)).
///
/// Retorna 1.0 en k = k_peak y decae gaussianamente con ancho σ.
pub fn bao_template(k: f64, k_peak: f64, sigma: f64) -> f64 {
    let dk = k - k_peak;
    (-0.5 * dk * dk / (sigma * sigma)).exp()
}

/// Promedio móvil simétrico con ventana de semi-ancho `half_window` bins.
fn running_average(values: &[f64], half_window: usize) -> Vec<f64> {
    let n = values.len();
    let mut smoothed = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(half_window);
        let hi = (i + half_window + 1).min(n);
        let count = hi - lo;
        let sum: f64 = values[lo..hi].iter().sum();
        smoothed.push(sum / count as f64);
    }
    smoothed
}

/// Ajuste de ley de potencias por mínimos cuadrados en espacio log-log.
///
/// Retorna (A, n) donde P_nw(k) = A · k^n.
fn fit_power_law(k_vals: &[f64], pk_vals: &[f64]) -> (f64, f64) {
    let n = k_vals.len().min(pk_vals.len());
    if n < 2 {
        return (1.0, -1.0);
    }
    let log_k: Vec<f64> = k_vals.iter().map(|k| k.ln()).collect();
    let log_p: Vec<f64> = pk_vals.iter().map(|p| p.ln()).collect();
    let mean_x: f64 = log_k.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = log_p.iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..n {
        let dx = log_k[i] - mean_x;
        let dy = log_p[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
    }
    if den.abs() < 1e-30 {
        return (mean_y.exp(), -1.0);
    }
    let slope = num / den;
    let intercept = mean_y - slope * mean_x;
    (intercept.exp(), slope)
}

/// Detecta el pico BAO en un espectro de potencias P(k).
///
/// # Algoritmo
///
/// 1. Selecciona bins en una ventana alrededor de `k_peak_expected` (±3·fit_width).
/// 2. Ajusta una ley de potencias P_nw(k) a los extremos de la ventana
///    para obtener el espectro *no-wiggle*.
/// 3. Calcula el residuo fraccional δ(k) = (P(k) − P_nw(k)) / P_nw(k).
/// 4. Suaviza los residuos con promedio móvil.
/// 5. Busca el pico de δ(k) más cercano a `k_peak_expected`.
/// 6. Estima la significancia como |δ_max| / σ(δ_edge).
pub fn detect_bao_peak(pk_bins: &[PkBin], params: &BaoParams) -> BaoResult {
    let default_result = || BaoResult {
        k_peak: params.k_peak_expected,
        amplitude: 0.0,
        significance: 0.0,
        detected: false,
        fit_bins: Vec::new(),
    };

    if pk_bins.len() < 10 {
        return default_result();
    }

    let k_lo = params.k_peak_expected - 3.0 * params.fit_width;
    let k_hi = params.k_peak_expected + 3.0 * params.fit_width;

    let window_bins: Vec<&PkBin> = pk_bins
        .iter()
        .filter(|b| b.k >= k_lo && b.k <= k_hi)
        .collect();

    if window_bins.len() < 5 {
        return default_result();
    }

    let k_inner_lo = params.k_peak_expected - params.fit_width;
    let k_inner_hi = params.k_peak_expected + params.fit_width;

    let mut edge_k = Vec::new();
    let mut edge_pk = Vec::new();
    for b in &window_bins {
        if b.k <= k_inner_lo || b.k >= k_inner_hi {
            edge_k.push(b.k);
            edge_pk.push(b.pk);
        }
    }

    let (a_nw, n_nw) = if edge_k.len() >= 2 {
        fit_power_law(&edge_k, &edge_pk)
    } else {
        let k_all: Vec<f64> = window_bins.iter().map(|b| b.k).collect();
        let pk_all: Vec<f64> = window_bins.iter().map(|b| b.pk).collect();
        fit_power_law(&k_all, &pk_all)
    };

    // Residuos fraccionales sin suavizar: δ(k) = (P - P_nw) / P_nw
    let residuals: Vec<f64> = window_bins
        .iter()
        .map(|b| {
            let p_nw = a_nw * b.k.powf(n_nw);
            if p_nw.abs() > 1e-30 {
                (b.pk - p_nw) / p_nw
            } else {
                0.0
            }
        })
        .collect();

    // Suavizar residuos (no P(k)) con promedio móvil
    let smoothed_residuals = running_average(&residuals, 2);

    // Encontrar el máximo de los residuos suavizados cerca de k_peak_expected
    let mut best_idx = 0_usize;
    let mut best_delta = f64::NEG_INFINITY;
    let mut best_dist = f64::INFINITY;
    for (i, &delta) in smoothed_residuals.iter().enumerate() {
        let dist = (window_bins[i].k - params.k_peak_expected).abs();
        if delta > best_delta || ((delta - best_delta).abs() < 1e-12 && dist < best_dist) {
            best_delta = delta;
            best_idx = i;
            best_dist = dist;
        }
    }

    let peak_k = window_bins[best_idx].k;
    let peak_amplitude = smoothed_residuals[best_idx];

    // Estimar σ usando residuos en los bordes (donde P_nw ≈ P y no hay BAO)
    let edge_residuals: Vec<f64> = window_bins
        .iter()
        .zip(residuals.iter())
        .filter(|(b, _)| b.k <= k_inner_lo || b.k >= k_inner_hi)
        .map(|(_, &r)| r)
        .collect();

    let sigma_noise = if edge_residuals.len() >= 2 {
        let mean_r: f64 = edge_residuals.iter().sum::<f64>() / edge_residuals.len() as f64;
        let var: f64 = edge_residuals
            .iter()
            .map(|&r| (r - mean_r) * (r - mean_r))
            .sum::<f64>()
            / (edge_residuals.len() - 1) as f64;
        var.sqrt()
    } else {
        1.0
    };

    let significance = if sigma_noise > 1e-8 {
        peak_amplitude / sigma_noise
    } else {
        0.0
    };

    let detected = significance > 2.0 && peak_amplitude > 0.01;

    BaoResult {
        k_peak: peak_k,
        amplitude: peak_amplitude,
        significance,
        detected,
        fit_bins: window_bins.into_iter().cloned().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_power_law(n: usize, k_min: f64, k_max: f64, amplitude: f64, ns: f64) -> Vec<PkBin> {
        (0..n)
            .map(|i| {
                let k = k_min + (k_max - k_min) * i as f64 / (n - 1) as f64;
                PkBin {
                    k,
                    pk: amplitude * k.powf(ns),
                    n_modes: 100,
                }
            })
            .collect()
    }

    #[test]
    fn no_peak_on_pure_power_law() {
        let bins = generate_power_law(200, 0.01, 0.5, 5000.0, -1.5);
        let params = BaoParams::default();
        let result = detect_bao_peak(&bins, &params);
        assert!(
            !result.detected,
            "Pure power law should not trigger BAO detection; significance = {}",
            result.significance
        );
    }

    #[test]
    fn detect_injected_peak() {
        let mut bins = generate_power_law(200, 0.01, 0.5, 5000.0, -1.5);
        let k_peak = 0.1;
        let sigma = 0.015;
        let amplitude_bao = 0.15;
        for b in &mut bins {
            let bump = 1.0 + amplitude_bao * bao_template(b.k, k_peak, sigma);
            b.pk *= bump;
        }
        let params = BaoParams::default();
        let result = detect_bao_peak(&bins, &params);
        assert!(
            result.detected,
            "Should detect BAO peak at k={}; got k_peak={}, significance={}",
            k_peak, result.k_peak, result.significance
        );
        assert!(
            (result.k_peak - k_peak).abs() < 0.05,
            "Peak position should be near k={}; got {}",
            k_peak,
            result.k_peak
        );
    }

    #[test]
    fn template_shape() {
        let k_peak = 0.1;
        let sigma = 0.02;
        assert!(
            (bao_template(k_peak, k_peak, sigma) - 1.0).abs() < 1e-12,
            "Template at k_peak should be 1.0"
        );
        assert!(
            bao_template(k_peak - sigma, k_peak, sigma) > 0.5,
            "Template at k_peak - sigma should be > 0.5"
        );
        assert!(
            bao_template(k_peak + sigma, k_peak, sigma) > 0.5,
            "Template at k_peak + sigma should be > 0.5"
        );
        assert!(
            bao_template(k_peak + 4.0 * sigma, k_peak, sigma) < 0.02,
            "Template far from peak should be ~0"
        );
    }

    #[test]
    fn params_default() {
        let params = BaoParams::default();
        assert!(
            (params.k_peak_expected - 0.1).abs() < 1e-10,
            "Default k_peak_expected should be 0.1"
        );
        assert!(
            (params.fit_width - 0.03).abs() < 1e-10,
            "Default fit_width should be 0.03"
        );
    }
}
