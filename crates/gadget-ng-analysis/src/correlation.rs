//! Función de correlación de 2 puntos ξ(r).
//!
//! Dos implementaciones complementarias:
//!
//! 1. **`two_point_correlation_fft`** — transforma P(k) a ξ(r) via la suma de Hankel discreta:
//!    ```text
//!    ξ(r) = (1/2π²) Σ_k  k² P(k) sinc(k·r) Δk
//!    ```
//!    Rápida (O(N_k × N_r)), válida para cajas periódicas.
//!
//! 2. **`two_point_correlation_pairs`** — conteo directo de pares DD.
//!    Estimador de Davis-Peebles: ξ(r) = DD/RR − 1,
//!    con RR analítico (distribución uniforme).
//!    Usa O(N²) operaciones; solo apta para N pequeño (<10⁴ partículas).

use crate::power_spectrum::PkBin;
use gadget_ng_core::Vec3;
use std::f64::consts::PI;

// ── Tipos públicos ────────────────────────────────────────────────────────────

/// Un bin de la función de correlación de 2 puntos.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct XiBin {
    /// Radio central del bin (en Mpc/h, o las mismas unidades que `box_size`).
    pub r: f64,
    /// Valor de ξ(r) en el bin.
    pub xi: f64,
    /// Número de pares contados (0 para estimación vía FFT).
    pub n_pairs: u64,
}

// ── Implementación FFT (desde P(k)) ──────────────────────────────────────────

/// Transforma el espectro de potencia `pk` a la función de correlación ξ(r)
/// mediante la suma de Hankel discreta.
///
/// # Parámetros
/// - `pk`: vector de bins de P(k) tal como lo devuelve [`crate::power_spectrum::power_spectrum`].
/// - `box_size`: tamaño de la caja en Mpc/h (para determinar el rango de r).
/// - `n_r_bins`: número de bins logarítmicos en r.
///
/// # Retorna
/// Vector de `XiBin` con r ∈ [r_min, r_max] donde `r_min = box_size / 2N_k`
/// y `r_max = box_size / 2`.
pub fn two_point_correlation_fft(pk: &[PkBin], box_size: f64, n_r_bins: usize) -> Vec<XiBin> {
    if pk.is_empty() || n_r_bins == 0 {
        return Vec::new();
    }

    // Límites de r: desde la escala de Nyquist hasta la mitad de la caja.
    let k_max = pk.last().map(|b| b.k).unwrap_or(1.0);
    let r_min = (PI / k_max).max(box_size * 1e-3);
    let r_max = box_size * 0.5;

    if r_min >= r_max {
        return Vec::new();
    }

    let log_r_min = r_min.ln();
    let log_r_max = r_max.ln();
    let d_log_r = (log_r_max - log_r_min) / n_r_bins as f64;

    // Calcular Δk para cada bin P(k) usando diferencias centradas.
    let dk: Vec<f64> = (0..pk.len())
        .map(|i| {
            let k_lo = if i == 0 {
                0.0
            } else {
                0.5 * (pk[i - 1].k + pk[i].k)
            };
            let k_hi = if i + 1 < pk.len() {
                0.5 * (pk[i].k + pk[i + 1].k)
            } else {
                pk[i].k + (pk[i].k - pk[i - 1].k) * 0.5
            };
            k_hi - k_lo
        })
        .collect();

    let mut bins = Vec::with_capacity(n_r_bins);
    for j in 0..n_r_bins {
        let r = (log_r_min + (j as f64 + 0.5) * d_log_r).exp();
        // ξ(r) = (1/2π²) Σ_k k² P(k) sinc(k r) Δk
        let xi: f64 = pk
            .iter()
            .zip(dk.iter())
            .map(|(b, &dki)| {
                let kr = b.k * r;
                let sinc_kr = if kr < 1e-10 { 1.0 } else { kr.sin() / kr };
                b.k * b.k * b.pk * sinc_kr * dki
            })
            .sum::<f64>()
            / (2.0 * PI * PI);

        bins.push(XiBin { r, xi, n_pairs: 0 });
    }
    bins
}

// ── Implementación de conteo de pares ────────────────────────────────────────

/// Calcula ξ(r) mediante conteo directo de pares.
///
/// Estimador de Davis-Peebles: ξ(r) = DD/RR − 1, donde RR es el esperado
/// para una distribución uniforme en la caja periódica.
///
/// **Complejidad O(N²)** — solo apto para N < ~10 000 partículas.
///
/// # Parámetros
/// - `positions`: posiciones de las partículas.
/// - `box_size`: tamaño de la caja (condiciones de contorno periódicas).
/// - `r_min`, `r_max`: rango de separaciones en las mismas unidades.
/// - `n_bins`: número de bins logarítmicos.
pub fn two_point_correlation_pairs(
    positions: &[Vec3],
    box_size: f64,
    r_min: f64,
    r_max: f64,
    n_bins: usize,
) -> Vec<XiBin> {
    if positions.is_empty() || n_bins == 0 || r_min >= r_max {
        return Vec::new();
    }

    let n = positions.len() as f64;
    let log_r_min = r_min.ln();
    let log_r_max = r_max.ln();
    let d_log_r = (log_r_max - log_r_min) / n_bins as f64;

    let mut dd = vec![0u64; n_bins];

    // Conteo de pares DD con condiciones de contorno periódicas.
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let dx = periodic_diff(positions[i].x - positions[j].x, box_size);
            let dy = periodic_diff(positions[i].y - positions[j].y, box_size);
            let dz = periodic_diff(positions[i].z - positions[j].z, box_size);
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < r_min || r >= r_max {
                continue;
            }
            let bin_idx = ((r.ln() - log_r_min) / d_log_r) as usize;
            if bin_idx < n_bins {
                dd[bin_idx] += 1;
            }
        }
    }

    let volume = box_size * box_size * box_size;

    (0..n_bins)
        .map(|j| {
            let r_lo = (log_r_min + j as f64 * d_log_r).exp();
            let r_hi = (log_r_min + (j as f64 + 1.0) * d_log_r).exp();
            let r_cen = (log_r_min + (j as f64 + 0.5) * d_log_r).exp();

            // RR analítico: pares esperados en anillo [r_lo, r_hi] para distribución uniforme.
            let shell_volume = (4.0 / 3.0) * PI * (r_hi.powi(3) - r_lo.powi(3));
            // Número esperado de pares por partícula × N_pares_totales / N_partículas
            let n_pairs_total = n * (n - 1.0) / 2.0;
            let rr = n_pairs_total * shell_volume / volume;

            let xi = if rr > 0.0 {
                (dd[j] as f64) / rr - 1.0
            } else {
                0.0
            };

            XiBin {
                r: r_cen,
                xi,
                n_pairs: dd[j],
            }
        })
        .collect()
}

// ── Helpers internos ──────────────────────────────────────────────────────────

#[inline]
fn periodic_diff(mut d: f64, box_size: f64) -> f64 {
    let half = box_size * 0.5;
    if d > half {
        d -= box_size;
    } else if d < -half {
        d += box_size;
    }
    d
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::power_spectrum::PkBin;

    fn flat_pk(n_bins: usize, k_min: f64, k_max: f64, amplitude: f64) -> Vec<PkBin> {
        (0..n_bins)
            .map(|i| {
                let k = k_min + (k_max - k_min) * (i as f64 + 0.5) / n_bins as f64;
                PkBin {
                    k,
                    pk: amplitude,
                    n_modes: 1,
                }
            })
            .collect()
    }

    /// ξ(r) desde P(k)=cte debe ser no-negativa a r pequeño.
    #[test]
    fn xi_fft_flat_pk_nonneg_small_r() {
        let box_size = 300.0_f64;
        let pk = flat_pk(50, 2.0 * PI / box_size, 5.0, 1e4);
        let xi = two_point_correlation_fft(&pk, box_size, 20);
        assert!(!xi.is_empty());
        // El primer bin (r pequeño) debe ser positivo para P(k) plano.
        assert!(xi[0].xi > 0.0, "ξ(r_min) = {} debe ser > 0", xi[0].xi);
        println!("[xi_test] ξ(r={}): {:.4}", xi[0].r, xi[0].xi);
    }

    /// Conteo de pares en distribución aleatoria uniforme: ξ estadísticamente cerca de 0.
    ///
    /// Nota: con N pequeño las fluctuaciones de Poisson son grandes (σ ~ 1/√DD).
    /// Se usa N=1000 para que las fluctuaciones sean menores y la prueba sea robusta.
    #[test]
    fn xi_pairs_uniform_near_zero() {
        use gadget_ng_core::Vec3;
        // Genera posiciones pseudo-aleatorias deterministicas (LCG de Knuth).
        let n = 1000usize;
        let box_size = 100.0_f64;
        let mut seed = 12345u64;
        let next_f64 = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Extraer 53 bits (precisión doble) del bit superior del u64.
            ((*s >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
        };
        let positions: Vec<Vec3> = (0..n)
            .map(|_| {
                let x = next_f64(&mut seed) * box_size;
                let y = next_f64(&mut seed) * box_size;
                let z = next_f64(&mut seed) * box_size;
                Vec3::new(x, y, z)
            })
            .collect();

        // Usar bins en el rango donde hay suficientes pares (evitar escalas < separación media).
        let xi = two_point_correlation_pairs(&positions, box_size, 10.0, 40.0, 6);
        assert!(!xi.is_empty());

        // Para distribución uniforme, ξ debe oscilar alrededor de 0.
        // Con N=1000 y escalas donde hay >30 pares/bin, Poisson da σ ~ 0.2-0.5.
        let mean_xi: f64 = xi.iter().map(|b| b.xi.abs()).sum::<f64>() / xi.len() as f64;
        assert!(
            mean_xi < 1.0,
            "|ξ̄| = {mean_xi:.3} demasiado alto para distribución uniforme (N=1000)"
        );
        println!("[xi_test] |ξ̄| uniforme (N=1000) = {mean_xi:.4}");
    }

    /// ξ(r) vacío para inputs vacíos.
    #[test]
    fn xi_empty_inputs() {
        let r1 = two_point_correlation_fft(&[], 100.0, 10);
        assert!(r1.is_empty());
        let r2 = two_point_correlation_pairs(&[], 100.0, 1.0, 50.0, 10);
        assert!(r2.is_empty());
    }
}
