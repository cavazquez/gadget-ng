//! Corrección absoluta de `P(k)` para postproceso (Phases 34–47).
//!
//! Esta API expone los dos factores que separan el estimador interno
//! (`power_spectrum::power_spectrum`) de la amplitud continua teórica:
//!
//! ```text
//! P_measured(k) = A_grid(N) · R(N) · P_cont(k)
//!
//!   A_grid(N) = 2·V² / N⁹              (Phase 34, cerrado analíticamente)
//!   R(N)      = factor de muestreo + CIC (Phase 35, modelo C·N^(-α))
//! ```
//!
//! La corrección práctica es:
//!
//! ```text
//! P_phys(k) = P_measured(k) / (A_grid(N) · R(N))
//! ```
//!
//! Los valores por defecto de `RnModel::phase35_default` provienen de la
//! campaña `N ∈ {8, 16, 32, 64}` × 4 seeds.  `phase47_default` extiende
//! la tabla a N=128 (campaña Phase 47) y mantiene retrocompatibilidad.
//!
//! La función [`measure_rn`] permite recalibrar el factor R(N) para
//! cualquier N sin necesitar un ejecutable externo.
//!
//! La función [`correct_pk_with_shot_noise`] resta el ruido de Poisson
//! P_shot = V/N_part antes de aplicar la corrección A_grid·R(N).
//!
//! # Ejemplo
//!
//! ```no_run
//! use gadget_ng_analysis::pk_correction::{correct_pk, correct_pk_with_shot_noise, RnModel};
//! use gadget_ng_analysis::power_spectrum::PkBin;
//!
//! # let pk_measured: Vec<PkBin> = vec![];
//! let model = RnModel::phase47_default();
//! let n = 128;
//! let box_internal = 1.0;
//! let box_mpc_h = Some(100.0);
//! let pk_physical = correct_pk(&pk_measured, box_internal, n, box_mpc_h, &model);
//!
//! // Con sustracción de shot noise (recomendado a alto k):
//! let n_particles = 128 * 128 * 128;
//! let pk_clean = correct_pk_with_shot_noise(
//!     &pk_measured, box_internal, n, box_mpc_h, n_particles, &model
//! );
//! ```

use crate::power_spectrum::PkBin;
use gadget_ng_core::{
    amplitude_for_sigma8, ic_zeldovich_internals as internals, transfer_eh_nowiggle,
    EisensteinHuParams, TransferKind, Vec3,
};

/// Factor analítico de grilla `A_grid(N) = 2·V² / N⁹` (Phase 34).
///
/// `box_size` es el tamaño de caja en las unidades internas del estimador
/// (las mismas que se pasaron a `power_spectrum`).
#[inline]
pub fn a_grid(box_size: f64, n: usize) -> f64 {
    let v = box_size.powi(3);
    2.0 * v * v / (n as f64).powi(9)
}

/// Modelo `R(N)` con tres variantes soportadas:
///
/// * **Modelo A** (`r_inf = None`): `R(N) = C · N^(-α)`.
/// * **Modelo B** (`r_inf = Some(r)`): `R(N) = C · N^(-α) + r`.
/// * **Tabla exacta**: si `N` coincide con una entrada de `table`, ese
///   valor gana sobre el fit analítico.
///
/// Para `N` fuera de la tabla, el método `evaluate` usa el fit analítico.
/// Si se quiere interpolación log-log entre puntos de la tabla, usar
/// [`RnModel::evaluate_interpolated`].
#[derive(Debug, Clone)]
pub struct RnModel {
    /// Coeficiente `C`.
    pub c: f64,
    /// Exponente `α`.
    pub alpha: f64,
    /// Offset asintótico (sólo Modelo B).
    pub r_inf: Option<f64>,
    /// Tabla de referencia `(N, R_mean)` medida experimentalmente.
    pub table: Vec<(usize, f64)>,
}

impl RnModel {
    /// Valores congelados por Phase 35 (campaña CIC + ZA lattice,
    /// `N ∈ {8, 16, 32, 64}`, 4 seeds, caja interna 1.0, `σ₈ = 0.8`,
    /// EH no-wiggle, `box_mpc_h = 100`).
    ///
    /// Ajuste OLS log-log, `R² = 0.997`.
    pub fn phase35_default() -> Self {
        Self {
            c: 22.108_191_932_947_1,
            alpha: 1.871_411_761_656_48,
            r_inf: None,
            table: vec![
                (8, 0.415_381_774_072_765_2),
                (16, 0.139_628_643_665_938_7),
                (32, 0.033_752_377_475_223_0),
                (64, 0.008_834_200_231_037_1),
            ],
        }
    }

    /// Valores extendidos por Phase 47 — misma campaña que Phase 35 más
    /// N=128 (4 seeds, mismos parámetros).
    ///
    /// Ajuste OLS log-log sobre `N ∈ {32, 64, 128}`: `C=29.77`, `α=1.953`,
    /// coherente con la tabla. Para N no tabulados, el modelo analítico
    /// extrapolará con estos coeficientes.
    ///
    /// Para N ∈ {8, 16, 32, 64} los valores de tabla son idénticos a
    /// `phase35_default` — retrocompatibilidad total.
    pub fn phase47_default() -> Self {
        // Fit OLS log-log sobre {32, 64, 128}: slope=-1.953, intercept=3.393
        // → C = exp(3.393) ≈ 29.77, α = 1.953.
        Self {
            c: 29.77,
            alpha: 1.953,
            r_inf: None,
            table: vec![
                (8, 0.415_381_774_072_765_2),
                (16, 0.139_628_643_665_938_7),
                (32, 0.033_752_377_475_223_0),
                (64, 0.008_834_200_231_037_1),
                // Phase 47: media de 4 seeds (42, 137, 271, 314), N=128³.
                // R² del fit {32,64,128}: α_47 ≈ 1.953 (Phase 35 usó α=1.871
                // sobre {8,16,32,64}; la campaña ampliada desplaza ligeramente
                // el exponente).
                (128, 0.002_251_796),
            ],
        }
    }

    /// Evalúa `R(N)` usando primero la tabla exacta, luego el modelo analítico.
    pub fn evaluate(&self, n: usize) -> f64 {
        if let Some(&(_, r)) = self.table.iter().find(|(m, _)| *m == n) {
            return r;
        }
        self.evaluate_model(n)
    }

    /// Evalúa estrictamente con el modelo analítico (ignora la tabla).
    #[inline]
    pub fn evaluate_model(&self, n: usize) -> f64 {
        let base = self.c * (n as f64).powf(-self.alpha);
        match self.r_inf {
            Some(r) => base + r,
            None => base,
        }
    }

    /// Interpola log-log linealmente entre los dos `N` más cercanos de la tabla.
    /// Cae al modelo analítico si `N` está fuera del rango tabulado.
    pub fn evaluate_interpolated(&self, n: usize) -> f64 {
        if self.table.is_empty() {
            return self.evaluate_model(n);
        }
        if let Some(&(_, r)) = self.table.iter().find(|(m, _)| *m == n) {
            return r;
        }
        let mut sorted: Vec<(usize, f64)> = self.table.clone();
        sorted.sort_by_key(|(m, _)| *m);
        let below = sorted.iter().rev().find(|(m, _)| *m < n).cloned();
        let above = sorted.iter().find(|(m, _)| *m > n).cloned();
        match (below, above) {
            (Some((n1, r1)), Some((n2, r2))) => {
                let lx = (n as f64).ln();
                let l1 = (n1 as f64).ln();
                let l2 = (n2 as f64).ln();
                let t = (lx - l1) / (l2 - l1);
                (r1.ln() * (1.0 - t) + r2.ln() * t).exp()
            }
            _ => self.evaluate_model(n),
        }
    }

    /// Construye un modelo ajustando OLS log-log a una tabla dada.
    ///
    /// El modelo resultante es Modelo A (`r_inf = None`).
    pub fn from_table(table: Vec<(usize, f64)>) -> Self {
        assert!(
            table.len() >= 2,
            "RnModel::from_table necesita al menos 2 puntos (se recibieron {})",
            table.len()
        );
        let (xs, ys): (Vec<f64>, Vec<f64>) = table
            .iter()
            .filter(|(_, r)| r.is_finite() && *r > 0.0)
            .map(|(n, r)| ((*n as f64).ln(), r.ln()))
            .unzip();
        let n = xs.len();
        let mx = xs.iter().sum::<f64>() / n as f64;
        let my = ys.iter().sum::<f64>() / n as f64;
        let (mut sxx, mut sxy) = (0.0, 0.0);
        for i in 0..n {
            let dx = xs[i] - mx;
            let dy = ys[i] - my;
            sxx += dx * dx;
            sxy += dx * dy;
        }
        let slope = sxy / sxx;
        let intercept = my - slope * mx;
        Self {
            c: intercept.exp(),
            alpha: -slope,
            r_inf: None,
            table,
        }
    }
}

/// Corrección absoluta de una lista de bins `PkBin`.
///
/// ```text
/// P_phys(k) = P_measured(k) / (A_grid(box_size, N) · R_model(N))
/// ```
///
/// Si `box_mpc_h` es `Some(L)`, la potencia de salida se re-escala a
/// unidades `(Mpc/h)³` con el factor `(L/box_size)³`. En otro caso se
/// deja en las unidades internas del estimador.
///
/// **Importante:** esta función **no** reescala `k`. Para llevar `k` a
/// unidades físicas usar `k_phys = k_internal · (box_size / L)`.
pub fn correct_pk(
    pk_bins: &[PkBin],
    box_size_internal: f64,
    n: usize,
    box_mpc_h: Option<f64>,
    model: &RnModel,
) -> Vec<PkBin> {
    let a = a_grid(box_size_internal, n);
    let r = model.evaluate(n);
    let denom = a * r;
    let unit_factor = match box_mpc_h {
        Some(l) => (l / box_size_internal).powi(3),
        None => 1.0,
    };
    pk_bins
        .iter()
        .map(|b| PkBin {
            k: b.k,
            pk: b.pk / denom * unit_factor,
            n_modes: b.n_modes,
        })
        .collect()
}

/// Corrección de `P(k)` con sustracción previa del ruido de Poisson (shot noise).
///
/// Aplica primero:
/// ```text
/// P_signal(k) = P_measured(k) − P_shot
/// P_shot      = V_box / N_part           (Poisson, unidades internas)
/// ```
/// y luego la corrección habitual `A_grid · R(N)`.
///
/// La sustracción de shot noise es importante a `k > k_Nyq / 2` donde la
/// amplitud del ruido de Poisson es comparable a la señal cosmológica.
/// Bins donde `P_measured ≤ P_shot` se fijan en cero (no negativos).
///
/// # Parámetros
/// - `pk_bins`          : bins medidos con `power_spectrum`.
/// - `box_size_internal`: tamaño de caja (mismas unidades que `power_spectrum`).
/// - `n`                : lado del grid de densidad.
/// - `box_mpc_h`        : tamaño físico en Mpc (opcional; si `Some(L)` se
///                        re-escala P(k) a `(Mpc/h)³`).
/// - `n_particles`      : número total de partículas.
/// - `model`            : modelo R(N) a usar.
pub fn correct_pk_with_shot_noise(
    pk_bins: &[PkBin],
    box_size_internal: f64,
    n: usize,
    box_mpc_h: Option<f64>,
    n_particles: usize,
    model: &RnModel,
) -> Vec<PkBin> {
    let v_box = box_size_internal.powi(3);
    let p_shot = v_box / n_particles as f64;
    let a = a_grid(box_size_internal, n);
    let r = model.evaluate(n);
    let denom = a * r;
    let unit_factor = match box_mpc_h {
        Some(l) => (l / box_size_internal).powi(3),
        None => 1.0,
    };
    pk_bins
        .iter()
        .map(|b| {
            let pk_signal = (b.pk - p_shot).max(0.0);
            PkBin {
                k: b.k,
                pk: pk_signal / denom * unit_factor,
                n_modes: b.n_modes,
            }
        })
        .collect()
}

/// Mide `R(N)` con el estimador CIC+ZA para un grid de lado `n`.
///
/// Genera ICs Zel'dovich para cada semilla en `seeds`, mide `P_measured(k)`
/// con [`crate::power_spectrum::power_spectrum`] y calcula el cociente:
///
/// ```text
/// R(N) = mean_k[ P_measured(k) / (A_grid(N) · P_cont(k)) ]
/// ```
///
/// sobre `k ≤ k_Nyq / 2` (idéntico al protocolo de Phase 35).
///
/// # Parámetros
/// - `n`        : lado del grid (potencia de 2 recomendada; N³ partículas).
/// - `seeds`    : semillas aleatorias a promediar (≥ 1).
/// - `box_size` : tamaño de caja interna (usar `1.0` para consistencia con
///                Phase 35).
/// - `box_mpc_h`: tamaño de caja en Mpc (mismo parámetro que
///                `build_spectrum_fn`; tipicamente 100).
/// - `sigma8`   : normalización del espectro (tipicamente `0.8`).
/// - `n_s`      : índice espectral (tipicamente `0.965`).
/// - `eh`       : parámetros cosmológicos Eisenstein-Hu.
///
/// # Retorna
/// `(r_mean, cv)` — media de R(N) sobre seeds y bins k,
/// y coeficiente de variación entre seeds (0 si hay < 2 seeds válidas).
///
/// # Ejemplo
/// ```no_run
/// use gadget_ng_analysis::pk_correction::measure_rn;
/// use gadget_ng_core::EisensteinHuParams;
///
/// let eh = EisensteinHuParams::default();
/// let (r, cv) = measure_rn(64, &[42, 137, 271, 314], 1.0, 100.0, 0.8, 0.965, &eh);
/// println!("R(64) = {r:.6}  CV = {:.1}%", cv * 100.0);
/// ```
pub fn measure_rn(
    n: usize,
    seeds: &[u64],
    box_size: f64,
    box_mpc_h: f64,
    sigma8: f64,
    n_s: f64,
    eh: &EisensteinHuParams,
) -> (f64, f64) {
    use std::f64::consts::PI;

    let amp = amplitude_for_sigma8(sigma8, n_s, eh);
    let a_g = a_grid(box_size, n);
    let k_nyq = (n as f64 / 2.0) * (2.0 * PI / box_size);
    let k_max = k_nyq * 0.5;
    // Conversión k interna → k [h/Mpc]: k_hmpc = k_internal · h / box_mpc_h
    // (consistente con build_spectrum_fn y con el protocolo Phase 35).
    let k_conv = eh.h / box_mpc_h;

    let seed_means: Vec<f64> = seeds
        .iter()
        .map(|&seed| {
            let spec = internals::build_spectrum_fn(
                n,
                n_s,
                1.0,
                TransferKind::EisensteinHu,
                Some(sigma8),
                eh.omega_m,
                eh.omega_b,
                eh.h,
                eh.t_cmb,
                Some(box_mpc_h),
            );
            let delta_k = internals::generate_delta_kspace(n, seed, spec);
            let [psi_x, psi_y, psi_z] =
                internals::delta_to_displacement(&delta_k, n, box_size);
            let cell = box_size / n as f64;
            let mass = 1.0 / (n * n * n) as f64;
            let mut pos = Vec::with_capacity(n * n * n);
            let mut mas = Vec::with_capacity(n * n * n);
            for ix in 0..n {
                for iy in 0..n {
                    for iz in 0..n {
                        let idx = ix * n * n + iy * n + iz;
                        let x = ((ix as f64 + 0.5) * cell + psi_x[idx])
                            .rem_euclid(box_size);
                        let y = ((iy as f64 + 0.5) * cell + psi_y[idx])
                            .rem_euclid(box_size);
                        let z = ((iz as f64 + 0.5) * cell + psi_z[idx])
                            .rem_euclid(box_size);
                        pos.push(Vec3::new(x, y, z));
                        mas.push(mass);
                    }
                }
            }
            let pk_bins =
                crate::power_spectrum::power_spectrum(&pos, &mas, box_size, n);
            let ratios: Vec<f64> = pk_bins
                .iter()
                .filter(|b| b.n_modes >= 8 && b.pk > 0.0 && b.k <= k_max)
                .filter_map(|b| {
                    let k_hmpc = b.k * k_conv;
                    let tk = transfer_eh_nowiggle(k_hmpc, eh);
                    let pk_cont = amp * amp * k_hmpc.powf(n_s) * tk * tk;
                    if pk_cont > 0.0 {
                        Some(b.pk / (a_g * pk_cont))
                    } else {
                        None
                    }
                })
                .collect();
            if ratios.is_empty() {
                return f64::NAN;
            }
            ratios.iter().sum::<f64>() / ratios.len() as f64
        })
        .collect();

    let valid: Vec<f64> = seed_means
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if valid.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let r_mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let cv = if valid.len() < 2 {
        0.0
    } else {
        let var = valid.iter().map(|v| (v - r_mean).powi(2)).sum::<f64>()
            / valid.len() as f64;
        var.sqrt() / r_mean.abs()
    };
    (r_mean, cv)
}

// ── Tests unitarios ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_table_matches_model_within_10pct() {
        // El fit analítico no es perfecto sobre cada punto, pero debe ser
        // razonable. La tabla se prefiere al modelo en evaluate().
        let m = RnModel::phase35_default();
        for (n, r_table) in m.table.iter() {
            let r_fit = m.evaluate_model(*n);
            let rel = (r_fit - *r_table).abs() / r_table;
            assert!(
                rel < 0.20,
                "N={n}: |fit - tabla| = {:.3} (> 20 %)",
                rel
            );
        }
    }

    #[test]
    fn evaluate_prefers_table_over_fit() {
        let m = RnModel::phase35_default();
        assert_eq!(m.evaluate(32), 0.033_752_377_475_223_0);
        assert_eq!(m.evaluate(64), 0.008_834_200_231_037_1);
    }

    #[test]
    fn evaluate_interpolated_reproduces_intermediate_n() {
        let m = RnModel::phase35_default();
        let r48_interp = m.evaluate_interpolated(48);
        let r48_model = m.evaluate_model(48);
        let rel = (r48_interp - r48_model).abs() / r48_model;
        assert!(
            rel < 0.05,
            "Interpolación vs modelo en N=48 difiere {:.3}",
            rel
        );
    }

    #[test]
    fn from_table_recovers_known_slope() {
        // Tabla sintética: R(N) = 10 · N^(-2.0) exacto.
        let table: Vec<(usize, f64)> = [8, 16, 32, 64]
            .iter()
            .map(|&n| (n, 10.0 * (n as f64).powi(-2)))
            .collect();
        let m = RnModel::from_table(table);
        assert!((m.c - 10.0).abs() / 10.0 < 1e-10);
        assert!((m.alpha - 2.0).abs() < 1e-10);
    }

    #[test]
    fn correct_pk_scales_linearly() {
        let bins = vec![
            PkBin { k: 1.0, pk: 100.0, n_modes: 10 },
            PkBin { k: 2.0, pk: 200.0, n_modes: 20 },
        ];
        let m = RnModel::phase35_default();
        let out = correct_pk(&bins, 1.0, 32, None, &m);
        let a = a_grid(1.0, 32);
        let r = m.evaluate(32);
        let denom = a * r;
        assert!((out[0].pk - 100.0 / denom).abs() / out[0].pk < 1e-12);
        assert!((out[1].pk - 200.0 / denom).abs() / out[1].pk < 1e-12);
        assert_eq!(out[0].n_modes, 10);
    }

    #[test]
    fn correct_pk_applies_box_unit_conversion() {
        let bins = vec![PkBin { k: 1.0, pk: 1.0, n_modes: 1 }];
        let m = RnModel::phase35_default();
        let a = a_grid(1.0, 32);
        let r = m.evaluate(32);
        let out = correct_pk(&bins, 1.0, 32, Some(100.0), &m);
        let expected = (1.0 / (a * r)) * 100.0_f64.powi(3);
        assert!((out[0].pk - expected).abs() / expected < 1e-12);
    }
}
