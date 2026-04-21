//! Corrección absoluta de `P(k)` para postproceso (Phase 35).
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
//! campaña `N ∈ {8, 16, 32, 64}` × 4 seeds descrita en
//! `docs/reports/2026-04-phase35-rn-modeling.md`.
//!
//! # Ejemplo
//!
//! ```no_run
//! use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
//! use gadget_ng_analysis::power_spectrum::PkBin;
//!
//! # let pk_measured: Vec<PkBin> = vec![];
//! let model = RnModel::phase35_default();
//! let n = 64;
//! let box_internal = 1.0;
//! let box_mpc_h = Some(100.0);
//! let pk_physical = correct_pk(&pk_measured, box_internal, n, box_mpc_h, &model);
//! ```

use crate::power_spectrum::PkBin;

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
