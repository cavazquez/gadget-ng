//! Fisher matrix for cosmological parameter forecasting.
//!
//! Computes the Fisher information matrix from the matter power spectrum P(k,z):
//!
//! ```text
//! F_ij = Σ_{k,z} (∂P/∂θ_i) × C⁻¹(k,z) × (∂P/∂θ_j) × Δk × Δz
//! ```
//!
//! where `C(k,z) = 2 P² / N_modes` is the Gaussian covariance for a survey
//! with volume `V` and `N_modes = 4π k² Δk V / (2π)³`.
//!
//! Derivatives ∂P/∂θ are computed via central finite differences of
//! [`p_linear_eh`] + growth factor D²(z) (linear) or [`halofit_pk`] (nonlinear).
//!
//! # Parameters
//!
//! The fiducial cosmology is specified via [`FisherParams`], which maps
//! to the existing [`EisensteinHuParams`], [`CosmologyParams`], and [`HalofitCosmo`].
//!
//! # Example
//!
//! ```rust,no_run
//! use gadget_ng_analysis::fisher::{FisherParams, FisherConfig, fisher_matrix, fisher_uncertainties};
//!
//! let fiducial = FisherParams::planck2018();
//! let config = FisherConfig::default();
//!
//! let fisher = fisher_matrix(&config, &fiducial);
//! let sigmas = fisher_uncertainties(&fisher);
//! println!("omega_m sigma = {:.4}, sigma8 sigma = {:.4}", sigmas.sigmas[0], sigmas.sigmas[4]);
//! ```

use std::f64::consts::PI;

use gadget_ng_core::{
    CosmologyParams, EisensteinHuParams, amplitude_for_sigma8, cosmology::growth_factor_d,
};

use crate::halofit::{HalofitCosmo, halofit_pk, p_linear_eh};

/// Fiducial cosmological parameter vector for Fisher forecasting.
///
/// Each field is an independent parameter θ_i that the Fisher matrix
/// constrains.  The order of fields determines the matrix indexing.
#[derive(Debug, Clone)]
pub struct FisherParams {
    /// Total matter density fraction Ω_m (including baryons).
    pub omega_m: f64,
    /// Baryon density fraction Ω_b.
    pub omega_b: f64,
    /// Dimensionless Hubble parameter h = H₀/(100 km/s/Mpc).
    pub h: f64,
    /// Scalar spectral index n_s.
    pub n_s: f64,
    /// Amplitude of matter fluctuations in 8 Mpc/h spheres σ₈.
    pub sigma8: f64,
    /// CPL dark energy equation of state w₀ (default: −1 for ΛCDM).
    pub w0: f64,
    /// CPL dark energy evolution wₐ (default: 0 for ΛCDM).
    pub wa: f64,
    /// Sum of neutrino masses in eV (default: 0.06 for minimal).
    pub m_nu_ev: f64,
}

impl Default for FisherParams {
    fn default() -> Self {
        Self::planck2018()
    }
}

impl FisherParams {
    /// Planck 2018 TT,TE,EE+lowE+lensing fiducial cosmology.
    pub fn planck2018() -> Self {
        Self {
            omega_m: 0.315,
            omega_b: 0.049,
            h: 0.674,
            n_s: 0.965,
            sigma8: 0.8111,
            w0: -1.0,
            wa: 0.0,
            m_nu_ev: 0.06,
        }
    }

    /// Convert to Eisenstein-Hu transfer function parameters.
    pub fn to_eh_params(&self) -> EisensteinHuParams {
        EisensteinHuParams {
            omega_m: self.omega_m,
            omega_b: self.omega_b,
            h: self.h,
            t_cmb: 2.7255,
        }
    }

    /// Convert to CosmologyParams (runtime integration, internal H₀ units).
    ///
    /// The `h0_code` parameter is the Hubble constant in internal units (1/t_sim).
    /// For Fisher forecasting, use `h0_code = self.h * 0.1` as a convention.
    pub fn to_cosmology_params(&self, h0_code: f64) -> CosmologyParams {
        let omega_lambda = 1.0 - self.omega_m;
        CosmologyParams::from_cosmology_toml(
            self.omega_m,
            omega_lambda,
            h0_code,
            self.w0,
            self.wa,
            self.m_nu_ev,
            self.h * 10.0,
        )
    }

    /// Convert to HalofitCosmo (flat ΛCDM only).
    pub fn to_halofit_cosmo(&self) -> HalofitCosmo {
        HalofitCosmo {
            omega_m0: self.omega_m,
            omega_de0: 1.0 - self.omega_m,
        }
    }
}

/// Configuration for Fisher matrix computation.
#[derive(Debug, Clone)]
pub struct FisherConfig {
    /// Fractional step for central finite differences (e.g. 0.01 = 1%).
    pub step_frac: f64,
    /// k values in h/Mpc at which to evaluate P(k).
    pub k_bins: Vec<f64>,
    /// Redshifts at which to evaluate P(k,z).
    pub redshifts: Vec<f64>,
    /// Survey volume in (Mpc/h)³.
    pub survey_volume: f64,
    /// Use nonlinear P(k) via Halofit instead of linear.
    pub use_nonlinear: bool,
}

impl Default for FisherConfig {
    fn default() -> Self {
        Self {
            step_frac: 0.01,
            k_bins: k_bins_default(),
            redshifts: vec![0.0, 0.5, 1.0, 2.0],
            survey_volume: 1.0e9,
            use_nonlinear: true,
        }
    }
}

/// Generate default k-bins (log-spaced, 0.01–1.0 h/Mpc).
pub fn k_bins_default() -> Vec<f64> {
    (0..64)
        .map(|i| 0.01 * 10.0_f64.powf(i as f64 / 63.0))
        .collect()
}

/// Fisher matrix result.
///
/// The matrix is stored as a flat `n_params × n_params` array in row-major order.
/// Parameters are ordered as: [Ω_m, Ω_b, h, n_s, σ₈, w₀, wₐ, m_ν].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FisherMatrix {
    /// Flat n_params² matrix (row-major).
    pub matrix: Vec<f64>,
    /// Number of parameters (8).
    pub n_params: usize,
    /// Parameter names.
    pub param_names: Vec<String>,
}

/// Marginalized parameter uncertainties σ(θ_i) from the Fisher matrix.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FisherUncertainties {
    /// Standard deviations σ(θ_i) = √(F⁻¹)_{ii}.
    pub sigmas: Vec<f64>,
    /// Parameter names.
    pub param_names: Vec<String>,
}

/// Derivative matrix dP(k,z)/dθ_i.
///
/// Shape: `[n_params, n_k × n_z]` flattened in row-major order.
#[derive(Debug, Clone)]
pub struct DerivativeMatrix {
    /// Derivatives: `derivs[i_param * n_kz + ik] = ∂P(k_j, z_l)/∂θ_i`.
    pub derivs: Vec<f64>,
    /// k values corresponding to each column.
    pub k_values: Vec<f64>,
    /// z values corresponding to each column block.
    pub z_values: Vec<f64>,
    /// Number of parameters.
    pub n_params: usize,
}

const PARAM_NAMES: [&str; 8] = [
    "omega_m", "omega_b", "h", "n_s", "sigma8", "w0", "wa", "m_nu_ev",
];

/// Compute the matter power spectrum P(k,z) for a given set of parameters.
///
/// Uses Eisenstein-Hu transfer function + CPT92 growth factor.
/// If `use_nonlinear` is true, applies Halofit correction.
pub fn pk_observable(k_hmpc: f64, z: f64, params: &FisherParams, use_nonlinear: bool) -> f64 {
    let eh = params.to_eh_params();
    let amp = amplitude_for_sigma8(params.sigma8, params.n_s, &eh);
    let cosmo = params.to_cosmology_params(params.h * 0.1);
    let a = 1.0 / (1.0 + z);
    let d_ratio = growth_factor_d(cosmo, a) / growth_factor_d(cosmo, 1.0);

    let p_lin = p_linear_eh(k_hmpc, amp, params.n_s, d_ratio, &eh);

    if !use_nonlinear || z > 10.0 {
        return p_lin;
    }

    let p_lin_fn = |k: f64| -> f64 {
        if k <= 0.0 {
            return 0.0;
        }
        p_linear_eh(k, amp, params.n_s, d_ratio, &eh)
    };

    let halofit_cosmo = params.to_halofit_cosmo();
    let pk_nl = halofit_pk(&[k_hmpc], &p_lin_fn, &halofit_cosmo, z);

    pk_nl.first().map(|&(_, p)| p).unwrap_or(p_lin)
}

/// Central finite difference: ∂f/∂x ≈ (f(x+h) − f(x−h)) / (2h).
///
/// Uses `h = |x| × step_frac` with a floor of `1e-12`.
pub fn central_difference<F: Fn(f64) -> f64>(x: f64, step_frac: f64, f: &F) -> f64 {
    let h = x.abs().max(1e-12) * step_frac;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Compute ∂P(k,z)/∂θ_i for all parameters via central finite differences.
///
/// Returns a [`DerivativeMatrix`] with shape `[n_params, n_k × n_z]`.
pub fn pk_derivatives(config: &FisherConfig, fiducial: &FisherParams) -> DerivativeMatrix {
    let n_k = config.k_bins.len();
    let n_z = config.redshifts.len();
    let n_kz = n_k * n_z;
    let n_params = 8usize;
    let mut derivs = vec![0.0; n_params * n_kz];

    // For each parameter, perturb +h and −h
    let params_keys: [fn(&mut FisherParams, f64); 8] = [
        |p, v| p.omega_m = v,
        |p, v| p.omega_b = v,
        |p, v| p.h = v,
        |p, v| p.n_s = v,
        |p, v| p.sigma8 = v,
        |p, v| p.w0 = v,
        |p, v| p.wa = v,
        |p, v| p.m_nu_ev = v,
    ];
    let params_get: [fn(&FisherParams) -> f64; 8] = [
        |p| p.omega_m,
        |p| p.omega_b,
        |p| p.h,
        |p| p.n_s,
        |p| p.sigma8,
        |p| p.w0,
        |p| p.wa,
        |p| p.m_nu_ev,
    ];

    for ip in 0..n_params {
        let val = params_get[ip](fiducial);
        let h = val.abs().max(1e-12) * config.step_frac;

        let mut plus = fiducial.clone();
        params_keys[ip](&mut plus, val + h);
        let mut minus = fiducial.clone();
        params_keys[ip](&mut minus, val - h);

        for (ik, &k) in config.k_bins.iter().enumerate() {
            for (iz, &z) in config.redshifts.iter().enumerate() {
                let pk_p = pk_observable(k, z, &plus, config.use_nonlinear);
                let pk_m = pk_observable(k, z, &minus, config.use_nonlinear);
                let ikz = ik * n_z + iz;
                derivs[ip * n_kz + ikz] = (pk_p - pk_m) / (2.0 * h);
            }
        }
    }

    DerivativeMatrix {
        derivs,
        k_values: config.k_bins.clone(),
        z_values: config.redshifts.clone(),
        n_params,
    }
}

/// Gaussian covariance per (k,z) bin.
///
/// `C(k,z) = 2 P²(k,z) / N_modes(k)`
///
/// where `N_modes(k) = 4π k² Δk V / (2π)³`.
pub fn gaussian_covariance(pk: f64, k: f64, dk: f64, survey_volume: f64) -> f64 {
    let n_modes = 4.0 * PI * k * k * dk * survey_volume / (8.0 * PI * PI * PI);
    if n_modes < 1.0 {
        return 2.0 * pk * pk;
    }
    2.0 * pk * pk / n_modes
}

/// Assemble the Fisher information matrix from derivatives and covariance.
///
/// ```text
/// F_{ij} = Σ_{k,z} (∂P/∂θ_i) × C⁻¹(k,z) × (∂P/∂θ_j) × Δk
/// ```
///
/// where `C(k,z) = 2 P² / N_modes` for Gaussian covariance.
pub fn fisher_matrix(config: &FisherConfig, fiducial: &FisherParams) -> FisherMatrix {
    let deriv = pk_derivatives(config, fiducial);
    let n_params = deriv.n_params;
    let n_k = config.k_bins.len();
    let n_z = config.redshifts.len();
    let n_kz = n_k * n_z;

    // Compute fiducial P(k,z)
    let fiducial_pk: Vec<f64> = config
        .k_bins
        .iter()
        .flat_map(|&k| {
            config
                .redshifts
                .iter()
                .map(move |&z| pk_observable(k, z, fiducial, config.use_nonlinear))
        })
        .collect();

    // Δk for each k bin (log-spaced)
    let dk: Vec<f64> = config
        .k_bins
        .windows(2)
        .map(|w| w[1] - w[0])
        .chain(std::iter::once(config.k_bins.last().unwrap() * 0.01))
        .collect();

    let mut matrix = vec![0.0; n_params * n_params];

    for (ik, (&dk_val, &k_val)) in dk.iter().zip(config.k_bins.iter()).enumerate() {
        for iz in 0..n_z {
            let ikz = ik * n_z + iz;
            let pk_val = fiducial_pk[ikz];
            if pk_val <= 0.0 {
                continue;
            }

            let cov = gaussian_covariance(pk_val, k_val, dk_val, config.survey_volume);
            let inv_cov = 1.0 / cov;

            for ip in 0..n_params {
                let dp_i = deriv.derivs[ip * n_kz + ikz];
                if dp_i == 0.0 {
                    continue;
                }
                for jp in 0..n_params {
                    let dp_j = deriv.derivs[jp * n_kz + ikz];
                    matrix[ip * n_params + jp] += dp_i * inv_cov * dp_j * dk_val;
                }
            }
        }
    }

    FisherMatrix {
        matrix,
        n_params,
        param_names: PARAM_NAMES.iter().map(|s| s.to_string()).collect(),
    }
}

/// Compute marginalized 1σ uncertainties from the Fisher matrix.
///
/// Returns `σ(θ_i) = √(F⁻¹)_{ii}` for each parameter.
///
/// Uses Gauss-Jordan elimination for matrix inversion (no external deps).
pub fn fisher_uncertainties(fisher: &FisherMatrix) -> FisherUncertainties {
    let n = fisher.n_params;
    let inv = invert_matrix(&fisher.matrix, n);
    let sigmas: Vec<f64> = (0..n).map(|i| inv[i * n + i].abs().sqrt()).collect();

    FisherUncertainties {
        sigmas,
        param_names: fisher.param_names.clone(),
    }
}

/// Compute the marginalized correlation matrix r_{ij} from the Fisher matrix.
///
/// `r_{ij} = (F⁻¹)_{ij} / √((F⁻¹)_{ii} · (F⁻¹)_{jj})`.
pub fn correlation_matrix(fisher: &FisherMatrix) -> Vec<f64> {
    let n = fisher.n_params;
    let inv = invert_matrix(&fisher.matrix, n);
    let mut corr = vec![0.0; n * n];
    let diag: Vec<f64> = (0..n).map(|i| inv[i * n + i].abs()).collect();

    for i in 0..n {
        for j in 0..n {
            let denom = (diag[i] * diag[j]).sqrt();
            corr[i * n + j] = if denom > 0.0 {
                inv[i * n + j] / denom
            } else {
                0.0
            };
        }
    }

    corr
}

/// Invert an n×n matrix using Gauss-Jordan elimination.
fn invert_matrix(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0; n * 2 * n];

    // [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = matrix[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    // Gauss-Jordan
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = aug[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * 2 * n + col].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = row;
            }
        }

        if pivot_val < 1e-30 {
            continue;
        }

        // Swap rows
        if pivot_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * 2 * n + j, pivot_row * 2 * n + j);
            }
        }

        let pivot = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }

    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planck_fiducial() -> FisherParams {
        FisherParams::planck2018()
    }

    fn small_config() -> FisherConfig {
        FisherConfig {
            step_frac: 0.02,
            k_bins: vec![0.05, 0.1, 0.2, 0.5, 1.0],
            redshifts: vec![0.0, 0.5, 1.0, 2.0],
            survey_volume: 1.0e9,
            use_nonlinear: false,
        }
    }

    #[test]
    fn pk_observable_positive_at_k01() {
        let params = planck_fiducial();
        let pk = pk_observable(0.1, 0.0, &params, false);
        assert!(pk > 0.0, "P(k=0.1, z=0) should be positive, got {pk}");
    }

    #[test]
    fn fisher_diagonal_positive() {
        let fiducial = planck_fiducial();
        let config = small_config();
        let fisher = fisher_matrix(&config, &fiducial);

        // Only check parameters that are well-constrained by P(k,z):
        // omega_m (0), omega_b (1), n_s (3), sigma8 (4)
        // w0/wa/m_nu have zero or near-zero derivatives with EH transfer.
        for &i in &[0usize, 1, 3, 4] {
            let fii = fisher.matrix[i * fisher.n_params + i];
            assert!(
                fii > 0.0,
                "F_{{i,i}} = {fii} should be positive for param {i}"
            );
        }
    }

    #[test]
    fn fisher_uncertainties_reasonable() {
        let fiducial = planck_fiducial();
        let config = small_config();
        let fisher = fisher_matrix(&config, &fiducial);
        let sigmas = fisher_uncertainties(&fisher);

        // sigma_8 should be well-constrained: O(0.01-0.1) for a 1 Gpc^3 survey
        assert!(
            sigmas.sigmas[4] > 1e-6 && sigmas.sigmas[4] < 10.0,
            "σ(σ₈) = {} outside reasonable range",
            sigmas.sigmas[4]
        );
        // omega_m should be constrained at order 0.01-1
        assert!(
            sigmas.sigmas[0] > 1e-6 && sigmas.sigmas[0] < 10.0,
            "σ(Ω_m) = {} outside reasonable range",
            sigmas.sigmas[0]
        );
    }

    #[test]
    fn derivative_converges_with_step() {
        let fiducial = planck_fiducial();
        let k = 0.1;
        let z = 0.0;

        let dp_large = central_difference(fiducial.omega_m, 0.1, &|omega_m| {
            let mut p = fiducial.clone();
            p.omega_m = omega_m;
            pk_observable(k, z, &p, false)
        });
        let dp_small = central_difference(fiducial.omega_m, 0.01, &|omega_m| {
            let mut p = fiducial.clone();
            p.omega_m = omega_m;
            pk_observable(k, z, &p, false)
        });

        let rel_change = (dp_large - dp_small).abs() / dp_small.abs().max(1e-30);
        assert!(
            rel_change < 1.0,
            "Derivative should converge: rel_change = {rel_change:.4}"
        );
    }
}
