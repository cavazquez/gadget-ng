//! Errores de validación de configuración ([`crate::config::RunConfig::validate`]).
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("universo plano: omega_m + omega_lambda = {sum:.6} (se espera ~1, tol ±{tol})")]
    NonFlatUniverse { sum: f64, tol: f64 },
    #[error("cosmology.a_init debe ser > 0 (actual: {0})")]
    AInitNonPositive(f64),
    #[error("simulation.softening debe ser > 0 (actual: {0})")]
    SofteningNonPositive(f64),
    #[error(
        "particle_count = {0} no es un cubo perfecto (requerido para IC tipo retícula / Zel'dovich)"
    )]
    ParticleCountNotPerfectCube(usize),
    #[error("gravity.pm_grid_size = {0} debe ser una potencia de 2")]
    PmGridNotPowerOfTwo(usize),
    #[error("solver PM/TreePM requiere cosmology.periodic = true")]
    PeriodicRequiredForPm,
    #[error("sph.agn.pbh_m_seed debe ser > 0 cuando PBH seeding está activo (actual: {0})")]
    PbhSeedMassNonPositive(f64),
    #[error("sph.agn.pbh_n_seeds debe ser > 0 cuando PBH seeding está activo")]
    PbhSeedCountZero,
    #[error("sph.agn.pbh_min_host_mass debe ser >= 0 (actual: {0})")]
    PbhMinHostMassNegative(f64),
    #[error("sph.agn.initial_spin debe estar en [-0.998, 0.998] (actual: {0})")]
    AgnInitialSpinOutOfRange(f64),
    #[error("sph.gas_fraction debe estar en [0, 1] (actual: {0})")]
    GasFractionOutOfRange(f64),
    #[error("{field} debe ser >= 0 (actual: {value})")]
    NegativeParameter { field: &'static str, value: f64 },
    #[error("{field} debe ser > 0 (actual: {value})")]
    NonPositiveParameter { field: &'static str, value: f64 },
    #[error("{feature} requiere activar {requires}")]
    FeatureRequires {
        feature: &'static str,
        requires: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_non_flat_universe_mentions_omegas() {
        let err = ConfigError::NonFlatUniverse {
            sum: 0.5,
            tol: 0.05,
        };
        let msg = err.to_string();
        assert!(msg.contains("omega_m"));
        assert!(msg.contains("0.5"));
    }

    #[test]
    fn display_feature_requires_names_dependency() {
        let err = ConfigError::FeatureRequires {
            feature: "sph.feedback",
            requires: "sph.enabled",
        };
        let msg = err.to_string();
        assert!(msg.contains("sph.feedback"));
        assert!(msg.contains("sph.enabled"));
    }

    #[test]
    fn display_pm_grid_not_power_of_two() {
        let err = ConfigError::PmGridNotPowerOfTwo(30);
        assert!(err.to_string().contains("30"));
    }
}
