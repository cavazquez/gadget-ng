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
}
