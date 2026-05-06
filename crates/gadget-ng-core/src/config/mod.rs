//! Configuración global de la simulación.
//!
//! Tipos por dominio viven en el submódulo [`sections`] (archivos por área:
//! simulación/IC, gravedad, salida/rendimiento, etc.); [`RunConfig`] los agrupa.

mod error;
mod sections;

pub use error::ConfigError;
pub use sections::*;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub simulation: SimulationSection,
    pub initial_conditions: InitialConditionsSection,
    #[serde(default)]
    pub output: OutputSection,
    #[serde(default)]
    pub gravity: GravitySection,
    #[serde(default)]
    pub performance: PerformanceSection,
    #[serde(default)]
    pub timestep: TimestepSection,
    #[serde(default)]
    pub cosmology: CosmologySection,
    /// Sistema de unidades físicas (opcional; `enabled = false` por defecto).
    #[serde(default)]
    pub units: UnitsSection,
    /// Configuración de descomposición de dominio (opcional; balanceo por coste de árbol).
    #[serde(default)]
    pub decomposition: DecompositionConfig,
    /// Análisis in-situ durante el loop `stepping` (opcional; desactivado por defecto).
    #[serde(default)]
    pub insitu_analysis: InsituAnalysisSection,
    /// Módulo SPH cosmológico (Phase 66; opcional; desactivado por defecto).
    #[serde(default)]
    pub sph: SphSection,
    /// Solver de transferencia radiativa M1 (Phase 81; opcional; desactivado por defecto).
    #[serde(default)]
    pub rt: RtSection,
    /// Reionización del Universo: fuentes UV puntuales (Phase 89; opcional).
    #[serde(default)]
    pub reionization: ReionizationSection,
    /// Magnetohidrodinámica ideal (Phase 126; opcional; desactivado por defecto).
    #[serde(default)]
    pub mhd: MhdSection,
    /// Forzado de turbulencia MHD Ornstein-Uhlenbeck (Phase 140; opcional; desactivado).
    #[serde(default)]
    pub turbulence: TurbulenceSection,
    /// Plasma de dos fluidos T_e ≠ T_i (Phase 149; opcional; desactivado).
    #[serde(default)]
    pub two_fluid: TwoFluidSection,
    /// Materia oscura auto-interactuante SIDM (Phase 157; opcional; desactivado).
    #[serde(default)]
    pub sidm: SidmSection,
    /// Gravedad modificada f(R) con screening chameleon (Phase 158; opcional; desactivado).
    #[serde(default)]
    pub modified_gravity: ModifiedGravitySection,
}

// ── RunConfig ─────────────────────────────────────────────────────────────────
impl RunConfig {
    pub fn softening_squared(&self) -> f64 {
        let e = self.simulation.softening;
        e * e
    }

    /// Verifica la combinación softening/cosmología y retorna advertencias si las hay.
    ///
    /// - `physical_softening = true` sin `cosmology.enabled = true` no tiene efecto.
    /// - `physical_softening = false` con `cosmology.enabled = true` usa softening comóvil
    ///   constante (comportamiento legacy).
    pub fn softening_warnings(&self) -> Vec<&'static str> {
        let mut warnings = Vec::new();
        if self.simulation.physical_softening && !self.cosmology.enabled {
            warnings.push(
                "physical_softening = true no tiene efecto sin cosmology.enabled = true \
                 (el softening comóvil fijo ya es constante en simulaciones newtonianas)",
            );
        }
        warnings
    }

    /// Densidad media de materia en el cubo [0, box_size)³, asumiendo masa total = 1.
    pub fn rho_bar(&self) -> f64 {
        let l = self.simulation.box_size;
        if l <= 0.0 {
            1.0
        } else {
            1.0 / (l * l * l)
        }
    }

    /// Constante gravitacional efectiva en el modo de integración actual.
    ///
    /// Prioridad (de mayor a menor):
    ///
    /// 1. `units.enabled = true` → `G_int = G_KPC_MSUN_KMPS × mass/length/v²`
    /// 2. `cosmology.enabled = true && cosmology.auto_g = true`
    ///    → `G = 3·Ω_m·H₀²/(8π)` (condición de Friedmann para ρ̄_m=1)
    /// 3. Fallback → `simulation.gravitational_constant`
    ///
    /// Para diagnosticar inconsistencias usa `cosmo_g_diagnostic`.
    pub fn effective_g(&self) -> f64 {
        if self.units.enabled {
            self.units.compute_g()
        } else if self.cosmology.enabled && self.cosmology.auto_g {
            crate::cosmology::g_code_consistent(
                self.cosmology.omega_m,
                self.cosmology.h0,
                self.rho_bar(),
            )
        } else {
            self.simulation.gravitational_constant
        }
    }

    /// Diagnóstico de consistencia cosmológica de G.
    ///
    /// Devuelve `Some((g_consistent, error_relativo))` cuando `cosmology.enabled = true`
    /// y se puede calcular G auto-consistente a partir de `omega_m` y `h0`.
    /// El error relativo mide cuánto difiere `effective_g()` del valor Friedmann-consistente.
    ///
    /// Devuelve `None` si la cosmología está desactivada.
    pub fn cosmo_g_diagnostic(&self) -> Option<(f64, f64)> {
        if !self.cosmology.enabled {
            return None;
        }
        let g_consistent = crate::cosmology::g_code_consistent(
            self.cosmology.omega_m,
            self.cosmology.h0,
            self.rho_bar(),
        );
        let g_used = self.effective_g();
        let rel_err = if g_consistent > 0.0 {
            (g_used - g_consistent).abs() / g_consistent
        } else {
            f64::INFINITY
        };
        Some((g_consistent, rel_err))
    }

    /// Comprueba parámetros que suelen provocar ejecuciones inválidas o incoherentes.
    ///
    /// No sustituye validación física completa; activar desde CLI/despawn opcional.
    pub fn validate(&self) -> Result<(), ConfigError> {
        const FLAT_TOL: f64 = 5e-2;
        if self.cosmology.enabled {
            let sum = self.cosmology.omega_m + self.cosmology.omega_lambda;
            if (sum - 1.0).abs() > FLAT_TOL {
                return Err(ConfigError::NonFlatUniverse { sum, tol: FLAT_TOL });
            }
            if self.cosmology.a_init <= 0.0 {
                return Err(ConfigError::AInitNonPositive(self.cosmology.a_init));
            }
        }

        if self.simulation.softening < 0.0 {
            return Err(ConfigError::SofteningNonPositive(self.simulation.softening));
        }

        let needs_cube = matches!(
            self.initial_conditions.kind,
            IcKind::Lattice | IcKind::PerturbedLattice { .. } | IcKind::Zeldovich { .. }
        );
        if needs_cube {
            let n = self.simulation.particle_count;
            let c = (n as f64).cbrt().round() as usize;
            if c * c * c != n {
                return Err(ConfigError::ParticleCountNotPerfectCube(n));
            }
        }

        let nm = self.gravity.pm_grid_size;
        if matches!(self.gravity.solver, SolverKind::Pm | SolverKind::TreePm) {
            if !self.cosmology.periodic {
                return Err(ConfigError::PeriodicRequiredForPm);
            }
            if nm == 0 || (nm & (nm - 1)) != 0 {
                return Err(ConfigError::PmGridNotPowerOfTwo(nm));
            }
        }

        Ok(())
    }
}
