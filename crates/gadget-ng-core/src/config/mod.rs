//! Configuración global de la simulación.
//!
//! Tipos por dominio viven en el submódulo `sections` (archivos por área:
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
    /// Materia oscura warm/fuzzy: cutoff de pequeña escala en ICs (Phase 184).
    #[serde(default)]
    pub dark_matter: DarkMatterSection,
    /// Kernels CUDA opt-in por módulo de física (smoke/parity; requiere `use_gpu_cuda = true`).
    #[serde(default)]
    pub accelerators: AcceleratorsSection,
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
        if l <= 0.0 { 1.0 } else { 1.0 / (l * l * l) }
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
        if !(0.0..=1.0).contains(&self.sph.gas_fraction) {
            return Err(ConfigError::GasFractionOutOfRange(self.sph.gas_fraction));
        }
        if self.sph.feedback.wind.enabled && !self.sph.feedback.enabled {
            return Err(ConfigError::FeatureRequires {
                feature: "sph.feedback.wind",
                requires: "sph.feedback",
            });
        }
        if self.sph.feedback.enabled && !self.sph.enabled {
            return Err(ConfigError::FeatureRequires {
                feature: "sph.feedback",
                requires: "sph",
            });
        }
        if self.sph.cr.enabled && !self.sph.enabled {
            return Err(ConfigError::FeatureRequires {
                feature: "sph.cr",
                requires: "sph",
            });
        }
        if self.sph.cr.anisotropic_diffusion && !self.mhd.enabled {
            return Err(ConfigError::FeatureRequires {
                feature: "sph.cr.anisotropic_diffusion",
                requires: "mhd",
            });
        }
        if self.sph.cr.streaming_coefficient > 0.0 && !self.mhd.enabled {
            return Err(ConfigError::FeatureRequires {
                feature: "sph.cr.streaming_coefficient",
                requires: "mhd",
            });
        }
        if self.turbulence.enabled && !self.mhd.enabled {
            return Err(ConfigError::FeatureRequires {
                feature: "turbulence",
                requires: "mhd",
            });
        }
        if self.mhd.ambipolar_diffusion_enabled && !self.mhd.enabled {
            return Err(ConfigError::FeatureRequires {
                feature: "mhd.ambipolar_diffusion",
                requires: "mhd",
            });
        }
        if self.rt.multifrequency_enabled && !self.rt.enabled {
            return Err(ConfigError::FeatureRequires {
                feature: "rt.multifrequency_enabled",
                requires: "rt",
            });
        }
        if self.reionization.enabled && !self.rt.enabled {
            return Err(ConfigError::FeatureRequires {
                feature: "reionization",
                requires: "rt",
            });
        }
        if self.sph.feedback.wind.v_wind_km_s < 0.0 {
            return Err(ConfigError::NegativeParameter {
                field: "sph.feedback.wind.v_wind_km_s",
                value: self.sph.feedback.wind.v_wind_km_s,
            });
        }
        if self.sph.cr.kappa_cr < 0.0 {
            return Err(ConfigError::NegativeParameter {
                field: "sph.cr.kappa_cr",
                value: self.sph.cr.kappa_cr,
            });
        }
        if self.turbulence.amplitude < 0.0 {
            return Err(ConfigError::NegativeParameter {
                field: "turbulence.amplitude",
                value: self.turbulence.amplitude,
            });
        }
        if self.mhd.ambipolar_eta < 0.0 {
            return Err(ConfigError::NegativeParameter {
                field: "mhd.ambipolar_eta",
                value: self.mhd.ambipolar_eta,
            });
        }
        if self.mhd.ambipolar_ion_floor <= 0.0 {
            return Err(ConfigError::NonPositiveParameter {
                field: "mhd.ambipolar_ion_floor",
                value: self.mhd.ambipolar_ion_floor,
            });
        }
        if self.mhd.ambipolar_dust_coupling < 0.0 {
            return Err(ConfigError::NegativeParameter {
                field: "mhd.ambipolar_dust_coupling",
                value: self.mhd.ambipolar_dust_coupling,
            });
        }
        if self.mhd.ohmic_enabled && !self.mhd.enabled {
            return Err(ConfigError::FeatureRequires {
                feature: "mhd.ohmic_diffusion",
                requires: "mhd.enabled",
            });
        }
        if self.mhd.ohmic_eta < 0.0 {
            return Err(ConfigError::NegativeParameter {
                field: "mhd.ohmic_eta",
                value: self.mhd.ohmic_eta,
            });
        }
        if self.sidm.enabled && self.sidm.sigma_m < 0.0 {
            return Err(ConfigError::NegativeParameter {
                field: "sidm.sigma_m",
                value: self.sidm.sigma_m,
            });
        }
        if self.modified_gravity.enabled && self.modified_gravity.f_r0 <= 0.0 {
            return Err(ConfigError::NonPositiveParameter {
                field: "modified_gravity.f_r0",
                value: self.modified_gravity.f_r0,
            });
        }
        if self.dark_matter.enabled && self.dark_matter.m_wdm_kev <= 0.0 {
            return Err(ConfigError::NonPositiveParameter {
                field: "dark_matter.m_wdm_kev",
                value: self.dark_matter.m_wdm_kev,
            });
        }
        if self.dark_matter.enabled && self.dark_matter.m_fdm_22 <= 0.0 {
            return Err(ConfigError::NonPositiveParameter {
                field: "dark_matter.m_fdm_22",
                value: self.dark_matter.m_fdm_22,
            });
        }
        if self.sph.dust.enabled {
            if self.sph.dust.d_to_g_max < 0.0 {
                return Err(ConfigError::NegativeParameter {
                    field: "sph.dust.d_to_g_max",
                    value: self.sph.dust.d_to_g_max,
                });
            }
            if self.sph.dust.silicate_fraction < 0.0 {
                return Err(ConfigError::NegativeParameter {
                    field: "sph.dust.silicate_fraction",
                    value: self.sph.dust.silicate_fraction,
                });
            }
            if self.sph.dust.graphite_fraction < 0.0 {
                return Err(ConfigError::NegativeParameter {
                    field: "sph.dust.graphite_fraction",
                    value: self.sph.dust.graphite_fraction,
                });
            }
            if self.sph.dust.silicate_fraction + self.sph.dust.graphite_fraction <= 0.0 {
                return Err(ConfigError::NonPositiveParameter {
                    field: "sph.dust.silicate_fraction + sph.dust.graphite_fraction",
                    value: 0.0,
                });
            }
            if self.sph.dust.kappa_silicate_uv < 0.0 {
                return Err(ConfigError::NegativeParameter {
                    field: "sph.dust.kappa_silicate_uv",
                    value: self.sph.dust.kappa_silicate_uv,
                });
            }
            if self.sph.dust.kappa_graphite_uv < 0.0 {
                return Err(ConfigError::NegativeParameter {
                    field: "sph.dust.kappa_graphite_uv",
                    value: self.sph.dust.kappa_graphite_uv,
                });
            }
            if self.sph.dust.h2_shielding_boost < 0.0 {
                return Err(ConfigError::NegativeParameter {
                    field: "sph.dust.h2_shielding_boost",
                    value: self.sph.dust.h2_shielding_boost,
                });
            }
        }
        if self.sph.agn.pbh_seeding_enabled {
            if self.sph.agn.pbh_n_seeds == 0 {
                return Err(ConfigError::PbhSeedCountZero);
            }
            if self.sph.agn.pbh_m_seed <= 0.0 {
                return Err(ConfigError::PbhSeedMassNonPositive(self.sph.agn.pbh_m_seed));
            }
            if self.sph.agn.pbh_min_host_mass < 0.0 {
                return Err(ConfigError::PbhMinHostMassNegative(
                    self.sph.agn.pbh_min_host_mass,
                ));
            }
            if !(-0.998..=0.998).contains(&self.sph.agn.initial_spin) {
                return Err(ConfigError::AgnInitialSpinOutOfRange(
                    self.sph.agn.initial_spin,
                ));
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CosmologySection, GravitySection, IcKind, InitialConditionsSection, OutputSection,
        PerformanceSection, SimulationSection, SolverKind, TimestepSection, UnitsSection,
    };

    fn minimal_run_config() -> RunConfig {
        RunConfig {
            simulation: SimulationSection {
                dt: 0.01,
                num_steps: 10,
                softening: 0.05,
                physical_softening: false,
                gravitational_constant: 1.0,
                particle_count: 8,
                box_size: 2.0,
                seed: 42,
                integrator: Default::default(),
            },
            initial_conditions: InitialConditionsSection {
                kind: IcKind::Lattice,
            },
            output: OutputSection::default(),
            gravity: GravitySection::default(),
            performance: PerformanceSection::default(),
            timestep: TimestepSection::default(),
            cosmology: CosmologySection::default(),
            units: UnitsSection::default(),
            decomposition: Default::default(),
            insitu_analysis: Default::default(),
            sph: Default::default(),
            rt: Default::default(),
            reionization: Default::default(),
            mhd: Default::default(),
            turbulence: Default::default(),
            two_fluid: Default::default(),
            sidm: Default::default(),
            modified_gravity: Default::default(),
            dark_matter: Default::default(),
            accelerators: Default::default(),
        }
    }

    #[test]
    fn validate_accepts_minimal_plummer_config() {
        let mut cfg = minimal_run_config();
        cfg.simulation.particle_count = 512;
        cfg.initial_conditions.kind = IcKind::Plummer { a: 1.0 };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_non_flat_cosmology() {
        let mut cfg = minimal_run_config();
        cfg.cosmology.enabled = true;
        cfg.cosmology.omega_m = 0.2;
        cfg.cosmology.omega_lambda = 0.2;
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::NonFlatUniverse { .. }));
    }

    #[test]
    fn validate_rejects_sph_feedback_without_sph() {
        let mut cfg = minimal_run_config();
        cfg.sph.feedback.enabled = true;
        let err = cfg.validate().unwrap_err();
        assert!(matches!(
            err,
            ConfigError::FeatureRequires {
                feature: "sph.feedback",
                ..
            }
        ));
    }

    #[test]
    fn validate_pm_requires_periodic_and_power_of_two_grid() {
        let mut cfg = minimal_run_config();
        cfg.simulation.particle_count = 27;
        cfg.gravity.solver = SolverKind::Pm;
        cfg.gravity.pm_grid_size = 32;
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::PeriodicRequiredForPm));

        cfg.cosmology.periodic = true;
        cfg.gravity.pm_grid_size = 30;
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::PmGridNotPowerOfTwo(30)));
    }

    #[test]
    fn softening_squared_and_rho_bar() {
        let mut cfg = minimal_run_config();
        cfg.simulation.softening = 0.2;
        cfg.simulation.box_size = 4.0;
        assert!((cfg.softening_squared() - 0.04).abs() < 1e-15);
        assert!((cfg.rho_bar() - 1.0 / 64.0).abs() < 1e-15);
    }

    #[test]
    fn softening_warnings_when_physical_without_cosmology() {
        let mut cfg = minimal_run_config();
        cfg.simulation.physical_softening = true;
        let w = cfg.softening_warnings();
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn plummer_toml_round_trip_and_validate() {
        let toml = r#"
[simulation]
dt = 0.01
num_steps = 100
softening = 0.1
particle_count = 512
box_size = 20.0
seed = 1234

[initial_conditions]
kind = { plummer = { a = 1.0 } }

[gravity]
solver = "barnes_hut"
theta = 0.5
"#;
        let cfg: RunConfig = toml::from_str(toml).expect("deserialize plummer example");
        assert!(
            matches!(cfg.initial_conditions.kind, IcKind::Plummer { a } if (a - 1.0).abs() < 1e-12)
        );
        assert!(cfg.validate().is_ok());
        let back = toml::to_string(&cfg).expect("serialize");
        let cfg2: RunConfig = toml::from_str(&back).expect("round-trip");
        assert_eq!(
            cfg.simulation.particle_count,
            cfg2.simulation.particle_count
        );
    }
}
