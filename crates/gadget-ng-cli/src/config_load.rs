use crate::error::CliError;
use gadget_ng_core::RunConfig;
use sha2::{Digest, Sha256};
use std::path::Path;

use figment::providers::Format;
use figment::{Figment, providers::Env};

pub fn load_run_config(path: &Path) -> Result<RunConfig, CliError> {
    let figment = Figment::new()
        .merge(figment::providers::Toml::file(path))
        .merge(Env::prefixed("GADGET_NG_").split("__"));
    let cfg: RunConfig = figment.extract::<RunConfig>().map_err(CliError::from)?;
    cfg.validate()?;
    Ok(cfg)
}

pub fn config_canonical_hash(cfg: &RunConfig) -> Result<String, toml::ser::Error> {
    let s = toml::to_string(cfg)?;
    let h = Sha256::digest(s.as_bytes());
    Ok(hex::encode(h))
}

pub fn print_resolved_config(cfg: &RunConfig) -> Result<(), serde_json::Error> {
    println!("{}", serde_json::to_string_pretty(cfg)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{
        CosmologySection, GravitySection, IcKind, InitialConditionsSection, OutputSection,
        PerformanceSection, RunConfig, SimulationSection, TimestepSection, UnitsSection,
    };

    fn minimal_cfg() -> RunConfig {
        RunConfig {
            simulation: SimulationSection {
                dt: 0.01,
                num_steps: 4,
                softening: 0.05,
                physical_softening: false,
                gravitational_constant: 1.0,
                particle_count: 8,
                box_size: 1.0,
                seed: 1,
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
    fn config_canonical_hash_is_deterministic() {
        let cfg = minimal_cfg();
        let h1 = config_canonical_hash(&cfg).expect("hash");
        let h2 = config_canonical_hash(&cfg).expect("hash");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn load_run_config_from_example_plummer() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/plummer_sphere.toml");
        if !path.exists() {
            return;
        }
        let cfg = load_run_config(&path).expect("plummer config");
        assert_eq!(cfg.simulation.particle_count, 512);
    }
}
