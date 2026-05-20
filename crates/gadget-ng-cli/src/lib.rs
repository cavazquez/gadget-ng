pub mod analyze_cmd;
pub mod config_load;
pub mod engine;
pub mod error;
pub mod fisher_cmd;
pub mod insitu;
pub mod mah_cmd;
pub mod merge_tree_cmd;

pub use engine::run_stepping;
pub use engine::{cmd_config_print, render_snapshot_visualization, run_snapshot, run_visualize};

use error::CliError;
use gadget_ng_core::{
    BFieldKind, CoolingKind, DarkMatterModel, DustSpeciesModel, PbhHostKind, RunConfig,
    StarFormationModel, StellarFeedbackMode,
};
use gadget_ng_parallel::ParallelRuntime;
#[cfg(not(feature = "mpi"))]
pub use gadget_ng_parallel::SerialRuntime;

pub fn parse_pbh_host_kind(value: &str) -> Result<PbhHostKind, CliError> {
    match value {
        "dark_matter" | "dm" => Ok(PbhHostKind::DarkMatter),
        "star" | "stars" => Ok(PbhHostKind::Star),
        "collisionless" | "all_collisionless" => Ok(PbhHostKind::Collisionless),
        other => Err(CliError::InvalidConfig(format!(
            "pbh_host_kind inválido: {other}; usar dark_matter, star o collisionless"
        ))),
    }
}

pub fn parse_cooling_kind(value: &str) -> Result<CoolingKind, CliError> {
    match value {
        "none" => Ok(CoolingKind::None),
        "atomic_h_he" | "atomic" | "atomic_hhe" => Ok(CoolingKind::AtomicHHe),
        "metal_cooling" | "metal" => Ok(CoolingKind::MetalCooling),
        "metal_tabular" | "tabular" => Ok(CoolingKind::MetalTabular),
        "uv_background" | "uvb" => Ok(CoolingKind::UvBackground),
        other => Err(CliError::InvalidConfig(format!(
            "cooling inválido: {other}; usar none, atomic_h_he, metal_cooling, metal_tabular o uv_background"
        ))),
    }
}

pub fn parse_sf_model(value: &str) -> Result<StarFormationModel, CliError> {
    match value {
        "density_law" | "density" => Ok(StarFormationModel::DensityLaw),
        "pressure_law" | "pressure" => Ok(StarFormationModel::PressureLaw),
        other => Err(CliError::InvalidConfig(format!(
            "sf_model inválido: {other}; usar density_law o pressure_law"
        ))),
    }
}

pub fn parse_feedback_mode(value: &str) -> Result<StellarFeedbackMode, CliError> {
    match value {
        "kinetic" => Ok(StellarFeedbackMode::Kinetic),
        "thermal_stochastic" | "thermal" => Ok(StellarFeedbackMode::ThermalStochastic),
        other => Err(CliError::InvalidConfig(format!(
            "feedback_mode inválido: {other}; usar kinetic o thermal_stochastic"
        ))),
    }
}

pub fn parse_bfield_kind(value: &str) -> Result<BFieldKind, CliError> {
    match value {
        "none" => Ok(BFieldKind::None),
        "uniform" => Ok(BFieldKind::Uniform),
        "random" => Ok(BFieldKind::Random),
        "spiral" => Ok(BFieldKind::Spiral),
        other => Err(CliError::InvalidConfig(format!(
            "bfield inválido: {other}; usar none, uniform, random o spiral"
        ))),
    }
}

pub fn parse_dark_matter_model(value: &str) -> Result<DarkMatterModel, CliError> {
    match value {
        "cold" | "cdm" => Ok(DarkMatterModel::Cold),
        "warm" | "wdm" => Ok(DarkMatterModel::Warm),
        "fuzzy" | "fdm" => Ok(DarkMatterModel::Fuzzy),
        other => Err(CliError::InvalidConfig(format!(
            "dark_matter inválido: {other}; usar cold, warm o fuzzy"
        ))),
    }
}

pub fn parse_dust_species_model(value: &str) -> Result<DustSpeciesModel, CliError> {
    match value {
        "single" => Ok(DustSpeciesModel::Single),
        "silicate_graphite" | "active" | "colibre" => Ok(DustSpeciesModel::SilicateGraphite),
        other => Err(CliError::InvalidConfig(format!(
            "dust_species inválido: {other}; usar single o silicate_graphite"
        ))),
    }
}

#[expect(clippy::struct_excessive_bools)]
pub struct RuntimeCliOverrides {
    pub pbh_seeding: bool,
    pub pbh_n_seeds: Option<usize>,
    pub pbh_m_seed: Option<f64>,
    pub pbh_min_host_mass: Option<f64>,
    pub pbh_seed: Option<u64>,
    pub pbh_host_kind: Option<String>,
    pub sph: bool,
    pub gas_fraction: Option<f64>,
    pub cooling: Option<String>,
    pub feedback: bool,
    pub sf_model: Option<String>,
    pub feedback_mode: Option<String>,
    pub winds: bool,
    pub wind_velocity: Option<f64>,
    pub agn: bool,
    pub agn_n_bh: Option<usize>,
    pub agn_m_seed: Option<f64>,
    pub agn_eps_feedback: Option<f64>,
    pub agn_radio: bool,
    pub agn_f_edd_threshold: Option<f64>,
    pub agn_spin: Option<f64>,
    pub agn_mergers: bool,
    pub cr: bool,
    pub cr_kappa: Option<f64>,
    pub cr_anisotropic: bool,
    pub cr_streaming: Option<f64>,
    pub mhd: bool,
    pub bfield: Option<String>,
    pub b0x: Option<f64>,
    pub b0y: Option<f64>,
    pub b0z: Option<f64>,
    pub turbulence: bool,
    pub turb_amplitude: Option<f64>,
    pub two_fluid: bool,
    pub ambipolar: bool,
    pub ambipolar_eta: Option<f64>,
    pub ambipolar_ion_floor: Option<f64>,
    pub ambipolar_dust_coupling: Option<f64>,
    pub sidm: bool,
    pub sidm_sigma_m: Option<f64>,
    pub fr: bool,
    pub fr_f_r0: Option<f64>,
    pub fr_n: Option<f64>,
    pub fr_nonlinear_mesh: bool,
    pub rt: bool,
    pub rt_multifrequency: bool,
    pub reionization: bool,
    pub dark_matter: Option<String>,
    pub wdm_mass_kev: Option<f64>,
    pub fdm_mass_22: Option<f64>,
    pub dust: bool,
    pub dust_species: Option<String>,
    pub dust_silicate_fraction: Option<f64>,
    pub dust_graphite_fraction: Option<f64>,
    pub dust_kappa_silicate_uv: Option<f64>,
    pub dust_kappa_graphite_uv: Option<f64>,
    pub dust_h2_shielding_boost: Option<f64>,
}

pub fn apply_runtime_cli_overrides(
    cfg: &mut RunConfig,
    overrides: RuntimeCliOverrides,
) -> Result<(), CliError> {
    if overrides.sph {
        cfg.sph.enabled = true;
    }
    if let Some(v) = overrides.gas_fraction {
        cfg.sph.gas_fraction = v;
    }
    if let Some(v) = overrides.cooling {
        cfg.sph.enabled = true;
        cfg.sph.cooling = parse_cooling_kind(&v)?;
    }
    if overrides.feedback {
        cfg.sph.enabled = true;
        cfg.sph.feedback.enabled = true;
    }
    if let Some(v) = overrides.sf_model {
        cfg.sph.enabled = true;
        cfg.sph.feedback.enabled = true;
        cfg.sph.feedback.sf_model = parse_sf_model(&v)?;
    }
    if let Some(v) = overrides.feedback_mode {
        cfg.sph.enabled = true;
        cfg.sph.feedback.enabled = true;
        cfg.sph.feedback.feedback_mode = parse_feedback_mode(&v)?;
    }
    if overrides.winds {
        cfg.sph.enabled = true;
        cfg.sph.feedback.enabled = true;
        cfg.sph.feedback.wind.enabled = true;
    }
    if let Some(v) = overrides.wind_velocity {
        cfg.sph.enabled = true;
        cfg.sph.feedback.enabled = true;
        cfg.sph.feedback.wind.enabled = true;
        cfg.sph.feedback.wind.v_wind_km_s = v;
    }
    if overrides.agn {
        cfg.sph.agn.enabled = true;
    }
    if let Some(v) = overrides.agn_n_bh {
        cfg.sph.agn.enabled = true;
        cfg.sph.agn.n_agn_bh = v;
    }
    if let Some(v) = overrides.agn_m_seed {
        cfg.sph.agn.enabled = true;
        cfg.sph.agn.m_seed = v;
    }
    if let Some(v) = overrides.agn_eps_feedback {
        cfg.sph.agn.enabled = true;
        cfg.sph.agn.eps_feedback = v;
    }
    if overrides.agn_radio {
        cfg.sph.agn.enabled = true;
        cfg.sph.agn.eps_radio = cfg.sph.agn.eps_radio.max(0.0);
    }
    if let Some(v) = overrides.agn_f_edd_threshold {
        cfg.sph.agn.enabled = true;
        cfg.sph.agn.f_edd_threshold = v;
    }
    if let Some(v) = overrides.agn_spin {
        cfg.sph.agn.enabled = true;
        cfg.sph.agn.spin_enabled = true;
        cfg.sph.agn.initial_spin = v;
    }
    if overrides.agn_mergers {
        cfg.sph.agn.enabled = true;
        cfg.sph.agn.mergers_enabled = true;
    }
    if overrides.pbh_seeding {
        cfg.sph.agn.enabled = true;
        cfg.sph.agn.pbh_seeding_enabled = true;
    }
    if let Some(v) = overrides.pbh_n_seeds {
        cfg.sph.agn.pbh_n_seeds = v;
    }
    if let Some(v) = overrides.pbh_m_seed {
        cfg.sph.agn.pbh_m_seed = v;
    }
    if let Some(v) = overrides.pbh_min_host_mass {
        cfg.sph.agn.pbh_min_host_mass = v;
    }
    if let Some(v) = overrides.pbh_seed {
        cfg.sph.agn.pbh_seed = v;
    }
    if let Some(v) = overrides.pbh_host_kind {
        cfg.sph.agn.pbh_host_kind = parse_pbh_host_kind(&v)?;
    }
    if overrides.cr {
        cfg.sph.enabled = true;
        cfg.sph.cr.enabled = true;
    }
    if let Some(v) = overrides.cr_kappa {
        cfg.sph.enabled = true;
        cfg.sph.cr.enabled = true;
        cfg.sph.cr.kappa_cr = v;
    }
    if overrides.cr_anisotropic {
        cfg.sph.enabled = true;
        cfg.sph.cr.enabled = true;
        cfg.sph.cr.anisotropic_diffusion = true;
    }
    if let Some(v) = overrides.cr_streaming {
        cfg.sph.enabled = true;
        cfg.sph.cr.enabled = true;
        cfg.sph.cr.streaming_coefficient = v;
    }
    if overrides.mhd {
        cfg.mhd.enabled = true;
    }
    if let Some(v) = overrides.bfield {
        cfg.mhd.enabled = true;
        cfg.mhd.b0_kind = parse_bfield_kind(&v)?;
    }
    if let Some(v) = overrides.b0x {
        cfg.mhd.enabled = true;
        cfg.mhd.b0_uniform[0] = v;
    }
    if let Some(v) = overrides.b0y {
        cfg.mhd.enabled = true;
        cfg.mhd.b0_uniform[1] = v;
    }
    if let Some(v) = overrides.b0z {
        cfg.mhd.enabled = true;
        cfg.mhd.b0_uniform[2] = v;
    }
    if overrides.turbulence {
        cfg.mhd.enabled = true;
        cfg.turbulence.enabled = true;
    }
    if let Some(v) = overrides.turb_amplitude {
        cfg.mhd.enabled = true;
        cfg.turbulence.enabled = true;
        cfg.turbulence.amplitude = v;
    }
    if overrides.two_fluid {
        cfg.two_fluid.enabled = true;
    }
    if overrides.ambipolar {
        cfg.mhd.enabled = true;
        cfg.mhd.ambipolar_diffusion_enabled = true;
    }
    if let Some(v) = overrides.ambipolar_eta {
        cfg.mhd.enabled = true;
        cfg.mhd.ambipolar_diffusion_enabled = true;
        cfg.mhd.ambipolar_eta = v;
    }
    if let Some(v) = overrides.ambipolar_ion_floor {
        cfg.mhd.enabled = true;
        cfg.mhd.ambipolar_diffusion_enabled = true;
        cfg.mhd.ambipolar_ion_floor = v;
    }
    if let Some(v) = overrides.ambipolar_dust_coupling {
        cfg.mhd.enabled = true;
        cfg.mhd.ambipolar_diffusion_enabled = true;
        cfg.mhd.ambipolar_dust_coupling = v;
    }
    if overrides.sidm {
        cfg.sidm.enabled = true;
    }
    if let Some(v) = overrides.sidm_sigma_m {
        cfg.sidm.enabled = true;
        cfg.sidm.sigma_m = v;
    }
    if overrides.fr {
        cfg.modified_gravity.enabled = true;
    }
    if let Some(v) = overrides.fr_f_r0 {
        cfg.modified_gravity.enabled = true;
        cfg.modified_gravity.f_r0 = v;
    }
    if let Some(v) = overrides.fr_n {
        cfg.modified_gravity.enabled = true;
        cfg.modified_gravity.n = v;
    }
    if overrides.fr_nonlinear_mesh {
        cfg.modified_gravity.enabled = true;
        cfg.modified_gravity.nonlinear_mesh = true;
    }
    if overrides.rt {
        cfg.rt.enabled = true;
    }
    if overrides.rt_multifrequency {
        cfg.rt.enabled = true;
        cfg.rt.multifrequency_enabled = true;
    }
    if overrides.reionization {
        cfg.rt.enabled = true;
        cfg.reionization.enabled = true;
    }
    if let Some(v) = overrides.dark_matter {
        cfg.dark_matter.enabled = true;
        cfg.dark_matter.model = parse_dark_matter_model(&v)?;
    }
    if let Some(v) = overrides.wdm_mass_kev {
        cfg.dark_matter.enabled = true;
        cfg.dark_matter.model = DarkMatterModel::Warm;
        cfg.dark_matter.m_wdm_kev = v;
    }
    if let Some(v) = overrides.fdm_mass_22 {
        cfg.dark_matter.enabled = true;
        cfg.dark_matter.model = DarkMatterModel::Fuzzy;
        cfg.dark_matter.m_fdm_22 = v;
    }
    if overrides.dust {
        cfg.sph.enabled = true;
        cfg.sph.dust.enabled = true;
    }
    if let Some(v) = overrides.dust_species {
        cfg.sph.enabled = true;
        cfg.sph.dust.enabled = true;
        cfg.sph.dust.species_model = parse_dust_species_model(&v)?;
    }
    if let Some(v) = overrides.dust_silicate_fraction {
        cfg.sph.enabled = true;
        cfg.sph.dust.enabled = true;
        cfg.sph.dust.silicate_fraction = v;
    }
    if let Some(v) = overrides.dust_graphite_fraction {
        cfg.sph.enabled = true;
        cfg.sph.dust.enabled = true;
        cfg.sph.dust.graphite_fraction = v;
    }
    if let Some(v) = overrides.dust_kappa_silicate_uv {
        cfg.sph.enabled = true;
        cfg.sph.dust.enabled = true;
        cfg.sph.dust.kappa_silicate_uv = v;
    }
    if let Some(v) = overrides.dust_kappa_graphite_uv {
        cfg.sph.enabled = true;
        cfg.sph.dust.enabled = true;
        cfg.sph.dust.kappa_graphite_uv = v;
    }
    if let Some(v) = overrides.dust_h2_shielding_boost {
        cfg.sph.enabled = true;
        cfg.sph.dust.enabled = true;
        cfg.sph.dust.h2_shielding_boost = v;
    }
    cfg.validate()?;
    Ok(())
}

pub fn run_with_runtime<F>(f: F) -> Result<(), CliError>
where
    F: for<'a> FnOnce(&'a dyn ParallelRuntime) -> Result<(), CliError>,
{
    #[cfg(feature = "mpi")]
    {
        let rt = gadget_ng_parallel::MpiRuntime::new();
        f(&rt)
    }
    #[cfg(not(feature = "mpi"))]
    {
        let rt = gadget_ng_parallel::SerialRuntime;
        f(&rt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::RunConfig;

    fn minimal_run_config() -> RunConfig {
        toml::from_str(
            r#"
[simulation]
dt = 0.01
num_steps = 2
softening = 0.05
particle_count = 8
box_size = 1.0
seed = 42

[initial_conditions]
kind = "lattice"
"#,
        )
        .expect("minimal RunConfig")
    }

    // ── parse_* helpers ───────────────────────────────────────────────────────

    #[test]
    fn parse_cooling_kind_known_aliases() {
        assert!(matches!(
            parse_cooling_kind("atomic_h_he"),
            Ok(CoolingKind::AtomicHHe)
        ));
        assert!(matches!(
            parse_cooling_kind("atomic"),
            Ok(CoolingKind::AtomicHHe)
        ));
        assert!(matches!(parse_cooling_kind("none"), Ok(CoolingKind::None)));
    }

    #[test]
    fn parse_cooling_kind_unknown_is_err() {
        assert!(parse_cooling_kind("flux_capacitor").is_err());
    }

    #[test]
    fn parse_sf_model_known() {
        assert!(matches!(
            parse_sf_model("density_law"),
            Ok(StarFormationModel::DensityLaw)
        ));
        assert!(matches!(
            parse_sf_model("pressure"),
            Ok(StarFormationModel::PressureLaw)
        ));
    }

    #[test]
    fn parse_feedback_mode_known() {
        assert!(matches!(
            parse_feedback_mode("kinetic"),
            Ok(StellarFeedbackMode::Kinetic)
        ));
        assert!(matches!(
            parse_feedback_mode("thermal"),
            Ok(StellarFeedbackMode::ThermalStochastic)
        ));
    }

    #[test]
    fn parse_bfield_kind_all_variants() {
        assert!(parse_bfield_kind("none").is_ok());
        assert!(parse_bfield_kind("uniform").is_ok());
        assert!(parse_bfield_kind("random").is_ok());
        assert!(parse_bfield_kind("spiral").is_ok());
        assert!(parse_bfield_kind("monopole").is_err());
    }

    #[test]
    fn parse_dark_matter_model_aliases() {
        assert!(matches!(
            parse_dark_matter_model("cdm"),
            Ok(DarkMatterModel::Cold)
        ));
        assert!(matches!(
            parse_dark_matter_model("fdm"),
            Ok(DarkMatterModel::Fuzzy)
        ));
        assert!(parse_dark_matter_model("unknown").is_err());
    }

    #[test]
    fn parse_dust_species_model_aliases() {
        assert!(matches!(
            parse_dust_species_model("single"),
            Ok(DustSpeciesModel::Single)
        ));
        assert!(matches!(
            parse_dust_species_model("colibre"),
            Ok(DustSpeciesModel::SilicateGraphite)
        ));
        assert!(parse_dust_species_model("bogus").is_err());
    }

    #[test]
    fn parse_pbh_host_kind_known() {
        assert!(parse_pbh_host_kind("dm").is_ok());
        assert!(parse_pbh_host_kind("star").is_ok());
        assert!(parse_pbh_host_kind("collisionless").is_ok());
        assert!(parse_pbh_host_kind("unknown").is_err());
    }

    // ── apply_runtime_cli_overrides ───────────────────────────────────────────

    fn default_overrides() -> RuntimeCliOverrides {
        RuntimeCliOverrides {
            pbh_seeding: false,
            pbh_n_seeds: None,
            pbh_m_seed: None,
            pbh_min_host_mass: None,
            pbh_seed: None,
            pbh_host_kind: None,
            sph: false,
            gas_fraction: None,
            cooling: None,
            feedback: false,
            sf_model: None,
            feedback_mode: None,
            winds: false,
            wind_velocity: None,
            agn: false,
            agn_n_bh: None,
            agn_m_seed: None,
            agn_eps_feedback: None,
            agn_radio: false,
            agn_f_edd_threshold: None,
            agn_spin: None,
            agn_mergers: false,
            cr: false,
            cr_kappa: None,
            cr_anisotropic: false,
            cr_streaming: None,
            mhd: false,
            bfield: None,
            b0x: None,
            b0y: None,
            b0z: None,
            turbulence: false,
            turb_amplitude: None,
            two_fluid: false,
            ambipolar: false,
            ambipolar_eta: None,
            ambipolar_ion_floor: None,
            ambipolar_dust_coupling: None,
            sidm: false,
            sidm_sigma_m: None,
            fr: false,
            fr_f_r0: None,
            fr_n: None,
            fr_nonlinear_mesh: false,
            rt: false,
            rt_multifrequency: false,
            reionization: false,
            dark_matter: None,
            wdm_mass_kev: None,
            fdm_mass_22: None,
            dust: false,
            dust_species: None,
            dust_silicate_fraction: None,
            dust_graphite_fraction: None,
            dust_kappa_silicate_uv: None,
            dust_kappa_graphite_uv: None,
            dust_h2_shielding_boost: None,
        }
    }

    #[test]
    fn apply_overrides_noop_is_ok() {
        let mut cfg = minimal_run_config();
        assert!(apply_runtime_cli_overrides(&mut cfg, default_overrides()).is_ok());
    }

    #[test]
    fn apply_overrides_sph_flag_enables_sph() {
        let mut cfg = minimal_run_config();
        let mut ov = default_overrides();
        ov.sph = true;
        apply_runtime_cli_overrides(&mut cfg, ov).expect("ok");
        assert!(cfg.sph.enabled);
    }

    #[test]
    fn apply_overrides_mhd_flag_enables_mhd() {
        let mut cfg = minimal_run_config();
        let mut ov = default_overrides();
        ov.mhd = true;
        apply_runtime_cli_overrides(&mut cfg, ov).expect("ok");
        assert!(cfg.mhd.enabled);
    }

    #[test]
    fn apply_overrides_cooling_enables_sph() {
        let mut cfg = minimal_run_config();
        let mut ov = default_overrides();
        ov.cooling = Some("atomic_h_he".into());
        apply_runtime_cli_overrides(&mut cfg, ov).expect("ok");
        assert!(cfg.sph.enabled);
        assert!(matches!(cfg.sph.cooling, CoolingKind::AtomicHHe));
    }

    #[test]
    fn apply_overrides_invalid_cooling_returns_err() {
        let mut cfg = minimal_run_config();
        let mut ov = default_overrides();
        ov.cooling = Some("flux_capacitor".into());
        assert!(apply_runtime_cli_overrides(&mut cfg, ov).is_err());
    }

    #[test]
    fn apply_overrides_sidm_sigma_m() {
        let mut cfg = minimal_run_config();
        let mut ov = default_overrides();
        ov.sidm_sigma_m = Some(1.5);
        apply_runtime_cli_overrides(&mut cfg, ov).expect("ok");
        assert!(cfg.sidm.enabled);
        assert!((cfg.sidm.sigma_m - 1.5).abs() < 1e-12);
    }

    #[test]
    fn run_with_runtime_serial_executes_closure() {
        let mut executed = false;
        run_with_runtime(|_rt| {
            executed = true;
            Ok(())
        })
        .expect("run_with_runtime ok");
        assert!(executed);
    }
}
