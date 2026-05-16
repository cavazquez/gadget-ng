mod analyze_cmd;
mod config_load;
mod engine;
mod error;
mod fisher_cmd;
mod insitu;
mod mah_cmd;
mod merge_tree_cmd;

use clap::{Parser, Subcommand, ValueEnum};
use error::CliError;
use gadget_ng_core::{
    BFieldKind, CoolingKind, DarkMatterModel, DustSpeciesModel, PbhHostKind, RunConfig,
    StarFormationModel, StellarFeedbackMode,
};
use gadget_ng_parallel::ParallelRuntime;
#[cfg(not(feature = "mpi"))]
use gadget_ng_parallel::SerialRuntime;
use std::path::PathBuf;

/// Modelo de P(k) para el subcomando `fisher` (lineal vs Halofit).
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum FisherPkModel {
    /// P(k) lineal (transfer Eisenstein–Hu y factor de crecimiento).
    Linear,
    /// P(k) no lineal con Halofit.
    #[default]
    Nonlinear,
}

#[derive(Parser)]
#[command(
    name = "gadget-ng",
    version,
    about = "Simulador cosmológico (N-body, SPH, MHD, RT, …): configuración, integración temporal, snapshots y análisis.",
    after_help = "Comandos por grupo:\n  Preparación y corrida: config, stepping, snapshot\n  Post-proceso y viz: visualize, analyze, merge-tree, mah\n  Forecasting: fisher\n\nLa corrida principal y sus flags están en `gadget-ng stepping --help`; la física base se define en el TOML y variables `GADGET_NG_*`."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
#[expect(clippy::large_enum_variant)]
enum Commands {
    /// Valida y muestra la configuración efectiva (TOML + env `GADGET_NG_`).
    #[command(display_order = 0)]
    Config {
        #[arg(long, help = "Ruta al archivo TOML de experimento")]
        config: PathBuf,
    },
    /// Integración leapfrog KDK (`num_steps` en TOML); flags opcionales aquí sobrescriben el TOML para esta corrida.
    #[command(display_order = 1)]
    Stepping {
        #[arg(long, help = "Ruta al archivo TOML de experimento")]
        config: PathBuf,
        #[arg(long, help = "Directorio de salida (diagnósticos y snapshot opcional)")]
        out: PathBuf,
        #[arg(long, help = "Escribir snapshot final bajo `<out>/snapshot_final`")]
        snapshot: bool,
        /// Reanudar desde el checkpoint guardado en `<RESUME>/checkpoint/`.
        #[arg(
            long,
            help = "Directorio de salida de una corrida anterior para reanudar"
        )]
        resume: Option<PathBuf>,
        /// Guardar imagen PPM del estado de partículas cada N pasos (0 = desactivado).
        /// Los archivos se escriben como `<out>/snap_NNNNNN.ppm` o `.png`.
        #[arg(long, default_value_t = 0)]
        vis_snapshot: u64,
        /// Proyección para el render de snapshot: `xy`, `xz`, `yz`.
        #[arg(long, default_value = "xy")]
        vis_proj: String,
        /// Modo de renderizado: `points` (puntos blancos) o `density` (mapa de densidad Viridis).
        #[arg(long, default_value = "points")]
        vis_mode: String,
        /// Formato de salida: `ppm` o `png`.
        #[arg(long, default_value = "ppm")]
        vis_format: String,
        /// Activar PBH seeding en `[sph.agn]` para esta corrida.
        #[arg(long)]
        pbh_seeding: bool,
        /// Número de semillas PBH a crear.
        #[arg(long)]
        pbh_n_seeds: Option<usize>,
        /// Masa de cada semilla PBH [M_sol/h].
        #[arg(long)]
        pbh_m_seed: Option<f64>,
        /// Masa mínima de partícula host para alojar una PBH.
        #[arg(long)]
        pbh_min_host_mass: Option<f64>,
        /// Semilla determinista para seleccionar hosts PBH.
        #[arg(long)]
        pbh_seed: Option<u64>,
        /// Tipo de host PBH: `dark_matter`, `star` o `collisionless`.
        #[arg(long)]
        pbh_host_kind: Option<String>,
        /// Activar SPH para esta corrida.
        #[arg(long)]
        sph: bool,
        /// Sobrescribir fracción inicial de gas SPH.
        #[arg(long)]
        gas_fraction: Option<f64>,
        /// Cooling SPH: none, atomic_h_he, metal_cooling, metal_tabular, uv_background.
        #[arg(long)]
        cooling: Option<String>,
        /// Activar feedback estelar.
        #[arg(long)]
        feedback: bool,
        /// Ley SF: density_law o pressure_law.
        #[arg(long)]
        sf_model: Option<String>,
        /// Modo feedback: kinetic o thermal_stochastic.
        #[arg(long)]
        feedback_mode: Option<String>,
        /// Activar vientos galácticos.
        #[arg(long)]
        winds: bool,
        /// Velocidad de viento galáctico [km/s].
        #[arg(long)]
        wind_velocity: Option<f64>,
        /// Activar AGN.
        #[arg(long)]
        agn: bool,
        /// Número de BH AGN clásicos.
        #[arg(long)]
        agn_n_bh: Option<usize>,
        /// Masa semilla AGN clásica [M_sol/h].
        #[arg(long)]
        agn_m_seed: Option<f64>,
        /// Eficiencia de feedback AGN.
        #[arg(long)]
        agn_eps_feedback: Option<f64>,
        /// Activar modo radio AGN.
        #[arg(long)]
        agn_radio: bool,
        /// Umbral Eddington modo quasar/radio.
        #[arg(long)]
        agn_f_edd_threshold: Option<f64>,
        /// Spin inicial de semillas BH.
        #[arg(long)]
        agn_spin: Option<f64>,
        /// Activar mergers de BH.
        #[arg(long)]
        agn_mergers: bool,
        /// Activar rayos cósmicos.
        #[arg(long)]
        cr: bool,
        /// Coeficiente de difusión CR.
        #[arg(long)]
        cr_kappa: Option<f64>,
        /// Activar difusión CR anisótropa (requiere MHD).
        #[arg(long)]
        cr_anisotropic: bool,
        /// Coeficiente de streaming CR paralelo al campo B (requiere MHD).
        #[arg(long)]
        cr_streaming: Option<f64>,
        /// Activar MHD.
        #[arg(long)]
        mhd: bool,
        /// Tipo de campo B inicial: none, uniform, random, spiral.
        #[arg(long)]
        bfield: Option<String>,
        /// Amplitud uniforme de B en x.
        #[arg(long)]
        b0x: Option<f64>,
        /// Amplitud uniforme de B en y.
        #[arg(long)]
        b0y: Option<f64>,
        /// Amplitud uniforme de B en z.
        #[arg(long)]
        b0z: Option<f64>,
        /// Activar turbulencia MHD OU.
        #[arg(long)]
        turbulence: bool,
        /// Amplitud de forzado turbulento.
        #[arg(long)]
        turb_amplitude: Option<f64>,
        /// Activar plasma de dos fluidos.
        #[arg(long)]
        two_fluid: bool,
        /// Activar difusión ambipolar no-ideal (requiere MHD).
        #[arg(long)]
        ambipolar: bool,
        /// Coeficiente base de difusión ambipolar.
        #[arg(long)]
        ambipolar_eta: Option<f64>,
        /// Piso de ionización para difusión ambipolar.
        #[arg(long)]
        ambipolar_ion_floor: Option<f64>,
        /// Acoplamiento polvo-ionización para difusión ambipolar.
        #[arg(long)]
        ambipolar_dust_coupling: Option<f64>,
        /// Activar SIDM.
        #[arg(long)]
        sidm: bool,
        /// Sección eficaz SIDM por masa.
        #[arg(long)]
        sidm_sigma_m: Option<f64>,
        /// Activar f(R) Hu-Sawicki.
        #[arg(long)]
        fr: bool,
        /// |f_R0| para f(R).
        #[arg(long)]
        fr_f_r0: Option<f64>,
        /// Índice n para f(R).
        #[arg(long)]
        fr_n: Option<f64>,
        /// Activar screening no lineal en malla f(R).
        #[arg(long)]
        fr_nonlinear_mesh: bool,
        /// Activar RT M1.
        #[arg(long)]
        rt: bool,
        /// Activar RT multifrecuencia.
        #[arg(long)]
        rt_multifrequency: bool,
        /// Activar reionización.
        #[arg(long)]
        reionization: bool,
        /// Activar WDM/FDM en ICs.
        #[arg(long)]
        dark_matter: Option<String>,
        /// Masa WDM [keV].
        #[arg(long)]
        wdm_mass_kev: Option<f64>,
        /// Masa FDM en 1e-22 eV.
        #[arg(long)]
        fdm_mass_22: Option<f64>,
        /// Activar polvo SPH.
        #[arg(long)]
        dust: bool,
        /// Modelo de especies de polvo: single o silicate_graphite.
        #[arg(long)]
        dust_species: Option<String>,
        /// Fracción de masa de polvo en silicatos.
        #[arg(long)]
        dust_silicate_fraction: Option<f64>,
        /// Fracción de masa de polvo en grafitos.
        #[arg(long)]
        dust_graphite_fraction: Option<f64>,
        /// Opacidad UV de silicatos [cm²/g].
        #[arg(long)]
        dust_kappa_silicate_uv: Option<f64>,
        /// Opacidad UV de grafitos [cm²/g].
        #[arg(long)]
        dust_kappa_graphite_uv: Option<f64>,
        /// Boost máximo de shielding H2 por polvo activo.
        #[arg(long)]
        dust_h2_shielding_boost: Option<f64>,
    },
    /// Escribe un snapshot del estado inicial (IC) resuelto.
    #[command(display_order = 2)]
    Snapshot {
        #[arg(long, help = "Ruta al archivo TOML de experimento")]
        config: PathBuf,
        #[arg(long, help = "Directorio de salida del snapshot")]
        out: PathBuf,
    },
    /// Renderiza un snapshot (JSONL) a imagen PNG con proyección y coloración configurables.
    ///
    /// Lee posiciones y velocidades del directorio de snapshot y escribe un PNG.
    ///
    /// Ejemplo:
    ///   gadget-ng visualize --snapshot out/snapshot_final --output frame.png --color velocity
    #[command(display_order = 10)]
    Visualize {
        /// Directorio del snapshot a renderizar.
        #[arg(long)]
        snapshot: PathBuf,
        /// Ruta del archivo PNG de salida.
        #[arg(long)]
        output: PathBuf,
        /// Ancho en píxeles.
        #[arg(long, default_value_t = 1024)]
        width: u32,
        /// Alto en píxeles.
        #[arg(long, default_value_t = 1024)]
        height: u32,
        /// Proyección: `xy`, `xz`, `yz`.
        #[arg(long, default_value = "xy")]
        projection: String,
        /// Coloración: `white`, `velocity`.
        #[arg(long, default_value = "velocity")]
        color: String,
    },
    /// Análisis completo de un snapshot: FoF + P(k) + ξ(r) + c(M).
    ///
    /// Lee posiciones y velocidades del directorio de snapshot (JSONL), ejecuta:
    /// - Friends-of-Friends (FoF) con parámetro b configurable.
    /// - Espectro de potencia P(k) via CIC + FFT 3D.
    /// - Función de correlación de 2 puntos ξ(r) via transformada de Hankel.
    /// - Concentración c(M) NFW para halos con N ≥ nfw-min-part.
    ///
    /// Escribe el JSON principal en la ruta `--output` (por defecto `results.json` en el cwd).
    ///
    /// Ejemplo:
    ///   gadget-ng analyze --snapshot out/snap --output out/snap/results.json --fof-b 0.2 --xi-bins 20
    #[command(display_order = 11)]
    Analyze {
        /// Directorio del snapshot a analizar.
        #[arg(long)]
        snapshot: PathBuf,
        /// Archivo JSON de salida (por defecto: `results.json`).
        #[arg(long, default_value = "results.json")]
        output: PathBuf,
        /// Parámetro de enlace FoF (fracción de la separación media).
        #[arg(long, default_value_t = 0.2)]
        fof_b: f64,
        /// Número mínimo de partículas para halo FoF.
        #[arg(long, default_value_t = 8)]
        min_particles: usize,
        /// Tamaño del grid para P(k) (por lado; total = mesh³).
        #[arg(long, default_value_t = 64)]
        pk_mesh: usize,
        /// Número de bins logarítmicos para ξ(r).
        #[arg(long, default_value_t = 20)]
        xi_bins: usize,
        /// Número mínimo de partículas para ajuste NFW en c(M).
        #[arg(long, default_value_t = 50)]
        nfw_min_part: usize,
        /// Tamaño físico de la caja en Mpc/h (para unidades de c(M) y ξ(r)).
        #[arg(long)]
        box_size_mpc_h: Option<f64>,
        /// Ejecutar SUBFIND sobre cada halo para identificar subestructura (Phase 68).
        #[arg(long, default_value_t = false)]
        subfind: bool,
        /// Número mínimo de partículas de halo para ejecutar SUBFIND (default: 50).
        #[arg(long, default_value_t = 50)]
        subfind_min_particles: usize,
        /// Escribir catálogo de halos en HDF5/JSONL además de results.json (Phase 82d).
        #[arg(long, default_value_t = false)]
        hdf5_catalog: bool,
        /// Calcular estadísticas 21cm (δT_b, P(k)₂₁cm) → analyze/cm21_output.json [Phase 104]
        #[arg(long, default_value_t = false)]
        cm21: bool,
        /// Calcular perfil de temperatura IGM T(z) → analyze/igm_temp.json [Phase 104]
        #[arg(long, default_value_t = false)]
        igm_temp: bool,
        /// Calcular estadísticas de BH AGN → analyze/agn_stats.json [Phase 104]
        #[arg(long, default_value_t = false)]
        agn_stats: bool,
        /// Calcular fracción de ionización x_HII media → analyze/eor_state.json [Phase 104]
        #[arg(long, default_value_t = false)]
        eor_state: bool,
        /// Calcular luminosidad y colores (B-V, g-r) → analyze/luminosity.json [Phase 118]
        #[arg(long, default_value_t = false)]
        luminosity: bool,
        /// Calcular luminosidad X bremsstrahlung → analyze/xray.json [AP-17]
        #[arg(long, default_value_t = false)]
        xray: bool,
    },
    /// Construye el merger tree conectando catálogos FoF de snapshots consecutivos.
    ///
    /// Sigue partículas entre snapshots para identificar progenitores y mergers.
    /// Escribe el JSON en la ruta `--out` (por defecto `merger_tree.json` en el cwd).
    ///
    /// Ejemplo:
    ///   gadget-ng merge-tree \
    ///     --catalogs "runs/cosmo/halos_000.jsonl,runs/cosmo/halos_001.jsonl" \
    ///     --snapshots "runs/cosmo/snap_000,runs/cosmo/snap_001" \
    ///     --out runs/cosmo/merger_tree.json
    #[command(display_order = 12)]
    MergeTree {
        /// Lista de directorios de snapshot separados por coma (orden cronológico).
        #[arg(long)]
        snapshots: String,
        /// Lista de archivos de catálogo JSONL separados por coma (mismo orden).
        #[arg(long)]
        catalogs: String,
        /// Archivo JSON de salida.
        #[arg(long, default_value = "merger_tree.json")]
        out: PathBuf,
        /// Fracción mínima de partículas compartidas para registrar un progenitor.
        #[arg(long, default_value_t = 0.1)]
        min_shared: f64,
    },
    /// Extrae la Historia de Acreción de Masa (MAH) a lo largo de la rama principal.
    ///
    /// Lee un merger tree JSON generado por `merge-tree` y extrae la MAH del halo raíz,
    /// comparándola con el ajuste analítico de McBride+2009.
    ///
    /// Ejemplo:
    ///   gadget-ng mah \
    ///     --merger-tree runs/cosmo/merger_tree.json \
    ///     --redshifts "49,10,5,2,1,0.5,0" \
    ///     --root-id 0 \
    ///     --out runs/cosmo/mah.json
    #[command(display_order = 13)]
    Mah {
        /// Ruta al archivo JSON del merger tree (salida de `merge-tree`).
        #[arg(long)]
        merger_tree: PathBuf,
        /// Redshifts de cada snapshot separados por coma (orden cronológico, del más antiguo z_max al más reciente z=0).
        #[arg(long)]
        redshifts: String,
        /// ID del halo raíz en el snapshot más reciente.
        #[arg(long, default_value_t = 0)]
        root_id: u64,
        /// Parámetro α del ajuste McBride+2009 (default: 1.0).
        #[arg(long, default_value_t = 1.0)]
        alpha: f64,
        /// Parámetro β del ajuste McBride+2009 (default: 0.0).
        #[arg(long, default_value_t = 0.0)]
        beta: f64,
        /// Archivo JSON de salida.
        #[arg(long, default_value = "mah.json")]
        out: PathBuf,
    },
    /// Matriz de Fisher para P(k) cosmológico (diferencias centrales en parámetros; fase 173).
    ///
    /// Aproxima ∂P(k,z)/∂θ y arma F_ij con la covarianza del survey en (Mpc/h)³.
    ///
    /// Ejemplo:
    ///   gadget-ng fisher \
    ///     --omega-m 0.315 --sigma8 0.8111 \
    ///     --survey-volume 1e9 \
    ///     --pk-model nonlinear \
    ///     --out runs/fisher/fisher_output.json
    #[command(display_order = 20)]
    Fisher {
        /// Ω_m: fracción de densidad de materia total (Planck 2018 por defecto: 0.315).
        #[arg(long, default_value_t = 0.315)]
        omega_m: f64,
        /// Ω_b: fracción de densidad bariónica (por defecto: 0.049).
        #[arg(long, default_value_t = 0.049)]
        omega_b: f64,
        /// h: parámetro de Hubble adimensional H₀/(100 km/s/Mpc) (por defecto: 0.674).
        #[arg(long, default_value_t = 0.674)]
        h: f64,
        /// n_s: índice espectral escalar (por defecto: 0.965).
        #[arg(long, default_value_t = 0.965)]
        n_s: f64,
        /// σ₈: amplitud de fluctuaciones de materia a 8 Mpc/h (por defecto: 0.8111).
        #[arg(long, default_value_t = 0.8111)]
        sigma8: f64,
        /// w₀: estado de ecuación energía oscura CPL (por defecto −1, ΛCDM).
        #[arg(long, default_value_t = -1.0)]
        w0: f64,
        /// wₐ: evolución CPL de la energía oscura (por defecto 0, ΛCDM).
        #[arg(long, default_value_t = 0.0)]
        wa: f64,
        /// Σm_ν en eV (por defecto 0.06, jerarquía mínima).
        #[arg(long, default_value_t = 0.06)]
        m_nu_ev: f64,
        /// Paso fraccional para diferencias centrales (por defecto 0.01 = 1 %).
        #[arg(long, default_value_t = 0.01)]
        step_frac: f64,
        /// Volumen del survey en (Mpc/h)³ (por defecto 1e9 ≈ 1 Gpc³).
        #[arg(long, default_value_t = 1.0e9)]
        survey_volume: f64,
        /// Modelo de P(k): lineal (EH + crecimiento) o `nonlinear` (Halofit).
        #[arg(long, value_enum, default_value_t = FisherPkModel::Nonlinear)]
        pk_model: FisherPkModel,
        /// Ruta del JSON de salida.
        #[arg(long, default_value = "fisher_output.json")]
        out: PathBuf,
    },
}

fn parse_pbh_host_kind(value: &str) -> Result<PbhHostKind, CliError> {
    match value {
        "dark_matter" | "dm" => Ok(PbhHostKind::DarkMatter),
        "star" | "stars" => Ok(PbhHostKind::Star),
        "collisionless" | "all_collisionless" => Ok(PbhHostKind::Collisionless),
        other => Err(CliError::InvalidConfig(format!(
            "pbh_host_kind inválido: {other}; usar dark_matter, star o collisionless"
        ))),
    }
}

fn parse_cooling_kind(value: &str) -> Result<CoolingKind, CliError> {
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

fn parse_sf_model(value: &str) -> Result<StarFormationModel, CliError> {
    match value {
        "density_law" | "density" => Ok(StarFormationModel::DensityLaw),
        "pressure_law" | "pressure" => Ok(StarFormationModel::PressureLaw),
        other => Err(CliError::InvalidConfig(format!(
            "sf_model inválido: {other}; usar density_law o pressure_law"
        ))),
    }
}

fn parse_feedback_mode(value: &str) -> Result<StellarFeedbackMode, CliError> {
    match value {
        "kinetic" => Ok(StellarFeedbackMode::Kinetic),
        "thermal_stochastic" | "thermal" => Ok(StellarFeedbackMode::ThermalStochastic),
        other => Err(CliError::InvalidConfig(format!(
            "feedback_mode inválido: {other}; usar kinetic o thermal_stochastic"
        ))),
    }
}

fn parse_bfield_kind(value: &str) -> Result<BFieldKind, CliError> {
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

fn parse_dark_matter_model(value: &str) -> Result<DarkMatterModel, CliError> {
    match value {
        "cold" | "cdm" => Ok(DarkMatterModel::Cold),
        "warm" | "wdm" => Ok(DarkMatterModel::Warm),
        "fuzzy" | "fdm" => Ok(DarkMatterModel::Fuzzy),
        other => Err(CliError::InvalidConfig(format!(
            "dark_matter inválido: {other}; usar cold, warm o fuzzy"
        ))),
    }
}

fn parse_dust_species_model(value: &str) -> Result<DustSpeciesModel, CliError> {
    match value {
        "single" => Ok(DustSpeciesModel::Single),
        "silicate_graphite" | "active" | "colibre" => Ok(DustSpeciesModel::SilicateGraphite),
        other => Err(CliError::InvalidConfig(format!(
            "dust_species inválido: {other}; usar single o silicate_graphite"
        ))),
    }
}

#[expect(clippy::struct_excessive_bools)]
struct RuntimeCliOverrides {
    pbh_seeding: bool,
    pbh_n_seeds: Option<usize>,
    pbh_m_seed: Option<f64>,
    pbh_min_host_mass: Option<f64>,
    pbh_seed: Option<u64>,
    pbh_host_kind: Option<String>,
    sph: bool,
    gas_fraction: Option<f64>,
    cooling: Option<String>,
    feedback: bool,
    sf_model: Option<String>,
    feedback_mode: Option<String>,
    winds: bool,
    wind_velocity: Option<f64>,
    agn: bool,
    agn_n_bh: Option<usize>,
    agn_m_seed: Option<f64>,
    agn_eps_feedback: Option<f64>,
    agn_radio: bool,
    agn_f_edd_threshold: Option<f64>,
    agn_spin: Option<f64>,
    agn_mergers: bool,
    cr: bool,
    cr_kappa: Option<f64>,
    cr_anisotropic: bool,
    cr_streaming: Option<f64>,
    mhd: bool,
    bfield: Option<String>,
    b0x: Option<f64>,
    b0y: Option<f64>,
    b0z: Option<f64>,
    turbulence: bool,
    turb_amplitude: Option<f64>,
    two_fluid: bool,
    ambipolar: bool,
    ambipolar_eta: Option<f64>,
    ambipolar_ion_floor: Option<f64>,
    ambipolar_dust_coupling: Option<f64>,
    sidm: bool,
    sidm_sigma_m: Option<f64>,
    fr: bool,
    fr_f_r0: Option<f64>,
    fr_n: Option<f64>,
    fr_nonlinear_mesh: bool,
    rt: bool,
    rt_multifrequency: bool,
    reionization: bool,
    dark_matter: Option<String>,
    wdm_mass_kev: Option<f64>,
    fdm_mass_22: Option<f64>,
    dust: bool,
    dust_species: Option<String>,
    dust_silicate_fraction: Option<f64>,
    dust_graphite_fraction: Option<f64>,
    dust_kappa_silicate_uv: Option<f64>,
    dust_kappa_graphite_uv: Option<f64>,
    dust_h2_shielding_boost: Option<f64>,
}

fn apply_runtime_cli_overrides(
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

fn run_with_runtime<F>(f: F) -> Result<(), CliError>
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
        let rt = SerialRuntime;
        f(&rt)
    }
}

fn main() -> Result<(), CliError> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Config { config } => engine::cmd_config_print(&config)?,
        Commands::Stepping {
            config,
            out,
            snapshot,
            resume,
            vis_snapshot,
            vis_proj,
            vis_mode,
            vis_format,
            pbh_seeding,
            pbh_n_seeds,
            pbh_m_seed,
            pbh_min_host_mass,
            pbh_seed,
            pbh_host_kind,
            sph,
            gas_fraction,
            cooling,
            feedback,
            sf_model,
            feedback_mode,
            winds,
            wind_velocity,
            agn,
            agn_n_bh,
            agn_m_seed,
            agn_eps_feedback,
            agn_radio,
            agn_f_edd_threshold,
            agn_spin,
            agn_mergers,
            cr,
            cr_kappa,
            cr_anisotropic,
            cr_streaming,
            mhd,
            bfield,
            b0x,
            b0y,
            b0z,
            turbulence,
            turb_amplitude,
            two_fluid,
            ambipolar,
            ambipolar_eta,
            ambipolar_ion_floor,
            ambipolar_dust_coupling,
            sidm,
            sidm_sigma_m,
            fr,
            fr_f_r0,
            fr_n,
            fr_nonlinear_mesh,
            rt,
            rt_multifrequency,
            reionization,
            dark_matter,
            wdm_mass_kev,
            fdm_mass_22,
            dust,
            dust_species,
            dust_silicate_fraction,
            dust_graphite_fraction,
            dust_kappa_silicate_uv,
            dust_kappa_graphite_uv,
            dust_h2_shielding_boost,
        } => {
            let mut cfg = config_load::load_run_config(&config)?;
            apply_runtime_cli_overrides(
                &mut cfg,
                RuntimeCliOverrides {
                    pbh_seeding,
                    pbh_n_seeds,
                    pbh_m_seed,
                    pbh_min_host_mass,
                    pbh_seed,
                    pbh_host_kind,
                    sph,
                    gas_fraction,
                    cooling,
                    feedback,
                    sf_model,
                    feedback_mode,
                    winds,
                    wind_velocity,
                    agn,
                    agn_n_bh,
                    agn_m_seed,
                    agn_eps_feedback,
                    agn_radio,
                    agn_f_edd_threshold,
                    agn_spin,
                    agn_mergers,
                    cr,
                    cr_kappa,
                    cr_anisotropic,
                    cr_streaming,
                    mhd,
                    bfield,
                    b0x,
                    b0y,
                    b0z,
                    turbulence,
                    turb_amplitude,
                    two_fluid,
                    ambipolar,
                    ambipolar_eta,
                    ambipolar_ion_floor,
                    ambipolar_dust_coupling,
                    sidm,
                    sidm_sigma_m,
                    fr,
                    fr_f_r0,
                    fr_n,
                    fr_nonlinear_mesh,
                    rt,
                    rt_multifrequency,
                    reionization,
                    dark_matter,
                    wdm_mass_kev,
                    fdm_mass_22,
                    dust,
                    dust_species,
                    dust_silicate_fraction,
                    dust_graphite_fraction,
                    dust_kappa_silicate_uv,
                    dust_kappa_graphite_uv,
                    dust_h2_shielding_boost,
                },
            )?;
            run_with_runtime(|rt| {
                engine::run_stepping(rt, &cfg, &out, snapshot, resume.as_deref())
            })?;
            // Si vis_snapshot > 0, renderizar el snapshot final.
            if vis_snapshot > 0 {
                engine::render_snapshot_visualization(
                    &out,
                    vis_snapshot,
                    &vis_proj,
                    &vis_mode,
                    &vis_format,
                );
            }
        }
        Commands::Snapshot { config, out } => {
            let cfg = config_load::load_run_config(&config)?;
            run_with_runtime(|rt| engine::run_snapshot(rt, &cfg, &out))?;
        }
        Commands::Visualize {
            snapshot,
            output,
            width,
            height,
            projection,
            color,
        } => {
            engine::run_visualize(&snapshot, &output, width, height, &projection, &color)?;
        }
        Commands::Analyze {
            snapshot,
            output,
            fof_b,
            min_particles,
            pk_mesh,
            xi_bins,
            nfw_min_part,
            box_size_mpc_h,
            subfind,
            subfind_min_particles,
            hdf5_catalog,
            cm21,
            igm_temp,
            agn_stats,
            eor_state,
            luminosity,
            xray,
        } => {
            let params = analyze_cmd::AnalyzeParams {
                snapshot_dir: &snapshot,
                out_path: &output,
                fof_b,
                min_particles,
                pk_mesh,
                xi_bins,
                nfw_min_part,
                cosmology: None,
                box_size_mpc_h,
                subfind,
                subfind_min_particles,
                hdf5_catalog,
                cm21,
                igm_temp,
                agn_stats,
                eor_state,
                luminosity,
                xray,
                cuda_analysis: false,
            };
            analyze_cmd::run_analyze(&params)?;
        }
        Commands::MergeTree {
            snapshots,
            catalogs,
            out,
            min_shared,
        } => {
            let snap_dirs: Vec<std::path::PathBuf> = snapshots
                .split(',')
                .map(|s| std::path::PathBuf::from(s.trim()))
                .collect();
            let catalog_paths: Vec<std::path::PathBuf> = catalogs
                .split(',')
                .map(|s| std::path::PathBuf::from(s.trim()))
                .collect();
            merge_tree_cmd::run_merge_tree(&snap_dirs, &catalog_paths, &out, min_shared)?;
        }
        Commands::Mah {
            merger_tree,
            redshifts,
            root_id,
            alpha,
            beta,
            out,
        } => {
            let zs: Vec<f64> = redshifts
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            mah_cmd::run_mah(&merger_tree, &zs, root_id, alpha, beta, &out)?;
        }
        Commands::Fisher {
            omega_m,
            omega_b,
            h,
            n_s,
            sigma8,
            w0,
            wa,
            m_nu_ev,
            step_frac,
            survey_volume,
            pk_model,
            out,
        } => {
            let use_nonlinear = matches!(pk_model, FisherPkModel::Nonlinear);
            fisher_cmd::run_fisher(
                omega_m,
                omega_b,
                h,
                n_s,
                sigma8,
                w0,
                wa,
                m_nu_ev,
                step_frac,
                survey_volume,
                use_nonlinear,
                &out,
            )?;
        }
    }
    Ok(())
}
