mod analyze_cmd;
mod config_load;
mod engine;
mod error;

use clap::{Parser, Subcommand};
use error::CliError;
use gadget_ng_parallel::ParallelRuntime;
#[cfg(not(feature = "mpi"))]
use gadget_ng_parallel::SerialRuntime;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "gadget-ng",
    version,
    about = "Simulación N-body (MVP): configuración, pasos temporales y snapshots."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Valida y muestra la configuración efectiva (TOML + env `GADGET_NG_`).
    Config {
        #[arg(long, help = "Ruta al archivo TOML de experimento")]
        config: PathBuf,
    },
    /// Ejecuta integración leapfrog KDK durante `num_steps` pasos.
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
        /// Los archivos se escriben como `<out>/snap_NNNNNN.ppm`.
        #[arg(long, default_value_t = 0)]
        vis_snapshot: u64,
    },
    /// Escribe un snapshot del estado inicial (IC) resuelto.
    Snapshot {
        #[arg(long, help = "Ruta al archivo TOML de experimento")]
        config: PathBuf,
        #[arg(long, help = "Directorio de salida del snapshot")]
        out: PathBuf,
    },
    /// Renderiza un snapshot de partículas a imagen PNG.
    ///
    /// Lee las posiciones y velocidades del directorio de snapshot (JSONL)
    /// y genera un PNG con el campo de partículas proyectado.
    ///
    /// Ejemplo:
    ///   gadget-ng visualize --snapshot out/snapshot_final --output frame.png --color velocity
    Visualize {
        /// Directorio del snapshot a renderizar.
        #[arg(long)]
        snapshot: PathBuf,
        /// Archivo PNG de salida.
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
    /// Escribe `results.json` en el directorio de salida.
    ///
    /// Ejemplo:
    ///   gadget-ng analyze --snapshot out/snap --out analysis/ --fof-b 0.2 --xi-bins 20
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
    },
    /// Analiza un snapshot: Friends-of-Friends (halos) + espectro de potencia P(k).
    ///
    /// Escribe catálogos JSONL en `<out>/halos.jsonl` y `<out>/power_spectrum.jsonl`.
    ///
    /// Ejemplo:
    ///   gadget-ng analyse --snapshot out/snapshot_final --out analysis/
    Analyse {
        /// Directorio del snapshot a analizar.
        #[arg(long)]
        snapshot: PathBuf,
        /// Directorio de salida para catálogos.
        #[arg(long)]
        out: PathBuf,
        /// Longitud de enlace FoF (en fracción de la separación media entre partículas).
        #[arg(long, default_value_t = 0.2)]
        linking_length: f64,
        /// Número mínimo de partículas por halo.
        #[arg(long, default_value_t = 8)]
        min_particles: usize,
        /// Tamaño del grid para P(k) (por lado; total = mesh³).
        #[arg(long, default_value_t = 64)]
        pk_mesh: usize,
    },
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
        } => {
            let cfg = config_load::load_run_config(&config)?;
            run_with_runtime(|rt| {
                engine::run_stepping(rt, &cfg, &out, snapshot, resume.as_deref())
            })?;
            // Si vis_snapshot > 0, renderizar el snapshot final como PPM.
            if vis_snapshot > 0 {
                let snap_dir = out.join("snapshot_final");
                if snap_dir.exists() {
                    use gadget_ng_core::SnapshotFormat;
                    let data =
                        gadget_ng_io::read_snapshot_formatted(SnapshotFormat::Jsonl, &snap_dir);
                    if let Ok(data) = data {
                        let positions: Vec<gadget_ng_core::Vec3> =
                            data.particles.iter().map(|p| p.position).collect();
                        let pixels =
                            gadget_ng_vis::render_ppm(&positions, data.box_size, 1024, 1024);
                        let ppm_path = out.join("snapshot_final.ppm");
                        if let Err(e) = gadget_ng_vis::write_ppm(&ppm_path, &pixels, 1024, 1024) {
                            eprintln!("[vis] Error escribiendo PPM: {e}");
                        } else {
                            eprintln!("[vis] PPM escrito en {:?}", ppm_path);
                        }
                    }
                }
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
            };
            analyze_cmd::run_analyze(&params)?;
        }
        Commands::Analyse {
            snapshot,
            out,
            linking_length,
            min_particles,
            pk_mesh,
        } => {
            engine::run_analyse(&snapshot, &out, linking_length, min_particles, pk_mesh)?;
        }
    }
    Ok(())
}
