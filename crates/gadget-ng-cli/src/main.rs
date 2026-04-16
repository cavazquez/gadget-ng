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
        #[arg(long, help = "Directorio de salida de una corrida anterior para reanudar")]
        resume: Option<PathBuf>,
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
        Commands::Stepping { config, out, snapshot, resume } => {
            let cfg = config_load::load_run_config(&config)?;
            run_with_runtime(|rt| engine::run_stepping(rt, &cfg, &out, snapshot, resume.as_deref()))?;
        }
        Commands::Snapshot { config, out } => {
            let cfg = config_load::load_run_config(&config)?;
            run_with_runtime(|rt| engine::run_snapshot(rt, &cfg, &out))?;
        }
        Commands::Visualize { snapshot, output, width, height, projection, color } => {
            engine::run_visualize(&snapshot, &output, width, height, &projection, &color)?;
        }
        Commands::Analyse { snapshot, out, linking_length, min_particles, pk_mesh } => {
            engine::run_analyse(&snapshot, &out, linking_length, min_particles, pk_mesh)?;
        }
    }
    Ok(())
}
