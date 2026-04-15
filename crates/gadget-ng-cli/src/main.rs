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
    },
    /// Escribe un snapshot del estado inicial (IC) resuelto.
    Snapshot {
        #[arg(long, help = "Ruta al archivo TOML de experimento")]
        config: PathBuf,
        #[arg(long, help = "Directorio de salida del snapshot")]
        out: PathBuf,
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
        } => {
            let cfg = config_load::load_run_config(&config)?;
            run_with_runtime(|rt| engine::run_stepping(rt, &cfg, &out, snapshot))?;
        }
        Commands::Snapshot { config, out } => {
            let cfg = config_load::load_run_config(&config)?;
            run_with_runtime(|rt| engine::run_snapshot(rt, &cfg, &out))?;
        }
    }
    Ok(())
}
