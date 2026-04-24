mod analyze_cmd;
mod config_load;
mod engine;
mod error;
mod insitu;
mod mah_cmd;
mod merge_tree_cmd;

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
    /// Construye el merger tree conectando catálogos FoF de snapshots consecutivos.
    ///
    /// Sigue partículas entre snapshots para identificar progenitores y mergers.
    /// Escribe `merger_tree.json` en el directorio de salida.
    ///
    /// Ejemplo:
    ///   gadget-ng merge-tree \
    ///     --catalogs "runs/cosmo/halos_000.jsonl,runs/cosmo/halos_001.jsonl" \
    ///     --snapshots "runs/cosmo/snap_000,runs/cosmo/snap_001" \
    ///     --out runs/cosmo/merger_tree.json
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
        } => {
            let cfg = config_load::load_run_config(&config)?;
            run_with_runtime(|rt| {
                engine::run_stepping(rt, &cfg, &out, snapshot, resume.as_deref())
            })?;
            // Si vis_snapshot > 0, renderizar el snapshot final.
            if vis_snapshot > 0 {
                let snap_dir = out.join("snapshot_final");
                if snap_dir.exists() {
                    use gadget_ng_core::SnapshotFormat;
                    use gadget_ng_vis::Projection;
                    let data =
                        gadget_ng_io::read_snapshot_formatted(SnapshotFormat::Jsonl, &snap_dir);
                    if let Ok(data) = data {
                        let positions: Vec<gadget_ng_core::Vec3> =
                            data.particles.iter().map(|p| p.position).collect();
                        let proj = match vis_proj.to_lowercase().as_str() {
                            "xz" => Projection::XZ,
                            "yz" => Projection::YZ,
                            _ => Projection::XY,
                        };
                        let pixels = match vis_mode.to_lowercase().as_str() {
                            "density" => gadget_ng_vis::render_density_ppm(
                                &positions, data.box_size, 1024, 1024, proj,
                            ),
                            _ => gadget_ng_vis::render_ppm_projection(
                                &positions, data.box_size, 1024, 1024, proj,
                            ),
                        };
                        let ext = if vis_format.to_lowercase() == "png" { "png" } else { "ppm" };
                        let out_path = out.join(format!("snapshot_final.{ext}"));
                        let result = if ext == "png" {
                            gadget_ng_vis::write_png(&out_path, &pixels, 1024, 1024)
                        } else {
                            gadget_ng_vis::write_ppm(&out_path, &pixels, 1024, 1024)
                        };
                        match result {
                            Ok(()) => eprintln!("[vis] imagen escrita en {:?}", out_path),
                            Err(e) => eprintln!("[vis] Error escribiendo imagen: {e}"),
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
            subfind,
            subfind_min_particles,
            hdf5_catalog,
            cm21,
            igm_temp,
            agn_stats,
            eor_state,
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
    }
    Ok(())
}
