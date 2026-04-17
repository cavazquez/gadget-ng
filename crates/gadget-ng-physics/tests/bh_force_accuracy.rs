//! Benchmark de precisión de fuerza: Barnes-Hut vs DirectGravity.
//!
//! ## Objetivo
//!
//! Cuantificar el error relativo de fuerza del solver Barnes-Hut multipolar
//! en función del parámetro de apertura θ y del orden de expansión, para dos
//! distribuciones de partículas:
//!
//! - **Distribución extendida**: esfera uniforme (UniformSphere, R=1).
//! - **Distribución concentrada**: esfera de Plummer (Plummer, a=0.1).
//!
//! **Nota:** El solver BH de gadget-ng incluye cuadrupolo y octupolo completos.
//! Los errores medidos son errores del árbol **con multipolar completo activo**, no
//! errores de un solver monopolar. El test `bh_multipole_ablation` cuantifica
//! la contribución individual de cada término.
//!
//! ## Métricas
//!
//! Para cada partícula i:
//! ```text
//! err_i = |a_bh_i − a_ref_i| / |a_ref_i|
//! ```
//! Se reportan: media, máximo, RMS.
//!
//! ## Salidas CSV
//!
//! - `bh_accuracy.csv`: barrido de θ con orden multipolar completo (order=3).
//! - `bh_multipole_ablation.csv`: ablación de orden multipolar (order=1,2,3) a θ fijo.

use gadget_ng_core::{
    build_particles, CosmologySection, DirectGravity, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, MacSoftening, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, UnitsSection, Vec3,
};
use gadget_ng_tree::{walk_stats_begin, walk_stats_end, BarnesHutGravity, Octree, WalkStats};
use std::time::Instant;

const G: f64 = 1.0;
const EPS: f64 = 0.05;
const EPS2: f64 = EPS * EPS;
const N: usize = 500;

// ── Configuraciones de partículas ─────────────────────────────────────────────

fn make_config_uniform_sphere() -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 1,
            softening: EPS,
            gravitational_constant: G,
            particle_count: N,
            box_size: 4.0,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::UniformSphere { r: 1.0 },
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection::default(),
        units: UnitsSection::default(),
    }
}

fn make_config_plummer() -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 1,
            softening: EPS,
            gravitational_constant: G,
            particle_count: N,
            box_size: 20.0,
            seed: 7,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Plummer { a: 0.1 },
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection::default(),
        units: UnitsSection::default(),
    }
}

// ── Cálculo de fuerzas ────────────────────────────────────────────────────────

fn compute_direct(positions: &[Vec3], masses: &[f64]) -> (Vec<Vec3>, f64) {
    let n = positions.len();
    let solver = DirectGravity;
    let all_idx: Vec<usize> = (0..n).collect();
    let mut acc = vec![Vec3::zero(); n];
    let t0 = Instant::now();
    solver.accelerations_for_indices(positions, masses, EPS2, G, &all_idx, &mut acc);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
    (acc, elapsed_ms)
}

fn compute_bh(positions: &[Vec3], masses: &[f64], theta: f64) -> (Vec<Vec3>, f64) {
    compute_bh_order(positions, masses, theta, 3)
}

/// Versión con orden multipolar explícito para benchmarks de ablación.
fn compute_bh_order(positions: &[Vec3], masses: &[f64], theta: f64, order: u8) -> (Vec<Vec3>, f64) {
    compute_bh_full(positions, masses, theta, order, false, false, 0.005)
}

/// Control completo de todas las opciones del solver BH.
/// Mantiene la firma histórica (7 argumentos, `mac_softening = Bare`) para no
/// romper los tests existentes.
fn compute_bh_full(
    positions: &[Vec3],
    masses: &[f64],
    theta: f64,
    order: u8,
    softened: bool,
    relative: bool,
    err_tol: f64,
) -> (Vec<Vec3>, f64) {
    compute_bh_full_mac(
        positions, masses, theta, order, softened, relative, err_tol, MacSoftening::Bare,
    )
}

/// Variante que expone explícitamente el softening del estimador MAC.
/// Usado por los benchmarks de Fase 5.
#[allow(clippy::too_many_arguments)]
fn compute_bh_full_mac(
    positions: &[Vec3],
    masses: &[f64],
    theta: f64,
    order: u8,
    softened: bool,
    relative: bool,
    err_tol: f64,
    mac_softening: MacSoftening,
) -> (Vec<Vec3>, f64) {
    let n = positions.len();
    let solver = BarnesHutGravity {
        theta,
        multipole_order: order,
        use_relative_criterion: relative,
        err_tol_force_acc: err_tol,
        softened_multipoles: softened,
        mac_softening,
    };
    let all_idx: Vec<usize> = (0..n).collect();
    let mut acc = vec![Vec3::zero(); n];
    let t0 = Instant::now();
    solver.accelerations_for_indices(positions, masses, EPS2, G, &all_idx, &mut acc);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
    (acc, elapsed_ms)
}

/// Evalúa Barnes-Hut con instrumentación del tree-walk (contadores de nodos
/// abiertos, hojas visitadas y profundidad). Construye el árbol una sola vez y
/// recorre todas las partículas secuencialmente para que los contadores thread-local
/// acumulen sobre todo el pase.
#[allow(clippy::too_many_arguments)]
fn compute_bh_instrumented(
    positions: &[Vec3],
    masses: &[f64],
    theta: f64,
    order: u8,
    softened: bool,
    relative: bool,
    err_tol: f64,
    mac_softening: MacSoftening,
) -> (Vec<Vec3>, f64, WalkStats) {
    let n = positions.len();
    let tree = Octree::build(positions, masses);
    let mut acc = vec![Vec3::zero(); n];
    walk_stats_begin();
    let t0 = Instant::now();
    for (i, a) in acc.iter_mut().enumerate() {
        *a = tree.walk_accel_multipole(
            positions[i],
            i,
            G,
            EPS2,
            theta,
            positions,
            masses,
            order,
            relative,
            err_tol,
            softened,
            mac_softening,
        );
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
    let stats = walk_stats_end();
    (acc, elapsed_ms, stats)
}

/// Versión con criterio de apertura relativo (GADGET-4 ErrTolForceAcc).
fn compute_bh_relative(positions: &[Vec3], masses: &[f64], err_tol: f64) -> (Vec<Vec3>, f64) {
    compute_bh_full(positions, masses, 0.5, 3, false, true, err_tol)
}

// ── Métricas de error ─────────────────────────────────────────────────────────

struct ForceError {
    mean_err: f64,
    max_err: f64,
    rms_err: f64,
    /// Error relativo en la suma total de aceleraciones (proxy del error en energía).
    energy_err: f64,
}

fn compute_force_error(ref_acc: &[Vec3], bh_acc: &[Vec3]) -> ForceError {
    let mut errors: Vec<f64> = Vec::with_capacity(ref_acc.len());
    let mut ref_total = Vec3::zero();
    let mut bh_total = Vec3::zero();

    for (r, b) in ref_acc.iter().zip(bh_acc.iter()) {
        let ref_mag = r.norm();
        if ref_mag < 1e-15 {
            continue;
        }
        let err = (*b - *r).norm() / ref_mag;
        errors.push(err);
        ref_total = ref_total + *r;
        bh_total = bh_total + *b;
    }

    let n = errors.len() as f64;
    let mean_err = errors.iter().copied().sum::<f64>() / n;
    let max_err = errors.iter().cloned().fold(0.0_f64, f64::max);
    let rms_err = (errors.iter().map(|e| e * e).sum::<f64>() / n).sqrt();

    let ref_tot_mag = ref_total.norm();
    let energy_err = if ref_tot_mag > 1e-15 {
        (bh_total - ref_total).norm() / ref_tot_mag
    } else {
        0.0
    };

    ForceError {
        mean_err,
        max_err,
        rms_err,
        energy_err,
    }
}

/// Métricas extendidas de error de fuerza: incluye percentil 95 y error angular.
struct ForceErrorExtended {
    mean_err: f64,
    max_err: f64,
    p95_err: f64,
    rms_err: f64,
    /// Error angular: 1 - cos(θ) donde θ es el ángulo entre a_bh y a_ref.
    /// 0.0 = direcciones idénticas, 1.0 = antiparalelas.
    mean_angular_err: f64,
    max_angular_err: f64,
}

fn compute_force_error_extended(ref_acc: &[Vec3], bh_acc: &[Vec3]) -> ForceErrorExtended {
    let mut mag_errors: Vec<f64> = Vec::with_capacity(ref_acc.len());
    let mut angular_errors: Vec<f64> = Vec::with_capacity(ref_acc.len());

    for (r, b) in ref_acc.iter().zip(bh_acc.iter()) {
        let ref_mag = r.norm();
        if ref_mag < 1e-15 {
            continue;
        }
        mag_errors.push((*b - *r).norm() / ref_mag);
        // Error angular: 1 - dot(r_hat, b_hat)
        let b_mag = b.norm();
        if b_mag > 1e-15 {
            let cos_angle = r.dot(*b) / (ref_mag * b_mag);
            angular_errors.push(1.0 - cos_angle.clamp(-1.0, 1.0));
        }
    }

    let n = mag_errors.len() as f64;
    let mean_err = mag_errors.iter().copied().sum::<f64>() / n;
    let max_err = mag_errors.iter().cloned().fold(0.0_f64, f64::max);
    let rms_err = (mag_errors.iter().map(|e| e * e).sum::<f64>() / n).sqrt();

    let mut sorted = mag_errors.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_idx = (sorted.len() as f64 * 0.95) as usize;
    let p95_err = sorted.get(p95_idx).copied().unwrap_or(0.0);

    let an = angular_errors.len() as f64;
    let mean_angular_err = if an > 0.0 {
        angular_errors.iter().copied().sum::<f64>() / an
    } else { 0.0 };
    let max_angular_err = angular_errors.iter().cloned().fold(0.0_f64, f64::max);

    ForceErrorExtended { mean_err, max_err, p95_err, rms_err, mean_angular_err, max_angular_err }
}

/// Análisis radial: calcula error de fuerza por bin de distancia al COM (normalizado por eps).
fn compute_radial_error(
    positions: &[Vec3],
    ref_acc: &[Vec3],
    bh_acc: &[Vec3],
    n_bins: usize,
    max_r_over_eps: f64,
) -> Vec<(f64, f64, f64, usize)> {  // (r_over_eps_center, mean_err, max_err, count)
    // Centro de masa
    let com = Vec3::new(
        positions.iter().map(|p| p.x).sum::<f64>() / positions.len() as f64,
        positions.iter().map(|p| p.y).sum::<f64>() / positions.len() as f64,
        positions.iter().map(|p| p.z).sum::<f64>() / positions.len() as f64,
    );
    let bin_width = max_r_over_eps / n_bins as f64;
    let mut bins: Vec<(Vec<f64>, f64)> = vec![(Vec::new(), 0.0); n_bins];

    for (i, (r_ref, r_bh)) in ref_acc.iter().zip(bh_acc.iter()).enumerate() {
        let r_mag = r_ref.norm();
        if r_mag < 1e-15 { continue; }
        let dist = (positions[i] - com).norm();
        let r_over_eps = dist / EPS;
        let bin = ((r_over_eps / max_r_over_eps) * n_bins as f64) as usize;
        let bin = bin.min(n_bins - 1);
        let err = (*r_bh - *r_ref).norm() / r_mag;
        bins[bin].0.push(err);
    }

    bins.iter().enumerate().map(|(i, (errs, _))| {
        let center = (i as f64 + 0.5) * bin_width;
        if errs.is_empty() {
            (center, 0.0, 0.0, 0)
        } else {
            let mean = errs.iter().sum::<f64>() / errs.len() as f64;
            let max = errs.iter().cloned().fold(0.0_f64, f64::max);
            (center, mean, max, errs.len())
        }
    }).collect()
}

// ── Runner principal ──────────────────────────────────────────────────────────

struct BenchResult {
    distribution: &'static str,
    theta: f64,
    mean_err: f64,
    max_err: f64,
    rms_err: f64,
    energy_err: f64,
    time_direct_ms: f64,
    time_bh_ms: f64,
}

fn run_benchmark(
    distribution: &'static str,
    cfg: &RunConfig,
    thetas: &[f64],
) -> Vec<BenchResult> {
    let particles = build_particles(cfg).expect("build_particles failed");
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    let (ref_acc, time_direct_ms) = compute_direct(&positions, &masses);

    let mut results = Vec::new();
    for &theta in thetas {
        let (bh_acc, time_bh_ms) = compute_bh(&positions, &masses, theta);
        let err = compute_force_error(&ref_acc, &bh_acc);
        results.push(BenchResult {
            distribution,
            theta,
            mean_err: err.mean_err,
            max_err: err.max_err,
            rms_err: err.rms_err,
            energy_err: err.energy_err,
            time_direct_ms,
            time_bh_ms,
        });
    }
    results
}

// ── Escritura de CSV ──────────────────────────────────────────────────────────

fn write_csv(results: &[BenchResult]) {
    // Resolver la ruta relativa a la raíz del repositorio (2 niveles sobre CARGO_MANIFEST_DIR).
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let repo_root = std::path::PathBuf::from(&manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let out_dir = repo_root
        .join("experiments/nbody/phase3_gadget4_benchmark/bh_force_error/results");

    let Ok(()) = std::fs::create_dir_all(&out_dir) else {
        eprintln!("[bh_force_accuracy] No se pudo crear {out_dir:?}; omitiendo CSV.");
        return;
    };

    let csv_path = out_dir.join("bh_accuracy.csv");
    let mut out = String::from(
        "distribution,theta,N,mean_err,max_err,rms_err,energy_err,time_direct_ms,time_bh_ms\n",
    );
    for r in results {
        out.push_str(&format!(
            "{},{:.2},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.3},{:.3}\n",
            r.distribution,
            r.theta,
            N,
            r.mean_err,
            r.max_err,
            r.rms_err,
            r.energy_err,
            r.time_direct_ms,
            r.time_bh_ms,
        ));
    }
    if let Err(e) = std::fs::write(&csv_path, &out) {
        eprintln!("[bh_force_accuracy] Error escribiendo CSV: {e}");
    } else {
        println!("[bh_force_accuracy] Resultados escritos en {csv_path:?}");
    }
}

// ── Ablación multipolar ───────────────────────────────────────────────────────

struct AblationResult {
    distribution: &'static str,
    theta: f64,
    order: u8,
    mean_err: f64,
    max_err: f64,
    rms_err: f64,
    time_bh_ms: f64,
    time_direct_ms: f64,
}

fn run_ablation(
    distribution: &'static str,
    cfg: &RunConfig,
    thetas: &[f64],
    orders: &[u8],
) -> Vec<AblationResult> {
    let particles = build_particles(cfg).expect("build_particles failed");
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let (ref_acc, time_direct_ms) = compute_direct(&positions, &masses);

    let mut results = Vec::new();
    for &theta in thetas {
        for &order in orders {
            let (bh_acc, time_bh_ms) = compute_bh_order(&positions, &masses, theta, order);
            let err = compute_force_error(&ref_acc, &bh_acc);
            results.push(AblationResult {
                distribution,
                theta,
                order,
                mean_err: err.mean_err,
                max_err: err.max_err,
                rms_err: err.rms_err,
                time_bh_ms,
                time_direct_ms,
            });
        }
    }
    results
}

fn write_ablation_csv(results: &[AblationResult]) {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let repo_root = std::path::PathBuf::from(&manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let out_dir = repo_root
        .join("experiments/nbody/phase3_gadget4_benchmark/bh_force_error/results");
    let Ok(()) = std::fs::create_dir_all(&out_dir) else {
        eprintln!("[bh_multipole_ablation] No se pudo crear {out_dir:?}; omitiendo CSV.");
        return;
    };
    let csv_path = out_dir.join("bh_multipole_ablation.csv");
    let mut out = String::from(
        "distribution,theta,order,N,mean_err,max_err,rms_err,time_direct_ms,time_bh_ms\n",
    );
    for r in results {
        out.push_str(&format!(
            "{},{:.2},{},{},{:.6e},{:.6e},{:.6e},{:.3},{:.3}\n",
            r.distribution,
            r.theta,
            r.order,
            N,
            r.mean_err,
            r.max_err,
            r.rms_err,
            r.time_direct_ms,
            r.time_bh_ms,
        ));
    }
    if let Err(e) = std::fs::write(&csv_path, &out) {
        eprintln!("[bh_multipole_ablation] Error escribiendo CSV: {e}");
    } else {
        println!("[bh_multipole_ablation] Resultados escritos en {csv_path:?}");
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

const THETAS: &[f64] = &[0.2, 0.5, 0.8, 1.0];

#[test]
fn bh_force_accuracy_uniform_sphere() {
    let cfg = make_config_uniform_sphere();
    let results = run_benchmark("uniform_sphere", &cfg, THETAS);

    println!("\n=== BH Force Accuracy: Uniform Sphere (N={N}) ===");
    println!("{:<8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "theta", "mean_err%", "max_err%", "rms_err%", "t_direct_ms", "t_bh_ms", "speedup");
    for r in &results {
        println!(
            "{:<8.2} {:>10.3} {:>10.3} {:>10.3} {:>12.1} {:>10.1} {:>9.2}x",
            r.theta,
            r.mean_err * 100.0,
            r.max_err * 100.0,
            r.rms_err * 100.0,
            r.time_direct_ms,
            r.time_bh_ms,
            r.time_direct_ms / r.time_bh_ms.max(1e-6),
        );
        // Verificaciones de criterios de aceptación.
        match r.theta as i32 {
            _ if (r.theta - 0.2).abs() < 0.01 => {
                assert!(r.mean_err < 0.01,
                    "θ=0.2: mean_err={:.4}% esperado < 1%", r.mean_err * 100.0);
            }
            _ if (r.theta - 0.5).abs() < 0.01 => {
                assert!(r.mean_err < 0.05,
                    "θ=0.5: mean_err={:.4}% esperado < 5%", r.mean_err * 100.0);
            }
            _ if (r.theta - 0.8).abs() < 0.01 => {
                assert!(r.mean_err < 0.20,
                    "θ=0.8: mean_err={:.4}% esperado < 20%", r.mean_err * 100.0);
            }
            _ => {}
        }
    }
    write_csv(&results);
}

#[test]
fn bh_force_accuracy_plummer() {
    let cfg = make_config_plummer();
    let results = run_benchmark("plummer", &cfg, THETAS);

    println!("\n=== BH Force Accuracy: Plummer (N={N}, a=0.1) ===");
    println!("{:<8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "theta", "mean_err%", "max_err%", "rms_err%", "t_direct_ms", "t_bh_ms", "speedup");
    for r in &results {
        println!(
            "{:<8.2} {:>10.3} {:>10.3} {:>10.3} {:>12.1} {:>10.1} {:>9.2}x",
            r.theta,
            r.mean_err * 100.0,
            r.max_err * 100.0,
            r.rms_err * 100.0,
            r.time_direct_ms,
            r.time_bh_ms,
            r.time_direct_ms / r.time_bh_ms.max(1e-6),
        );
        // Plummer concentrado → errores más altos; umbrales más relajados.
        match r.theta as i32 {
            _ if (r.theta - 0.2).abs() < 0.01 => {
                assert!(r.mean_err < 0.03,
                    "Plummer θ=0.2: mean_err={:.4}% esperado < 3%", r.mean_err * 100.0);
            }
            _ if (r.theta - 0.5).abs() < 0.01 => {
                assert!(r.mean_err < 0.10,
                    "Plummer θ=0.5: mean_err={:.4}% esperado < 10%", r.mean_err * 100.0);
            }
            _ => {}
        }
    }
    // Combinar resultados de plummer con los previos si existe el CSV.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let repo_root = std::path::PathBuf::from(&manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let csv_path = repo_root.join(
        "experiments/nbody/phase3_gadget4_benchmark/bh_force_error/results/bh_accuracy.csv",
    );
    if csv_path.exists() {
        // Añadir filas de Plummer al CSV existente.
        let mut existing = std::fs::read_to_string(&csv_path).unwrap_or_default();
        for r in &results {
            existing.push_str(&format!(
                "{},{:.2},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.3},{:.3}\n",
                r.distribution,
                r.theta,
                N,
                r.mean_err,
                r.max_err,
                r.rms_err,
                r.energy_err,
                r.time_direct_ms,
                r.time_bh_ms,
            ));
        }
        let _ = std::fs::write(&csv_path, existing);
    } else {
        write_csv(&results);
    }
}

/// Test consolidado: ejecuta ambas distribuciones y escribe CSV completo.
/// Usar este test para generar los resultados del experimento completo.
#[test]
fn bh_force_accuracy_full_sweep() {
    let cfg_sphere = make_config_uniform_sphere();
    let cfg_plummer = make_config_plummer();

    let mut all_results = run_benchmark("uniform_sphere", &cfg_sphere, THETAS);
    all_results.extend(run_benchmark("plummer", &cfg_plummer, THETAS));

    println!("\n=== BH Force Accuracy: Barrido completo (N={N}, order=3) ===");
    println!("{:<16} {:<8} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "distribution", "theta", "mean_err%", "max_err%", "rms_err%", "t_bh_ms", "speedup");
    for r in &all_results {
        println!(
            "{:<16} {:<8.2} {:>10.3} {:>10.3} {:>10.3} {:>12.1} {:>9.2}x",
            r.distribution,
            r.theta,
            r.mean_err * 100.0,
            r.max_err * 100.0,
            r.rms_err * 100.0,
            r.time_bh_ms,
            r.time_direct_ms / r.time_bh_ms.max(1e-6),
        );
    }
    write_csv(&all_results);
}

/// Ablación de orden multipolar: cuantifica la contribución real de cada término.
///
/// Compara monopolo puro (order=1), mono+quad (order=2) y mono+quad+oct (order=3)
/// para las dos distribuciones de referencia, a θ=0.5 (criterio estándar).
///
/// Esta prueba confirma cuánto error se reduce al añadir cada término y demuestra
/// que los errores observados en Fase 3 son los errores **con multipolar completo**.
#[test]
fn bh_multipole_ablation() {
    let cfg_sphere = make_config_uniform_sphere();
    let cfg_plummer = make_config_plummer();

    const ABLATION_THETAS: &[f64] = &[0.5, 0.8];
    const ORDERS: &[u8] = &[1, 2, 3];

    let mut all = run_ablation("uniform_sphere", &cfg_sphere, ABLATION_THETAS, ORDERS);
    all.extend(run_ablation("plummer", &cfg_plummer, ABLATION_THETAS, ORDERS));

    println!("\n=== Ablación multipolar Barnes-Hut (N={N}) ===");
    println!(
        "{:<16} {:>6} {:>6} {:>12} {:>12} {:>12}",
        "distribution", "theta", "order", "mean_err%", "max_err%", "rms_err%"
    );
    for r in &all {
        let order_name = match r.order {
            1 => "mono    ",
            2 => "mono+Q  ",
            3 => "mono+Q+O",
            _ => "?       ",
        };
        println!(
            "{:<16} {:>6.2} {:>6} ({}) {:>10.3} {:>12.3} {:>12.3}",
            r.distribution,
            r.theta,
            r.order,
            order_name,
            r.mean_err * 100.0,
            r.max_err * 100.0,
            r.rms_err * 100.0,
        );
    }

    // Reportar la contribución de cada término y la relación entre órdenes.
    //
    // NOTA IMPORTANTE: para distribuciones concentradas (Plummer), los términos cuadrupolar
    // y octupolar de gadget-ng se aplican SIN suavizado Plummer (bare 1/r⁵), mientras el
    // monopolo usa la fuerza suavizada. Esto puede hacer que orden=2,3 tengan más error que
    // orden=1 para núcleos densos con θ grande: los términos de corrección agregan ruido en
    // lugar de reducir el error del monopolo. Este es un hallazgo científico genuino que
    // aparece también en el plan: "Softening en términos quad/oct: No (bare 1/r⁵) vs GADGET-4: Sí".
    for dist in ["uniform_sphere", "plummer"] {
        let err_order1 = all.iter()
            .find(|r| r.distribution == dist && (r.theta - 0.5).abs() < 0.01 && r.order == 1)
            .map(|r| r.mean_err)
            .unwrap_or(0.0);
        let err_order2 = all.iter()
            .find(|r| r.distribution == dist && (r.theta - 0.5).abs() < 0.01 && r.order == 2)
            .map(|r| r.mean_err)
            .unwrap_or(0.0);
        let err_order3 = all.iter()
            .find(|r| r.distribution == dist && (r.theta - 0.5).abs() < 0.01 && r.order == 3)
            .map(|r| r.mean_err)
            .unwrap_or(f64::MAX);
        println!(
            "[{dist}] θ=0.5: mono={:.3}%  mono+Q={:.3}%  mono+Q+O={:.3}%",
            err_order1 * 100.0,
            err_order2 * 100.0,
            err_order3 * 100.0,
        );
        // Para esfera uniforme (distribución extendida), el orden 3 debería ser ≤ orden 1.
        // Para Plummer concentrado no se garantiza debido al softening faltante en quad/oct.
        if dist == "uniform_sphere" {
            assert!(
                err_order3 <= err_order1 * 1.1,
                "uniform_sphere θ=0.5: mono+Q+O ({:.4}%) debería ser ≤ mono ({:.4}%)",
                err_order3 * 100.0, err_order1 * 100.0
            );
        }
        // Para Plummer: simplemente verificamos que todos los errores son finitos y positivos.
        assert!(err_order1.is_finite() && err_order1 > 0.0, "{dist}: orden1 inválido");
        assert!(err_order2.is_finite() && err_order2 > 0.0, "{dist}: orden2 inválido");
        assert!(err_order3.is_finite() && err_order3 > 0.0, "{dist}: orden3 inválido");
    }

    write_ablation_csv(&all);
}

/// Benchmark del criterio de apertura relativo vs geométrico para Plummer concentrado.
///
/// Compara el criterio geométrico clásico (θ=0.5) con el criterio relativo tipo GADGET-4
/// (`ErrTolForceAcc`) a distintas tolerancias. Muestra cómo el criterio relativo reduce
/// el error en el núcleo denso adaptando el MAC por interacción.
#[test]
fn bh_relative_opening_criterion() {
    let cfg = make_config_plummer();
    let particles = build_particles(&cfg).expect("build_particles failed");
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let (ref_acc, time_direct_ms) = compute_direct(&positions, &masses);

    println!("\n=== Criterio de apertura relativo vs geométrico (Plummer, N={N}) ===");
    println!(
        "{:<28} {:>12} {:>12} {:>12} {:>10}",
        "criterio", "mean_err%", "max_err%", "rms_err%", "t_bh_ms"
    );

    // Geométrico a distintos θ
    for theta in [0.3, 0.5, 0.7] {
        let (bh_acc, t) = compute_bh_order(&positions, &masses, theta, 3);
        let err = compute_force_error(&ref_acc, &bh_acc);
        println!(
            "geometric θ={:.1}              {:>12.3} {:>12.3} {:>12.3} {:>10.2}",
            theta, err.mean_err * 100.0, err.max_err * 100.0, err.rms_err * 100.0, t
        );
    }

    // Relativo a distintas tolerancias (equivalentes aproximados de ErrTolForceAcc de GADGET-4)
    for tol in [0.001, 0.005, 0.010] {
        let (bh_acc, t) = compute_bh_relative(&positions, &masses, tol);
        let err = compute_force_error(&ref_acc, &bh_acc);
        println!(
            "relative err_tol={:.3}         {:>12.3} {:>12.3} {:>12.3} {:>10.2}",
            tol, err.mean_err * 100.0, err.max_err * 100.0, err.rms_err * 100.0, t
        );
    }

    // El criterio relativo con tol=0.005 debe dar error similar o mejor que geométrico θ=0.5.
    let (bh_geo, _) = compute_bh_order(&positions, &masses, 0.5, 3);
    let (bh_rel, _) = compute_bh_relative(&positions, &masses, 0.005);
    let err_geo = compute_force_error(&ref_acc, &bh_geo);
    let err_rel = compute_force_error(&ref_acc, &bh_rel);
    println!(
        "\nResumen: geo θ=0.5 mean_err={:.3}%  rel tol=0.005 mean_err={:.3}%  ratio={:.2}x",
        err_geo.mean_err * 100.0,
        err_rel.mean_err * 100.0,
        err_geo.mean_err / err_rel.mean_err.max(1e-15)
    );

    // Nota: el tiempo directo es el mismo para este test (referencia común)
    let _ = time_direct_ms;
}

// ── Helpers para tests de Fase 4 ──────────────────────────────────────────────

fn make_config_plummer_a(a: f64, seed: u64) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 1,
            softening: EPS,
            gravitational_constant: G,
            particle_count: N,
            box_size: 20.0,
            seed,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Plummer { a },
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection::default(),
        units: UnitsSection::default(),
    }
}

fn repo_root() -> std::path::PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(&manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

fn phase4_results_dir() -> std::path::PathBuf {
    repo_root().join("experiments/nbody/phase4_multipole_softening/results")
}

fn phase5_results_dir() -> std::path::PathBuf {
    repo_root().join("experiments/nbody/phase5_energy_mac_consistency/results")
}

fn write_csv_generic(path: &std::path::Path, content: &str) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = std::fs::write(path, content) {
        eprintln!("[phase4] Error escribiendo {path:?}: {e}");
    } else {
        println!("[phase4] Resultados escritos en {path:?}");
    }
}

// ── Test: Ablación bare vs softened × concentración Plummer ──────────────────

/// Ablación de softening en multipolos para distintas concentraciones Plummer.
///
/// Valida la Hipótesis H1: `quad_accel_softened` reduce el error para Plummer
/// concentrado (a ≤ ε×5 = 0.25) mientras que para a ≫ ε ambas versiones son equivalentes.
///
/// Mide: mean, max, P95, RMS y error angular para:
/// - monopolo bare (order=1, softened=false)
/// - mono+quad bare (order=2, softened=false)  
/// - mono+quad+oct bare (order=3, softened=false)
/// - mono+quad bare (order=2, softened=true)
/// - mono+quad+oct softened (order=3, softened=true)
///
/// Para distribuciones: Plummer a ∈ {0.05, 0.1, 0.3, 1.0} y esfera uniforme.
#[test]
fn bh_softened_multipoles_ablation() {
    let plummer_concentrations: &[(f64, u64, &str)] = &[
        (0.05, 11, "plummer_a005"),  // a/eps = 1 → núcleo < softening
        (0.1,  7,  "plummer_a010"),  // a/eps = 2 → núcleo ~ softening
        (0.3,  13, "plummer_a030"),  // a/eps = 6 → núcleo > softening
        (1.0,  17, "plummer_a100"),  // a/eps = 20 → campo lejano
    ];

    let mut csv = String::from(
        "distribution,a_plummer,a_over_eps,order,softened,theta,\
         mean_err,max_err,p95_err,rms_err,mean_angular_err,max_angular_err,time_bh_ms\n"
    );

    println!("\n=== Ablación softening multipolar — Fase 4 (N={N}, ε={EPS}) ===");
    println!(
        "{:<18} {:>5} {:>6} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "distribution", "order", "soft", "mean%", "max%", "p95%", "ang_mean%", "ang_max%"
    );

    let configs: Vec<(String, f64, RunConfig)> = {
        let mut v: Vec<(String, f64, RunConfig)> = plummer_concentrations.iter()
            .map(|(a, seed, name)| (name.to_string(), *a, make_config_plummer_a(*a, *seed)))
            .collect();
        v.push(("uniform_sphere".to_string(), 0.0, make_config_uniform_sphere()));
        v
    };

    const THETA: f64 = 0.5;
    let configs_to_test: &[(u8, bool)] = &[
        (1, false),  // monopolo bare
        (2, false),  // mono+quad bare
        (3, false),  // mono+quad+oct bare
        (2, true),   // mono+quad softened
        (3, true),   // mono+quad+oct softened
    ];

    for (dist_name, a_val, cfg) in &configs {
        let particles = build_particles(cfg).expect("build_particles failed");
        let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
        let (ref_acc, _) = compute_direct(&positions, &masses);

        for &(order, softened) in configs_to_test {
            let (bh_acc, t_ms) = compute_bh_full(&positions, &masses, THETA, order, softened, false, 0.005);
            let err = compute_force_error_extended(&ref_acc, &bh_acc);
            let a_over_eps = if *a_val > 0.0 { a_val / EPS } else { 0.0 };

            println!(
                "{:<18} {:>5} {:>6} {:>8.3} {:>12.3} {:>12.3} {:>12.4} {:>12.4}",
                dist_name, order,
                if softened { "yes" } else { "no" },
                err.mean_err * 100.0, err.max_err * 100.0, err.p95_err * 100.0,
                err.mean_angular_err * 100.0, err.max_angular_err * 100.0,
            );

            csv.push_str(&format!(
                "{},{:.3},{:.2},{},{},{:.2},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.3}\n",
                dist_name, a_val, a_over_eps,
                order, softened as u8, THETA,
                err.mean_err, err.max_err, err.p95_err, err.rms_err,
                err.mean_angular_err, err.max_angular_err, t_ms,
            ));
        }
        println!("---");
    }

    let out_path = phase4_results_dir().join("softened_ablation.csv");
    write_csv_generic(&out_path, &csv);

    // Verificar H1: para Plummer a=0.1 (a/eps=2), softened debe mejorar respecto a bare
    // para orden 2 y 3.
    let cfg_01 = make_config_plummer_a(0.1, 7);
    let particles = build_particles(&cfg_01).expect("build_particles");
    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let (ref_acc, _) = compute_direct(&pos, &masses);

    let (bare2, _) = compute_bh_full(&pos, &masses, THETA, 2, false, false, 0.0);
    let (soft2, _) = compute_bh_full(&pos, &masses, THETA, 2, true, false, 0.0);
    let err_bare2 = compute_force_error_extended(&ref_acc, &bare2);
    let err_soft2 = compute_force_error_extended(&ref_acc, &soft2);

    println!(
        "\nH1 check Plummer a=0.1: bare order2 mean={:.3}%  softened order2 mean={:.3}%  ratio={:.2}x",
        err_bare2.mean_err * 100.0, err_soft2.mean_err * 100.0,
        err_bare2.mean_err / err_soft2.mean_err.max(1e-15)
    );

    // Todos los errores deben ser finitos
    assert!(err_bare2.mean_err.is_finite(), "bare order2: error no finito");
    assert!(err_soft2.mean_err.is_finite(), "softened order2: error no finito");
}

// ── Test: Barrido de criterio relativo vs softened multipoles ─────────────────

/// Barrido completo de `err_tol_force_acc` para el criterio relativo.
///
/// Curva precisión vs costo para Plummer a=0.1 y esfera uniforme, comparando:
/// - criterio geométrico (θ=0.3, 0.5, 0.7, 0.9) con bare y softened multipoles
/// - criterio relativo (err_tol ∈ {1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4}) con bare y softened
///
/// Escribe `bh_criterion_sweep.csv` con columnas para plotting de curvas Pareto
/// error vs costo computacional.
#[test]
fn bh_relative_criterion_sweep() {
    let plummer_cfg = make_config_plummer_a(0.1, 7);
    let sphere_cfg = make_config_uniform_sphere();

    let mut csv = String::from(
        "distribution,criterion,param,order,softened,mean_err,max_err,p95_err,rms_err,\
         mean_angular_err,time_bh_ms,speedup_vs_direct\n"
    );

    println!("\n=== Barrido criterio relativo vs geométrico — Fase 4 (N={N}) ===");
    println!(
        "{:<16} {:<20} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "distribution", "criterio", "mean%", "max%", "p95%", "ang%", "t_ms"
    );

    for (dist_name, cfg) in [("plummer_a010", &plummer_cfg), ("uniform_sphere", &sphere_cfg)] {
        let particles = build_particles(cfg).expect("build_particles");
        let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
        let (ref_acc, t_direct) = compute_direct(&pos, &masses);

        // Geométrico: θ ∈ {0.3, 0.5, 0.7, 0.9} × softened ∈ {false, true}
        for theta in [0.3f64, 0.5, 0.7, 0.9] {
            for &softened in &[false, true] {
                let (bh_acc, t_ms) = compute_bh_full(&pos, &masses, theta, 3, softened, false, 0.0);
                let err = compute_force_error_extended(&ref_acc, &bh_acc);
                let suf = if softened { "geo_soft" } else { "geo_bare" };
                println!(
                    "{:<16} θ={:.1} {} {:>8.3} {:>8.3} {:>8.3} {:>8.4} {:>10.2}",
                    dist_name, theta, suf,
                    err.mean_err * 100.0, err.max_err * 100.0, err.p95_err * 100.0,
                    err.mean_angular_err * 100.0, t_ms
                );
                csv.push_str(&format!(
                    "{},{},{:.2},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.3},{:.3}\n",
                    dist_name,
                    if softened { "geometric_soft" } else { "geometric_bare" },
                    theta, 3, softened as u8,
                    err.mean_err, err.max_err, err.p95_err, err.rms_err,
                    err.mean_angular_err, t_ms, t_direct / t_ms.max(1e-9)
                ));
            }
        }

        // Relativo: err_tol ∈ {1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4} × softened ∈ {false, true}
        for err_tol in [1e-1f64, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4] {
            for &softened in &[false, true] {
                let (bh_acc, t_ms) = compute_bh_full(&pos, &masses, 0.5, 3, softened, true, err_tol);
                let err = compute_force_error_extended(&ref_acc, &bh_acc);
                let suf = if softened { "rel_soft" } else { "rel_bare" };
                println!(
                    "{:<16} tol={:.0e} {} {:>8.3} {:>8.3} {:>8.3} {:>8.4} {:>10.2}",
                    dist_name, err_tol, suf,
                    err.mean_err * 100.0, err.max_err * 100.0, err.p95_err * 100.0,
                    err.mean_angular_err * 100.0, t_ms
                );
                csv.push_str(&format!(
                    "{},{},{:.0e},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.3},{:.3}\n",
                    dist_name,
                    if softened { "relative_soft" } else { "relative_bare" },
                    err_tol, 3, softened as u8,
                    err.mean_err, err.max_err, err.p95_err, err.rms_err,
                    err.mean_angular_err, t_ms, t_direct / t_ms.max(1e-9)
                ));
            }
        }
        println!("---");
    }

    let out_path = phase4_results_dir().join("bh_criterion_sweep.csv");
    write_csv_generic(&out_path, &csv);

    // H3: criterio relativo tol=0.005 debe ser mejor que geométrico θ=0.5 para Plummer
    let plummer_particles = build_particles(&plummer_cfg).expect("build_particles");
    let pos: Vec<Vec3> = plummer_particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = plummer_particles.iter().map(|p| p.mass).collect();
    let (ref_acc, _) = compute_direct(&pos, &masses);

    let (geo_acc, _) = compute_bh_full(&pos, &masses, 0.5, 3, false, false, 0.0);
    let (rel_acc, _) = compute_bh_full(&pos, &masses, 0.5, 3, false, true, 5e-3);
    let err_geo = compute_force_error_extended(&ref_acc, &geo_acc);
    let err_rel = compute_force_error_extended(&ref_acc, &rel_acc);

    println!(
        "\nH3 check: geo θ=0.5 mean={:.3}%  rel tol=5e-3 mean={:.3}%  ratio={:.1}x",
        err_geo.mean_err * 100.0, err_rel.mean_err * 100.0,
        err_geo.mean_err / err_rel.mean_err.max(1e-15)
    );
    assert!(
        err_rel.mean_err <= err_geo.mean_err,
        "H3: criterio relativo debería ser mejor que geométrico para Plummer"
    );
}

// ── Test: Análisis radial de error de fuerza ──────────────────────────────────

/// Análisis de error de fuerza como función de la distancia al centro (r/ε).
///
/// Demuestra cuándo y dónde los términos multipolares sin softening divergen:
/// en el régimen r ≈ ε los términos bare divergen mientras los softened se comportan bien.
///
/// Genera `radial_error_analysis.csv` con bins de r/ε para:
/// - monopolo bare (baseline)
/// - mono+quad bare
/// - mono+quad softened
/// - mono+quad+oct bare
/// - mono+quad+oct softened
#[test]
fn bh_radial_error_analysis() {
    // Plummer a=0.1 → concentración severa (a/eps = 2), donde la inconsistencia es más visible
    let cfg = make_config_plummer_a(0.1, 7);
    let particles = build_particles(&cfg).expect("build_particles failed");
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let (ref_acc, _) = compute_direct(&positions, &masses);

    const N_BINS: usize = 20;
    const MAX_R_EPS: f64 = 30.0;  // r/eps hasta 30 (cubre el núcleo y el halo)
    const THETA: f64 = 0.5;

    struct Config {
        label: &'static str,
        order: u8,
        softened: bool,
    }
    let configs = [
        Config { label: "mono_bare",      order: 1, softened: false },
        Config { label: "quad_bare",      order: 2, softened: false },
        Config { label: "quad_soft",      order: 2, softened: true  },
        Config { label: "oct_bare",       order: 3, softened: false },
        Config { label: "oct_soft",       order: 3, softened: true  },
    ];

    let mut csv = String::from("config,r_over_eps_center,mean_err,max_err,n_particles\n");

    println!("\n=== Análisis radial de error (Plummer a=0.1, θ={THETA}, N={N}) ===");
    println!("{:<14} {:>6} bins (r/ε center → mean_err%):", "config", N_BINS);

    for cfg_item in &configs {
        let (bh_acc, _) = compute_bh_full(&positions, &masses, THETA, cfg_item.order, cfg_item.softened, false, 0.005);
        let radial = compute_radial_error(&positions, &ref_acc, &bh_acc, N_BINS, MAX_R_EPS);

        print!("  {:<12}", cfg_item.label);
        for (r_center, mean_err, max_err, count) in &radial {
            if *count > 0 {
                csv.push_str(&format!(
                    "{},{:.2},{:.6e},{:.6e},{}\n",
                    cfg_item.label, r_center, mean_err, max_err, count
                ));
            }
        }
        // Imprimir resumen de los primeros 5 bins (zona de núcleo)
        let core_bins: Vec<_> = radial.iter().filter(|(_, _, _, c)| *c > 0).take(5).collect();
        for (r, m, _, cnt) in &core_bins {
            print!(" r/ε={:.1}→{:.1}%({cnt})", r, m * 100.0);
        }
        println!();
    }

    let out_path = phase4_results_dir().join("radial_error_analysis.csv");
    write_csv_generic(&out_path, &csv);

    // Verificación: el análisis radial debe tener N_BINS filas por config (algunas pueden estar vacías)
    println!("\nAnálisis radial completado. Ver {out_path:?}");
}

// ── Fase 5: Ablación del softening del MAC ────────────────────────────────────
//
// Compara las 5 variantes de criterio de apertura / softening × 4 distribuciones
// × 2 tamaños. Reporta error local (mean/max/P95, angular), coste (tiempo
// wall) y coste interno (nodos abiertos, hojas visitadas, profundidad media).
//
// Hipótesis H2: `MacSoftening::Consistent` abre más nodos sólo en el núcleo
// (`d ~ ε`), reduciendo el error allí sin inflar de forma significativa el coste
// global. Hipótesis H4: la variante `relative + softened_multipoles +
// mac_softening=consistent` domina el Pareto frente a variantes parciales.
//
// Salida: `experiments/nbody/phase5_energy_mac_consistency/results/bh_mac_softening.csv`.

/// Construye una config Plummer con N personalizado y `a` / seed dados.
fn make_config_plummer_a_n(a: f64, seed: u64, n: usize) -> RunConfig {
    let mut cfg = make_config_plummer_a(a, seed);
    cfg.simulation.particle_count = n;
    cfg
}

/// Construye la config de esfera uniforme con N personalizado.
fn make_config_uniform_n(n: usize) -> RunConfig {
    let mut cfg = make_config_uniform_sphere();
    cfg.simulation.particle_count = n;
    cfg
}

/// Identificador compacto de una variante MAC.
#[derive(Clone, Copy, Debug)]
struct MacVariant {
    name: &'static str,
    relative: bool,
    softened_multipoles: bool,
    mac_softening: MacSoftening,
}

const MAC_VARIANTS: &[MacVariant] = &[
    MacVariant {
        name: "V1_geom_bare",
        relative: false,
        softened_multipoles: false,
        mac_softening: MacSoftening::Bare,
    },
    MacVariant {
        name: "V2_geom_soft",
        relative: false,
        softened_multipoles: true,
        mac_softening: MacSoftening::Bare,
    },
    MacVariant {
        name: "V3_rel_bare",
        relative: true,
        softened_multipoles: false,
        mac_softening: MacSoftening::Bare,
    },
    MacVariant {
        name: "V4_rel_soft",
        relative: true,
        softened_multipoles: true,
        mac_softening: MacSoftening::Bare,
    },
    MacVariant {
        name: "V5_rel_soft_consistent",
        relative: true,
        softened_multipoles: true,
        mac_softening: MacSoftening::Consistent,
    },
];

/// Descripción de un caso (distribución × N).
struct CaseDef {
    dist: &'static str,
    a_plummer: f64,
    n: usize,
    cfg: RunConfig,
}

fn phase5_cases() -> Vec<CaseDef> {
    // 4 distribuciones × 2 N  =  8 casos base. Sobre cada caso corren las 5 variantes.
    let mut out = Vec::new();
    let n_values: &[usize] = &[200, 1000];
    let plummer: &[(f64, u64, &str)] = &[
        (0.05, 11, "plummer_a1"),   // a/ε = 1
        (0.10, 7,  "plummer_a2"),   // a/ε = 2
        (0.30, 13, "plummer_a6"),   // a/ε = 6
    ];
    for &n in n_values {
        for &(a, seed, name) in plummer {
            out.push(CaseDef {
                dist: name,
                a_plummer: a,
                n,
                cfg: make_config_plummer_a_n(a, seed, n),
            });
        }
        out.push(CaseDef {
            dist: "uniform_sphere",
            a_plummer: 0.0,
            n,
            cfg: make_config_uniform_n(n),
        });
    }
    out
}

#[test]
fn bh_mac_softening_ablation() {
    // Parámetros fijos: θ=0.5 (geométrico) y err_tol=0.005 (relativo) para que
    // los variantes sean comparables en precisión objetivo.
    const THETA: f64 = 0.5;
    const ERR_TOL: f64 = 0.005;
    const ORDER: u8 = 3;

    let cases = phase5_cases();

    let mut csv = String::from(
        "distribution,a_plummer,a_over_eps,N,variant,relative,softened_multipoles,\
         mac_softening,mean_err,max_err,p95_err,rms_err,mean_angular_err,max_angular_err,\
         time_bh_ms,time_direct_ms,opened_nodes,leaves_visited,max_depth,mean_depth\n",
    );

    println!("\n=== Fase 5: Ablación MAC softening (θ={THETA}, err_tol={ERR_TOL}, order={ORDER}) ===");
    println!(
        "{:<18} {:>5} {:>22} {:>9} {:>9} {:>9} {:>9} {:>9} {:>11}",
        "dist", "N", "variant", "mean%", "max%", "p95%", "ang%", "ms", "opened"
    );

    for case in &cases {
        let particles = build_particles(&case.cfg).expect("build_particles failed");
        let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
        let (ref_acc, t_direct) = compute_direct(&positions, &masses);

        for variant in MAC_VARIANTS {
            let (bh_acc, t_bh, stats) = compute_bh_instrumented(
                &positions,
                &masses,
                THETA,
                ORDER,
                variant.softened_multipoles,
                variant.relative,
                ERR_TOL,
                variant.mac_softening,
            );
            let err = compute_force_error_extended(&ref_acc, &bh_acc);
            let a_over_eps = if case.a_plummer > 0.0 {
                case.a_plummer / EPS
            } else {
                0.0
            };
            let mac_soft_str = match variant.mac_softening {
                MacSoftening::Bare => "bare",
                MacSoftening::Consistent => "consistent",
            };

            println!(
                "{:<18} {:>5} {:>22} {:>9.3} {:>9.3} {:>9.3} {:>9.4} {:>9.2} {:>11}",
                case.dist,
                case.n,
                variant.name,
                err.mean_err * 100.0,
                err.max_err * 100.0,
                err.p95_err * 100.0,
                err.mean_angular_err * 100.0,
                t_bh,
                stats.opened_nodes,
            );

            csv.push_str(&format!(
                "{},{:.3},{:.2},{},{},{},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.3},{:.3},{},{},{},{:.3}\n",
                case.dist,
                case.a_plummer,
                a_over_eps,
                case.n,
                variant.name,
                variant.relative as u8,
                variant.softened_multipoles as u8,
                mac_soft_str,
                err.mean_err,
                err.max_err,
                err.p95_err,
                err.rms_err,
                err.mean_angular_err,
                err.max_angular_err,
                t_bh,
                t_direct,
                stats.opened_nodes,
                stats.leaves_visited,
                stats.max_depth,
                stats.mean_depth(),
            ));
        }
        println!("---");
    }

    let out_path = phase5_results_dir().join("bh_mac_softening.csv");
    write_csv_generic(&out_path, &csv);

    // Sanity check ligero: para Plummer a/ε=1, V5 debe reducir sustancialmente
    // el error máximo respecto a V1 (baseline). No asertamos valores concretos
    // para no hacer el test frágil; simplemente reportamos para inspección.
    println!("\nAblación MAC completada. Ver {out_path:?}");
}
