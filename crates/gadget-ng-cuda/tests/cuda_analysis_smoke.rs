//! Tests smoke/parity para kernels CUDA de análisis in-situ.
//!
//! Verifica spin de halo, luminosidad galáctica y emisión X contra CPU.

use gadget_ng_analysis::{
    halo_spin::{SpinParams, halo_spin},
    luminosity::galaxy_luminosity,
    xray::total_xray_luminosity,
};
use gadget_ng_core::{Particle, ParticleType, Vec3};
use gadget_ng_cuda::CudaAnalysisSolver;

fn cuda_or_skip() -> Option<CudaAnalysisSolver> {
    match CudaAnalysisSolver::try_new_checked() {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("SKIP CudaAnalysisSolver: {e}");
            None
        }
    }
}

fn assert_close_rel(label: &str, got: f64, expected: f64, tol: f64) {
    let denom = expected.abs().max(1.0e-30);
    let rel = (got - expected).abs() / denom;
    assert!(
        rel <= tol,
        "{label}: got={got:.6e} expected={expected:.6e} rel={rel:.3e} tol={tol:.3e}"
    );
}

// ── Halo spin ─────────────────────────────────────────────────────────────

fn ring_halo(n: usize, r: f64, v: f64, m: f64) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>) {
    let mut pos = Vec::new();
    let mut vel = Vec::new();
    for i in 0..n {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        pos.push(Vec3::new(r * theta.cos(), r * theta.sin(), 0.0));
        vel.push(Vec3::new(-v * theta.sin(), v * theta.cos(), 0.0));
    }
    (pos, vel, vec![m; n])
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_analysis_halo_spin_matches_cpu() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let (pos, vel, mass) = ring_halo(64, 10.0, 5.0, 1e10);
    let params = SpinParams::default();
    let cpu = halo_spin(&pos, &vel, &mass, &params).unwrap();

    let mass_total: f64 = mass.iter().sum();
    let [cx, cy, cz] = cpu.pos_com;
    let [vcx, vcy, vcz] = cpu.vel_com;
    let [lx_gpu, ly_gpu, lz_gpu] = cuda
        .try_halo_spin(&pos, &vel, &mass, [cx, cy, cz], [vcx, vcy, vcz])
        .unwrap();

    let l_mag_gpu = (lx_gpu * lx_gpu + ly_gpu * ly_gpu + lz_gpu * lz_gpu).sqrt();

    // f32 precision: tolerance 1e-3 on |L|.
    assert_close_rel("l_mag", l_mag_gpu, cpu.l_mag, 1e-3);
    // L should point in +Z direction for CCW ring.
    assert!(lz_gpu > 0.0, "Lz GPU debe ser positivo: Lz={lz_gpu:.3e}");
    let _ = mass_total;
}

#[test]
fn cuda_analysis_solver_available_without_hardware() {
    let _ = CudaAnalysisSolver::try_new_checked();
}

// ── Galaxy luminosity ─────────────────────────────────────────────────────

fn stellar_particles(n: usize) -> Vec<Particle> {
    (0..n)
        .map(|i| {
            let mut p = Particle::new(i, 1e8, Vec3::new(i as f64 * 0.1, 0.0, 0.0), Vec3::zero());
            p.ptype = ParticleType::Star;
            p.stellar_age = 1.0 + (i % 5) as f64 * 0.5;
            p.metallicity = 0.01 + (i % 3) as f64 * 0.005;
            p
        })
        .collect()
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_analysis_luminosity_matches_cpu() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let particles = stellar_particles(256);
    let cpu = galaxy_luminosity(&particles);
    let (l_gpu, _bv_gpu, _gr_gpu, n_stars_gpu) = cuda.try_galaxy_luminosity(&particles).unwrap();

    assert_eq!(n_stars_gpu, cpu.n_stars, "n_stars debe coincidir");
    // f32 arithmetic: tolerance 1e-3.
    assert_close_rel("l_total", l_gpu, cpu.l_total, 1e-3);
}

// ── X-ray luminosity ──────────────────────────────────────────────────────

fn hot_gas_particles(n: usize, gamma: f64) -> Vec<Particle> {
    // Gas caliente: T ~ 1e7 K → u = T × kB/(mH μ) / (γ-1)
    // kB/(mH μ) ≈ 8.254e-3/0.6 (km/s)²/K
    let kb_over_mh_mu = 8.254e-3 / 0.6;
    let t_k = 1.0e7_f64;
    let u = t_k * kb_over_mh_mu / (gamma - 1.0);
    (0..n)
        .map(|i| {
            let mut p = Particle::new_gas(
                i,
                1e10,
                Vec3::new(i as f64 * 0.5, 0.0, 0.0),
                Vec3::zero(),
                u,
                0.5,
            );
            p.smoothing_length = 0.5;
            p
        })
        .collect()
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_analysis_xray_matches_cpu() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let gamma = 5.0 / 3.0;
    let particles = hot_gas_particles(256, gamma);
    let cpu = total_xray_luminosity(&particles, gamma);
    let gpu = cuda.try_xray_luminosity(&particles, gamma).unwrap();

    // Double precision reduction on GPU; tolerance 1e-5.
    assert_close_rel("xray_lum", gpu, cpu, 1e-5);
}
