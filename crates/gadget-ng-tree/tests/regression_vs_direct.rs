//! Regresión Barnes-Hut vs gravedad directa; precisión con theta > 0.
use approx::assert_relative_eq;
use gadget_ng_core::{
    build_particles, DirectGravity, GravitySolver, IcKind, InitialConditionsSection, RunConfig,
    SimulationSection, Vec3,
};
use gadget_ng_tree::BarnesHutGravity;

#[test]
fn two_particles_bh_matches_direct_theta_half() {
    let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
    let mass = vec![1.0_f64, 1.0_f64];
    let eps2 = 0.01_f64 * 0.01;
    let g = 1.0_f64;
    let idx = vec![0usize, 1];
    let mut acc_d = vec![Vec3::zero(); 2];
    let mut acc_bh = vec![Vec3::zero(); 2];
    DirectGravity.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut acc_d);
    BarnesHutGravity { theta: 0.5 }.accelerations_for_indices(
        &pos,
        &mass,
        eps2,
        g,
        &idx,
        &mut acc_bh,
    );
    for i in 0..2 {
        assert_relative_eq!(acc_bh[i].x, acc_d[i].x, epsilon = 1e-10);
        assert_relative_eq!(acc_bh[i].y, acc_d[i].y, epsilon = 1e-10);
        assert_relative_eq!(acc_bh[i].z, acc_d[i].z, epsilon = 1e-10);
    }
}

fn lattice_cfg(n: usize, seed: u64) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 1,
            softening: 0.03,
            gravitational_constant: 1.0,
            particle_count: n,
            box_size: 1.0,
            seed,
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Lattice,
        },
        output: Default::default(),
        gravity: Default::default(),
        performance: Default::default(),
        timestep: Default::default(),
        cosmology: Default::default(),
        units: Default::default(),
    }
}

#[test]
fn barnes_hut_theta_zero_matches_direct() {
    let cfg = lattice_cfg(27, 11);
    let parts = build_particles(&cfg).expect("ic");
    let n = parts.len();
    let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let eps2 = cfg.softening_squared();
    let g = cfg.simulation.gravitational_constant;
    let idx: Vec<usize> = (0..n).collect();
    let mut acc_d = vec![Vec3::zero(); n];
    let mut acc_bh = vec![Vec3::zero(); n];
    DirectGravity.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut acc_d);
    BarnesHutGravity { theta: 0.0 }.accelerations_for_indices(
        &pos,
        &mass,
        eps2,
        g,
        &idx,
        &mut acc_bh,
    );
    for i in 0..n {
        assert_relative_eq!(acc_bh[i].x, acc_d[i].x, epsilon = 1e-11);
        assert_relative_eq!(acc_bh[i].y, acc_d[i].y, epsilon = 1e-11);
        assert_relative_eq!(acc_bh[i].z, acc_d[i].z, epsilon = 1e-11);
    }
}

#[test]
fn barnes_hut_theta_half_mean_relative_error_small_on_strong_accel() {
    // En un lattice casi simétrico, muchas partículas tienen |a_direct|≈0 por cancelación;
    // el error relativo carece de sentido ahí. Comparamos solo donde |a_direct| es notable.
    let cfg = lattice_cfg(125, 13);
    let parts = build_particles(&cfg).expect("ic");
    let n = parts.len();
    let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let eps2 = cfg.softening_squared();
    let g = cfg.simulation.gravitational_constant;
    let idx: Vec<usize> = (0..n).collect();
    let mut acc_d = vec![Vec3::zero(); n];
    let mut acc_bh = vec![Vec3::zero(); n];
    DirectGravity.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut acc_d);
    BarnesHutGravity { theta: 0.5 }.accelerations_for_indices(
        &pos,
        &mass,
        eps2,
        g,
        &idx,
        &mut acc_bh,
    );
    let max_ad = acc_d.iter().map(|a| a.norm()).fold(0.0_f64, f64::max);
    let thresh = (1e-9_f64).max(0.01 * max_ad);
    let mut sum_rel = 0.0_f64;
    let mut count = 0usize;
    let mut max_rel = 0.0_f64;
    for i in 0..n {
        let nd = acc_d[i].norm();
        if nd < thresh {
            continue;
        }
        let rel = (acc_bh[i] - acc_d[i]).norm() / nd;
        sum_rel += rel;
        max_rel = max_rel.max(rel);
        count += 1;
    }
    assert!(
        count >= 10,
        "muy pocas partículas con |a_direct|>thresh={thresh}"
    );
    let mean_rel = sum_rel / count as f64;
    // Monopolo puro + lattice casi simétrico: con theta=0.5 el error medio típico ~2–3 % en este subset.
    assert!(
        mean_rel < 0.03,
        "error relativo medio {mean_rel} >= 3% (theta=0.5, N=125, subset |a_direct|>1% del máximo)"
    );
    assert!(
        max_rel < 0.12,
        "error relativo máximo {max_rel} >= 12% en el mismo subset"
    );
}

/// Regresión más estricta con `theta = 0.25`: error medio relativo frente a directo bajo 1 %.
#[test]
fn barnes_hut_theta_quarter_mean_relative_error_under_1pct() {
    let cfg = lattice_cfg(125, 13);
    let parts = build_particles(&cfg).expect("ic");
    let n = parts.len();
    let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let eps2 = cfg.softening_squared();
    let g = cfg.simulation.gravitational_constant;
    let idx: Vec<usize> = (0..n).collect();
    let mut acc_d = vec![Vec3::zero(); n];
    let mut acc_bh = vec![Vec3::zero(); n];
    DirectGravity.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut acc_d);
    BarnesHutGravity { theta: 0.25 }.accelerations_for_indices(
        &pos,
        &mass,
        eps2,
        g,
        &idx,
        &mut acc_bh,
    );
    let max_ad = acc_d.iter().map(|a| a.norm()).fold(0.0_f64, f64::max);
    let thresh = (1e-9_f64).max(0.01 * max_ad);
    let mut sum_rel = 0.0_f64;
    let mut count = 0usize;
    let mut max_rel = 0.0_f64;
    for i in 0..n {
        let nd = acc_d[i].norm();
        if nd < thresh {
            continue;
        }
        let rel = (acc_bh[i] - acc_d[i]).norm() / nd;
        sum_rel += rel;
        max_rel = max_rel.max(rel);
        count += 1;
    }
    assert!(
        count >= 10,
        "muy pocas partículas con |a_direct|>thresh={thresh}"
    );
    let mean_rel = sum_rel / count as f64;
    assert!(
        mean_rel < 0.01,
        "error relativo medio {mean_rel} >= 1% (theta=0.25, N=125)"
    );
    assert!(
        max_rel < 0.05,
        "error relativo máximo {max_rel} >= 5% (theta=0.25, N=125)"
    );
}
