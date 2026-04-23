//! Paridad numérica: aceleraciones locales por bloque de `global_id` coinciden con cálculo global.
use gadget_ng_core::{
    build_particles, build_particles_for_gid_range, DirectGravity, GravitySolver, IcKind,
    InitialConditionsSection, RunConfig, SimulationSection, Vec3,
};

fn small_lattice_cfg() -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 1,
            softening: 0.03,
            physical_softening: false,
            gravitational_constant: 1.0,
            particle_count: 27,
            box_size: 1.0,
            seed: 1,
            integrator: Default::default(),
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
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(), reionization: Default::default(),
    }
}

#[test]
fn split_gid_range_matches_full_accelerations() {
    let cfg = small_lattice_cfg();
    let n = cfg.simulation.particle_count;
    let full = build_particles(&cfg).expect("ic");
    let eps2 = cfg.softening_squared();
    let g = cfg.simulation.gravitational_constant;
    let direct = DirectGravity;
    let pos: Vec<Vec3> = full.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = full.iter().map(|p| p.mass).collect();
    let mut acc_full = vec![Vec3::zero(); n];
    let idx_full: Vec<usize> = (0..n).collect();
    direct.accelerations_for_indices(&pos, &mass, eps2, g, &idx_full, &mut acc_full);

    let lo = 5usize;
    let hi = 14usize;
    let local = build_particles_for_gid_range(&cfg, lo, hi).expect("ic");
    let mut acc_loc = vec![Vec3::zero(); local.len()];
    let gids: Vec<usize> = local.iter().map(|p| p.global_id).collect();
    direct.accelerations_for_indices(&pos, &mass, eps2, g, &gids, &mut acc_loc);
    for (p, a) in local.iter().zip(acc_loc.iter()) {
        let af = acc_full[p.global_id];
        let d = (*a - af).norm();
        assert!(d < 1e-15, "gid {} mismatch d={}", p.global_id, d);
    }
}
