//! Conservación aproximada del momento lineal total con gravedad pareada y leapfrog KDK.
use gadget_ng_core::{
    accelerations_all_particles, build_particles, DirectGravity, IcKind, InitialConditionsSection,
    RunConfig, SimulationSection, Vec3,
};
use gadget_ng_integrators::leapfrog_kdk_step;

fn lattice_config() -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.001,
            num_steps: 30,
            softening: 0.02,
            physical_softening: false,
            gravitational_constant: 1.0,
            particle_count: 8,
            box_size: 1.0,
            seed: 7,
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

fn total_momentum(parts: &[gadget_ng_core::Particle]) -> Vec3 {
    let mut p = Vec3::zero();
    for x in parts {
        p += x.velocity * x.mass;
    }
    p
}

#[test]
fn linear_momentum_stays_small_for_symmetric_lattice() {
    let cfg = lattice_config();
    let eps2 = cfg.softening_squared();
    let g = cfg.simulation.gravitational_constant;
    let dt = cfg.simulation.dt;
    let mut parts = build_particles(&cfg).expect("ic");
    let mut scratch = vec![Vec3::zero(); parts.len()];
    let direct = DirectGravity;
    let p0 = total_momentum(&parts);
    assert!(
        p0.norm() < 1e-14,
        "lattice simétrico debería tener momento inicial ~0, got {p0:?}"
    );
    for _ in 0..cfg.simulation.num_steps {
        leapfrog_kdk_step(&mut parts, dt, &mut scratch, |p, acc| {
            accelerations_all_particles(&direct, p, eps2, g, acc);
        });
    }
    let p1 = total_momentum(&parts);
    let tol = 5e-11_f64;
    assert!(
        p1.norm() < tol,
        "momento final {p1:?} debería permanecer acotado (tol={tol})"
    );
}
