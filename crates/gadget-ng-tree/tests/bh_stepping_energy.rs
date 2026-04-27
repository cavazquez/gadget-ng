//! Integración corta con Barnes-Hut: la energía cinética no explota (cota conservadora).
use gadget_ng_core::{
    IcKind, InitialConditionsSection, RunConfig, SimulationSection, Vec3,
    accelerations_all_particles, build_particles,
};
use gadget_ng_integrators::leapfrog_kdk_step;
use gadget_ng_tree::BarnesHutGravity;

fn kinetic(parts: &[gadget_ng_core::Particle]) -> f64 {
    parts
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity))
        .sum()
}

#[test]
fn stepping_with_barnes_hut_kinetic_bounded() {
    let cfg = RunConfig {
        simulation: SimulationSection {
            dt: 0.0005,
            num_steps: 50,
            softening: 0.04,
            physical_softening: false,
            gravitational_constant: 1.0,
            particle_count: 27,
            box_size: 1.0,
            seed: 21,
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
        rt: Default::default(),
        reionization: Default::default(),
        mhd: Default::default(),
        turbulence: Default::default(),
        two_fluid: Default::default(),
        sidm: Default::default(),
        modified_gravity: Default::default(),
    };
    let eps2 = cfg.softening_squared();
    let g = cfg.simulation.gravitational_constant;
    let dt = cfg.simulation.dt;
    let mut parts = build_particles(&cfg).expect("ic");
    let mut scratch = vec![Vec3::zero(); parts.len()];
    let bh = BarnesHutGravity {
        theta: 0.5,
        ..Default::default()
    };
    let ke0 = kinetic(&parts);
    assert!(ke0.is_finite() && ke0 >= 0.0);
    let mut ke_max = ke0;
    for _ in 0..cfg.simulation.num_steps {
        leapfrog_kdk_step(&mut parts, dt, &mut scratch, |p, acc| {
            accelerations_all_particles(&bh, p, eps2, g, acc);
        });
        let ke = kinetic(&parts);
        assert!(ke.is_finite() && ke >= 0.0);
        ke_max = ke_max.max(ke);
    }
    assert!(
        ke_max < 1e6,
        "energía cinética máxima {ke_max} no finita o demasiado grande"
    );
}
