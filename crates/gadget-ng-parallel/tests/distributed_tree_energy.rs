//! Test de integración: árbol distribuido (serial) conserva energía cinética acotada.
//!
//! En modo serial, `exchange_halos_by_x` devuelve vacío (no hay vecinos).
//! La fuerza se calcula solo con las partículas locales, equivalente al Barnes-Hut estándar.
//! El test verifica que la energía cinética no explote en una integración corta.

use gadget_ng_core::{
    build_particles, IcKind, InitialConditionsSection, RunConfig, SimulationSection, Vec3,
};
use gadget_ng_integrators::leapfrog_kdk_step;
use gadget_ng_parallel::{ParallelRuntime, SerialRuntime, SlabDecomposition};
use gadget_ng_tree::Octree;

fn kinetic(parts: &[gadget_ng_core::Particle]) -> f64 {
    parts
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity))
        .sum()
}

/// Calcula fuerzas usando árbol local + halos (vacíos en serial).
fn force_dtree(
    parts: &[gadget_ng_core::Particle],
    halos: &[gadget_ng_core::Particle],
    theta: f64,
    g: f64,
    eps2: f64,
    out: &mut [Vec3],
) {
    let all_pos: Vec<Vec3> = parts
        .iter()
        .chain(halos.iter())
        .map(|p| p.position)
        .collect();
    let all_mass: Vec<f64> = parts.iter().chain(halos.iter()).map(|p| p.mass).collect();
    let tree = Octree::build(&all_pos, &all_mass);
    for (li, acc_out) in out.iter_mut().enumerate() {
        *acc_out = tree.walk_accel(parts[li].position, li, g, eps2, theta, &all_pos, &all_mass);
    }
}

#[test]
fn distributed_tree_serial_kinetic_bounded() {
    let cfg = RunConfig {
        simulation: SimulationSection {
            dt: 0.0005,
            num_steps: 40,
            softening: 0.04,
            gravitational_constant: 1.0,
            particle_count: 27,
            box_size: 1.0,
            seed: 99,
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
    };

    let rt = SerialRuntime;
    let eps2 = cfg.softening_squared();
    let g = cfg.simulation.gravitational_constant;
    let dt = cfg.simulation.dt;
    let theta = cfg.gravity.theta; // default 0.5

    let mut parts = build_particles(&cfg).expect("ic");
    let mut scratch = vec![Vec3::zero(); parts.len()];

    let ke0 = kinetic(&parts);
    assert!(ke0.is_finite() && ke0 >= 0.0);
    let mut ke_max = ke0;

    for _ in 0..cfg.simulation.num_steps {
        // Re-calcular bounds y decomposición cada paso
        let x_lo_loc = parts
            .iter()
            .map(|p| p.position.x)
            .fold(f64::INFINITY, f64::min);
        let x_hi_loc = parts
            .iter()
            .map(|p| p.position.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let x_lo = rt.allreduce_min_f64(x_lo_loc);
        let x_hi = rt.allreduce_max_f64(x_hi_loc);
        let decomp = SlabDecomposition::new(x_lo, x_hi, rt.size());
        let (my_x_lo, my_x_hi) = decomp.bounds(rt.rank());
        let hw = decomp.halo_width(0.5);

        // Dominio: serial → no-op
        rt.exchange_domain_by_x(&mut parts, my_x_lo, my_x_hi);

        leapfrog_kdk_step(&mut parts, dt, &mut scratch, |p, acc| {
            let halos = rt.exchange_halos_by_x(p, my_x_lo, my_x_hi, hw);
            force_dtree(p, &halos, theta, g, eps2, acc);
        });

        ke_max = ke_max.max(kinetic(&parts));
    }

    // La energía cinética no debe divergir (tolerar hasta ×10 el valor inicial).
    assert!(
        ke_max < ke0 * 10.0 + 1.0,
        "ke_max={ke_max:.4} demasiado alto (ke0={ke0:.4})"
    );
}

#[test]
fn slab_decomposition_covers_all_particles() {
    let decomp = SlabDecomposition::new(0.0, 3.0, 3);
    let xs = [0.1, 1.1, 2.1, 0.9, 1.9, 2.9];
    let expected_ranks = [0, 1, 2, 0, 1, 2];
    for (x, &er) in xs.iter().zip(expected_ranks.iter()) {
        assert_eq!(
            decomp.rank_for_x(*x),
            er,
            "x={x} → rank {} (esperado {er})",
            decomp.rank_for_x(*x)
        );
    }
}

#[test]
fn halo_exchange_serial_returns_empty() {
    let cfg = RunConfig {
        simulation: SimulationSection {
            particle_count: 8,
            box_size: 1.0,
            seed: 1,
            dt: 0.001,
            num_steps: 1,
            softening: 0.05,
            gravitational_constant: 1.0,
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
    };
    let parts = build_particles(&cfg).expect("ic");
    let rt = SerialRuntime;
    let halos = rt.exchange_halos_by_x(&parts, -0.5, 0.5, 0.1);
    assert!(halos.is_empty(), "serial no debe tener halos");
}
