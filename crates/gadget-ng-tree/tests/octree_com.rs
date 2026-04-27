//! Masa y centro de masa de la raíz del octree coinciden con el sistema de partículas.
use gadget_ng_core::{
    IcKind, InitialConditionsSection, RunConfig, SimulationSection, Vec3, build_particles,
};
use gadget_ng_tree::Octree;

#[test]
fn root_mass_and_com_match_particles() {
    let cfg = RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 1,
            softening: 0.02,
            physical_softening: false,
            gravitational_constant: 1.0,
            particle_count: 8,
            box_size: 1.0,
            seed: 3,
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
    let parts = build_particles(&cfg).expect("ic");
    let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let tree = Octree::build(&pos, &mass);
    let root = &tree.nodes[tree.root as usize];
    let m_tot: f64 = mass.iter().sum();
    assert!((root.mass - m_tot).abs() < 1e-14);
    let mut com = Vec3::zero();
    for p in &parts {
        com += p.position * p.mass;
    }
    com /= m_tot;
    assert!(
        (root.com - com).norm() < 1e-13,
        "com arbol {:?} vs analitico {:?}",
        root.com,
        com
    );
}
