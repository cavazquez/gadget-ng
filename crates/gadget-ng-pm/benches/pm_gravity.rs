use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gadget_ng_core::{
    build_particles, GravitySolver, IcKind, InitialConditionsSection, RunConfig, SimulationSection,
    SolverKind, UnitsSection,
};
use gadget_ng_pm::PmSolver;

fn lattice_cfg(n: usize, grid: usize) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 1,
            softening: 0.04,
            gravitational_constant: 1.0,
            particle_count: n,
            box_size: 1.0,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Lattice,
        },
        gravity: gadget_ng_core::GravitySection {
            solver: SolverKind::Pm,
            pm_grid_size: grid,
            ..Default::default()
        },
        output: Default::default(),
        performance: Default::default(),
        timestep: Default::default(),
        cosmology: Default::default(),
        units: UnitsSection::default(),
        decomposition: Default::default(),
    }
}

fn bench_pm(c: &mut Criterion) {
    let mut group = c.benchmark_group("PM gravity");
    // Probamos con N = 64 (4×4×4) y grid 32³.
    for (n, grid) in [(64, 32), (512, 64)] {
        let cfg = lattice_cfg(n, grid);
        let particles = build_particles(&cfg).unwrap();
        let positions: Vec<_> = particles.iter().map(|p| p.position).collect();
        let masses: Vec<_> = particles.iter().map(|p| p.mass).collect();
        let indices: Vec<_> = (0..n).collect();
        let eps2 = cfg.softening_squared();
        let g = cfg.effective_g();
        let solver = PmSolver {
            grid_size: grid,
            box_size: 1.0,
        };
        let mut acc = vec![gadget_ng_core::Vec3::zero(); n];

        group.bench_with_input(BenchmarkId::new(format!("grid{grid}"), n), &n, |b, _| {
            b.iter(|| {
                solver.accelerations_for_indices(&positions, &masses, eps2, g, &indices, &mut acc);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pm);
criterion_main!(benches);
