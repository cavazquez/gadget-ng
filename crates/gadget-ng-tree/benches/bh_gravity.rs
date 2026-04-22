use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gadget_ng_core::{
    build_particles, GravitySolver, IcKind, InitialConditionsSection, RunConfig, SimulationSection,
    Vec3,
};
use gadget_ng_tree::{BarnesHutGravity, Octree};

fn lattice_cfg(n: usize) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 1,
            softening: 0.04,
            gravitational_constant: 1.0,
            particle_count: n,
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
    }
}

fn bench_bh_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bh_build");
    for n in [125usize, 512, 1000] {
        let cfg = lattice_cfg(n);
        let parts = build_particles(&cfg).unwrap();
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let _ = Octree::build(&pos, &mass);
            });
        });
    }
    group.finish();
}

fn bench_bh_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("bh_serial_theta0.5");
    for n in [125usize, 512] {
        let cfg = lattice_cfg(n);
        let parts = build_particles(&cfg).unwrap();
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let eps2 = cfg.softening_squared();
        let g = cfg.simulation.gravitational_constant;
        let idx: Vec<usize> = (0..n).collect();
        let mut out = vec![Vec3::zero(); n];
        let solver = BarnesHutGravity { theta: 0.5, ..Default::default() };
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                solver.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut out);
            });
        });
    }
    group.finish();
}

#[cfg(feature = "simd")]
fn bench_bh_rayon(c: &mut Criterion) {
    use gadget_ng_tree::RayonBarnesHutGravity;
    let mut group = c.benchmark_group("bh_rayon_theta0.5");
    for n in [125usize, 512] {
        let cfg = lattice_cfg(n);
        let parts = build_particles(&cfg).unwrap();
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let eps2 = cfg.softening_squared();
        let g = cfg.simulation.gravitational_constant;
        let idx: Vec<usize> = (0..n).collect();
        let mut out = vec![Vec3::zero(); n];
        let solver = RayonBarnesHutGravity { theta: 0.5, ..Default::default() };
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                solver.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut out);
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "simd"))]
fn bench_bh_rayon(_c: &mut Criterion) {}

criterion_group!(benches, bench_bh_build, bench_bh_serial, bench_bh_rayon);
criterion_main!(benches);
