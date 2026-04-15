use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gadget_ng_core::config::{OutputSection, PerformanceSection, TimestepSection};
use gadget_ng_core::{
    build_particles, DirectGravity, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, RunConfig, SimulationSection, Vec3,
};

fn lattice_cfg(n: usize) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 1,
            softening: 0.05,
            gravitational_constant: 1.0,
            particle_count: n,
            box_size: 1.0,
            seed: 42,
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Lattice,
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
    }
}

fn bench_direct_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_serial");
    for n in [27usize, 125, 512] {
        let cfg = lattice_cfg(n);
        let parts = build_particles(&cfg).unwrap();
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let eps2 = cfg.softening_squared();
        let g = cfg.simulation.gravitational_constant;
        let idx: Vec<usize> = (0..n).collect();
        let mut out = vec![Vec3::zero(); n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                DirectGravity.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut out);
            });
        });
    }
    group.finish();
}

#[cfg(feature = "simd")]
fn bench_direct_rayon(c: &mut Criterion) {
    use gadget_ng_core::RayonDirectGravity;
    let mut group = c.benchmark_group("direct_rayon");
    for n in [27usize, 125, 512] {
        let cfg = lattice_cfg(n);
        let parts = build_particles(&cfg).unwrap();
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let eps2 = cfg.softening_squared();
        let g = cfg.simulation.gravitational_constant;
        let idx: Vec<usize> = (0..n).collect();
        let mut out = vec![Vec3::zero(); n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                RayonDirectGravity.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut out);
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "simd"))]
fn bench_direct_rayon(_c: &mut Criterion) {}

#[cfg(feature = "simd")]
fn bench_direct_simd(c: &mut Criterion) {
    use gadget_ng_core::SimdDirectGravity;
    let mut group = c.benchmark_group("direct_simd");
    for n in [27usize, 125, 512] {
        let cfg = lattice_cfg(n);
        let parts = build_particles(&cfg).unwrap();
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let eps2 = cfg.softening_squared();
        let g = cfg.simulation.gravitational_constant;
        let idx: Vec<usize> = (0..n).collect();
        let mut out = vec![Vec3::zero(); n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                SimdDirectGravity.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut out);
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "simd"))]
fn bench_direct_simd(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_direct_serial,
    bench_direct_rayon,
    bench_direct_simd
);
criterion_main!(benches);
