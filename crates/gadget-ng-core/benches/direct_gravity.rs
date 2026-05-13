//! Benchmark Criterion: gravedad directa N² — serial vs Rayon vs SIMD.
//!
//! Comparación + gráfico: `bash scripts/bench_direct_cpu_plot.sh` genera
//! `runs/benchmarks/direct-cpu/direct_gravity_mean_times.{csv,png}`.
//!
//! Matriz de **features** (cada pasada compila lo que aplica):
//! - sin features → `direct_cpu/serial`
//! - `--features rayon` → + `direct_cpu/rayon_scalar_inner`
//! - `--features simd,rayon` → + SIMD monohilo y Rayon+SIMD (AVX2/512 en x86 si la CPU lo expone)
//!
//! Los tiers AVX2/AVX-512 forzados requieren x86 con esos flags; si no, el kernel cae a escalar.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gadget_ng_core::config::{
    CosmologySection, OutputSection, PerformanceSection, TimestepSection, UnitsSection,
};
use gadget_ng_core::{
    DirectGravity, GravitySection, GravitySolver, IcKind, InitialConditionsSection, RunConfig,
    SimulationSection, Vec3, build_particles,
};

fn lattice_cfg(n: usize) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 1,
            softening: 0.05,
            physical_softening: false,
            gravitational_constant: 1.0,
            particle_count: n,
            box_size: 1.0,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Lattice,
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection::default(),
        units: UnitsSection::default(),
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
        dark_matter: Default::default(),
    }
}

fn lattice_buffers(n: usize) -> (Vec<Vec3>, Vec<f64>, f64, f64, Vec<usize>, Vec<Vec3>) {
    let cfg = lattice_cfg(n);
    let parts = build_particles(&cfg).unwrap();
    let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let eps2 = cfg.softening_squared();
    let g = cfg.simulation.gravitational_constant;
    let idx: Vec<usize> = (0..n).collect();
    let out = vec![Vec3::zero(); n];
    (pos, mass, eps2, g, idx, out)
}

fn bench_gravity_solver<S: GravitySolver + Copy>(
    c: &mut Criterion,
    group_name: &str,
    solver: S,
    sizes: &[usize],
) {
    let mut group = c.benchmark_group(group_name);
    for &n in sizes {
        let (pos, mass, eps2, g, idx, mut out) = lattice_buffers(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                solver.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut out);
            });
        });
    }
    group.finish();
}

fn direct_gravity_cpu_tiers(c: &mut Criterion) {
    const SIZES: &[usize] = &[27, 125, 512];

    bench_gravity_solver(c, "direct_cpu/serial", DirectGravity, SIZES);

    #[cfg(all(feature = "rayon", not(feature = "simd")))]
    {
        use gadget_ng_core::RayonDirectGravity;
        bench_gravity_solver(
            c,
            "direct_cpu/rayon_scalar_inner",
            RayonDirectGravity,
            SIZES,
        );
    }

    #[cfg(feature = "simd")]
    {
        use gadget_ng_core::SimdDirectGravity;
        bench_gravity_solver(
            c,
            "direct_cpu/simd_serial_runtime",
            SimdDirectGravity,
            SIZES,
        );

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use gadget_ng_core::{GravSimdTier, SimdDirectGravityTier};
            use std::arch::is_x86_feature_detected;
            bench_gravity_solver(
                c,
                "direct_cpu/simd_serial_avx2",
                SimdDirectGravityTier(GravSimdTier::Avx2Fma),
                SIZES,
            );
            if is_x86_feature_detected!("avx512f") {
                bench_gravity_solver(
                    c,
                    "direct_cpu/simd_serial_avx512",
                    SimdDirectGravityTier(GravSimdTier::Avx512),
                    SIZES,
                );
            }
        }
    }

    #[cfg(all(feature = "rayon", feature = "simd"))]
    {
        use gadget_ng_core::RayonDirectGravity;
        bench_gravity_solver(
            c,
            "direct_cpu/rayon_simd_runtime",
            RayonDirectGravity,
            SIZES,
        );

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use gadget_ng_core::{GravSimdTier, RayonDirectGravitySimdTier};
            use std::arch::is_x86_feature_detected;
            bench_gravity_solver(
                c,
                "direct_cpu/rayon_simd_avx2",
                RayonDirectGravitySimdTier(GravSimdTier::Avx2Fma),
                SIZES,
            );
            if is_x86_feature_detected!("avx512f") {
                bench_gravity_solver(
                    c,
                    "direct_cpu/rayon_simd_avx512",
                    RayonDirectGravitySimdTier(GravSimdTier::Avx512),
                    SIZES,
                );
            }
        }
    }
}

criterion_group!(benches, direct_gravity_cpu_tiers);
criterion_main!(benches);
