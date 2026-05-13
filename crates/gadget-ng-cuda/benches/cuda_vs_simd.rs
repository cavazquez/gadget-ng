use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gadget_ng_core::{DirectGravity, GravitySolver, SimdDirectGravity, Vec3};
use gadget_ng_cuda::CudaDirectGravity;
use std::hint::black_box;

fn make_particles(n: usize) -> (Vec<Vec3>, Vec<f64>, Vec<[f32; 3]>, Vec<f32>) {
    let positions64: Vec<Vec3> = (0..n)
        .map(|i| {
            let t = i as f64 * 0.37;
            Vec3::new(t.sin() * 10.0, t.cos() * 10.0, (t * 0.5).sin() * 10.0)
        })
        .collect();
    let masses64 = vec![1.0_f64 / n as f64; n];
    let positions32 = positions64
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();
    let masses32 = masses64.iter().map(|&m| m as f32).collect();
    (positions64, masses64, positions32, masses32)
}

fn bench_direct_cuda_vs_simd(c: &mut Criterion) {
    let cuda = CudaDirectGravity::try_new_checked(0.1).ok();
    if cuda.is_none() {
        eprintln!("cuda_vs_simd: CUDA no disponible; se omite la serie cuda");
    }

    let mut group = c.benchmark_group("cuda_vs_simd_direct");
    group.sample_size(10);

    for n in [512usize, 1024, 2048] {
        let (positions64, masses64, positions32, masses32) = make_particles(n);
        let indices: Vec<usize> = (0..n).collect();
        let eps2 = 0.01_f64;
        let g = 1.0_f64;
        let mut out = vec![Vec3::zero(); n];

        group.bench_with_input(BenchmarkId::new("cpu_serial", n), &n, |b, _| {
            b.iter(|| {
                DirectGravity.accelerations_for_indices(
                    black_box(&positions64),
                    black_box(&masses64),
                    black_box(eps2),
                    black_box(g),
                    black_box(&indices),
                    black_box(&mut out),
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("cpu_simd", n), &n, |b, _| {
            b.iter(|| {
                SimdDirectGravity.accelerations_for_indices(
                    black_box(&positions64),
                    black_box(&masses64),
                    black_box(eps2),
                    black_box(g),
                    black_box(&indices),
                    black_box(&mut out),
                );
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    black_box(cuda.compute(black_box(&positions32), black_box(&masses32)));
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_direct_cuda_vs_simd);
criterion_main!(benches);
