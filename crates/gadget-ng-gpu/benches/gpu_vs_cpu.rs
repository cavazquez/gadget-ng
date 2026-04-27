//! Benchmark: GPU (wgpu WGSL) vs gravedad directa CPU.
//!
//! Mide el tiempo de cómputo de la gravedad de Plummer directa (O(N²)) en función
//! de N tanto en GPU (compute shader via wgpu) como en CPU (Rust single-thread).
//!
//! Ejecutar con:
//!   cargo bench -p gadget-ng-gpu --bench gpu_vs_cpu
//!
//! Los resultados se guardan en `target/criterion/` en formato HTML.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use gadget_ng_gpu::GpuDirectGravity;

// ── Gravedad directa CPU (referencia) ────────────────────────────────────────

fn direct_gravity_cpu(positions: &[[f32; 3]], masses: &[f32], eps2: f32, g: f32) -> Vec<[f32; 3]> {
    let n = positions.len();
    (0..n)
        .map(|qi| {
            let [px, py, pz] = positions[qi];
            let (mut ax, mut ay, mut az) = (0.0f32, 0.0, 0.0);
            for (j, &[jx, jy, jz]) in positions.iter().enumerate() {
                if j == qi {
                    continue;
                }
                let dx = jx - px;
                let dy = jy - py;
                let dz = jz - pz;
                let r2soft = dx * dx + dy * dy + dz * dz + eps2;
                let r3inv = r2soft.sqrt().recip() / r2soft;
                ax += masses[j] * dx * r3inv;
                ay += masses[j] * dy * r3inv;
                az += masses[j] * dz * r3inv;
            }
            [g * ax, g * ay, g * az]
        })
        .collect()
}

// ── Helpers para generar partículas de prueba ─────────────────────────────────

fn make_particles(n: usize) -> (Vec<[f32; 3]>, Vec<f32>) {
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let t = i as f32 * 0.37;
            [t.sin() * 10.0, t.cos() * 10.0, (t * 0.5).sin() * 10.0]
        })
        .collect();
    let masses = vec![1.0f32 / n as f32; n];
    (positions, masses)
}

// ── Benchmarks CPU ────────────────────────────────────────────────────────────

fn bench_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gravity_cpu");
    group.sample_size(10);

    for n in [100usize, 250, 500, 1000] {
        let (positions, masses) = make_particles(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(direct_gravity_cpu(
                    black_box(&positions),
                    black_box(&masses),
                    0.01,
                    1.0,
                ))
            });
        });
    }
    group.finish();
}

// ── Benchmarks GPU ────────────────────────────────────────────────────────────

fn bench_gpu(c: &mut Criterion) {
    let Some(gpu) = GpuDirectGravity::try_new() else {
        eprintln!("bench_gpu: SKIP — sin GPU disponible");
        return;
    };

    let mut group = c.benchmark_group("gravity_gpu");
    group.sample_size(10);

    for n in [100usize, 250, 500, 1000] {
        let (positions, masses) = make_particles(n);
        let flat_pos: Vec<f32> = positions.iter().flat_map(|p| *p).collect();
        let idx: Vec<u32> = (0..n as u32).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(gpu.compute_accelerations_raw(
                    black_box(&flat_pos),
                    black_box(&masses),
                    black_box(&idx),
                    0.01,
                    1.0,
                ))
            });
        });
    }
    group.finish();
}

// ── Benchmark comparativo (CPU vs GPU en el mismo grupo) ──────────────────────

fn bench_comparison(c: &mut Criterion) {
    let gpu_opt = GpuDirectGravity::try_new();
    let mut group = c.benchmark_group("gravity_comparison");
    group.sample_size(10);

    for n in [100usize, 500, 1000] {
        let (positions, masses) = make_particles(n);

        // CPU
        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |b, _| {
            b.iter(|| {
                black_box(direct_gravity_cpu(
                    black_box(&positions),
                    black_box(&masses),
                    0.01,
                    1.0,
                ))
            });
        });

        // GPU (si disponible)
        if let Some(ref gpu) = gpu_opt {
            let flat_pos: Vec<f32> = positions.iter().flat_map(|p| *p).collect();
            let idx: Vec<u32> = (0..n as u32).collect();
            group.bench_with_input(BenchmarkId::new("gpu", n), &n, |b, _| {
                b.iter(|| {
                    black_box(gpu.compute_accelerations_raw(
                        black_box(&flat_pos),
                        black_box(&masses),
                        black_box(&idx),
                        0.01,
                        1.0,
                    ))
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_cpu, bench_gpu, bench_comparison);
criterion_main!(benches);
