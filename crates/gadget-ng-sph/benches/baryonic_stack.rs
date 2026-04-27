/// Benchmark del stack bariónico completo (Phase 120).
///
/// Mide el overhead combinado de ISM, CRs y vientos estelares sobre 1000 partículas.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use gadget_ng_core::Particle;
use gadget_ng_core::{CrSection, FeedbackSection, IsmSection, Vec3};
use gadget_ng_sph::{diffuse_cr, inject_cr_from_sn, update_ism_phases};

fn make_gas_particles(n: usize) -> Vec<Particle> {
    (0..n)
        .map(|i| {
            let x = (i % 10) as f64 * 0.1;
            let y = ((i / 10) % 10) as f64 * 0.1;
            let z = (i / 100) as f64 * 0.1;
            let mut p = Particle::new_gas(i, 1.0, Vec3::new(x, y, z), Vec3::zero(), 1.0, 0.1);
            p.metallicity = 0.01;
            p
        })
        .collect()
}

fn bench_ism_update(c: &mut Criterion) {
    let mut particles = make_gas_particles(1000);
    let sfr = vec![0.5_f64; 1000];
    let cfg = IsmSection {
        enabled: true,
        q_star: 2.5,
        f_cold: 0.5,
    };
    c.bench_function("update_ism_phases_1000", |b| {
        b.iter(|| {
            update_ism_phases(
                black_box(&mut particles),
                black_box(&sfr),
                black_box(0.1),
                black_box(&cfg),
                black_box(0.01),
            )
        });
    });
}

fn bench_cr_injection(c: &mut Criterion) {
    let mut particles = make_gas_particles(1000);
    let sfr = vec![0.5_f64; 1000];
    c.bench_function("inject_cr_from_sn_1000", |b| {
        b.iter(|| {
            inject_cr_from_sn(
                black_box(&mut particles),
                black_box(&sfr),
                black_box(0.1),
                black_box(0.01),
            )
        });
    });
}

fn bench_cr_diffusion(c: &mut Criterion) {
    let mut particles = make_gas_particles(100); // difusión es O(N²), usar N más pequeño
    for p in &mut particles {
        p.cr_energy = 1.0;
    }
    c.bench_function("diffuse_cr_100", |b| {
        b.iter(|| diffuse_cr(black_box(&mut particles), black_box(3e-3), black_box(0.01)));
    });
}

criterion_group!(
    benches,
    bench_ism_update,
    bench_cr_injection,
    bench_cr_diffusion
);
criterion_main!(benches);
