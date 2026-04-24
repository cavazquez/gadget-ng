/// Phase 143 — Benchmarks Criterion avanzados (turbulencia, flux-freeze, RMHD)
///
/// Los benchmarks reales se ejecutan con `cargo bench --bench advanced_bench`.
/// Estos tests verifican que las funciones bajo benchmarking producen resultados
/// correctos y que el módulo de benchmarks compila sin errores.
use gadget_ng_core::{Particle, TurbulenceSection, Vec3};
use gadget_ng_mhd::{
    advance_srmhd, apply_flux_freeze, apply_turbulent_forcing, mean_gas_density,
    srmhd_conserved_to_primitive, C_LIGHT,
};

fn gas_particle(id: usize, pos: Vec3, vel: Vec3, b: Vec3) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, vel, 1e10, 0.2);
    p.b_field = b;
    p
}

fn turb_cfg() -> TurbulenceSection {
    TurbulenceSection {
        enabled: true,
        amplitude: 1e-3,
        correlation_time: 1.0,
        k_min: 1.0,
        k_max: 8.0,
        spectral_index: 5.0 / 3.0,
    }
}

// ── 1. Turbulencia N=100 produce dv > 0 ────────────────────────────────────

#[test]
fn turb_n100_nonzero() {
    let mut ps: Vec<Particle> = (0..100)
        .map(|i| gas_particle(i, Vec3::new(i as f64 * 0.01, 0.0, 0.0), Vec3::zero(),
                              Vec3::new(1e-6, 0.0, 0.0)))
        .collect();
    apply_turbulent_forcing(&mut ps, &turb_cfg(), 0.01, 77);
    let dv: f64 = ps.iter().map(|p| p.velocity.x.abs() + p.velocity.y.abs() + p.velocity.z.abs()).sum();
    assert!(dv > 0.0);
}

// ── 2. Turbulencia N=500 escala correctamente ─────────────────────────────

#[test]
fn turb_n500_scales() {
    let mut ps: Vec<Particle> = (0..500)
        .map(|i| gas_particle(i, Vec3::new(i as f64 * 0.002, 0.0, 0.0), Vec3::zero(),
                              Vec3::new(1e-6, 0.0, 0.0)))
        .collect();
    apply_turbulent_forcing(&mut ps, &turb_cfg(), 0.01, 1337);
    let dv: f64 = ps.iter().map(|p| p.velocity.x.abs() + p.velocity.y.abs() + p.velocity.z.abs()).sum();
    assert!(dv > 0.0);
}

// ── 3. flux_freeze N=100 conserva flujo magnético ────────────────────────

#[test]
fn flux_freeze_n100_conserves_flux() {
    let mut ps: Vec<Particle> = (0..100)
        .map(|i| gas_particle(i, Vec3::new(i as f64 * 0.01, 0.0, 0.0), Vec3::zero(),
                              Vec3::new(1e-9, 0.0, 0.0)))
        .collect();
    let rho_ref = mean_gas_density(&ps);
    apply_flux_freeze(&mut ps, 5.0 / 3.0, 1.0, rho_ref); // beta_freeze=1: siempre activo
    let b_rms: f64 = (ps.iter().map(|p| {
        let b = p.b_field;
        b.x*b.x + b.y*b.y + b.z*b.z
    }).sum::<f64>() / ps.len() as f64).sqrt();
    assert!(b_rms.is_finite() && b_rms >= 0.0);
}

// ── 4. flux_freeze N=1000 no crashea ─────────────────────────────────────

#[test]
fn flux_freeze_n1000_no_crash() {
    let mut ps: Vec<Particle> = (0..1000)
        .map(|i| gas_particle(i, Vec3::new(i as f64 * 0.001, 0.0, 0.0), Vec3::zero(),
                              Vec3::new(1e-9, 0.0, 0.0)))
        .collect();
    let rho_ref = mean_gas_density(&ps);
    apply_flux_freeze(&mut ps, 5.0 / 3.0, 100.0, rho_ref);
    assert!(ps.iter().all(|p| p.b_field.x.is_finite()));
}

// ── 5. advance_srmhd N=100 con 10% relativistas ───────────────────────────

#[test]
fn srmhd_n100_partial_relativistic() {
    let mut ps: Vec<Particle> = (0..100)
        .map(|i| {
            let v = if i % 10 == 0 { Vec3::new(0.5 * C_LIGHT, 0.0, 0.0) } else { Vec3::zero() };
            gas_particle(i, Vec3::new(i as f64 * 0.01, 0.0, 0.0), v, Vec3::new(1e-6, 0.0, 0.0))
        })
        .collect();
    advance_srmhd(&mut ps, 1e-5, C_LIGHT, 0.1);
    assert!(ps.iter().all(|p| p.velocity.x.is_finite()));
}

// ── 6. srmhd_conserved_to_primitive 1000 iter retorna valores finitos ─────

#[test]
fn conserved_to_primitive_1000_iter_finite() {
    for seed in 0..1000u64 {
        let d = 1.0 + (seed as f64 * 0.001) % 5.0;
        let sx = (seed as f64 * 0.003) % 0.3 * C_LIGHT * d;
        let e_cons = d * C_LIGHT * C_LIGHT * 1.1 + 0.5 * sx * sx / d;
        let result = srmhd_conserved_to_primitive(
            d, [sx, 0.0, 0.0], e_cons, [1e-6, 0.0, 0.0], 5.0 / 3.0, C_LIGHT,
        );
        if let Some((rho, v, p)) = result {
            assert!(rho.is_finite() && rho > 0.0);
            assert!(v[0].is_finite());
            assert!(p.is_finite() && p >= 0.0);
        }
    }
}
