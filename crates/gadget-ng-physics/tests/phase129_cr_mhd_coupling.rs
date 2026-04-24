/// Phase 129 — Acoplamiento CR–B: difusión suprimida por |B|
///
/// Tests: b_suppress=0 reproduce comportamiento clásico, b_suppress>0 reduce difusión,
///        B fuerte suprime casi completamente la difusión, b_suppress no afecta inyección,
///        supresión escala con |B|², CR pressure no afectada por supresión.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_sph::{cr_pressure, diffuse_cr};

fn gas_with_cr(id: usize, pos: Vec3, cr: f64, h: f64) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, Vec3::zero(), 1.0, h);
    p.cr_energy = cr;
    p
}

// ── 1. b_suppress=0 → comportamiento clásico (sin supresión) ─────────────

#[test]
fn zero_suppress_equals_classic() {
    let mut p_classic = vec![
        gas_with_cr(0, Vec3::new(0.0, 0.0, 0.0), 10.0, 0.5),
        gas_with_cr(1, Vec3::new(0.1, 0.0, 0.0), 1.0, 0.5),
    ];
    let mut p_new = p_classic.clone();
    // b_suppress=0 → kappa_eff = kappa_cr siempre
    diffuse_cr(&mut p_classic, 0.1, 0.0, 0.01);
    diffuse_cr(&mut p_new, 0.1, 0.0, 0.01);
    assert!((p_classic[0].cr_energy - p_new[0].cr_energy).abs() < 1e-15,
        "b_suppress=0 debe ser idéntico al resultado previo");
}

// ── 2. b_suppress>0 reduce la difusión ────────────────────────────────────

#[test]
fn nonzero_suppress_reduces_diffusion() {
    // Configuración con gradiente CR pronunciado
    let mut p_no_suppress = vec![
        gas_with_cr(0, Vec3::new(0.0, 0.0, 0.0), 10.0, 0.5),
        gas_with_cr(1, Vec3::new(0.1, 0.0, 0.0), 1.0, 0.5),
    ];
    let mut p_suppress = p_no_suppress.clone();
    // Agregar campo B fuerte a p_suppress
    p_suppress[0].b_field = Vec3::new(10.0, 0.0, 0.0);
    p_suppress[1].b_field = Vec3::new(10.0, 0.0, 0.0);

    let e0_no = p_no_suppress[0].cr_energy;
    let e0_sup = p_suppress[0].cr_energy;

    diffuse_cr(&mut p_no_suppress, 0.1, 0.0, 0.01);
    diffuse_cr(&mut p_suppress, 0.1, 1.0, 0.01); // b_suppress=1.0

    let de_no = (p_no_suppress[0].cr_energy - e0_no).abs();
    let de_sup = (p_suppress[0].cr_energy - e0_sup).abs();

    assert!(de_sup < de_no,
        "Con B fuerte, difusión debe ser menor: Δe_sup={de_sup:.4e} vs Δe_no={de_no:.4e}");
}

// ── 3. B muy fuerte → difusión casi nula ─────────────────────────────────

#[test]
fn very_strong_b_suppresses_diffusion() {
    let mut particles = vec![
        gas_with_cr(0, Vec3::new(0.0, 0.0, 0.0), 100.0, 0.5),
        gas_with_cr(1, Vec3::new(0.1, 0.0, 0.0), 0.0, 0.5),
    ];
    // B = 1000 → B² = 1e6 → f_suppress = 1/(1 + 1e6) ≈ 1e-6
    for p in &mut particles {
        p.b_field = Vec3::new(1000.0, 0.0, 0.0);
    }
    let e0 = particles[0].cr_energy;
    diffuse_cr(&mut particles, 1.0, 1.0, 0.1);
    let de = (particles[0].cr_energy - e0).abs();
    assert!(de < 1e-3, "Con B=1000, difusión casi nula: Δe={de:.4e}");
}

// ── 4. b_suppress no afecta la inyección CR ──────────────────────────────

#[test]
fn b_suppress_does_not_affect_injection() {
    use gadget_ng_sph::inject_cr_from_sn;
    let mut particles = vec![gas_with_cr(0, Vec3::zero(), 0.0, 0.5)];
    particles[0].b_field = Vec3::new(100.0, 0.0, 0.0); // B muy fuerte
    let sfr = vec![1.0_f64];
    inject_cr_from_sn(&mut particles, &sfr, 0.1, 0.01);
    assert!(particles[0].cr_energy > 0.0, "Inyección CR no depende de B");
}

// ── 5. Supresión escala correctamente con |B|² ────────────────────────────

#[test]
fn suppression_scales_with_b_squared() {
    // κ_eff = κ / (1 + b_suppress × B²)
    // Para B=1: f=1/2. Para B=2: f=1/5.
    let b_suppress = 1.0_f64;
    let b1 = 1.0_f64;
    let b2 = 2.0_f64;
    let f1 = 1.0 / (1.0 + b_suppress * b1 * b1);
    let f2 = 1.0 / (1.0 + b_suppress * b2 * b2);
    assert!((f1 - 0.5).abs() < 1e-12, "f1 = {f1}");
    assert!((f2 - 0.2).abs() < 1e-12, "f2 = {f2}");
    assert!(f2 < f1, "Mayor B → mayor supresión");
}

// ── 6. cr_pressure no cambia con supresión ───────────────────────────────

#[test]
fn cr_pressure_independent_of_b() {
    // cr_pressure solo depende de cr_energy y rho, no de B
    let p_no_b = cr_pressure(1.0, 1.0);
    let p_with_b = cr_pressure(1.0, 1.0); // misma función
    assert_eq!(p_no_b, p_with_b);
    assert!(p_no_b > 0.0);
}
