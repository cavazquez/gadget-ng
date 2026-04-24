/// Phase 117 — Rayos cósmicos básicos
///
/// Tests: inyección CR desde SN, DM no recibe CR, difusión iguala energía,
///        cr_pressure formula, cr_energy >= 0, serde de CrSection.
use gadget_ng_core::{CrSection, Particle, ParticleType, Vec3};
use gadget_ng_sph::{cr_pressure, diffuse_cr, inject_cr_from_sn};

fn gas(id: usize, x: f64) -> Particle {
    Particle::new_gas(id, 1.0, Vec3::new(x, 0.0, 0.0), Vec3::zero(), 1.0, 0.5)
}

// ── 1. Inyección CR aumenta cr_energy del gas ─────────────────────────────

#[test]
fn injection_increases_cr_energy() {
    let mut particles = vec![gas(0, 0.0)];
    let sfr = vec![1.0];
    let cr_before = particles[0].cr_energy;
    inject_cr_from_sn(&mut particles, &sfr, 0.1, 0.01);
    assert!(particles[0].cr_energy > cr_before, "cr_energy debe aumentar tras inyección");
}

// ── 2. SFR = 0 → sin inyección ────────────────────────────────────────────

#[test]
fn zero_sfr_no_cr_injection() {
    let mut particles = vec![gas(0, 0.0)];
    let sfr = vec![0.0];
    inject_cr_from_sn(&mut particles, &sfr, 0.1, 1.0);
    assert_eq!(particles[0].cr_energy, 0.0, "Sin SFR, no debe inyectarse CR");
}

// ── 3. DM no recibe inyección CR ──────────────────────────────────────────

#[test]
fn dm_not_injected_with_cr() {
    let mut particles = vec![Particle::new(0, 1.0, Vec3::zero(), Vec3::zero())];
    let sfr = vec![1.0];
    inject_cr_from_sn(&mut particles, &sfr, 0.1, 1.0);
    assert_eq!(particles[0].cr_energy, 0.0, "DM no debe recibir CR");
}

// ── 4. Difusión iguala energía CR entre vecinos ────────────────────────────

#[test]
fn diffusion_equalizes_cr_energy() {
    // Partículas muy cercanas (dentro del radio de suavizado)
    let mut p0 = gas(0, 0.0);
    let mut p1 = gas(1, 0.1); // vecino muy cercano (r << h=0.5)
    p0.cr_energy = 2.0;
    p1.cr_energy = 0.0;
    p0.smoothing_length = 1.0; // h grande para cubrir a p1
    p1.smoothing_length = 1.0;
    let mut particles = vec![p0, p1];

    diffuse_cr(&mut particles, 0.1, 1.0);

    // La partícula con más CR debe perder algo; la otra debe ganar
    assert!(particles[1].cr_energy > 0.0, "p1 debe recibir CR por difusión");
    assert!(particles[0].cr_energy < 2.0, "p0 debe perder CR por difusión");
}

// ── 5. cr_pressure formula correcta ──────────────────────────────────────

#[test]
fn cr_pressure_formula() {
    let e_cr = 1.0;
    let rho = 2.0;
    let expected = (4.0 / 3.0 - 1.0) * rho * e_cr; // = 2/3
    let p_cr = cr_pressure(e_cr, rho);
    assert!((p_cr - expected).abs() < 1e-12, "P_cr incorrecto: {p_cr} vs {expected}");
}

// ── 6. serde de CrSection ────────────────────────────────────────────────

#[test]
fn cr_section_serde() {
    let cfg = CrSection { enabled: true, cr_fraction: 0.15, kappa_cr: 5e-3 };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: CrSection = serde_json::from_str(&json).unwrap();
    assert!(cfg2.enabled);
    assert!((cfg2.cr_fraction - 0.15).abs() < 1e-15);
    assert!((cfg2.kappa_cr - 5e-3).abs() < 1e-20);
}
