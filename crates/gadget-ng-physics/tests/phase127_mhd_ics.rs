/// Phase 127 — ICs magnetizadas + CFL magnético
///
/// Tests: b0_kind None = B=0, Uniform inicializa B correcto, Random tiene |B| correcto,
///        Spiral tiene estructura espacial, alfven_dt da valor finito con B≠0,
///        alfven_dt=∞ con B=0.
use gadget_ng_core::{BFieldKind, MhdSection, Particle, Vec3};
use gadget_ng_mhd::{alfven_dt, init_b_field};

fn gas(id: usize, pos: Vec3) -> Particle {
    Particle::new_gas(id, 1.0, pos, Vec3::zero(), 1.0, 0.1)
}

fn mhd_cfg(kind: BFieldKind, b0: [f64; 3]) -> MhdSection {
    MhdSection { enabled: true, b0_kind: kind, b0_uniform: b0, cfl_mhd: 0.3, ..Default::default() }
}

// ── 1. b0_kind None → B permanece cero ───────────────────────────────────

#[test]
fn none_leaves_b_zero() {
    let cfg = mhd_cfg(BFieldKind::None, [1.0, 0.0, 0.0]);
    let mut particles = vec![gas(0, Vec3::zero()), gas(1, Vec3::new(0.1, 0.0, 0.0))];
    init_b_field(&mut particles, &cfg, 1.0);
    for p in &particles {
        assert_eq!(p.b_field, Vec3::zero(), "None: B debe ser 0");
    }
}

// ── 2. Uniform inicializa todas las partículas con b0_uniform ─────────────

#[test]
fn uniform_sets_b_correctly() {
    let b0 = [1.0, 0.5, 0.2];
    let cfg = mhd_cfg(BFieldKind::Uniform, b0);
    let mut particles = vec![gas(0, Vec3::zero()), gas(1, Vec3::new(1.0, 0.0, 0.0))];
    init_b_field(&mut particles, &cfg, 10.0);
    for p in &particles {
        assert!((p.b_field.x - b0[0]).abs() < 1e-12, "B.x != b0[0]");
        assert!((p.b_field.y - b0[1]).abs() < 1e-12, "B.y != b0[1]");
        assert!((p.b_field.z - b0[2]).abs() < 1e-12, "B.z != b0[2]");
    }
}

// ── 3. Random: |B| ≈ |b0_uniform| para cada partícula ────────────────────

#[test]
fn random_preserves_b_magnitude() {
    let b0 = [2.0, 0.0, 0.0];
    let b_mag = 2.0_f64;
    let cfg = mhd_cfg(BFieldKind::Random, b0);
    let mut particles: Vec<Particle> = (0..10).map(|i| gas(i, Vec3::new(i as f64 * 0.1, 0.0, 0.0))).collect();
    init_b_field(&mut particles, &cfg, 10.0);
    for p in &particles {
        let b = (p.b_field.x.powi(2) + p.b_field.y.powi(2) + p.b_field.z.powi(2)).sqrt();
        assert!((b - b_mag).abs() < 1e-10, "|B| = {b}, esperado {b_mag}");
    }
}

// ── 4. Spiral tiene patrón espacial coherente ─────────────────────────────

#[test]
fn spiral_spatially_varying() {
    let b0 = [1.0, 0.0, 0.0];
    let cfg = mhd_cfg(BFieldKind::Spiral, b0);
    let mut p1 = gas(0, Vec3::new(0.0, 0.0, 0.0));
    let mut p2 = gas(1, Vec3::new(0.25, 0.0, 0.0)); // x/L = 0.25
    init_b_field(&mut std::slice::from_mut(&mut p1), &cfg, 1.0);
    init_b_field(&mut std::slice::from_mut(&mut p2), &cfg, 1.0);
    // B.y = cos(2π x/L): para x=0 → cos(0)=1, para x=0.25 → cos(π/2)≈0
    assert!((p1.b_field.y - 1.0).abs() < 1e-10, "p1.B.y = {}", p1.b_field.y);
    assert!(p2.b_field.y.abs() < 1e-10, "p2.B.y ≈ 0, obtenido {}", p2.b_field.y);
}

// ── 5. alfven_dt da valor finito con B≠0 ─────────────────────────────────

#[test]
fn alfven_dt_finite_with_nonzero_b() {
    let mut particles = vec![gas(0, Vec3::zero())];
    particles[0].b_field = Vec3::new(1.0, 0.0, 0.0);
    let dt = alfven_dt(&particles, 0.3);
    assert!(dt.is_finite() && dt > 0.0, "dt_alfven debe ser finito y positivo: {dt}");
}

// ── 6. alfven_dt = ∞ con B=0 ─────────────────────────────────────────────

#[test]
fn alfven_dt_infinite_with_zero_b() {
    let particles = vec![gas(0, Vec3::zero())]; // B=0 por defecto
    let dt = alfven_dt(&particles, 0.3);
    assert!(dt.is_infinite(), "dt_alfven debe ser ∞ con B=0: {dt}");
}
