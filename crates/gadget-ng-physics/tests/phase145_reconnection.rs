/// Phase 145 — Reconexión magnética Sweet-Parker (reconnection.rs)
///
/// Verifica: B antiparalelo detectado correctamente, energía liberada > 0,
/// B paralelo ignorado, tasa Sweet-Parker teórica, f_rec=0 es no-op,
/// campo B reducido tras reconexión.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::{apply_magnetic_reconnection, sweet_parker_rate};

fn gas(id: usize, pos: Vec3, b: Vec3, u: f64) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, Vec3::zero(), u, 0.2);
    p.b_field = b;
    p
}

// ── 1. B antiparalelo libera calor ────────────────────────────────────────

#[test]
fn antiparallel_b_releases_heat() {
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0),
        gas(1, Vec3::new(0.05, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 1.0),
    ];
    let u0 = particles[0].internal_energy;
    apply_magnetic_reconnection(&mut particles, 0.1, 5.0 / 3.0, 0.1);
    assert!(
        particles[0].internal_energy > u0,
        "B antiparalelo debe liberar calor"
    );
}

// ── 2. B paralelo NO libera calor ─────────────────────────────────────────

#[test]
fn parallel_b_no_heat_release() {
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0),
        gas(1, Vec3::new(0.05, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0),
    ];
    let u0 = particles[0].internal_energy;
    apply_magnetic_reconnection(&mut particles, 0.1, 5.0 / 3.0, 0.1);
    assert_eq!(
        particles[0].internal_energy, u0,
        "B paralelo NO debe liberar calor"
    );
}

// ── 3. B reduce su magnitud tras reconexión ───────────────────────────────

#[test]
fn b_magnitude_decreases_after_reconnection() {
    let b_init = 5.0;
    let mut particles = vec![
        gas(
            0,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(b_init, 0.0, 0.0),
            1.0,
        ),
        gas(
            1,
            Vec3::new(0.05, 0.0, 0.0),
            Vec3::new(-b_init, 0.0, 0.0),
            1.0,
        ),
    ];
    apply_magnetic_reconnection(&mut particles, 0.2, 5.0 / 3.0, 0.1);
    let b0_after = particles[0].b_field.x.abs();
    assert!(
        b0_after < b_init,
        "Reconexión debe reducir |B|: {b0_after:.4} < {b_init}"
    );
}

// ── 4. f_reconnection = 0 → no-op ────────────────────────────────────────

#[test]
fn f_rec_zero_no_op() {
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0),
        gas(1, Vec3::new(0.05, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 1.0),
    ];
    let u0 = particles[0].internal_energy;
    let b0 = particles[0].b_field.x;
    apply_magnetic_reconnection(&mut particles, 0.0, 5.0 / 3.0, 0.1);
    assert_eq!(particles[0].internal_energy, u0);
    assert_eq!(particles[0].b_field.x, b0);
}

// ── 5. Partículas fuera de 2h no reconectan ───────────────────────────────

#[test]
fn far_particles_no_reconnection() {
    // h = 0.2 → 2h = 0.4; separación = 10 >> 2h
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0),
        gas(1, Vec3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 1.0),
    ];
    let u0 = particles[0].internal_energy;
    apply_magnetic_reconnection(&mut particles, 0.1, 5.0 / 3.0, 0.1);
    assert_eq!(
        particles[0].internal_energy, u0,
        "Partículas lejanas no deben reconectar"
    );
}

// ── 6. Tasa Sweet-Parker teórica ──────────────────────────────────────────

#[test]
fn sweet_parker_rate_formula() {
    let v_a = 100.0_f64;
    let l = 10.0_f64;
    let eta = 0.001_f64;
    let rm = l * v_a / eta; // Rm = 1_000_000
    let v_rec_expected = v_a / rm.sqrt(); // = v_A / sqrt(Rm) = 0.1
    let v_rec = sweet_parker_rate(v_a, l, eta);
    assert!(
        (v_rec - v_rec_expected).abs() < 1e-10,
        "Sweet-Parker: {v_rec:.6e} ≠ {v_rec_expected:.6e}"
    );
    assert!(v_rec < v_a, "Tasa de reconexión < velocidad de Alfvén");
}
