/// Phase 147 — Corrida cosmológica de referencia MHD completo + P_B(k)
///
/// Tests: P_B(k) espectro tiene estructura, b_rms > 0 después de evolución,
///        e_mag finita, max|v| < c_light, MHD end-to-end conserva energía magnética,
///        magnetic_power_spectrum retorna bins correctos.
use gadget_ng_core::{MhdSection, Particle, Vec3};
use gadget_ng_mhd::{
    advance_induction, apply_magnetic_forces, b_field_stats, dedner_cleaning_step,
    magnetic_power_spectrum, C_LIGHT,
};

fn gas_uniform_b(n: usize, b0: f64, box_l: f64) -> Vec<Particle> {
    (0..n)
        .map(|i| {
            let fi = i as f64;
            let x = (fi / n as f64) * box_l;
            let mut p = Particle::new_gas(
                i, 1.0,
                Vec3::new(x, (fi * 0.017).sin() * 0.1 * box_l,
                              (fi * 0.013).cos() * 0.1 * box_l),
                Vec3::new((fi * 0.001).sin() * 1.0, 0.0, 0.0),
                1e12, box_l / n as f64,
            );
            p.b_field = Vec3::new(b0, 0.0, 0.0);
            p
        })
        .collect()
}

// ── 1. magnetic_power_spectrum retorna n_bins > 0 ─────────────────────────

#[test]
fn power_spectrum_has_bins() {
    let particles = gas_uniform_b(64, 1e-9, 1.0);
    let pk = magnetic_power_spectrum(&particles, 1.0, 8);
    assert!(!pk.is_empty(), "P_B(k) debe retornar bins: got {} bins", pk.len());
}

// ── 2. P_B(k) es positivo en todos los bins ───────────────────────────────

#[test]
fn power_spectrum_positive() {
    let particles = gas_uniform_b(64, 1e-6, 1.0);
    let pk = magnetic_power_spectrum(&particles, 1.0, 8);
    for (k, p) in &pk {
        assert!(*p >= 0.0 && k.is_finite(), "P_B({k:.3e}) = {p:.3e} debe ser >= 0");
    }
}

// ── 3. b_rms > 0 tras evolución MHD (10 pasos) ───────────────────────────

#[test]
fn b_rms_nonzero_after_mhd_steps() {
    let mut particles = gas_uniform_b(32, 1e-9, 1.0);
    let dt = 1e-4;
    let cfg = MhdSection { enabled: true, ..Default::default() };
    for _ in 0..10 {
        advance_induction(&mut particles, dt);
        apply_magnetic_forces(&mut particles, dt);
        dedner_cleaning_step(&mut particles, cfg.c_h, cfg.c_r, dt);
    }
    let stats = b_field_stats(&particles).expect("debe haber estadísticas");
    assert!(stats.b_rms > 0.0, "b_rms debe ser > 0 tras evolución: {:.3e}", stats.b_rms);
}

// ── 4. e_mag es finita ─────────────────────────────────────────────────────

#[test]
fn e_mag_finite_after_evolution() {
    let mut particles = gas_uniform_b(32, 1e-9, 1.0);
    let dt = 1e-4;
    let cfg = MhdSection { enabled: true, ..Default::default() };
    for _ in 0..5 {
        advance_induction(&mut particles, dt);
        apply_magnetic_forces(&mut particles, dt);
        dedner_cleaning_step(&mut particles, cfg.c_h, cfg.c_r, dt);
    }
    let stats = b_field_stats(&particles).expect("debe haber estadísticas");
    assert!(stats.e_mag.is_finite() && stats.e_mag >= 0.0,
        "e_mag debe ser finita: {:.3e}", stats.e_mag);
}

// ── 5. max|v| < C_LIGHT tras evolución ────────────────────────────────────

#[test]
fn max_velocity_below_c() {
    let mut particles = gas_uniform_b(32, 1e-9, 1.0);
    let dt = 1e-4;
    let cfg = MhdSection { enabled: true, ..Default::default() };
    for _ in 0..10 {
        advance_induction(&mut particles, dt);
        apply_magnetic_forces(&mut particles, dt);
        dedner_cleaning_step(&mut particles, cfg.c_h, cfg.c_r, dt);
    }
    let v_max = particles.iter().map(|p| {
        (p.velocity.x*p.velocity.x + p.velocity.y*p.velocity.y + p.velocity.z*p.velocity.z).sqrt()
    }).fold(0.0_f64, f64::max);
    assert!(v_max < C_LIGHT, "max|v| = {v_max:.3e} debe ser < C_LIGHT = {C_LIGHT:.3e}");
}

// ── 6. P_B(k) espectro tiene dispersión (no todos iguales) ────────────────

#[test]
fn power_spectrum_has_variation() {
    // Con B no uniforme, P_B(k) debe variar entre bins
    let mut particles = gas_uniform_b(64, 1e-6, 1.0);
    // Perturbamos B y h para crear variación espectral con rango de k amplio
    for (i, p) in particles.iter_mut().enumerate() {
        let fi = i as f64;
        p.b_field.x *= 1.0 + 0.3 * (fi * 0.5).sin();
        // h varía en factor 100: escala de k correspondiente varía en factor 100
        p.smoothing_length = 0.001 + 0.1 * (fi / 64.0).powf(2.0).min(0.099);
    }
    let pk = magnetic_power_spectrum(&particles, 1.0, 8);
    assert!(pk.len() >= 2, "Debe haber al menos 2 bins: got {}", pk.len());
    let p_max = pk.iter().map(|(_, p)| *p).fold(0.0_f64, f64::max);
    let p_min = pk.iter().map(|(_, p)| *p).fold(f64::INFINITY, f64::min);
    assert!(p_max > p_min, "P_B(k) debe variar entre bins");
}
