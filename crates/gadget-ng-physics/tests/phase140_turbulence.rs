/// Phase 140 — Turbulencia MHD: forzado Ornstein-Uhlenbeck + espectro de Kolmogorov
///
/// Tests: disabled no modifica velocidades, enabled perturba velocidades,
///        amplitude=0 no-op, turbulence_stats con v=0 devuelve 0,
///        TurbulenceSection defaults, stats Mach positivo con v>0.
use gadget_ng_core::{Particle, TurbulenceSection, Vec3};
use gadget_ng_mhd::{apply_turbulent_forcing, turbulence_stats};

fn gas(id: usize, vel: Vec3, b: Vec3) -> Particle {
    let mut p = Particle::new_gas(id, 1.0,
        Vec3::new(id as f64 * 0.1, 0.0, 0.0), vel, 1e8, 0.2);
    p.b_field = b;
    p
}

fn turb_cfg(enabled: bool, amplitude: f64) -> TurbulenceSection {
    TurbulenceSection {
        enabled,
        amplitude,
        correlation_time: 1.0,
        k_min: 1.0,
        k_max: 4.0,
        spectral_index: 5.0 / 3.0,
    }
}

// ── 1. disabled → no modifica velocidades ────────────────────────────────

#[test]
fn disabled_no_effect() {
    let mut particles = vec![gas(0, Vec3::zero(), Vec3::new(1.0, 0.0, 0.0))];
    let cfg = turb_cfg(false, 1e-3);
    apply_turbulent_forcing(&mut particles, &cfg, 0.01, 0);
    assert_eq!(particles[0].velocity.x, 0.0, "disabled: v debe ser 0");
}

// ── 2. amplitude=0 → no-op ────────────────────────────────────────────────

#[test]
fn zero_amplitude_no_op() {
    let mut particles = vec![gas(0, Vec3::zero(), Vec3::new(1.0, 0.0, 0.0))];
    let cfg = turb_cfg(true, 0.0);
    apply_turbulent_forcing(&mut particles, &cfg, 0.01, 0);
    assert_eq!(particles[0].velocity.x, 0.0, "amplitude=0: v debe ser 0");
}

// ── 3. enabled + amplitude>0 → perturba velocidades ──────────────────────

#[test]
fn enabled_perturbs_velocities() {
    let mut particles: Vec<Particle> = (0..8).map(|i| gas(i, Vec3::zero(), Vec3::new(1.0, 0.0, 0.0))).collect();
    let cfg = turb_cfg(true, 1e-2);
    apply_turbulent_forcing(&mut particles, &cfg, 0.01, 1);
    let v_total: f64 = particles.iter().map(|p| p.velocity.x.abs() + p.velocity.y.abs() + p.velocity.z.abs()).sum();
    assert!(v_total > 0.0, "El forzado debe perturbar velocidades: v_total={v_total:.4e}");
}

// ── 4. turbulence_stats con v=0 y B>0 → Mach=0, Alfvén Mach=0 ───────────

#[test]
fn stats_zero_velocity() {
    let particles = vec![
        gas(0, Vec3::zero(), Vec3::new(1.0, 0.0, 0.0)),
        gas(1, Vec3::zero(), Vec3::new(1.0, 0.0, 0.0)),
    ];
    let (mach, alfven_mach) = turbulence_stats(&particles, 5.0/3.0);
    assert_eq!(mach, 0.0, "Mach debe ser 0 con v=0");
    assert_eq!(alfven_mach, 0.0, "Alfvén Mach debe ser 0 con v=0");
}

// ── 5. TurbulenceSection defaults correctos ───────────────────────────────

#[test]
fn turbulence_section_defaults() {
    let cfg = TurbulenceSection::default();
    assert!(!cfg.enabled);
    assert!((cfg.amplitude - 1e-3).abs() < 1e-10);
    assert!((cfg.correlation_time - 1.0).abs() < 1e-10);
    assert!((cfg.k_min - 1.0).abs() < 1e-10);
    assert!((cfg.k_max - 4.0).abs() < 1e-10);
    assert!((cfg.spectral_index - 5.0/3.0).abs() < 1e-10);
}

// ── 6. Mach positivo con partículas en movimiento ─────────────────────────

#[test]
fn stats_positive_mach_with_velocity() {
    let v = Vec3::new(1.0, 0.0, 0.0);
    let b = Vec3::new(0.1, 0.0, 0.0);
    let particles = vec![gas(0, v, b), gas(1, v, b)];
    let (mach, alfven_mach) = turbulence_stats(&particles, 5.0/3.0);
    assert!(mach > 0.0, "Mach debe ser positivo: {mach}");
    assert!(alfven_mach > 0.0, "Alfvén Mach debe ser positivo: {alfven_mach}");
}
