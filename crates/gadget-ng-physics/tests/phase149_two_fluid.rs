/// Phase 149 — Plasma de dos fluidos: T_e ≠ T_i, acoplamiento Coulomb
///
/// Tests: inicialización T_e = T_i, acoplamiento reduce |T_e - T_i|,
///        T_e siempre positiva, mean_te_over_ti en equilibrio = 1,
///        T_e inicializada a 0 = T_i en primer paso, cfg desactivado es no-op.
use gadget_ng_core::{Particle, TwoFluidSection, Vec3};
use gadget_ng_mhd::{apply_electron_ion_coupling, mean_te_over_ti};

fn gas_with_te(id: usize, u: f64, t_e: f64) -> Particle {
    let mut p = Particle::new_gas(
        id,
        1.0,
        Vec3::new(id as f64 * 0.1, 0.0, 0.0),
        Vec3::zero(),
        u,
        0.2,
    );
    p.t_electron = t_e;
    p
}

fn cfg_active() -> TwoFluidSection {
    TwoFluidSection {
        enabled: true,
        nu_ei_coeff: 10.0,
        t_e_init_k: 0.0,
    }
}

// ── 1. T_e = 0 → inicializada a T_i ─────────────────────────────────────

#[test]
fn te_zero_initialized_to_ti() {
    let mut particles = vec![gas_with_te(0, 1e12, 0.0)]; // t_e = 0
    let cfg = cfg_active();
    apply_electron_ion_coupling(&mut particles, &cfg, 0.01);
    // Después del primer paso, t_electron debe igualarse a T_i y no ser 0
    // (el paso con t_e=0 solo inicializa, no evoluye)
    let t_i = 2.0 / 3.0 * particles[0].internal_energy; // (γ-1)×u
    let t_e = particles[0].t_electron;
    assert!(
        (t_e - t_i).abs() < t_i * 1e-10 || t_e == t_i,
        "T_e debe inicializarse a T_i: T_e={t_e:.3e}, T_i={t_i:.3e}"
    );
}

// ── 2. Acoplamiento reduce |T_e - T_i| con el tiempo ─────────────────────

#[test]
fn coupling_reduces_temperature_gap() {
    // T_i >> T_e: acoplamiento debe aumentar T_e
    let u_hot = 1e15; // T_i alto
    let t_e_init = 1.0; // T_e muy fría
    let mut particles = vec![gas_with_te(0, u_hot, t_e_init)];
    let cfg = cfg_active();
    let gap_before = {
        let t_i = 2.0 / 3.0 * particles[0].internal_energy;
        (t_i - particles[0].t_electron).abs()
    };
    apply_electron_ion_coupling(&mut particles, &cfg, 0.1);
    let t_i = 2.0 / 3.0 * particles[0].internal_energy;
    let gap_after = (t_i - particles[0].t_electron).abs();
    assert!(
        gap_after < gap_before,
        "Acoplamiento debe reducir |T_e - T_i|: antes={gap_before:.3e}, después={gap_after:.3e}"
    );
}

// ── 3. T_e siempre ≥ 0 ───────────────────────────────────────────────────

#[test]
fn te_always_non_negative() {
    let mut particles: Vec<Particle> = (0..20)
        .map(|i| {
            let u = if i < 10 { 1e15 } else { 1e6 };
            let t_e = if i < 5 { 1.0 } else { 1e12 };
            gas_with_te(i, u, t_e)
        })
        .collect();
    let cfg = cfg_active();
    for _ in 0..10 {
        apply_electron_ion_coupling(&mut particles, &cfg, 0.01);
    }
    for p in &particles {
        assert!(
            p.t_electron >= 0.0,
            "T_e debe ser ≥ 0: {:.3e}",
            p.t_electron
        );
    }
}

// ── 4. mean_te_over_ti ≈ 1 en equilibrio ──────────────────────────────────

#[test]
fn mean_te_over_ti_equilibrium() {
    // Cuando T_e = T_i → ratio = 1
    let u = 1e12;
    let particles: Vec<Particle> = (0..10)
        .map(|i| {
            let t_i = 2.0 / 3.0 * u;
            gas_with_te(i, u, t_i) // T_e = T_i exactamente
        })
        .collect();
    let ratio = mean_te_over_ti(&particles);
    assert!(
        (ratio - 1.0).abs() < 1e-10,
        "T_e/T_i en equilibrio debe ser 1: {ratio:.6}"
    );
}

// ── 5. mean_te_over_ti < 1 cuando T_e << T_i ─────────────────────────────

#[test]
fn mean_te_over_ti_below_one_out_of_equilibrium() {
    let u = 1e15;
    let t_e_cold = 1.0; // muy frío respecto a T_i
    let particles: Vec<Particle> = (0..5).map(|i| gas_with_te(i, u, t_e_cold)).collect();
    let ratio = mean_te_over_ti(&particles);
    assert!(ratio < 1.0, "T_e << T_i → ratio < 1: {ratio:.6e}");
}

// ── 6. Partículas no-gas ignoradas por apply_electron_ion_coupling ────────

#[test]
fn non_gas_particles_ignored() {
    let mut p_dm = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero());
    p_dm.t_electron = 5.0;
    let mut particles = vec![p_dm];
    let cfg = cfg_active();
    let t_e_before = particles[0].t_electron;
    apply_electron_ion_coupling(&mut particles, &cfg, 0.1);
    assert_eq!(
        particles[0].t_electron, t_e_before,
        "Partículas DM deben ser ignoradas"
    );
}
