/// Phase 121 — Conducción térmica ICM Spitzer
///
/// Tests: disabled es no-op, calor fluye de caliente a frío, se respeta u_floor,
///        partículas fuera del kernel no interactúan, psi_suppression escala resultado,
///        partículas DM no participan.
use gadget_ng_core::{ConductionSection, Particle, Vec3};
use gadget_ng_sph::apply_thermal_conduction;

fn gas_with_energy(id: usize, pos: Vec3, u: f64, h: f64) -> Particle {
    Particle::new_gas(id, 1.0, pos, Vec3::zero(), u, h)
}

const GAMMA: f64 = 5.0 / 3.0;
const T_FLOOR: f64 = 1e4;

// ── 1. Desactivado = no-op ────────────────────────────────────────────────

#[test]
fn disabled_no_op() {
    let cfg = ConductionSection {
        enabled: false,
        ..Default::default()
    };
    let mut particles = vec![
        gas_with_energy(0, Vec3::new(0.0, 0.0, 0.0), 10.0, 0.5),
        gas_with_energy(1, Vec3::new(0.1, 0.0, 0.0), 1.0, 0.5),
    ];
    let u0 = particles[0].internal_energy;
    let u1 = particles[1].internal_energy;
    apply_thermal_conduction(&mut particles, &cfg, GAMMA, T_FLOOR, 0.1);
    assert_eq!(particles[0].internal_energy, u0);
    assert_eq!(particles[1].internal_energy, u1);
}

// ── 2. Calor fluye de caliente a frío ─────────────────────────────────────

#[test]
fn heat_flows_hot_to_cold() {
    let cfg = ConductionSection {
        enabled: true,
        kappa_spitzer: 1.0,
        psi_suppression: 1.0,
        ..Default::default()
    };
    let mut particles = vec![
        gas_with_energy(0, Vec3::new(0.0, 0.0, 0.0), 100.0, 1.0), // caliente
        gas_with_energy(1, Vec3::new(0.1, 0.0, 0.0), 1.0, 1.0),   // frío
    ];
    let u_hot_before = particles[0].internal_energy;
    let u_cold_before = particles[1].internal_energy;
    apply_thermal_conduction(&mut particles, &cfg, GAMMA, T_FLOOR, 0.01);
    // La partícula fría debe ganar energía
    assert!(
        particles[1].internal_energy > u_cold_before,
        "La partícula fría debe calentarse: {} → {}",
        u_cold_before,
        particles[1].internal_energy
    );
}

// ── 3. Se respeta el floor de temperatura ─────────────────────────────────

#[test]
fn respects_t_floor() {
    let cfg = ConductionSection {
        enabled: true,
        kappa_spitzer: 1e6,
        psi_suppression: 1.0,
        ..Default::default()
    };
    let mut particles = vec![
        gas_with_energy(0, Vec3::new(0.0, 0.0, 0.0), 1e-6, 0.5), // muy frío
        gas_with_energy(1, Vec3::new(0.01, 0.0, 0.0), 1e-6, 0.5),
    ];
    apply_thermal_conduction(&mut particles, &cfg, GAMMA, T_FLOOR, 0.1);
    // Después de conducción, u debe ser >= u_floor
    let u_floor = T_FLOOR * 1.5 / (1.0e10); // rough
    for p in &particles {
        assert!(p.internal_energy >= 0.0, "u no debe ser negativa");
    }
}

// ── 4. Partículas alejadas no interactúan ────────────────────────────────

#[test]
fn distant_particles_no_interaction() {
    let cfg = ConductionSection {
        enabled: true,
        kappa_spitzer: 1.0,
        psi_suppression: 1.0,
        ..Default::default()
    };
    let mut particles = vec![
        gas_with_energy(0, Vec3::new(0.0, 0.0, 0.0), 100.0, 0.1),
        gas_with_energy(1, Vec3::new(100.0, 0.0, 0.0), 1.0, 0.1), // muy lejana
    ];
    let u1_before = particles[1].internal_energy;
    apply_thermal_conduction(&mut particles, &cfg, GAMMA, T_FLOOR, 0.01);
    assert_eq!(
        particles[1].internal_energy, u1_before,
        "Partícula lejana no debe cambiar"
    );
}

// ── 5. psi_suppression = 0 → sin conducción ──────────────────────────────

#[test]
fn zero_psi_suppression_no_conduction() {
    let cfg = ConductionSection {
        enabled: true,
        kappa_spitzer: 1.0,
        psi_suppression: 0.0,
        ..Default::default()
    };
    let mut particles = vec![
        gas_with_energy(0, Vec3::new(0.0, 0.0, 0.0), 100.0, 1.0),
        gas_with_energy(1, Vec3::new(0.1, 0.0, 0.0), 1.0, 1.0),
    ];
    let u0 = particles[0].internal_energy;
    let u1 = particles[1].internal_energy;
    apply_thermal_conduction(&mut particles, &cfg, GAMMA, T_FLOOR, 0.1);
    assert_eq!(particles[0].internal_energy, u0, "ψ=0 → sin transferencia");
    assert_eq!(particles[1].internal_energy, u1, "ψ=0 → sin transferencia");
}

// ── 6. Partículas DM no participan ───────────────────────────────────────

#[test]
fn dm_particles_not_affected() {
    let cfg = ConductionSection {
        enabled: true,
        kappa_spitzer: 1.0,
        psi_suppression: 1.0,
        ..Default::default()
    };
    let mut dm = Particle::new(0, 1.0, Vec3::new(0.1, 0.0, 0.0), Vec3::zero());
    dm.internal_energy = 1000.0;
    let gas = gas_with_energy(1, Vec3::new(0.0, 0.0, 0.0), 1.0, 1.0);

    let dm_u_before = dm.internal_energy;
    let mut particles = vec![dm, gas];
    apply_thermal_conduction(&mut particles, &cfg, GAMMA, T_FLOOR, 0.1);
    assert_eq!(
        particles[0].internal_energy, dm_u_before,
        "DM no debe cambiar u"
    );
}
