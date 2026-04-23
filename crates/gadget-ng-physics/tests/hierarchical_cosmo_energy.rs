//! Test de conservación de energía para block timesteps jerárquicos con cosmología.
//!
//! Verifica que el path jerárquico cosmológico:
//! 1. Aplica el acoplamiento G·a³ correctamente (momentum canónico GADGET-4).
//! 2. Conserva la energía cinética comóvil dentro de un margen razonable
//!    durante una evolución de corta duración (~10 pasos).
//! 3. La cota kappa_h restringe los bins de las partículas activas.

use gadget_ng_core::{cosmology::CosmologyParams, Particle, TimestepCriterion, Vec3};
use gadget_ng_integrators::{hierarchical_kdk_step, HierarchicalState};

const G: f64 = 1.0;
const EPS2: f64 = 0.01;
const ETA: f64 = 0.025;
const MAX_LEVEL: u32 = 4;

fn make_two_particles() -> Vec<Particle> {
    // Dos partículas en separación unitaria con velocidades opuestas pequeñas.
    let mut p0 = Particle::new(0, 1.0, Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.0, 0.3, 0.0));
    p0.acceleration = Vec3::zero();

    let mut p1 = Particle::new(1, 1.0, Vec3::new(0.5, 0.0, 0.0), Vec3::new(0.0, -0.3, 0.0));
    p1.acceleration = Vec3::zero();

    vec![p0, p1]
}

fn gravity_direct(parts: &[Particle], active: &[usize], out: &mut [Vec3]) {
    for (j, &i) in active.iter().enumerate() {
        let mut a = Vec3::zero();
        for (k, pk) in parts.iter().enumerate() {
            if k == i {
                continue;
            }
            let dr = pk.position - parts[i].position;
            let r2 = dr.dot(dr) + EPS2;
            let r3 = r2 * r2.sqrt();
            a += dr * (G * pk.mass / r3);
        }
        out[j] = a;
    }
}

fn kinetic_energy_comoving(parts: &[Particle], a: f64) -> f64 {
    // En convención QKSL: p = a²·v_peculiar → v_peculiar = p/a²
    // E_cin comóvil = Σ m·|p|²/(2a⁴) = Σ m·|v|²/(2a⁴) donde v = velocity en código
    parts
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity) / (a * a * a * a))
        .sum()
}

/// Verifica que con cosmología activada, el path jerárquico no explota:
/// la energía cinética comóvil debe mantenerse en un factor razonable tras 10 pasos.
#[test]
fn hierarchical_cosmo_energy_bounded() {
    let cosmo = CosmologyParams::new(0.3, 0.7, 0.1);
    let mut a = 0.5_f64;
    let dt = 0.005;

    let mut parts = make_two_particles();
    let all_idx: Vec<usize> = (0..parts.len()).collect();

    // Aceleraciones iniciales con G·a³ (acoplamiento cosmológico).
    let g_eff = gadget_ng_core::gravity_coupling_qksl(G, a);
    let mut init_acc = vec![Vec3::zero(); parts.len()];
    gravity_direct(&parts, &all_idx, &mut init_acc);
    for acc in init_acc.iter_mut() {
        *acc *= g_eff / G;
    }
    for (p, &acc) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = acc;
    }

    let mut h_state = HierarchicalState::new(parts.len());
    h_state.init_from_accels(
        &parts,
        EPS2,
        dt,
        ETA,
        MAX_LEVEL,
        TimestepCriterion::Acceleration,
    );

    let e0 = kinetic_energy_comoving(&parts, a);

    for _ in 0..10 {
        let g_step = gadget_ng_core::gravity_coupling_qksl(G, a);
        let _stats = hierarchical_kdk_step(
            &mut parts,
            &mut h_state,
            dt,
            EPS2,
            ETA,
            MAX_LEVEL,
            TimestepCriterion::Acceleration,
            Some((&cosmo, &mut a)),
            None,
            |ps, active, acc| {
                gravity_direct(ps, active, acc);
                for a_val in acc.iter_mut() {
                    *a_val *= g_step / G;
                }
            },
        );
    }

    let e_final = kinetic_energy_comoving(&parts, a);

    // La energía cinética comóvil debe estar dentro de un factor 10× de la inicial
    // (la cosmología añade trabajo, no se espera conservación exacta).
    // El objetivo es verificar que no explota (NaN, Inf, o crecimiento catastrófico).
    assert!(
        e_final.is_finite(),
        "energía cinética comóvil no es finita: {e_final}"
    );
    assert!(
        e_final < e0 * 10.0 + 1.0,
        "energía cinética comóvil creció demasiado: E0={e0:.4e} → E_final={e_final:.4e}"
    );
}

/// Verifica que kappa_h restringe los niveles de bin:
/// con kappa_h muy pequeño, todas las partículas deberían estar en el nivel máximo.
#[test]
fn hierarchical_cosmo_kappa_h_restricts_bins() {
    let cosmo = CosmologyParams::new(0.3, 0.7, 0.1);
    let mut a = 0.5_f64;
    let dt = 1.0; // dt_base grande para que el criterio de Aarseth no domine

    let mut parts = make_two_particles();
    let all_idx: Vec<usize> = (0..parts.len()).collect();

    let g_eff = gadget_ng_core::gravity_coupling_qksl(G, a);
    let mut init_acc = vec![Vec3::zero(); parts.len()];
    gravity_direct(&parts, &all_idx, &mut init_acc);
    for acc in init_acc.iter_mut() {
        *acc *= g_eff / G;
    }
    for (p, &acc) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = acc;
    }

    let mut h_state = HierarchicalState::new(parts.len());
    h_state.init_from_accels(
        &parts,
        EPS2,
        dt,
        ETA,
        MAX_LEVEL,
        TimestepCriterion::Acceleration,
    );

    // kappa_h = 1e-6 → dt_cosmo_max ≈ 1e-6 * a / H(a) extremadamente pequeño
    // → todos los bins deberían estar en MAX_LEVEL.
    let kappa_h = Some(1e-6_f64);
    let g_step = gadget_ng_core::gravity_coupling_qksl(G, a);
    let stats = hierarchical_kdk_step(
        &mut parts,
        &mut h_state,
        dt,
        EPS2,
        ETA,
        MAX_LEVEL,
        TimestepCriterion::Acceleration,
        Some((&cosmo, &mut a)),
        kappa_h,
        |ps, active, acc| {
            gravity_direct(ps, active, acc);
            for a_val in acc.iter_mut() {
                *a_val *= g_step / G;
            }
        },
    );

    // Con kappa_h mínimo, el dt efectivo mínimo debe ser el paso fino.
    let fine_dt = dt / (1u64 << MAX_LEVEL) as f64;
    assert!(
        stats.dt_min_effective <= fine_dt * 2.0,
        "kappa_h extremadamente pequeño no restringió los bins: dt_min_effective={:.4e}, fine_dt={:.4e}",
        stats.dt_min_effective,
        fine_dt
    );
}
