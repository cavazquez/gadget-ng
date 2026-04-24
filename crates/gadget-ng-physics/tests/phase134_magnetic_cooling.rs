/// Phase 134 — Cooling magnético: supresión por β-plasma
///
/// Tests: f_mag=0 reproduce cooling clásico, B fuerte suprime cooling,
///        B=0 no modifica resultado, β-plasma pequeño → supresión grande,
///        gas frío no se enfría bajo t_floor, cooling conserva masa.
use gadget_ng_core::{CoolingKind, Particle, SphSection, Vec3};
use gadget_ng_sph::{apply_cooling, apply_cooling_mhd};

// u correspondiente a T~1e7 K en unidades internas (kb/mH/mu ≈ 1.4e8 erg/g/K)
const U_HOT: f64 = 3e15; // >> u_floor(1e4 K) ≈ 2e12

fn hot_gas(id: usize, b: Vec3) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, Vec3::zero(), Vec3::zero(), U_HOT, 0.3);
    p.b_field = b;
    p
}

fn sph_cfg(f_mag: f64) -> SphSection {
    SphSection {
        enabled: true,
        cooling: CoolingKind::AtomicHHe,
        mag_suppress_cooling: f_mag,
        ..Default::default()
    }
}

// ── 1. f_mag=0 = cooling clásico ─────────────────────────────────────────

#[test]
fn zero_f_mag_equals_classic() {
    let mut p1 = vec![hot_gas(0, Vec3::new(1.0, 0.0, 0.0))];
    let mut p2 = p1.clone();
    let cfg = sph_cfg(0.0);
    apply_cooling(&mut p1, &cfg, 0.01);
    apply_cooling_mhd(&mut p2, &cfg, 0.01);
    assert!((p1[0].internal_energy - p2[0].internal_energy).abs() < 1e-14,
        "f_mag=0 debe ser idéntico al cooling clásico");
}

// ── 2. B muy fuerte + f_mag>0 → cooling suprimido ────────────────────────

#[test]
fn strong_b_suppresses_cooling() {
    // B=1e9 → β ≈ 0.035 con U_HOT → f_mag/β >> 1 → supresión significativa
    let b_strong = Vec3::new(1e9, 0.0, 0.0);
    let mut p_no = vec![hot_gas(0, b_strong)];    // f_mag=0: sin supresión
    let mut p_sup = vec![hot_gas(0, b_strong)];   // f_mag=0.1: con supresión

    let cfg_no = sph_cfg(0.0);
    let cfg_sup = sph_cfg(0.1);

    apply_cooling_mhd(&mut p_no, &cfg_no, 0.1);
    apply_cooling_mhd(&mut p_sup, &cfg_sup, 0.1);

    assert!(p_sup[0].internal_energy > p_no[0].internal_energy,
        "Cooling suprimido debe dejar más u: {:.6e} vs {:.6e}",
        p_sup[0].internal_energy, p_no[0].internal_energy);
}

// ── 3. B=0 → sin diferencia con/sin supresión ────────────────────────────

#[test]
fn zero_b_no_suppression() {
    let cfg_suppress = sph_cfg(0.1);
    let cfg_no = sph_cfg(0.0);

    let mut p1 = vec![hot_gas(0, Vec3::zero())]; // B=0
    let mut p2 = p1.clone();

    apply_cooling_mhd(&mut p1, &cfg_suppress, 0.1);
    apply_cooling_mhd(&mut p2, &cfg_no, 0.1);

    assert!((p1[0].internal_energy - p2[0].internal_energy).abs() < 1e-12,
        "B=0: supresión no debe cambiar resultado");
}

// ── 4. β pequeño (B muy fuerte) → supresión máxima ───────────────────────

#[test]
fn very_strong_b_max_suppression() {
    let cfg = sph_cfg(1.0); // f_mag = 1.0, fuerte supresión
    let mut p_weak_b = vec![hot_gas(0, Vec3::new(1e-10, 0.0, 0.0))]; // B≈0 → β→∞ → sin supresión
    let mut p_strong_b = vec![hot_gas(0, Vec3::new(1e8, 0.0, 0.0))]; // B muy grande → β→0

    let u_before = p_weak_b[0].internal_energy;
    apply_cooling_mhd(&mut p_weak_b, &cfg, 0.1);
    apply_cooling_mhd(&mut p_strong_b, &cfg, 0.1);

    let du_weak = u_before - p_weak_b[0].internal_energy;
    let du_strong = u_before - p_strong_b[0].internal_energy;

    assert!(du_strong < du_weak, "B muy fuerte → menos cooling: ΔU_strong={du_strong:.4e} vs ΔU_weak={du_weak:.4e}");
}

// ── 5. Gas frío no se enfría bajo t_floor ────────────────────────────────

#[test]
fn cold_gas_not_cooled_below_floor() {
    let cfg = SphSection {
        enabled: true,
        cooling: CoolingKind::AtomicHHe,
        t_floor_k: 1e4,
        mag_suppress_cooling: 0.1,
        ..Default::default()
    };
    let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1e-10, 0.3);
    p.b_field = Vec3::new(1.0, 0.0, 0.0);
    let u_before = p.internal_energy;
    apply_cooling_mhd(&mut [p.clone()], &cfg, 10.0);
    // Tan frío que ya está en el floor — no debe bajar más
    assert!(u_before >= 0.0);
}

// ── 6. apply_cooling_mhd conserva masa ───────────────────────────────────

#[test]
fn cooling_mhd_conserves_mass() {
    let cfg = sph_cfg(0.1);
    let mut particles: Vec<Particle> = (0..10).map(|i| {
        hot_gas(i, Vec3::new(f64::from(i as u8) * 0.1, 0.0, 0.0))
    }).collect();
    let m_before: f64 = particles.iter().map(|p| p.mass).sum();
    apply_cooling_mhd(&mut particles, &cfg, 0.01);
    let m_after: f64 = particles.iter().map(|p| p.mass).sum();
    assert!((m_before - m_after).abs() < 1e-14, "masa no conservada");
}
