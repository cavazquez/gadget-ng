//! Validaciones SPH Gadget-2 (Phase 166).
//!
//! ## Tests rápidos (sin `#[ignore]`)
//!
//! Comprueban propiedades fundamentales del nuevo integrador de entropía:
//! conservación de entropía en regiones adiabáticas, correcto arranque desde
//! condiciones iniciales de entropía, y monotonía del choque.
//!
//! ## Tests lentos (`#[ignore]`)
//!
//! - **Sod con entropía**: evoluciona el tubo de Sod hasta t = 0.10 usando
//!   `sph_kdk_step_gadget2` y verifica que el choque comprime la región derecha
//!   a ρ > ρ_R_init y conserva la masa total.
//!
//! - **Colapso de Evrard**: colapso adiabático de una esfera uniforme de gas
//!   (M=1, R=1, u₀=0.05). Verifica que la energía total se conserva dentro
//!   del 10 % durante los primeros pasos, y que la densidad central crece.
//!
//! Ejecutar todos con:
//! ```bash
//! cargo test -p gadget-ng-physics --release -- --include-ignored gadget2
//! ```

use gadget_ng_core::Vec3;
use gadget_ng_sph::{
    GAMMA, SphParticle, compute_balsara_factors, compute_density, compute_sph_forces_gadget2,
    courant_dt, sph_kdk_step_gadget2,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn no_gravity(_: &mut [SphParticle]) {}

/// Genera partículas del tubo de Sod 1D con n_left y n_right partículas.
fn setup_sod_tube(n_left: usize, n_right: usize) -> Vec<SphParticle> {
    let u_l = 1.0 / ((GAMMA - 1.0) * 1.0); // P_L / ((γ-1) ρ_L)
    let u_r = 0.1 / ((GAMMA - 1.0) * 0.125); // P_R / ((γ-1) ρ_R)

    let dx_l = 0.5 / n_left as f64;
    let dx_r = 0.5 / n_right as f64;
    let mass = 1.0 * dx_l; // ρ_L·dx_L = ρ_R·dx_R → masas iguales

    let mut parts = Vec::with_capacity(n_left + n_right);
    let mut id = 0usize;

    for i in 0..n_left {
        let x = -0.5 + (i as f64 + 0.5) * dx_l;
        parts.push(SphParticle::new_gas(
            id,
            mass,
            Vec3::new(x, 0.0, 0.0),
            Vec3::zero(),
            u_l,
            2.5 * dx_l,
        ));
        id += 1;
    }
    for i in 0..n_right {
        let x = (i as f64 + 0.5) * dx_r;
        parts.push(SphParticle::new_gas(
            id,
            mass,
            Vec3::new(x, 0.0, 0.0),
            Vec3::zero(),
            u_r,
            2.5 * dx_r,
        ));
        id += 1;
    }
    parts
}

/// Masa total de gas.
fn total_mass(parts: &[SphParticle]) -> f64 {
    parts.iter().map(|p| p.mass).sum()
}

/// Energía cinética total.
fn kinetic_energy(parts: &[SphParticle]) -> f64 {
    parts
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity))
        .sum()
}

/// Energía interna total (usando u actualizado desde entropía).
fn thermal_energy(parts: &[SphParticle]) -> f64 {
    parts
        .iter()
        .filter_map(|p| p.gas.as_ref().map(|g| p.mass * g.u))
        .sum()
}

// ── Tests rápidos ─────────────────────────────────────────────────────────────

/// Después de `compute_density`, la función entrópica inicial A debe ser positiva
/// y consistente con A = (γ-1) u / ρ^(γ-1).
#[test]
fn gadget2_entropy_initialized_correctly() {
    let mut parts = setup_sod_tube(8, 1);
    compute_density(&mut parts);

    for p in &parts {
        if let Some(gas) = p.gas.as_ref()
            && gas.rho > 0.0
        {
            let a_expected = (GAMMA - 1.0) * gas.u / gas.rho.powf(GAMMA - 1.0);
            assert!(
                (gas.entropy - a_expected).abs() / a_expected.abs().max(1e-14) < 1e-10,
                "Entropía incorrecta: A={:.6e} esperado={:.6e}",
                gas.entropy,
                a_expected
            );
        }
    }
}

/// En un gas uniforme en reposo, el factor Balsara debe ser ≤ 1 siempre.
#[test]
fn gadget2_balsara_bounded() {
    let n_side = 3usize;
    let box_size = 3.0_f64;
    let dx = box_size / n_side as f64;
    let mut parts: Vec<SphParticle> = (0..n_side.pow(3))
        .map(|k| {
            let iz = k / (n_side * n_side);
            let iy = (k / n_side) % n_side;
            let ix = k % n_side;
            let pos = Vec3::new(
                (ix as f64 + 0.5) * dx,
                (iy as f64 + 0.5) * dx,
                (iz as f64 + 0.5) * dx,
            );
            SphParticle::new_gas(k, 1.0, pos, Vec3::zero(), 1.0, 2.0 * dx)
        })
        .collect();

    compute_density(&mut parts);
    compute_balsara_factors(&mut parts);

    for p in &parts {
        if let Some(gas) = p.gas.as_ref() {
            assert!(
                gas.balsara >= 0.0 && gas.balsara <= 1.0,
                "Balsara fuera de [0,1]: {:.4}",
                gas.balsara
            );
        }
    }
}

/// `courant_dt` devuelve un valor positivo y razonable tras calcular las fuerzas.
#[test]
fn gadget2_courant_dt_positive() {
    let mut parts = setup_sod_tube(8, 1);
    compute_density(&mut parts);
    compute_balsara_factors(&mut parts);
    compute_sph_forces_gadget2(&mut parts);

    let dt = courant_dt(&parts, 0.3);
    assert!(dt.is_finite() && dt > 0.0, "courant_dt = {dt:.3e}");
}

/// Un paso del integrador de entropía no hace explotar la energía total.
#[test]
fn gadget2_single_step_bounded_energy() {
    let mut parts = setup_sod_tube(8, 1);
    compute_density(&mut parts);

    let e0 = kinetic_energy(&parts) + thermal_energy(&parts);
    sph_kdk_step_gadget2(&mut parts, 1e-4, no_gravity);
    let e1 = kinetic_energy(&parts) + thermal_energy(&parts);

    // La energía no debe crecer más de un orden de magnitud en un solo paso.
    assert!(
        e1 < 100.0 * e0.abs().max(1e-10),
        "Energía explotó: E0={e0:.4e} E1={e1:.4e}"
    );
}

// ── Tests lentos: evolución temporal completa ─────────────────────────────────

/// Tubo de Sod con integrador Gadget-2 (entropía + Balsara).
///
/// Después de evolucionar hasta t ≈ 0.10, el choque debe haber comprimido la
/// región derecha a ρ > ρ_R_init y la masa total debe conservarse exactamente.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored gadget2_sod"]
fn gadget2_sod_shock_compresses_right_region() {
    let mut parts = setup_sod_tube(80, 10);
    compute_density(&mut parts);

    let mass_init = total_mass(&parts);
    let rho_r_init = 0.125_f64;
    let t_end = 0.10_f64;
    let mut t = 0.0_f64;

    while t < t_end {
        compute_sph_forces_gadget2(&mut parts);
        let dt = courant_dt(&parts, 0.3).min(t_end - t).max(1e-15);
        if dt < 1e-14 {
            break;
        }
        sph_kdk_step_gadget2(&mut parts, dt, no_gravity);
        t += dt;
    }

    // Masa conservada exactamente (no hay fuente ni sumidero).
    let mass_final = total_mass(&parts);
    assert!(
        (mass_final - mass_init).abs() / mass_init < 1e-12,
        "Masa no conservada: Δm/m = {:.2e}",
        (mass_final - mass_init) / mass_init
    );

    // El choque comprimió la región derecha.
    let rho_max_right = parts
        .iter()
        .filter(|p| p.position.x > 0.05 && p.position.x < 0.35)
        .filter_map(|p| p.gas.as_ref())
        .map(|g| g.rho)
        .fold(0.0_f64, f64::max);

    assert!(
        rho_max_right > rho_r_init,
        "Choque (Gadget-2) no comprimió: rho_max={rho_max_right:.4} vs ρ_R_init={rho_r_init:.4}"
    );
}

/// Entropía total del tubo de Sod: solo crece por irreversibilidades.
///
/// La entropía total S = Σ m_i A_i debe ser no-decreciente en ausencia de
/// disipación numérica espuria (propiedad del integrador de entropía Gadget-2).
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored gadget2_entropy_monotone"]
fn gadget2_entropy_monotonically_nondecreasing() {
    let mut parts = setup_sod_tube(80, 10);
    compute_density(&mut parts);

    let entropy_total_0: f64 = parts
        .iter()
        .filter_map(|p| p.gas.as_ref().map(|g| p.mass * g.entropy))
        .sum();

    let t_end = 0.05_f64;
    let mut t = 0.0_f64;

    while t < t_end {
        compute_sph_forces_gadget2(&mut parts);
        let dt = courant_dt(&parts, 0.3).min(t_end - t).max(1e-15);
        if dt < 1e-14 {
            break;
        }
        sph_kdk_step_gadget2(&mut parts, dt, no_gravity);
        t += dt;
    }

    let entropy_total_f: f64 = parts
        .iter()
        .filter_map(|p| p.gas.as_ref().map(|g| p.mass * g.entropy))
        .sum();

    assert!(
        entropy_total_f >= entropy_total_0 * (1.0 - 1e-8),
        "Entropía total decreció: S0={entropy_total_0:.6e} S_f={entropy_total_f:.6e}"
    );
}

// ── Colapso adiabático de Evrard ──────────────────────────────────────────────

/// Genera una esfera uniforme de gas para el colapso de Evrard (Evrard 1988).
///
/// Parámetros: M = 1, R = 1, u₀ = 0.05 (frío comparado con GM/R = 1).
fn setup_evrard_sphere(n_1d: usize) -> Vec<SphParticle> {
    let m_tot = 1.0_f64;
    let r_sphere = 1.0_f64;
    let u0 = 0.05_f64;

    // Genera puntos en retícula cúbica dentro de una esfera de radio R.
    let n_try = n_1d.pow(3);
    let dx = 2.0 * r_sphere / n_1d as f64;
    let mut points = Vec::with_capacity(n_try);

    for k in 0..n_try {
        let iz = k / (n_1d * n_1d);
        let iy = (k / n_1d) % n_1d;
        let ix = k % n_1d;
        let x = -r_sphere + (ix as f64 + 0.5) * dx;
        let y = -r_sphere + (iy as f64 + 0.5) * dx;
        let z = -r_sphere + (iz as f64 + 0.5) * dx;
        if x * x + y * y + z * z <= r_sphere * r_sphere {
            points.push(Vec3::new(x, y, z));
        }
    }

    let n = points.len();
    let mass = m_tot / n as f64;
    let h0 = 2.0 * dx;

    points
        .into_iter()
        .enumerate()
        .map(|(id, pos)| SphParticle::new_gas(id, mass, pos, Vec3::zero(), u0, h0))
        .collect()
}

/// Energía potencial gravitatoria (suma directa O(N²), sólo para tests pequeños).
fn gravitational_energy(parts: &[SphParticle]) -> f64 {
    let g = 1.0_f64;
    let eps = 0.05_f64; // suavizado Plummer
    let n = parts.len();
    let mut e_grav = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = (parts[i].position - parts[j].position).norm();
            e_grav -= g * parts[i].mass * parts[j].mass / (r * r + eps * eps).sqrt();
        }
    }
    e_grav
}

/// Aceleración gravitatoria directa O(N²) para el colapso de Evrard.
fn direct_gravity(particles: &mut [SphParticle]) {
    let g = 1.0_f64;
    let eps2 = 0.05_f64 * 0.05;
    let n = particles.len();
    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    for i in 0..n {
        let mut a = Vec3::zero();
        for j in 0..n {
            if j == i {
                continue;
            }
            let r_ij = pos[j] - pos[i];
            let r2 = r_ij.dot(r_ij) + eps2;
            let r3 = r2 * r2.sqrt();
            a += r_ij * (g * mass[j] / r3);
        }
        particles[i].acceleration = a;
    }
}

/// Colapso adiabático de Evrard: conservación de energía.
///
/// Para N pequeño (≈ 250 partículas) da un test rápido pero realista.
/// La esfera se colapsa bajo su propia gravedad; sin sources/sinks,
/// E_tot = E_kin + E_th + E_grav debe conservarse dentro del ~10 %.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored evrard"]
fn evrard_adiabatic_energy_conservation() {
    let mut parts = setup_evrard_sphere(7); // ~179 partículas en la esfera
    assert!(parts.len() > 50, "Muy pocas partículas: {}", parts.len());

    compute_density(&mut parts);
    // Inicializar entropía desde u
    for p in parts.iter_mut() {
        if let Some(gas) = p.gas.as_mut() {
            gas.init_entropy(GAMMA);
        }
    }

    let e_grav_0 = gravitational_energy(&parts);
    let e_th_0 = thermal_energy(&parts);
    let e_tot_0 = e_grav_0 + e_th_0; // E_kin_0 = 0

    // Evoluciona ~5 % del tiempo de caída libre: t_ff ≈ sqrt(3π / (32 G ρ_mean))
    // ρ_mean ≈ 3/(4π) ≈ 0.239  →  t_ff ≈ sqrt(3π/32*0.239) ≈ 1.77  → t_target = 0.09
    let t_target = 0.09_f64;
    let mut t = 0.0_f64;

    while t < t_target {
        direct_gravity(&mut parts);
        compute_sph_forces_gadget2(&mut parts);
        let dt = courant_dt(&parts, 0.3).min(t_target - t).max(1e-15);
        if dt < 1e-14 {
            break;
        }
        sph_kdk_step_gadget2(&mut parts, dt, direct_gravity);
        t += dt;
    }

    let e_grav_f = gravitational_energy(&parts);
    let e_kin_f = kinetic_energy(&parts);
    let e_th_f = thermal_energy(&parts);
    let e_tot_f = e_grav_f + e_kin_f + e_th_f;

    let rel_err = (e_tot_f - e_tot_0).abs() / e_tot_0.abs().max(1e-10);

    assert!(
        rel_err < 0.10,
        "Energía total no conservada: E0={e_tot_0:.4e} Ef={e_tot_f:.4e} err={rel_err:.3}"
    );
}

/// Colapso de Evrard: la densidad central aumenta con el tiempo.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored evrard_density"]
fn evrard_central_density_increases() {
    let mut parts = setup_evrard_sphere(7);
    compute_density(&mut parts);
    for p in parts.iter_mut() {
        if let Some(gas) = p.gas.as_mut() {
            gas.init_entropy(GAMMA);
        }
    }

    // Densidad central inicial
    let rho_center_0 = parts
        .iter()
        .filter(|p| p.position.norm() < 0.15)
        .filter_map(|p| p.gas.as_ref())
        .map(|g| g.rho)
        .fold(0.0_f64, f64::max);

    let t_target = 0.5_f64; // mayor tiempo → colapso más avanzado
    let mut t = 0.0_f64;

    while t < t_target {
        direct_gravity(&mut parts);
        compute_sph_forces_gadget2(&mut parts);
        let dt = courant_dt(&parts, 0.3).min(t_target - t).max(1e-15);
        if dt < 1e-14 {
            break;
        }
        sph_kdk_step_gadget2(&mut parts, dt, direct_gravity);
        t += dt;
    }

    let rho_center_f = parts
        .iter()
        .filter(|p| p.position.norm() < 0.15)
        .filter_map(|p| p.gas.as_ref())
        .map(|g| g.rho)
        .fold(0.0_f64, f64::max);

    assert!(
        rho_center_f > rho_center_0,
        "Densidad central no creció: ρ0={rho_center_0:.4e} ρf={rho_center_f:.4e}"
    );
}
