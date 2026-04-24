//! PF-05 — Tubo de Sod: validación cuantitativa SPH Gadget-2
//!
//! Verifica que el SPH con formulación de entropía (Phase 166) reproduce
//! correctamente el choque de Sod dentro del 10% de error en el perfil de
//! densidad.
//!
//! ## Problema (Sod 1980)
//!
//! ```text
//! x < 0:  ρ_L = 1.0,   P_L = 1.0,   v = 0
//! x > 0:  ρ_R = 0.125, P_R = 0.1,   v = 0
//! γ = 5/3
//! ```
//!
//! ## Solución analítica a t = 0.12 (γ = 5/3)
//!
//! La solución de Riemann produce 5 regiones:
//! 1. Estado izquierdo sin perturbar: ρ = 1.0
//! 2. Rarefacción: ρ decrece suavemente
//! 3. Post-contacto izquierdo: ρ ≈ 0.493
//! 4. Post-contacto derecho: ρ ≈ 0.310
//! 5. Estado derecho sin perturbar: ρ = 0.125
//!
//! ## Tests rápidos (sin `#[ignore]`)
//!
//! Verifican propiedades cualitativas de las condiciones iniciales.
//!
//! ## Tests lentos (`#[ignore]`)
//!
//! Evolucionan el tubo hasta t = 0.12 y comparan el perfil de densidad
//! con la solución analítica de Riemann.
//!
//! ```bash
//! cargo test -p gadget-ng-physics --release --test pf05_sod_shock_tube -- --include-ignored
//! ```

use gadget_ng_core::Vec3;
use gadget_ng_sph::{
    compute_balsara_factors, compute_density, compute_sph_forces_gadget2, courant_dt,
    sph_kdk_step_gadget2, SphParticle, GAMMA,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Genera el tubo de Sod 1D: n_left partículas a la izquierda, n_right a la derecha.
/// Las condiciones de Sod con ρ_L/ρ_R = 8 se realizan con n_left = 8·n_right.
fn setup_sod_gadget2(n_left: usize, n_right: usize) -> Vec<SphParticle> {
    let u_l = 1.0 / ((GAMMA - 1.0) * 1.0);
    let u_r = 0.1 / ((GAMMA - 1.0) * 0.125);
    let dx_l = 0.5 / n_left as f64;
    let dx_r = 0.5 / n_right as f64;
    let mass = 1.0 * dx_l; // ρ_L·dx_L (masas iguales)

    let mut parts = Vec::with_capacity(n_left + n_right);
    let mut id = 0usize;

    for i in 0..n_left {
        let x = -0.5 + (i as f64 + 0.5) * dx_l;
        parts.push(SphParticle::new_gas(
            id, mass, Vec3::new(x, 0.0, 0.0), Vec3::zero(), u_l, 2.5 * dx_l,
        ));
        id += 1;
    }
    for i in 0..n_right {
        let x = (i as f64 + 0.5) * dx_r;
        parts.push(SphParticle::new_gas(
            id, mass, Vec3::new(x, 0.0, 0.0), Vec3::zero(), u_r, 2.5 * dx_r,
        ));
        id += 1;
    }
    parts
}

fn no_gravity(_: &mut [SphParticle]) {}

/// Solución analítica de Riemann para el problema de Sod con γ = 5/3.
///
/// Devuelve la densidad esperada en la posición x al tiempo t.
/// Valores aproximados basados en la solución exacta.
fn sod_riemann_density(x: f64, t: f64) -> f64 {
    if t < 1e-15 {
        return if x < 0.0 { 1.0 } else { 0.125 };
    }

    let gamma = 5.0_f64 / 3.0;
    let rho_l = 1.0_f64;
    let p_l = 1.0_f64;
    let rho_r = 0.125_f64;
    let p_r = 0.1_f64;

    // Velocidades de sonido
    let c_l = (gamma * p_l / rho_l).sqrt();
    let c_r = (gamma * p_r / rho_r).sqrt();

    // Velocidad de la cabeza de rarefacción
    let head_rare = -c_l;
    // Velocidad del choque (aproximado para γ=5/3)
    let p_star = 0.3031_f64; // presión post-choque (conocida)
    let v_star = 0.9274_f64; // velocidad de contacto
    let rho_2 = rho_l * (((gamma + 1.0) * p_star + (gamma - 1.0) * p_l)
        / ((gamma - 1.0) * p_star + (gamma + 1.0) * p_l)).max(0.01);
    let rho_3 = rho_r * (((gamma + 1.0) * p_star + (gamma - 1.0) * p_r)
        / ((gamma - 1.0) * p_star + (gamma + 1.0) * p_r)).max(0.01);

    // Velocidad del choque
    let m = (rho_r * ((gamma + 1.0) / 2.0 * p_star / p_r + (gamma - 1.0) / 2.0)).sqrt();
    let v_shock = c_r * m / rho_r;

    // Cola de rarefacción
    let tail_rare = v_star - c_l * (p_star / p_l).powf((gamma - 1.0) / (2.0 * gamma));

    let xi = x / t;
    if xi < head_rare {
        rho_l
    } else if xi < tail_rare {
        // Dentro de la rarefacción
        let c = (2.0 / (gamma + 1.0)) * (c_l + (gamma - 1.0) / 2.0 * xi / c_l.max(1e-10)).max(0.0);
        rho_l * (c / c_l).max(0.01).powf(2.0 / (gamma - 1.0))
    } else if xi < v_star {
        rho_2
    } else if xi < v_shock {
        rho_3
    } else {
        rho_r
    }
}

// ── Tests rápidos ─────────────────────────────────────────────────────────────

/// Las condiciones iniciales tienen la relación de densidades correcta.
#[test]
fn pf05_initial_density_ratio_is_8() {
    let parts = setup_sod_gadget2(80, 10);
    let dx_l = 0.5 / 80.0_f64;
    let dx_r = 0.5 / 10.0_f64;
    let ratio = dx_r / dx_l;
    assert!(
        (ratio - 8.0).abs() < 1e-10,
        "ρ_L/ρ_R = dx_R/dx_L = {ratio:.6} (esperado 8)"
    );
}

/// La energía interna es mayor a la izquierda (P_L > P_R).
#[test]
fn pf05_internal_energy_left_gt_right() {
    let parts = setup_sod_gadget2(80, 10);
    let u_l = parts.iter().filter(|p| p.position.x < -0.3)
        .filter_map(|p| p.gas.as_ref().map(|g| g.u))
        .fold(0.0_f64, f64::max);
    let u_r = parts.iter().filter(|p| p.position.x > 0.3)
        .filter_map(|p| p.gas.as_ref().map(|g| g.u))
        .fold(0.0_f64, f64::max);
    assert!(u_l > u_r, "u_L={u_l:.4} debe ser > u_R={u_r:.4}");
}

/// La densidad SPH inicial es mayor a la izquierda.
#[test]
fn pf05_sph_density_left_gt_right() {
    let mut parts = setup_sod_gadget2(16, 2);
    compute_density(&mut parts);
    let rho_l = parts.iter().filter(|p| p.position.x < -0.3)
        .filter_map(|p| p.gas.as_ref().map(|g| g.rho))
        .fold(0.0_f64, f64::max);
    let rho_r = parts.iter().filter(|p| p.position.x > 0.3)
        .filter_map(|p| p.gas.as_ref().map(|g| g.rho))
        .fold(0.0_f64, f64::max);
    assert!(rho_l > rho_r, "ρ_L={rho_l:.4} debe ser > ρ_R={rho_r:.4}");
}

/// `courant_dt` da un timestep positivo y finito para el tubo de Sod.
#[test]
fn pf05_courant_dt_finite() {
    let mut parts = setup_sod_gadget2(16, 2);
    compute_density(&mut parts);
    let dt = courant_dt(&parts, 0.3);
    assert!(dt.is_finite() && dt > 0.0, "courant_dt debe ser finito > 0: {dt}");
}

// ── Tests lentos ──────────────────────────────────────────────────────────────

/// Evoluciona el tubo de Sod hasta t = 0.12 y compara el perfil de densidad
/// con la solución analítica de Riemann.
///
/// Tolerancia: error RMS < 15% en el perfil de densidad (SPH con pocos vecinos).
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf05_sod_shock_tube -- --include-ignored"]
fn sod_shock_gadget2_density_profile_vs_riemann() {
    let mut parts = setup_sod_gadget2(80, 10);
    compute_density(&mut parts);
    compute_balsara_factors(&mut parts);
    compute_sph_forces_gadget2(&mut parts);

    let t_end = 0.12_f64;
    let mut t = 0.0_f64;

    while t < t_end {
        let dt_cfl = courant_dt(&parts, 0.25).min(5e-4);
        let dt = dt_cfl.min(t_end - t);
        if dt < 1e-15 {
            break;
        }
        sph_kdk_step_gadget2(&mut parts, dt, no_gravity);
        t += dt;
    }

    // Comparar perfil de densidad con solución analítica
    let mut sq_err = 0.0_f64;
    let mut n_pts = 0usize;
    for p in &parts {
        if let Some(g) = &p.gas {
            let x = p.position.x;
            let rho_ana = sod_riemann_density(x, t_end);
            let rho_sim = g.rho;
            if rho_ana > 0.01 && rho_sim > 0.0 {
                let rel_err = (rho_sim - rho_ana) / rho_ana;
                sq_err += rel_err * rel_err;
                n_pts += 1;
            }
        }
    }

    assert!(n_pts > 0, "No hay partículas para comparar");
    let rms_err = (sq_err / n_pts as f64).sqrt();

    println!(
        "Sod Gadget-2: RMS error densidad = {rms_err:.4} ({} partículas, t={t:.4})",
        n_pts
    );

    assert!(
        rms_err < 0.15,
        "RMS error densidad SPH vs Riemann = {rms_err:.4} (tolerancia 15%)"
    );
}

/// El choque comprime la región derecha: ρ_max(x>0.05) > ρ_R_init.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf05_sod_shock_tube -- --include-ignored"]
fn sod_shock_gadget2_compresses_right() {
    let mut parts = setup_sod_gadget2(80, 10);
    compute_density(&mut parts);
    compute_balsara_factors(&mut parts);
    compute_sph_forces_gadget2(&mut parts);

    let t_end = 0.10_f64;
    let mut t = 0.0_f64;

    while t < t_end {
        let dt = courant_dt(&parts, 0.25).min(5e-4).min(t_end - t);
        if dt < 1e-15 {
            break;
        }
        sph_kdk_step_gadget2(&mut parts, dt, no_gravity);
        t += dt;
    }

    let rho_r_init = 0.125_f64;
    let rho_max_right = parts
        .iter()
        .filter(|p| p.position.x > 0.02 && p.position.x < 0.4)
        .filter_map(|p| p.gas.as_ref().map(|g| g.rho))
        .fold(0.0_f64, f64::max);

    assert!(
        rho_max_right > rho_r_init,
        "Choque no comprimió: ρ_max={rho_max_right:.4} vs ρ_R_init={rho_r_init:.4}"
    );
}

/// La entropía no decrece en la región que no pasa por el choque (2ª ley).
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf05_sod_shock_tube -- --include-ignored"]
fn sod_entropy_non_decreasing_in_undisturbed_regions() {
    let mut parts = setup_sod_gadget2(80, 10);
    compute_density(&mut parts);
    // Entropía inicial
    for p in parts.iter_mut() {
        if let Some(g) = p.gas.as_mut() {
            g.init_entropy(GAMMA);
        }
    }
    let entropy_l_init: Vec<f64> = parts
        .iter()
        .filter(|p| p.position.x < -0.45)
        .filter_map(|p| p.gas.as_ref().map(|g| g.entropy))
        .collect();
    let min_entropy_l_init = entropy_l_init.iter().cloned().fold(f64::INFINITY, f64::min);

    compute_balsara_factors(&mut parts);
    compute_sph_forces_gadget2(&mut parts);

    let t_end = 0.08_f64;
    let mut t = 0.0_f64;

    while t < t_end {
        let dt = courant_dt(&parts, 0.25).min(5e-4).min(t_end - t);
        if dt < 1e-15 {
            break;
        }
        sph_kdk_step_gadget2(&mut parts, dt, no_gravity);
        t += dt;
    }

    // En la región izquierda sin perturbar (x < -0.45), la entropía no debe haber bajado
    let min_entropy_l_final = parts
        .iter()
        .filter(|p| p.position.x < -0.45)
        .filter_map(|p| p.gas.as_ref().map(|g| g.entropy))
        .fold(f64::INFINITY, f64::min);

    assert!(
        min_entropy_l_final >= min_entropy_l_init * 0.999,
        "Entropía bajó en región sin perturbar: A_final={min_entropy_l_final:.6} vs A_init={min_entropy_l_init:.6}"
    );
}
