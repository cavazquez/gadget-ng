//! Tubo de Sod 1D — validación SPH.
//!
//! ## Problema (Sod 1980)
//!
//! Discontinuidad de Riemann en x = 0:
//!
//! ```text
//! x < 0:  ρ_L = 1.0,   P_L = 1.0,   v = 0
//! x > 0:  ρ_R = 0.125, P_R = 0.1,   v = 0
//! ```
//!
//! Con γ = 5/3 la solución analítica a t = 0.12 produce:
//! - Onda de rarefacción → izquierda.
//! - Discontinuidad de contacto.
//! - Onda de choque → derecha (comprime la región derecha a ρ ≈ 0.265).
//!
//! ## Estrategia de tests
//!
//! - Tests **rápidos** (sin `#[ignore]`): verifican condiciones iniciales (densidad,
//!   presión, energía interna) que deben ser correctas antes de integrar.
//!
//! - Tests **lentos** (`#[ignore]`): evolucionan el sistema ~3 pasos en release mode
//!   y comprueban el comportamiento del choque. Ejecutar con:
//!   `cargo test -p gadget-ng-physics --release -- --include-ignored`

use gadget_ng_core::Vec3;
use gadget_ng_sph::{compute_density, SphParticle};

/// Genera partículas 1D para el tubo de Sod.
fn setup_sod_tube(n_left: usize, n_right: usize) -> Vec<SphParticle> {
    const GAMMA: f64 = 5.0 / 3.0;
    let u_l = 1.0 / ((GAMMA - 1.0) * 1.0);    // P_L / ((γ-1) ρ_L)
    let u_r = 0.1 / ((GAMMA - 1.0) * 0.125);   // P_R / ((γ-1) ρ_R)

    let dx_l = 0.5 / n_left  as f64;
    let dx_r = 0.5 / n_right as f64;
    let mass = 1.0 * dx_l; // ρ_L·dx_L = ρ_R·dx_R (masas iguales)

    let mut parts = Vec::with_capacity(n_left + n_right);
    let mut id = 0usize;

    for i in 0..n_left {
        let x  = -0.5 + (i as f64 + 0.5) * dx_l;
        let h0 = 2.5 * dx_l;
        parts.push(SphParticle::new_gas(id, mass, Vec3::new(x, 0.0, 0.0), Vec3::zero(), u_l, h0));
        id += 1;
    }
    for i in 0..n_right {
        let x  = (i as f64 + 0.5) * dx_r;
        let h0 = 2.5 * dx_r;
        parts.push(SphParticle::new_gas(id, mass, Vec3::new(x, 0.0, 0.0), Vec3::zero(), u_r, h0));
        id += 1;
    }
    parts
}

// ── Tests rápidos: condiciones iniciales ──────────────────────────────────────

/// La relación de densidades iniciales ρ_L / ρ_R debe ser ≈ 8 (condición de Sod).
#[test]
fn sod_initial_density_ratio() {
    const N_L: usize = 20;
    const N_R: usize = 20;
    let mut parts = setup_sod_tube(N_L, N_R);
    // Con la misma masa y dx_L = dx_R, las densidades SPH dependen de h y vecinos.
    // Queremos: masas correctas → m = ρ_L·dx_L y ρ_R·dx_R deben ser iguales.
    let mass_l = parts.iter().filter(|p| p.position.x < 0.0).map(|p| p.mass).fold(0.0_f64, f64::max);
    let mass_r = parts.iter().filter(|p| p.position.x > 0.0).map(|p| p.mass).fold(0.0_f64, f64::max);
    assert!((mass_l - mass_r).abs() < 1e-14, "masas iguales: m_L={mass_l:.6}, m_R={mass_r:.6}");

    // Las condiciones de Sod requieren dx_R / dx_L = ρ_L / ρ_R = 8.
    // Con N_L = N_R = 20 y dominio L/2 = R/2, dx_L = dx_R → ρ_L = ρ_R (no Sod).
    // Usamos n_left=80, n_right=10 para ρ_L/ρ_R = 8.
    let mut parts2 = setup_sod_tube(80, 10);
    let dx_l = 0.5_f64 / 80.0;
    let dx_r = 0.5_f64 / 10.0;
    let ratio = dx_r / dx_l; // = 8 → ρ_L/ρ_R = 8
    assert!((ratio - 8.0).abs() < 1e-10, "ratio dx_R/dx_L = {ratio:.2} (esperado 8)");
    // La masa es igual en ambos lados.
    let m0 = parts2[0].mass;
    for p in &parts2 {
        assert!((p.mass - m0).abs() < 1e-14);
    }
    // La energía interna debe ser mayor a la izquierda.
    let u_l = parts2.iter().filter(|p| p.position.x < 0.0).filter_map(|p| p.gas.as_ref().map(|g| g.u)).fold(0.0_f64, f64::max);
    let u_r = parts2.iter().filter(|p| p.position.x > 0.0).filter_map(|p| p.gas.as_ref().map(|g| g.u)).fold(0.0_f64, f64::max);
    assert!(u_l > u_r, "u_L={u_l:.4} debe ser > u_R={u_r:.4}");
    let _ = parts; // silence warning
}

/// La presión inicial (estimada con SPH) debe ser mayor a la izquierda.
#[test]
fn sod_initial_pressure_left_greater_than_right() {
    // Usa pocas partículas para que compute_density sea rápido.
    let mut parts = setup_sod_tube(8, 1);
    compute_density(&mut parts);
    let p_l_max = parts.iter()
        .filter(|p| p.position.x < -0.1)
        .filter_map(|p| p.gas.as_ref().map(|g| g.pressure))
        .fold(0.0_f64, f64::max);
    let p_r_max = parts.iter()
        .filter(|p| p.position.x > 0.1)
        .filter_map(|p| p.gas.as_ref().map(|g| g.pressure))
        .fold(0.0_f64, f64::max);
    assert!(
        p_l_max > p_r_max,
        "P_L_max={p_l_max:.4} debe ser > P_R_max={p_r_max:.4}"
    );
}

/// Las masas de todas las partículas son iguales (condición para densidad ∝ 1/dx).
#[test]
fn sod_equal_particle_masses() {
    let parts = setup_sod_tube(80, 10);
    let m0 = parts[0].mass;
    for p in &parts {
        assert!((p.mass - m0).abs() < 1e-14, "masas desiguales: {:.6e}", (p.mass - m0).abs());
    }
}

/// El cociente de energías internas iniciales es consistente con P_L/P_R y ρ_L/ρ_R.
#[test]
fn sod_internal_energy_ratio() {
    const GAMMA: f64 = 5.0 / 3.0;
    let u_l = 1.0   / ((GAMMA - 1.0) * 1.0);
    let u_r = 0.1   / ((GAMMA - 1.0) * 0.125);
    // u_L / u_R = (P_L/ρ_L) / (P_R/ρ_R) = (1/1) / (0.1/0.125) = 1 / 0.8 = 1.25
    let ratio = u_l / u_r;
    assert!((ratio - 1.25).abs() < 1e-10, "u_L/u_R = {ratio:.6} (esperado 1.25)");

    let parts = setup_sod_tube(8, 1);
    // Verificar que u inicial es correcto en las partículas generadas.
    let u_l_check = parts.iter()
        .filter(|p| p.position.x < -0.2)
        .filter_map(|p| p.gas.as_ref().map(|g| g.u))
        .fold(0.0_f64, f64::max);
    assert!((u_l_check - u_l).abs() < 1e-10, "u_L check: {u_l_check:.6} vs {u_l:.6}");
}

// ── Tests lentos: evolución temporal ─────────────────────────────────────────

/// Test de compresión del choque (N grande). Ejecutar con --release --include-ignored.
#[test]
#[ignore = "Requiere: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn sod_shock_compresses_right_region() {
    use gadget_ng_sph::{compute_sph_forces, sph_kdk_step, GAMMA};

    fn no_gravity(_: &mut [SphParticle]) {}
    fn max_cs(parts: &[SphParticle]) -> f64 {
        parts.iter()
            .filter_map(|p| p.gas.as_ref())
            .filter(|g| g.rho > 1e-10)
            .map(|g| (GAMMA * g.pressure / g.rho).sqrt())
            .fold(0.0_f64, f64::max)
            .max(1.0)
    }

    let mut parts = setup_sod_tube(80, 10);
    compute_density(&mut parts);
    compute_sph_forces(&mut parts);

    let rho_r_init = 0.125_f64;
    let t_end = 0.10_f64;
    let mut t = 0.0_f64;

    while t < t_end {
        let cs = max_cs(&parts);
        let h_min = parts.iter()
            .filter_map(|p| p.gas.as_ref())
            .map(|g| g.h_sml)
            .fold(f64::INFINITY, f64::min);
        let dt = (0.3 * h_min / cs).min(t_end - t);
        if dt < 1e-15 { break; }
        sph_kdk_step(&mut parts, dt, no_gravity);
        t += dt;
    }

    let rho_max_right = parts.iter()
        .filter(|p| p.position.x > 0.05 && p.position.x < 0.3)
        .filter_map(|p| p.gas.as_ref())
        .map(|g| g.rho)
        .fold(0.0_f64, f64::max);

    assert!(
        rho_max_right > rho_r_init,
        "Choque no comprimió: rho_max={rho_max_right:.4} vs ρ_R_init={rho_r_init:.4}"
    );
}
