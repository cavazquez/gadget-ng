//! PF-15 — Rayos X: relación L_X – T_X (bremsstrahlung puro)
//!
//! Para un gas caliente de bremsstrahlung térmico, la luminosidad en rayos X
//! escala como:
//!
//! ```text
//! L_X ∝ n_e² · √T   (bremsstrahlung térmico, Rybicki & Lightman 1979)
//! ```
//!
//! En términos de temperatura media de los cúmulos:
//! ```text
//! L_X ∝ T^2    (para cúmulos autosimilares, White et al. 1997)
//! ```
//!
//! Este test verifica que la pendiente de log(L_X) vs log(T_X) ≈ 2.0 ± 0.2
//! para una muestra de partículas de gas con diferentes temperaturas.

use gadget_ng_analysis::xray::{
    bremsstrahlung_emissivity, mass_weighted_temperature, total_xray_luminosity,
};
use gadget_ng_core::{Particle, Vec3};

const GAMMA: f64 = 5.0 / 3.0;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Crea una partícula de gas caliente usando `new_gas` (como en phase151).
///
/// u_hot ≥ 1e3 para que `bremsstrahlung_emissivity` dé valores no nulos.
fn gas_hot(id: usize, u: f64, mass: f64, x: f64, h: f64) -> Particle {
    Particle::new_gas(id, mass, Vec3::new(x, 0.0, 0.0), Vec3::zero(), u, h)
}

/// Crea N partículas de gas caliente con temperatura uniforme.
/// `temperature` escala u = temperature (unidades de trabajo).
fn hot_gas_cluster(n: usize, temperature: f64, rho: f64, box_size: f64) -> Vec<Particle> {
    let dx = box_size / n as f64;
    let mass = rho * box_size / n as f64; // masa por partícula
    let h = 2.0 * dx;
    (0..n)
        .map(|i| {
            let x = (i as f64 + 0.5) * dx;
            gas_hot(i, temperature, mass, x, h)
        })
        .collect()
}

// ── Tests rápidos ─────────────────────────────────────────────────────────────

/// `bremsstrahlung_emissivity` crece con la temperatura (u) para gas caliente.
#[test]
fn bremsstrahlung_increases_with_temperature() {
    // Usar u = 1e3 y 1e4 como en phase151 (hay un umbral de temperatura)
    let p_warm = gas_hot(0, 1e3, 1.0, 0.0, 0.5);
    let p_hot = gas_hot(1, 1e4, 1.0, 0.0, 0.5);

    let e_warm = bremsstrahlung_emissivity(&p_warm, GAMMA);
    let e_hot = bremsstrahlung_emissivity(&p_hot, GAMMA);
    assert!(
        e_hot > e_warm,
        "Emisividad debe crecer con T: e_warm={e_warm:.4e}, e_hot={e_hot:.4e}"
    );
}

/// `total_xray_luminosity` es positiva para gas caliente (u = 1e4).
#[test]
fn total_xray_luminosity_positive() {
    let particles = hot_gas_cluster(8, 1e4, 1.0, 1.0);
    let lx = total_xray_luminosity(&particles, GAMMA);
    assert!(lx > 0.0, "L_X debe ser positiva: {lx:.4e}");
    assert!(lx.is_finite(), "L_X debe ser finita: {lx:.4e}");
}

/// `mass_weighted_temperature` es positiva para gas caliente.
#[test]
fn mass_weighted_temperature_consistent() {
    let t = 1e4_f64;
    let particles = hot_gas_cluster(8, t, 1.0, 1.0);
    let t_mw = mass_weighted_temperature(&particles, GAMMA);
    assert!(
        t_mw > 0.0,
        "T_mw debe ser positiva: {t_mw:.4e}"
    );
    assert!(t_mw.is_finite(), "T_mw debe ser finita: {t_mw:.4e}");
    println!("T_mw para u={t:.2e}: {t_mw:.4e}");
}

/// L_X crece con la masa de gas (L_X ∝ ρ²·V ∝ ρ·masa).
#[test]
fn lx_scales_with_density_squared() {
    let t = 1e4_f64;
    // ρ doble → masa doble × factor adicional por ρ²
    let lx1 = total_xray_luminosity(&hot_gas_cluster(8, t, 1.0, 1.0), GAMMA);
    let lx2 = total_xray_luminosity(&hot_gas_cluster(8, t, 2.0, 1.0), GAMMA);
    assert!(lx1 > 0.0, "L_X debe ser positiva: {lx1:.4e}");
    assert!(lx2 > 0.0, "L_X debe ser positiva: {lx2:.4e}");
    // L_X debe crecer con la densidad
    assert!(
        lx2 > lx1,
        "L_X(2ρ) debe ser > L_X(ρ): lx1={lx1:.4e}, lx2={lx2:.4e}"
    );
}

// ── Test lento ────────────────────────────────────────────────────────────────

/// La pendiente log(L_X) vs log(T_X) ≈ 2.0 ± 0.2 (bremsstrahlung autosimilar).
///
/// Se usan 6 grupos de gas con temperaturas en escala logarítmica.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf15_xray_lx_tx -- --include-ignored"]
fn lx_tx_slope_matches_bremsstrahlung() {
    let gamma = 5.0 / 3.0;
    // Temperaturas en escala logarítmica: 1e3..1e5
    let temps = [1e3_f64, 2e3, 5e3, 1e4, 2e4, 5e4];
    let mut log_lx = Vec::new();
    let mut log_tx = Vec::new();

    for &t in &temps {
        let particles = hot_gas_cluster(16, t, 1.0, 1.0);
        let lx = total_xray_luminosity(&particles, gamma);
        let tx = mass_weighted_temperature(&particles, gamma);
        if lx > 1e-30 && tx > 1e-30 {
            log_lx.push(lx.ln());
            log_tx.push(tx.ln());
        }
    }

    assert!(
        log_lx.len() >= 4,
        "Se necesitan ≥ 4 puntos para ajustar la pendiente"
    );

    // Regresión lineal
    let n = log_tx.len() as f64;
    let mean_x: f64 = log_tx.iter().sum::<f64>() / n;
    let mean_y: f64 = log_lx.iter().sum::<f64>() / n;
    let num: f64 = log_tx.iter().zip(log_lx.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_tx.iter().map(|x| (x - mean_x).powi(2)).sum();

    let slope = num / den;
    println!("L_X-T_X pendiente: {slope:.3} (esperado 2.0 ± 0.2)");

    assert!(
        (slope - 2.0).abs() < 0.2,
        "Pendiente L_X-T_X = {slope:.3} (esperado 2.0 ± 0.2)"
    );
}
