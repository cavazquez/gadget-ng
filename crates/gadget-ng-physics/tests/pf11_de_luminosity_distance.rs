//! PF-11 — Energía oscura dinámica: distancia de luminosidad vs ΛCDM
//!
//! Verifica que la distancia de luminosidad d_L(z) con la parametrización
//! CPL (w₀=-1, wₐ=0) coincide con ΛCDM dentro del 0.1%, y que para
//! (w₀=-0.9, wₐ=0.1) la diferencia con ΛCDM es < 5% a z=1.

use gadget_ng_core::{dark_energy_eos, hubble_param, CosmologyParams};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Calcula la distancia comóvil χ(z) por integración numérica de Romberg.
/// `cosmo.advance_a` integra da/dt pero necesitamos dχ/dz = c/H(z).
/// Aquí usamos integración simple de 1/[a²·H(a)].
fn comoving_distance(cosmo: &CosmologyParams, z: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let a_start = 1.0 / (1.0 + z);
    let a_end = 1.0_f64;
    let n = 1000usize;
    let da = (a_end - a_start) / n as f64;

    // χ = ∫ c·da / (a²·H(a)) = ∫ dz / H(z) (en unidades con c=1)
    let mut chi = 0.0_f64;
    for i in 0..n {
        let a = a_start + (i as f64 + 0.5) * da;
        let h_a = hubble_param(*cosmo, a);
        chi += da / (a * a * h_a);
    }
    chi
}

/// Distancia de luminosidad: d_L = (1+z) · χ(z).
fn luminosity_distance(cosmo: &CosmologyParams, z: f64) -> f64 {
    (1.0 + z) * comoving_distance(cosmo, z)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Con w₀=-1, wₐ=0, CPL es idéntico a ΛCDM: |d_L_CPL / d_L_ΛCDM - 1| < 0.1%.
#[test]
fn cpl_de_matches_lcdm_for_w0_minus1() {
    let lcdm = CosmologyParams::new(0.3, 0.7, 0.1);
    let cpl = CosmologyParams::new_cpl(0.3, 0.7, 0.1, -1.0, 0.0);

    for &z in &[0.1, 0.5, 1.0, 2.0] {
        let dl_lcdm = luminosity_distance(&lcdm, z);
        let dl_cpl = luminosity_distance(&cpl, z);
        let rel = (dl_cpl / dl_lcdm - 1.0).abs();
        assert!(
            rel < 0.001,
            "CPL vs ΛCDM (z={z:.1}): |d_L_CPL/d_L_ΛCDM - 1| = {rel:.4e} (tolerancia 0.1%)"
        );
    }
}

/// Con w₀=-0.9, wₐ=0.1, la diferencia con ΛCDM es < 5% a z=1.
#[test]
fn cpl_w0_m09_differs_from_lcdm_by_less_than_5pct() {
    let lcdm = CosmologyParams::new(0.3, 0.7, 0.1);
    let cpl = CosmologyParams::new_cpl(0.3, 0.7, 0.1, -0.9, 0.1);

    let z = 1.0;
    let dl_lcdm = luminosity_distance(&lcdm, z);
    let dl_cpl = luminosity_distance(&cpl, z);
    let rel = (dl_cpl / dl_lcdm - 1.0).abs();

    println!("CPL (w0=-0.9, wa=0.1) vs ΛCDM a z=1: d_L diff = {:.3}%", rel * 100.0);
    assert!(
        rel < 0.05,
        "Diferencia d_L CPL vs ΛCDM (z=1): {rel:.4} (tolerancia 5%)"
    );
}

/// d_L crece monótonamente con z para modelos de energía oscura físicos.
#[test]
fn luminosity_distance_monotone_with_z() {
    let cosmo = CosmologyParams::new_cpl(0.3, 0.7, 0.1, -0.8, 0.2);
    let z_vals = [0.1, 0.5, 1.0, 2.0, 3.0];
    let dl: Vec<f64> = z_vals.iter().map(|&z| luminosity_distance(&cosmo, z)).collect();

    for i in 1..dl.len() {
        assert!(
            dl[i] > dl[i - 1],
            "d_L debe crecer con z: d_L({})={:.4}, d_L({})={:.4}",
            z_vals[i - 1], dl[i - 1],
            z_vals[i], dl[i]
        );
    }
}

/// `dark_energy_eos` devuelve w₀ para wₐ=0 a todos los redshifts.
#[test]
fn de_eos_constant_wa_zero() {
    for &a in &[0.1, 0.5, 1.0] {
        let w = dark_energy_eos(a, -0.8, 0.0);
        assert!(
            (w + 0.8).abs() < 1e-12,
            "w(a={a}, w0=-0.8, wa=0) = {w:.6} (esperado -0.8)"
        );
    }
}

/// La distancia de luminosidad a z=0 es nula.
#[test]
fn luminosity_distance_zero_at_z0() {
    let cosmo = CosmologyParams::new(0.3, 0.7, 0.1);
    let dl = luminosity_distance(&cosmo, 0.0);
    assert!(dl.abs() < 1e-8, "d_L(z=0) debe ser 0: {dl:.4e}");
}
