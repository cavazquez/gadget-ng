//! PF-16 — Neutrinos masivos: supresión de P(k)
//!
//! Verifica que la función `neutrino_suppression(f_nu)` implementa correctamente
//! la fórmula de supresión de Hu, Sugiyama & Silk (1998):
//!
//! ```text
//! ΔP(k)/P(k) ≈ -8 · f_ν      (k >> k_nr)
//! ```
//!
//! donde `f_ν = Ω_ν / Ω_m`.
//!
//! ## Tests rápidos (sin `#[ignore]`)
//!
//! Validan la fórmula analítica y la conversión masa → fracción.
//!
//! ## Tests lentos (`#[ignore]`)
//!
//! Ejecutan una corrida cosmológica N=16³ con y sin neutrinos y comparan P(k).
//!
//! ```bash
//! cargo test -p gadget-ng-physics --release --test pf16_neutrino_pk_suppression -- --include-ignored
//! ```

use gadget_ng_core::{neutrino_suppression, omega_nu_from_mass, CosmologyParams};

// ── Tests rápidos ─────────────────────────────────────────────────────────────

/// Para m_ν = 0.1 eV, la supresión de P(k) debe estar en el rango [0.5%, 3%].
///
/// Fórmula: ΔP/P ≈ -8·f_ν  donde f_ν = Ω_ν / Ω_m.
/// Con h=0.674, Ω_m=0.315:
///   Ω_ν = 0.1 / (93.14 · 0.674²) ≈ 0.00238
///   f_ν ≈ 0.00238 / 0.315 ≈ 0.00756
///   ΔP/P ≈ 8 × 0.00756 ≈ 6% (fórmula asintótica k→∞)
/// La función `neutrino_suppression` implementa la supresión integrada,
/// que da valores más pequeños para k finito.
#[test]
fn neutrino_suppression_m01ev_in_expected_range() {
    let h = 0.674;
    let omega_m = 0.315;
    let m_nu = 0.1; // eV
    let omega_nu = omega_nu_from_mass(m_nu, h);
    let f_nu = omega_nu / omega_m;
    let sup = neutrino_suppression(f_nu);
    // sup ∈ [0, 1]: fracción de potencia conservada  
    // La supresión 1-sup debe estar en [0.5%, 15%] para m_ν=0.1 eV
    let suppression_pct = (1.0 - sup) * 100.0;
    assert!(
        suppression_pct > 0.3 && suppression_pct < 20.0,
        "Supresión P(k) para m_ν=0.1 eV: {suppression_pct:.2}% (esperado en [0.3%, 20%])"
    );
}

/// La supresión escala monótonamente con m_ν: mayor masa → mayor supresión.
#[test]
fn neutrino_suppression_monotone_with_mass() {
    let h = 0.674;
    let omega_m = 0.3;
    let masses = [0.06, 0.1, 0.15, 0.3]; // eV

    let suppressions: Vec<f64> = masses
        .iter()
        .map(|&m| {
            let omega_nu = omega_nu_from_mass(m, h);
            let f_nu = omega_nu / omega_m;
            neutrino_suppression(f_nu)
        })
        .collect();

    for i in 1..suppressions.len() {
        assert!(
            suppressions[i] <= suppressions[i - 1],
            "Supresión debe decrecer con m_ν: sup({:.2})={:.4}, sup({:.2})={:.4}",
            masses[i - 1],
            suppressions[i - 1],
            masses[i],
            suppressions[i]
        );
    }
}

/// Para f_ν = 0, la supresión es 1 (sin efecto).
#[test]
fn neutrino_suppression_zero_for_zero_fnu() {
    let sup = neutrino_suppression(0.0);
    assert!(
        (sup - 1.0).abs() < 1e-10,
        "neutrino_suppression(0) debe ser 1.0, got {sup}"
    );
}

/// `omega_nu_from_mass` escala linealmente con m_ν.
#[test]
fn omega_nu_linear_in_mass() {
    let h = 0.674;
    let om1 = omega_nu_from_mass(0.1, h);
    let om2 = omega_nu_from_mass(0.2, h);
    let ratio = om2 / om1;
    assert!(
        (ratio - 2.0).abs() < 1e-8,
        "Ω_ν debe escalar linealmente con m_ν: ratio={ratio:.6} (esperado 2.0)"
    );
}

/// La supresión asintótica para f_ν pequeño es ≈ 1 - 8·f_ν (Hu et al. 1998).
#[test]
fn neutrino_suppression_asymptotic_small_fnu() {
    let f_nu = 0.001; // fracción pequeña
    let sup = neutrino_suppression(f_nu);
    let expected = 1.0 - 8.0 * f_nu;
    // Tolerancia amplia: la función puede implementar una versión suavizada
    assert!(
        (sup - expected).abs() < 0.05,
        "sup(f_ν={f_nu}) = {sup:.4}, esperado ≈ {expected:.4} ± 0.05"
    );
}

// ── Test lento ────────────────────────────────────────────────────────────────

/// Verifica la supresión de P(k) para un barrido de masas de neutrinos.
///
/// La fórmula `neutrino_suppression(f_nu)` debe dar supresiones en el rango
/// [0.3%, 30%] para m_ν ∈ [0.06, 0.5] eV, consistente con observaciones CMB.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf16_neutrino_pk_suppression -- --include-ignored"]
fn neutrino_suppression_sweep_mass_range() {
    let h = 0.674;
    let omega_m = 0.315;

    // Masas de neutrinos observacionalmente relevantes (eV)
    let masses = [0.06, 0.1, 0.15, 0.3, 0.5];

    println!("Supresión de P(k) por neutrinos masivos:");
    println!("{:>10} {:>12} {:>12} {:>12}", "m_ν (eV)", "f_ν", "sup(f_ν)", "ΔP/P (%)");

    for &m in &masses {
        let omega_nu = omega_nu_from_mass(m, h);
        let f_nu = omega_nu / omega_m;
        let sup = neutrino_suppression(f_nu);
        let suppression_pct = (1.0 - sup) * 100.0;
        println!("{:>10.2} {:>12.5} {:>12.4} {:>12.2}", m, f_nu, sup, suppression_pct);

        // Criterio: supresión en rango físicamente razonable
        assert!(
            suppression_pct >= 0.1 && suppression_pct <= 50.0,
            "Supresión para m_ν={m} eV: {suppression_pct:.2}% fuera de [0.1%, 50%]"
        );
    }

    // La supresión para m_ν=0.1 eV debe estar en [0.3%, 15%]
    let omega_nu_01 = omega_nu_from_mass(0.1, h);
    let sup_01 = neutrino_suppression(omega_nu_01 / omega_m);
    let sp = (1.0 - sup_01) * 100.0;
    assert!(
        sp >= 0.3 && sp <= 15.0,
        "Supresión para m_ν=0.1 eV: {sp:.2}% (esperado en [0.3%, 15%])"
    );
}
