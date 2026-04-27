//! PF-08 — Reconexión magnética: escalado Sweet-Parker con resistividad
//!
//! Verifica que la tasa de reconexión Sweet-Parker escala como `Γ ∝ η^{1/2}`:
//!
//! ```text
//! sweet_parker_rate(v_A, l, η) = v_A / sqrt(R_m)   con R_m = v_A·l/η
//! ```
//!
//! Al variar η por un factor de 10, la tasa debe cambiar por `√10 ≈ 3.162`.

use gadget_ng_mhd::sweet_parker_rate;

/// La tasa Sweet-Parker escala como √η al variar η.
#[test]
fn sweet_parker_rate_scales_as_sqrt_eta() {
    let v_a = 1.0_f64;
    let l = 1.0_f64;
    let eta1 = 1e-3_f64;
    let eta2 = 1e-2_f64; // factor 10

    let r1 = sweet_parker_rate(v_a, l, eta1);
    let r2 = sweet_parker_rate(v_a, l, eta2);
    let ratio = r2 / r1;
    let expected = (eta2 / eta1).sqrt(); // = √10 ≈ 3.162

    assert!(
        (ratio - expected).abs() / expected < 0.01,
        "Escalado SP: ratio={ratio:.4} (esperado {expected:.4} ± 1%)"
    );
}

/// La tasa Sweet-Parker crece con v_Alfvén.
#[test]
fn sweet_parker_rate_increases_with_alfven() {
    let l = 1.0_f64;
    let eta = 1e-3_f64;
    let r1 = sweet_parker_rate(0.5, l, eta);
    let r2 = sweet_parker_rate(1.0, l, eta);
    let r3 = sweet_parker_rate(2.0, l, eta);
    assert!(
        r2 > r1,
        "Γ_SP debe crecer con v_A: r1={r1:.4e}, r2={r2:.4e}"
    );
    assert!(
        r3 > r2,
        "Γ_SP debe crecer con v_A: r2={r2:.4e}, r3={r3:.4e}"
    );
}

/// La tasa Sweet-Parker decrece con la longitud de la capa de difusión.
#[test]
fn sweet_parker_rate_decreases_with_length() {
    let v_a = 1.0_f64;
    let eta = 1e-3_f64;
    let r1 = sweet_parker_rate(v_a, 0.5, eta);
    let r2 = sweet_parker_rate(v_a, 1.0, eta);
    let r3 = sweet_parker_rate(v_a, 2.0, eta);
    assert!(
        r1 > r2,
        "Γ_SP debe decrecer con l: r1={r1:.4e} > r2={r2:.4e}"
    );
    assert!(
        r2 > r3,
        "Γ_SP debe decrecer con l: r2={r2:.4e} > r3={r3:.4e}"
    );
}

/// La fórmula Sweet-Parker es consistente: Γ = v_A / √(R_m) = v_A · √(η/(v_A·l)).
#[test]
fn sweet_parker_formula_consistent() {
    let v_a = 2.0_f64;
    let l = 0.5_f64;
    let eta = 1e-2_f64;
    let r_m = v_a * l / eta;
    let expected = v_a / r_m.sqrt();
    let result = sweet_parker_rate(v_a, l, eta);
    assert!(
        (result - expected).abs() / expected.abs() < 1e-10,
        "Fórmula SP: result={result:.6e}, expected={expected:.6e}"
    );
}

/// La tasa es positiva para parámetros físicos.
#[test]
fn sweet_parker_rate_positive() {
    let rate = sweet_parker_rate(1.0, 1.0, 1e-3);
    assert!(rate > 0.0, "La tasa Sweet-Parker debe ser positiva: {rate}");
    assert!(
        rate.is_finite(),
        "La tasa Sweet-Parker debe ser finita: {rate}"
    );
}
