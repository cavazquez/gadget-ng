//! Kernel de suavizado Wendland C2 en 3D.
//!
//! ## Fórmula
//!
//! ```text
//! W(r, h) = σ₃ / h³ · (1 − q/2)⁴ · (2q + 1)   para q = r/h ∈ [0, 2]
//! W(r, h) = 0                                      para q > 2
//! ```
//!
//! Normalización en 3D: `σ₃ = 21 / (2π)`.
//!
//! El gradiente es:
//!
//! ```text
//! ∇W(r, h) = (dW/dr) · r̂ / h
//! dW/dr = σ₃ / h⁴ · (−5q(1 − q/2)³)
//! ```

/// Factor de normalización 3D del kernel Wendland C2.
/// ∫ W(r,h) 4πr² dr = 4π σ₃ · (4/21) = 1  →  σ₃ = 21 / (16π).
const SIGMA3: f64 = 21.0 / (16.0 * std::f64::consts::PI);

/// Evalúa el kernel `W(r, h)`.
#[inline]
pub fn w(r: f64, h: f64) -> f64 {
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let t = 1.0 - 0.5 * q;
    SIGMA3 / (h * h * h) * t * t * t * t * (2.0 * q + 1.0)
}

/// Evalúa `dW/dq · 1/h` (factor para el gradiente vectorial `∇W = dW/dq · r / (r·h)`).
///
/// Retorna el escalar tal que `∇W = grad_w(r, h) · r_vec / r` donde `r = |r_vec|`.
/// (Si `r = 0` retorna 0.)
#[inline]
pub fn grad_w(r: f64, h: f64) -> f64 {
    let q = r / h;
    if q >= 2.0 || r < 1e-300 {
        return 0.0;
    }
    let t = 1.0 - 0.5 * q;
    // dW/dr = σ₃ / h⁴ · (-5q · t³)
    SIGMA3 / (h * h * h * h) * (-5.0 * q * t * t * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_normalized_approx() {
        // ∫ W(r,h) 4π r² dr = 1 (integración numérica con 10 000 puntos)
        let h = 1.0_f64;
        let n = 10_000usize;
        let dr = 2.0 * h / n as f64;
        let integral: f64 = (0..n)
            .map(|k| {
                let r = (k as f64 + 0.5) * dr;
                w(r, h) * 4.0 * std::f64::consts::PI * r * r * dr
            })
            .sum();
        assert!((integral - 1.0).abs() < 1e-3, "integral = {integral:.6}");
    }

    #[test]
    fn kernel_zero_beyond_support() {
        let h = 1.5_f64;
        assert_eq!(w(3.0, h), 0.0);
        assert_eq!(w(2.0 * h, h), 0.0);
        assert_eq!(w(2.0 * h + 1e-9, h), 0.0);
    }

    #[test]
    fn gradient_sign_negative() {
        // El kernel decrece con r → dW/dr < 0.
        let h = 1.0_f64;
        assert!(grad_w(0.5, h) < 0.0, "dW/dr debe ser negativo");
    }
}
