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
//!
//! ## Funciones batch SIMD
//!
//! Las funciones `w_batch`, `grad_w_batch` y `w_and_grad_w_batch` evalúan el
//! kernel y/o su gradiente para múltiples radios con un solo `h`, usando
//! auto-vectorización vía `#[target_feature]` con dispatch en runtime:
//!
//! - **AVX-512**: procesa 8×f64 por iteración (ZMM registers).
//! - **AVX2+FMA**: procesa 4×f64 por iteración (YMM registers).
//! - **Scalar**: fallback sin SIMD.

/// Factor de normalización 3D del kernel Wendland C2.
/// ∫ W(r,h) 4πr² dr = 4π σ₃ · (4/21) = 1  →  σ₃ = 21 / (16π).
const SIGMA3: f64 = 21.0 / (16.0 * std::f64::consts::PI);

// ── Funciones escalares ──────────────────────────────────────────────────────

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
    SIGMA3 / (h * h * h * h) * (-5.0 * q * t * t * t)
}

// ── Funciones internas branch-free (auto-vectorizables) ───────────────────────

/// Kernel Wendland C2 branch-free para auto-vectorización.
///
/// Usa `q.min(2.0)` para eliminar el branch: cuando q >= 2, `t = 1 - q/2 = 0`,
/// así `t⁴ = 0` y el resultado es exactamente 0.
#[inline]
fn w_branchfree(q: f64, inv_h3: f64) -> f64 {
    let qc = q.min(2.0);
    let t = 1.0 - 0.5 * qc;
    SIGMA3 * inv_h3 * t * t * t * t * (2.0 * qc + 1.0)
}

/// Gradiente del kernel Wendland C2 branch-free para auto-vectorización.
///
/// Usa `q.min(2.0)`: cuando q >= 2, `t = 0`, resultado = 0.
/// Cuando q = 0, el resultado es 0 (porque `-5 * 0 * t³ = 0`).
#[inline]
fn grad_w_branchfree(q: f64, inv_h4: f64) -> f64 {
    let qc = q.min(2.0);
    let t = 1.0 - 0.5 * qc;
    SIGMA3 * inv_h4 * (-5.0 * qc * t * t * t)
}

// ── Bucle interno escalar (fallback) ─────────────────────────────────────────

fn w_batch_scalar(r: &[f64], h: f64, out: &mut [f64]) {
    let inv_h = 1.0 / h;
    let inv_h3 = inv_h * inv_h * inv_h;
    for (i, &ri) in r.iter().enumerate() {
        let q = ri * inv_h;
        out[i] = w_branchfree(q, inv_h3);
    }
}

fn grad_w_batch_scalar(r: &[f64], h: f64, out: &mut [f64]) {
    let inv_h = 1.0 / h;
    let inv_h4 = inv_h * inv_h * inv_h * inv_h;
    for (i, &ri) in r.iter().enumerate() {
        let q = ri * inv_h;
        out[i] = grad_w_branchfree(q, inv_h4);
    }
}

fn w_and_grad_w_batch_scalar(r: &[f64], h: f64, w_out: &mut [f64], gw_out: &mut [f64]) {
    let inv_h = 1.0 / h;
    let inv_h3 = inv_h * inv_h * inv_h;
    let inv_h4 = inv_h * inv_h3;
    for (i, &ri) in r.iter().enumerate() {
        let q = ri * inv_h;
        let qc = q.min(2.0);
        let t = 1.0 - 0.5 * qc;
        let t3 = t * t * t;
        w_out[i] = SIGMA3 * inv_h3 * t3 * t * (2.0 * qc + 1.0);
        gw_out[i] = SIGMA3 * inv_h4 * (-5.0 * qc * t3);
    }
}

// ── Kernels AVX2+FMA (4×f64 por iteración) ──────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn w_batch_avx2(r: &[f64], h: f64, out: &mut [f64]) {
    w_batch_scalar(r, h, out)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn grad_w_batch_avx2(r: &[f64], h: f64, out: &mut [f64]) {
    grad_w_batch_scalar(r, h, out)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn w_and_grad_w_batch_avx2(r: &[f64], h: f64, w_out: &mut [f64], gw_out: &mut [f64]) {
    w_and_grad_w_batch_scalar(r, h, w_out, gw_out)
}

// ── Kernels AVX-512 (8×f64 por iteración) ────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn w_batch_avx512(r: &[f64], h: f64, out: &mut [f64]) {
    w_batch_scalar(r, h, out)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn grad_w_batch_avx512(r: &[f64], h: f64, out: &mut [f64]) {
    grad_w_batch_scalar(r, h, out)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn w_and_grad_w_batch_avx512(r: &[f64], h: f64, w_out: &mut [f64], gw_out: &mut [f64]) {
    w_and_grad_w_batch_scalar(r, h, w_out, gw_out)
}

// ── API pública con dispatch en runtime ──────────────────────────────────────

/// Evalúa el kernel `W(r, h)` para múltiples radios con un solo smoothing length.
///
/// Procesa `r.len()` valores, escribiendo en `out`. Selecciona AVX-512, AVX2+FMA
/// o escalar en runtime según las capacidades de la CPU.
///
/// # Panics
///
/// Si `r.len() != out.len()`.
pub fn w_batch(r: &[f64], h: f64, out: &mut [f64]) {
    assert_eq!(
        r.len(),
        out.len(),
        "w_batch: r y out deben tener la misma longitud"
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { w_batch_avx512(r, h, out) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { w_batch_avx2(r, h, out) };
            return;
        }
    }
    w_batch_scalar(r, h, out);
}

/// Evalúa el gradiente del kernel `grad_w(r, h)` para múltiples radios con un
/// solo smoothing length.
///
/// Procesa `r.len()` valores, escribiendo en `out`. Selecciona AVX-512, AVX2+FMA
/// o escalar en runtime.
///
/// # Panics
///
/// Si `r.len() != out.len()`.
pub fn grad_w_batch(r: &[f64], h: f64, out: &mut [f64]) {
    assert_eq!(
        r.len(),
        out.len(),
        "grad_w_batch: r y out deben tener la misma longitud"
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { grad_w_batch_avx512(r, h, out) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { grad_w_batch_avx2(r, h, out) };
            return;
        }
    }
    grad_w_batch_scalar(r, h, out);
}

/// Evalúa kernel y gradiente simultáneamente para múltiples radios.
///
/// Más eficiente que llamar `w_batch` + `grad_w_batch` por separado, ya que
/// comparte el cómputo de `q`, `t`, `t³` entre ambas evaluaciones.
///
/// # Panics
///
/// Si `r.len() != w_out.len() || r.len() != gw_out.len()`.
pub fn w_and_grad_w_batch(r: &[f64], h: f64, w_out: &mut [f64], gw_out: &mut [f64]) {
    assert_eq!(
        r.len(),
        w_out.len(),
        "w_and_grad_w_batch: r y w_out deben tener la misma longitud"
    );
    assert_eq!(
        r.len(),
        gw_out.len(),
        "w_and_grad_w_batch: r y gw_out deben tener la misma longitud"
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { w_and_grad_w_batch_avx512(r, h, w_out, gw_out) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { w_and_grad_w_batch_avx2(r, h, w_out, gw_out) };
            return;
        }
    }
    w_and_grad_w_batch_scalar(r, h, w_out, gw_out);
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

    #[test]
    fn batch_matches_scalar_w() {
        let h = 1.0_f64;
        let radii: Vec<f64> = (0..100).map(|i| i as f64 * 0.04).collect();
        let mut out_batch = vec![0.0; radii.len()];
        let mut out_scalar = vec![0.0; radii.len()];

        w_batch(&radii, h, &mut out_batch);
        for (i, &r) in radii.iter().enumerate() {
            out_scalar[i] = w(r, h);
        }

        for (i, (&b, &s)) in out_batch.iter().zip(out_scalar.iter()).enumerate() {
            let rel_err = if s.abs() > 1e-30 {
                (b - s).abs() / s.abs()
            } else {
                b.abs()
            };
            assert!(
                rel_err < 1e-12,
                "w_batch[{i}]: batch={b}, scalar={s}, rel_err={rel_err:.2e}"
            );
        }
    }

    #[test]
    fn batch_matches_scalar_grad_w() {
        let h = 1.0_f64;
        let radii: Vec<f64> = (1..100).map(|i| i as f64 * 0.04).collect();
        let mut out_batch = vec![0.0; radii.len()];
        let mut out_scalar = vec![0.0; radii.len()];

        grad_w_batch(&radii, h, &mut out_batch);
        for (i, &r) in radii.iter().enumerate() {
            out_scalar[i] = grad_w(r, h);
        }

        for (i, (&b, &s)) in out_batch.iter().zip(out_scalar.iter()).enumerate() {
            let rel_err = if s.abs() > 1e-30 {
                (b - s).abs() / s.abs()
            } else {
                b.abs()
            };
            assert!(
                rel_err < 1e-12,
                "grad_w_batch[{i}]: batch={b}, scalar={s}, rel_err={rel_err:.2e}"
            );
        }
    }

    #[test]
    fn batch_combined_matches_individual() {
        let h = 0.75_f64;
        let radii: Vec<f64> = (0..80).map(|i| i as f64 * 0.04).collect();
        let mut w_out = vec![0.0; radii.len()];
        let mut gw_out = vec![0.0; radii.len()];
        let mut w_ref = vec![0.0; radii.len()];
        let mut gw_ref = vec![0.0; radii.len()];

        w_and_grad_w_batch(&radii, h, &mut w_out, &mut gw_out);

        for (i, &r) in radii.iter().enumerate() {
            w_ref[i] = w(r, h);
            gw_ref[i] = grad_w(r, h);
        }

        for (i, (&wb, &ws)) in w_out.iter().zip(w_ref.iter()).enumerate() {
            let rel_err = if ws.abs() > 1e-30 {
                (wb - ws).abs() / ws.abs()
            } else {
                wb.abs()
            };
            assert!(
                rel_err < 1e-12,
                "w_combined[{i}]: batch={wb}, scalar={ws}, rel_err={rel_err:.2e}"
            );
        }

        for (i, (&gwb, &gws)) in gw_out.iter().zip(gw_ref.iter()).enumerate() {
            let rel_err = if gws.abs() > 1e-30 {
                (gwb - gws).abs() / gws.abs()
            } else {
                gwb.abs()
            };
            assert!(
                rel_err < 1e-12,
                "gw_combined[{i}]: batch={gwb}, scalar={gws}, rel_err={rel_err:.2e}"
            );
        }
    }

    #[test]
    fn batch_zero_beyond_support() {
        let h = 1.0_f64;
        let radii: Vec<f64> = vec![2.0, 2.1, 3.0, 5.0, 100.0];
        let mut w_out = vec![0.0; radii.len()];
        let mut gw_out = vec![0.0; radii.len()];

        w_batch(&radii, h, &mut w_out);
        grad_w_batch(&radii, h, &mut gw_out);

        for (i, &v) in w_out.iter().enumerate() {
            assert!(v.abs() < 1e-15, "w_batch beyond support [{i}]: {v}");
        }
        for (i, &v) in gw_out.iter().enumerate() {
            assert!(v.abs() < 1e-15, "gw_batch beyond support [{i}]: {v}");
        }
    }

    #[test]
    fn batch_returns_exact_zero_at_q2() {
        // At q=2.0, the branch-free version should give exactly 0 (since t=0, t^4=0)
        let h = 1.0_f64;
        let radii = vec![2.0];
        let mut w_out = vec![0.0; 1];
        let mut gw_out = vec![0.0; 1];

        w_batch(&radii, h, &mut w_out);
        grad_w_batch(&radii, h, &mut gw_out);

        assert_eq!(w_out[0], 0.0, "w at q=2 should be exactly 0");
        assert_eq!(gw_out[0], 0.0, "grad_w at q=2 should be exactly 0");
    }
}
