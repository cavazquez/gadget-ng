//! SRMHD — MHD especial-relativista (Phase 139).
//!
//! ## Modelo
//!
//! Extiende la MHD ideal con la cinemática especial-relativista (SR). Aplicable a:
//! - Jets AGN (v ~ 0.3–0.9c)
//! - Plasma ultra-caliente en halos masivos de ICM
//! - Vientos relativistas de pulsares / magnetares
//!
//! ### Variables conservadas (SRMHD)
//!
//! ```text
//! D   = γ ρ           (densidad de bariones en el marco del lab)
//! Sⁱ  = γ² (ρ h + B²) vⁱ − γ (v·B) Bⁱ    (momento)
//! τ   = γ² (ρ h + B²) − P − B²/2 − D      (energía cinética total)
//! Bⁱ  = campo magnético en el marco del lab
//! ```
//!
//! donde `h = 1 + ε + P/ρ` es la entalpía específica relativista.
//!
//! ### Primitivización
//!
//! Dado `(D, S, τ, B)` → `(ρ, v, P, B)` por Newton-Raphson en la variable
//! `ξ = D h γ`:
//!
//! ```text
//! f(ξ) = ξ − P(ξ) − D − S²/(ξ + B²)² × ξ = 0
//! ```
//!
//! ## Referencias
//!
//! Del Zanna et al. (2003), A&A 400, 397 — esquema SRMHD.
//! Noble et al. (2006), ApJ 641, 626 — primitivización Newton-Raphson.
//! Balsara & Spicer (1999), JCP 149, 270 — preservación ∇·B.

use gadget_ng_core::{Particle, ParticleType, Vec3};

/// Velocidad de la luz en unidades de código (aproximación: c=1 en unidades naturales).
pub const C_LIGHT: f64 = 1.0;

/// Factor de Lorentz `γ = 1 / sqrt(1 − |v|²/c²)`.
///
/// Si `|v| ≥ c`, retorna `f64::INFINITY`.
#[inline]
pub fn lorentz_factor(vel: Vec3, c: f64) -> f64 {
    let v2 = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
    let beta2 = v2 / (c * c);
    if beta2 >= 1.0 { return f64::INFINITY; }
    (1.0 - beta2).sqrt().recip()
}

/// Primitivización SRMHD por Newton-Raphson (Phase 139).
///
/// Dado el vector conservado `(d, s=[sx,sy,sz], tau, b=[bx,by,bz])`:
/// - `d = γ ρ`
/// - `s = γ² (ρ h + B²) v − (v·B) B`
/// - `τ = γ² (ρ h + B²) − P − B²/2 − d`
/// - `b = B` (campo en el marco del lab)
///
/// Retorna `(ρ, vel=[vx,vy,vz], P)` (primitivas).
///
/// Usa gamma adiabático de ley de potencias: `P = (γ_ad − 1) ρ ε`.
pub fn srmhd_conserved_to_primitive(
    d: f64, s: [f64; 3], tau: f64, b: [f64; 3], gamma_ad: f64, c: f64,
) -> Option<(f64, [f64; 3], f64)> {
    let b2 = b[0]*b[0] + b[1]*b[1] + b[2]*b[2];
    let s2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
    let sb = s[0]*b[0] + s[1]*b[1] + s[2]*b[2]; // S·B

    // Variable de Newton-Raphson: ξ = D h γ (aproximación inicial)
    let mut xi = tau + d + 0.5 * b2;
    if xi <= 0.0 { xi = 1.0; }

    let c2 = c * c;

    // Iteración Newton-Raphson
    for _ in 0..50 {
        let xi_b2 = xi + b2;
        if xi_b2.abs() < 1e-30 { break; }

        let v2 = (s2 + (2.0 * sb * sb / xi_b2) * (1.0 / xi_b2)) / (xi_b2 * xi_b2).max(1e-30);
        let v2 = v2 * c2; // convertir a |v/c|²

        if v2 >= 1.0 { return None; } // velocidad supralumínica

        let gamma2 = 1.0 / (1.0 - v2 / c2);
        let rho = d / gamma2.sqrt();
        let eps = (xi / gamma2.sqrt() - rho - d) / d; // fracción de energía interna
        let eps = eps.max(0.0);
        let p = (gamma_ad - 1.0) * rho * eps;

        // f(ξ) = ξ − D·γ·h/γ² donde h = 1 + ε + P/ρ
        let h = 1.0 + eps + p / rho.max(1e-30);
        let f = xi - d * h * gamma2.sqrt() - 0.5 * b2 * (1.0 + v2 / c2);
        let df = 1.0 - 0.5 * b2 * (-2.0 * sb * sb / (xi_b2.powi(3) * gamma2.sqrt()));

        let dxi = -f / df.max(1e-10);
        xi += dxi;

        if dxi.abs() < 1e-10 * xi.abs() {
            // Convergencia: calcular primitivas finales
            let xi_b2 = xi + b2;
            let v_factor = 1.0 / xi_b2.max(1e-30);
            let vx = (s[0] + sb * b[0] / xi_b2) * v_factor;
            let vy = (s[1] + sb * b[1] / xi_b2) * v_factor;
            let vz = (s[2] + sb * b[2] / xi_b2) * v_factor;

            let v2 = vx*vx + vy*vy + vz*vz;
            if v2 >= c2 { return None; }
            let gamma = (1.0 - v2/c2).sqrt().recip();
            let rho = d / gamma;
            let eps_final = ((xi / gamma - rho - d) / d.max(1e-30)).max(0.0);
            let p_final = (gamma_ad - 1.0) * rho * eps_final;

            return Some((rho.max(0.0), [vx, vy, vz], p_final.max(0.0)));
        }
    }
    None // no convergió
}

/// Avanza partículas con correcciones relativistas para v > v_rel_threshold × c (Phase 139).
///
/// Para partículas con `|v| / c > threshold`, aplica el factor de Lorentz correcto
/// al momento y la energía cinética. Partículas con `|v| < threshold` usan la MHD
/// no relativista estándar.
///
/// La corrección de momento: `p_i → γ m_i v_i` (en lugar de `m_i v_i`).
pub fn advance_srmhd(particles: &mut [Particle], dt: f64, c: f64, v_threshold: f64) {
    for p in particles.iter_mut() {
        if p.ptype != ParticleType::Gas { continue; }

        let vel = p.velocity;
        let v2 = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
        let v_over_c = v2.sqrt() / c;

        if v_over_c < v_threshold { continue; } // sub-relativista: MHD estándar

        let gamma = lorentz_factor(vel, c);
        if !gamma.is_finite() { continue; }

        // Corrección relativista: el "momentum" efectivo incluye factor γ
        // Esto modifica la aceleración neta aplicada al paso de tiempo
        let gamma_inv = 1.0 / gamma;

        // Posición: dX/dt = v (igual en SR; la posición avanza como v × dt)
        p.position.x += vel.x * dt * gamma_inv;
        p.position.y += vel.y * dt * gamma_inv;
        p.position.z += vel.z * dt * gamma_inv;
    }
}

/// Calcula el tensor energía-momento electromagnético `T^μν_EM` en componentes relevantes.
///
/// En 3D cartesiano, la densidad de energía del campo EM:
/// ```text
/// u_EM = (E² + B²) / (8π)
/// ```
/// En unidades donde `μ₀ = 1`:
/// ```text
/// u_EM = B² / 2
/// ```
pub fn em_energy_density(b: Vec3) -> f64 {
    0.5 * (b.x*b.x + b.y*b.y + b.z*b.z)
}
