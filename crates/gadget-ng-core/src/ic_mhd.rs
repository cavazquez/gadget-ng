//! Condiciones iniciales para campo magnético primordial (Phase 161 / V3).
//!
//! ## Física
//!
//! Un campo magnético primordial comoving se genera con espectro de potencias:
//!
//! ```text
//! P_B(k) ∝ k^n_B,   n_B típico: -2.9 (nearly scale-invariant)
//! ```
//!
//! El módulo implementa dos modos:
//!
//! 1. **Campo uniforme**: `b0` en dirección z, sin variación espacial. Útil para
//!    tests de ondas de Alfvén y flux-freeze.
//!
//! 2. **Campo aleatorio Gaussiano**: amplitudes en espacio de k con fase aleatoria,
//!    transformada inversa a espacio real. Satisface `∇·B = 0` por construcción
//!    (usando solo la parte transversal del campo).
//!
//! ## Normalización
//!
//! La presión magnética comoving es `P_B = |B|²/(8π)` (unidades Gaussianas) o
//! `P_B = |B|²/2` si el código usa `μ₀ = 1` (unidades internas).
//!
//! El parámetro β = P_gas / P_mag debe ser >> 1 en las ICs cosmológicas (el campo
//! no debe dominar sobre la presión térmica en el universo temprano).
//! Para B₀ comoving ~ 1 nGauss a z=50, β ~ 10⁶.
//!
//! ## Hermeticidad
//!
//! Para el campo aleatorio, se impone simetría Hermitiana en espacio-k y se usa
//! solo la componente transversal para garantizar `∇·B = 0`.
//!
//! ## Unidades
//!
//! El valor `b0` está en las unidades internas del código. Para convertir desde
//! nGauss comoving: `B_int = B_nG * unidades_B_factor`.

use crate::{particle::Particle, vec3::Vec3};

// ── Generador LCG simple (reproducible, sin dependencias) ─────────────────────

fn lcg_u64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f64) / (u32::MAX as f64)
}

/// Genera dos valores normales N(0,1) con Box-Muller.
fn lcg_normal2(state: &mut u64) -> (f64, f64) {
    let u1 = lcg_u64(state).max(1e-300);
    let u2 = lcg_u64(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

// ─────────────────────────────────────────────────────────────────────────────
// API pública
// ─────────────────────────────────────────────────────────────────────────────

/// Asigna un campo magnético uniforme `B = (0, 0, b0)` a todas las partículas.
///
/// Es el modo más simple para tests de ondas de Alfvén: el campo no tiene
/// variación espacial, lo que permite comparar directamente con la solución
/// analítica (velocidad de Alfvén `v_A = b0/√ρ` en unidades con `μ₀=1`).
///
/// # Parámetros
/// - `particles` — slice mutable de partículas (gas o DM)
/// - `b0` — amplitud en unidades internas del código
pub fn uniform_bfield_ic(particles: &mut [Particle], b0: f64) {
    for p in particles.iter_mut() {
        p.b_field = Vec3::new(0.0, 0.0, b0);
    }
}

/// Asigna campo magnético inicial con espectro de potencias B(k) ∝ k^`spectral_index`.
///
/// El campo resultante es solenoidal (`∇·B = 0`) por construcción: se genera en
/// espacio de Fourier usando solo la componente transversal y luego se transforma
/// de vuelta al espacio real mediante IFFT.
///
/// Para tests cosmológicos, `spectral_index = -2.9` reproduce el espectro
/// "nearly scale-invariant" estándar.
///
/// # Parámetros
/// - `particles` — debe ser un grid regular 1D o 3D ordenado por posición X
/// - `b0` — amplitud RMS del campo en unidades internas
/// - `spectral_index` — índice espectral `n_B` de la potencia (`P_B ∝ k^n_B`)
/// - `seed` — semilla para reproducibilidad
///
/// # Limitación
///
/// La implementación actual usa una aproximación 1D a lo largo de la dirección X
/// para el campo `B_y`. Esta es suficiente para el test V3-T5 (β_plasma).
/// Una implementación 3D completa requiere FFT 3D y se puede agregar en el futuro.
pub fn primordial_bfield_ic(
    particles: &mut [Particle],
    b0: f64,
    spectral_index: f64,
    seed: u64,
) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    let mut rng = seed;

    // Generar amplitudes espectrales B(k) ∝ k^(n_B/2) con fase aleatoria.
    // Para la componente B_y como función de la posición x.
    let mut by_field = vec![0.0_f64; n];

    for ki in 1..=(n / 2) {
        let k = ki as f64;
        // Amplitud espectral: σ_k ∝ k^(n_B/2)
        let sigma = b0 * k.powf(spectral_index / 2.0);
        let (re, im) = lcg_normal2(&mut rng);
        let amp_re = sigma * re;
        let amp_im = sigma * im;

        // Transformada inversa discreta (DFT inversa parcial)
        for (xi, by) in by_field.iter_mut().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * ki as f64 * xi as f64 / n as f64;
            *by += 2.0 * (amp_re * phase.cos() - amp_im * phase.sin()) / n as f64;
        }
    }

    // Normalizar al RMS pedido
    let rms = (by_field.iter().map(|b| b * b).sum::<f64>() / n as f64).sqrt();
    let scale = if rms > 0.0 { b0 / rms } else { 1.0 };

    for (i, p) in particles.iter_mut().enumerate() {
        p.b_field = Vec3::new(0.0, by_field[i] * scale, 0.0);
    }
}

/// Calcula el parámetro β_plasma medio = P_gas / P_mag.
///
/// - `P_gas = (γ-1) · u · ρ` donde `ρ = masa/vol` con vol estimado como
///   `(smoothing_length)³` para SPH, o `1/N` para distribución uniforme.
/// - `P_mag = |B|² / 2` (unidades internas con `μ₀ = 1`).
///
/// Devuelve `f64::INFINITY` si no hay campo magnético (`|B| = 0` en todas las
/// partículas), lo que se interpreta como "beta infinito" (sin campo).
///
/// # Parámetros
/// - `particles` — slice de partículas (debe incluir partículas de gas con `u > 0`)
/// - `gamma` — índice adiabático (típico: 5/3)
pub fn check_plasma_beta(particles: &[Particle], gamma: f64) -> f64 {
    let gas: Vec<&Particle> = particles.iter().filter(|p| p.internal_energy > 0.0).collect();
    if gas.is_empty() {
        return f64::INFINITY;
    }

    let mut sum_beta = 0.0;
    let mut n_counted = 0usize;

    for p in &gas {
        let b2 = p.b_field.dot(p.b_field);
        if b2 < 1e-300 {
            continue;
        }
        // Estimación de densidad: masa / volumen de la celda de suavizado
        let h = p.smoothing_length;
        let rho = if h > 0.0 {
            p.mass / (h * h * h)
        } else {
            // Fallback: asumir densidad unitaria
            1.0
        };
        let p_gas = (gamma - 1.0) * p.internal_energy * rho;
        let p_mag = b2 / 2.0;
        if p_mag > 0.0 {
            sum_beta += p_gas / p_mag;
            n_counted += 1;
        }
    }

    if n_counted == 0 {
        return f64::INFINITY;
    }
    sum_beta / n_counted as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests unitarios
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::ParticleType;

    fn gas_particle(id: usize, x: f64, u: f64, h: f64) -> Particle {
        let mut p = Particle::new_gas(id, 1.0, Vec3::new(x, 0.0, 0.0), Vec3::zero(), u, h);
        p.ptype = ParticleType::Gas;
        p
    }

    #[test]
    fn uniform_bfield_sets_bz() {
        let mut ps: Vec<Particle> = (0..8)
            .map(|i| gas_particle(i, i as f64, 1.0, 0.5))
            .collect();
        uniform_bfield_ic(&mut ps, 2.0);
        for p in &ps {
            assert!((p.b_field.z - 2.0).abs() < 1e-14);
            assert_eq!(p.b_field.x, 0.0);
            assert_eq!(p.b_field.y, 0.0);
        }
    }

    #[test]
    fn primordial_bfield_rms_matches_b0() {
        let n = 64;
        let b0 = 0.5;
        let mut ps: Vec<Particle> = (0..n)
            .map(|i| gas_particle(i, i as f64 / n as f64, 1.0, 0.5))
            .collect();
        primordial_bfield_ic(&mut ps, b0, -2.9, 42);
        let rms = (ps.iter().map(|p| p.b_field.y.powi(2)).sum::<f64>() / n as f64).sqrt();
        assert!((rms - b0).abs() / b0 < 0.02, "RMS={rms:.4} b0={b0}");
    }

    #[test]
    fn check_plasma_beta_returns_infinity_without_field() {
        let mut ps: Vec<Particle> = (0..4)
            .map(|i| gas_particle(i, i as f64, 2.0, 0.5))
            .collect();
        // Sin campo magnético
        let beta = check_plasma_beta(&ps, 5.0 / 3.0);
        assert!(beta.is_infinite(), "beta={beta}");

        // Con campo: beta finito
        uniform_bfield_ic(&mut ps, 1.0);
        let beta2 = check_plasma_beta(&ps, 5.0 / 3.0);
        assert!(beta2.is_finite(), "beta2={beta2}");
        assert!(beta2 > 0.0);
    }
}
