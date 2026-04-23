//! Integrador simpléctico Yoshida de 4º orden (Yoshida 1990).
//!
//! ## Formulación
//!
//! El esquema se construye como composición triple del leapfrog KDK con pesos:
//!
//! ```text
//! c  = 2^(1/3)
//! w1 =  1 / (2 - c)   ≈ +1.3512071919596576
//! w0 = -c · w1         ≈ -1.7024143839193153
//! w1 + w0 + w1 = 1
//! ```
//!
//! La composición es `ψ₄(dt) = ψ₂(w₁·dt) ∘ ψ₂(w₀·dt) ∘ ψ₂(w₁·dt)`, donde `ψ₂`
//! es el leapfrog KDK estándar. Después de fusionar los half-kicks
//! consecutivos entre sub-pasos, el algoritmo ejecuta exactamente **4
//! evaluaciones de fuerza por paso** (vs 2 del leapfrog KDK), manteniendo
//! orden 4 de precisión local, simplecticidad exacta y reversibilidad
//! temporal.
//!
//! ## Secuencia de operaciones (Newtoniana)
//!
//! ```text
//! K(w1·dt/2)  D(w1·dt)  K((w1+w0)/2·dt)  D(w0·dt)
//! K((w0+w1)/2·dt)  D(w1·dt)  K(w1·dt/2)
//! ```
//!
//! donde `K(α)` es un kick de peso `α` y `D(β)` un drift de peso `β`.
use gadget_ng_core::{Particle, Vec3};

use crate::leapfrog::CosmoFactors;

/// Peso externo de la composición de Yoshida 4º orden.
///
/// `w₁ = 1 / (2 − 2^{1/3})`.
pub const YOSHIDA4_W1: f64 = 1.351_207_191_959_657_7_f64;

/// Peso central de la composición de Yoshida 4º orden.
///
/// `w₀ = −2^{1/3} · w₁ = 1 − 2·w₁`.
pub const YOSHIDA4_W0: f64 = -1.702_414_383_919_315_3_f64;

/// Un paso Yoshida 4º orden en forma KDK con paso `dt` fijo.
///
/// Contrato igual a [`crate::leapfrog::leapfrog_kdk_step`]:
/// - `compute` debe escribir aceleraciones alineadas con `particles`.
/// - Al finalizar, `particle.acceleration` contiene `a` al final del paso
///   (no se reutiliza entre pasos, pero se persiste por coherencia).
///
/// Coste: 4 llamadas a `compute` por paso.
pub fn yoshida4_kdk_step(
    particles: &mut [Particle],
    dt: f64,
    scratch: &mut [Vec3],
    mut compute: impl FnMut(&[Particle], &mut [Vec3]),
) {
    assert_eq!(particles.len(), scratch.len());

    // El orden de las operaciones debe coincidir con `CosmoFactors::flat(w·dt)`
    // para preservar bit-exactness con `yoshida4_cosmo_kdk_step` en modo plano.
    let k_outer = YOSHIDA4_W1 * dt * 0.5;
    let k_inner_half = YOSHIDA4_W0 * dt * 0.5;
    let k_mix = k_outer + k_inner_half;
    let d_outer = YOSHIDA4_W1 * dt;
    let d_inner = YOSHIDA4_W0 * dt;

    compute(particles, scratch);
    for (p, &a) in particles.iter_mut().zip(scratch.iter()) {
        p.velocity += a * k_outer;
    }
    for p in particles.iter_mut() {
        p.position += p.velocity * d_outer;
    }

    compute(particles, scratch);
    for (p, &a) in particles.iter_mut().zip(scratch.iter()) {
        p.velocity += a * k_mix;
    }
    for p in particles.iter_mut() {
        p.position += p.velocity * d_inner;
    }

    compute(particles, scratch);
    for (p, &a) in particles.iter_mut().zip(scratch.iter()) {
        p.velocity += a * k_mix;
    }
    for p in particles.iter_mut() {
        p.position += p.velocity * d_outer;
    }

    compute(particles, scratch);
    for (p, &a) in particles.iter_mut().zip(scratch.iter()) {
        p.velocity += a * k_outer;
        p.acceleration = a;
    }
}

/// Un paso Yoshida 4º orden cosmológico.
///
/// El caller debe pre-calcular los 3 `CosmoFactors` correspondientes a los
/// sub-intervalos de pesos `w₁·dt`, `w₀·dt`, `w₁·dt` sobre `a(t)` actual,
/// avanzando el factor de escala tras cada drift.
///
/// Pasar `[CosmoFactors::flat(w₁·dt), CosmoFactors::flat(w₀·dt),
/// CosmoFactors::flat(w₁·dt)]` recupera bit-a-bit [`yoshida4_kdk_step`].
pub fn yoshida4_cosmo_kdk_step(
    particles: &mut [Particle],
    cf: [CosmoFactors; 3],
    scratch: &mut [Vec3],
    mut compute: impl FnMut(&[Particle], &mut [Vec3]),
) {
    assert_eq!(particles.len(), scratch.len());

    compute(particles, scratch);
    for (p, &a) in particles.iter_mut().zip(scratch.iter()) {
        p.velocity += a * cf[0].kick_half;
    }
    for p in particles.iter_mut() {
        p.position += p.velocity * cf[0].drift;
    }

    compute(particles, scratch);
    for (p, &a) in particles.iter_mut().zip(scratch.iter()) {
        p.velocity += a * (cf[0].kick_half2 + cf[1].kick_half);
    }
    for p in particles.iter_mut() {
        p.position += p.velocity * cf[1].drift;
    }

    compute(particles, scratch);
    for (p, &a) in particles.iter_mut().zip(scratch.iter()) {
        p.velocity += a * (cf[1].kick_half2 + cf[2].kick_half);
    }
    for p in particles.iter_mut() {
        p.position += p.velocity * cf[2].drift;
    }

    compute(particles, scratch);
    for (p, &a) in particles.iter_mut().zip(scratch.iter()) {
        p.velocity += a * cf[2].kick_half2;
        p.acceleration = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yoshida4_weights_sum_to_one() {
        let sum = YOSHIDA4_W1 + YOSHIDA4_W0 + YOSHIDA4_W1;
        assert!(
            (sum - 1.0).abs() < 1e-14,
            "w1 + w0 + w1 debe ser exactamente 1, got {sum}"
        );
    }

    /// Con `CosmoFactors::flat` la versión cosmológica debe reproducir bit-exacto
    /// la Newtoniana plana.
    #[test]
    fn cosmo_flat_equals_newtonian() {
        let make_particle = || {
            let mut p = Particle::new(0, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.0));
            p.acceleration = Vec3::zero();
            p
        };

        let dt = 0.05_f64;
        let mut p1 = vec![make_particle()];
        let mut p2 = vec![make_particle()];
        let mut a1 = vec![Vec3::zero()];
        let mut a2 = vec![Vec3::zero()];

        let force = |ps: &[Particle], out: &mut [Vec3]| {
            out[0] = -ps[0].position;
        };

        let cf = [
            CosmoFactors::flat(YOSHIDA4_W1 * dt),
            CosmoFactors::flat(YOSHIDA4_W0 * dt),
            CosmoFactors::flat(YOSHIDA4_W1 * dt),
        ];

        yoshida4_kdk_step(&mut p1, dt, &mut a1, force);
        yoshida4_cosmo_kdk_step(&mut p2, cf, &mut a2, force);

        assert_eq!(p1[0].position, p2[0].position);
        assert_eq!(p1[0].velocity, p2[0].velocity);
    }
}
