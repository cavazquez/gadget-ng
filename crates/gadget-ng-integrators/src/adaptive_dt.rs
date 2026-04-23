//! Timestep adaptativo global (Phase 43).
//!
//! Implementa un único criterio de paso temporal **global** (un `dt` compartido
//! por todas las partículas, recalculado al final de cada paso base) pensado
//! para integradores Leapfrog KDK como [`crate::leapfrog_cosmo_kdk_step`].
//!
//! No es un block-timestep jerárquico (eso ya existe en
//! [`crate::hierarchical`]); es una envolvente mínima para probar si reducir
//! `dt` según la dinámica real mejora la fidelidad del crecimiento lineal sin
//! tener que reescribir el solver.
//!
//! ## Criterios disponibles
//!
//! * [`AdaptiveDtCriterion::Fixed`] — control a `dt` constante (equivalente a
//!   no usar adaptativo, expuesto para uniformar la API del caller).
//! * [`AdaptiveDtCriterion::Acceleration`] — criterio tipo Aarseth global:
//!   `dt = η · sqrt(ε / a_max)` con `a_max = max_i |a_i|`. `ε` es el
//!   softening físico (mismo que usa el solver).
//! * [`AdaptiveDtCriterion::CosmoAcceleration`] — combina el criterio de
//!   aceleración con una cota cosmológica `dt ≤ κ · H(a)^{-1} · a`. Esto
//!   evita que `dt` crezca sin control en épocas tempranas donde `a_max` es
//!   pequeño (típico en 2LPT ICs a `a_init = 0.02`) y mantiene la expansión
//!   de fondo resuelta con al menos `1/κ` pasos por Hubble.
//!
//! ## Integrador KDK no se rompe si `dt` cambia entre pasos
//!
//! El Leapfrog KDK es simpléctico sólo con `dt` **constante** dentro de un
//! paso. Si cambiamos `dt` al final de un paso y usamos el nuevo `dt` en el
//! siguiente, perdemos simplecticidad formal pero mantenemos:
//!
//! 1. **Conservación de segundo orden** en cada paso individual
//!    (`O(dt²)` de error local).
//! 2. **Reversibilidad temporal** si los kicks ven el mismo `dt` inicial y
//!    final (garantizado por KDK).
//! 3. **Estabilidad lineal** mientras `dt ≤ dt_stab ≈ 2/ω_max`, donde
//!    `ω_max = sqrt(|a_max|/ε)` es la frecuencia característica del modo más
//!    rápido; el criterio `dt = η · sqrt(ε/a_max)` con `η ≤ 0.25` sobreestima
//!    `dt_stab` por factores ≥ 8×.
//!
//! Formalmente es un "symplectic with small drift" sobre pasos largos; la
//! energía puede crecer linealmente en `T = N·dt`, pero para evoluciones
//! tempranas (~cientos de pasos, hasta `a ≤ 0.1`) el drift es despreciable
//! frente al error físico de amplitud inicial (Phase 37/39).
//!
//! ## Clamps
//!
//! Los parámetros [`AdaptiveDtCriterion::dt_min`] y [`AdaptiveDtCriterion::dt_max`]
//! imponen un intervalo de `dt` admisible. El `dt_max` evita saltos grandes
//! cuando la dinámica está congelada (ICs homogéneos); `dt_min` evita
//! deadlocks numéricos si un par de partículas se acerca patológicamente.

use gadget_ng_core::{cosmology::CosmologyParams, Vec3};

/// Selección del criterio de `dt` global.
#[derive(Debug, Clone, Copy)]
pub enum AdaptiveDtCriterion {
    /// Paso temporal constante: devuelve siempre el mismo valor.
    Fixed(f64),
    /// `dt = η · sqrt(ε / a_max)` (tipo Aarseth), clamped a `[dt_min, dt_max]`.
    Acceleration {
        /// Coeficiente dimensionless. Valores típicos 0.01–0.1.
        eta: f64,
        /// Softening físico (misma unidad que el solver). `>0`.
        eps: f64,
        /// Paso mínimo absoluto.
        dt_min: f64,
        /// Paso máximo absoluto.
        dt_max: f64,
    },
    /// Combina [`Acceleration`] con una cota cosmológica
    /// `dt ≤ kappa_h · a / H(a)`. Usa el valor más restrictivo.
    ///
    /// Con `kappa_h = 0.02` garantiza ≥50 pasos por e-folding del factor de
    /// escala.
    CosmoAcceleration {
        eta: f64,
        eps: f64,
        kappa_h: f64,
        dt_min: f64,
        dt_max: f64,
    },
}

impl AdaptiveDtCriterion {
    /// Construcción ergonómica del criterio de aceleración puro.
    pub fn acceleration(eta: f64, eps: f64, dt_min: f64, dt_max: f64) -> Self {
        Self::Acceleration {
            eta,
            eps,
            dt_min,
            dt_max,
        }
    }

    /// Construcción ergonómica del criterio combinado aceleración + cosmología.
    pub fn cosmo_acceleration(eta: f64, eps: f64, kappa_h: f64, dt_min: f64, dt_max: f64) -> Self {
        Self::CosmoAcceleration {
            eta,
            eps,
            kappa_h,
            dt_min,
            dt_max,
        }
    }
}

/// Calcula el `dt` global que se usará en el próximo paso, dadas las
/// aceleraciones actuales y el estado cosmológico.
///
/// * `accelerations` — una por partícula, en las mismas unidades que usa el
///   solver (coordenadas comóviles internas).
/// * `cosmo` — opcional; requerido sólo para el modo [`AdaptiveDtCriterion::CosmoAcceleration`].
/// * `a` — factor de escala actual.
///
/// Devuelve un `dt > 0` y finito. Si las aceleraciones son todas nulas o no
/// finitas, devuelve `dt_max` del criterio (o el `Fixed`).
pub fn compute_global_adaptive_dt(
    criterion: AdaptiveDtCriterion,
    accelerations: &[Vec3],
    cosmo: Option<CosmologyParams>,
    a: f64,
) -> f64 {
    match criterion {
        AdaptiveDtCriterion::Fixed(dt) => dt,
        AdaptiveDtCriterion::Acceleration {
            eta,
            eps,
            dt_min,
            dt_max,
        } => {
            let a_max = max_accel_magnitude(accelerations);
            clamp_finite(accel_dt(eta, eps, a_max, dt_max), dt_min, dt_max)
        }
        AdaptiveDtCriterion::CosmoAcceleration {
            eta,
            eps,
            kappa_h,
            dt_min,
            dt_max,
        } => {
            let a_max = max_accel_magnitude(accelerations);
            let dt_a = accel_dt(eta, eps, a_max, dt_max);
            let dt_h = cosmo
                .map(|c| cosmo_dt_limit(c, a, kappa_h, dt_max))
                .unwrap_or(dt_max);
            clamp_finite(dt_a.min(dt_h), dt_min, dt_max)
        }
    }
}

/// Magnitud máxima de aceleración (L² por componente) en un slice.
/// Devuelve 0 si el slice está vacío o todas las entradas son no finitas.
#[inline]
pub fn max_accel_magnitude(accels: &[Vec3]) -> f64 {
    let mut max2 = 0.0_f64;
    for a in accels {
        let m2 = a.x * a.x + a.y * a.y + a.z * a.z;
        if m2.is_finite() && m2 > max2 {
            max2 = m2;
        }
    }
    max2.sqrt()
}

#[inline]
fn accel_dt(eta: f64, eps: f64, a_max: f64, dt_fallback: f64) -> f64 {
    if a_max > 0.0 && eps > 0.0 && eta > 0.0 {
        eta * (eps / a_max).sqrt()
    } else {
        dt_fallback
    }
}

#[inline]
fn cosmo_dt_limit(cosmo: CosmologyParams, a: f64, kappa_h: f64, dt_fallback: f64) -> f64 {
    let h = gadget_ng_core::cosmology::hubble_param(cosmo, a);
    if h > 0.0 && kappa_h > 0.0 && a > 0.0 {
        // dt (código, unidades de a) ≤ kappa_h · a / H(a)
        kappa_h * a / h
    } else {
        dt_fallback
    }
}

#[inline]
fn clamp_finite(dt: f64, dt_min: f64, dt_max: f64) -> f64 {
    if !dt.is_finite() || dt <= 0.0 {
        return dt_max.max(dt_min);
    }
    dt.clamp(dt_min, dt_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_returns_constant() {
        let c = AdaptiveDtCriterion::Fixed(1.23e-4);
        let dt = compute_global_adaptive_dt(c, &[Vec3::zero()], None, 0.02);
        assert_eq!(dt, 1.23e-4);
    }

    #[test]
    fn acceleration_scales_as_sqrt_eps_over_amax() {
        let eta = 0.05;
        let eps = 1e-4;
        let dt_min = 1e-8;
        let dt_max = 1.0;
        let crit = AdaptiveDtCriterion::acceleration(eta, eps, dt_min, dt_max);

        let accels = [Vec3::new(4.0, 0.0, 0.0)]; // |a| = 4
        let dt = compute_global_adaptive_dt(crit, &accels, None, 1.0);
        let expected = eta * (eps / 4.0_f64).sqrt();
        assert!(
            (dt - expected).abs() < 1e-14,
            "dt={dt}, expected={expected}"
        );
    }

    #[test]
    fn clamp_to_dt_max_when_accelerations_vanish() {
        let crit = AdaptiveDtCriterion::acceleration(0.05, 1e-4, 1e-8, 0.01);
        let accels = [Vec3::zero(); 4];
        let dt = compute_global_adaptive_dt(crit, &accels, None, 1.0);
        assert_eq!(dt, 0.01);
    }

    #[test]
    fn clamp_to_dt_min_when_accelerations_huge() {
        let crit = AdaptiveDtCriterion::acceleration(0.05, 1e-4, 1e-6, 1e-2);
        let accels = [Vec3::new(1e20, 0.0, 0.0)];
        let dt = compute_global_adaptive_dt(crit, &accels, None, 1.0);
        assert_eq!(dt, 1e-6);
    }

    #[test]
    fn cosmo_limit_kicks_in_when_accel_small() {
        // En el límite acel=0, el criterio de aceleración devolvería dt_max.
        // El límite cosmológico debe imponerse si κ·a/H(a) < dt_max.
        let cosmo = CosmologyParams::new(1.0, 0.0, 0.1); // EdS, H0 = 0.1 (interna)
        let a = 0.02;
        // H(a) = H0·a^(-3/2) en EdS.
        let h = gadget_ng_core::cosmology::hubble_param(cosmo, a);
        let kappa_h = 0.02;
        let crit = AdaptiveDtCriterion::cosmo_acceleration(0.05, 1e-4, kappa_h, 1e-10, 1.0);
        let dt = compute_global_adaptive_dt(crit, &[Vec3::zero()], Some(cosmo), a);
        let expected = kappa_h * a / h;
        assert!(
            (dt - expected).abs() / expected < 1e-10,
            "dt={dt}, expected={expected}"
        );
    }

    #[test]
    fn cosmo_combined_picks_the_more_restrictive() {
        let cosmo = CosmologyParams::new(1.0, 0.0, 0.1);
        let crit = AdaptiveDtCriterion::cosmo_acceleration(0.05, 1e-4, 0.02, 1e-10, 1.0);
        // Aceleración grande → el criterio accel domina.
        let accels = [Vec3::new(10.0, 0.0, 0.0)];
        let dt_combined = compute_global_adaptive_dt(crit, &accels, Some(cosmo), 0.02);
        let dt_accel_only = 0.05 * (1e-4_f64 / 10.0).sqrt();
        assert!(dt_combined <= dt_accel_only + 1e-14);
    }
}
