//! Rayos cósmicos básicos: inyección desde SN y difusión SPH (Phase 117).
//!
//! ## Modelo
//!
//! Los rayos cósmicos (CRs) son partículas relativistas (principalmente protones)
//! aceleradas en frentes de choque de supernovas. El módulo implementa:
//!
//! 1. **Inyección**: una fracción `cr_fraction` de la energía de SN II se deposita
//!    como energía CR en las partículas de gas con SFR activa.
//!
//! 2. **Difusión**: difusión isótropa simplificada usando SPH kernel:
//!    `Δe_cr,i = κ_CR × Σ_j (e_cr,j - e_cr,i) × w(r_ij, h_i) × dt`
//!
//! 3. **Presión de CRs**: contribución a la presión total con γ_CR = 4/3.
//!
//! ## Referencia
//!
//! Jubelgas et al. (2008) A&A 481, 33 — CRs en SPH cosmológico.
//! Pfrommer et al. (2017) MNRAS 465, 4500 — transporte de CRs.

use gadget_ng_core::{Particle, ParticleType};

/// Constante de energía de SN en unidades internas [(km/s)² por 10¹⁰ M_sun].
const E_SN_CODE: f64 = 1.54e-3;

/// Kernel SPH simple (Wendland C2) para difusión.
#[inline]
fn kernel_w_cr(r: f64, h: f64) -> f64 {
    if h <= 0.0 { return 0.0; }
    let q = r / h;
    if q > 2.0 { return 0.0; }
    let t = 1.0 - 0.5 * q;
    (21.0 / (2.0 * std::f64::consts::PI)) / (h * h * h) * t.powi(4) * (1.0 + 2.0 * q)
}

/// Presión de rayos cósmicos para γ_CR = 4/3 (relativista) (Phase 117).
///
/// `P_cr = (γ_cr - 1) × ρ × e_cr`
#[inline]
pub fn cr_pressure(cr_energy: f64, rho: f64) -> f64 {
    const GAMMA_CR: f64 = 4.0 / 3.0;
    (GAMMA_CR - 1.0) * rho * cr_energy.max(0.0)
}

/// Inyecta energía CR desde eventos de SN II (Phase 117).
///
/// Para cada partícula de gas con `sfr[i] > 0`:
/// `Δe_cr,i = cr_fraction × E_SN × sfr[i] × dt`
///
/// La energía CR se suma directamente al campo `cr_energy` de la partícula.
///
/// # Parámetros
///
/// - `particles`: slice mutable de partículas
/// - `sfr`: tasa de formación estelar por partícula
/// - `cr_fraction`: fracción de E_SN → CRs
/// - `dt`: paso de tiempo
pub fn inject_cr_from_sn(
    particles: &mut [Particle],
    sfr: &[f64],
    cr_fraction: f64,
    dt: f64,
) {
    assert_eq!(particles.len(), sfr.len());
    for i in 0..particles.len() {
        if particles[i].ptype != ParticleType::Gas { continue; }
        if sfr[i] <= 0.0 { continue; }
        let delta_cr = cr_fraction * E_SN_CODE * sfr[i] * dt;
        particles[i].cr_energy += delta_cr;
    }
}

/// Difusión isótropa de CRs con supresión magnética opcional (Phase 117 + 129).
///
/// Implementa: `Δe_cr,i = κ_CR_eff × Σ_j (e_cr,j - e_cr,i) × w(r_ij, h_i) × dt`
///
/// donde `κ_CR_eff = κ_CR / (1 + b_suppress × |B_i|²)` (Phase 129).
/// Con `b_suppress = 0.0` se recupera el comportamiento clásico isótropo.
///
/// # Parámetros
///
/// - `particles`: slice mutable de partículas
/// - `kappa_cr`: coeficiente de difusión [unidades internas]
/// - `b_suppress`: factor de supresión por campo B (Phase 129); `0.0` = sin supresión
/// - `dt`: paso de tiempo
#[allow(clippy::needless_range_loop)]
pub fn diffuse_cr(particles: &mut [Particle], kappa_cr: f64, b_suppress: f64, dt: f64) {
    let n = particles.len();
    if n == 0 { return; }

    // Calculamos el flujo neto para evitar aliasing
    let mut delta_cr = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas { continue; }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let e_i = particles[i].cr_energy;

        // Phase 129: κ efectiva modulada por |B|²
        let b2_i = particles[i].b_field.x.powi(2)
            + particles[i].b_field.y.powi(2)
            + particles[i].b_field.z.powi(2);
        let kappa_eff = kappa_cr / (1.0 + b_suppress * b2_i);

        for j in 0..n {
            if i == j { continue; }
            if particles[j].ptype != ParticleType::Gas { continue; }

            let dx = particles[j].position.x - pos_i.x;
            let dy = particles[j].position.y - pos_i.y;
            let dz = particles[j].position.z - pos_i.z;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();

            let w = kernel_w_cr(r, 2.0 * h_i);
            if w > 0.0 {
                delta_cr[i] += kappa_eff * (particles[j].cr_energy - e_i) * w * dt;
            }
        }
    }

    // Aplicar incrementos, asegurar no-negatividad
    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].cr_energy = (particles[i].cr_energy + delta_cr[i]).max(0.0);
        }
    }
}
