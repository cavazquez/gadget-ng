//! Conducción térmica y difusión CR anisótropa a lo largo de líneas de campo B (Phase 133).
//!
//! ## Modelo
//!
//! En un plasma magnetizado, el transporte de calor y de rayos cósmicos es altamente
//! anisótropo: la difusión es eficiente paralela a B y suprimida en la dirección
//! perpendicular por la giración de las partículas en torno a las líneas de campo.
//!
//! ### Tensor de difusión
//!
//! ```text
//! D = κ_∥ (B̂ ⊗ B̂) + κ_⊥ (I − B̂ ⊗ B̂)
//! ```
//!
//! El flujo de calor entre partículas `i` y `j`:
//!
//! ```text
//! q_ij = (κ_∥ − κ_⊥) (B̂_i · r̂_ij)² + κ_⊥) × (T_j − T_i) × W(r_ij, h_i) × dt
//! ```
//!
//! donde `r̂_ij = (r_j − r_i) / |r_j − r_i|` es la dirección entre partículas.
//!
//! Para `κ_⊥ = 0` y `κ_∥ >> 0`: conducción puramente paralela a B.
//! Para `κ_∥ = κ_⊥ = κ`: recupera conducción isótropa de Spitzer.
//!
//! ## Referencias
//!
//! Braginskii (1965), Rev. Plasma Phys. 1, 205 — tensor de transporte anisótropo.
//! Parrish & Stone (2005), ApJ 633, 334 — conducción anisótropa en SPH.
//! Sharma & Hammett (2007), J. Comput. Phys. 227, 123 — implementación discreta.

use crate::MU0;
use gadget_ng_core::{Particle, ParticleType, Vec3};

/// Kernel SPH simple para difusión (Wendland C2).
#[inline]
fn kernel_w(r: f64, h: f64) -> f64 {
    if h <= 0.0 {
        return 0.0;
    }
    let q = r / h;
    if q > 2.0 {
        return 0.0;
    }
    let t = 1.0 - 0.5 * q;
    (21.0 / (2.0 * std::f64::consts::PI * h * h * h)) * t.powi(4) * (1.0 + 2.0 * q)
}

/// Convierte energía interna `u` a temperatura con `γ = gamma`.
#[inline]
fn u_to_t(u: f64, gamma: f64) -> f64 {
    const MU_MEAN: f64 = 0.588; // peso molecular medio (H+He totalmente ionizado)
    const KB_OVER_MU: f64 = 8.314e7; // erg/g/K (R en CGS)
    u.max(0.0) * (gamma - 1.0) / (KB_OVER_MU / MU_MEAN)
}

/// Conducción térmica anisótropa ∥B: `D = κ_∥ (B̂⊗B̂) + κ_⊥ (I − B̂⊗B̂)` (Phase 133).
///
/// El flujo de calor entre dos partículas depende del coseno del ángulo entre `r_ij` y `B_i`.
/// La conductividad efectiva en la dirección r̂_ij es:
///
/// ```text
/// κ_eff(θ) = κ_⊥ + (κ_∥ − κ_⊥) cos²(θ)
/// ```
///
/// donde `θ` es el ángulo entre r̂_ij y B̂_i.
///
/// # Parámetros
///
/// - `particles`: slice mutable de partículas
/// - `kappa_par`: conductividad paralela a B
/// - `kappa_perp`: conductividad perpendicular a B (normalmente << kappa_par)
/// - `gamma`: índice adiabático
/// - `dt`: paso de tiempo
pub fn apply_anisotropic_conduction(
    particles: &mut [Particle],
    kappa_par: f64,
    kappa_perp: f64,
    gamma: f64,
    dt: f64,
) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    let mut delta_u = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let t_i = u_to_t(particles[i].internal_energy, gamma);
        let b_i = particles[i].b_field;
        let b_mag_i = (b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z)
            .sqrt()
            .max(1e-30);
        let bhat_i = Vec3::new(b_i.x / b_mag_i, b_i.y / b_mag_i, b_i.z / b_mag_i);

        for j in (i + 1)..n {
            if particles[j].ptype != ParticleType::Gas {
                continue;
            }

            let dx = particles[j].position.x - pos_i.x;
            let dy = particles[j].position.y - pos_i.y;
            let dz = particles[j].position.z - pos_i.z;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-14 {
                continue;
            }

            let h_avg = 0.5 * (h_i + particles[j].smoothing_length.max(1e-10));
            let w = kernel_w(r, 2.0 * h_avg);
            if w <= 0.0 {
                continue;
            }

            // cos²(θ) = (B̂_i · r̂_ij)²
            let rhat_x = dx / r;
            let rhat_y = dy / r;
            let rhat_z = dz / r;
            let cos_theta = bhat_i.x * rhat_x + bhat_i.y * rhat_y + bhat_i.z * rhat_z;
            let cos2 = cos_theta * cos_theta;

            let kappa_eff = kappa_perp + (kappa_par - kappa_perp) * cos2;
            let t_j = u_to_t(particles[j].internal_energy, gamma);
            let flux = kappa_eff * (t_j - t_i) * w * dt;

            delta_u[i] += flux;
            delta_u[j] -= flux; // conservación de energía
        }
    }

    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].internal_energy = (particles[i].internal_energy + delta_u[i]).max(0.0);
        }
    }
}

/// Difusión CR anisótropa a lo largo de B (Phase 133).
///
/// El flujo CR en la dirección r̂_ij tiene un factor geométrico cos²(θ_B):
/// ```text
/// ΔE_cr,i = κ_CR × cos²(θ_B) × (e_cr,j − e_cr,i) × W(r_ij) × dt
/// ```
///
/// Con `B = 0` degenera en difusión isótropa.
#[allow(clippy::needless_range_loop)]
pub fn diffuse_cr_anisotropic(particles: &mut [Particle], kappa_cr: f64, b_suppress: f64, dt: f64) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    let mut delta_cr = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let e_i = particles[i].cr_energy;
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        let b_mag_i = b2_i.sqrt().max(1e-30);
        let bhat_i = Vec3::new(b_i.x / b_mag_i, b_i.y / b_mag_i, b_i.z / b_mag_i);

        // Supresión por |B|² (Phase 129)
        let kappa_eff_base = kappa_cr / (1.0 + b_suppress * b2_i);

        for j in 0..n {
            if i == j {
                continue;
            }
            if particles[j].ptype != ParticleType::Gas {
                continue;
            }

            let dx = particles[j].position.x - pos_i.x;
            let dy = particles[j].position.y - pos_i.y;
            let dz = particles[j].position.z - pos_i.z;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-14 {
                continue;
            }

            let w = kernel_w(r, 2.0 * h_i);
            if w <= 0.0 {
                continue;
            }

            // Dirección de difusión: cos²(θ_B) = (B̂ · r̂)²
            let rhat_x = dx / r;
            let rhat_y = dy / r;
            let rhat_z = dz / r;
            let cos_theta = bhat_i.x * rhat_x + bhat_i.y * rhat_y + bhat_i.z * rhat_z;
            let cos2 = cos_theta * cos_theta;

            // Con B=0: cos2 promedia a 1/3 (isótropo); con B fuerte: solo ∥B
            let kappa_aniso = kappa_eff_base * cos2;
            delta_cr[i] += kappa_aniso * (particles[j].cr_energy - e_i) * w * dt;
        }
    }

    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].cr_energy = (particles[i].cr_energy + delta_cr[i]).max(0.0);
        }
    }
}

/// Calcula el factor β-plasma: `β = 2μ₀ P_th / |B|²`.
///
/// Un β grande (>1) indica que la presión térmica domina sobre la presión magnética.
/// Un β pequeño (<1) indica que el campo magnético domina.
pub fn beta_plasma(p_thermal: f64, b: Vec3) -> f64 {
    let b2 = b.x * b.x + b.y * b.y + b.z * b.z;
    if b2 < 1e-60 {
        return f64::INFINITY;
    }
    2.0 * MU0 * p_thermal / b2
}
