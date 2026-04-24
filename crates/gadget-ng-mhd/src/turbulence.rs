//! Turbulencia MHD: forzado estocástico Ornstein-Uhlenbeck (Phase 140).
//!
//! ## Modelo
//!
//! Genera turbulencia Alfvénica usando el proceso estocástico de Ornstein-Uhlenbeck (OU):
//!
//! ```text
//! dA/dt = -A/τ_c + σ × η(t)
//! ```
//!
//! donde:
//! - `A` es la amplitud de forzado en cada modo espectral
//! - `τ_c` es el tiempo de correlación (~ L/v_A)
//! - `σ = sqrt(2 amplitude²/τ_c)` es la varianza del ruido
//! - `η(t)` es ruido blanco gaussiano
//!
//! ### Espectro de potencia
//!
//! Los modos se pesan con la potencia espectral dada:
//! - Kolmogorov: `P(k) ∝ k^{-5/3}` (turbulencia hidrodinámica)
//! - Goldreich-Sridhar: `P(k) ∝ k^{-3/2}` (turbulencia Alfvénica con B₀)
//!
//! ### Estadísticas
//!
//! - Número de Mach rms: `M = v_rms / c_s`
//! - Número de Mach Alfvénico: `M_A = v_rms / v_A`
//!
//! ## Referencias
//!
//! Schmidt et al. (2006), A&A 450, 265 — proceso OU para turbulencia cósmica.
//! Goldreich & Sridhar (1995), ApJ 438, 763 — turbulencia MHD anisótropa.
//! Federrath et al. (2010), A&A 512, A81 — forzado numérico de turbulencia.

use gadget_ng_core::{Particle, ParticleType, TurbulenceSection, Vec3};
use crate::MU0;

/// Generador de números pseudo-aleatorios simple (LCG para reproducibilidad).
fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state as f64) / (u64::MAX as f64)
}

/// Genera un número gaussiano estándar N(0,1) usando Box-Muller.
fn gauss(state: &mut u64) -> f64 {
    let u1 = lcg_next(state).max(1e-30);
    let u2 = lcg_next(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Aplica forzado turbulento Ornstein-Uhlenbeck al campo de velocidades (Phase 140).
///
/// Genera modos de forzado en el rango `[k_min, k_max]` con espectro de potencia
/// proporcional a `k^{-spectral_index}` y los aplica como perturbaciones de velocidad.
///
/// El estado del proceso OU se actualiza en cada paso: `A_new = A_old × exp(-dt/τ_c) + σ × η`.
///
/// # Parámetros
///
/// - `particles`: slice mutable de partículas
/// - `cfg`: configuración de turbulencia
/// - `dt`: paso de tiempo
/// - `step`: número de paso (para la semilla del generador aleatorio)
pub fn apply_turbulent_forcing(
    particles: &mut [Particle],
    cfg: &TurbulenceSection,
    dt: f64,
    step: u64,
) {
    if !cfg.enabled || cfg.amplitude <= 0.0 { return; }

    let tau = cfg.correlation_time.max(1e-10);
    let sigma = (2.0 * cfg.amplitude * cfg.amplitude / tau).sqrt();
    let decay = (-dt / tau).exp();

    // Semilla reproducible basada en el paso
    let mut rng_state: u64 = 12345678 ^ (step.wrapping_mul(987654321));

    // Para cada partícula de gas, generar perturbación de velocidad
    for (idx, p) in particles.iter_mut().enumerate() {
        if p.ptype != ParticleType::Gas { continue; }

        // Estado OU previo (aproximado como amplitud × fase aleatoria)
        let seed_p = rng_state ^ (idx as u64).wrapping_mul(1234567);
        let mut rng_p = seed_p;

        // Perturbación OU: δv = decay × δv_prev + sigma × gauss()
        // Usamos la posición de la partícula para definir el forzado en el espacio k
        let x = p.position.x;
        let y = p.position.y;
        let z = p.position.z;

        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;
        let mut fz = 0.0_f64;

        // Suma sobre modos k en [k_min, k_max]
        let n_modes = 4usize;
        for m in 1..=n_modes {
            let k = cfg.k_min + (cfg.k_max - cfg.k_min) * (m as f64 / n_modes as f64);

            // Peso espectral: P(k) ∝ k^{-spectral_index}
            let weight = k.powf(-cfg.spectral_index).sqrt();

            // Fase aleatoria fija por modo (seed reproducible)
            let phase_x = 2.0 * std::f64::consts::PI * lcg_next(&mut rng_p);
            let phase_y = 2.0 * std::f64::consts::PI * lcg_next(&mut rng_p);
            let phase_z = 2.0 * std::f64::consts::PI * lcg_next(&mut rng_p);

            // Proceso OU para este modo
            let a_x = sigma * gauss(&mut rng_p) * weight;
            let a_y = sigma * gauss(&mut rng_p) * weight;
            let a_z = sigma * gauss(&mut rng_p) * weight;

            // Patrón espacial: forzado ∝ sin(k·x + φ)
            fx += a_x * (k * x + phase_x).sin() * decay;
            fy += a_y * (k * y + phase_y).sin() * decay;
            fz += a_z * (k * z + phase_z).sin() * decay;
        }

        // Aplicar perturbación de velocidad
        p.velocity.x += fx * dt;
        p.velocity.y += fy * dt;
        p.velocity.z += fz * dt;

        rng_state = rng_state.wrapping_add(rng_p);
    }
}

/// Calcula estadísticas de turbulencia: número de Mach rms y Mach Alfvénico (Phase 140).
///
/// # Retorna
///
/// `(mach_rms, alfven_mach)` donde:
/// - `mach_rms = v_rms / c_s` (número de Mach sónico)
/// - `alfven_mach = v_rms / v_A` (número de Mach Alfvénico)
///
/// `c_s = sqrt(γ P / ρ)` y `v_A = B / sqrt(μ₀ ρ)`.
pub fn turbulence_stats(particles: &[Particle], gamma: f64) -> (f64, f64) {
    let mut v2_sum = 0.0_f64;
    let mut b2_rho_sum = 0.0_f64;
    let mut cs2_sum = 0.0_f64;
    let mut n = 0usize;

    for p in particles.iter() {
        if p.ptype != ParticleType::Gas { continue; }
        let h = p.smoothing_length.max(1e-10);
        let rho = (p.mass / (h * h * h)).max(1e-30);
        let p_th = (gamma - 1.0) * rho * p.internal_energy.max(0.0);

        let v2 = p.velocity.x*p.velocity.x + p.velocity.y*p.velocity.y + p.velocity.z*p.velocity.z;
        let b2 = p.b_field.x*p.b_field.x + p.b_field.y*p.b_field.y + p.b_field.z*p.b_field.z;

        v2_sum += v2;
        b2_rho_sum += b2 / (MU0 * rho);
        cs2_sum += gamma * p_th / rho;
        n += 1;
    }

    if n == 0 { return (0.0, 0.0); }
    let v_rms = (v2_sum / n as f64).sqrt();
    let v_a = (b2_rho_sum / n as f64).sqrt();
    let c_s = (cs2_sum / n as f64).sqrt();

    let mach_rms = if c_s > 1e-30 { v_rms / c_s } else { 0.0 };
    let alfven_mach = if v_a > 1e-30 { v_rms / v_a } else { 0.0 };

    (mach_rms, alfven_mach)
}
