//! Integrador leapfrog KDK para partículas SPH.
//!
//! El paso de integración para gas combina:
//! - Gravedad externa `particle.acceleration` (calculada por el solver gravitatorio).
//! - Aceleración hidrodinámica SPH `gas.acc_sph`.
//! - Energía interna integrada por Euler explícito: `u += du_dt · dt`.

use crate::density::compute_density;
use crate::forces::{compute_sph_forces, compute_sph_forces_gadget2};
use crate::kernel::{grad_w, w};
use crate::particle::SphParticle;
use crate::viscosity::compute_balsara_factors;
use gadget_ng_core::{Particle, ParticleType, Vec3};

// ─── SPH cosmológico sobre gadget_ng_core::Particle ──────────────────────────

/// Aceleración + tasa de energía interna SPH para partículas `Particle` con `ptype == Gas`.
///
/// Devuelve `(acc_sph[i], du_dt[i])` para cada partícula gas, usando el índice original.
fn sph_accel_and_dudt(
    particles: &[Particle],
    rho: &[f64],
    pressure: &[f64],
    alpha_visc: f64,
    gamma: f64,
) -> Vec<(Vec3, f64)> {
    let n = particles.len();
    let mut result = vec![(Vec3::zero(), 0.0f64); n];
    const EPS_VISC: f64 = 0.01;

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas || rho[i] < 1e-200 {
            continue;
        }
        let pi = particles[i].position;
        let vi = particles[i].velocity;
        let hi = particles[i].smoothing_length.max(1e-10);
        let pi_rho2 = pressure[i] / (rho[i] * rho[i]);
        let cs_i = (gamma * pressure[i] / rho[i]).sqrt().max(0.0);

        let mut acc = Vec3::zero();
        let mut dudt = 0.0_f64;

        for j in 0..n {
            if j == i || particles[j].ptype != ParticleType::Gas || rho[j] < 1e-200 {
                continue;
            }
            let r_ij = pi - particles[j].position;
            let r = r_ij.norm();
            let hj = particles[j].smoothing_length.max(1e-10);

            let gw_i = grad_w(r, hi);
            let gw_j = grad_w(r, hj);
            if gw_i == 0.0 && gw_j == 0.0 {
                continue;
            }

            let r_hat = if r > 1e-300 { r_ij * (1.0 / r) } else { Vec3::zero() };
            let nabla_w_i = r_hat * (gw_i / hi);
            let nabla_w_j = r_hat * (gw_j / hj);
            let pj_rho2 = pressure[j] / (rho[j] * rho[j]);

            let v_ij = vi - particles[j].velocity;
            let h_bar = 0.5 * (hi + hj);
            let mu_ij = if v_ij.dot(r_ij) < 0.0 {
                let rho_bar = 0.5 * (rho[i] + rho[j]);
                let cs_j = (gamma * pressure[j] / rho[j]).sqrt().max(0.0);
                let cs_bar = 0.5 * (cs_i + cs_j);
                let mu = h_bar * v_ij.dot(r_ij) / (r * r + EPS_VISC * h_bar * h_bar);
                -alpha_visc * cs_bar * mu / rho_bar
            } else {
                0.0
            };

            let coeff_i = pi_rho2 + 0.5 * mu_ij;
            let coeff_j = pj_rho2 + 0.5 * mu_ij;
            acc -= (nabla_w_i * coeff_i + nabla_w_j * coeff_j) * particles[j].mass;
            dudt += particles[j].mass * pi_rho2 * v_ij.dot(nabla_w_i);
        }
        result[i] = (acc, dudt);
    }
    result
}

/// Calcula densidad y presión SPH usando los campos de `Particle`.
///
/// Devuelve `(rho[i], pressure[i])` para cada partícula.
fn compute_rho_pressure(particles: &mut [Particle], n_neigh: f64, gamma: f64) -> (Vec<f64>, Vec<f64>) {
    let n = particles.len();
    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    const MAX_ITER: usize = 30;
    let mut rho_out = vec![0.0_f64; n];
    let mut pressure_out = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let pi = pos[i];
        let m_i = mass[i];
        let mut h = particles[i].smoothing_length.max(1e-10);

        for _ in 0..MAX_ITER {
            let mut rho = 0.0_f64;
            let mut drho = 0.0_f64;
            for j in 0..n {
                if particles[j].ptype != ParticleType::Gas { continue; }
                let r = (pos[j] - pi).norm();
                rho += mass[j] * w(r, h);
                let gw = grad_w(r, h);
                drho += mass[j] * (-1.0 / h) * (3.0 * w(r, h) + r * gw);
            }
            let n_eff = (4.0 * std::f64::consts::PI / 3.0) * (2.0 * h).powi(3) * rho / m_i;
            if (n_eff - n_neigh).abs() < 1e-2 {
                break;
            }
            let dn_dh = (4.0 * std::f64::consts::PI / 3.0)
                * (24.0 * h * h * rho + (2.0 * h).powi(3) * drho)
                / m_i;
            let dh = -(n_eff - n_neigh) / (dn_dh + 1e-100);
            h = (h + dh.clamp(-0.5 * h, 0.5 * h)).max(1e-10);
        }
        let rho_final: f64 = (0..n)
            .filter(|&j| particles[j].ptype == ParticleType::Gas)
            .map(|j| mass[j] * w((pos[j] - pi).norm(), h))
            .sum();
        particles[i].smoothing_length = h;
        rho_out[i] = rho_final;
        pressure_out[i] = (gamma - 1.0) * rho_final * particles[i].internal_energy;
    }
    (rho_out, pressure_out)
}

/// Un paso leapfrog KDK cosmológico (Kick-Drift-Kick) para un sistema SPH+DM.
///
/// Integra posición, velocidad y energía interna de partículas `gadget_ng_core::Particle`
/// usando factores cosmológicos. Las partículas DM siguen solo el paso gravitacional;
/// las partículas Gas reciben también las fuerzas SPH.
///
/// # Parámetros
/// - `particles`: lista mezclada (DM + Gas). Modificada in-place.
/// - `cf`: factores cosmológicos `CosmoFactors` del paso actual.
/// - `gamma`: índice adiabático (p. ej. 5/3).
/// - `alpha_visc`: parámetro de viscosidad artificial (p. ej. 1.0).
/// - `n_neigh`: número objetivo de vecinos SPH (p. ej. 32.0).
/// - `gravity_accel`: función que escribe `particle.acceleration` para todas las partículas.
pub fn sph_cosmo_kdk_step<F>(
    particles: &mut [Particle],
    cf: gadget_ng_integrators::CosmoFactors,
    gamma: f64,
    alpha_visc: f64,
    n_neigh: f64,
    mut gravity_accel: F,
) where
    F: FnMut(&mut [Particle]),
{
    // ── Densidad y presión al inicio del paso ─────────────────────────────────
    let (rho, pressure) = compute_rho_pressure(particles, n_neigh, gamma);

    // ── Fuerzas SPH al inicio ─────────────────────────────────────────────────
    let sph_forces = sph_accel_and_dudt(particles, &rho, &pressure, alpha_visc, gamma);

    // ── Kick 1 (½ kick) ──────────────────────────────────────────────────────
    for (i, p) in particles.iter_mut().enumerate() {
        let grav_acc = p.acceleration;
        p.velocity += grav_acc * cf.kick_half;
        if p.ptype == ParticleType::Gas {
            let (acc_sph, du_dt) = sph_forces[i];
            p.velocity += acc_sph * cf.kick_half;
            p.internal_energy = (p.internal_energy + du_dt * cf.kick_half).max(0.0);
        }
    }

    // ── Drift ─────────────────────────────────────────────────────────────────
    for p in particles.iter_mut() {
        p.position += p.velocity * cf.drift;
    }

    // ── Fuerzas al nuevo tiempo ───────────────────────────────────────────────
    gravity_accel(particles);
    let (rho2, pressure2) = compute_rho_pressure(particles, n_neigh, gamma);
    let sph_forces2 = sph_accel_and_dudt(particles, &rho2, &pressure2, alpha_visc, gamma);

    // ── Kick 2 (½ kick) ──────────────────────────────────────────────────────
    for (i, p) in particles.iter_mut().enumerate() {
        let grav_acc = p.acceleration;
        p.velocity += grav_acc * cf.kick_half2;
        if p.ptype == ParticleType::Gas {
            let (acc_sph2, du_dt2) = sph_forces2[i];
            p.velocity += acc_sph2 * cf.kick_half2;
            p.internal_energy = (p.internal_energy + du_dt2 * cf.kick_half2).max(0.0);
        }
    }
}

/// Un paso leapfrog KDK completo (Kick-Drift-Kick) para un sistema SPH.
///
/// # Parámetros
/// - `particles`: lista de partículas (DM + gas mezcladas).
/// - `dt`: paso de tiempo.
/// - `gravity_accel`: función que calcula la aceleración gravitatoria y la escribe
///   en `particle.acceleration` para todas las partículas (DM y gas).
pub fn sph_kdk_step<F>(particles: &mut [SphParticle], dt: f64, gravity_accel: F)
where
    F: Fn(&mut [SphParticle]),
{
    let dt2 = 0.5 * dt;

    // ── Kick 1 (½ dt) ────────────────────────────────────────────────────────
    for p in particles.iter_mut() {
        let total_acc = total_acceleration(p);
        p.velocity += total_acc * dt2;
        if let Some(gas) = p.gas.as_mut() {
            gas.u = (gas.u + gas.du_dt * dt2).max(0.0);
        }
    }

    // ── Drift (dt) ───────────────────────────────────────────────────────────
    for p in particles.iter_mut() {
        p.position += p.velocity * dt;
    }

    // ── Fuerzas al nuevo tiempo ───────────────────────────────────────────────
    gravity_accel(particles);
    compute_density(particles);
    compute_sph_forces(particles);

    // ── Kick 2 (½ dt) ────────────────────────────────────────────────────────
    for p in particles.iter_mut() {
        let total_acc = total_acceleration(p);
        p.velocity += total_acc * dt2;
        if let Some(gas) = p.gas.as_mut() {
            gas.u = (gas.u + gas.du_dt * dt2).max(0.0);
        }
    }
}

/// Aceleración total: gravedad + SPH (sólo para gas).
#[inline]
fn total_acceleration(p: &SphParticle) -> Vec3 {
    let mut a = p.acceleration;
    if let Some(gas) = p.gas.as_ref() {
        a += gas.acc_sph;
    }
    a
}

// ─── Integrador KDK Gadget-2 (entropía + Balsara) ────────────────────────────

/// Un paso leapfrog KDK completo para SPH con formulación de **entropía** Gadget-2.
///
/// A diferencia de `sph_kdk_step` que evoluciona la energía interna `u`,
/// este integrador evoluciona la **función entrópica** `A = P/ρ^γ`:
///
/// ```text
/// dA_i/dt = (γ-1) / (2 ρ_i^(γ-1))  Σ_j m_j Π_ij v_ij · ∇W̄_ij
/// ```
///
/// En regiones adiabáticas (sin viscosidad), `dA/dt = 0` exactamente → la
/// entropía se conserva a nivel de máquina, eliminando la producción numérica
/// de entropía del integrador energético clásico.
///
/// ## Flujo por paso
///
/// 1. Cálculo de fuerzas (densidad, Balsara, `compute_sph_forces_gadget2`).
/// 2. Kick ½·dt: `v += (a_grav + a_sph)·dt/2`, `A += da_dt·dt/2`.
/// 3. Drift dt: `r += v·dt`.
/// 4. Gravedad + fuerzas SPH al nuevo tiempo.
/// 5. Kick ½·dt final + sincronización P, u desde A.
///
/// # Parámetros
///
/// - `particles`: lista de partículas (DM + gas mezcladas).
/// - `dt`: paso de tiempo.
/// - `gravity_accel`: función que calcula la aceleración gravitatoria in-place.
pub fn sph_kdk_step_gadget2<F>(particles: &mut [SphParticle], dt: f64, gravity_accel: F)
where
    F: Fn(&mut [SphParticle]),
{
    use crate::density::GAMMA;

    let dt2 = 0.5 * dt;

    // ── Fuerzas al tiempo actual ──────────────────────────────────────────────
    compute_density(particles);
    compute_balsara_factors(particles);
    compute_sph_forces_gadget2(particles);

    // ── Kick 1 (½ dt) ────────────────────────────────────────────────────────
    for p in particles.iter_mut() {
        let total_acc = total_acceleration(p);
        p.velocity += total_acc * dt2;
        if let Some(gas) = p.gas.as_mut() {
            gas.entropy = (gas.entropy + gas.da_dt * dt2).max(0.0);
        }
    }

    // ── Drift (dt) ───────────────────────────────────────────────────────────
    for p in particles.iter_mut() {
        p.position += p.velocity * dt;
    }

    // ── Fuerzas al nuevo tiempo ───────────────────────────────────────────────
    gravity_accel(particles);
    compute_density(particles);
    // Después de re-calcular ρ, sincronizar P y u desde A actualizada
    for p in particles.iter_mut() {
        if let Some(gas) = p.gas.as_mut() {
            gas.sync_from_entropy(GAMMA);
        }
    }
    compute_balsara_factors(particles);
    compute_sph_forces_gadget2(particles);

    // ── Kick 2 (½ dt) ────────────────────────────────────────────────────────
    for p in particles.iter_mut() {
        let total_acc = total_acceleration(p);
        p.velocity += total_acc * dt2;
        if let Some(gas) = p.gas.as_mut() {
            gas.entropy = (gas.entropy + gas.da_dt * dt2).max(0.0);
            // Sincronizar P y u finales desde entropía
            gas.sync_from_entropy(GAMMA);
        }
    }
}

/// Calcula el paso de tiempo de Courant hidrodinámica mínimo entre todas las partículas.
///
/// ```text
/// dt_i = C_courant · h_i / max(max_vsig_i, c_s_i)
/// ```
///
/// Usa `max_vsig` cuando está disponible (calculado por `compute_sph_forces_gadget2`).
/// En condiciones de reposo (max_vsig = 0) cae back a la velocidad del sonido local,
/// garantizando siempre un dt finito.
///
/// # Parámetros
/// - `c_courant`: número de Courant (típico: 0.3).
pub fn courant_dt(particles: &[SphParticle], c_courant: f64) -> f64 {
    use crate::density::GAMMA;
    particles
        .iter()
        .filter_map(|p| p.gas.as_ref())
        .filter(|g| g.h_sml > 0.0 && g.rho > 0.0)
        .map(|g| {
            let cs = (GAMMA * g.pressure / g.rho).sqrt().max(0.0);
            let vsig = g.max_vsig.max(cs).max(1e-300);
            c_courant * g.h_sml / vsig
        })
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::compute_density;
    use crate::forces::compute_sph_forces;
    use gadget_ng_core::Vec3;

    fn no_gravity(_: &mut [SphParticle]) {}

    /// Un blob de gas en reposo sin gravedad externa no debe explotar
    /// en pocas iteraciones (energía interna acotada).
    #[test]
    fn gas_blob_no_gravity_bounded_energy() {
        let n_side = 3usize;
        let box_size = 3.0_f64;
        let dx = box_size / n_side as f64;
        let mut parts: Vec<SphParticle> = (0..n_side.pow(3))
            .map(|k| {
                let iz = k / (n_side * n_side);
                let iy = (k / n_side) % n_side;
                let ix = k % n_side;
                let pos = Vec3::new(
                    (ix as f64 + 0.5) * dx,
                    (iy as f64 + 0.5) * dx,
                    (iz as f64 + 0.5) * dx,
                );
                SphParticle::new_gas(k, 1.0, pos, Vec3::zero(), 1.0, 2.0 * dx)
            })
            .collect();

        compute_density(&mut parts);
        compute_sph_forces(&mut parts);

        let dt = 1e-3_f64;
        for _ in 0..10 {
            sph_kdk_step(&mut parts, dt, no_gravity);
        }

        let total_u: f64 = parts
            .iter()
            .filter_map(|p| p.gas.as_ref())
            .map(|g| g.u)
            .sum();
        assert!(
            total_u > 0.0 && total_u < 1000.0,
            "energía interna total = {total_u:.4}"
        );
    }
}
