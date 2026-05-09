//! Streaming de rayos cósmicos lungo líneas de campo B (Phase 170).
//!
//! ## Modelo
//!
//! Los CRs relativistas streaming lungo las líneas de campo magnético a la velocidad
//! de Alfvén `v_A = B / sqrt(μ₀ ρ)`. El término compressional de la ecuación de energía:
//!
//! ```text
//! de_cr/dt = -(1/3) e_cr * (∇·v)
//! ```
//!
//! Este término captura el trabajo realizado por los CRs durante la compresión/expansión
//! del gas (efecto del virial). En zonas de compresión (∇·v < 0), los CRs ganan energía;
//! en zonas de expansión (∇·v > 0), pierden energía.
//!
//! La pérdida de energía por streaming propiamente dicho (excitación de Alfvén waves)
//! se modela como `η_stream = v_A * e_cr / L_min` donde `L_min` es la escala de longitud
//! más pequeña del campo de velocidades.
//!
//! ## Implementación numérica
//!
//! 1. Calcular divergencia de velocidad via SPH: `∇·v = Σ_j m_j (v_i - v_j) · ∇W(r_ij, h_i)`
//! 2. Aplicar término compressional: `e_cr ← e_cr * (1 - dt * (1/3) * div_v)`
//! 3. Aplicar pérdidas por streaming (opcional, activo si streaming_coefficient > 0)
//!
//! ## Referencia
//!
//! Uhlig et al. (2012) MNRAS 423, 2374 — CR streaming.
//! Pakmor et al. (2016) MNRAS 455, 1134 — CR backreaction en EAGLE.

use crate::MU0;
use gadget_ng_core::{Particle, ParticleType, Vec3};

const GAMMA_CR: f64 = 4.0 / 3.0;

fn alfven_speed_magnitude(b2: f64, rho: f64) -> f64 {
    if b2 < 1e-60 || rho < 1e-60 {
        return 0.0;
    }
    (b2 / (MU0 * rho)).sqrt()
}

fn div_v_single(
    i: usize,
    particles: &[Particle],
    periodic_box: Option<f64>,
    _rho_i: f64,
    pos_i: Vec3,
    h_i: f64,
) -> f64 {
    let mut div = 0.0_f64;

    for j in 0..particles.len() {
        if j == i || particles[j].ptype != ParticleType::Gas {
            continue;
        }

        let h_j = particles[j].smoothing_length.max(1e-10);

        let mut dr = particles[j].position - pos_i;
        if let Some(l) = periodic_box {
            dr.x -= l * (dr.x / l).round();
            dr.y -= l * (dr.y / l).round();
            dr.z -= l * (dr.z / l).round();
        }

        let r = dr.norm();
        if r < 1e-14 || r > 2.0 * h_i {
            continue;
        }

        let rhat_x = dr.x / r;
        let rhat_y = dr.y / r;
        let rhat_z = dr.z / r;

        let dwdr = grad_w_approx(r, h_i);

        let v_ij_x = particles[i].velocity.x - particles[j].velocity.x;
        let v_ij_y = particles[i].velocity.y - particles[j].velocity.y;
        let v_ij_z = particles[i].velocity.z - particles[j].velocity.z;

        let vol_j = particles[j].mass / (4.0 / 3.0 * std::f64::consts::PI * h_j * h_j * h_j);

        div += particles[j].mass / vol_j.max(1e-30)
            * (v_ij_x * rhat_x + v_ij_y * rhat_y + v_ij_z * rhat_z)
            * dwdr
            / h_i.max(1e-10);
    }

    div
}

#[inline]
fn grad_w_approx(r: f64, h: f64) -> f64 {
    if h <= 0.0 || r > 2.0 * h {
        return 0.0;
    }
    let q = r / h;
    if q > 1.0 {
        let t = 0.5 * (2.0 - q);
        -(21.0 / (2.0 * std::f64::consts::PI)) / (h * h * h) * 4.0 * t.powi(3) * (1.0 + q) * (-0.5)
    } else {
        let t = 1.0 - 0.5 * q;
        -(21.0 / (2.0 * std::f64::consts::PI)) / (h * h * h) * 4.0 * t.powi(3) * (-1.5 - 2.0 * q)
            / h
    }
}

#[inline]
fn periodic_delta(pi: Vec3, pj: Vec3, periodic_box: Option<f64>) -> Vec3 {
    let mut d = pj - pi;
    if let Some(l) = periodic_box
        && l > 0.0
    {
        d.x -= l * (d.x / l).round();
        d.y -= l * (d.y / l).round();
        d.z -= l * (d.z / l).round();
    }
    d
}

fn density_approx(p: &Particle) -> f64 {
    let h = p.smoothing_length.max(1e-10);
    p.mass / ((4.0 / 3.0) * std::f64::consts::PI * h * h * h).max(1e-100)
}

pub fn streaming_crk(
    particles: &mut [Particle],
    dt: f64,
    streaming_coefficient: f64,
    periodic_box: Option<f64>,
) {
    for i in 0..particles.len() {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }

        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;

        let e_cr = particles[i].cr_energy;
        if e_cr <= 0.0 {
            continue;
        }

        let div_v_i = div_v_single(
            i,
            particles,
            periodic_box,
            density_approx(&particles[i]),
            pos_i,
            h_i,
        );

        let compressional_term = -(1.0 / 3.0) * e_cr * div_v_i;

        let b2 = particles[i].b_field.x.powi(2)
            + particles[i].b_field.y.powi(2)
            + particles[i].b_field.z.powi(2);
        let rho_i = density_approx(&particles[i]);
        let v_a = alfven_speed_magnitude(b2, rho_i);

        let streaming_loss = if streaming_coefficient > 0.0 && v_a > 1e-30 {
            let l_min = h_i;
            streaming_coefficient * v_a * e_cr / l_min.max(1e-30)
        } else {
            0.0
        };

        let total_loss = compressional_term + streaming_loss;
        particles[i].cr_energy = (e_cr + total_loss * dt).max(0.0);
    }
}

pub fn cr_pressure_backreaction(particles: &mut [Particle], periodic_box: Option<f64>) {
    let n = particles.len();

    let mut grad_p_cr_x = vec![0.0_f64; n];
    let mut grad_p_cr_y = vec![0.0_f64; n];
    let mut grad_p_cr_z = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }

        let h_i = particles[i].smoothing_length.max(1e-10);
        let rho_i = density_approx(&particles[i]);
        if rho_i < 1e-30 {
            continue;
        }

        let p_cr_i = (GAMMA_CR - 1.0) * rho_i * particles[i].cr_energy.max(0.0);

        for j in (i + 1)..n {
            if particles[j].ptype != ParticleType::Gas {
                continue;
            }

            let h_j = particles[j].smoothing_length.max(1e-10);
            let rho_j = density_approx(&particles[j]);
            let p_cr_j = if rho_j > 1e-30 {
                (GAMMA_CR - 1.0) * rho_j * particles[j].cr_energy.max(0.0)
            } else {
                0.0
            };

            let dr = periodic_delta(particles[i].position, particles[j].position, periodic_box);
            let r = dr.norm();
            if r < 1e-14 || r > 2.0 * h_i.max(h_j) {
                continue;
            }

            let dwdr = grad_w_approx(r, 0.5 * (h_i + h_j));
            let rhat_x = dr.x / r;
            let rhat_y = dr.y / r;
            let rhat_z = dr.z / r;

            let p_avg = (p_cr_i + p_cr_j) / 2.0;
            let term = p_avg * dwdr;

            grad_p_cr_x[i] += particles[j].mass * term * rhat_x;
            grad_p_cr_y[i] += particles[j].mass * term * rhat_y;
            grad_p_cr_z[i] += particles[j].mass * term * rhat_z;

            grad_p_cr_x[j] -= particles[i].mass * term * rhat_x;
            grad_p_cr_y[j] -= particles[i].mass * term * rhat_y;
            grad_p_cr_z[j] -= particles[i].mass * term * rhat_z;
        }
    }

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        particles[i].acceleration.x += grad_p_cr_x[i];
        particles[i].acceleration.y += grad_p_cr_y[i];
        particles[i].acceleration.z += grad_p_cr_z[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gas(x: f64, y: f64, z: f64, cr_energy: f64, bz: f64) -> Particle {
        let mut p = Particle::new(0, 1.0, Vec3::new(x, y, z), Vec3::zero());
        p.ptype = ParticleType::Gas;
        p.cr_energy = cr_energy;
        p.b_field = Vec3::new(0.0, 0.0, bz);
        p.smoothing_length = 1.0;
        p
    }

    #[test]
    fn streaming_no_compression_no_change() {
        let mut particles = vec![
            make_gas(0.0, 0.0, 0.0, 1.0, 0.0),
            make_gas(2.0, 0.0, 0.0, 1.0, 0.0),
        ];

        let e_before = particles[0].cr_energy;
        streaming_crk(&mut particles, 0.01, 0.0, None);
        assert!((particles[0].cr_energy - e_before).abs() < 1e-10);
    }

    #[test]
    fn cr_pressure_incompressible_zero_force() {
        let mut particles = vec![
            make_gas(0.0, 0.0, 0.0, 0.0, 1.0),
            make_gas(2.0, 0.0, 0.0, 0.0, 1.0),
        ];

        cr_pressure_backreaction(&mut particles, None);

        assert!((particles[0].acceleration.x).abs() < 1e-10);
        assert!((particles[1].acceleration.x).abs() < 1e-10);
    }

    #[test]
    fn cr_pressure_symmetry() {
        let mut particles = vec![
            make_gas(0.0, 0.0, 0.0, 1.0, 1.0),
            make_gas(2.0, 0.0, 0.0, 1.0, 1.0),
        ];

        cr_pressure_backreaction(&mut particles, None);

        assert!((particles[0].acceleration.x + particles[1].acceleration.x).abs() < 1e-10);
    }
}
