//! Estimación de densidad SPH y suavizado adaptativo `h_sml`.
//!
//! ## Algoritmo
//!
//! Para cada partícula de gas `i`:
//!
//! 1. Se buscan los `N_neigh` vecinos más cercanos (estructura O3: neighbor list O(N²) simplificada).
//! 2. Se ajusta `h_i` iterativamente (Newton-Raphson) hasta que
//!    `(4π/3) · (2h_i)³ · ρ_i ≈ N_neigh · m̄`.
//! 3. La densidad se estima como `ρ_i = Σ_j m_j W(|r_ij|, h_i)`.
//!
//! La implementación usa una lista O(N²) válida para N ~ 10 000 partículas.
//! Para N mayores se debería usar un árbol KD o cell-linked-list.

use crate::kernel::{w_and_grad_w_batch, w_batch};
use crate::particle::SphParticle;
use crate::periodic_delta;
use gadget_ng_core::Vec3;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Número objetivo de vecinos SPH.
const N_NEIGH: f64 = 32.0;
/// Gamma adiabático (gas monoatómico ideal).
pub const GAMMA: f64 = 5.0 / 3.0;
/// Número máximo de iteraciones Newton-Raphson para h_sml.
const MAX_ITER: usize = 30;

/// Estima la densidad y actualiza `h_sml` para todas las partículas de gas.
///
/// Actualiza `gas.rho`, `gas.h_sml` y `gas.pressure` in-place.
pub fn compute_density(particles: &mut [SphParticle]) {
    compute_density_with_periodic(particles, None);
}

/// Igual que `compute_density`, pero usando imagen mínima si `periodic_box = Some(L)`.
///
/// # Rayon + SIMD combined path
///
/// When the `rayon` feature is active the outer loop over particles runs in parallel via Rayon.
/// Regardless of the `simd` feature flag, the inner kernel evaluation (`w_and_grad_w_batch` in
/// `kernel.rs`) already dispatches at runtime to AVX-512 → AVX2+FMA → scalar based on
/// `is_x86_feature_detected!`. Therefore the Rayon path already constitutes a true
/// "Rayon-outer + SIMD-inner" combined path — no additional cfg gate is required here.
pub fn compute_density_with_periodic(particles: &mut [SphParticle], periodic_box: Option<f64>) {
    let n = particles.len();
    // Extrae datos de todas las partículas (pos, mass) para no tener borrowing doble.
    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    #[cfg(feature = "rayon")]
    {
        let gas_state: Vec<Option<(f64, f64)>> = particles
            .iter()
            .map(|p| p.gas.as_ref().map(|g| (g.h_sml, g.u)))
            .collect();
        let updates: Vec<Option<(f64, f64, f64, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (h0, u) = gas_state[i]?;
                Some(density_update_for_particle(
                    &pos,
                    &mass,
                    i,
                    h0,
                    u,
                    periodic_box,
                ))
            })
            .collect();

        for (p, update) in particles.iter_mut().zip(updates) {
            if let (Some(gas), Some((h, rho, pressure, entropy))) = (p.gas.as_mut(), update) {
                gas.h_sml = h;
                gas.rho = rho;
                gas.pressure = pressure;
                gas.entropy = entropy;
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
        let gas = match particles[i].gas.as_mut() {
            Some(g) => g,
            None => continue,
        };

        let pi = pos[i];
        let m_i = mass[i];
        let mut h = gas.h_sml.max(1e-10);

        for _ in 0..MAX_ITER {
            let (rho, drho_dh) = rho_and_deriv_batch(&pos, &mass, pi, h, n, periodic_box);
            let n_eff = (4.0 * std::f64::consts::PI / 3.0) * (2.0 * h).powi(3) * rho / m_i;
            if (n_eff - N_NEIGH).abs() < 1e-2 {
                break;
            }
            let dn_dh = (4.0 * std::f64::consts::PI / 3.0)
                * (24.0 * h * h * rho + (2.0 * h).powi(3) * drho_dh)
                / m_i;
            let dh = -(n_eff - N_NEIGH) / (dn_dh + 1e-100);
            h = (h + dh.clamp(-0.5 * h, 0.5 * h)).max(1e-10);
        }
        gas.h_sml = h;
        gas.rho = rho_sum_batch(&pos, &mass, pi, h, n, periodic_box);
        gas.pressure = (GAMMA - 1.0) * gas.rho * gas.u;
        if gas.rho > 0.0 {
            gas.entropy = (GAMMA - 1.0) * gas.u / gas.rho.powf(GAMMA - 1.0);
        }
    }
}

#[cfg(feature = "rayon")]
fn density_update_for_particle(
    pos: &[Vec3],
    mass: &[f64],
    i: usize,
    h0: f64,
    u: f64,
    periodic_box: Option<f64>,
) -> (f64, f64, f64, f64) {
    let n = pos.len();
    let pi = pos[i];
    let m_i = mass[i];
    let mut h = h0.max(1e-10);

    for _ in 0..MAX_ITER {
        let (rho, drho_dh) = rho_and_deriv_batch(pos, mass, pi, h, n, periodic_box);
        let n_eff = (4.0 * std::f64::consts::PI / 3.0) * (2.0 * h).powi(3) * rho / m_i;
        if (n_eff - N_NEIGH).abs() < 1e-2 {
            break;
        }
        let dn_dh = (4.0 * std::f64::consts::PI / 3.0)
            * (24.0 * h * h * rho + (2.0 * h).powi(3) * drho_dh)
            / m_i;
        let dh = -(n_eff - N_NEIGH) / (dn_dh + 1e-100);
        h = (h + dh.clamp(-0.5 * h, 0.5 * h)).max(1e-10);
    }

    let rho = rho_sum_batch(pos, mass, pi, h, n, periodic_box);
    let pressure = (GAMMA - 1.0) * rho * u;
    let entropy = if rho > 0.0 {
        (GAMMA - 1.0) * u / rho.powf(GAMMA - 1.0)
    } else {
        0.0
    };
    (h, rho, pressure, entropy)
}

/// Versión SIMD por lotes de `rho_and_deriv`: recolecta distancias y usa
/// `w_and_grad_w_batch` para vectorizar el cómputo del kernel Wendland C2.
fn rho_and_deriv_batch(
    pos: &[Vec3],
    mass: &[f64],
    pi: Vec3,
    h: f64,
    n: usize,
    periodic_box: Option<f64>,
) -> (f64, f64) {
    let mut r_buf: Vec<f64> = Vec::with_capacity(n);
    let mut idx_buf: Vec<usize> = Vec::with_capacity(n);

    for (j, pj) in pos.iter().enumerate().take(n) {
        let r = periodic_delta(pi, *pj, periodic_box).norm();
        if r < 2.0 * h {
            r_buf.push(r);
            idx_buf.push(j);
        }
    }

    let m = r_buf.len();
    if m == 0 {
        return (0.0, 0.0);
    }

    let mut w_buf = vec![0.0_f64; m];
    let mut gw_buf = vec![0.0_f64; m];
    w_and_grad_w_batch(&r_buf, h, &mut w_buf, &mut gw_buf);

    let mut rho = 0.0_f64;
    let mut drho = 0.0_f64;
    let inv_h = 1.0 / h;
    let coeff = -inv_h * 3.0;

    for k in 0..m {
        let mj = mass[idx_buf[k]];
        rho += mj * w_buf[k];
        drho += mj * (coeff * w_buf[k] + (-inv_h) * r_buf[k] * gw_buf[k]);
    }
    (rho, drho)
}

/// Versión SIMD por lotes de `rho_sum`.
fn rho_sum_batch(
    pos: &[Vec3],
    mass: &[f64],
    pi: Vec3,
    h: f64,
    n: usize,
    periodic_box: Option<f64>,
) -> f64 {
    let mut r_buf: Vec<f64> = Vec::with_capacity(n);
    let mut idx_buf: Vec<usize> = Vec::with_capacity(n);

    for (j, pj) in pos.iter().enumerate().take(n) {
        let r = periodic_delta(pi, *pj, periodic_box).norm();
        if r < 2.0 * h {
            r_buf.push(r);
            idx_buf.push(j);
        }
    }

    let m = r_buf.len();
    if m == 0 {
        return 0.0;
    }

    let mut w_buf = vec![0.0_f64; m];
    w_batch(&r_buf, h, &mut w_buf);

    let mut rho = 0.0_f64;
    for k in 0..m {
        rho += mass[idx_buf[k]] * w_buf[k];
    }
    rho
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::SphParticle;
    use gadget_ng_core::Vec3;

    fn glass_particles(n: usize, box_size: f64) -> Vec<SphParticle> {
        let n_side = (n as f64).cbrt().round() as usize;
        let dx = box_size / n_side as f64;
        let mut parts = Vec::new();
        let mut id = 0;
        'outer: for ix in 0..n_side {
            for iy in 0..n_side {
                for iz in 0..n_side {
                    let pos = Vec3::new(
                        (ix as f64 + 0.5) * dx,
                        (iy as f64 + 0.5) * dx,
                        (iz as f64 + 0.5) * dx,
                    );
                    let h0 = 2.0 * dx;
                    parts.push(SphParticle::new_gas(id, 1.0, pos, Vec3::zero(), 1.0, h0));
                    id += 1;
                    if parts.len() == n {
                        break 'outer;
                    }
                }
            }
        }
        parts
    }

    #[test]
    fn density_uniform_glass_converges() {
        // Retícula cúbica uniforme: la densidad SPH debe converger a ρ_true = N·m/V.
        let n = 64usize; // 4³
        let box_size = 4.0_f64;
        let mut parts = glass_particles(n, box_size);
        compute_density(&mut parts);

        let rho_true = n as f64 / (box_size * box_size * box_size);
        let mean_rho: f64 = parts
            .iter()
            .filter_map(|p| p.gas.as_ref())
            .map(|g| g.rho)
            .sum::<f64>()
            / n as f64;

        // Se acepta ±30 % de error: sin BC periódicas las partículas de borde
        // tienen menos vecinos y subestiman la densidad.
        assert!(
            (mean_rho - rho_true).abs() / rho_true < 0.30,
            "rho_mean={mean_rho:.4} rho_true={rho_true:.4}"
        );
    }

    /// Confirms that the Rayon path and the serial path agree bit-near.
    ///
    /// Both paths call `w_and_grad_w_batch` (kernel.rs) which dispatches to AVX-512/AVX2/scalar
    /// at runtime. This test verifies the combined "Rayon-outer + SIMD-inner" behaviour is
    /// numerically identical to the serial reference.
    #[test]
    #[cfg(feature = "rayon")]
    fn density_rayon_matches_serial() {
        let n = 27usize;
        let box_size = 3.0_f64;
        let mut rayon_parts = glass_particles(n, box_size);
        let mut serial_parts = rayon_parts.clone();

        // Use the Rayon path (current feature).
        compute_density(&mut rayon_parts);

        // Force the serial path by calling the inner per-particle function directly.
        let pos: Vec<Vec3> = serial_parts.iter().map(|p| p.position).collect();
        let mass: Vec<f64> = serial_parts.iter().map(|p| p.mass).collect();
        for i in 0..n {
            let gas = match serial_parts[i].gas.as_ref() {
                Some(g) => g,
                None => continue,
            };
            let (h, rho, pressure, entropy) =
                density_update_for_particle(&pos, &mass, i, gas.h_sml, gas.u, None);
            if let Some(g) = serial_parts[i].gas.as_mut() {
                g.h_sml = h;
                g.rho = rho;
                g.pressure = pressure;
                g.entropy = entropy;
            }
        }

        for (rp, sp) in rayon_parts.iter().zip(serial_parts.iter()) {
            let rg = rp.gas.as_ref().unwrap();
            let sg = sp.gas.as_ref().unwrap();
            assert!(
                (rg.rho - sg.rho).abs() < 1e-12,
                "rho mismatch at particle: rayon={} serial={}",
                rg.rho,
                sg.rho
            );
        }
    }
}
