//! Perfiles de velocidad σ_v(r) dentro de halos (Phase 73).
//!
//! Calcula la dispersión de velocidad radial y 3D en anillos esféricos
//! alrededor del centro del halo.
//!
//! ## Estadísticas calculadas por bin
//!
//! - `v_r_mean`  — velocidad radial media ⟨v_r⟩(r) (flujo neto de partículas).
//! - `sigma_r`   — dispersión de velocidad radial σ_r(r) = sqrt(⟨v_r²⟩ - ⟨v_r⟩²).
//! - `sigma_t`   — dispersión de velocidad tangencial σ_t(r).
//! - `sigma_3d`  — dispersión total 3D σ₃D(r) = sqrt(σ_r² + 2σ_t²) (isotrópia).
//! - `n_part`    — número de partículas en el bin.
//!
//! ## Referencia
//!
//! Springel et al. (2001); NFW (1997) §4; Klypin et al. (2001).

use gadget_ng_core::Vec3;

/// Un bin del perfil de velocidad.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VelocityProfileBin {
    /// Radio central del bin (unidades de la simulación).
    pub r: f64,
    /// Radio mínimo del bin.
    pub r_lo: f64,
    /// Radio máximo del bin.
    pub r_hi: f64,
    /// Velocidad radial media ⟨v_r⟩.
    pub v_r_mean: f64,
    /// Dispersión de velocidad radial σ_r.
    pub sigma_r: f64,
    /// Dispersión de velocidad tangencial σ_t.
    pub sigma_t: f64,
    /// Dispersión total 3D σ₃D.
    pub sigma_3d: f64,
    /// Número de partículas en el bin.
    pub n_part: usize,
}

/// Parámetros para el cálculo del perfil de velocidad.
#[derive(Debug, Clone)]
pub struct VelocityProfileParams {
    /// Número de bins radiales (escala logarítmica).
    pub n_bins: usize,
    /// Radio mínimo del perfil (ej. softening).
    pub r_min: f64,
    /// Radio máximo del perfil (ej. R_200).
    pub r_max: f64,
    /// Usar bins logarítmicos (true) o lineales (false).
    pub log_bins: bool,
}

impl Default for VelocityProfileParams {
    fn default() -> Self {
        Self {
            n_bins: 20,
            r_min: 0.01,
            r_max: 1.0,
            log_bins: true,
        }
    }
}

/// Calcula el perfil de velocidad de un halo.
///
/// # Parámetros
/// - `positions`  — posiciones de las partículas.
/// - `velocities` — velocidades de las partículas.
/// - `masses`     — masas de las partículas.
/// - `center`     — posición del centro del halo (ej. partícula más densa).
/// - `v_center`   — velocidad del centro de masa del halo.
/// - `params`     — parámetros de binning.
///
/// # Retorna
/// Vec de `VelocityProfileBin` ordenado por r creciente. Los bins vacíos se omiten.
pub fn velocity_profile(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    center: Vec3,
    v_center: Vec3,
    params: &VelocityProfileParams,
) -> Vec<VelocityProfileBin> {
    let n_bins = params.n_bins;

    // ── Construir bordes del bin ─────────────────────────────────────────
    let edges = if params.log_bins {
        log_edges(params.r_min, params.r_max, n_bins)
    } else {
        lin_edges(params.r_min, params.r_max, n_bins)
    };

    // ── Acumuladores ─────────────────────────────────────────────────────
    let mut vr_sum = vec![0.0f64; n_bins];
    let mut vr2_sum = vec![0.0f64; n_bins];
    let mut vt2_sum = vec![0.0f64; n_bins];
    let mut count = vec![0usize; n_bins];

    for (i, (&pos, &vel)) in positions.iter().zip(velocities.iter()).enumerate() {
        let _m = if i < masses.len() { masses[i] } else { 1.0 };

        // Vector relativo
        let rx = pos.x - center.x;
        let ry = pos.y - center.y;
        let rz = pos.z - center.z;
        let r = (rx * rx + ry * ry + rz * rz).sqrt();

        if r < params.r_min || r > params.r_max {
            continue;
        }

        // Velocidad relativa al centro de masa
        let vx = vel.x - v_center.x;
        let vy = vel.y - v_center.y;
        let vz = vel.z - v_center.z;

        // Velocidad radial v_r = (v · r̂)
        let r_hat_x = rx / r;
        let r_hat_y = ry / r;
        let r_hat_z = rz / r;
        let v_r = vx * r_hat_x + vy * r_hat_y + vz * r_hat_z;

        // Velocidad total y tangencial v_t = sqrt(|v|² - v_r²)
        let v2 = vx * vx + vy * vy + vz * vz;
        let vt2 = (v2 - v_r * v_r).max(0.0);

        // Bin
        let bin = find_bin(&edges, r);
        if let Some(b) = bin {
            vr_sum[b] += v_r;
            vr2_sum[b] += v_r * v_r;
            vt2_sum[b] += vt2;
            count[b] += 1;
        }
    }

    // ── Calcular dispersiones ─────────────────────────────────────────────
    let mut result = Vec::new();
    for b in 0..n_bins {
        let n = count[b];
        if n == 0 {
            continue;
        }
        let nf = n as f64;
        let vr_mean = vr_sum[b] / nf;
        let sigma_r = (vr2_sum[b] / nf - vr_mean * vr_mean).max(0.0).sqrt();
        let sigma_t = (vt2_sum[b] / nf / 2.0).sqrt(); // cada componente tangencial
        let sigma_3d = (sigma_r * sigma_r + 2.0 * sigma_t * sigma_t).sqrt();

        let r_lo = edges[b];
        let r_hi = edges[b + 1];
        let r_cen = if params.log_bins {
            (r_lo * r_hi).sqrt()
        } else {
            (r_lo + r_hi) * 0.5
        };

        result.push(VelocityProfileBin {
            r: r_cen,
            r_lo,
            r_hi,
            v_r_mean: vr_mean,
            sigma_r,
            sigma_t,
            sigma_3d,
            n_part: n,
        });
    }

    result
}

/// Calcula la dispersión de velocidad 1D (isótropa) σ_1D = σ₃D / sqrt(3).
///
/// Útil para comparar con observaciones de dispersión de velocidad de la
/// línea de visión.
pub fn sigma_1d(sigma_3d: f64) -> f64 {
    sigma_3d / 3.0f64.sqrt()
}

/// Calcula el parámetro de anisotropía de Binney β(r) = 1 − σ_t² / σ_r².
///
/// - β = 0: isótropa.
/// - β = 1: puramente radial.
/// - β < 0: órbitas tangenciales dominan.
pub fn velocity_anisotropy(profile: &[VelocityProfileBin]) -> Vec<(f64, f64)> {
    profile
        .iter()
        .map(|b| {
            let beta = if b.sigma_r > 0.0 {
                1.0 - b.sigma_t * b.sigma_t / (b.sigma_r * b.sigma_r)
            } else {
                0.0
            };
            (b.r, beta)
        })
        .collect()
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn log_edges(r_min: f64, r_max: f64, n: usize) -> Vec<f64> {
    let log_min = r_min.ln();
    let log_max = r_max.ln();
    let d = (log_max - log_min) / n as f64;
    (0..=n).map(|i| (log_min + i as f64 * d).exp()).collect()
}

fn lin_edges(r_min: f64, r_max: f64, n: usize) -> Vec<f64> {
    let d = (r_max - r_min) / n as f64;
    (0..=n).map(|i| r_min + i as f64 * d).collect()
}

fn find_bin(edges: &[f64], r: f64) -> Option<usize> {
    if r < edges[0] || r >= *edges.last().unwrap() {
        return None;
    }
    let b = edges.partition_point(|&e| e <= r).saturating_sub(1);
    Some(b.min(edges.len() - 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    /// Crea una distribución esférica isotrópica de partículas.
    fn make_isotropic(n: usize, r_max: f64, v0: f64) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>) {
        let mut pos = Vec::new();
        let mut vel = Vec::new();
        let mass = vec![1.0; n];
        let mut seed = 42u64;
        let lcg = |s: &mut u64| {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 33) as f64) / (u32::MAX as f64)
        };
        for _ in 0..n {
            // Posición aleatoria en esfera de radio r_max (rechazo)
            loop {
                let x = (lcg(&mut seed) - 0.5) * 2.0 * r_max;
                let y = (lcg(&mut seed) - 0.5) * 2.0 * r_max;
                let z = (lcg(&mut seed) - 0.5) * 2.0 * r_max;
                if x * x + y * y + z * z <= r_max * r_max {
                    pos.push(Vec3::new(x, y, z));
                    // Velocidad aleatoria isótropa
                    let vx = (lcg(&mut seed) - 0.5) * 2.0 * v0;
                    let vy = (lcg(&mut seed) - 0.5) * 2.0 * v0;
                    let vz = (lcg(&mut seed) - 0.5) * 2.0 * v0;
                    vel.push(Vec3::new(vx, vy, vz));
                    break;
                }
            }
        }
        (pos, vel, mass)
    }

    #[test]
    fn profile_bins_nonempty() {
        let (pos, vel, mass) = make_isotropic(200, 1.0, 1.0);
        let params = VelocityProfileParams {
            n_bins: 10,
            r_min: 0.01,
            r_max: 1.0,
            log_bins: true,
        };
        let center = Vec3::new(0.0, 0.0, 0.0);
        let v_cen = Vec3::new(0.0, 0.0, 0.0);
        let profile = velocity_profile(&pos, &vel, &mass, center, v_cen, &params);
        assert!(!profile.is_empty(), "Perfil no debe estar vacío");
    }

    #[test]
    fn profile_sigma_positive() {
        let (pos, vel, mass) = make_isotropic(500, 1.0, 2.0);
        let params = VelocityProfileParams {
            n_bins: 8,
            r_min: 0.05,
            r_max: 0.9,
            log_bins: true,
        };
        let center = Vec3::new(0.0, 0.0, 0.0);
        let v_cen = Vec3::new(0.0, 0.0, 0.0);
        let profile = velocity_profile(&pos, &vel, &mass, center, v_cen, &params);
        for b in &profile {
            assert!(b.sigma_r >= 0.0, "σ_r debe ser >= 0: {}", b.sigma_r);
            assert!(b.sigma_3d >= 0.0, "σ₃D debe ser >= 0: {}", b.sigma_3d);
        }
    }

    #[test]
    fn profile_radial_ordering() {
        let (pos, vel, mass) = make_isotropic(300, 1.0, 1.0);
        let params = VelocityProfileParams::default();
        let center = Vec3::new(0.0, 0.0, 0.0);
        let v_cen = Vec3::new(0.0, 0.0, 0.0);
        let profile = velocity_profile(&pos, &vel, &mass, center, v_cen, &params);
        for i in 1..profile.len() {
            assert!(profile[i].r > profile[i - 1].r, "r debe ser creciente");
        }
    }

    #[test]
    fn profile_isotropic_beta_finite() {
        // Para distribución isótropa, β debe ser finito y acotado
        let (pos, vel, mass) = make_isotropic(5000, 1.0, 1.0);
        let params = VelocityProfileParams {
            n_bins: 3,
            r_min: 0.2,
            r_max: 0.85,
            log_bins: false,
        };
        let center = Vec3::new(0.0, 0.0, 0.0);
        let v_cen = Vec3::new(0.0, 0.0, 0.0);
        let profile = velocity_profile(&pos, &vel, &mass, center, v_cen, &params);
        let anisotropy = velocity_anisotropy(&profile);
        // β ∈ (-∞, 1], verificar que es finito y <= 1
        for (r, beta) in &anisotropy {
            assert!(beta.is_finite(), "β no finito en r={r}");
            assert!(*beta <= 1.0 + 1e-10, "β > 1 en r={r}: {beta}");
        }
    }

    #[test]
    fn sigma_1d_relationship() {
        // σ_1D = σ_3D / sqrt(3)
        let sigma_3d = 300.0;
        let s1d = sigma_1d(sigma_3d);
        assert!((s1d - sigma_3d / 3.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn lin_bins_cover_range() {
        let edges = lin_edges(0.1, 1.0, 5);
        assert_eq!(edges.len(), 6);
        assert!((edges[0] - 0.1).abs() < 1e-10);
        assert!((edges[5] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn log_bins_cover_range() {
        let edges = log_edges(0.01, 10.0, 10);
        assert_eq!(edges.len(), 11);
        assert!((edges[0] - 0.01).abs() < 1e-10);
        assert!((edges[10] - 10.0).abs() < 1e-8);
    }

    #[test]
    fn radial_only_particles_high_beta() {
        // Partículas con dispersión predominantemente radial → β > 0
        let n = 500;
        let mut pos = Vec::new();
        let mut vel = Vec::new();
        let mass = vec![1.0; n];
        let mut seed = 77u64;
        let lcg = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 33) as f64) / (u32::MAX as f64)
        };
        for _ in 0..n {
            let x = (lcg(&mut seed) - 0.5) * 2.0;
            let y = (lcg(&mut seed) - 0.5) * 2.0;
            let z = (lcg(&mut seed) - 0.5) * 2.0;
            let r = (x * x + y * y + z * z).sqrt().max(1e-6);
            pos.push(Vec3::new(x, y, z));
            // Velocidad: componente radial grande + ruido tangencial pequeño
            let vr = (lcg(&mut seed) - 0.5) * 2.0 * 3.0; // σ_r ≈ 3
            let vt_scale = 0.3;                             // σ_t ≈ 0.3
            let tx = lcg(&mut seed) - 0.5;
            let ty = lcg(&mut seed) - 0.5;
            let tz = lcg(&mut seed) - 0.5;
            // Crear un vector tangente ortogonal al radio
            let r_hat = [x / r, y / r, z / r];
            let dot = tx * r_hat[0] + ty * r_hat[1] + tz * r_hat[2];
            let tang = [
                (tx - dot * r_hat[0]) * vt_scale,
                (ty - dot * r_hat[1]) * vt_scale,
                (tz - dot * r_hat[2]) * vt_scale,
            ];
            vel.push(Vec3::new(
                vr * r_hat[0] + tang[0],
                vr * r_hat[1] + tang[1],
                vr * r_hat[2] + tang[2],
            ));
        }
        let params = VelocityProfileParams {
            n_bins: 2,
            r_min: 0.2,
            r_max: 1.4,
            log_bins: false,
        };
        let center = Vec3::new(0.0, 0.0, 0.0);
        let v_cen = Vec3::new(0.0, 0.0, 0.0);
        let profile = velocity_profile(&pos, &vel, &mass, center, v_cen, &params);
        let anisotropy = velocity_anisotropy(&profile);
        for (r, beta) in &anisotropy {
            assert!(*beta > 0.0, "β debe ser positivo (radial) en r={r}: {beta}");
        }
    }
}
