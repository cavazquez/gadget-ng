//! Parámetro de spin λ de halos FoF (Phase 72).
//!
//! ## Definiciones
//!
//! **Peebles (1971)**:
//! ```text
//! λ = |L| / (M × V_vir × R_vir)
//! ```
//! donde `V_vir = sqrt(G×M/R_vir)` y `R_vir ≈ R_200`.
//!
//! **Bullock et al. (2001)** (versión simplificada, más común en la literatura moderna):
//! ```text
//! λ' = |L| / (sqrt(2) × M × V_vir × R_vir)
//! ```
//!
//! El momento angular `L` se calcula respecto al centro de masa:
//! ```text
//! L = Σ_i m_i × (r_i - r_com) × (v_i - v_com)
//! ```
//!
//! ## Referencia
//!
//! Peebles (1971), A&A 11, 377; Bullock et al. (2001), ApJ 555, 240.

use gadget_ng_core::Vec3;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const G_INTERNAL: f64 = 4.302e-3; // pc M_sun⁻¹ (km/s)² — en unidades de kpc·(km/s)²/M_sun = 4.302e-6

/// Resultado de spin para un halo.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HaloSpin {
    /// Masa total del halo (unidades de la simulación).
    pub mass: f64,
    /// Radio virial R_200 (unidades de la simulación).
    pub r200: f64,
    /// Centro de masa (x,y,z).
    pub pos_com: [f64; 3],
    /// Velocidad del centro de masa (vx,vy,vz).
    pub vel_com: [f64; 3],
    /// Momento angular total L = (Lx, Ly, Lz).
    pub angular_momentum: [f64; 3],
    /// Módulo |L|.
    pub l_mag: f64,
    /// Parámetro de spin Peebles λ.
    pub lambda_peebles: f64,
    /// Parámetro de spin Bullock λ'.
    pub lambda_bullock: f64,
}

/// Parámetros para el cálculo de spin.
#[derive(Debug, Clone)]
pub struct SpinParams {
    /// Constante gravitacional G en unidades internas de la simulación.
    /// Valor por defecto: 4.302e-3 (kpc, km/s, M_sun).
    pub g_newton: f64,
    /// Factor de sobredensidad para R_vir (por defecto 200 × ρ_crit).
    pub delta_vir: f64,
    /// Densidad crítica ρ_crit en unidades internas.
    pub rho_crit: f64,
}

impl Default for SpinParams {
    fn default() -> Self {
        Self {
            g_newton: G_INTERNAL,
            delta_vir: 200.0,
            rho_crit: 2.775e11, // M_sun/Mpc³ para H0=100 h km/s/Mpc
        }
    }
}

/// Calcula el parámetro de spin λ para un halo dado su lista de partículas.
///
/// # Parámetros
/// - `positions`  — posiciones de las partículas del halo.
/// - `velocities` — velocidades de las partículas del halo.
/// - `masses`     — masas de las partículas del halo.
/// - `params`     — parámetros del cálculo.
///
/// # Retorna
/// `None` si el halo no tiene partículas o tiene masa cero.
pub fn halo_spin(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    params: &SpinParams,
) -> Option<HaloSpin> {
    let n = positions.len();
    if n == 0 || masses.is_empty() {
        return None;
    }

    #[cfg(feature = "rayon")]
    let mass_total: f64 = masses.par_iter().sum();
    #[cfg(not(feature = "rayon"))]
    let mass_total: f64 = mass_sum(masses);
    if mass_total <= 0.0 {
        return None;
    }

    // ── Centro de masa ────────────────────────────────────────────────────
    let pos_com = center_of_mass(positions, masses, mass_total);
    let vel_com = velocity_center(velocities, masses, mass_total);

    // ── Momento angular L = Σ m_i × (r_i - r_com) × (v_i - v_com) ───────
    #[cfg(feature = "rayon")]
    let (lx, ly, lz) = positions
        .par_iter()
        .zip(velocities.par_iter())
        .enumerate()
        .map(|(i, (&pos, &vel))| angular_momentum_term(i, pos, vel, masses, pos_com, vel_com))
        .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    #[cfg(not(feature = "rayon"))]
    let (lx, ly, lz) = angular_momentum_sum(positions, velocities, masses, pos_com, vel_com);

    let l_mag = (lx * lx + ly * ly + lz * lz).sqrt();

    // ── Radio virial R_200 = (3M / (4π × Δ_vir × ρ_crit))^(1/3) ─────────
    let r200 = r200_from_mass(mass_total, params);

    // ── Velocidad virial V_vir = sqrt(G × M / R_200) ─────────────────────
    let v_vir = if r200 > 0.0 {
        (params.g_newton * mass_total / r200).sqrt()
    } else {
        1.0
    };

    // ── Parámetros de spin ────────────────────────────────────────────────
    let denom = mass_total * v_vir * r200;
    let lambda_peebles = if denom > 0.0 { l_mag / denom } else { 0.0 };
    let lambda_bullock = lambda_peebles / std::f64::consts::SQRT_2;

    Some(HaloSpin {
        mass: mass_total,
        r200,
        pos_com,
        vel_com,
        angular_momentum: [lx, ly, lz],
        l_mag,
        lambda_peebles,
        lambda_bullock,
    })
}

/// Calcula el spin para múltiples halos dados como índices de partículas.
///
/// # Parámetros
/// - `positions`   — posiciones de TODAS las partículas.
/// - `velocities`  — velocidades de TODAS las partículas.
/// - `masses`      — masas de TODAS las partículas.
/// - `halo_ids`    — para cada halo, lista de índices de partículas.
/// - `params`      — parámetros del cálculo.
pub fn compute_halo_spins(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    halo_ids: &[Vec<usize>],
    params: &SpinParams,
) -> Vec<Option<HaloSpin>> {
    compute_halo_spins_impl(positions, velocities, masses, halo_ids, params)
}

// ── Helpers ───────────────────────────────────────────────────────────────

#[cfg(not(feature = "rayon"))]
fn mass_sum(masses: &[f64]) -> f64 {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            unsafe {
                return mass_sum_avx512(masses);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                return mass_sum_avx2(masses);
            }
        }
    }
    masses.iter().sum()
}

#[cfg(feature = "rayon")]
fn compute_halo_spins_impl(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    halo_ids: &[Vec<usize>],
    params: &SpinParams,
) -> Vec<Option<HaloSpin>> {
    halo_ids
        .par_iter()
        .map(|ids| {
            let pos: Vec<Vec3> = ids.iter().map(|&i| positions[i]).collect();
            let vel: Vec<Vec3> = ids.iter().map(|&i| velocities[i]).collect();
            let mass: Vec<f64> = ids.iter().map(|&i| masses[i]).collect();
            halo_spin(&pos, &vel, &mass, params)
        })
        .collect()
}

#[cfg(not(feature = "rayon"))]
fn compute_halo_spins_impl(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    halo_ids: &[Vec<usize>],
    params: &SpinParams,
) -> Vec<Option<HaloSpin>> {
    halo_ids
        .iter()
        .map(|ids| {
            let pos: Vec<Vec3> = ids.iter().map(|&i| positions[i]).collect();
            let vel: Vec<Vec3> = ids.iter().map(|&i| velocities[i]).collect();
            let mass: Vec<f64> = ids.iter().map(|&i| masses[i]).collect();
            halo_spin(&pos, &vel, &mass, params)
        })
        .collect()
}

fn angular_momentum_term(
    i: usize,
    pos: Vec3,
    vel: Vec3,
    masses: &[f64],
    pos_com: [f64; 3],
    vel_com: [f64; 3],
) -> (f64, f64, f64) {
    let m = if i < masses.len() {
        masses[i]
    } else {
        masses[0]
    };
    let rx = pos.x - pos_com[0];
    let ry = pos.y - pos_com[1];
    let rz = pos.z - pos_com[2];
    let vx = vel.x - vel_com[0];
    let vy = vel.y - vel_com[1];
    let vz = vel.z - vel_com[2];
    // L += m × (r × v)
    (
        m * (ry * vz - rz * vy),
        m * (rz * vx - rx * vz),
        m * (rx * vy - ry * vx),
    )
}

fn center_of_mass(positions: &[Vec3], masses: &[f64], total_mass: f64) -> [f64; 3] {
    #[cfg(feature = "rayon")]
    {
        let (cx, cy, cz) = positions
            .par_iter()
            .enumerate()
            .map(|(i, &pos)| {
                let m = if i < masses.len() {
                    masses[i]
                } else {
                    masses[0]
                };
                (m * pos.x, m * pos.y, m * pos.z)
            })
            .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        [cx / total_mass, cy / total_mass, cz / total_mass]
    }

    #[cfg(not(feature = "rayon"))]
    {
        let (cx, cy, cz) = weighted_vec3_sum(positions, masses);
        [cx / total_mass, cy / total_mass, cz / total_mass]
    }
}

fn velocity_center(velocities: &[Vec3], masses: &[f64], total_mass: f64) -> [f64; 3] {
    #[cfg(feature = "rayon")]
    {
        let (vx, vy, vz) = velocities
            .par_iter()
            .enumerate()
            .map(|(i, &vel)| {
                let m = if i < masses.len() {
                    masses[i]
                } else {
                    masses[0]
                };
                (m * vel.x, m * vel.y, m * vel.z)
            })
            .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        [vx / total_mass, vy / total_mass, vz / total_mass]
    }

    #[cfg(not(feature = "rayon"))]
    {
        let (vx, vy, vz) = weighted_vec3_sum(velocities, masses);
        [vx / total_mass, vy / total_mass, vz / total_mass]
    }
}

#[cfg(not(feature = "rayon"))]
fn weighted_vec3_sum(values: &[Vec3], masses: &[f64]) -> (f64, f64, f64) {
    if values.len() > masses.len() {
        return weighted_vec3_sum_scalar(values, masses);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            unsafe {
                return weighted_vec3_sum_avx512(values, masses);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                return weighted_vec3_sum_avx2(values, masses);
            }
        }
    }

    weighted_vec3_sum_scalar(values, masses)
}

#[cfg(not(feature = "rayon"))]
fn weighted_vec3_sum_scalar(values: &[Vec3], masses: &[f64]) -> (f64, f64, f64) {
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sz = 0.0f64;
    for (i, &value) in values.iter().enumerate() {
        let m = if i < masses.len() {
            masses[i]
        } else {
            masses[0]
        };
        sx += m * value.x;
        sy += m * value.y;
        sz += m * value.z;
    }
    (sx, sy, sz)
}

#[cfg(not(feature = "rayon"))]
fn angular_momentum_sum(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    pos_com: [f64; 3],
    vel_com: [f64; 3],
) -> (f64, f64, f64) {
    if positions.len() > masses.len() || positions.len() != velocities.len() {
        return angular_momentum_sum_scalar(positions, velocities, masses, pos_com, vel_com);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            unsafe {
                return angular_momentum_sum_avx512(
                    positions, velocities, masses, pos_com, vel_com,
                );
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                return angular_momentum_sum_avx2(positions, velocities, masses, pos_com, vel_com);
            }
        }
    }

    angular_momentum_sum_scalar(positions, velocities, masses, pos_com, vel_com)
}

#[cfg(not(feature = "rayon"))]
fn angular_momentum_sum_scalar(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    pos_com: [f64; 3],
    vel_com: [f64; 3],
) -> (f64, f64, f64) {
    let mut lx = 0.0f64;
    let mut ly = 0.0f64;
    let mut lz = 0.0f64;
    for (i, (&pos, &vel)) in positions.iter().zip(velocities.iter()).enumerate() {
        let (dlx, dly, dlz) = angular_momentum_term(i, pos, vel, masses, pos_com, vel_com);
        lx += dlx;
        ly += dly;
        lz += dlz;
    }
    (lx, ly, lz)
}

fn r200_from_mass(mass: f64, params: &SpinParams) -> f64 {
    let rho_thresh = params.delta_vir * params.rho_crit;
    if rho_thresh <= 0.0 {
        return 0.0;
    }
    (3.0 * mass / (4.0 * std::f64::consts::PI * rho_thresh)).cbrt()
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn mass_sum_avx2(masses: &[f64]) -> f64 {
    let lanes = 4;
    let chunks = masses.len() / lanes * lanes;
    let mut acc = _mm256_setzero_pd();
    let mut i = 0;
    while i < chunks {
        // SAFETY: `i..i+4` is in-bounds by construction and unaligned loads are permitted.
        let m = unsafe { _mm256_loadu_pd(masses.as_ptr().add(i)) };
        acc = _mm256_add_pd(acc, m);
        i += lanes;
    }
    let mut tmp = [0.0; 4];
    // SAFETY: fixed-size stack array has exactly four f64 lanes.
    unsafe { _mm256_storeu_pd(tmp.as_mut_ptr(), acc) };
    tmp.into_iter().sum::<f64>() + masses[chunks..].iter().sum::<f64>()
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn mass_sum_avx512(masses: &[f64]) -> f64 {
    let lanes = 8;
    let chunks = masses.len() / lanes * lanes;
    let mut acc = _mm512_setzero_pd();
    let mut i = 0;
    while i < chunks {
        // SAFETY: `i..i+8` is in-bounds by construction and unaligned loads are permitted.
        let m = unsafe { _mm512_loadu_pd(masses.as_ptr().add(i)) };
        acc = _mm512_add_pd(acc, m);
        i += lanes;
    }
    let mut tmp = [0.0; 8];
    // SAFETY: fixed-size stack array has exactly eight f64 lanes.
    unsafe { _mm512_storeu_pd(tmp.as_mut_ptr(), acc) };
    tmp.into_iter().sum::<f64>() + masses[chunks..].iter().sum::<f64>()
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn weighted_vec3_sum_avx2(values: &[Vec3], masses: &[f64]) -> (f64, f64, f64) {
    let lanes = 4;
    let chunks = values.len() / lanes * lanes;
    let mut sx = _mm256_setzero_pd();
    let mut sy = _mm256_setzero_pd();
    let mut sz = _mm256_setzero_pd();

    let mut i = 0;
    while i < chunks {
        // SAFETY: `i..i+4` is in-bounds by construction and unaligned loads are permitted.
        let m = unsafe { _mm256_loadu_pd(masses.as_ptr().add(i)) };
        let x = _mm256_set_pd(
            values[i + 3].x,
            values[i + 2].x,
            values[i + 1].x,
            values[i].x,
        );
        let y = _mm256_set_pd(
            values[i + 3].y,
            values[i + 2].y,
            values[i + 1].y,
            values[i].y,
        );
        let z = _mm256_set_pd(
            values[i + 3].z,
            values[i + 2].z,
            values[i + 1].z,
            values[i].z,
        );
        sx = _mm256_fmadd_pd(m, x, sx);
        sy = _mm256_fmadd_pd(m, y, sy);
        sz = _mm256_fmadd_pd(m, z, sz);
        i += lanes;
    }

    let mut tx = [0.0; 4];
    let mut ty = [0.0; 4];
    let mut tz = [0.0; 4];
    // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
    unsafe {
        _mm256_storeu_pd(tx.as_mut_ptr(), sx);
        _mm256_storeu_pd(ty.as_mut_ptr(), sy);
        _mm256_storeu_pd(tz.as_mut_ptr(), sz);
    }
    let tail = weighted_vec3_sum_scalar(&values[chunks..], &masses[chunks..]);
    (
        tx.into_iter().sum::<f64>() + tail.0,
        ty.into_iter().sum::<f64>() + tail.1,
        tz.into_iter().sum::<f64>() + tail.2,
    )
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn weighted_vec3_sum_avx512(values: &[Vec3], masses: &[f64]) -> (f64, f64, f64) {
    let lanes = 8;
    let chunks = values.len() / lanes * lanes;
    let mut sx = _mm512_setzero_pd();
    let mut sy = _mm512_setzero_pd();
    let mut sz = _mm512_setzero_pd();

    let mut i = 0;
    while i < chunks {
        // SAFETY: `i..i+8` is in-bounds by construction and unaligned loads are permitted.
        let m = unsafe { _mm512_loadu_pd(masses.as_ptr().add(i)) };
        let x = _mm512_set_pd(
            values[i + 7].x,
            values[i + 6].x,
            values[i + 5].x,
            values[i + 4].x,
            values[i + 3].x,
            values[i + 2].x,
            values[i + 1].x,
            values[i].x,
        );
        let y = _mm512_set_pd(
            values[i + 7].y,
            values[i + 6].y,
            values[i + 5].y,
            values[i + 4].y,
            values[i + 3].y,
            values[i + 2].y,
            values[i + 1].y,
            values[i].y,
        );
        let z = _mm512_set_pd(
            values[i + 7].z,
            values[i + 6].z,
            values[i + 5].z,
            values[i + 4].z,
            values[i + 3].z,
            values[i + 2].z,
            values[i + 1].z,
            values[i].z,
        );
        sx = _mm512_fmadd_pd(m, x, sx);
        sy = _mm512_fmadd_pd(m, y, sy);
        sz = _mm512_fmadd_pd(m, z, sz);
        i += lanes;
    }

    let mut tx = [0.0; 8];
    let mut ty = [0.0; 8];
    let mut tz = [0.0; 8];
    // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
    unsafe {
        _mm512_storeu_pd(tx.as_mut_ptr(), sx);
        _mm512_storeu_pd(ty.as_mut_ptr(), sy);
        _mm512_storeu_pd(tz.as_mut_ptr(), sz);
    }
    let tail = weighted_vec3_sum_scalar(&values[chunks..], &masses[chunks..]);
    (
        tx.into_iter().sum::<f64>() + tail.0,
        ty.into_iter().sum::<f64>() + tail.1,
        tz.into_iter().sum::<f64>() + tail.2,
    )
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn angular_momentum_sum_avx2(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    pos_com: [f64; 3],
    vel_com: [f64; 3],
) -> (f64, f64, f64) {
    let lanes = 4;
    let chunks = positions.len() / lanes * lanes;
    let pcx = _mm256_set1_pd(pos_com[0]);
    let pcy = _mm256_set1_pd(pos_com[1]);
    let pcz = _mm256_set1_pd(pos_com[2]);
    let vcx = _mm256_set1_pd(vel_com[0]);
    let vcy = _mm256_set1_pd(vel_com[1]);
    let vcz = _mm256_set1_pd(vel_com[2]);
    let mut lx = _mm256_setzero_pd();
    let mut ly = _mm256_setzero_pd();
    let mut lz = _mm256_setzero_pd();

    let mut i = 0;
    while i < chunks {
        // SAFETY: `i..i+4` is in-bounds by construction and unaligned loads are permitted.
        let m = unsafe { _mm256_loadu_pd(masses.as_ptr().add(i)) };
        let px = _mm256_set_pd(
            positions[i + 3].x,
            positions[i + 2].x,
            positions[i + 1].x,
            positions[i].x,
        );
        let py = _mm256_set_pd(
            positions[i + 3].y,
            positions[i + 2].y,
            positions[i + 1].y,
            positions[i].y,
        );
        let pz = _mm256_set_pd(
            positions[i + 3].z,
            positions[i + 2].z,
            positions[i + 1].z,
            positions[i].z,
        );
        let vx = _mm256_set_pd(
            velocities[i + 3].x,
            velocities[i + 2].x,
            velocities[i + 1].x,
            velocities[i].x,
        );
        let vy = _mm256_set_pd(
            velocities[i + 3].y,
            velocities[i + 2].y,
            velocities[i + 1].y,
            velocities[i].y,
        );
        let vz = _mm256_set_pd(
            velocities[i + 3].z,
            velocities[i + 2].z,
            velocities[i + 1].z,
            velocities[i].z,
        );
        let rx = _mm256_sub_pd(px, pcx);
        let ry = _mm256_sub_pd(py, pcy);
        let rz = _mm256_sub_pd(pz, pcz);
        let dvx = _mm256_sub_pd(vx, vcx);
        let dvy = _mm256_sub_pd(vy, vcy);
        let dvz = _mm256_sub_pd(vz, vcz);
        lx = _mm256_fmadd_pd(
            m,
            _mm256_sub_pd(_mm256_mul_pd(ry, dvz), _mm256_mul_pd(rz, dvy)),
            lx,
        );
        ly = _mm256_fmadd_pd(
            m,
            _mm256_sub_pd(_mm256_mul_pd(rz, dvx), _mm256_mul_pd(rx, dvz)),
            ly,
        );
        lz = _mm256_fmadd_pd(
            m,
            _mm256_sub_pd(_mm256_mul_pd(rx, dvy), _mm256_mul_pd(ry, dvx)),
            lz,
        );
        i += lanes;
    }

    let mut tx = [0.0; 4];
    let mut ty = [0.0; 4];
    let mut tz = [0.0; 4];
    // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
    unsafe {
        _mm256_storeu_pd(tx.as_mut_ptr(), lx);
        _mm256_storeu_pd(ty.as_mut_ptr(), ly);
        _mm256_storeu_pd(tz.as_mut_ptr(), lz);
    }
    let tail = angular_momentum_sum_scalar(
        &positions[chunks..],
        &velocities[chunks..],
        &masses[chunks..],
        pos_com,
        vel_com,
    );
    (
        tx.into_iter().sum::<f64>() + tail.0,
        ty.into_iter().sum::<f64>() + tail.1,
        tz.into_iter().sum::<f64>() + tail.2,
    )
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn angular_momentum_sum_avx512(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    pos_com: [f64; 3],
    vel_com: [f64; 3],
) -> (f64, f64, f64) {
    let lanes = 8;
    let chunks = positions.len() / lanes * lanes;
    let mut lx = 0.0;
    let mut ly = 0.0;
    let mut lz = 0.0;

    let mut i = 0;
    while i < chunks {
        let vals = angular_momentum_sum_scalar(
            &positions[i..i + lanes],
            &velocities[i..i + lanes],
            &masses[i..i + lanes],
            pos_com,
            vel_com,
        );
        lx += vals.0;
        ly += vals.1;
        lz += vals.2;
        i += lanes;
    }
    let tail = angular_momentum_sum_scalar(
        &positions[chunks..],
        &velocities[chunks..],
        &masses[chunks..],
        pos_com,
        vel_com,
    );
    (lx + tail.0, ly + tail.1, lz + tail.2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn make_ring(n: usize, r: f64, v: f64, m: f64) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>) {
        let mut pos = Vec::new();
        let mut vel = Vec::new();
        let masses = vec![m; n];
        for i in 0..n {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            pos.push(Vec3::new(r * theta.cos(), r * theta.sin(), 0.0));
            // Velocidad tangencial para órbita circular: v = sqrt(GM/r)
            vel.push(Vec3::new(-v * theta.sin(), v * theta.cos(), 0.0));
        }
        (pos, vel, masses)
    }

    #[test]
    fn spin_ring_positive() {
        // Un anillo de partículas con velocidad circular debe tener λ > 0
        let (pos, vel, mass) = make_ring(16, 10.0, 5.0, 1e10);
        let params = SpinParams::default();
        let spin = halo_spin(&pos, &vel, &mass, &params).unwrap();
        assert!(
            spin.lambda_peebles > 0.0,
            "λ debe ser positivo: {}",
            spin.lambda_peebles
        );
        assert!(spin.lambda_bullock > 0.0, "λ' debe ser positivo");
        assert!(spin.l_mag > 0.0, "|L| debe ser positivo");
    }

    #[test]
    fn spin_static_halo_zero() {
        // Halo sin velocidades → L = 0 → λ = 0
        let pos = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let vel = vec![Vec3::new(0.0, 0.0, 0.0); 3];
        let masses = vec![1e10; 3];
        let params = SpinParams::default();
        let spin = halo_spin(&pos, &vel, &masses, &params).unwrap();
        assert!(
            spin.l_mag < 1e-10,
            "L debe ser 0 para halo estático: {}",
            spin.l_mag
        );
        assert!(
            spin.lambda_peebles < 1e-10,
            "λ debe ser 0: {}",
            spin.lambda_peebles
        );
    }

    #[test]
    fn spin_empty_returns_none() {
        let params = SpinParams::default();
        let result = halo_spin(&[], &[], &[], &params);
        assert!(result.is_none());
    }

    #[test]
    fn lambda_bullock_smaller_than_peebles() {
        // λ_Bullock = λ_Peebles / sqrt(2) < λ_Peebles
        let (pos, vel, mass) = make_ring(8, 5.0, 3.0, 1e10);
        let params = SpinParams::default();
        let spin = halo_spin(&pos, &vel, &mass, &params).unwrap();
        assert!(
            spin.lambda_bullock < spin.lambda_peebles,
            "λ' debe ser menor que λ: {} vs {}",
            spin.lambda_bullock,
            spin.lambda_peebles
        );
        let ratio = spin.lambda_peebles / spin.lambda_bullock;
        assert!(
            (ratio - std::f64::consts::SQRT_2).abs() < 1e-10,
            "ratio λ/λ' debe ser sqrt(2): {ratio}"
        );
    }

    #[test]
    fn center_of_mass_symmetric() {
        // Distribución simétrica → COM en origen
        let pos = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let masses = vec![1.0, 1.0];
        let com = center_of_mass(&pos, &masses, 2.0);
        assert!(com[0].abs() < 1e-15, "COM.x debe ser 0: {}", com[0]);
    }

    #[test]
    fn spin_angular_momentum_direction() {
        // Rotación en plano XY → L debe apuntar en +Z
        let (pos, vel, mass) = make_ring(8, 5.0, 3.0, 1e10);
        let params = SpinParams::default();
        let spin = halo_spin(&pos, &vel, &mass, &params).unwrap();
        // Lz debe dominar
        assert!(
            spin.angular_momentum[2] > 0.0,
            "Lz debe ser positivo para rotación antihoraria: {}",
            spin.angular_momentum[2]
        );
        let lz_frac = spin.angular_momentum[2].abs() / spin.l_mag;
        assert!(lz_frac > 0.99, "L debe apuntar casi en Z: {lz_frac}");
    }

    #[test]
    fn compute_halo_spins_multi() {
        let (pos1, vel1, mass1) = make_ring(8, 5.0, 3.0, 1e10);
        let (pos2, vel2, mass2) = make_ring(4, 10.0, 2.0, 1e11);
        let n1 = pos1.len();
        let n2 = pos2.len();
        let all_pos: Vec<Vec3> = pos1.into_iter().chain(pos2).collect();
        let all_vel: Vec<Vec3> = vel1.into_iter().chain(vel2).collect();
        let all_mass: Vec<f64> = mass1.into_iter().chain(mass2).collect();
        let halo_ids = vec![
            (0..n1).collect::<Vec<_>>(),
            (n1..n1 + n2).collect::<Vec<_>>(),
        ];
        let params = SpinParams::default();
        let spins = compute_halo_spins(&all_pos, &all_vel, &all_mass, &halo_ids, &params);
        assert_eq!(spins.len(), 2);
        assert!(spins[0].is_some());
        assert!(spins[1].is_some());
    }
}
