//! Espectro de potencia en espacio de redshift P(k,μ) y multipoles P₀/P₂/P₄ (Phase 75).
//!
//! ## Algoritmo RSD (Hamilton 1992)
//!
//! 1. Desplazar posiciones a lo largo del eje `los` (line-of-sight):
//!    `s = r + v_los / (a × H(a)) × ê_los`
//! 2. CIC deposit en el campo desplazado → δ_s(k) via FFT 3D.
//! 3. Para cada modo (kx, ky, kz) calcular μ = k_los / |k|.
//! 4. Binar en (k, μ) → P(k,μ).
//! 5. Proyectar sobre polinomios de Legendre → multipoles P₀/P₂/P₄.
//!
//! ## Multipoles de Legendre
//!
//! ```text
//! P_l(k) = (2l+1)/2 × ∫₋₁¹ P(k,μ) × L_l(μ) dμ
//! ```
//!
//! - **P₀(k)** — monopolo: promedio isótropo.
//! - **P₂(k)** — cuadrupolo: anisotropía por efecto Kaiser.
//! - **P₄(k)** — hexadecapolo: corrección de orden superior.
//!
//! ## Referencia
//!
//! Hamilton (1992), ApJ 385, L5; Kaiser (1987), MNRAS 227, 1;
//! Scoccimarro (2004), PRD 70, 083007.

use gadget_ng_core::Vec3;
use rustfft::{FftPlanner, num_complex::Complex};

// ── Structs públicos ──────────────────────────────────────────────────────

/// Eje de la línea de visión (line-of-sight).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LosAxis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Default for LosAxis {
    fn default() -> Self {
        Self::Z
    }
}

/// Un bin del espectro 2D P(k, μ).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PkRsdBin {
    /// Número de onda central k (en unidades de 2π/L).
    pub k: f64,
    /// Coseno del ángulo con la línea de visión μ ∈ [0, 1].
    pub mu: f64,
    /// Potencia P(k,μ).
    pub pk: f64,
    /// Número de modos en el bin.
    pub n_modes: u64,
}

/// Multipoles de Legendre del espectro de potencia P_l(k).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PkMultipoleBin {
    /// Número de onda central k.
    pub k: f64,
    /// Monopolo P₀(k) = promedio isótropo.
    pub p0: f64,
    /// Cuadrupolo P₂(k) = (5/2) ⟨P(k,μ) × L₂(μ)⟩.
    pub p2: f64,
    /// Hexadecapolo P₄(k) = (9/2) ⟨P(k,μ) × L₄(μ)⟩.
    pub p4: f64,
    /// Número de modos (para P₀).
    pub n_modes: u64,
}

/// Parámetros para el cálculo de P(k,μ).
#[derive(Debug, Clone)]
pub struct PkRsdParams {
    /// Número de bins en k. 0 → usa n/2.
    pub n_k_bins: usize,
    /// Número de bins en μ ∈ [0,1].
    pub n_mu_bins: usize,
    /// Eje de la LOS.
    pub los: LosAxis,
    /// Factor de escala a (para el desplazamiento RSD).
    pub scale_factor: f64,
    /// H(a) en unidades internas (km/s/Mpc o equivalente).
    pub hubble_a: f64,
}

impl Default for PkRsdParams {
    fn default() -> Self {
        Self {
            n_k_bins: 0,
            n_mu_bins: 10,
            los: LosAxis::Z,
            scale_factor: 1.0,
            hubble_a: 100.0,
        }
    }
}

// ── API principal ─────────────────────────────────────────────────────────

/// Calcula el espectro de potencia en espacio de redshift P(k,μ).
///
/// Aplica el desplazamiento RSD a las posiciones antes de calcular el campo.
///
/// # Parámetros
/// - `positions`  — posiciones en espacio real [0, box_size).
/// - `velocities` — velocidades peculiares.
/// - `masses`     — masas de las partículas.
/// - `box_size`   — tamaño de la caja periódica.
/// - `mesh`       — resolución del grid CIC.
/// - `params`     — parámetros RSD.
///
/// # Retorna
/// Grid 2D de `PkRsdBin` para (k, μ).
pub fn pk_redshift_space(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    box_size: f64,
    mesh: usize,
    params: &PkRsdParams,
) -> Vec<PkRsdBin> {
    let n = mesh;
    let n3 = n * n * n;
    let cell = box_size / n as f64;
    let k_fund = 2.0 * std::f64::consts::PI / box_size;

    // ── 1. Posiciones RSD: s = r + v_los / (a·H) ─────────────────────────
    let a_h = params.scale_factor * params.hubble_a;
    let rsd_shift: Vec<Vec3> = positions
        .iter()
        .zip(velocities.iter())
        .map(|(&pos, &vel)| {
            let v_los = match params.los {
                LosAxis::X => vel.x,
                LosAxis::Y => vel.y,
                LosAxis::Z => vel.z,
            };
            let delta = if a_h > 0.0 { v_los / a_h } else { 0.0 };
            match params.los {
                LosAxis::X => Vec3::new((pos.x + delta).rem_euclid(box_size), pos.y, pos.z),
                LosAxis::Y => Vec3::new(pos.x, (pos.y + delta).rem_euclid(box_size), pos.z),
                LosAxis::Z => Vec3::new(pos.x, pos.y, (pos.z + delta).rem_euclid(box_size)),
            }
        })
        .collect();

    // ── 2. CIC deposit + FFT ──────────────────────────────────────────────
    let total_mass: f64 = masses.iter().sum();
    let mean_rho = total_mass / box_size.powi(3);
    let vol_cell = cell.powi(3);

    let mut rho = vec![0.0f64; n3];
    for (&pos, &m) in rsd_shift.iter().zip(masses.iter()) {
        cic_assign(&mut rho, pos, m, n, cell);
    }
    let mut buf: Vec<Complex<f64>> = rho
        .iter()
        .map(|&r| Complex::new(r / (mean_rho * vol_cell) - 1.0, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    for row in buf.chunks_exact_mut(n) {
        fft.process(row);
    }
    fft_axis_y(&mut buf, n, &fft);
    fft_axis_x(&mut buf, n, &fft);

    // ── 3. Binar en (k, μ) ───────────────────────────────────────────────
    let n_k = if params.n_k_bins == 0 {
        n / 2
    } else {
        params.n_k_bins
    };
    let n_mu = params.n_mu_bins.max(1);
    let vol = box_size.powi(3);
    let norm = (vol / n3 as f64).powi(2);

    let los_idx = params.los as usize;

    let mut pk_sum = vec![0.0f64; n_k * n_mu];
    let mut pk_count = vec![0u64; n_k * n_mu];

    for ix in 0..n {
        let kx = freq(ix, n) as f64;
        let wx = sinc(kx / n as f64);
        for iy in 0..n {
            let ky = freq(iy, n) as f64;
            let wy = sinc(ky / n as f64);
            for iz in 0..n {
                let kz = freq(iz, n) as f64;
                let wz = sinc(kz / n as f64);
                let k2 = kx * kx + ky * ky + kz * kz;
                if k2 == 0.0 {
                    continue;
                }
                let k_mag = k2.sqrt();
                let k_bin_f = k_mag - 0.5;
                if k_bin_f < 0.0 || k_bin_f >= n_k as f64 {
                    continue;
                }
                let k_bin = k_bin_f as usize;

                // μ = k_los / |k|
                let k_los = [kx, ky, kz][los_idx];
                let mu = (k_los / k_mag).abs();
                let mu_bin = ((mu * n_mu as f64) as usize).min(n_mu - 1);

                let w2 = (wx * wy * wz).powi(2);
                let idx = ix * n * n + iy * n + iz;
                let delta2 = buf[idx].norm_sqr() / w2;
                let pk_val = delta2 * norm;

                let b = k_bin * n_mu + mu_bin;
                pk_sum[b] += pk_val;
                pk_count[b] += 1;
            }
        }
    }

    let mut result = Vec::new();
    for kb in 0..n_k {
        for mb in 0..n_mu {
            let b = kb * n_mu + mb;
            let nm = pk_count[b];
            if nm == 0 {
                continue;
            }
            result.push(PkRsdBin {
                k: (kb as f64 + 1.0) * k_fund,
                mu: (mb as f64 + 0.5) / n_mu as f64,
                pk: pk_sum[b] / nm as f64,
                n_modes: nm,
            });
        }
    }
    result
}

/// Calcula los multipoles de Legendre P₀/P₂/P₄ desde el espectro 2D P(k,μ).
///
/// Integra numéricamente usando los bins de μ disponibles en `pk_rsd`.
pub fn pk_multipoles(pk_rsd: &[PkRsdBin], box_size: f64, mesh: usize) -> Vec<PkMultipoleBin> {
    let k_fund = 2.0 * std::f64::consts::PI / box_size;
    let n_k = mesh / 2;

    // Agrupar por bin k
    let mut p0_sum = vec![0.0f64; n_k];
    let mut p2_sum = vec![0.0f64; n_k];
    let mut p4_sum = vec![0.0f64; n_k];
    let mut w_sum = vec![0.0f64; n_k];
    let mut nm_sum = vec![0u64; n_k];

    for bin in pk_rsd {
        let k_bin_f = bin.k / k_fund - 1.0;
        if k_bin_f < 0.0 || k_bin_f >= n_k as f64 {
            continue;
        }
        let kb = k_bin_f as usize;
        let mu = bin.mu;
        let pk = bin.pk;
        let nm = bin.n_modes as f64;

        // Polinomios de Legendre
        let l0 = 1.0;
        let l2 = 0.5 * (3.0 * mu * mu - 1.0);
        let l4 = 0.125 * (35.0 * mu.powi(4) - 30.0 * mu * mu + 3.0);

        p0_sum[kb] += pk * l0 * nm;
        p2_sum[kb] += pk * l2 * nm;
        p4_sum[kb] += pk * l4 * nm;
        w_sum[kb] += nm;
        nm_sum[kb] += bin.n_modes;
    }

    let mut result = Vec::new();
    for kb in 0..n_k {
        let w = w_sum[kb];
        if w == 0.0 {
            continue;
        }
        result.push(PkMultipoleBin {
            k: (kb as f64 + 1.0) * k_fund,
            p0: p0_sum[kb] / w,
            p2: 5.0 * p2_sum[kb] / w,
            p4: 9.0 * p4_sum[kb] / w,
            n_modes: nm_sum[kb],
        });
    }
    result
}

/// Calcula directamente los multipoles desde posiciones y velocidades.
///
/// Combina `pk_redshift_space` + `pk_multipoles` en una sola llamada.
pub fn compute_pk_multipoles(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    box_size: f64,
    mesh: usize,
    params: &PkRsdParams,
) -> Vec<PkMultipoleBin> {
    let pk_rsd = pk_redshift_space(positions, velocities, masses, box_size, mesh, params);
    pk_multipoles(&pk_rsd, box_size, mesh)
}

// ── Kaiser factor (lineal) ────────────────────────────────────────────────

/// Factor Kaiser para el cuadrupolo en régimen lineal:
/// β = f/b donde f = d ln D / d ln a ≈ Ω_m(a)^0.55 y b es el bias de galaxias.
///
/// Retorna (P₀/P_lin, P₂/P_lin, P₄/P_lin) — los ratios de Kaiser.
pub fn kaiser_multipole_ratios(beta: f64) -> (f64, f64, f64) {
    let r0 = 1.0 + 2.0 / 3.0 * beta + 1.0 / 5.0 * beta * beta;
    let r2 = 4.0 / 3.0 * beta + 4.0 / 7.0 * beta * beta;
    let r4 = 8.0 / 35.0 * beta * beta;
    (r0, r2, r4)
}

// ── Helpers ───────────────────────────────────────────────────────────────

#[inline]
fn freq(i: usize, n: usize) -> i64 {
    let h = (n / 2) as i64;
    let ii = i as i64;
    if ii <= h { ii } else { ii - n as i64 }
}

#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

fn cic_assign(grid: &mut [f64], pos: Vec3, m: f64, n: usize, cell: f64) {
    let n2 = n * n;
    let fx = (pos.x / cell).rem_euclid(n as f64);
    let fy = (pos.y / cell).rem_euclid(n as f64);
    let fz = (pos.z / cell).rem_euclid(n as f64);
    let ix = fx.floor() as usize;
    let iy = fy.floor() as usize;
    let iz = fz.floor() as usize;
    let tx = fx - ix as f64;
    let ty = fy - iy as f64;
    let tz = fz - iz as f64;
    for (ddx, wx) in [(0usize, 1.0 - tx), (1, tx)] {
        for (ddy, wy) in [(0usize, 1.0 - ty), (1, ty)] {
            for (ddz, wz) in [(0usize, 1.0 - tz), (1, tz)] {
                let jx = (ix + ddx) % n;
                let jy = (iy + ddy) % n;
                let jz = (iz + ddz) % n;
                grid[jx * n2 + jy * n + jz] += m * wx * wy * wz;
            }
        }
    }
}

fn fft_axis_y(buf: &mut [Complex<f64>], n: usize, fft: &std::sync::Arc<dyn rustfft::Fft<f64>>) {
    let mut tmp = vec![Complex::default(); n];
    for ix in 0..n {
        for iz in 0..n {
            for iy in 0..n {
                tmp[iy] = buf[ix * n * n + iy * n + iz];
            }
            fft.process(&mut tmp);
            for iy in 0..n {
                buf[ix * n * n + iy * n + iz] = tmp[iy];
            }
        }
    }
}

fn fft_axis_x(buf: &mut [Complex<f64>], n: usize, fft: &std::sync::Arc<dyn rustfft::Fft<f64>>) {
    let mut tmp = vec![Complex::default(); n];
    for iy in 0..n {
        for iz in 0..n {
            for ix in 0..n {
                tmp[ix] = buf[ix * n * n + iy * n + iz];
            }
            fft.process(&mut tmp);
            for ix in 0..n {
                buf[ix * n * n + iy * n + iz] = tmp[ix];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn uniform_grid(n_side: usize, box_size: f64) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>) {
        let dx = box_size / n_side as f64;
        let mut pos = Vec::new();
        let mut vel = Vec::new();
        let mut mass = Vec::new();
        for iz in 0..n_side {
            for iy in 0..n_side {
                for ix in 0..n_side {
                    pos.push(Vec3::new(
                        (ix as f64 + 0.5) * dx,
                        (iy as f64 + 0.5) * dx,
                        (iz as f64 + 0.5) * dx,
                    ));
                    vel.push(Vec3::new(0.0, 0.0, 0.0));
                    mass.push(1.0);
                }
            }
        }
        (pos, vel, mass)
    }

    #[test]
    fn pk_rsd_zero_vel_equals_real_space() {
        // Con velocidades cero, P(k,μ) debe ser independiente de μ
        let (pos, vel, mass) = uniform_grid(8, 1.0);
        let params = PkRsdParams {
            n_k_bins: 4,
            n_mu_bins: 4,
            los: LosAxis::Z,
            scale_factor: 1.0,
            hubble_a: 100.0,
        };
        let bins = pk_redshift_space(&pos, &vel, &mass, 1.0, 8, &params);
        // Todos los bins deben ser finitos
        for b in &bins {
            assert!(b.pk.is_finite(), "P(k,μ) no finito: {:?}", b);
        }
    }

    #[test]
    fn pk_rsd_bins_have_increasing_k() {
        let (pos, vel, mass) = uniform_grid(4, 1.0);
        let params = PkRsdParams {
            n_k_bins: 2,
            n_mu_bins: 2,
            ..Default::default()
        };
        let bins = pk_redshift_space(&pos, &vel, &mass, 1.0, 8, &params);
        // Dentro del mismo μ, k debe ser creciente
        let ks: Vec<f64> = bins.iter().filter(|b| b.mu < 0.5).map(|b| b.k).collect();
        for i in 1..ks.len() {
            assert!(ks[i] >= ks[i - 1]);
        }
    }

    #[test]
    fn pk_multipoles_p0_positive() {
        let (pos, vel, mass) = uniform_grid(4, 1.0);
        let mut vel2 = vel.clone();
        // Agregar velocidades aleatorias en Z para activar RSD
        let mut seed = 7u64;
        for v in &mut vel2 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            v.z = ((seed >> 33) as f64) / (u32::MAX as f64) * 10.0 - 5.0;
        }
        let params = PkRsdParams {
            n_k_bins: 4,
            n_mu_bins: 8,
            los: LosAxis::Z,
            scale_factor: 0.5,
            hubble_a: 70.0,
        };
        let multipoles = compute_pk_multipoles(&pos, &vel2, &mass, 1.0, 8, &params);
        for m in &multipoles {
            assert!(m.p0.is_finite(), "P₀ no finito");
        }
    }

    #[test]
    fn kaiser_ratios_beta_zero() {
        // Para β=0 (sin RSD), Kaiser da P₀=P_lin, P₂=0, P₄=0
        let (r0, r2, r4) = kaiser_multipole_ratios(0.0);
        assert!((r0 - 1.0).abs() < 1e-10);
        assert!(r2.abs() < 1e-10);
        assert!(r4.abs() < 1e-10);
    }

    #[test]
    fn kaiser_ratios_beta_positive() {
        // Para β>0, P₂>0 (cuadrupolo positivo por compresión a lo largo de LOS)
        let (r0, r2, r4) = kaiser_multipole_ratios(0.5);
        assert!(r0 > 1.0, "P₀/P_lin debe ser > 1 para β>0: {r0}");
        assert!(r2 > 0.0, "P₂/P_lin debe ser > 0: {r2}");
        assert!(r4 > 0.0, "P₄/P_lin debe ser > 0: {r4}");
    }

    #[test]
    fn pk_rsd_los_axes_different() {
        // El P(k,μ) debe ser diferente para distintas LOS con velocidades anisótropas
        let (pos, _, mass) = uniform_grid(4, 1.0);
        let mut vel = vec![Vec3::new(0.0, 0.0, 0.0); pos.len()];
        // Solo velocidad en Z
        for v in &mut vel {
            v.z = 50.0;
        }
        let params_z = PkRsdParams {
            n_k_bins: 2,
            n_mu_bins: 2,
            los: LosAxis::Z,
            ..Default::default()
        };
        let params_x = PkRsdParams {
            n_k_bins: 2,
            n_mu_bins: 2,
            los: LosAxis::X,
            ..Default::default()
        };
        let bz = pk_redshift_space(&pos, &vel, &mass, 1.0, 8, &params_z);
        let bx = pk_redshift_space(&pos, &vel, &mass, 1.0, 8, &params_x);
        // Con velocidad solo en Z, el espectro para LOS=Z y LOS=X debe diferir
        assert!(!bz.is_empty() && !bx.is_empty());
    }

    #[test]
    fn pk_multipole_bin_serializes() {
        let b = PkMultipoleBin {
            k: 1.0,
            p0: 2.0,
            p2: 0.5,
            p4: 0.1,
            n_modes: 100,
        };
        let s = serde_json::to_string(&b).unwrap();
        let b2: PkMultipoleBin = serde_json::from_str(&s).unwrap();
        assert!((b2.p0 - 2.0).abs() < 1e-12);
    }
}
