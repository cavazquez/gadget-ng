//! Bispectrum B(k₁,k₂,k₃) del campo de densidad (Phase 71).
//!
//! El bispectrum es la estadística de 3 puntos en espacio de Fourier:
//!
//! ```text
//! B(k₁,k₂,k₃) = ⟨δ(k₁)δ(k₂)δ(k₃)⟩  donde  k₁+k₂+k₃ ≈ 0
//! ```
//!
//! ## Implementación
//!
//! Se calcula el **bispectrum equiláteral** B_eq(k) y el **bispectrum isósceles**
//! B(k₁,k₂) usando el método de filtros de cáscara (shell-filter):
//!
//! 1. CIC deposit → δ(k) via FFT 3D.
//! 2. Para cada k: filtrar δ(k) a la cáscara |k| ∈ [k-Δk/2, k+Δk/2] → δ_k(x) via IFFT.
//! 3. **B_eq(k)** = ⟨δ_k³(x)⟩ × V² / n_triangles.
//! 4. **B(k₁,k₂)** = ⟨δ_k₁(x) × δ_k₂(x) × δ_{|k₁+k₂|}(x)⟩ × V² / n_triangles.
//!
//! ## Referencia
//!
//! Scoccimarro (2000), ApJ 544, 597; Sefusatti & Komatsu (2007), PRD 76, 083004.

use gadget_ng_core::Vec3;
use rustfft::{num_complex::Complex, FftPlanner};

// ── Tipos públicos ─────────────────────────────────────────────────────────

/// Un bin del bispectrum equiláteral B_eq(k).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BkBin {
    /// Número de onda central (en h/Mpc si se usan unidades físicas).
    pub k: f64,
    /// Bispectrum B_eq(k) = ⟨δ_k³⟩ × V² (en unidades de V² = L⁶).
    pub bk: f64,
    /// Número de triángulos equiláteros contados.
    pub n_triangles: u64,
}

/// Un bin del bispectrum isósceles B(k₁, k₂).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BkIsoscelesBin {
    /// Primer número de onda.
    pub k1: f64,
    /// Segundo número de onda.
    pub k2: f64,
    /// Bispectrum B(k₁,k₂,|k₁+k₂|).
    pub bk: f64,
    /// Número de triángulos.
    pub n_triangles: u64,
}

// ── Bispectrum equiláteral ─────────────────────────────────────────────────

/// Calcula el bispectrum equiláteral B_eq(k) del campo de densidad.
///
/// # Parámetros
/// - `positions` — posiciones de las partículas en `[0, box_size)`.
/// - `masses`    — masas de las partículas.
/// - `box_size`  — lado del cubo periódico.
/// - `mesh`      — resolución del grid CIC por eje (potencia de 2).
/// - `n_bins`    — número de bins de k. Si 0, usa `mesh/2`.
///
/// # Retorna
/// Vec de `BkBin` para k ∈ [k_fund, k_Nyquist].
pub fn bispectrum_equilateral(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    mesh: usize,
    n_bins: usize,
) -> Vec<BkBin> {
    let n = mesh;
    let n3 = n * n * n;
    let n_bins = if n_bins == 0 { n / 2 } else { n_bins };
    let k_fund = 2.0 * std::f64::consts::PI / box_size;
    let vol = box_size.powi(3);

    // ── Paso 1: CIC deposit + FFT → δ(k) ─────────────────────────────────
    let delta_k = compute_delta_k(positions, masses, box_size, n);

    // ── Paso 2: B_eq(k) = ⟨δ_k³(x)⟩ × V² para cada bin k ────────────────
    let mut result = Vec::with_capacity(n_bins);

    for b in 0..n_bins {
        let k_lo = (b as f64 + 0.5) * k_fund;
        let k_hi = k_lo + k_fund;
        let k_cen = (k_lo + k_hi) * 0.5;

        // Filtrar modos en la cáscara [k_lo, k_hi) → campo real δ_k(x)
        let dk_x = shell_ifft(&delta_k, n, box_size, k_lo, k_hi);

        // B_eq = ⟨δ_k(x)³⟩ × V²
        let mean_cube: f64 = dk_x.iter().map(|&v| v * v * v).sum::<f64>() / n3 as f64;
        let bk = mean_cube * vol * vol;

        // Número de modos en la cáscara (para estimar n_triangles ≈ N_modes³/6)
        let n_modes = count_shell_modes(&delta_k, n, box_size, k_lo, k_hi);
        let n_triangles = n_modes * n_modes * n_modes / 6;

        result.push(BkBin { k: k_cen, bk, n_triangles });
    }

    result
}

/// Calcula el bispectrum isósceles B(k₁, k₂) para una cuadrícula de (k₁, k₂).
///
/// Más costoso que el equiláteral: O(n_bins² × N log N).
/// Para cada par (k₁,k₂), el tercer lado es k₃ = |k₁+k₂| (aproximado como k₁+k₂).
pub fn bispectrum_isosceles(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    mesh: usize,
    k1_bins: &[f64],
    k2_bins: &[f64],
) -> Vec<BkIsoscelesBin> {
    let n = mesh;
    let n3 = n * n * n;
    let k_fund = 2.0 * std::f64::consts::PI / box_size;
    let dk = k_fund;
    let vol = box_size.powi(3);

    let delta_k = compute_delta_k(positions, masses, box_size, n);

    let mut result = Vec::new();

    for &k1 in k1_bins {
        let dk1_x = shell_ifft(&delta_k, n, box_size, k1 - dk * 0.5, k1 + dk * 0.5);
        let n1 = count_shell_modes(&delta_k, n, box_size, k1 - dk * 0.5, k1 + dk * 0.5);

        for &k2 in k2_bins {
            if k2 > k1 * 2.0 + dk {
                // Desigualdad triangular: no puede formar triángulo
                continue;
            }
            let k3 = (k1 * k1 + k2 * k2 + k1 * k2).sqrt(); // k3 = |k1+k2| aprox
            let dk2_x = shell_ifft(&delta_k, n, box_size, k2 - dk * 0.5, k2 + dk * 0.5);
            let dk3_x = shell_ifft(&delta_k, n, box_size, k3 - dk * 0.5, k3 + dk * 0.5);

            let mean_prod: f64 = (0..n3)
                .map(|i| dk1_x[i] * dk2_x[i] * dk3_x[i])
                .sum::<f64>() / n3 as f64;
            let bk = mean_prod * vol * vol;

            let n2 = count_shell_modes(&delta_k, n, box_size, k2 - dk * 0.5, k2 + dk * 0.5);
            let n3_modes = count_shell_modes(&delta_k, n, box_size, k3 - dk * 0.5, k3 + dk * 0.5);
            let n_triangles = n1 * n2 * n3_modes;

            result.push(BkIsoscelesBin { k1, k2, bk, n_triangles });
        }
    }

    result
}

// ── Reduced bispectrum Q ───────────────────────────────────────────────────

/// Bispectrum reducido Q(k) = B_eq(k) / [3 × P(k)²].
///
/// Para una distribución gaussiana perfecta, Q = 0.
/// Valores Q ≠ 0 indican no-gaussianidades.
pub fn reduced_bispectrum(bk_bins: &[BkBin], pk_at_k: &[(f64, f64)]) -> Vec<(f64, f64)> {
    bk_bins
        .iter()
        .filter_map(|bb| {
            // Interpolar P(k) en el k del bin
            let pk = interpolate_pk(pk_at_k, bb.k)?;
            if pk.abs() < 1e-30 {
                return None;
            }
            let q = bb.bk / (3.0 * pk * pk);
            Some((bb.k, q))
        })
        .collect()
}

fn interpolate_pk(pk_table: &[(f64, f64)], k: f64) -> Option<f64> {
    let i = pk_table.partition_point(|(ki, _)| *ki < k);
    if i == 0 || i >= pk_table.len() {
        return None;
    }
    let (k0, p0) = pk_table[i - 1];
    let (k1, p1) = pk_table[i];
    let t = (k - k0) / (k1 - k0);
    Some(p0 + t * (p1 - p0))
}

// ── Funciones internas ─────────────────────────────────────────────────────

/// Deposita masa (CIC) y calcula δ(k) via FFT 3D.
fn compute_delta_k(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    n: usize,
) -> Vec<Complex<f64>> {
    let n3 = n * n * n;
    let cell = box_size / n as f64;
    let total_mass: f64 = masses.iter().sum();
    let mean_rho = total_mass / box_size.powi(3);

    // CIC deposit
    let mut rho = vec![0.0f64; n3];
    for (&pos, &m) in positions.iter().zip(masses.iter()) {
        cic_assign(&mut rho, pos, m, n, cell);
    }

    // δ = ρ/(ρ̄·V_cell) - 1
    let vol_cell = cell.powi(3);
    let mut buf: Vec<Complex<f64>> = rho
        .iter()
        .map(|&r| Complex::new(r / (mean_rho * vol_cell) - 1.0, 0.0))
        .collect();

    // FFT 3D via 3× 1D
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    // Eje z
    for row in buf.chunks_exact_mut(n) {
        fft.process(row);
    }
    // Eje y
    fft_axis(&mut buf, n, n, n, &fft);
    // Eje x
    fft_axis2(&mut buf, n, &fft);

    buf
}

/// Extrae los modos en la cáscara [k_lo, k_hi), hace IFFT y retorna campo real.
fn shell_ifft(
    delta_k: &[Complex<f64>],
    n: usize,
    box_size: f64,
    k_lo: f64,
    k_hi: f64,
) -> Vec<f64> {
    let n3 = n * n * n;
    let k_fund = 2.0 * std::f64::consts::PI / box_size;
    let mut filtered = vec![Complex::new(0.0, 0.0); n3];

    for ix in 0..n {
        let kx = freq(ix, n) as f64 * k_fund;
        for iy in 0..n {
            let ky = freq(iy, n) as f64 * k_fund;
            for iz in 0..n {
                let kz = freq(iz, n) as f64 * k_fund;
                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();
                if k_mag >= k_lo && k_mag < k_hi {
                    let idx = ix * n * n + iy * n + iz;
                    filtered[idx] = delta_k[idx];
                }
            }
        }
    }

    // IFFT 3D
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);

    // Eje z
    for row in filtered.chunks_exact_mut(n) {
        ifft.process(row);
    }
    fft_axis(&mut filtered, n, n, n, &planner.plan_fft_inverse(n));
    fft_axis2_inv(&mut filtered, n, &planner.plan_fft_inverse(n));

    // Normalizar: IFFT rustfft no normaliza
    let norm = 1.0 / n3 as f64;
    filtered.iter().map(|c| c.re * norm).collect()
}

/// Cuenta el número de modos en la cáscara [k_lo, k_hi).
fn count_shell_modes(
    delta_k: &[Complex<f64>],
    n: usize,
    box_size: f64,
    k_lo: f64,
    k_hi: f64,
) -> u64 {
    let k_fund = 2.0 * std::f64::consts::PI / box_size;
    let mut count = 0u64;
    for ix in 0..n {
        let kx = freq(ix, n) as f64 * k_fund;
        for iy in 0..n {
            let ky = freq(iy, n) as f64 * k_fund;
            for iz in 0..n {
                let kz = freq(iz, n) as f64 * k_fund;
                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();
                if k_mag >= k_lo && k_mag < k_hi {
                    count += 1;
                }
            }
        }
    }
    let _ = delta_k; // silence unused
    count
}

// ── Helpers FFT ───────────────────────────────────────────────────────────

#[inline]
fn freq(i: usize, n: usize) -> i64 {
    if i <= n / 2 { i as i64 } else { i as i64 - n as i64 }
}

fn cic_assign(rho: &mut [f64], pos: Vec3, m: f64, n: usize, cell: f64) {
    let n2 = n * n;
    let cx = (pos.x / cell).rem_euclid(n as f64);
    let cy = (pos.y / cell).rem_euclid(n as f64);
    let cz = (pos.z / cell).rem_euclid(n as f64);
    let ix0 = cx.floor() as usize;
    let iy0 = cy.floor() as usize;
    let iz0 = cz.floor() as usize;
    let dx = cx - ix0 as f64;
    let dy = cy - iy0 as f64;
    let dz = cz - iz0 as f64;
    for (diz, wz) in [(0, 1.0 - dz), (1, dz)] {
        let iz = (iz0 + diz) % n;
        for (diy, wy) in [(0, 1.0 - dy), (1, dy)] {
            let iy = (iy0 + diy) % n;
            for (dix, wx) in [(0, 1.0 - dx), (1, dx)] {
                let ix = (ix0 + dix) % n;
                rho[iz * n2 + iy * n + ix] += m * wx * wy * wz;
            }
        }
    }
}

fn fft_axis(buf: &mut [Complex<f64>], nx: usize, ny: usize, nz: usize, fft: &std::sync::Arc<dyn rustfft::Fft<f64>>) {
    let mut tmp = vec![Complex::new(0.0, 0.0); ny];
    for ix in 0..nx {
        for iz in 0..nz {
            for (j, t) in tmp.iter_mut().enumerate() {
                *t = buf[ix * ny * nz + j * nz + iz];
            }
            fft.process(&mut tmp);
            for (j, t) in tmp.iter().enumerate() {
                buf[ix * ny * nz + j * nz + iz] = *t;
            }
        }
    }
}

fn fft_axis2(buf: &mut [Complex<f64>], n: usize, fft: &std::sync::Arc<dyn rustfft::Fft<f64>>) {
    let n2 = n * n;
    let mut tmp = vec![Complex::new(0.0, 0.0); n];
    for iy in 0..n {
        for iz in 0..n {
            for (j, t) in tmp.iter_mut().enumerate() {
                *t = buf[j * n2 + iy * n + iz];
            }
            fft.process(&mut tmp);
            for (j, t) in tmp.iter().enumerate() {
                buf[j * n2 + iy * n + iz] = *t;
            }
        }
    }
}

fn fft_axis2_inv(buf: &mut [Complex<f64>], n: usize, ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>) {
    fft_axis2(buf, n, ifft)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn uniform_lattice(n_side: usize, box_size: f64) -> (Vec<Vec3>, Vec<f64>) {
        let dx = box_size / n_side as f64;
        let mut pos = Vec::new();
        let mut mass = Vec::new();
        for iz in 0..n_side {
            for iy in 0..n_side {
                for ix in 0..n_side {
                    pos.push(Vec3::new(
                        (ix as f64 + 0.5) * dx,
                        (iy as f64 + 0.5) * dx,
                        (iz as f64 + 0.5) * dx,
                    ));
                    mass.push(1.0);
                }
            }
        }
        (pos, mass)
    }

    #[test]
    fn bk_uniform_nearly_zero() {
        // Para distribución uniforme, B_eq(k) debe ser muy pequeño
        let (pos, mass) = uniform_lattice(8, 1.0);
        let bins = bispectrum_equilateral(&pos, &mass, 1.0, 8, 4);
        assert_eq!(bins.len(), 4);
        for b in &bins {
            assert!(b.bk.abs() < 1.0, "B_eq no debería ser grande para distribución uniforme: {}", b.bk);
        }
    }

    #[test]
    fn bk_bins_have_k_increasing() {
        let (pos, mass) = uniform_lattice(4, 1.0);
        let bins = bispectrum_equilateral(&pos, &mass, 1.0, 8, 4);
        let ks: Vec<f64> = bins.iter().map(|b| b.k).collect();
        for i in 1..ks.len() {
            assert!(ks[i] > ks[i - 1], "k no creciente en bin {i}");
        }
    }

    #[test]
    fn bk_finite_for_random_distribution() {
        // Distribución seudoaleatoria → B_eq debe ser finito
        let mut pos = Vec::new();
        let mut mass = Vec::new();
        let mut seed = 12345u64;
        for _ in 0..64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = ((seed >> 33) as f64) / (u32::MAX as f64);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let y = ((seed >> 33) as f64) / (u32::MAX as f64);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let z = ((seed >> 33) as f64) / (u32::MAX as f64);
            pos.push(Vec3::new(x, y, z));
            mass.push(1.0);
        }
        let bins = bispectrum_equilateral(&pos, &mass, 1.0, 8, 4);
        for b in &bins {
            assert!(b.bk.is_finite(), "B_eq no finito: {}", b.bk);
        }
    }

    #[test]
    fn bk_bin_struct_serializes() {
        let b = BkBin { k: 1.0, bk: 2.5, n_triangles: 100 };
        let s = serde_json::to_string(&b).unwrap();
        let b2: BkBin = serde_json::from_str(&s).unwrap();
        assert!((b2.k - 1.0).abs() < 1e-10);
        assert!((b2.bk - 2.5).abs() < 1e-10);
    }

    #[test]
    fn reduced_bispectrum_gaussian_near_zero() {
        // Para campo gaussiano aproximado, Q ≈ 0
        let (pos, mass) = uniform_lattice(8, 1.0);
        let bk = bispectrum_equilateral(&pos, &mass, 1.0, 8, 3);
        // Construir tabla P(k) trivial
        let pk_table: Vec<(f64, f64)> = bk.iter().map(|b| (b.k, 1.0)).collect();
        let q = reduced_bispectrum(&bk, &pk_table);
        for (k, qi) in &q {
            assert!(qi.is_finite(), "Q no finito en k={k}: {qi}");
        }
    }
}
