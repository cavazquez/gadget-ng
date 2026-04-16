//! Power spectrum P(k) del campo de densidad de masa.
//!
//! Algoritmo:
//! 1. Asignación de masa al grid con Cloud-In-Cell (CIC).
//! 2. FFT 3D real → compleja con `rustfft`.
//! 3. Deconvolución del kernel CIC en k-space: W(k) = sinc²(kx/2) sinc²(ky/2) sinc²(kz/2).
//! 4. Estimador de potencia: P(k) = |δ(k)|² × V / N_modes, binado en anillos esféricos.
//!
//! La salida es un vector de `PkBin`, ordenado por k creciente.

use gadget_ng_core::Vec3;
use rustfft::{FftPlanner, num_complex::Complex};

/// Un bin de potencia.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PkBin {
    /// Número de onda central del bin (en unidades de 2π/L).
    pub k: f64,
    /// Potencia media P(k) en el bin (mismas unidades que L³ / N²).
    pub pk: f64,
    /// Número de modos en el bin.
    pub n_modes: u64,
}

/// Calcula el power spectrum P(k) de un conjunto de partículas.
///
/// # Parámetros
/// - `positions`: posiciones de las partículas en [0, box_size).
/// - `masses`: masas de las partículas.
/// - `box_size`: tamaño de la caja.
/// - `mesh`: número de celdas por eje del grid (potencia de 2 recomendada).
///
/// # Retorna
/// Vector de `PkBin` ordenado por k creciente, para k ∈ [k_fund, k_Nyquist].
pub fn power_spectrum(
    positions: &[Vec3],
    masses:    &[f64],
    box_size:  f64,
    mesh:      usize,
) -> Vec<PkBin> {
    let n = mesh;
    let n3 = n * n * n;
    let cell = box_size / n as f64;

    // ── 1. Asignación CIC ─────────────────────────────────────────────────────
    let mut rho = vec![0.0f64; n3];
    let total_mass: f64 = masses.iter().sum();
    let mean_rho   = total_mass / (box_size * box_size * box_size);

    for (&pos, &m) in positions.iter().zip(masses.iter()) {
        cic_assign(&mut rho, pos, m, n, cell);
    }

    // Convertir a sobre-densidad δ = ρ/ρ̄ − 1.
    let vol_cell = cell * cell * cell;
    for v in &mut rho {
        *v = *v / (mean_rho * vol_cell) - 1.0;
    }

    // ── 2. FFT 3D (real → complejo como 3× FFT 1D para usar rustfft) ─────────
    // Convertir a complejo.
    let mut buf: Vec<Complex<f64>> = rho.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    // FFT por filas (eje z), luego eje y, luego eje x.
    // Eje z: n×n filas de longitud n.
    for row in buf.chunks_exact_mut(n) {
        fft.process(row);
    }
    // Eje y: n×n filas de longitud n (stride n).
    fft_axis(&mut buf, n, n, n, &fft);
    // Eje x: n² filas de longitud n (stride n²).
    fft_axis2(&mut buf, n, &fft);

    // ── 3. Deconvolución CIC y cálculo de P(k) ────────────────────────────────
    let n_nyq = n / 2;
    let k_fund = 2.0 * std::f64::consts::PI / box_size; // k fundamental
    let n_bins = n_nyq;
    let mut pk_sum   = vec![0.0f64; n_bins];
    let mut n_modes  = vec![0u64;   n_bins];

    let vol = box_size * box_size * box_size;
    let norm = (vol / (n3 as f64)).powi(2);

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
                let bin_f = k_mag - 0.5;
                if bin_f < 0.0 || bin_f >= n_bins as f64 {
                    continue;
                }
                let bin = bin_f as usize;

                let w2 = (wx * wy * wz).powi(2);
                let idx = ix * n * n + iy * n + iz;
                let delta2 = buf[idx].norm_sqr() / w2;
                pk_sum[bin]  += delta2 * norm;
                n_modes[bin] += 1;
            }
        }
    }

    pk_sum.iter()
        .zip(n_modes.iter())
        .enumerate()
        .filter(|(_, (_, &nm))| nm > 0)
        .map(|(bin, (&ps, &nm))| PkBin {
            k:       (bin as f64 + 1.0) * k_fund,
            pk:      ps / nm as f64,
            n_modes: nm,
        })
        .collect()
}

/// Frecuencia discreta en [-n/2, n/2) para el índice `i`.
#[inline]
fn freq(i: usize, n: usize) -> i32 {
    let h = (n / 2) as i32;
    let ii = i as i32;
    if ii <= h { ii } else { ii - n as i32 }
}

/// sinc(x) = sin(πx) / (πx) para la deconvolución CIC (ventana de orden 2).
#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// CIC assignment: distribuye masa `m` de la partícula en `pos` a 8 celdas vecinas.
fn cic_assign(grid: &mut [f64], pos: Vec3, m: f64, n: usize, cell: f64) {
    let fx = pos.x / cell;
    let fy = pos.y / cell;
    let fz = pos.z / cell;
    let ix = fx.floor() as isize;
    let iy = fy.floor() as isize;
    let iz = fz.floor() as isize;
    let tx = fx - fx.floor();
    let ty = fy - fy.floor();
    let tz = fz - fz.floor();
    let ni = n as isize;
    for (ddx, wx) in [(0, 1.0 - tx), (1, tx)] {
        for (ddy, wy) in [(0, 1.0 - ty), (1, ty)] {
            for (ddz, wz) in [(0, 1.0 - tz), (1, tz)] {
                let jx = ((ix + ddx).rem_euclid(ni)) as usize;
                let jy = ((iy + ddy).rem_euclid(ni)) as usize;
                let jz = ((iz + ddz).rem_euclid(ni)) as usize;
                grid[jx * n * n + jy * n + jz] += m * wx * wy * wz;
            }
        }
    }
}

/// FFT sobre el eje Y: para cada (ix, iz), hace FFT de los n elementos a lo largo de y.
fn fft_axis(
    buf: &mut [Complex<f64>],
    nx:  usize,
    ny:  usize,
    nz:  usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
) {
    let mut tmp = vec![Complex::default(); ny];
    for ix in 0..nx {
        for iz in 0..nz {
            for iy in 0..ny {
                tmp[iy] = buf[ix * ny * nz + iy * nz + iz];
            }
            fft.process(&mut tmp);
            for iy in 0..ny {
                buf[ix * ny * nz + iy * nz + iz] = tmp[iy];
            }
        }
    }
}

/// FFT sobre el eje X: para cada (iy, iz), hace FFT de los n elementos a lo largo de x.
fn fft_axis2(
    buf: &mut [Complex<f64>],
    n:   usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
) {
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

    /// Un campo uniforme (todos iguales) debe dar P(k) ≈ 0 para k > 0.
    #[test]
    fn pk_uniform_field_is_near_zero() {
        // Lattice cúbica uniforme → δ = 0 en todas partes → P(k) ≈ 0.
        let n = 4usize;
        let box_size = 1.0f64;
        let cell = box_size / n as f64;
        let mut pos = Vec::new();
        let mut mass = Vec::new();
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    pos.push(Vec3::new(
                        (ix as f64 + 0.5) * cell,
                        (iy as f64 + 0.5) * cell,
                        (iz as f64 + 0.5) * cell,
                    ));
                    mass.push(1.0f64);
                }
            }
        }
        let bins = power_spectrum(&pos, &mass, box_size, n);
        for b in &bins {
            assert!(b.pk < 1e-20, "P(k={}) = {} debe ser ~0", b.k, b.pk);
        }
    }

    /// Con N=1 partícula, |δ(k)| = constante → P(k) no debe ser NaN/inf.
    #[test]
    fn pk_single_particle_no_nan() {
        let pos  = vec![Vec3::new(0.5, 0.5, 0.5)];
        let mass = vec![1.0f64];
        let bins = power_spectrum(&pos, &mass, 1.0, 8);
        for b in &bins {
            assert!(b.pk.is_finite(), "P(k={}) no debe ser NaN/inf", b.k);
        }
    }
}
