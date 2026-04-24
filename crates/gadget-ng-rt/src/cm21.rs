//! Estadísticas de la línea de 21cm del hidrógeno neutro (Phase 94 + FFT real).
//!
//! Calcula la temperatura de brillo diferencial δT_b por partícula y el
//! power spectrum P(k)₂₁cm usando deposición CIC trilineal + FFT 3D real (rustfft).
//!
//! ## Física
//!
//! La temperatura de brillo diferencial frente al CMB es:
//! $$\delta T_b \approx 27 x_{HI}(1+\delta)\left(\frac{1+z}{10}\right)^{1/2} \text{ mK}$$
//!
//! donde x_HI = 1 - x_HII es la fracción neutra, (1+δ) = ρ/ρ̄ es la sobredensidad.
//!
//! ## Power spectrum
//!
//! 1. Deposición CIC trilineal del campo δT_b en malla N³
//! 2. Sustracción de la media (campo de contraste)
//! 3. FFT 3D via 3 pasadas de FFT 1D complejas (separabilidad de la DFT)
//! 4. Binning esférico de |δ̃(k)|² → P(k) [mK² (Mpc/h)³]
//! 5. Varianza dimensional Δ²(k) = k³ P(k) / (2π²) [mK²]

use crate::ChemState;
use gadget_ng_core::Particle;
use rustfft::{num_complex::Complex, FftPlanner};

/// Parámetros para el cálculo de estadísticas 21cm.
#[derive(Debug, Clone)]
pub struct Cm21Params {
    /// Temperatura de spin T_S [K] (default: 1000 K >> T_CMB).
    pub t_s_kelvin: f64,
    /// Frecuencia de la línea 21cm [MHz] (default: 1420.406 MHz).
    pub nu_21cm_mhz: f64,
}

impl Default for Cm21Params {
    fn default() -> Self {
        Self {
            t_s_kelvin: 1000.0,
            nu_21cm_mhz: 1_420.405_751_768,
        }
    }
}

/// Un bin en el power spectrum P(k)₂₁cm.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cm21PkBin {
    /// Número de onda central [h Mpc⁻¹].
    pub k: f64,
    /// Varianza dimensional Δ²₂₁(k) = k³ P(k) / (2π²) [mK²].
    pub delta_sq: f64,
}

/// Salida completa del análisis 21cm en un snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cm21Output {
    /// Redshift del snapshot.
    pub z: f64,
    /// Temperatura de brillo media <δT_b> [mK].
    pub delta_tb_mean: f64,
    /// Dispersión σ(δT_b) [mK].
    pub delta_tb_sigma: f64,
    /// Power spectrum dimensional Δ²₂₁(k) [mK²].
    pub pk_21cm: Vec<Cm21PkBin>,
}

/// Calcula la temperatura de brillo diferencial δT_b para una partícula de gas [mK].
///
/// Usa la fórmula estándar:
/// δT_b ≈ 27 x_HI (1+δ) √((1+z)/10) mK
///
/// # Argumentos
/// - `x_hii`: fracción ionizada (entre 0 y 1)
/// - `overdensity`: (1 + δ) = ρ / ρ̄, sobredensidad local
/// - `z`: redshift
/// - `_params`: parámetros 21cm (reservado para extensiones)
pub fn brightness_temperature(x_hii: f64, overdensity: f64, z: f64, _params: &Cm21Params) -> f64 {
    let x_hi = (1.0 - x_hii).max(0.0);
    27.0 * x_hi * overdensity * ((1.0 + z) / 10.0_f64).sqrt()
}

/// Calcula el campo de temperatura de brillo δT_b para cada partícula de gas.
///
/// Devuelve un vector con δT_b [mK] por partícula.
pub fn compute_delta_tb_field(
    particles: &[Particle],
    chem_states: &[ChemState],
    z: f64,
    params: &Cm21Params,
) -> Vec<f64> {
    if particles.is_empty() || chem_states.is_empty() {
        return Vec::new();
    }
    let n = particles.len().min(chem_states.len());

    // Densidad media para calcular sobredensidad
    let total_mass: f64 = particles[..n].iter().map(|p| p.mass).sum();
    let total_vol: f64 = particles[..n]
        .iter()
        .map(|p| {
            let h = p.smoothing_length.max(1e-30);
            h * h * h
        })
        .sum();
    let rho_mean = if total_vol > 0.0 {
        total_mass / total_vol
    } else {
        1.0
    };

    particles[..n]
        .iter()
        .zip(chem_states[..n].iter())
        .map(|(p, chem)| {
            let h = p.smoothing_length.max(1e-30);
            let rho_local = p.mass / (h * h * h);
            let overdensity = (rho_local / rho_mean).max(0.0);
            brightness_temperature(chem.x_hii, overdensity, z, params)
        })
        .collect()
}

/// Calcula estadísticas 21cm completas: <δT_b>, σ, y P(k)₂₁cm.
///
/// El power spectrum se calcula proyectando el campo δT_b en una malla CIC
/// trilineal y aplicando FFT 3D real via `rustfft`.
///
/// # Argumentos
/// - `particles`: partículas de gas
/// - `chem_states`: estados de química por partícula
/// - `box_size`: tamaño de la caja en Mpc/h
/// - `z`: redshift
/// - `n_mesh`: resolución del grid CIC (por lado). Debe ser ≥ 4.
/// - `n_pk_bins`: número de bins en P(k)
/// - `params`: parámetros 21cm
pub fn compute_cm21_output(
    particles: &[Particle],
    chem_states: &[ChemState],
    box_size: f64,
    z: f64,
    n_mesh: usize,
    n_pk_bins: usize,
    params: &Cm21Params,
) -> Cm21Output {
    let delta_tb = compute_delta_tb_field(particles, chem_states, z, params);

    if delta_tb.is_empty() {
        return Cm21Output { z, delta_tb_mean: 0.0, delta_tb_sigma: 0.0, pk_21cm: Vec::new() };
    }

    let mean = delta_tb.iter().sum::<f64>() / delta_tb.len() as f64;
    let variance = delta_tb.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / delta_tb.len() as f64;
    let sigma = variance.sqrt();

    let pk_21cm = if n_mesh >= 4 && n_pk_bins >= 1 {
        compute_pk_21cm_fft(&delta_tb, particles, box_size, n_mesh, n_pk_bins)
    } else {
        Vec::new()
    };

    Cm21Output { z, delta_tb_mean: mean, delta_tb_sigma: sigma, pk_21cm }
}

/// Deposita el campo δT_b en una malla 3D usando CIC trilineal periódico.
///
/// Cada partícula contribuye a las 8 celdas vecinas con peso proporcional
/// a la fracción volumétrica dentro de cada celda.
fn deposit_cic(
    delta_tb: &[f64],
    particles: &[Particle],
    n_mesh: usize,
    dx: f64,
) -> Vec<f64> {
    let n3 = n_mesh * n_mesh * n_mesh;
    let mut grid = vec![0.0_f64; n3];
    let n = particles.len().min(delta_tb.len());

    for i in 0..n {
        let p = &particles[i];
        let dtb = delta_tb[i];

        // Posición normalizada en unidades de celda
        let xc = p.position.x / dx - 0.5;
        let yc = p.position.y / dx - 0.5;
        let zc = p.position.z / dx - 0.5;

        let ix0 = xc.floor() as isize;
        let iy0 = yc.floor() as isize;
        let iz0 = zc.floor() as isize;

        let tx = xc - ix0 as f64; // peso fraccionario [0,1)
        let ty = yc - iy0 as f64;
        let tz = zc - iz0 as f64;

        // 8 vértices CIC con periodicidad
        let nm = n_mesh as isize;
        for (dx_i, wx) in [(0, 1.0 - tx), (1, tx)] {
            for (dy_i, wy) in [(0, 1.0 - ty), (1, ty)] {
                for (dz_i, wz) in [(0, 1.0 - tz), (1, tz)] {
                    let ix = ((ix0 + dx_i).rem_euclid(nm)) as usize;
                    let iy = ((iy0 + dy_i).rem_euclid(nm)) as usize;
                    let iz = ((iz0 + dz_i).rem_euclid(nm)) as usize;
                    grid[ix * n_mesh * n_mesh + iy * n_mesh + iz] += dtb * wx * wy * wz;
                }
            }
        }
    }
    grid
}

/// Aplica FFT 3D a un grid real usando 3 pasadas de FFT 1D complejas (separabilidad DFT).
///
/// Retorna el grid en espacio de Fourier como Vec<Complex<f64>> de longitud N³.
/// El layout es [ix][iy][iz] con stride n_mesh² / n_mesh / 1.
fn fft3d_real(grid_real: &[f64], n_mesh: usize) -> Vec<Complex<f64>> {
    let n3 = n_mesh * n_mesh * n_mesh;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_mesh);

    // Convertir a complejo
    let mut data: Vec<Complex<f64>> = grid_real
        .iter()
        .map(|&v| Complex::new(v, 0.0))
        .collect();

    // FFT sobre el eje Z (dim 2, stride 1, n_mesh elementos consecutivos)
    for ix in 0..n_mesh {
        for iy in 0..n_mesh {
            let base = ix * n_mesh * n_mesh + iy * n_mesh;
            let slice = &mut data[base..base + n_mesh];
            fft.process(slice);
        }
    }

    // FFT sobre el eje Y (dim 1, stride n_mesh)
    let mut scratch_y = vec![Complex::new(0.0, 0.0); n_mesh];
    for ix in 0..n_mesh {
        for iz in 0..n_mesh {
            for iy in 0..n_mesh {
                scratch_y[iy] = data[ix * n_mesh * n_mesh + iy * n_mesh + iz];
            }
            fft.process(&mut scratch_y);
            for iy in 0..n_mesh {
                data[ix * n_mesh * n_mesh + iy * n_mesh + iz] = scratch_y[iy];
            }
        }
    }

    // FFT sobre el eje X (dim 0, stride n_mesh²)
    let mut scratch_x = vec![Complex::new(0.0, 0.0); n_mesh];
    for iy in 0..n_mesh {
        for iz in 0..n_mesh {
            for ix in 0..n_mesh {
                scratch_x[ix] = data[ix * n_mesh * n_mesh + iy * n_mesh + iz];
            }
            fft.process(&mut scratch_x);
            for ix in 0..n_mesh {
                data[ix * n_mesh * n_mesh + iy * n_mesh + iz] = scratch_x[ix];
            }
        }
    }

    let _ = n3;
    data
}

/// Calcula P(k)₂₁cm via CIC trilineal + FFT 3D real + binning esférico.
///
/// Pipeline:
/// 1. CIC trilineal → malla δT_b(x) de N³ celdas
/// 2. Substracción de la media → campo de contraste
/// 3. FFT 3D (3 × FFT 1D complejas) → δ̃T_b(k)
/// 4. |δ̃T_b(k)|² × V / N³ → estimador de P(k) [mK² (Mpc/h)³]
/// 5. Binning esférico → Δ²(k) = k³ P(k) / (2π²) [mK²]
fn compute_pk_21cm_fft(
    delta_tb: &[f64],
    particles: &[Particle],
    box_size: f64,
    n_mesh: usize,
    n_pk_bins: usize,
) -> Vec<Cm21PkBin> {
    let n3 = n_mesh * n_mesh * n_mesh;
    let dx = box_size / n_mesh as f64;

    // 1. Deposición CIC
    let mut grid = deposit_cic(delta_tb, particles, n_mesh, dx);

    // 2. Substraer media (campo de contraste δ = T/T̄ - 1 en el grid)
    let mean_grid = grid.iter().sum::<f64>() / n3 as f64;
    for v in grid.iter_mut() {
        *v -= mean_grid;
    }

    // 3. FFT 3D
    let fft_grid = fft3d_real(&grid, n_mesh);

    // 4. Binning esférico de |δ̃|² en k
    let k_fund = 2.0 * std::f64::consts::PI / box_size; // k fundamental [h/Mpc]
    let k_nyq = std::f64::consts::PI / dx;               // k Nyquist
    let dk = (k_nyq - k_fund) / n_pk_bins as f64;
    let vol = box_size.powi(3);
    let norm = vol / (n3 * n3) as f64; // factor de normalización de la DFT discreta

    let mut pk_bins: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); n_pk_bins];

    for ix in 0..n_mesh {
        let kx = freq_to_k(ix, n_mesh, k_fund);
        for iy in 0..n_mesh {
            let ky = freq_to_k(iy, n_mesh, k_fund);
            for iz in 0..n_mesh {
                let kz = freq_to_k(iz, n_mesh, k_fund);
                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();
                if k_mag < k_fund * 0.5 {
                    continue; // excluir modo cero
                }
                let bin_idx = ((k_mag - k_fund) / dk) as usize;
                if bin_idx < n_pk_bins {
                    let c = fft_grid[ix * n_mesh * n_mesh + iy * n_mesh + iz];
                    let pk = (c.re * c.re + c.im * c.im) * norm;
                    pk_bins[bin_idx].0 += k_mag;
                    pk_bins[bin_idx].1 += pk;
                    pk_bins[bin_idx].2 += 1;
                }
            }
        }
    }

    // 5. Calcular Δ²(k) = k³ P(k) / (2π²)
    let two_pi_sq = 2.0 * std::f64::consts::PI * std::f64::consts::PI;
    pk_bins
        .into_iter()
        .filter(|(_, _, count)| *count > 0)
        .map(|(k_sum, pk_sum, count)| {
            let k_mean = k_sum / count as f64;
            let pk_mean = pk_sum / count as f64;
            let delta_sq = k_mean.powi(3) * pk_mean / two_pi_sq;
            Cm21PkBin { k: k_mean, delta_sq }
        })
        .collect()
}

/// Convierte índice FFT a frecuencia física k [h/Mpc] (convenio centrado).
#[inline]
fn freq_to_k(idx: usize, n: usize, k_fund: f64) -> f64 {
    let i = idx as isize;
    let n = n as isize;
    if i <= n / 2 { i as f64 * k_fund } else { (i - n) as f64 * k_fund }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, Vec3};

    fn make_particle(x: f64, y: f64, z_pos: f64, mass: f64, h: f64) -> Particle {
        let mut p = Particle::new(0, mass, Vec3::new(x, y, z_pos), Vec3::zero());
        p.smoothing_length = h;
        p.internal_energy = 100.0;
        p
    }

    fn make_chem(x_hii: f64) -> ChemState {
        ChemState {
            x_hi: 1.0 - x_hii,
            x_hii,
            x_hei: 1.0,
            x_heii: 0.0,
            x_heiii: 0.0,
            x_e: x_hii,
        }
    }

    #[test]
    fn delta_tb_zero_at_full_ionization() {
        let params = Cm21Params::default();
        let dtb = brightness_temperature(1.0, 1.5, 8.0, &params);
        assert!(dtb.abs() < 1e-12, "δT_b debe ser 0 con x_HII=1, got {}", dtb);
    }

    #[test]
    fn delta_tb_positive_before_reionization() {
        let params = Cm21Params::default();
        let dtb = brightness_temperature(0.0, 1.0, 9.0, &params);
        assert!(dtb > 0.0, "δT_b debe ser positiva antes de reionización, got {}", dtb);
        let expected = 27.0 * ((1.0 + 9.0) / 10.0_f64).sqrt();
        assert!(
            (dtb - expected).abs() < 0.1,
            "δT_b = {} ≠ {} esperado",
            dtb,
            expected
        );
    }

    #[test]
    fn delta_tb_scales_with_overdensity() {
        let params = Cm21Params::default();
        let dtb1 = brightness_temperature(0.0, 1.0, 8.0, &params);
        let dtb2 = brightness_temperature(0.0, 2.0, 8.0, &params);
        assert!(
            (dtb2 - 2.0 * dtb1).abs() < 1e-10,
            "δT_b debe escalar con overdensity: {} vs {}",
            dtb2,
            dtb1
        );
    }

    #[test]
    fn compute_cm21_output_basic() {
        let params = Cm21Params::default();
        let box_size = 10.0;
        let n = 8;
        let mut particles = Vec::new();
        let mut chem = Vec::new();
        for i in 0..n {
            let x = (i as f64 + 0.5) * box_size / n as f64;
            particles.push(make_particle(x, x, x, 1.0, 0.5));
            chem.push(make_chem(if i < n / 2 { 0.1 } else { 0.9 }));
        }

        let out = compute_cm21_output(&particles, &chem, box_size, 8.0, 4, 3, &params);
        assert_eq!(out.z, 8.0);
        assert!(out.delta_tb_mean >= 0.0);
        assert!(out.delta_tb_sigma >= 0.0);
    }

    #[test]
    fn compute_cm21_output_empty() {
        let params = Cm21Params::default();
        let out = compute_cm21_output(&[], &[], 10.0, 8.0, 4, 3, &params);
        assert_eq!(out.delta_tb_mean, 0.0);
        assert!(out.pk_21cm.is_empty());
    }

    /// La FFT 3D de un campo constante (solo modo k=0) no debe producir
    /// bins de P(k) en la parte de señal (todos los bins en k>0 deben ser ~0).
    #[test]
    fn pk_fft_uniform_field_zero_signal() {
        let params = Cm21Params::default();
        let box_size = 10.0;
        let n_mesh = 8;
        let n_part = n_mesh * n_mesh * n_mesh;

        // Partículas en retícula regular, todas con el mismo δT_b (gas completamente neutro)
        let dx = box_size / n_mesh as f64;
        let mut particles = Vec::new();
        let mut chem = Vec::new();
        for ix in 0..n_mesh {
            for iy in 0..n_mesh {
                for iz in 0..n_mesh {
                    let x = (ix as f64 + 0.5) * dx;
                    let y = (iy as f64 + 0.5) * dx;
                    let z = (iz as f64 + 0.5) * dx;
                    particles.push(make_particle(x, y, z, 1.0, 0.5 * dx));
                    chem.push(make_chem(0.0)); // x_HII=0 → δT_b uniforme
                }
            }
        }

        let out = compute_cm21_output(&particles, &chem, box_size, 9.0, n_mesh, 4, &params);

        // Campo uniforme → δ̃(k) = 0 para k≠0 → todos los bins de P(k) deben ser ≈ 0
        for bin in &out.pk_21cm {
            assert!(
                bin.delta_sq.abs() < 1e-6,
                "campo uniforme debe tener Δ²(k)≈0 en k={:.3}, got {:.3e}",
                bin.k,
                bin.delta_sq
            );
        }
        assert_eq!(out.pk_21cm.len(), 4, "deben haber 4 bins k");
        assert!(out.delta_tb_mean > 0.0, "señal media debe ser positiva antes de reionización");
        let _ = n_part;
    }

    /// El P(k) con una onda sinusoidal inyectada debe tener pico en el bin correcto.
    #[test]
    fn pk_fft_sinusoidal_signal_peak() {
        let params = Cm21Params::default();
        let box_size = 10.0;
        let n_mesh = 8;
        let dx = box_size / n_mesh as f64;
        let k_fund = 2.0 * std::f64::consts::PI / box_size;

        // Retícula regular con δT_b modulado por cos(k_fund x) en la dirección X
        let mut particles = Vec::new();
        let mut chem_base = Vec::new();
        for ix in 0..n_mesh {
            for iy in 0..n_mesh {
                for iz in 0..n_mesh {
                    let x = (ix as f64 + 0.5) * dx;
                    let y = (iy as f64 + 0.5) * dx;
                    let z = (iz as f64 + 0.5) * dx;
                    let p = make_particle(x, y, z, 1.0, 0.4 * dx);
                    particles.push(p);
                    // Modulamos x_hii con una sinusoidal para crear variación en δT_b
                    let x_hii = 0.5 + 0.3 * (k_fund * x).cos();
                    chem_base.push(make_chem(x_hii.clamp(0.0, 1.0)));
                }
            }
        }

        let out = compute_cm21_output(&particles, &chem_base, box_size, 9.0, n_mesh, 4, &params);

        // Debe haber señal en P(k) (campo no uniforme → FFT no nula)
        let has_signal = out.pk_21cm.iter().any(|b| b.delta_sq > 1e-10);
        assert!(has_signal, "campo modulado debe producir señal en P(k)");

        // El bin de menor k debe tener la mayor potencia (modo fundamental)
        let pk_vals: Vec<f64> = out.pk_21cm.iter().map(|b| b.delta_sq).collect();
        if pk_vals.len() >= 2 {
            assert!(
                pk_vals[0] >= pk_vals[pk_vals.len() - 1],
                "el modo fundamental debe dominar: Δ²(k_min)={:.3e} vs Δ²(k_max)={:.3e}",
                pk_vals[0], pk_vals[pk_vals.len() - 1]
            );
        }
    }

    /// CIC trilineal debe conservar la masa total del campo.
    #[test]
    fn cic_deposit_conserves_total() {
        let box_size = 10.0;
        let n_mesh = 8;
        let dx = box_size / n_mesh as f64;
        let n = 64;

        let mut particles = Vec::new();
        let mut dtb = Vec::new();
        for i in 0..n {
            let x = (i as f64 + 0.5) * box_size / n as f64;
            particles.push(make_particle(x % box_size, (x * 1.3) % box_size, (x * 0.7) % box_size, 1.0, 0.3));
            dtb.push(1.0_f64); // δT_b = 1 para todos
        }

        let grid = deposit_cic(&dtb, &particles, n_mesh, dx);
        let total: f64 = grid.iter().sum();
        assert!(
            (total - n as f64).abs() < 1e-10,
            "CIC debe conservar la suma total: {} ≠ {}",
            total, n
        );
    }

    /// fft3d_real de un impulso unitario debe tener módulo constante = 1.
    #[test]
    fn fft3d_impulse_flat_spectrum() {
        let n = 4;
        let n3 = n * n * n;
        let mut grid = vec![0.0_f64; n3];
        grid[0] = 1.0; // impulso en el origen

        let fft_out = fft3d_real(&grid, n);

        // Para un impulso δ[0,0,0], la DFT debe ser plana: |F(k)| = 1 para todo k
        for c in &fft_out {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-10,
                "DFT de impulso debe ser plana, got |F| = {}",
                mag
            );
        }
    }
}
