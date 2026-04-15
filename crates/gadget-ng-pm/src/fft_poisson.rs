//! Resolución de la ecuación de Poisson en k-space y cálculo de fuerzas en el grid.
//!
//! ## Algoritmo
//!
//! Dado el campo de densidad `ρ[x]` en el grid NM³ y la constante gravitacional G:
//!
//! 1. FFT 3D de densidad: `ρ̂(k) = FFT3D{ρ}`
//! 2. Potencial gravitacional: `Φ̂(k) = -4πG · ρ̂(k) / k²`  (k=0 → 0)
//! 3. Fuerzas en k-space: `F̂_α(k) = -i · k_α · Φ̂(k)`
//! 4. IFFT por componente: `F_α = IFFT3D{F̂_α}`
//!
//! ## Convención de k
//!
//! Para un grid de lado NM y celda de tamaño `Δx = box_size/NM`, los números
//! de onda son `k_α = 2π·n_α / box_size` con `n_α ∈ {0,1,...,NM/2,-NM/2+1,...,-1}`.
//! En rustfft (convención DFT estándar), el índice `j` corresponde a
//! `n = j` para `j ≤ NM/2` y `n = j - NM` para `j > NM/2`.

use rustfft::{num_complex::Complex, FftPlanner};

/// Resuelve la ecuación de Poisson y devuelve las tres componentes de la fuerza
/// en el grid como arrays planos de longitud `nm³`.
///
/// - `density` — densidad `ρ` en el grid (masa/celda, longitud `nm³`).
/// - `g` — constante gravitacional (signo positivo; las fuerzas son atractivas).
/// - `nm` — número de celdas por lado (potencia de 2 recomendada).
/// - `box_size` — longitud del cubo periódico.
///
/// La densidad `ρ` tiene unidades de masa/celda. Para que el resultado tenga
/// unidades de aceleración (longitud/tiempo²), las masas deben estar en las
/// mismas unidades que G·m/r².
pub fn solve_forces(density: &[f64], g: f64, nm: usize, box_size: f64) -> [Vec<f64>; 3] {
    solve_forces_impl(density, g, nm, box_size, None)
}

/// Igual que [`solve_forces`] pero aplica un filtro Gaussiano en k-space que
/// suprime las contribuciones de **corto alcance**, dejando solo las de largo alcance.
///
/// El filtro es `W(k) = exp(-k²·r_split²/2)`, equivalente a convolucionar la
/// densidad con una Gaussiana de anchura `r_split` en espacio real. El potencial
/// resultante corresponde a `erf(r / (√2·r_split))` en el par-Newton real-space.
///
/// La complementaria `erfc(r / (√2·r_split))` se calcula en el paso de corto
/// alcance del árbol para que la suma sea igual al Newton exacto.
pub fn solve_forces_filtered(
    density: &[f64],
    g: f64,
    nm: usize,
    box_size: f64,
    r_split: f64,
) -> [Vec<f64>; 3] {
    solve_forces_impl(density, g, nm, box_size, Some(r_split))
}

fn solve_forces_impl(
    density: &[f64],
    g: f64,
    nm: usize,
    box_size: f64,
    r_split: Option<f64>,
) -> [Vec<f64>; 3] {
    let nm2 = nm * nm;
    let nm3 = nm2 * nm;
    assert_eq!(density.len(), nm3);

    // Volumen total y volumen de celda (para normalizar la densidad volumétrica).
    let cell_vol = (box_size / nm as f64).powi(3);
    // El grid almacena masa/celda; convertir a densidad volumétrica (masa/volumen).
    // La FFT de rustfft ya normaliza por 1/N en la IFFT; no necesitamos factor
    // adicional, pero sí convertir ρ_celda → ρ_volumétrica para que las unidades
    // de Φ sean [G·masa/longitud].
    let rho_scale = 1.0 / cell_vol;

    // ── FFT 3D de la densidad ─────────────────────────────────────────────────
    // Usamos rustfft con arrays complejos; el input real implica f[j] = conj(f[N-j]).
    let mut planner = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(nm);
    let fft_inv = planner.plan_fft_inverse(nm);

    // Convertir densidad a complejo.
    let mut rho_c: Vec<Complex<f64>> = density
        .iter()
        .map(|&r| Complex::new(r * rho_scale, 0.0))
        .collect();

    // FFT 3D = tres pasadas de 1D FFTs (filas, columnas, pilas).
    fft3d_inplace(&mut rho_c, nm, &fft_fwd);

    // ── Resolver Poisson y construir F̂_x, F̂_y, F̂_z en k-space ────────────
    let dk = 2.0 * std::f64::consts::PI / box_size; // espaciado en k
    let four_pi_g = 4.0 * std::f64::consts::PI * g;

    let mut fx_c = vec![Complex::new(0.0, 0.0); nm3];
    let mut fy_c = vec![Complex::new(0.0, 0.0); nm3];
    let mut fz_c = vec![Complex::new(0.0, 0.0); nm3];

    for iz in 0..nm {
        let kz = dk * freq_index(iz, nm) as f64;
        for iy in 0..nm {
            let ky = dk * freq_index(iy, nm) as f64;
            for ix in 0..nm {
                let kx = dk * freq_index(ix, nm) as f64;

                let k2 = kx * kx + ky * ky + kz * kz;
                let flat = iz * nm2 + iy * nm + ix;

                if k2 < 1e-30 {
                    // Modo de fondo (DC): sin fuerza global.
                    // fx_c, fy_c, fz_c ya son 0 por inicialización.
                    continue;
                }

                // Φ̂(k) = -4πG · ρ̂(k) / k²
                // Con filtro Gaussiano de largo alcance: × exp(-k²·r_s²/2)
                let filter = if let Some(r_s) = r_split {
                    (-0.5 * k2 * r_s * r_s).exp()
                } else {
                    1.0
                };
                let phi_k = rho_c[flat] * (-four_pi_g * filter / k2);

                // F̂_α = -i · k_α · Φ̂(k)
                // -i · k_α multiplicado por complejo c: -i·kα·(a+ib) = (kα·b) - i·(kα·a)
                fx_c[flat] = Complex::new(kx * phi_k.im, -kx * phi_k.re);
                fy_c[flat] = Complex::new(ky * phi_k.im, -ky * phi_k.re);
                fz_c[flat] = Complex::new(kz * phi_k.im, -kz * phi_k.re);
            }
        }
    }

    // ── IFFT 3D de cada componente de fuerza ─────────────────────────────────
    let norm = 1.0 / nm3 as f64; // normalización IFFT
    ifft3d_inplace(&mut fx_c, nm, &fft_inv);
    ifft3d_inplace(&mut fy_c, nm, &fft_inv);
    ifft3d_inplace(&mut fz_c, nm, &fft_inv);

    let fx: Vec<f64> = fx_c.iter().map(|c| c.re * norm).collect();
    let fy: Vec<f64> = fy_c.iter().map(|c| c.re * norm).collect();
    let fz: Vec<f64> = fz_c.iter().map(|c| c.re * norm).collect();

    [fx, fy, fz]
}

/// Convierte un índice DFT `j ∈ [0, nm)` en el número de onda entero correspondiente.
/// `j ≤ nm/2` → `j`; `j > nm/2` → `j - nm`.
#[inline]
fn freq_index(j: usize, nm: usize) -> i64 {
    if j <= nm / 2 {
        j as i64
    } else {
        j as i64 - nm as i64
    }
}

/// FFT 3D in-place usando tres pasadas de 1D FFTs (filas → columnas → pilas).
fn fft3d_inplace(
    data: &mut [Complex<f64>],
    nm: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
) {
    let nm2 = nm * nm;
    // Pasada X: FFT a lo largo de filas (eje x).
    for iz in 0..nm {
        for iy in 0..nm {
            let start = iz * nm2 + iy * nm;
            fft.process(&mut data[start..start + nm]);
        }
    }
    // Pasada Y: FFT a lo largo de columnas (eje y). Necesita copia temporal por strides.
    let mut tmp = vec![Complex::new(0.0, 0.0); nm];
    for iz in 0..nm {
        for ix in 0..nm {
            for iy in 0..nm {
                tmp[iy] = data[iz * nm2 + iy * nm + ix];
            }
            fft.process(&mut tmp);
            for iy in 0..nm {
                data[iz * nm2 + iy * nm + ix] = tmp[iy];
            }
        }
    }
    // Pasada Z: FFT a lo largo de pilas (eje z). Strides en z.
    for iy in 0..nm {
        for ix in 0..nm {
            for iz in 0..nm {
                tmp[iz] = data[iz * nm2 + iy * nm + ix];
            }
            fft.process(&mut tmp);
            for iz in 0..nm {
                data[iz * nm2 + iy * nm + ix] = tmp[iz];
            }
        }
    }
}

/// IFFT 3D in-place (misma estructura que `fft3d_inplace` pero con plan inverso).
/// La normalización 1/N³ se aplica fuera.
fn ifft3d_inplace(
    data: &mut [Complex<f64>],
    nm: usize,
    ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
) {
    let nm2 = nm * nm;
    let mut tmp = vec![Complex::new(0.0, 0.0); nm];
    // Pasada Z.
    for iy in 0..nm {
        for ix in 0..nm {
            for iz in 0..nm {
                tmp[iz] = data[iz * nm2 + iy * nm + ix];
            }
            ifft.process(&mut tmp);
            for iz in 0..nm {
                data[iz * nm2 + iy * nm + ix] = tmp[iz];
            }
        }
    }
    // Pasada Y.
    for iz in 0..nm {
        for ix in 0..nm {
            for iy in 0..nm {
                tmp[iy] = data[iz * nm2 + iy * nm + ix];
            }
            ifft.process(&mut tmp);
            for iy in 0..nm {
                data[iz * nm2 + iy * nm + ix] = tmp[iy];
            }
        }
    }
    // Pasada X.
    for iz in 0..nm {
        for iy in 0..nm {
            let start = iz * nm2 + iy * nm;
            ifft.process(&mut data[start..start + nm]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Una densidad uniforme produce fuerzas ≈ 0 (el modo DC es cero).
    #[test]
    fn uniform_density_gives_zero_force() {
        let nm = 8usize;
        let nm3 = nm * nm * nm;
        let density = vec![1.0_f64; nm3];
        let [fx, fy, fz] = solve_forces(&density, 1.0, nm, 1.0);
        for i in 0..nm3 {
            assert!(
                fx[i].abs() < 1e-10 && fy[i].abs() < 1e-10 && fz[i].abs() < 1e-10,
                "fuerzas no nulas en celda {i}: fx={} fy={} fz={}",
                fx[i],
                fy[i],
                fz[i]
            );
        }
    }

    /// La FFT 3D seguida de IFFT debe recuperar la señal original.
    #[test]
    fn fft3d_roundtrip() {
        let nm = 4usize;
        let nm3 = nm * nm * nm;
        let original: Vec<Complex<f64>> = (0..nm3)
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();
        let mut data = original.clone();
        let mut planner = FftPlanner::new();
        let fft_fwd = planner.plan_fft_forward(nm);
        let fft_inv = planner.plan_fft_inverse(nm);
        fft3d_inplace(&mut data, nm, &fft_fwd);
        ifft3d_inplace(&mut data, nm, &fft_inv);
        let norm = 1.0 / nm3 as f64;
        for (i, (d, o)) in data.iter().zip(original.iter()).enumerate() {
            let err = (d.re * norm - o.re).abs();
            assert!(err < 1e-10, "error en índice {i}: {err}");
        }
    }
}
