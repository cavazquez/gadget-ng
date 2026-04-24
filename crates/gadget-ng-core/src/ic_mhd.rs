//! Condiciones iniciales para campo magnético primordial (Phase 161 / V3).
//!
//! ## Física
//!
//! Un campo magnético primordial comoving se genera con espectro de potencias:
//!
//! ```text
//! P_B(k) ∝ k^n_B,   n_B típico: -2.9 (nearly scale-invariant)
//! ```
//!
//! El módulo implementa dos modos:
//!
//! 1. **Campo uniforme**: `b0` en dirección z, sin variación espacial. Útil para
//!    tests de ondas de Alfvén y flux-freeze.
//!
//! 2. **Campo aleatorio Gaussiano**: amplitudes en espacio de k con fase aleatoria,
//!    transformada inversa a espacio real. Satisface `∇·B = 0` por construcción
//!    (usando solo la parte transversal del campo).
//!
//! ## Normalización
//!
//! La presión magnética comoving es `P_B = |B|²/(8π)` (unidades Gaussianas) o
//! `P_B = |B|²/2` si el código usa `μ₀ = 1` (unidades internas).
//!
//! El parámetro β = P_gas / P_mag debe ser >> 1 en las ICs cosmológicas (el campo
//! no debe dominar sobre la presión térmica en el universo temprano).
//! Para B₀ comoving ~ 1 nGauss a z=50, β ~ 10⁶.
//!
//! ## Hermeticidad
//!
//! Para el campo aleatorio, se impone simetría Hermitiana en espacio-k y se usa
//! solo la componente transversal para garantizar `∇·B = 0`.
//!
//! ## Unidades
//!
//! El valor `b0` está en las unidades internas del código. Para convertir desde
//! nGauss comoving: `B_int = B_nG * unidades_B_factor`.

use rustfft::{num_complex::Complex, FftPlanner};

use crate::{particle::Particle, vec3::Vec3};

// ── Generador LCG simple (reproducible, sin dependencias) ─────────────────────

fn lcg_u64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f64) / (u32::MAX as f64)
}

/// Genera dos valores normales N(0,1) con Box-Muller.
fn lcg_normal2(state: &mut u64) -> (f64, f64) {
    let u1 = lcg_u64(state).max(1e-300);
    let u2 = lcg_u64(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

// ─────────────────────────────────────────────────────────────────────────────
// API pública
// ─────────────────────────────────────────────────────────────────────────────

/// Asigna un campo magnético uniforme `B = (0, 0, b0)` a todas las partículas.
///
/// Es el modo más simple para tests de ondas de Alfvén: el campo no tiene
/// variación espacial, lo que permite comparar directamente con la solución
/// analítica (velocidad de Alfvén `v_A = b0/√ρ` en unidades con `μ₀=1`).
///
/// # Parámetros
/// - `particles` — slice mutable de partículas (gas o DM)
/// - `b0` — amplitud en unidades internas del código
pub fn uniform_bfield_ic(particles: &mut [Particle], b0: f64) {
    for p in particles.iter_mut() {
        p.b_field = Vec3::new(0.0, 0.0, b0);
    }
}

/// Asigna campo magnético inicial con espectro de potencias B(k) ∝ k^`spectral_index`.
///
/// El campo resultante es solenoidal (`∇·B = 0`) por construcción: se genera en
/// espacio de Fourier usando solo la componente transversal y luego se transforma
/// de vuelta al espacio real mediante IFFT.
///
/// Para tests cosmológicos, `spectral_index = -2.9` reproduce el espectro
/// "nearly scale-invariant" estándar.
///
/// # Parámetros
/// - `particles` — debe ser un grid regular 1D o 3D ordenado por posición X
/// - `b0` — amplitud RMS del campo en unidades internas
/// - `spectral_index` — índice espectral `n_B` de la potencia (`P_B ∝ k^n_B`)
/// - `seed` — semilla para reproducibilidad
///
/// # Limitación
///
/// La implementación actual usa una aproximación 1D a lo largo de la dirección X
/// para el campo `B_y`. Esta es suficiente para el test V3-T5 (β_plasma).
/// Una implementación 3D completa requiere FFT 3D y se puede agregar en el futuro.
pub fn primordial_bfield_ic(
    particles: &mut [Particle],
    b0: f64,
    spectral_index: f64,
    seed: u64,
) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    let mut rng = seed;

    // Generar amplitudes espectrales B(k) ∝ k^(n_B/2) con fase aleatoria.
    // Para la componente B_y como función de la posición x.
    let mut by_field = vec![0.0_f64; n];

    for ki in 1..=(n / 2) {
        let k = ki as f64;
        // Amplitud espectral: σ_k ∝ k^(n_B/2)
        let sigma = b0 * k.powf(spectral_index / 2.0);
        let (re, im) = lcg_normal2(&mut rng);
        let amp_re = sigma * re;
        let amp_im = sigma * im;

        // Transformada inversa discreta (DFT inversa parcial)
        for (xi, by) in by_field.iter_mut().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * ki as f64 * xi as f64 / n as f64;
            *by += 2.0 * (amp_re * phase.cos() - amp_im * phase.sin()) / n as f64;
        }
    }

    // Normalizar al RMS pedido
    let rms = (by_field.iter().map(|b| b * b).sum::<f64>() / n as f64).sqrt();
    let scale = if rms > 0.0 { b0 / rms } else { 1.0 };

    for (i, p) in particles.iter_mut().enumerate() {
        p.b_field = Vec3::new(0.0, by_field[i] * scale, 0.0);
    }
}

/// Genera un campo magnético primordial **3D completo y solenoidal** (`∇·B = 0`).
///
/// ## Método
///
/// El campo se construye íntegramente en espacio de Fourier:
///
/// 1. Para cada modo `k = (kx, ky, kz)` con `|k| > 0` se genera un vector complejo
///    aleatorio Gaussiano con amplitud `σ_k ∝ |k|^(n_B/2)` (espectro B(k)∝k^n_B).
/// 2. El vector se proyecta sobre el plano perpendicular a `k` (proyección transversal
///    `P_⊥ = I - k̂⊗k̂`), lo que garantiza `k·B_k = 0` y por tanto `∇·B = 0` al
///    transformar al espacio real.
/// 3. Se impone simetría Hermitiana `B_k(-k) = B_k(k)*` para que el campo sea
///    puramente real tras la IFFT.
/// 4. IFFT 3D con `rustfft` (una IFFT 1D por cada eje, in-place) sobre cada
///    componente `(Bx, By, Bz)` por separado.
/// 5. Se normaliza el campo al RMS pedido `b0`.
///
/// ## Parámetros
///
/// - `particles` — debe estar ordenado como grilla cúbica regular con índice
///   `p = ix + iy*n + iz*n²` (`ix` varía más rápido, orden C por planos).
///   Se requiere `n_side³ == particles.len()`.
/// - `n_side` — número de partículas por lado de la grilla.
/// - `box_size` — tamaño de la caja (sólo afecta a la normalización de k).
/// - `b0` — amplitud RMS objetivo en unidades internas.
/// - `spectral_index` — índice espectral `n_B` (`P_B ∝ k^n_B`; típico: -2.9).
/// - `seed` — semilla LCG para reproducibilidad.
///
/// ## Panics
///
/// Panics si `particles.len() != n_side³`.
pub fn primordial_bfield_ic_3d(
    particles: &mut [Particle],
    n_side: usize,
    box_size: f64,
    b0: f64,
    spectral_index: f64,
    seed: u64,
) {
    let n3 = n_side * n_side * n_side;
    assert_eq!(
        particles.len(),
        n3,
        "primordial_bfield_ic_3d: particles.len()={} != n_side³={}",
        particles.len(),
        n3
    );
    if n_side == 0 || b0 == 0.0 {
        return;
    }

    let n = n_side;
    let pi2 = 2.0 * std::f64::consts::PI;

    // ── 1. Allocar buffers complejos para los tres componentes del campo ──────
    // Layout: [ix + iy*n + iz*n²] — ix varía más rápido (mismo que las partículas).
    let mut bx_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let mut by_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let mut bz_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

    let mut rng = seed;

    // ── 2. Generar amplitudes en espacio de k y proyectar ⊥ a k ─────────────
    // Iteramos solo sobre el hemisferio kz ≥ 0 para luego imponer Hermitiana.
    for iz in 0..n {
        let kz_int: i64 = if iz <= n / 2 { iz as i64 } else { iz as i64 - n as i64 };
        let kz = kz_int as f64 * pi2 / box_size;

        for iy in 0..n {
            let ky_int: i64 = if iy <= n / 2 { iy as i64 } else { iy as i64 - n as i64 };
            let ky = ky_int as f64 * pi2 / box_size;

            for ix in 0..n {
                let kx_int: i64 = if ix <= n / 2 { ix as i64 } else { ix as i64 - n as i64 };
                let kx = kx_int as f64 * pi2 / box_size;

                let k2 = kx * kx + ky * ky + kz * kz;
                if k2 < 1e-300 {
                    continue; // modo cero → campo medio nulo
                }
                let k_mag = k2.sqrt();

                // Para la proyección transversal usamos el k "efectivo" del operador de
                // diferencias centrales: k̃_α = sin(2π·kα_int/N) · N/L
                // Esto garantiza que la divergencia DISCRETA (diferencias centrales) sea
                // exactamente cero, lo que permite validar ∇·B=0 numéricamente.
                let ktx = (kx_int as f64 * pi2 / n as f64).sin() * n as f64 / box_size;
                let kty = (ky_int as f64 * pi2 / n as f64).sin() * n as f64 / box_size;
                let ktz = (kz_int as f64 * pi2 / n as f64).sin() * n as f64 / box_size;
                let kt2 = ktx * ktx + kty * kty + ktz * ktz;

                // Solo llenar el hemisferio kz > 0, o kz=0 con ky > 0,
                // o kz=0,ky=0 con kx > 0 (para evitar duplicar el modo y su conjugado).
                let is_positive_half = kz_int > 0
                    || (kz_int == 0 && ky_int > 0)
                    || (kz_int == 0 && ky_int == 0 && kx_int > 0);
                if !is_positive_half {
                    continue;
                }

                // Amplitud espectral: σ_k ∝ k^(n_B/2)
                let sigma = k_mag.powf(spectral_index / 2.0);

                // Generar vector complejo Gaussiano para (Bx, By, Bz)
                let (ax_re, ax_im) = lcg_normal2(&mut rng);
                let (ay_re, ay_im) = lcg_normal2(&mut rng);
                let (az_re, az_im) = lcg_normal2(&mut rng);

                let mut bx_re = sigma * ax_re;
                let mut bx_im = sigma * ax_im;
                let mut by_re = sigma * ay_re;
                let mut by_im = sigma * ay_im;
                let mut bz_re = sigma * az_re;
                let mut bz_im = sigma * az_im;

                // Proyección transversal usando k̃ (diferencias centrales discretas):
                // B_k -= (B_k · k̃_hat) · k̃_hat, con k̃_hat = k̃ / |k̃|.
                // Si |k̃| ≈ 0 (modo Nyquist: sin(π) = 0), no proyectamos.
                if kt2 > 1e-300 {
                    let kt_inv = 1.0 / kt2.sqrt();
                    let ktx_n = ktx * kt_inv;
                    let kty_n = kty * kt_inv;
                    let ktz_n = ktz * kt_inv;

                    let dot_re = bx_re * ktx_n + by_re * kty_n + bz_re * ktz_n;
                    let dot_im = bx_im * ktx_n + by_im * kty_n + bz_im * ktz_n;

                    bx_re -= dot_re * ktx_n;
                    bx_im -= dot_im * ktx_n;
                    by_re -= dot_re * kty_n;
                    by_im -= dot_im * kty_n;
                    bz_re -= dot_re * ktz_n;
                    bz_im -= dot_im * ktz_n;
                }

                let idx = ix + iy * n + iz * n * n;
                bx_k[idx] = Complex::new(bx_re, bx_im);
                by_k[idx] = Complex::new(by_re, by_im);
                bz_k[idx] = Complex::new(bz_re, bz_im);

                // Simetría Hermitiana: modo conjugado en -k.
                // El índice de -k_x en el array DFT de longitud n es simplemente (n - ix) % n.
                let ix_neg = (n - ix) % n;
                let iy_neg = (n - iy) % n;
                let iz_neg = (n - iz) % n;
                let idx_neg = ix_neg + iy_neg * n + iz_neg * n * n;
                bx_k[idx_neg] = Complex::new(bx_re, -bx_im);
                by_k[idx_neg] = Complex::new(by_re, -by_im);
                bz_k[idx_neg] = Complex::new(bz_re, -bz_im);
            }
        }
    }

    // ── 3. IFFT 3D in-place usando rustfft ────────────────────────────────────
    // Estrategia: 3 pases de IFFT 1D (eje Z, Y, X) en-lugar sobre el buffer lineal.
    // Esto es equivalente a una IFFT 3D separable.
    let inv_n3 = 1.0 / n3 as f64;
    for buf in [&mut bx_k, &mut by_k, &mut bz_k] {
        ifft_3d_inplace(buf, n);
        // Normalizar (rustfft IFFT no normaliza)
        for c in buf.iter_mut() {
            *c *= inv_n3;
        }
    }

    // ── 4. Asignar campo y normalizar al RMS b0 ───────────────────────────────
    // El campo real es la parte real del resultado (la imaginaria es O(eps) por
    // la simetría Hermitiana; se descarta).
    let rms2: f64 = (0..n3)
        .map(|i| bx_k[i].re.powi(2) + by_k[i].re.powi(2) + bz_k[i].re.powi(2))
        .sum::<f64>()
        / n3 as f64;
    let rms = rms2.sqrt();
    let scale = if rms > 1e-300 { b0 / rms } else { 1.0 };

    for (i, p) in particles.iter_mut().enumerate() {
        p.b_field = Vec3::new(
            bx_k[i].re * scale,
            by_k[i].re * scale,
            bz_k[i].re * scale,
        );
    }
}

/// Aplica IFFT 3D separable in-place sobre un buffer lineal de tamaño `n³`.
///
/// Realiza tres pases de IFFT 1D:
/// 1. A lo largo del eje X (stride 1, cada línea de n elementos).
/// 2. A lo largo del eje Y (stride n).
/// 3. A lo largo del eje Z (stride n²).
///
/// El resultado es equivalente a una IFFT 3D completa (separabilidad de la DFT).
/// La normalización `1/n³` debe aplicarse por el llamador.
fn ifft_3d_inplace(buf: &mut Vec<Complex<f64>>, n: usize) {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);

    let n2 = n * n;
    let n3 = n2 * n;

    // Pase 1: IFFT a lo largo del eje X (líneas contiguas de longitud n)
    for iz in 0..n {
        for iy in 0..n {
            let base = iy * n + iz * n2;
            ifft.process(&mut buf[base..base + n]);
        }
    }

    // Pase 2: IFFT a lo largo del eje Y (stride n, longitud n)
    let mut tmp = vec![Complex::new(0.0, 0.0); n];
    for iz in 0..n {
        for ix in 0..n {
            for iy in 0..n {
                tmp[iy] = buf[ix + iy * n + iz * n2];
            }
            ifft.process(&mut tmp);
            for iy in 0..n {
                buf[ix + iy * n + iz * n2] = tmp[iy];
            }
        }
    }

    // Pase 3: IFFT a lo largo del eje Z (stride n², longitud n)
    for iy in 0..n {
        for ix in 0..n {
            for iz in 0..n {
                tmp[iz] = buf[ix + iy * n + iz * n2];
            }
            ifft.process(&mut tmp);
            for iz in 0..n {
                buf[ix + iy * n + iz * n2] = tmp[iz];
            }
        }
    }
    let _ = n3; // evitar warning
}

/// Calcula el parámetro β_plasma medio = P_gas / P_mag.
///
/// - `P_gas = (γ-1) · u · ρ` donde `ρ = masa/vol` con vol estimado como
///   `(smoothing_length)³` para SPH, o `1/N` para distribución uniforme.
/// - `P_mag = |B|² / 2` (unidades internas con `μ₀ = 1`).
///
/// Devuelve `f64::INFINITY` si no hay campo magnético (`|B| = 0` en todas las
/// partículas), lo que se interpreta como "beta infinito" (sin campo).
///
/// # Parámetros
/// - `particles` — slice de partículas (debe incluir partículas de gas con `u > 0`)
/// - `gamma` — índice adiabático (típico: 5/3)
pub fn check_plasma_beta(particles: &[Particle], gamma: f64) -> f64 {
    let gas: Vec<&Particle> = particles.iter().filter(|p| p.internal_energy > 0.0).collect();
    if gas.is_empty() {
        return f64::INFINITY;
    }

    let mut sum_beta = 0.0;
    let mut n_counted = 0usize;

    for p in &gas {
        let b2 = p.b_field.dot(p.b_field);
        if b2 < 1e-300 {
            continue;
        }
        // Estimación de densidad: masa / volumen de la celda de suavizado
        let h = p.smoothing_length;
        let rho = if h > 0.0 {
            p.mass / (h * h * h)
        } else {
            // Fallback: asumir densidad unitaria
            1.0
        };
        let p_gas = (gamma - 1.0) * p.internal_energy * rho;
        let p_mag = b2 / 2.0;
        if p_mag > 0.0 {
            sum_beta += p_gas / p_mag;
            n_counted += 1;
        }
    }

    if n_counted == 0 {
        return f64::INFINITY;
    }
    sum_beta / n_counted as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests unitarios
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::ParticleType;
    #[allow(unused_imports)]
    use super::primordial_bfield_ic_3d;

    fn gas_particle(id: usize, x: f64, u: f64, h: f64) -> Particle {
        let mut p = Particle::new_gas(id, 1.0, Vec3::new(x, 0.0, 0.0), Vec3::zero(), u, h);
        p.ptype = ParticleType::Gas;
        p
    }

    #[test]
    fn uniform_bfield_sets_bz() {
        let mut ps: Vec<Particle> = (0..8)
            .map(|i| gas_particle(i, i as f64, 1.0, 0.5))
            .collect();
        uniform_bfield_ic(&mut ps, 2.0);
        for p in &ps {
            assert!((p.b_field.z - 2.0).abs() < 1e-14);
            assert_eq!(p.b_field.x, 0.0);
            assert_eq!(p.b_field.y, 0.0);
        }
    }

    #[test]
    fn primordial_bfield_rms_matches_b0() {
        let n = 64;
        let b0 = 0.5;
        let mut ps: Vec<Particle> = (0..n)
            .map(|i| gas_particle(i, i as f64 / n as f64, 1.0, 0.5))
            .collect();
        primordial_bfield_ic(&mut ps, b0, -2.9, 42);
        let rms = (ps.iter().map(|p| p.b_field.y.powi(2)).sum::<f64>() / n as f64).sqrt();
        assert!((rms - b0).abs() / b0 < 0.02, "RMS={rms:.4} b0={b0}");
    }

    /// El RMS del campo 3D debe coincidir con `b0` al 2%.
    #[test]
    fn primordial_bfield_3d_rms_matches_b0() {
        let n = 8_usize; // 8³ = 512 partículas
        let b0 = 1.5_f64;
        let box_size = 1.0_f64;
        let mut ps: Vec<Particle> = (0..n * n * n)
            .map(|gid| {
                let ix = gid % n;
                let iy = (gid / n) % n;
                let iz = gid / (n * n);
                let x = (ix as f64 + 0.5) / n as f64 * box_size;
                let y = (iy as f64 + 0.5) / n as f64 * box_size;
                let z = (iz as f64 + 0.5) / n as f64 * box_size;
                let mut p = gas_particle(gid, x, 1.0, 0.5);
                p.position = Vec3::new(x, y, z);
                p
            })
            .collect();
        primordial_bfield_ic_3d(&mut ps, n, box_size, b0, -2.9, 42);
        let n3 = n * n * n;
        let rms = (ps.iter().map(|p| p.b_field.dot(p.b_field)).sum::<f64>() / n3 as f64).sqrt();
        let err = (rms - b0).abs() / b0;
        assert!(err < 0.02, "RMS 3D = {rms:.6} b0 = {b0} err = {err:.3e}");
    }

    /// La divergencia numérica `∇·B` debe ser compatible con cero (< 1e-10) en una
    /// grilla uniforme, verificada con diferencias finitas de primer orden.
    #[test]
    fn primordial_bfield_3d_divergence_free() {
        let n = 8_usize;
        let box_size = 1.0_f64;
        let b0 = 1.0_f64;
        let mut ps: Vec<Particle> = (0..n * n * n)
            .map(|gid| {
                let ix = gid % n;
                let iy = (gid / n) % n;
                let iz = gid / (n * n);
                let x = (ix as f64 + 0.5) / n as f64 * box_size;
                let y = (iy as f64 + 0.5) / n as f64 * box_size;
                let z = (iz as f64 + 0.5) / n as f64 * box_size;
                let mut p = gas_particle(gid, x, 1.0, 0.5);
                p.position = Vec3::new(x, y, z);
                p
            })
            .collect();
        primordial_bfield_ic_3d(&mut ps, n, box_size, b0, -2.9, 7);

        // ∇·B ≈ (Bx[i+1]-Bx[i-1])/(2dx) + ... (diferencias centrales periódicas)
        let dx = box_size / n as f64;
        let idx = |ix: usize, iy: usize, iz: usize| ix + iy * n + iz * n * n;
        let mut max_div = 0.0_f64;

        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let ix_p = (ix + 1) % n;
                    let ix_m = (ix + n - 1) % n;
                    let iy_p = (iy + 1) % n;
                    let iy_m = (iy + n - 1) % n;
                    let iz_p = (iz + 1) % n;
                    let iz_m = (iz + n - 1) % n;

                    let dbx_dx = (ps[idx(ix_p, iy, iz)].b_field.x
                        - ps[idx(ix_m, iy, iz)].b_field.x)
                        / (2.0 * dx);
                    let dby_dy = (ps[idx(ix, iy_p, iz)].b_field.y
                        - ps[idx(ix, iy_m, iz)].b_field.y)
                        / (2.0 * dx);
                    let dbz_dz = (ps[idx(ix, iy, iz_p)].b_field.z
                        - ps[idx(ix, iy, iz_m)].b_field.z)
                        / (2.0 * dx);

                    let div = (dbx_dx + dby_dy + dbz_dz).abs();
                    if div > max_div {
                        max_div = div;
                    }
                }
            }
        }

        // La divergencia numérica debe ser mucho menor que b0/dx (amplitud / escala)
        let tol = b0 / dx * 1e-10;
        println!("∇·B máximo = {max_div:.4e}, tolerancia = {tol:.4e}");
        assert!(
            max_div < tol,
            "Campo no solenoidal: max |∇·B| = {max_div:.4e} > {tol:.4e}"
        );
    }

    #[test]
    fn check_plasma_beta_returns_infinity_without_field() {
        let mut ps: Vec<Particle> = (0..4)
            .map(|i| gas_particle(i, i as f64, 2.0, 0.5))
            .collect();
        // Sin campo magnético
        let beta = check_plasma_beta(&ps, 5.0 / 3.0);
        assert!(beta.is_infinite(), "beta={beta}");

        // Con campo: beta finito
        uniform_bfield_ic(&mut ps, 1.0);
        let beta2 = check_plasma_beta(&ps, 5.0 / 3.0);
        assert!(beta2.is_finite(), "beta2={beta2}");
        assert!(beta2 > 0.0);
    }
}
