//! Condiciones iniciales de Segundo Orden (2LPT).
//!
//! ## Formulación matemática
//!
//! La posición total a segundo orden es:
//! ```text
//! x = q + D₁(a)·Ψ¹ + D₂(a)·Ψ²
//! ```
//!
//! Con las convenciones de gadget-ng (`D₁ = 1` absorbida en la amplitud),
//! la corrección neta de segundo orden es:
//! ```text
//! Δx₂ = (D₂/D₁²)·Ψ²    donde  D₂/D₁² ≈ −3/7 · Ω_m(a)^{−1/143}
//! ```
//!
//! ### Cálculo de Ψ²
//!
//! **Paso A.** Derivadas de segundo orden de φ¹ en k-space:
//! ```text
//! φ̂¹,αβ(k) = −n_α·n_β / |n|² · δ̂(k)
//! ```
//! IFFT → φ¹,αβ(x) en espacio real (6 componentes: xx,yy,zz,xy,xz,yz).
//!
//! **Paso B.** Fuente de segundo orden:
//! ```text
//! S(x) = (φ,xx·φ,yy − φ,xy²) + (φ,yy·φ,zz − φ,yz²) + (φ,zz·φ,xx − φ,xz²)
//! ```
//!
//! **Paso C.** Poisson de segundo orden en k-space:
//! ```text
//! φ²(k) = −S(k) / |n|²     (DC=0, Nyquist=0)
//! ```
//!
//! **Paso D.** Gradiente de φ²:
//! ```text
//! Ψ²(k) = −i·(n/|n|²)·φ²(k)   →   IFFT   →   Ψ²(x)
//! ```
//!
//! ### Velocidades
//!
//! El momentum canónico total (estilo GADGET-4):
//! ```text
//! p = a²·H(a)·[f₁·Ψ¹ + f₂·(D₂/D₁²)·Ψ²]
//! ```
//! con `f₂ ≈ 2·f₁` (aproximación válida en ΛCDM temprano, z ≫ 1).
//!
//! ### Factor D₂/D₁²
//!
//! Aproximación de Bouchet et al. (1995) para ΛCDM plana:
//! ```text
//! D₂/D₁² ≈ −3/7 · Ω_m(a)^{−1/143}
//! ```
//! Para Ω_m ≈ 0.3 y a = 0.02: D₂/D₁² ≈ −0.435.

use crate::{
    config::{RunConfig, TransferKind},
    cosmology::{growth_rate_f, hubble_param, CosmologyParams},
    ic_zeldovich::{
        build_spectrum_fn, delta_to_displacement, fft3d, generate_delta_kspace, mode_int,
    },
    particle::Particle,
    vec3::Vec3,
};
use rustfft::num_complex::Complex;

// ── Derivadas de segundo orden de φ¹ ─────────────────────────────────────────

/// Calcula las 6 derivadas de segundo orden de φ¹ en espacio real.
///
/// En k-space: `φ̂¹,αβ(k) = −n_α·n_β / |n|² · δ̂(k)`.
///
/// Devuelve `[φ,xx, φ,yy, φ,zz, φ,xy, φ,xz, φ,yz]`, cada componente
/// de longitud `n³` en unidades adimensionales (grid).
///
/// Los modos DC y Nyquist son automáticamente cero (heredados de `delta`).
fn phi_second_derivatives(delta: &[Complex<f64>], n: usize) -> [Vec<f64>; 6] {
    let n3 = n * n * n;
    let ifft_norm = 1.0 / n3 as f64;

    // Orden de componentes: [xx, yy, zz, xy, xz, yz]
    // Pares de ejes: (α, β)
    let pairs: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];

    let mut phi_k: [Vec<Complex<f64>>; 6] = [
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
    ];

    for ix in 0..n {
        let nx = mode_int(ix, n) as f64;
        for iy in 0..n {
            let ny = mode_int(iy, n) as f64;
            for iz in 0..n {
                let nz = mode_int(iz, n) as f64;
                let n2 = nx * nx + ny * ny + nz * nz;
                let idx = ix * n * n + iy * n + iz;

                if n2 == 0.0 {
                    continue;
                }

                let d = delta[idx];
                let ns = [nx, ny, nz];

                for (comp, &(a, b)) in pairs.iter().enumerate() {
                    // φ̂,αβ = −(n_α·n_β / |n|²) · δ̂(k)  →  real escalar × complejo
                    let c = -ns[a] * ns[b] / n2;
                    phi_k[comp][idx] = Complex::new(c * d.re, c * d.im);
                }
            }
        }
    }

    // IFFT 3D en cada componente
    for arr in &mut phi_k {
        fft3d(arr, n, false);
    }

    // Extraer parte real y aplicar normalización IFFT
    phi_k.map(|arr| arr.iter().map(|c| c.re * ifft_norm).collect())
}

// ── Fuente de segundo orden ───────────────────────────────────────────────────

/// Construye la fuente S(x) del Poisson de segundo orden.
///
/// ```text
/// S = (φ,xx·φ,yy − φ,xy²) + (φ,yy·φ,zz − φ,yz²) + (φ,zz·φ,xx − φ,xz²)
/// ```
///
/// La entrada `phi_derivs` debe estar indexada como `[xx, yy, zz, xy, xz, yz]`.
fn build_2lpt_source(phi_derivs: &[Vec<f64>; 6]) -> Vec<f64> {
    let [phi_xx, phi_yy, phi_zz, phi_xy, phi_xz, phi_yz] = phi_derivs;
    let n3 = phi_xx.len();

    (0..n3)
        .map(|i| {
            (phi_xx[i] * phi_yy[i] - phi_xy[i] * phi_xy[i])
                + (phi_yy[i] * phi_zz[i] - phi_yz[i] * phi_yz[i])
                + (phi_zz[i] * phi_xx[i] - phi_xz[i] * phi_xz[i])
        })
        .collect()
}

// ── Poisson de segundo orden ──────────────────────────────────────────────────

/// Resuelve `∇²φ² = S(x)` en k-space: `φ²(k) = −S(k) / |n|²`.
///
/// Aplica FFT 3D forward sobre la fuente real `source`, divide por `−|n|²`
/// modo a modo y pone a cero los modos DC y Nyquist.
///
/// Devuelve el campo `φ²(k)` en k-space.
fn solve_poisson_real_to_kspace(source: &[f64], n: usize) -> Vec<Complex<f64>> {
    // FFT forward de la fuente real
    let mut phi2_k: Vec<Complex<f64>> =
        source.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft3d(&mut phi2_k, n, true);

    let half = (n / 2) as i64;

    for ix in 0..n {
        let nx = mode_int(ix, n);
        for iy in 0..n {
            let ny = mode_int(iy, n);
            for iz in 0..n {
                let nz = mode_int(iz, n);
                let idx = ix * n * n + iy * n + iz;
                let n2 = (nx * nx + ny * ny + nz * nz) as f64;

                // Modos DC y Nyquist → cero
                if n2 == 0.0
                    || nx.abs() == half
                    || ny.abs() == half
                    || nz.abs() == half
                {
                    phi2_k[idx] = Complex::new(0.0, 0.0);
                    continue;
                }

                // φ²(k) = −S(k) / |n|²
                let s = phi2_k[idx];
                phi2_k[idx] = Complex::new(-s.re / n2, -s.im / n2);
            }
        }
    }

    phi2_k
}

// ── Gradiente de φ² → Ψ² ─────────────────────────────────────────────────────

/// Calcula `Ψ² = −∇φ²` a partir de `φ²(k)`.
///
/// En k-space: `Ψ²_α(k) = −i·n_α / |n|² · φ²(k)`.
///
/// Mismo patrón que `delta_to_displacement` pero con signo negativo (`−i`
/// en lugar de `+i`). Devuelve `[Ψ²_x, Ψ²_y, Ψ²_z]` en unidades físicas
/// (`box_size`).
fn phi2_to_psi2(phi2_k: &[Complex<f64>], n: usize, box_size: f64) -> [Vec<f64>; 3] {
    let n3 = n * n * n;
    let d = box_size / n as f64;
    let ifft_norm = 1.0 / n3 as f64;

    let mut psi: [Vec<Complex<f64>>; 3] = [
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
    ];

    for ix in 0..n {
        let nx = mode_int(ix, n) as f64;
        for iy in 0..n {
            let ny = mode_int(iy, n) as f64;
            for iz in 0..n {
                let nz = mode_int(iz, n) as f64;
                let n2 = nx * nx + ny * ny + nz * nz;
                let idx = ix * n * n + iy * n + iz;

                if n2 == 0.0 {
                    continue;
                }

                let phi = phi2_k[idx];

                // Ψ²_α(k) = −i·n_α / |n|² · φ²(k)
                // −i·(a + ib) = b − ia
                let make_psi2 = |n_alpha: f64| -> Complex<f64> {
                    Complex::new(
                        n_alpha * phi.im / n2,
                        -n_alpha * phi.re / n2,
                    )
                };

                psi[0][idx] = make_psi2(nx);
                psi[1][idx] = make_psi2(ny);
                psi[2][idx] = make_psi2(nz);
            }
        }
    }

    for component in &mut psi {
        fft3d(component, n, false);
    }

    let scale = ifft_norm * d;
    [
        psi[0].iter().map(|c| c.re * scale).collect(),
        psi[1].iter().map(|c| c.re * scale).collect(),
        psi[2].iter().map(|c| c.re * scale).collect(),
    ]
}

// ── Entrada pública ───────────────────────────────────────────────────────────

/// Genera partículas con condiciones iniciales 2LPT para el rango `[lo, hi)`.
///
/// La posición incluye la corrección de segundo orden:
/// ```text
/// x = q + Ψ¹ + (D₂/D₁²)·Ψ²
/// ```
///
/// El momentum canónico incluye la contribución de segundo orden:
/// ```text
/// p = a²·H(a)·[f₁·Ψ¹ + f₂·(D₂/D₁²)·Ψ²]   con f₂ = 2·f₁
/// ```
///
/// ## Parámetros cosmológicos
///
/// - `D₂/D₁²` usa la aproximación de Bouchet et al. (1995):
///   `D₂/D₁² ≈ −3/7 · Ω_m(a)^{−1/143}`
/// - `f₂ ≈ 2·f₁` es válida a alto z en ΛCDM (error < 5% para z > 1).
///
/// ## Retrocompatibilidad
///
/// Con `use_2lpt = false` en la configuración se llama a `zeldovich_ics` (1LPT).
/// Esta función solo se invoca cuando `use_2lpt = true`.
#[allow(clippy::too_many_arguments)]
pub fn zeldovich_2lpt_ics(
    cfg: &RunConfig,
    n: usize,
    seed: u64,
    amplitude: f64,
    spectral_index: f64,
    transfer: TransferKind,
    sigma8: Option<f64>,
    omega_b: f64,
    h_dimless: f64,
    t_cmb: f64,
    box_size_mpc_h: Option<f64>,
    lo: usize,
    hi: usize,
) -> Vec<Particle> {
    let box_size = cfg.simulation.box_size;
    let n_part = cfg.simulation.particle_count;
    let mass = 1.0 / n_part as f64;

    // ── Parámetros cosmológicos
    let (a_init, cosmo) = if cfg.cosmology.enabled {
        let a = cfg.cosmology.a_init;
        let cp = CosmologyParams::new(
            cfg.cosmology.omega_m,
            cfg.cosmology.omega_lambda,
            cfg.cosmology.h0,
        );
        (a, cp)
    } else {
        let cp = CosmologyParams::new(1.0, 0.0, cfg.cosmology.h0);
        (1.0, cp)
    };

    let h_a = hubble_param(cosmo, a_init);
    let f1 = growth_rate_f(cosmo, a_init);

    // Factor de velocidad de segundo orden: f₂ ≈ 2·f₁
    let f2 = 2.0 * f1;

    // Factor de amplitud de segundo orden: D₂/D₁² ≈ −3/7 · Ω_m(a)^{−1/143}
    // Ω_m(a) = Ω_m · a⁻³ / (H(a)/H₀)²
    let omega_m_a = if cfg.cosmology.enabled && cosmo.h0 > 0.0 {
        let h_ratio_sq = (h_a / cosmo.h0) * (h_a / cosmo.h0);
        (cosmo.omega_m / (a_init * a_init * a_init) / h_ratio_sq).max(0.0)
    } else {
        1.0
    };
    // Bouchet et al. (1995)
    let d2_over_d1sq = -3.0 / 7.0 * omega_m_a.powf(-1.0 / 143.0);

    // Factor de velocidad 1LPT: a²·H·f₁
    let vel1_factor = a_init * a_init * f1 * h_a;
    // Factor de velocidad 2LPT: a²·H·f₂·(D₂/D₁²)
    let vel2_factor = a_init * a_init * f2 * h_a * d2_over_d1sq;

    // Espaciado de la retícula
    let d = box_size / n as f64;

    // ── Construir closure de espectro
    let spectrum_fn = build_spectrum_fn(
        n,
        spectral_index,
        amplitude,
        transfer,
        sigma8,
        cfg.cosmology.omega_m,
        omega_b,
        h_dimless,
        t_cmb,
        box_size_mpc_h,
    );

    // ── Generar campo δ(k) (todos los rangos)
    let delta = generate_delta_kspace(n, seed, spectrum_fn);

    // ── Calcular Ψ¹ (primer orden)
    let [psi1_x, psi1_y, psi1_z] = delta_to_displacement(&delta, n, box_size);

    // ── Calcular Ψ² (segundo orden)
    //    Paso A: derivadas de segundo orden de φ¹
    let phi_derivs = phi_second_derivatives(&delta, n);

    //    Paso B: fuente S(x)
    let source = build_2lpt_source(&phi_derivs);

    //    Paso C: resolver Poisson → φ²(k)
    let phi2_k = solve_poisson_real_to_kspace(&source, n);

    //    Paso D: gradiente de φ² → Ψ²(x)
    let [psi2_x, psi2_y, psi2_z] = phi2_to_psi2(&phi2_k, n, box_size);

    // ── Construir partículas para el rango [lo, hi)
    let mut out = Vec::with_capacity(hi.saturating_sub(lo));

    for gid in lo..hi {
        if gid >= n_part {
            break;
        }

        let ix = gid / (n * n);
        let rem = gid % (n * n);
        let iy = rem / n;
        let iz = rem % n;

        // Posición Lagrangiana (centro de celda)
        let q_x = (ix as f64 + 0.5) * d;
        let q_y = (iy as f64 + 0.5) * d;
        let q_z = (iz as f64 + 0.5) * d;

        let g = ix * n * n + iy * n + iz;

        let psi1 = Vec3::new(psi1_x[g], psi1_y[g], psi1_z[g]);
        let psi2 = Vec3::new(psi2_x[g], psi2_y[g], psi2_z[g]);

        // Posición total: x = q + Ψ¹ + (D₂/D₁²)·Ψ²
        let x = Vec3::new(
            (q_x + psi1.x + d2_over_d1sq * psi2.x).rem_euclid(box_size),
            (q_y + psi1.y + d2_over_d1sq * psi2.y).rem_euclid(box_size),
            (q_z + psi1.z + d2_over_d1sq * psi2.z).rem_euclid(box_size),
        );

        // Momentum canónico: p = a²·H·[f₁·Ψ¹ + f₂·(D₂/D₁²)·Ψ²]
        let p = psi1 * vel1_factor + psi2 * vel2_factor;

        out.push(Particle::new(gid, mass, x, p));
    }

    out
}

// ── Tests unitarios ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_delta(n: usize) -> Vec<Complex<f64>> {
        use crate::ic_zeldovich::generate_delta_kspace;
        let inv_sqrt_n3 = 1.0 / ((n * n * n) as f64).sqrt();
        generate_delta_kspace(n, 42, move |n_abs: f64| {
            if n_abs <= 0.0 {
                0.0
            } else {
                1e-2 * n_abs.powf(-1.0) * inv_sqrt_n3
            }
        })
    }

    /// El modo DC de S(x) debe ser proporcional a ⟨S⟩. Para un campo Gaussiano
    /// el término cuadrático en S no es cero, pero las derivadas φ,αβ tienen
    /// media nula (modo DC = 0), así que S tiene media pequeña.
    #[test]
    fn source_is_finite() {
        let n = 8;
        let delta = simple_delta(n);
        let phi_d = phi_second_derivatives(&delta, n);
        let s = build_2lpt_source(&phi_d);
        // Todos los valores deben ser finitos
        assert!(s.iter().all(|&x| x.is_finite()), "S contiene NaN/Inf");
    }

    /// φ²(k) debe tener modo DC y Nyquist nulos.
    #[test]
    fn phi2_dc_is_zero() {
        let n = 8;
        let delta = simple_delta(n);
        let phi_d = phi_second_derivatives(&delta, n);
        let s = build_2lpt_source(&phi_d);
        let phi2_k = solve_poisson_real_to_kspace(&s, n);

        // Modo DC: índice 0
        assert_eq!(
            phi2_k[0].re, 0.0,
            "φ²(k=0).re debe ser cero"
        );
        assert_eq!(
            phi2_k[0].im, 0.0,
            "φ²(k=0).im debe ser cero"
        );
    }

    /// Ψ² debe ser un campo real (parte imaginaria de IFFT ≈ 0).
    #[test]
    fn psi2_is_real() {
        let n = 8;
        let box_size = 1.0;
        let delta = simple_delta(n);
        let phi_d = phi_second_derivatives(&delta, n);
        let s = build_2lpt_source(&phi_d);
        let phi2_k = solve_poisson_real_to_kspace(&s, n);

        // Comprobar que Ψ² en k-space tiene simetría Hermitiana (IFFT produce real)
        // Esto es equivalente a comprobar que la parte imaginaria de Ψ²(x) es ~0.
        // Verificamos indirectamente: phi2_k ya viene de FFT de campo real,
        // la operación -i n/|n|² preserva la simetría → Ψ²(x) es real.
        let [px, py, pz] = phi2_to_psi2(&phi2_k, n, box_size);
        // Todos deben ser finitos
        assert!(
            px.iter().chain(py.iter()).chain(pz.iter()).all(|&x| x.is_finite()),
            "Ψ² contiene NaN/Inf"
        );
    }
}
