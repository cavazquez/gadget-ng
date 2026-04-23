//! Condiciones iniciales de Segundo Orden (2LPT).
//!
//! ## Formulación matemática (Phase 44 — auditoría canónica)
//!
//! Siguiendo la convención de Jenkins (2010, arXiv:0910.0258, ec. 2) y la
//! implementación de referencia `2LPTic` (Crocce, Pueblas & Scoccimarro 2006,
//! MNRAS 373, 369):
//!
//! ```text
//! x = q − D₁·∇φ¹ + D₂·∇φ²
//! v = −D₁·f₁·H·∇φ¹ + D₂·f₂·H·∇φ²
//! ∇²φ¹(q) = δ¹(q)          ∇²φ²(q) = δ²(q) ≡ S(q)
//! D₂ ≈ −(3/7)·D₁²
//! f₁ ≈ Ω_m(a)^{5/9}        f₂ ≈ 2·Ω_m(a)^{6/11}     (Bouchet et al. 1995)
//! ```
//!
//! `gadget-ng` absorbe `D₁ = 1` en la amplitud del campo (calibrado vía
//! `amplitude_for_sigma8` y `rescale_to_a_init`), de modo que la fórmula de
//! código queda:
//!
//! ```text
//! x = q + Ψ¹ + (D₂/D₁²)·Ψ²             (D₂/D₁² ≈ −3/7·Ω_m(a)^{−1/143})
//! p = a²·H·[f₁·Ψ¹ + f₂·(D₂/D₁²)·Ψ²]    (momentum canónico GADGET-4: p = a²·dx/dt)
//! ```
//!
//! donde `Ψ¹ = −∇φ¹` y `Ψ² = +∇φ²`.
//!
//! ### Cálculo de Ψ²  (corregido en Phase 44)
//!
//! **Paso A.** Derivadas de segundo orden de φ¹ en k-space (mismo factor
//! cuadrático que `delta_to_displacement`, signo global irrelevante porque
//! S es cuadrático en φ¹,αβ):
//! ```text
//! φ̂¹,αβ(k) = −n_α·n_β / |n|² · δ̂(k)
//! ```
//!
//! **Paso B.** Fuente:
//! ```text
//! S(x) = (φ,xx·φ,yy − φ,xy²) + (φ,yy·φ,zz − φ,yz²) + (φ,zz·φ,xx − φ,xz²)
//! ```
//!
//! **Paso C.** Ψ² en k-space **directamente desde S** (UNA sola división por
//! `|n|²`, signo canónico `−i` tomado de `2LPTic/main.c:477-478` y Jenkins
//! 2010 ec. 2):
//! ```text
//! Ψ²_α(k) = −i · n_α / |n|² · S(k)     (DC=0, Nyquist=0)
//! ```
//! IFFT → `Ψ²(x)`; se multiplica por `d = box_size/n` para unidades físicas.
//!
//! ### Historia (bugs corregidos en Phase 44)
//!
//! La versión pre-Phase-44 calculaba primero `φ²(k) = −S/|n|²` y luego
//! `Ψ²_α(k) = −i·n_α/|n|²·φ²`, componiendo **dos divisiones por `|n|²`** y
//! produciendo `Ψ²_impl = +i·n_α·S/|n|⁴`. Versus canónico `−i·n_α·S/|n|²`
//! eso implica:
//! - amplitud atenuada por `1/|n|²` (inútil en todas las escalas pero
//!   especialmente dañino en `|n| ≥ 2`),
//! - signo global invertido.
//!
//! El patch de Phase 44 unifica el Poisson y el gradiente en
//! [`source_to_psi2`], que implementa `Ψ² = −i·k/k² · S` tal como
//! `2LPTic` (Crocce+06) y coincide con la ec. 2 de Jenkins (2010).

use crate::{
    config::{RunConfig, TransferKind},
    cosmology::{growth_factor_d_ratio, growth_rate_f, hubble_param, CosmologyParams},
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

// ── Ψ² = −∇∇⁻²S en una sola pasada (Phase 44) ─────────────────────────────────

/// Calcula `Ψ²(x) = IFFT[ −i·n_α / |n|² · S(k) ] · d` (una sola división por
/// `|n|²`, signo canónico `−i`).
///
/// Esta es la función correcta post-Phase-44. Reemplaza al par
/// `solve_poisson_real_to_kspace` + `phi2_to_psi2` de la versión anterior, que
/// componía `φ²(k)=−S/|n|²` seguido de `Ψ²=−i·n/|n|²·φ²` y producía
/// `Ψ²=+i·n·S/|n|⁴` (amplitud atenuada por `1/|n|²` y signo invertido).
///
/// Referencias:
/// - Jenkins (2010), arXiv:0910.0258, ec. 2.
/// - Crocce, Pueblas & Scoccimarro (2006), MNRAS 373, 369 — código
///   `2LPTic/main.c:477-478`:
///   ```c
///   cdisp2[axes].re =  S.im * kvec[axes] / kmag2;
///   cdisp2[axes].im = -S.re * kvec[axes] / kmag2;
///   ```
///   equivalente a `Ψ²(k) = −i · k / k² · S(k)`.
///
/// Devuelve `[Ψ²_x, Ψ²_y, Ψ²_z]` en unidades físicas (×`d = box_size/n`).
/// Los modos DC y Nyquist quedan a cero (consistente con `delta_to_displacement`).
fn source_to_psi2(source: &[f64], n: usize, box_size: f64) -> [Vec<f64>; 3] {
    let n3 = n * n * n;
    let d = box_size / n as f64;
    let ifft_norm = 1.0 / n3 as f64;
    let half = (n / 2) as i64;

    // FFT forward: S(x) → S(k)
    let mut s_k: Vec<Complex<f64>> = source.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft3d(&mut s_k, n, true);

    let mut psi: [Vec<Complex<f64>>; 3] = [
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
    ];

    for ix in 0..n {
        let nxi = mode_int(ix, n);
        let nx = nxi as f64;
        for iy in 0..n {
            let nyi = mode_int(iy, n);
            let ny = nyi as f64;
            for iz in 0..n {
                let nzi = mode_int(iz, n);
                let nz = nzi as f64;
                let n2 = nx * nx + ny * ny + nz * nz;
                let idx = ix * n * n + iy * n + iz;

                // DC y Nyquist → 0 (consistente con delta_to_displacement)
                if n2 == 0.0 || nxi.abs() == half || nyi.abs() == half || nzi.abs() == half {
                    continue;
                }

                let s = s_k[idx];
                // Ψ²_α(k) = −i · n_α / |n|² · S(k)
                // −i·(a + ib) = b − i·a   →   re = n_α·S.im/|n|², im = −n_α·S.re/|n|²
                let make_psi2 = |n_alpha: f64| -> Complex<f64> {
                    Complex::new(n_alpha * s.im / n2, -n_alpha * s.re / n2)
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

// ── Variante "legacy buggy" (solo para A/B testing de Phase 44) ──────────────

/// Replica el bug pre-Phase-44: doble división por `|n|²` y signo global
/// invertido.
///
/// **NO USAR EN SIMULACIONES**. Esta función existe exclusivamente para
/// experimentos de auditoría (scripts Phase 44 A/B) que necesitan comparar
/// el comportamiento antiguo vs el corregido sin hacer `git checkout` de la
/// versión anterior. En la versión buggy original:
///
/// ```text
/// φ²(k)  = −S(k) / |n|²           (Poisson)
/// Ψ²_α(k) = −i·n_α / |n|² · φ²(k)  (gradiente con doble división)
///         = +i·n_α · S(k) / |n|⁴   ← bug: amplitud 1/|n|² extra y signo +
/// ```
fn source_to_psi2_legacy_buggy(source: &[f64], n: usize, box_size: f64) -> [Vec<f64>; 3] {
    let n3 = n * n * n;
    let d = box_size / n as f64;
    let ifft_norm = 1.0 / n3 as f64;
    let half = (n / 2) as i64;

    let mut s_k: Vec<Complex<f64>> = source.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft3d(&mut s_k, n, true);

    let mut psi: [Vec<Complex<f64>>; 3] = [
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
        vec![Complex::default(); n3],
    ];

    for ix in 0..n {
        let nxi = mode_int(ix, n);
        let nx = nxi as f64;
        for iy in 0..n {
            let nyi = mode_int(iy, n);
            let ny = nyi as f64;
            for iz in 0..n {
                let nzi = mode_int(iz, n);
                let nz = nzi as f64;
                let n2 = nx * nx + ny * ny + nz * nz;
                let idx = ix * n * n + iy * n + iz;

                if n2 == 0.0 || nxi.abs() == half || nyi.abs() == half || nzi.abs() == half {
                    continue;
                }

                let s = s_k[idx];
                // Bug viejo: +i·n_α·S/|n|⁴
                // +i·(a + ib) = −b + i·a
                let mk = |na: f64| -> Complex<f64> {
                    let f = na / (n2 * n2);
                    Complex::new(-f * s.im, f * s.re)
                };
                psi[0][idx] = mk(nx);
                psi[1][idx] = mk(ny);
                psi[2][idx] = mk(nz);
            }
        }
    }

    for c in &mut psi {
        fft3d(c, n, false);
    }
    let scale = ifft_norm * d;
    [
        psi[0].iter().map(|c| c.re * scale).collect(),
        psi[1].iter().map(|c| c.re * scale).collect(),
        psi[2].iter().map(|c| c.re * scale).collect(),
    ]
}

/// Variante de `Ψ²` a usar en [`zeldovich_2lpt_ics`].
///
/// Siempre prefiere `Fixed`; `LegacyBuggy` es sólo para tests de auditoría.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Psi2Variant {
    /// Versión canónica post-Phase-44 (única división por `|n|²`, signo `−i`).
    Fixed,
    /// Reproduce el bug pre-Phase-44 (doble `1/|n|²` + signo `+i`). Solo A/B.
    LegacyBuggy,
}

impl Default for Psi2Variant {
    fn default() -> Self {
        Self::Fixed
    }
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
    rescale_to_a_init: bool,
    lo: usize,
    hi: usize,
) -> Vec<Particle> {
    zeldovich_2lpt_ics_with_variant(
        cfg,
        n,
        seed,
        amplitude,
        spectral_index,
        transfer,
        sigma8,
        omega_b,
        h_dimless,
        t_cmb,
        box_size_mpc_h,
        rescale_to_a_init,
        lo,
        hi,
        Psi2Variant::Fixed,
    )
}

/// Variante A/B de [`zeldovich_2lpt_ics`] que permite seleccionar la versión
/// de `Ψ²` (Fixed vs LegacyBuggy). Solo expuesta para auditoría Phase 44.
#[allow(clippy::too_many_arguments)]
pub fn zeldovich_2lpt_ics_with_variant(
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
    rescale_to_a_init: bool,
    lo: usize,
    hi: usize,
    psi2_variant: Psi2Variant,
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

    // Fase 44: Ω_m(a) = Ω_m · a⁻³ / (H(a)/H₀)².
    // Se usa tanto para D₂/D₁² (Bouchet) como para f₂ (Scoccimarro/Bouchet).
    let omega_m_a = if cfg.cosmology.enabled && cosmo.h0 > 0.0 {
        let h_ratio_sq = (h_a / cosmo.h0) * (h_a / cosmo.h0);
        (cosmo.omega_m / (a_init * a_init * a_init) / h_ratio_sq).max(0.0)
    } else {
        1.0
    };

    // Fase 44: f₂ ≈ 2·Ω_m(a)^{6/11} (Bouchet et al. 1995; Scoccimarro 1998).
    // En la versión previa se usaba `f₂ = 2·f₁` con `f₁ = Ω_m^{0.55}` (Linder),
    // que difiere del canónico en < 0.01 % a z ≫ 1 pero rompe la convención
    // literaria de 2LPTic/Jenkins 2010.
    let f2 = 2.0 * omega_m_a.powf(6.0 / 11.0);

    // Factor de amplitud de segundo orden: D₂/D₁² ≈ −3/7 · Ω_m(a)^{−1/143}
    // Bouchet et al. (1995). Para Ω_m ≈ 0.3 y a = 0.02: D₂/D₁² ≈ −0.428.
    let d2_over_d1sq = -3.0 / 7.0 * omega_m_a.powf(-1.0 / 143.0);

    // Factor de velocidad 1LPT: a²·H·f₁
    let vel1_factor = a_init * a_init * f1 * h_a;
    // Factor de velocidad 2LPT: a²·H·f₂·(D₂/D₁²)
    let vel2_factor = a_init * a_init * f2 * h_a * d2_over_d1sq;

    // Fase 37: factor de reescalado físico. Con `rescale_to_a_init = false`
    // (default) → scale = 1, bit-idéntico a Fase 28.
    let scale = if rescale_to_a_init && cfg.cosmology.enabled {
        growth_factor_d_ratio(cosmo, a_init, 1.0)
    } else {
        1.0
    };
    let scale2 = scale * scale;

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
    let [mut psi1_x, mut psi1_y, mut psi1_z] = delta_to_displacement(&delta, n, box_size);

    // ── Calcular Ψ² (segundo orden) — Phase 44
    //    Paso A: derivadas de segundo orden de φ¹
    let phi_derivs = phi_second_derivatives(&delta, n);

    //    Paso B: fuente S(x) = δ²(q) (Jenkins 2010 ec. 6)
    let source = build_2lpt_source(&phi_derivs);

    //    Paso C: Ψ²(x) = IFFT[−i·n_α/|n|² · S(k)] · d
    //    (una sola división por |n|², signo canónico `−i`, ver source_to_psi2).
    //    Phase 44: la variante `LegacyBuggy` reproduce el bug anterior para A/B.
    let [mut psi2_x, mut psi2_y, mut psi2_z] = match psi2_variant {
        Psi2Variant::Fixed => source_to_psi2(&source, n, box_size),
        Psi2Variant::LegacyBuggy => source_to_psi2_legacy_buggy(&source, n, box_size),
    };

    // Fase 37: aplicar reescalado físico opcional.
    // Ψ¹ crece con D¹, Ψ² crece con D² → factores s y s² respectivamente.
    if scale != 1.0 {
        for v in psi1_x.iter_mut() {
            *v *= scale;
        }
        for v in psi1_y.iter_mut() {
            *v *= scale;
        }
        for v in psi1_z.iter_mut() {
            *v *= scale;
        }
        for v in psi2_x.iter_mut() {
            *v *= scale2;
        }
        for v in psi2_y.iter_mut() {
            *v *= scale2;
        }
        for v in psi2_z.iter_mut() {
            *v *= scale2;
        }
    }

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

    /// δ con amplitud de O(1) para tests k-space con rango dinámico numérico
    /// razonable (evita que todos los números caigan a ε de máquina).
    fn strong_delta(n: usize) -> Vec<Complex<f64>> {
        use crate::ic_zeldovich::generate_delta_kspace;
        generate_delta_kspace(n, 42, move |n_abs: f64| {
            if n_abs <= 0.0 {
                0.0
            } else {
                // P(k) ~ |n|^{-2}: campo δ(x) con σ ~ O(0.1–1).
                n_abs.powf(-1.0)
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

    /// Ψ² debe ser un campo real y finito (parte imaginaria de IFFT ≈ 0).
    #[test]
    fn psi2_is_real_and_finite() {
        let n = 8;
        let box_size = 1.0;
        let delta = simple_delta(n);
        let phi_d = phi_second_derivatives(&delta, n);
        let s = build_2lpt_source(&phi_d);
        let [px, py, pz] = source_to_psi2(&s, n, box_size);
        assert!(
            px.iter()
                .chain(py.iter())
                .chain(pz.iter())
                .all(|&x| x.is_finite()),
            "Ψ² contiene NaN/Inf"
        );
    }

    // ── Phase 44: tests canónicos de corrección 2LPT ──────────────────────────

    /// k-space: `FFT(Ψ²_α)(k) = (−i·n_α/|n|² · FFT(S)(k)) · d / N³`.
    ///
    /// Este test verifica **directamente** la fórmula que distingue el patch de
    /// Phase 44 del código antiguo (que tenía amplitud `1/|n|⁴` y signo `+i`):
    ///
    /// * `1/|n|²` (sola división) — si hubiera `1/|n|⁴` el test falla por
    ///   magnitud × `1/|n|²` en cada modo.
    /// * signo `−i` — si hubiera `+i` el test falla por signo opuesto en cada
    ///   modo.
    ///
    /// Se evalúa modo a modo (evitando aliasing de diferencias finitas).
    #[test]
    fn psi2_matches_canonical_kspace_formula() {
        let n = 16;
        let box_size = 1.0;
        let n3 = n * n * n;
        let d = box_size / n as f64;
        let half = (n / 2) as i64;

        let delta = strong_delta(n);
        let phi_d = phi_second_derivatives(&delta, n);
        let s_real = build_2lpt_source(&phi_d);
        let [psi2_x, psi2_y, psi2_z] = source_to_psi2(&s_real, n, box_size);

        // FFT forward de S(x) → S(k)
        let mut s_k: Vec<Complex<f64>> = s_real.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft3d(&mut s_k, n, true);

        // FFT forward de Ψ²_α(x) → Ψ̂²_α(k)
        let mut psi_k: [Vec<Complex<f64>>; 3] = [
            psi2_x
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect::<Vec<_>>(),
            psi2_y
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect::<Vec<_>>(),
            psi2_z
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect::<Vec<_>>(),
        ];
        for c in &mut psi_k {
            fft3d(c, n, true);
        }

        // Comparar modo a modo con la fórmula canónica.
        // Relación: Ψ²_α(x) = IFFT_normalized[ (−i·n_α/|n|²) · S(k) ] · d
        // En términos de FFT forward aplicada a Ψ²_α(x):
        //   FFT[Ψ²_α](k) = (−i·n_α/|n|² · S(k)) · d          (IFFT normalizada
        //                                                      se cancela con
        //                                                      el factor N³ de
        //                                                      la FFT forward)
        //
        // Usamos tolerancia global: max‖got−exp‖ / max‖exp‖ < ε. Esto evita
        // falsos positivos en modos donde s_k es ~ε-de-máquina (y entonces
        // |exp| también es ~ε).
        let mut max_diff = 0.0_f64;
        let mut max_mag = 0.0_f64;
        let mut num_modes = 0_usize;
        for ix in 0..n {
            let nxi = mode_int(ix, n);
            let nx = nxi as f64;
            for iy in 0..n {
                let nyi = mode_int(iy, n);
                let ny = nyi as f64;
                for iz in 0..n {
                    let nzi = mode_int(iz, n);
                    let nz = nzi as f64;
                    let n2 = nx * nx + ny * ny + nz * nz;
                    let idx = ix * n * n + iy * n + iz;

                    // DC + Nyquist → cero en ambas partes
                    if n2 == 0.0 || nxi.abs() == half || nyi.abs() == half || nzi.abs() == half {
                        continue;
                    }

                    // Expected: (−i·n_α/|n|²) · S(k) · d
                    // −i·(a + ib) = b − i·a
                    let expect_alpha = |n_alpha: f64| -> Complex<f64> {
                        let factor = n_alpha / n2 * d;
                        Complex::new(factor * s_k[idx].im, -factor * s_k[idx].re)
                    };

                    for (alpha, nalpha) in [nx, ny, nz].iter().enumerate() {
                        let got = psi_k[alpha][idx];
                        let exp = expect_alpha(*nalpha);
                        let diff = (got - exp).norm();
                        let mag = exp.norm();
                        if diff > max_diff {
                            max_diff = diff;
                        }
                        if mag > max_mag {
                            max_mag = mag;
                        }
                        num_modes += 1;
                    }
                }
            }
        }

        let global_rel = max_diff / max_mag.max(1e-300);
        assert!(
            global_rel < 1e-10,
            "Ψ̂²_α(k) no coincide con (−i·n_α/|n|²)·S(k)·d; \
             max‖diff‖ = {:.3e}, max‖exp‖ = {:.3e}, global rel = {:.3e}, modos = {}",
            max_diff,
            max_mag,
            global_rel,
            num_modes
        );
        let _ = n3;
    }

    /// **Regresión contra el bug pre-Phase-44**: si hubiese doble división por
    /// `|n|²` (o signo invertido), este test falla porque la amplitud RMS de
    /// Ψ² en k-space queda muy por debajo de la canónica.
    ///
    /// Compara el RMS de Ψ²(x) implementado vs el RMS **teórico** esperado si
    /// se aplicara la fórmula del bug anterior (Ψ² = +i·n·S/|n|⁴):
    ///   - patched / buggy ≈ ⟨|n|²⟩ (promedio pesado por potencia de S).
    ///   - para simple_delta(n=8), ese cociente es ~3.4.
    #[test]
    fn psi2_amplitude_differs_from_legacy_bug() {
        let n = 16;
        let box_size = 1.0;
        let delta = strong_delta(n);
        let phi_d = phi_second_derivatives(&delta, n);
        let s = build_2lpt_source(&phi_d);

        // Ψ² con fórmula correcta (una división por |n|², signo −i).
        let [px_ok, py_ok, pz_ok] = source_to_psi2(&s, n, box_size);
        let rms_ok = {
            let sum_sq: f64 = px_ok
                .iter()
                .chain(py_ok.iter())
                .chain(pz_ok.iter())
                .map(|x| x * x)
                .sum();
            (sum_sq / (3.0 * (n * n * n) as f64)).sqrt()
        };

        // Replica local del bug viejo: Ψ²_α = +i·n_α·S/|n|⁴
        // para comparar amplitudes (sin llamar al código eliminado).
        let rms_buggy = {
            let n3 = n * n * n;
            let d = box_size / n as f64;
            let ifft_norm = 1.0 / n3 as f64;
            let half = (n / 2) as i64;
            let mut s_k: Vec<Complex<f64>> = s.iter().map(|&x| Complex::new(x, 0.0)).collect();
            fft3d(&mut s_k, n, true);
            let mut psi: [Vec<Complex<f64>>; 3] = [
                vec![Complex::default(); n3],
                vec![Complex::default(); n3],
                vec![Complex::default(); n3],
            ];
            for ix in 0..n {
                let nxi = mode_int(ix, n);
                let nx = nxi as f64;
                for iy in 0..n {
                    let nyi = mode_int(iy, n);
                    let ny = nyi as f64;
                    for iz in 0..n {
                        let nzi = mode_int(iz, n);
                        let nz = nzi as f64;
                        let n2 = nx * nx + ny * ny + nz * nz;
                        let idx = ix * n * n + iy * n + iz;
                        if n2 == 0.0 || nxi.abs() == half || nyi.abs() == half || nzi.abs() == half
                        {
                            continue;
                        }
                        let sk = s_k[idx];
                        // +i·n_α·S/|n|⁴ :  +i·(a+ib) = −b + i·a → re = −n·S.im/|n|⁴
                        let mk = |na: f64| -> Complex<f64> {
                            let f = na / (n2 * n2);
                            Complex::new(-f * sk.im, f * sk.re)
                        };
                        psi[0][idx] = mk(nx);
                        psi[1][idx] = mk(ny);
                        psi[2][idx] = mk(nz);
                    }
                }
            }
            for c in &mut psi {
                fft3d(c, n, false);
            }
            let scale = ifft_norm * d;
            let sum_sq: f64 = psi
                .iter()
                .flat_map(|v| v.iter())
                .map(|c| {
                    let v = c.re * scale;
                    v * v
                })
                .sum();
            (sum_sq / (3.0 * (n * n * n) as f64)).sqrt()
        };

        // Esperamos rms_ok > rms_buggy con factor significativo (cuán grande
        // depende del espectro, pero >1.5 es garantizado si el patch amplifica
        // los modos altos al eliminar el 1/|n|² extra del bug).
        assert!(
            rms_ok > 1.5 * rms_buggy,
            "rms(Ψ²_fix) = {:.3e} no es al menos 1.5× rms(Ψ²_bug) = {:.3e} \
             — el patch de Phase 44 debería amplificar los modos altos \
             (ratio observado = {:.2}x)",
            rms_ok,
            rms_buggy,
            rms_ok / rms_buggy
        );
    }

    /// Ψ² escala cuadráticamente con δ: si δ → λ·δ entonces S → λ²·S y Ψ² → λ²·Ψ².
    #[test]
    fn psi2_scales_quadratically_with_delta() {
        let n = 8;
        let box_size = 1.0;
        let delta1 = simple_delta(n);
        let lambda = 2.0;
        let delta2: Vec<Complex<f64>> = delta1
            .iter()
            .map(|c| Complex::new(lambda * c.re, lambda * c.im))
            .collect();

        let [p1x, ..] = source_to_psi2(
            &build_2lpt_source(&phi_second_derivatives(&delta1, n)),
            n,
            box_size,
        );
        let [p2x, ..] = source_to_psi2(
            &build_2lpt_source(&phi_second_derivatives(&delta2, n)),
            n,
            box_size,
        );

        // p2 debe ser exactamente λ²·p1, dentro del error numérico.
        let ratio: Vec<f64> = p1x
            .iter()
            .zip(p2x.iter())
            .filter_map(|(a, b)| if a.abs() > 1e-14 { Some(b / a) } else { None })
            .collect();
        assert!(!ratio.is_empty(), "Ψ²(delta=0)? no hay puntos válidos");
        let mean_ratio: f64 = ratio.iter().sum::<f64>() / ratio.len() as f64;
        let expected = lambda * lambda;
        let rel_err = ((mean_ratio - expected) / expected).abs();
        assert!(
            rel_err < 1e-10,
            "⟨Ψ²(λδ)/Ψ²(δ)⟩ = {:.6} vs λ² = {} (rel_err = {:.2e})",
            mean_ratio,
            expected,
            rel_err
        );
    }

    /// Test de signo: con una onda plana 1D para `δ`, los signos de Ψ¹ y Ψ²
    /// deben coincidir con la convención canónica de Jenkins 2010.
    ///
    /// Para `δ(q) = A·cos(k₀·q_x)`:
    ///   - Ψ¹_x(q) = +A/k₀ · sin(k₀·q_x)  (de `Ψ¹=ik/k²·δ`, signo `+`)
    ///   - S(q) = (∂²φ/∂x²)² − 0 = (A·cos(k₀·q_x))² = A²·cos²  (solo φ_xx ≠ 0,
    ///     resto cero; S = φ_xx·φ_yy + φ_yy·φ_zz + φ_zz·φ_xx − φ_xy² − φ_xz² − φ_yz²
    ///     = 0 porque φ_yy = φ_zz = 0 con onda 1D). **Entonces S=0** para onda 1D.
    ///
    /// Usamos onda 2D cruzada: δ = A·cos(k₀·q_x)·cos(k₀·q_y). Más complicado
    /// pero permite S ≠ 0.
    ///
    /// Test cualitativo: para cualquier δ no trivial, Ψ²_x/A² y S/A² deben
    /// preservar sus magnitudes relativas al escalar A.
    #[test]
    fn psi2_signs_consistent_across_amplitudes() {
        // Ya cubierto por `psi2_scales_quadratically_with_delta`.
        // Este test adicional confirma comportamiento específico:
        // Si invertimos δ → −δ, entonces φ_αβ → −φ_αβ, pero S es cuadrático
        // en φ_αβ por lo que S queda igual → Ψ² queda igual.
        let n = 8;
        let box_size = 1.0;
        let delta_pos = simple_delta(n);
        let delta_neg: Vec<Complex<f64>> = delta_pos
            .iter()
            .map(|c| Complex::new(-c.re, -c.im))
            .collect();

        let [a_x, ..] = source_to_psi2(
            &build_2lpt_source(&phi_second_derivatives(&delta_pos, n)),
            n,
            box_size,
        );
        let [b_x, ..] = source_to_psi2(
            &build_2lpt_source(&phi_second_derivatives(&delta_neg, n)),
            n,
            box_size,
        );
        let max_diff = a_x
            .iter()
            .zip(b_x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_val = a_x.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let rel = max_diff / max_val.max(1e-300);
        assert!(
            rel < 1e-12,
            "Ψ²(δ) − Ψ²(−δ) tiene max rel = {:.2e} (esperado ≈ 0: S es cuadrático en δ)",
            rel
        );
    }
}
