//! Generador de condiciones iniciales de Zel'dovich (1LPT).
//!
//! ## Formulación matemática
//!
//! La aproximación de Zel'dovich conecta las posiciones Lagrangianas `q` con las
//! posiciones Eulerianas `x` mediante un campo de desplazamiento `Ψ`:
//!
//! ```text
//! x_i = q_i + Ψ(q_i)
//! p_i = a²·f(a)·H(a)·Ψ(q_i)         (momentum canónico GADGET)
//! ```
//!
//! El campo `Ψ` se obtiene de un potencial de desplazamiento Φ definido por:
//!
//! ```text
//! ∇²Φ = −δ   →   Φ̂(k) = δ̂(k)/k²
//! Ψ = −∇Φ    →   Ψ̂_α(k) = i·k_α/k²·δ̂(k)
//! ```
//!
//! ## Espectro de potencia
//!
//! El campo de densidad gaussiano tiene espectro `P(k) = amplitude²·|n|^spectral_index`
//! en unidades de grid, donde `n = (nx,ny,nz)` es el vector de modo entero y
//! `|n|² = nx²+ny²+nz²`.
//!
//! La varianza por modo es: `σ²(k) = P(k)/N³`
//!
//! ## Convención de FFT (rustfft)
//!
//! Se usa DFT directa con signo negativo en el exponente para la transformada forward:
//! `f̂[j] = Σ_n f[n] · exp(−2πi·j·n/N)`.
//!
//! Los índices de modo: `n_α = j` para `j ≤ N/2`, `n_α = j − N` para `j > N/2`.
//! El factor de normalización de la IFFT de rustfft es `1/N` por eje → `1/N³` total.
//!
//! ## Simetría Hermitiana
//!
//! Para que `Ψ` sea real tras la IFFT, imponemos `δ̂(−k) = conj(δ̂(k))`.
//! El modo DC (`k=0`) se fija a cero: `δ̂(0)=0`.
//! Los modos Nyquist (`|n_α|=N/2`) se fijan a cero para evitar aliasing.
//!
//! ## Reproducibilidad en MPI
//!
//! El campo completo N³ se genera de forma determinista a partir de `seed`
//! en todos los rangos. Cada rango extrae luego su segmento `[lo, hi)` de `gid`.
//! Esto garantiza reproducibilidad exacta independientemente del número de rangos.
//! Es correcto para N ≤ 64³ (memoria < 200 MB por componente).

use crate::{
    config::{RunConfig, TransferKind},
    cosmology::{growth_factor_d_ratio, growth_rate_f, hubble_param, CosmologyParams},
    particle::Particle,
    transfer_fn::{amplitude_for_sigma8, transfer_eh_nowiggle, EisensteinHuParams},
    vec3::Vec3,
};
use rustfft::{num_complex::Complex, FftPlanner};

// ── Generador LCG ─────────────────────────────────────────────────────────────

/// Función de mezcla de bits determinista (finalizador de Murmur3).
#[inline]
fn hash_u64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// Mezcla la seed global con el índice del modo para reproducibilidad por modo.
#[inline]
fn mode_seed(global_seed: u64, ix: usize, iy: usize, iz: usize, n: usize) -> u64 {
    let packed = ix as u64 * n as u64 * n as u64 + iy as u64 * n as u64 + iz as u64;
    hash_u64(global_seed ^ hash_u64(packed))
}

/// Generador LCG simple (Knuth MMIX).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Número en `(0, 1)` usando 53 bits.
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Gaussiana estándar via Box-Muller.
    fn gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::TAU * u2).cos()
    }
}

// ── Índice de modo con signo ──────────────────────────────────────────────────

/// Convierte índice FFT `j ∈ [0, n)` al número de modo con signo `n ∈ [−n/2, n/2)`.
#[inline]
pub fn mode_int(j: usize, n: usize) -> i64 {
    let half = (n / 2) as i64;
    let jj = j as i64;
    if jj <= half {
        jj
    } else {
        jj - n as i64
    }
}

// ── Generación del campo gaussiano en k-space ─────────────────────────────────

/// Genera el campo de densidad gaussiano `δ̂(k)` con simetría Hermitiana.
///
/// Parámetros:
/// - `n`: lado del grid (número de celdas por eje).
/// - `seed`: semilla del generador.
/// - `spectrum_fn`: closure `|n_abs: f64| -> f64` que recibe el módulo del vector de
///   modo entero `|n| = sqrt(nx²+ny²+nz²)` y devuelve la desviación estándar `σ(n)`
///   por modo. Este closure encapsula toda la física del espectro (amplitud, índice,
///   función de transferencia, etc.) en una única función genérica.
///
/// Devuelve un array de longitud `n³` con los modos `δ̂`.
///
/// Convención de almacenamiento: índice `ix*n*n + iy*n + iz`.
pub fn generate_delta_kspace(
    n: usize,
    seed: u64,
    spectrum_fn: impl Fn(f64) -> f64,
) -> Vec<Complex<f64>> {
    let n3 = n * n * n;
    let mut delta = vec![Complex::new(0.0, 0.0); n3];

    // Paso 1: generar la mitad "upper" del campo (iz >= 0, iz=0 tomamos iy >= 0).
    // Garantizar simetría Hermitiana a posteriori.
    for ix in 0..n {
        let nx = mode_int(ix, n);
        for iy in 0..n {
            let ny = mode_int(iy, n);
            for iz in 0..n {
                let nz = mode_int(iz, n);

                let n2 = (nx * nx + ny * ny + nz * nz) as f64;

                // Modo DC: forzar a cero.
                if n2 == 0.0 {
                    continue;
                }

                // Modos Nyquist: forzar a cero para evitar aliasing.
                let half = (n / 2) as i64;
                if nx.abs() == half || ny.abs() == half || nz.abs() == half {
                    continue;
                }

                // σ(k) determinada por el closure (puede ser power-law, EH, etc.)
                let sigma = spectrum_fn(n2.sqrt());

                // Generador por modo: reproducible con seed+modo.
                let mut lcg = Lcg::new(mode_seed(seed, ix, iy, iz, n));
                let g_r = lcg.gauss();
                let g_i = lcg.gauss();

                let val = Complex::new(sigma * g_r, sigma * g_i);
                delta[ix * n * n + iy * n + iz] = val;
            }
        }
    }

    // Paso 2: imponer simetría Hermitiana: δ̂(−k) = conj(δ̂(k)).
    for ix in 0..n {
        let nx = mode_int(ix, n);
        for iy in 0..n {
            let ny = mode_int(iy, n);
            for iz in 0..n {
                let nz = mode_int(iz, n);

                let n2 = (nx * nx + ny * ny + nz * nz) as f64;
                if n2 == 0.0 {
                    continue;
                }
                let half = (n / 2) as i64;
                if nx.abs() == half || ny.abs() == half || nz.abs() == half {
                    continue;
                }

                // Índice del modo conjugado (−k).
                let ix_neg = ((-nx).rem_euclid(n as i64)) as usize;
                let iy_neg = ((-ny).rem_euclid(n as i64)) as usize;
                let iz_neg = ((-nz).rem_euclid(n as i64)) as usize;

                // Solo procesar cada par una vez (modo "upper": iz > 0, o iz=0 iy>0, o iz=0 iy=0 ix>0).
                let is_upper = iz > 0 || (iz == 0 && iy > 0) || (iz == 0 && iy == 0 && ix > 0);

                if is_upper {
                    let val = delta[ix * n * n + iy * n + iz];
                    delta[ix_neg * n * n + iy_neg * n + iz_neg] = val.conj();
                }
            }
        }
    }

    delta
}

/// Construye el closure `spectrum_fn` para el generador de k-space según la configuración.
///
/// Encapsula la lógica de selección entre:
/// - Ley de potencia (legacy): `σ(|n|) = amplitude · |n|^(n_s/2) / sqrt(N³)`
/// - Eisenstein–Hu + σ₈: `σ(|n|) = A · k_phys(n)^(n_s/2) · T(k_phys) / sqrt(N³)`
///
/// La firma del closure retornado es `|n_abs: f64| -> f64`, compatible con
/// `generate_delta_kspace`.
#[allow(clippy::too_many_arguments)]
pub fn build_spectrum_fn(
    n: usize,
    spectral_index: f64,
    amplitude: f64,
    transfer: TransferKind,
    sigma8: Option<f64>,
    omega_m: f64,
    omega_b: f64,
    h: f64,
    t_cmb: f64,
    box_size_mpc_h: Option<f64>,
) -> Box<dyn Fn(f64) -> f64> {
    let n3 = n * n * n;
    let inv_sqrt_n3 = 1.0 / (n3 as f64).sqrt();

    match transfer {
        TransferKind::PowerLaw => {
            // Legacy: σ(|n|) = A · |n|^(n_s/2) / sqrt(N³)
            // Se normaliza como σ² para la varianza: P(k) = A² · |n|^n_s
            // Pero generate_delta_kspace multiplica σ por un gaussiano N(0,1),
            // así que el factor correcto es A · |n|^(n_s/4) para que P(k) = A² · |n|^(n_s/2).
            // Compatibilidad con Fase 26: mantener la misma normalización.
            Box::new(move |n_abs: f64| {
                if n_abs <= 0.0 {
                    return 0.0;
                }
                amplitude * n_abs.powf(spectral_index / 4.0) * inv_sqrt_n3
            })
        }
        TransferKind::EisensteinHu => {
            let bsm = box_size_mpc_h.unwrap_or(100.0); // fallback 100 Mpc/h
            let eh_params = EisensteinHuParams {
                omega_m,
                omega_b,
                h,
                t_cmb,
            };

            // Calcular amplitud A para σ₈ si se especificó.
            let amp = if let Some(s8) = sigma8 {
                amplitude_for_sigma8(s8, spectral_index, &eh_params)
            } else {
                amplitude
            };

            // σ(|n|) = A · k_phys^(n_s/4) · sqrt(T(k_phys)) / sqrt(N³)
            // donde k_phys = 2π·|n|·h / box_size_mpc_h  [h/Mpc]
            // Nota: el exponente n_s/4 viene de que P(k) = amp² · k^n_s · T²(k)
            // y generate_delta_kspace genera δ ~ N(0, σ²), así que
            // σ = amp · k^(n_s/2) · T(k) / sqrt(N³) para que P = σ²·N³ = amp² k^n_s T² ✓
            // Pero la convención de Fase 26 usaba σ = A · |n|^(n_s/4) / sqrt(N³),
            // lo que da P(k) = A² · |n|^(n_s/2). Para EH usamos:
            // σ(n) = amp · k^(n_s/2) · T(k) / sqrt(N³)
            let two_pi_h_over_bsm = 2.0 * std::f64::consts::PI * h / bsm;
            Box::new(move |n_abs: f64| {
                if n_abs <= 0.0 {
                    return 0.0;
                }
                let k_hmpc = two_pi_h_over_bsm * n_abs;
                let tk = transfer_eh_nowiggle(k_hmpc, &eh_params);
                // σ² = amp² · k^n_s · T² → σ = amp · k^(n_s/2) · T(k)
                // Dividir por sqrt(N³) para normalizar por volumen del grid.
                amp * k_hmpc.powf(spectral_index / 2.0) * tk * inv_sqrt_n3
            })
        }
    }
}

// ── Campo de desplazamiento ───────────────────────────────────────────────────

/// FFT 3D in-place sobre el array `buf` de tamaño `n³`.
/// Convención: tres pasadas de FFT 1D (ejes z, y, x).
pub fn fft3d(buf: &mut [Complex<f64>], n: usize, forward: bool) {
    let mut planner = FftPlanner::new();
    let plan = if forward {
        planner.plan_fft_forward(n)
    } else {
        planner.plan_fft_inverse(n)
    };

    // Eje Z: n×n filas de longitud n (contiguas en memoria).
    for row in buf.chunks_exact_mut(n) {
        plan.process(row);
    }

    // Eje Y: para cada (ix, iz), extraer columna y transformar.
    let mut tmp = vec![Complex::default(); n];
    for ix in 0..n {
        for iz in 0..n {
            for iy in 0..n {
                tmp[iy] = buf[ix * n * n + iy * n + iz];
            }
            plan.process(&mut tmp);
            for iy in 0..n {
                buf[ix * n * n + iy * n + iz] = tmp[iy];
            }
        }
    }

    // Eje X: para cada (iy, iz), extraer columna y transformar.
    for iy in 0..n {
        for iz in 0..n {
            for ix in 0..n {
                tmp[ix] = buf[ix * n * n + iy * n + iz];
            }
            plan.process(&mut tmp);
            for ix in 0..n {
                buf[ix * n * n + iy * n + iz] = tmp[ix];
            }
        }
    }
}

/// Computa las tres componentes del campo de desplazamiento `Ψ_α(q)` en unidades de `d`.
///
/// Dado `δ̂(k)` en k-space, calcula:
/// `Ψ̂_α(k) = i·n_α/|n|²·δ̂(k)`
///
/// y aplica IFFT 3D para obtener `Ψ_α(q)` en espacio real (en unidades de grid).
/// El resultado se multiplica por `d = box_size/N` para obtener desplazamientos físicos.
///
/// Devuelve `[Ψ_x, Ψ_y, Ψ_z]`, cada uno de longitud `n³`, en unidades de `box_size`.
pub fn delta_to_displacement(delta: &[Complex<f64>], n: usize, box_size: f64) -> [Vec<f64>; 3] {
    let n3 = n * n * n;
    let d = box_size / n as f64; // spacing físico

    // Factor de normalización de la IFFT (rustfft no normaliza la inversa).
    let ifft_norm = 1.0 / n3 as f64;

    let mut psi: [Vec<Complex<f64>>; 3] = [
        vec![Complex::new(0.0, 0.0); n3],
        vec![Complex::new(0.0, 0.0); n3],
        vec![Complex::new(0.0, 0.0); n3],
    ];

    // Construir Ψ̂_α(k) = i·n_α/|n|²·δ̂(k) para α ∈ {x,y,z}.
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

                let d_hat = delta[idx];

                // Multiplicar por i·n_α/n²:
                // i·(a+ib) = −b+ia   →   i·n_α·(a+ib)/n² = (−n_α·b + i·n_α·a) / n²
                let make_psi = |n_alpha: f64| -> Complex<f64> {
                    Complex::new(-n_alpha * d_hat.im / n2, n_alpha * d_hat.re / n2)
                };

                psi[0][idx] = make_psi(nx);
                psi[1][idx] = make_psi(ny);
                psi[2][idx] = make_psi(nz);
            }
        }
    }

    // IFFT 3D en cada componente → Ψ_α en espacio real (en unidades de grid).
    for component in &mut psi {
        fft3d(component, n, false);
    }

    // Extraer parte real y convertir a unidades físicas (× d).
    let scale = ifft_norm * d;
    [
        psi[0].iter().map(|c| c.re * scale).collect(),
        psi[1].iter().map(|c| c.re * scale).collect(),
        psi[2].iter().map(|c| c.re * scale).collect(),
    ]
}

// ── Entrada pública ───────────────────────────────────────────────────────────

/// Genera partículas con condiciones iniciales de Zel'dovich para el rango `[lo, hi)`.
///
/// El campo completo se genera en todos los rangos (serial) y luego se extrae
/// el subconjunto de partículas `[lo, hi)`. Esto garantiza reproducibilidad en MPI.
///
/// ## Parámetros
///
/// - `cfg`: configuración completa de la simulación.
/// - `n`: lado de la retícula (`particle_count = n³`).
/// - `seed`, `amplitude`, `spectral_index`: parámetros del espectro.
/// - `transfer`, `sigma8`, `omega_b`, `h`, `t_cmb`, `box_size_mpc_h`: parámetros
///   de la función de transferencia y σ₈ (Fase 27).
/// - `lo`, `hi`: rango de `global_id` a generar (medio abierto).
///
/// ## Unidades
///
/// - Posiciones: `[0, box_size)` (coordenadas comóviles).
/// - Velocidades: momentum canónico `p = a²·f(a)·H(a)·Ψ` (estilo GADGET-4).
///
/// Con `cosmology.enabled = false`, se usa `a=1`, `f=1`, `H=h0` como fallback.
#[allow(clippy::too_many_arguments)]
pub fn zeldovich_ics(
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
    let box_size = cfg.simulation.box_size;
    let n_part = cfg.simulation.particle_count;
    let mass = 1.0 / n_part as f64;

    // Parámetros cosmológicos para las velocidades.
    let (a_init, cosmo) = if cfg.cosmology.enabled {
        let a = cfg.cosmology.a_init;
        // Phase 156: incluir Ω_ν si m_nu_ev > 0
        let omega_nu = crate::cosmology::omega_nu_from_mass(
            cfg.cosmology.m_nu_ev, cfg.cosmology.h0 * 10.0,
        );
        let mut cp = CosmologyParams::new(
            cfg.cosmology.omega_m,
            cfg.cosmology.omega_lambda,
            cfg.cosmology.h0,
        );
        // Phase 155: parámetros CPL
        cp.w0 = cfg.cosmology.w0;
        cp.wa = cfg.cosmology.wa;
        cp.omega_nu = omega_nu;
        (a, cp)
    } else {
        let cp = CosmologyParams::new(1.0, 0.0, cfg.cosmology.h0);
        (1.0, cp)
    };

    // Phase 156: factor de supresión por neutrinos masivos
    let nu_suppression = if cfg.cosmology.m_nu_ev > 0.0 && cfg.cosmology.omega_m > 0.0 {
        let omega_nu = crate::cosmology::omega_nu_from_mass(
            cfg.cosmology.m_nu_ev, cfg.cosmology.h0 * 10.0,
        );
        let f_nu = omega_nu / cfg.cosmology.omega_m;
        crate::cosmology::neutrino_suppression(f_nu)
    } else {
        1.0
    };

    let h_a = hubble_param(cosmo, a_init);
    let f_a = growth_rate_f(cosmo, a_init);

    // Factor de velocidad: p = a²·f·H·Ψ
    let vel_factor = a_init * a_init * f_a * h_a;

    // Fase 37: factor de reescalado físico s = D(a_init)/D(1).
    // Cuando `rescale_to_a_init = false` (default) → s = 1 bit-idéntico al
    // comportamiento de Fase 26/27. Solo se activa cuando cosmología está
    // habilitada; en otros casos el factor carece de significado.
    let scale = if rescale_to_a_init && cfg.cosmology.enabled {
        growth_factor_d_ratio(cosmo, a_init, 1.0)
    } else {
        1.0
    };

    // Espaciado de la retícula.
    let d = box_size / n as f64;

    // ── Construir el closure de espectro según la configuración.
    let spectrum_fn_base = build_spectrum_fn(
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
    // Phase 156: aplicar supresión de neutrinos al espectro (σ → σ × sqrt(suppression))
    let spectrum_fn: Box<dyn Fn(f64) -> f64> = if nu_suppression < 1.0 {
        let sqrt_sup = nu_suppression.sqrt();
        Box::new(move |n_abs: f64| spectrum_fn_base(n_abs) * sqrt_sup)
    } else {
        spectrum_fn_base
    };

    // ── Generar campo en k-space (todos los rangos).
    let delta = generate_delta_kspace(n, seed, spectrum_fn);

    // ── Calcular campo de desplazamiento Ψ.
    let [mut psi_x, mut psi_y, mut psi_z] = delta_to_displacement(&delta, n, box_size);

    // Fase 37: aplicar reescalado físico opcional.
    if scale != 1.0 {
        for v in psi_x.iter_mut() {
            *v *= scale;
        }
        for v in psi_y.iter_mut() {
            *v *= scale;
        }
        for v in psi_z.iter_mut() {
            *v *= scale;
        }
    }

    // ── Construir partículas para el rango [lo, hi).
    let mut out = Vec::with_capacity(hi.saturating_sub(lo));

    for gid in lo..hi {
        if gid >= n_part {
            break;
        }

        // Coordenadas de retícula en orden lexicográfico (ix, iy, iz).
        let ix = gid / (n * n);
        let rem = gid % (n * n);
        let iy = rem / n;
        let iz = rem % n;

        // Posición de la retícula (centro de la celda).
        let q_x = (ix as f64 + 0.5) * d;
        let q_y = (iy as f64 + 0.5) * d;
        let q_z = (iz as f64 + 0.5) * d;

        let grid_idx = ix * n * n + iy * n + iz;
        let psi_vec = Vec3::new(psi_x[grid_idx], psi_y[grid_idx], psi_z[grid_idx]);

        // Posición desplazada, envuelta periódicamente.
        let x = Vec3::new(
            (q_x + psi_vec.x).rem_euclid(box_size),
            (q_y + psi_vec.y).rem_euclid(box_size),
            (q_z + psi_vec.z).rem_euclid(box_size),
        );

        // Momentum canónico (estilo GADGET-4).
        let p = psi_vec * vel_factor;

        out.push(Particle::new(gid, mass, x, p));
    }

    out
}

// ── Convenciones de momentum/velocidad para ICs (Phase 45) ────────────────────

/// Convención usada para poblar `Particle::velocity` en los ICs de Zel'dovich.
///
/// En LPT, la velocidad comóvil lineal es `dx_c/dt = f·H·Ψ` donde `Ψ` es el
/// campo de desplazamiento ya escalado por `D(a)`. Distintos integradores
/// esperan distintos múltiplos de `a` en el slot de "velocity" del struct
/// `Particle`:
///
/// | Variante           | `velocity` slot         | Drift consistente   |
/// |--------------------|-------------------------|---------------------|
/// | `DxDt`             | `f·H·Ψ` (= `dx_c/dt`)   | `x += v · dt`       |
/// | `ADxDt`            | `a·f·H·Ψ`               | `x += v · dt/a`     |
/// | `A2DxDt` / `GadgetCanonical` | `a²·f·H·Ψ` (= `p = a² ẋ_c`) | `x += v · dt/a²` |
///
/// Ante la convención del integrador `leapfrog_cosmo_kdk_step`
/// (`drift = ∫ dt'/a²`), la única consistente es `A2DxDt`, que coincide
/// con `GadgetCanonical`.
///
/// Se expone como enum para permitir **A/B tests de convenciones** en
/// Phase 45 (audit de unidades IC↔integrador) sin reescribir el solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcMomentumConvention {
    /// `velocity = f·H·Ψ` (velocidad comóvil pura `dx_c/dt`).
    DxDt,
    /// `velocity = a·f·H·Ψ` (peculiar velocity: `a · dx_c/dt`).
    ADxDt,
    /// `velocity = a²·f·H·Ψ` (momentum canónico GADGET-4: `p = a² · dx_c/dt`).
    /// **Convención por defecto, usada por `zeldovich_ics`.**
    A2DxDt,
    /// Alias explícito de `A2DxDt` consistente con la documentación GADGET.
    GadgetCanonical,
}

impl IcMomentumConvention {
    /// Factor multiplicador aplicado a `Ψ` para obtener `velocity`:
    /// `velocity = factor · Ψ` con `factor = {1, a, a²} · f · H`.
    #[inline]
    pub fn velocity_factor(self, a: f64, f: f64, h: f64) -> f64 {
        match self {
            Self::DxDt => f * h,
            Self::ADxDt => a * f * h,
            Self::A2DxDt | Self::GadgetCanonical => a * a * f * h,
        }
    }
}

/// Versión de `zeldovich_ics` que permite elegir explícitamente la
/// [`IcMomentumConvention`]. Utilizada en auditorías de unidades (Phase 45)
/// y en A/B tests.
///
/// Salvo por el factor de velocidad, el comportamiento es bit-idéntico a
/// `zeldovich_ics` con la misma configuración. En particular:
/// `zeldovich_ics(...) == zeldovich_ics_with_convention(..., IcMomentumConvention::A2DxDt)`.
#[allow(clippy::too_many_arguments)]
pub fn zeldovich_ics_with_convention(
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
    convention: IcMomentumConvention,
) -> Vec<Particle> {
    let box_size = cfg.simulation.box_size;
    let n_part = cfg.simulation.particle_count;
    let mass = 1.0 / n_part as f64;

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
    let f_a = growth_rate_f(cosmo, a_init);
    let vel_factor = convention.velocity_factor(a_init, f_a, h_a);

    let scale = if rescale_to_a_init && cfg.cosmology.enabled {
        growth_factor_d_ratio(cosmo, a_init, 1.0)
    } else {
        1.0
    };

    let d = box_size / n as f64;

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

    let delta = generate_delta_kspace(n, seed, spectrum_fn);
    let [mut psi_x, mut psi_y, mut psi_z] = delta_to_displacement(&delta, n, box_size);

    if scale != 1.0 {
        for v in psi_x.iter_mut() {
            *v *= scale;
        }
        for v in psi_y.iter_mut() {
            *v *= scale;
        }
        for v in psi_z.iter_mut() {
            *v *= scale;
        }
    }

    let mut out = Vec::with_capacity(hi.saturating_sub(lo));
    for gid in lo..hi {
        if gid >= n_part {
            break;
        }
        let ix = gid / (n * n);
        let rem = gid % (n * n);
        let iy = rem / n;
        let iz = rem % n;

        let q_x = (ix as f64 + 0.5) * d;
        let q_y = (iy as f64 + 0.5) * d;
        let q_z = (iz as f64 + 0.5) * d;

        let grid_idx = ix * n * n + iy * n + iz;
        let psi_vec = Vec3::new(psi_x[grid_idx], psi_y[grid_idx], psi_z[grid_idx]);

        let x = Vec3::new(
            (q_x + psi_vec.x).rem_euclid(box_size),
            (q_y + psi_vec.y).rem_euclid(box_size),
            (q_z + psi_vec.z).rem_euclid(box_size),
        );

        let p = psi_vec * vel_factor;
        out.push(Particle::new(gid, mass, x, p));
    }
    out
}

/// Versión conveniente para ICs de Fase 26 (ley de potencia sin parámetros E-H).
///
/// Wrapper compatible con la firma original de Fase 26; construye los parámetros
/// por defecto para los campos nuevos de Fase 27.
pub fn zeldovich_ics_power_law(
    cfg: &RunConfig,
    n: usize,
    seed: u64,
    amplitude: f64,
    spectral_index: f64,
    lo: usize,
    hi: usize,
) -> Vec<Particle> {
    zeldovich_ics(
        cfg,
        n,
        seed,
        amplitude,
        spectral_index,
        TransferKind::PowerLaw,
        None,
        0.049,
        0.674,
        2.7255,
        None,
        false, // rescale_to_a_init: mantener comportamiento legacy Fase 26
        lo,
        hi,
    )
}

// ── API interna para tests de normalización (Phase 34) ───────────────────────

/// Internals expuestos únicamente para tests de normalización (Phase 34).
///
/// **No forman parte de la API estable.** El único propósito de este
/// submódulo es permitir que los tests de caracterización decompongan el
/// pipeline (`δ̂(k) → IFFT → δ(x) → FFT → P(k)`, `δ̂(k) → Ψ → partículas ZA`)
/// sin duplicar las primitivas del generador de ICs ni arriesgar divergencia
/// con el código de producción.
///
/// El consumo desde fuera del crate (por ejemplo desde
/// `crates/gadget-ng-physics/tests/phase34_discrete_normalization.rs`) debe
/// declararse explícitamente como "uso de internals" y estar acompañado de
/// una referencia al reporte Phase 34.
pub mod internals {
    pub use super::{
        build_spectrum_fn, delta_to_displacement, fft3d, generate_delta_kspace, mode_int,
    };
}

// ── Tests unitarios ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Construye closure de ley de potencia simple para tests.
    fn power_law_fn(n: usize, amplitude: f64, spectral_index: f64) -> impl Fn(f64) -> f64 {
        let inv_sqrt_n3 = 1.0 / ((n * n * n) as f64).sqrt();
        move |n_abs: f64| {
            if n_abs <= 0.0 {
                0.0
            } else {
                amplitude * n_abs.powf(spectral_index / 4.0) * inv_sqrt_n3
            }
        }
    }

    /// Verifica que el modo DC es exactamente cero en el campo generado.
    #[test]
    fn delta_dc_mode_is_zero() {
        let n = 8;
        let delta = generate_delta_kspace(n, 42, power_law_fn(n, 1e-4, -2.0));
        // Índice del modo k=0 es ix=0, iy=0, iz=0.
        assert_eq!(
            delta[0],
            Complex::new(0.0, 0.0),
            "Modo DC δ̂(0) debe ser exactamente cero"
        );
    }

    /// Verifica que el campo satisface simetría Hermitiana: δ̂(−k) = conj(δ̂(k)).
    #[test]
    fn delta_hermitian_symmetry() {
        let n = 8;
        let delta = generate_delta_kspace(n, 99, power_law_fn(n, 1e-4, -2.0));

        let mut max_err = 0.0_f64;
        for ix in 0..n {
            let nx = mode_int(ix, n);
            for iy in 0..n {
                let ny = mode_int(iy, n);
                for iz in 0..n {
                    let nz = mode_int(iz, n);

                    let half = (n / 2) as i64;
                    if nx.abs() == half || ny.abs() == half || nz.abs() == half {
                        continue;
                    }

                    let ix_neg = ((-nx).rem_euclid(n as i64)) as usize;
                    let iy_neg = ((-ny).rem_euclid(n as i64)) as usize;
                    let iz_neg = ((-nz).rem_euclid(n as i64)) as usize;

                    let val = delta[ix * n * n + iy * n + iz];
                    let conj = delta[ix_neg * n * n + iy_neg * n + iz_neg];
                    let err = (val.re - conj.re).abs() + (val.im + conj.im).abs();
                    max_err = max_err.max(err);
                }
            }
        }
        assert!(
            max_err < 1e-14,
            "Simetría Hermitiana violada: error máximo = {:.2e}",
            max_err
        );
    }

    /// Verifica que la IFFT del campo Hermitiano produce valores reales.
    #[test]
    fn displacement_field_is_real() {
        let n = 8;
        let box_size = 1.0;
        let delta = generate_delta_kspace(n, 7, power_law_fn(n, 1e-4, -2.0));

        let mut psi_x = delta.clone();
        // La parte imaginaria del resultado IFFT debe ser ~0.
        fft3d(&mut psi_x, n, false);
        let max_im: f64 = psi_x.iter().map(|c| c.im.abs()).fold(0.0, f64::max);
        assert!(
            max_im < 1e-10,
            "Parte imaginaria de Ψ no es ~0: max|Im| = {:.2e}",
            max_im
        );
        let _ = box_size; // suprime warning
    }

    /// El desplazamiento medio debe ser ~0 (modo DC nulo).
    #[test]
    fn displacement_mean_near_zero() {
        let n = 16;
        let box_size = 1.0;
        let delta = generate_delta_kspace(n, 1234, power_law_fn(n, 1e-4, -2.0));
        let [px, py, pz] = delta_to_displacement(&delta, n, box_size);

        let mean_x: f64 = px.iter().sum::<f64>() / px.len() as f64;
        let mean_y: f64 = py.iter().sum::<f64>() / py.len() as f64;
        let mean_z: f64 = pz.iter().sum::<f64>() / pz.len() as f64;

        let tol = 1e-12;
        assert!(mean_x.abs() < tol, "⟨Ψ_x⟩ = {:.2e} ≠ 0", mean_x);
        assert!(mean_y.abs() < tol, "⟨Ψ_y⟩ = {:.2e} ≠ 0", mean_y);
        assert!(mean_z.abs() < tol, "⟨Ψ_z⟩ = {:.2e} ≠ 0", mean_z);
    }
}
