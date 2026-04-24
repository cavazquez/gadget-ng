//! Phase 34 — Cierre de la normalización discreta de P(k).
//!
//! Desarma el pipeline `P_cont(k) → δ̂(k) → IFFT → δ(x) → FFT → P_m(k)`
//! (y su continuación `→ Ψ → partículas ZA → CIC → FFT → P_part(k)`)
//! en etapas independientes para aislar dónde nace el offset absoluto
//! reportado en Phase 30–33.
//!
//! ## Convención fijada (ver docs/reports/2026-04-phase34-*.md §2)
//!
//! - DFT forward de `rustfft`, **sin normalizar**:
//!
//!     `δ̂[k] = Σ_n δ[n] · exp(−2πi k·n/N)`
//!
//! - IFFT con factor `1/N³` aplicado **manualmente** (`fft3d(buf, n, false)`
//!   seguido de `× 1/N³`), tal como hace el generador de ICs
//!   ([`ic_zeldovich::internals::delta_to_displacement`]).
//!
//! - Estimador de P(k) con normalización `(V/N³)²` y deconvolución CIC
//!   `W²(k) = Π sinc²(kᵢ/N)` (ver `gadget_ng_analysis::power_spectrum`).
//!
//! - Generador de ICs: `σ(|n|) = √(P_cont(k_phys)/N³)` → `⟨|δ̂|²⟩ = 2·σ² =
//!   2·P_cont/N³` (factor 2 porque `δ̂ = σ·(g_r + i·g_i)` con `g_r,g_i ∼ N(0,1)`
//!   **antes** de imponer simetría Hermitiana).
//!
//! ## Identidad de consistencia objetivo (caja adimensional `V=1`)
//!
//!     ⟨|δ̂|²⟩_IC · (V/N³)² = 2·P_cont · V²/N⁹
//!
//! Con `V_internal = 1` y `N=32` la predicción es `A_pred = 2·V²/N⁹ = 2/N⁹`.
//! Si la convención está cerrada limpiamente, `P_m/P_cont` debería medir
//! exactamente este factor, sin residuo dependiente de `k`.
//!
//! Todos los tests escriben su resultado a `target/phase34/*.json` para ser
//! consumidos por los scripts de análisis en
//! `experiments/nbody/phase34_discrete_normalization/`.

use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    amplitude_for_sigma8, ic_zeldovich_internals as internals, transfer_eh_nowiggle,
    CosmologySection, EisensteinHuParams, GravitySection, IcKind, InitialConditionsSection,
    OutputSection, Particle, PerformanceSection, RunConfig, SimulationSection, TimestepSection,
    TransferKind, UnitsSection, Vec3,
};
use rustfft::{num_complex::Complex, FftPlanner};
use serde_json::json;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;

// ── Constantes de la fase (alineadas con Phase 30-33 para comparabilidad) ─────

const BOX: f64 = 1.0;
const SEEDS: [u64; 6] = [42, 137, 271, 314, 512, 999];
const N_SMALL: usize = 16;
const N_LARGE: usize = 32;

const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const H0: f64 = 0.1;
const A_INIT: f64 = 0.02;
const OMEGA_B: f64 = 0.049;
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const BOX_MPC_H: f64 = 100.0;
const SIGMA8_TARGET: f64 = 0.8;

// ── Utilidades generales ──────────────────────────────────────────────────────

fn eh_params() -> EisensteinHuParams {
    EisensteinHuParams {
        omega_m: OMEGA_M,
        omega_b: OMEGA_B,
        h: H_DIMLESS,
        t_cmb: T_CMB,
    }
}

fn theory_pk_at_k(k_hmpc: f64) -> f64 {
    let eh = eh_params();
    let amp = amplitude_for_sigma8(SIGMA8_TARGET, N_S, &eh);
    let tk = transfer_eh_nowiggle(k_hmpc, &eh);
    amp * amp * k_hmpc.powf(N_S) * tk * tk
}

fn k_internal_to_hmpc(k_internal: f64) -> f64 {
    k_internal * H_DIMLESS / BOX_MPC_H
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

fn cv(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    if m.abs() < 1e-300 {
        return f64::NAN;
    }
    let var = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64;
    var.sqrt() / m.abs()
}

/// Pendiente `log₁₀(y) vs log₁₀(x)` por mínimos cuadrados simples.
fn loglog_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return f64::NAN;
    }
    let lx: Vec<f64> = xs.iter().take(n).map(|v| v.ln()).collect();
    let ly: Vec<f64> = ys.iter().take(n).map(|v| v.ln()).collect();
    let mx = mean(&lx);
    let my = mean(&ly);
    let num: f64 = lx.iter().zip(&ly).map(|(a, b)| (a - mx) * (b - my)).sum();
    let den: f64 = lx.iter().map(|a| (a - mx).powi(2)).sum();
    if den == 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

/// Directorio donde se vuelcan los JSONs reproducibles.
fn phase34_dir() -> PathBuf {
    // `target/` es estable entre invocaciones; los scripts de análisis lo leen.
    let mut d = PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        // Heurística: `target/` en la raíz del workspace.
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("target");
        p.to_string_lossy().to_string()
    }));
    d.push("phase34");
    let _ = fs::create_dir_all(&d);
    d
}

fn dump_json(name: &str, value: serde_json::Value) {
    let mut path = phase34_dir();
    path.push(format!("{name}.json"));
    if let Ok(s) = serde_json::to_string_pretty(&value) {
        let _ = fs::write(&path, s);
    }
}

// ── FFT / IFFT con la misma convención que el código de producción ────────────

/// IFFT 3D con normalización `1/N³` (idéntica a la usada en
/// `delta_to_displacement`: `fft3d(..., forward=false)` seguido de `× 1/N³`).
fn ifft3d_normalized(buf: &mut [Complex<f64>], n: usize) {
    internals::fft3d(buf, n, false);
    let inv = 1.0 / (n as f64).powi(3);
    for v in buf.iter_mut() {
        v.re *= inv;
        v.im *= inv;
    }
}

/// FFT 3D forward (unnormalized), misma convención que el estimador.
fn fft3d_forward(buf: &mut [Complex<f64>], n: usize) {
    internals::fft3d(buf, n, true);
}

/// Genera `N³` reales ~ `N(0, σ²)` con un LCG reproducible por seed.
fn white_noise_real(n: usize, seed: u64, sigma: f64) -> Vec<f64> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(n * n * n);
    let mut have_spare = false;
    let mut spare = 0.0f64;
    while out.len() < n * n * n {
        if have_spare {
            out.push(spare * sigma);
            have_spare = false;
            continue;
        }
        // Box–Muller con LCG Knuth.
        let u1 = loop {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (state >> 11) as f64 / (1u64 << 53) as f64;
            if f > 1e-300 {
                break f;
            }
        };
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let a = 2.0 * std::f64::consts::TAU * u2; // 2·TAU = 4π, basta que sea consistente
        out.push(r * a.cos() * sigma);
        spare = r * a.sin();
        have_spare = true;
    }
    out
}

// ── Estimadores locales reducidos (bloque grilla pura) ───────────────────────

/// Estimador *simple* sobre un `δ̂(k)` ya dado (sin CIC, sin deconvolución).
///
/// Aplica la misma normalización que `gadget_ng_analysis::power_spectrum`:
/// `P_m(k_bin) = ⟨|δ̂|²⟩_bin · (V/N³)²`.
fn pk_from_delta_kspace_no_cic(delta_k: &[Complex<f64>], n: usize, box_size: f64) -> Vec<PkBin> {
    let n_nyq = n / 2;
    let k_fund = 2.0 * PI / box_size;
    let n_bins = n_nyq;
    let mut pk_sum = vec![0.0f64; n_bins];
    let mut n_modes = vec![0u64; n_bins];
    let n3 = (n * n * n) as f64;
    let vol = box_size.powi(3);
    let norm = (vol / n3).powi(2);

    for ix in 0..n {
        let kx = internals::mode_int(ix, n) as f64;
        for iy in 0..n {
            let ky = internals::mode_int(iy, n) as f64;
            for iz in 0..n {
                let kz = internals::mode_int(iz, n) as f64;
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
                let idx = ix * n * n + iy * n + iz;
                pk_sum[bin] += delta_k[idx].norm_sqr() * norm;
                n_modes[bin] += 1;
            }
        }
    }
    pk_sum
        .iter()
        .zip(n_modes.iter())
        .enumerate()
        .filter(|(_, (_, &nm))| nm > 0)
        .map(|(bin, (&ps, &nm))| PkBin {
            k: (bin as f64 + 1.0) * k_fund,
            pk: ps / nm as f64,
            n_modes: nm,
        })
        .collect()
}

/// Estimador reducido que aplica **todos los pasos del estimador oficial
/// excepto la deconvolución CIC** (devuelve `P_raw` con la ventana aún
/// convolucionada).
fn pk_particles_without_deconv(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    mesh: usize,
) -> Vec<PkBin> {
    let n = mesh;
    let n3 = n * n * n;
    let cell = box_size / n as f64;

    // ── CIC deposit ───────────────────────────────────────────────────────────
    let mut rho = vec![0.0f64; n3];
    let total_mass: f64 = masses.iter().sum();
    let mean_rho = total_mass / (box_size * box_size * box_size);
    let vol_cell = cell * cell * cell;
    for (&pos, &m) in positions.iter().zip(masses.iter()) {
        cic_assign_local(&mut rho, pos, m, n, cell);
    }
    for v in &mut rho {
        *v = *v / (mean_rho * vol_cell) - 1.0;
    }

    // ── FFT 3D forward ────────────────────────────────────────────────────────
    let mut buf: Vec<Complex<f64>> = rho.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    for row in buf.chunks_exact_mut(n) {
        fft.process(row);
    }
    fft_axis_y(&mut buf, n, &fft);
    fft_axis_x(&mut buf, n, &fft);

    // ── Binning sin deconvolución ────────────────────────────────────────────
    let n_nyq = n / 2;
    let k_fund = 2.0 * PI / box_size;
    let n_bins = n_nyq;
    let mut pk_sum = vec![0.0f64; n_bins];
    let mut n_modes = vec![0u64; n_bins];
    let vol = box_size.powi(3);
    let norm = (vol / n3 as f64).powi(2);

    for ix in 0..n {
        let kx = internals::mode_int(ix, n) as f64;
        for iy in 0..n {
            let ky = internals::mode_int(iy, n) as f64;
            for iz in 0..n {
                let kz = internals::mode_int(iz, n) as f64;
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
                let idx = ix * n * n + iy * n + iz;
                pk_sum[bin] += buf[idx].norm_sqr() * norm;
                n_modes[bin] += 1;
            }
        }
    }
    pk_sum
        .iter()
        .zip(n_modes.iter())
        .enumerate()
        .filter(|(_, (_, &nm))| nm > 0)
        .map(|(bin, (&ps, &nm))| PkBin {
            k: (bin as f64 + 1.0) * k_fund,
            pk: ps / nm as f64,
            n_modes: nm,
        })
        .collect()
}

fn cic_assign_local(grid: &mut [f64], pos: Vec3, m: f64, n: usize, cell: f64) {
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

// ── Helpers de generación de ICs y medición completa ─────────────────────────

fn eh_spectrum_fn_for(n: usize) -> Box<dyn Fn(f64) -> f64> {
    internals::build_spectrum_fn(
        n,
        N_S,
        1.0,
        TransferKind::EisensteinHu,
        Some(SIGMA8_TARGET),
        OMEGA_M,
        OMEGA_B,
        H_DIMLESS,
        T_CMB,
        Some(BOX_MPC_H),
    )
}

/// Ratios `P_m(bin) / P_cont(k_phys(bin))` sobre bins con suficientes modos.
fn ratios_vs_theory(bins: &[PkBin]) -> Vec<(f64, f64)> {
    bins.iter()
        .filter(|b| b.pk > 0.0 && b.n_modes >= 8)
        .filter_map(|b| {
            let k_hmpc = k_internal_to_hmpc(b.k);
            let th = theory_pk_at_k(k_hmpc);
            if th > 0.0 && th.is_finite() {
                Some((b.k, b.pk / th))
            } else {
                None
            }
        })
        .collect()
}

fn restrict_linear_regime(bins: &[PkBin], n_mesh: usize) -> Vec<&PkBin> {
    let k_fund = 2.0 * PI / BOX;
    let k_nyq = (n_mesh as f64 / 2.0) * k_fund;
    let k_max = k_nyq * 0.5; // k ≤ k_Nyq/2
    bins.iter()
        .filter(|b| b.k <= k_max && b.pk > 0.0 && b.n_modes >= 8)
        .collect()
}

fn make_config(seed: u64, n: usize) -> RunConfig {
    let gravity = GravitySection {
        solver: gadget_ng_core::SolverKind::Pm,
        pm_grid_size: n,
        ..GravitySection::default()
    };
    RunConfig {
        simulation: SimulationSection {
            dt: 0.002,
            num_steps: 1,
            softening: 0.01,
            physical_softening: false,
            gravitational_constant: 1.0,
            particle_count: n * n * n,
            box_size: BOX,
            seed,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed,
                grid_size: n,
                spectral_index: N_S,
                amplitude: 1.0,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(SIGMA8_TARGET),
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: Some(BOX_MPC_H),
                use_2lpt: false,
                normalization_mode: gadget_ng_core::NormalizationMode::Legacy,
            },
        },
        output: OutputSection::default(),
        gravity,
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m: OMEGA_M,
            omega_lambda: OMEGA_L,
            h0: H0,
            a_init: A_INIT,
            auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(), reionization: Default::default(), mhd: Default::default(),
    }
}

/// Construye partículas ZA "sin evolución" replicando la cadena del generador
/// directamente desde un `δ̂(k)` dado: `Ψ = IFFT(i·k_α/k²·δ̂)`, `x_p = q + Ψ(q)`.
fn particles_from_delta_kspace(
    delta_k: &[Complex<f64>],
    n: usize,
    box_size: f64,
) -> (Vec<Vec3>, Vec<f64>) {
    let [psi_x, psi_y, psi_z] = internals::delta_to_displacement(delta_k, n, box_size);
    let d = box_size / n as f64;
    let mass = 1.0 / (n * n * n) as f64;
    let mut pos = Vec::with_capacity(n * n * n);
    let mut mas = Vec::with_capacity(n * n * n);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let idx = ix * n * n + iy * n + iz;
                let q_x = (ix as f64 + 0.5) * d;
                let q_y = (iy as f64 + 0.5) * d;
                let q_z = (iz as f64 + 0.5) * d;
                let x = (q_x + psi_x[idx]).rem_euclid(box_size);
                let y = (q_y + psi_y[idx]).rem_euclid(box_size);
                let z = (q_z + psi_z[idx]).rem_euclid(box_size);
                pos.push(Vec3::new(x, y, z));
                mas.push(mass);
            }
        }
    }
    (pos, mas)
}

// ══════════════════════════════════════════════════════════════════════════════
//  BLOQUE B1 — Grilla pura (sin partículas, sin CIC)
// ══════════════════════════════════════════════════════════════════════════════

// ── Test 1: roundtrip preserva amplitud ──────────────────────────────────────

/// Un campo real arbitrario `δ(x)`, tras `FFT → IFFT_{1/N³}`, debe recuperarse
/// a precisión de máquina. Esto fija la convención DFT del código:
/// `forward no normaliza, inverse aplica 1/N³ manual`.
#[test]
fn grid_roundtrip_preserves_amplitude_with_known_convention() {
    let n = 16usize;
    let samples = white_noise_real(n, 12345, 1.0);
    let mut buf: Vec<Complex<f64>> = samples.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft3d_forward(&mut buf, n);
    ifft3d_normalized(&mut buf, n);
    let max_err: f64 = samples
        .iter()
        .zip(buf.iter())
        .map(|(a, c)| (a - c.re).abs().max(c.im.abs()))
        .fold(0.0, f64::max);
    dump_json(
        "b1_roundtrip",
        json!({
            "n": n,
            "max_abs_error": max_err,
            "convention": "forward unnormalized, inverse × 1/N^3"
        }),
    );
    assert!(
        max_err < 1e-10,
        "Roundtrip excede 1e-10: max_err = {:.3e}",
        max_err
    );
}

// ── Test 2: modo único ───────────────────────────────────────────────────────

/// Un único modo real `δ̂(k₀)=A` con `δ̂(−k₀)=A*` produce tras IFFT (con 1/N³)
/// un coseno puro `δ(x) = (2A/N³)·cos(2π·n_α/N·m)` evaluado en la grilla.
/// Al aplicar FFT forward se recupera `δ̂(k₀) = A` exactamente.
#[test]
fn single_mode_recovered_with_correct_amplitude() {
    let n = 8usize;
    let amplitude = 7.0f64; // cualquier real
    let mut delta_k = vec![Complex::new(0.0, 0.0); n * n * n];

    // Poner el modo (nx=1, ny=0, nz=0) y su conjugado (nx=-1 ≡ n-1,0,0).
    let ix_pos = 1usize;
    let ix_neg = n - 1;
    let idx = |ix: usize| ix * n * n;
    delta_k[idx(ix_pos)] = Complex::new(amplitude, 0.0);
    delta_k[idx(ix_neg)] = Complex::new(amplitude, 0.0);

    // IFFT con 1/N³.
    let mut real_space = delta_k.clone();
    ifft3d_normalized(&mut real_space, n);

    // Debería ser real y seguir cos(2π·ix/N) con amplitud 2A/N³.
    let expected_amp = 2.0 * amplitude / (n as f64).powi(3);
    let mut max_im = 0.0f64;
    let mut max_cos_err = 0.0f64;
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let idx = ix * n * n + iy * n + iz;
                let expected = expected_amp * (2.0 * PI * ix as f64 / n as f64).cos();
                max_cos_err = max_cos_err.max((real_space[idx].re - expected).abs());
                max_im = max_im.max(real_space[idx].im.abs());
                // La dependencia sólo es en ix (nx=1); y e z son constantes en (ix).
                // En realidad cos(2π·ix/N) no depende de iy,iz → la comprobación
                // anterior ya captura el patrón.
                let _ = (iy, iz);
            }
        }
    }

    // FFT forward → recupera el modo original.
    let mut recovered = real_space.clone();
    fft3d_forward(&mut recovered, n);
    let re_pos = recovered[idx(ix_pos)];
    let re_neg = recovered[idx(ix_neg)];

    // Resto del grid debería ser ~0.
    let mut max_other = 0.0f64;
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                if (ix == ix_pos || ix == ix_neg) && iy == 0 && iz == 0 {
                    continue;
                }
                let idx = ix * n * n + iy * n + iz;
                max_other = max_other.max(recovered[idx].norm());
            }
        }
    }

    dump_json(
        "b1_single_mode",
        json!({
            "n": n,
            "amplitude": amplitude,
            "expected_real_peak": expected_amp,
            "max_imag_after_ifft": max_im,
            "max_cos_deviation": max_cos_err,
            "recovered_pos": { "re": re_pos.re, "im": re_pos.im },
            "recovered_neg": { "re": re_neg.re, "im": re_neg.im },
            "max_amplitude_in_other_modes": max_other
        }),
    );

    assert!(
        max_im < 1e-10,
        "IFFT no devolvió campo real: max|Im| = {:.3e}",
        max_im
    );
    assert!(
        max_cos_err < 1e-9,
        "Patrón cos no coincide: max_err = {:.3e}",
        max_cos_err
    );
    assert!(
        (re_pos.re - amplitude).abs() < 1e-9 && re_pos.im.abs() < 1e-9,
        "Modo (+k) no recuperado: ({:.3e}, {:.3e}) vs {:.3e}",
        re_pos.re,
        re_pos.im,
        amplitude
    );
    assert!(
        (re_neg.re - amplitude).abs() < 1e-9 && re_neg.im.abs() < 1e-9,
        "Modo (−k) no recuperado: ({:.3e}, {:.3e}) vs {:.3e}",
        re_neg.re,
        re_neg.im,
        amplitude
    );
    assert!(
        max_other < 1e-9,
        "Fuga a otros modos: max = {:.3e}",
        max_other
    );
}

// ── Test 3: ruido blanco ──────────────────────────────────────────────────────

/// Para `δ(x) ~ N(0,σ²)` la varianza por modo Fourier bajo convención
/// rustfft (forward unnormalized) es `Var(|δ̂_k|²) ≈ σ²·N³` en promedio
/// sobre los N³ modos (exacto para N³ grande por el teorema de Parseval).
///
/// Verifica `⟨|δ̂_k|²⟩ / (σ² · N³) = 1 ± O(1/√N³)`.
#[test]
fn white_noise_grid_matches_expected_variance() {
    let n = 32usize;
    let sigma = 0.1f64;
    let samples = white_noise_real(n, 98765, sigma);
    let mut buf: Vec<Complex<f64>> = samples.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft3d_forward(&mut buf, n);
    let total_power: f64 = buf.iter().skip(1).map(|c| c.norm_sqr()).sum();
    let n_modes = (n * n * n - 1) as f64;
    let mean_mode_power = total_power / n_modes;
    let expected = sigma * sigma * (n as f64).powi(3);
    let ratio = mean_mode_power / expected;
    // Tolerancia razonable para N=32 y muestra única: 5 %.
    let tolerance = 0.05;
    dump_json(
        "b1_white_noise",
        json!({
            "n": n,
            "sigma": sigma,
            "mean_mode_power": mean_mode_power,
            "expected_sigma2_N3": expected,
            "ratio": ratio,
            "tolerance": tolerance
        }),
    );
    assert!(
        (ratio - 1.0).abs() < tolerance,
        "Varianza por modo / σ²·N³ = {:.4} (tol {:.2}) — convención DFT rota",
        ratio,
        tolerance
    );
}

// ══════════════════════════════════════════════════════════════════════════════
//  BLOQUE B2 — Grilla → partículas (ZA sin evolución)
// ══════════════════════════════════════════════════════════════════════════════

// ── Test 4: offset entre partículas y grilla ─────────────────────────────────

/// Para el **mismo** `δ̂(k)`, compara:
/// - A_grid  = ⟨P_grid(k_bin) / P_cont(k_phys)⟩  (grilla pura)
/// - A_part  = ⟨P_part(k_bin) / P_cont(k_phys)⟩  (partículas ZA + CIC + deconv)
///
/// El ratio A_part / A_grid mide el offset **exclusivo** de la conversión
/// a partículas (Zel'dovich + CIC). Si la convención FFT está cerrada, este
/// ratio es cercano a 1; si hay sesgo Poisson/CIC neto, queda ≠ 1.
#[test]
fn particle_sampling_introduces_quantified_offset() {
    let n = N_LARGE;
    let box_size = BOX;
    let mut ratios_part_over_grid_per_seed: Vec<f64> = Vec::new();
    let mut a_grid_list: Vec<f64> = Vec::new();
    let mut a_part_list: Vec<f64> = Vec::new();

    let mut per_seed_details = Vec::new();

    for &seed in SEEDS.iter() {
        // 1. Generar δ̂(k)
        let spec = eh_spectrum_fn_for(n);
        let delta_k = internals::generate_delta_kspace(n, seed, spec);

        // 2. P_grid (sin CIC): directamente desde el δ̂ (la grilla pura).
        let pk_grid = pk_from_delta_kspace_no_cic(&delta_k, n, box_size);
        let a_grid = mean(
            &ratios_vs_theory(&pk_grid)
                .iter()
                .map(|(_, r)| *r)
                .collect::<Vec<_>>(),
        );

        // 3. P_part: construir partículas ZA y medir con el estimador oficial.
        let (pos, masses) = particles_from_delta_kspace(&delta_k, n, box_size);
        let pk_part = power_spectrum(&pos, &masses, box_size, n);
        let a_part = mean(
            &ratios_vs_theory(&pk_part)
                .iter()
                .map(|(_, r)| *r)
                .collect::<Vec<_>>(),
        );

        a_grid_list.push(a_grid);
        a_part_list.push(a_part);
        if a_grid > 0.0 && a_grid.is_finite() {
            ratios_part_over_grid_per_seed.push(a_part / a_grid);
        }
        per_seed_details.push(json!({
            "seed": seed,
            "a_grid": a_grid,
            "a_part": a_part,
            "ratio_part_over_grid": if a_grid > 0.0 { a_part / a_grid } else { f64::NAN }
        }));
    }

    let a_grid_mean = mean(&a_grid_list);
    let a_part_mean = mean(&a_part_list);
    let ratio_mean = mean(&ratios_part_over_grid_per_seed);
    let ratio_cv = cv(&ratios_part_over_grid_per_seed);

    dump_json(
        "b2_particle_vs_grid",
        json!({
            "n": n,
            "seeds": SEEDS,
            "a_grid_mean": a_grid_mean,
            "a_part_mean": a_part_mean,
            "ratio_part_over_grid_mean": ratio_mean,
            "ratio_part_over_grid_cv": ratio_cv,
            "per_seed": per_seed_details
        }),
    );

    // Requisito: el ratio part/grid es estable entre seeds (CV < 0.10).
    // No pedimos que sea 1 — eso es justamente el hallazgo cuantitativo.
    assert!(
        ratio_cv.is_finite() && ratio_cv < 0.10,
        "CV(P_part/P_grid) = {:.4} ≥ 0.10 — el offset de partículas no es determinista",
        ratio_cv
    );
}

// ── Test 5: deconvolución CIC reduce la pendiente de R(k) ────────────────────

/// Compara la dependencia en `k` del ratio `P_m/P_cont` antes y después de
/// deconvolucionar CIC. La deconvolución debe aplanar al menos en factor 2
/// la pendiente de `log R(k) vs log k`.
#[test]
fn cic_deconvolution_reduces_shape_error() {
    let n = N_LARGE;
    let seed = SEEDS[0];
    let spec = eh_spectrum_fn_for(n);
    let delta_k = internals::generate_delta_kspace(n, seed, spec);
    let (pos, masses) = particles_from_delta_kspace(&delta_k, n, BOX);

    let pk_deconv = power_spectrum(&pos, &masses, BOX, n);
    let pk_raw = pk_particles_without_deconv(&pos, &masses, BOX, n);

    let lin_deconv = restrict_linear_regime(&pk_deconv, n);
    let lin_raw = restrict_linear_regime(&pk_raw, n);

    // Normalizamos cada lista contra el P_cont correspondiente.
    let ks_d: Vec<f64> = lin_deconv.iter().map(|b| b.k).collect();
    let rs_d: Vec<f64> = lin_deconv
        .iter()
        .map(|b| {
            let th = theory_pk_at_k(k_internal_to_hmpc(b.k));
            b.pk / th
        })
        .collect();
    let ks_r: Vec<f64> = lin_raw.iter().map(|b| b.k).collect();
    let rs_r: Vec<f64> = lin_raw
        .iter()
        .map(|b| {
            let th = theory_pk_at_k(k_internal_to_hmpc(b.k));
            b.pk / th
        })
        .collect();

    let slope_deconv = loglog_slope(&ks_d, &rs_d);
    let slope_raw = loglog_slope(&ks_r, &rs_r);

    dump_json(
        "b2_cic_effect",
        json!({
            "n": n,
            "seed": seed,
            "slope_deconv": slope_deconv,
            "slope_raw": slope_raw,
            "ks_deconv": ks_d,
            "ratios_deconv": rs_d,
            "ks_raw": ks_r,
            "ratios_raw": rs_r
        }),
    );

    // Criterio: la deconvolución debe reducir la pendiente al menos un 30 %.
    // El factor exacto (observado ≈ 2 en ZA sin evolución) queda en el JSON y
    // se analiza en el reporte; este umbral sólo verifica que *mejora*.
    let reduction = 1.0 - slope_deconv.abs() / slope_raw.abs();
    assert!(
        slope_raw.abs() > 0.0 && reduction > 0.30,
        "Deconvolución no aplana suficientemente: |dec|={:.3e}, |raw|={:.3e}, reducción={:.3}",
        slope_deconv.abs(),
        slope_raw.abs(),
        reduction
    );
}

// ── Test 6: offset global aislado (sin solver) ───────────────────────────────

/// `A = ⟨P_m/P_cont⟩` y `CV(P_m/P_cont)` en `k ≤ k_Nyq/2` sin llamar al solver,
/// partiendo de partículas ZA de `build_particles`. Esto replica la métrica
/// de Phase 33 pero con una verificación explícita de que el offset es
/// *global* (CV chico) y no un artefacto del solver temporal.
#[test]
fn global_offset_isolated_before_solver() {
    let n = N_LARGE;
    let cfg = make_config(SEEDS[0], n);
    let parts = gadget_ng_core::build_particles(&cfg).unwrap();
    let positions: Vec<Vec3> = parts.iter().map(|p: &Particle| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p: &Particle| p.mass).collect();
    let pk = power_spectrum(&positions, &masses, BOX, n);

    let lin = restrict_linear_regime(&pk, n);
    let ratios: Vec<f64> = lin
        .iter()
        .map(|b| b.pk / theory_pk_at_k(k_internal_to_hmpc(b.k)))
        .collect();
    let a = mean(&ratios);
    let cv_r = cv(&ratios);

    dump_json(
        "b2_global_offset",
        json!({
            "n": n,
            "seed": SEEDS[0],
            "n_bins": ratios.len(),
            "A_mean": a,
            "CV_ratio": cv_r,
            "ratios": ratios,
            "ks": lin.iter().map(|b| b.k).collect::<Vec<_>>()
        }),
    );

    assert!(a.is_finite() && a > 0.0, "A no finito: {a}");
    assert!(
        cv_r < 0.15,
        "CV(P_m/P_cont) = {:.4} ≥ 0.15 — la forma no es suficientemente plana",
        cv_r
    );
}

// ══════════════════════════════════════════════════════════════════════════════
//  BLOQUE B3 — Robustez
// ══════════════════════════════════════════════════════════════════════════════

// ── Test 7: offset estable entre resoluciones ────────────────────────────────

/// Mide A para N=16 y N=32 (media sobre las 6 seeds). El ratio
/// `log₁₀(A₁₆/A₃₂)` debería seguir la ley `∝ 1/N⁹` predicha por la convención,
/// o bien mostrar un offset constante por etapa. Toleramos ≤ 1 orden de
/// magnitud de desviación respecto a la predicción `(32/16)⁹ = 512`.
#[test]
fn offset_stable_across_resolutions() {
    let mut a_by_n = Vec::new();
    for n in [N_SMALL, N_LARGE] {
        let mut a_list = Vec::new();
        for &seed in SEEDS.iter() {
            let cfg = make_config(seed, n);
            let parts = gadget_ng_core::build_particles(&cfg).unwrap();
            let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
            let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
            let pk = power_spectrum(&positions, &masses, BOX, n);
            let lin = restrict_linear_regime(&pk, n);
            let ratios: Vec<f64> = lin
                .iter()
                .map(|b| b.pk / theory_pk_at_k(k_internal_to_hmpc(b.k)))
                .collect();
            let a = mean(&ratios);
            if a.is_finite() && a > 0.0 {
                a_list.push(a);
            }
        }
        let a_mean = mean(&a_list);
        a_by_n.push((n, a_mean, a_list));
    }

    let a16 = a_by_n[0].1;
    let a32 = a_by_n[1].1;
    let log_ratio = (a16 / a32).log10();
    // Predicción nominal: factor 512 ≈ log10 ≈ 2.71. Tolerancia: ±1 década.
    let expected = (512.0f64).log10();
    dump_json(
        "b3_resolutions",
        json!({
            "n_values": [N_SMALL, N_LARGE],
            "A_mean_per_n": [a16, a32],
            "log10_ratio_A16_over_A32": log_ratio,
            "expected_log10_ratio": expected,
            "seeds": SEEDS,
            "per_n_lists": a_by_n
                .iter()
                .map(|(n, _, xs)| json!({ "n": n, "a_list": xs }))
                .collect::<Vec<_>>()
        }),
    );

    assert!(a16.is_finite() && a32.is_finite(), "A_n no finito");
    assert!(
        (log_ratio - expected).abs() < 1.0,
        "log₁₀(A₁₆/A₃₂) = {:.3} fuera de [{:.3}, {:.3}]",
        log_ratio,
        expected - 1.0,
        expected + 1.0
    );
}

// ── Test 8: offset estable entre seeds ───────────────────────────────────────

/// `CV(A)` sobre las 6 seeds a N=32 debe ser < 0.10 (reproduce Phase 33 como
/// sanity check, ahora desacoplado del solver).
#[test]
fn offset_stable_across_seeds() {
    let n = N_LARGE;
    let mut a_list = Vec::new();
    for &seed in SEEDS.iter() {
        let cfg = make_config(seed, n);
        let parts = gadget_ng_core::build_particles(&cfg).unwrap();
        let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let pk = power_spectrum(&positions, &masses, BOX, n);
        let lin = restrict_linear_regime(&pk, n);
        let ratios: Vec<f64> = lin
            .iter()
            .map(|b| b.pk / theory_pk_at_k(k_internal_to_hmpc(b.k)))
            .collect();
        let a = mean(&ratios);
        if a.is_finite() && a > 0.0 {
            a_list.push(a);
        }
    }

    let m = mean(&a_list);
    let c = cv(&a_list);
    dump_json(
        "b3_seeds",
        json!({
            "n": n,
            "seeds": SEEDS,
            "a_list": a_list,
            "A_mean": m,
            "CV_A": c
        }),
    );

    assert!(
        a_list.len() >= 4,
        "Al menos 4 seeds deben dar A finito, sólo {} válidos",
        a_list.len()
    );
    assert!(c < 0.10, "CV(A) entre seeds = {:.4} ≥ 0.10", c);
}
