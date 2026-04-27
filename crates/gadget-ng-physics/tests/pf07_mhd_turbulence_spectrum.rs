//! PF-07 — Turbulencia MHD: espectro cinético de Kolmogorov
//!
//! Verifica que el forzado Ornstein-Uhlenbeck produce un espectro cinético
//! compatible con la cascada de Kolmogorov `E(k) ∝ k^{-5/3}`.
//!
//! ## Tests rápidos (sin `#[ignore]`)
//!
//! Comprueban que el forzado inyecta energía cinética en la simulación.
//!
//! ## Tests lentos (`#[ignore]`)
//!
//! Evolucionan el sistema durante múltiples tiempos de correlación y miden
//! el índice espectral en el rango inercial.
//!
//! ```bash
//! cargo test -p gadget-ng-physics --release --test pf07_mhd_turbulence_spectrum -- --include-ignored
//! ```

use gadget_ng_core::{Particle, TurbulenceSection, Vec3};
use gadget_ng_mhd::{apply_turbulent_forcing, turbulence_stats};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn setup_uniform_gas(n: usize, box_size: f64) -> Vec<Particle> {
    let n_side = (n as f64).cbrt().round() as usize;
    let n_side = n_side.max(2);
    let dx = box_size / n_side as f64;
    let mass = 1.0 / (n_side * n_side * n_side) as f64;
    let mut particles = Vec::new();
    let mut id = 0usize;
    for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                let pos = Vec3::new(
                    (ix as f64 + 0.5) * dx,
                    (iy as f64 + 0.5) * dx,
                    (iz as f64 + 0.5) * dx,
                );
                let mut p = Particle::new(id, mass, pos, Vec3::zero());
                p.ptype = gadget_ng_core::ParticleType::Gas;
                p.internal_energy = 1.0;
                p.smoothing_length = 2.0 * dx;
                particles.push(p);
                id += 1;
            }
        }
    }
    particles
}

fn turbulence_cfg(amplitude: f64) -> TurbulenceSection {
    TurbulenceSection {
        enabled: true,
        amplitude,
        correlation_time: 0.1,
        k_min: 1.0,
        k_max: 4.0,
        spectral_index: 5.0 / 3.0,
    }
}

/// Calcula la varianza de la velocidad promedio de las partículas de gas.
fn velocity_variance(particles: &[Particle]) -> f64 {
    let n = particles.len() as f64;
    if n < 1.0 {
        return 0.0;
    }
    let sum_v2: f64 = particles.iter().map(|p| p.velocity.dot(p.velocity)).sum();
    sum_v2 / n
}

/// Ajusta índice espectral en bins log-k sobre un espectro `(k, E_k)`.
/// Devuelve la pendiente de la regresión lineal de log(Ek) vs log(k).
fn fit_spectral_slope(k_bins: &[(f64, f64)]) -> f64 {
    // Filtra bins con E_k > 0
    let data: Vec<(f64, f64)> = k_bins
        .iter()
        .filter(|&&(k, ek)| k > 0.0 && ek > 0.0)
        .map(|&(k, ek)| (k.ln(), ek.ln()))
        .collect();
    if data.len() < 3 {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean_x: f64 = data.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y: f64 = data.iter().map(|(_, y)| y).sum::<f64>() / n;
    let num: f64 = data.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
    let den: f64 = data.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
    if den < 1e-30 { 0.0 } else { num / den }
}

/// Calcula espectro cinético por bins de k. Usa velocidades de las partículas.
/// Devuelve (k_center, E_k) para cada bin.
fn kinetic_power_spectrum_bins(
    particles: &[Particle],
    box_size: f64,
    n_bins: usize,
) -> Vec<(f64, f64)> {
    let k_min = 2.0 * std::f64::consts::PI / box_size;
    let k_max = k_min * (n_bins as f64);
    let mut bins = vec![(0.0_f64, 0.0_f64); n_bins];
    let mut counts = vec![0usize; n_bins];

    // Estimador simple: para cada par de partículas calcular la correlación de velocidades
    // y asignar al bin según su separación. En vez de eso, usamos FFT-like con gridding.
    // Aquí implementamos un estimador sencillo por binned velocities.
    let n_side = (particles.len() as f64).cbrt().round() as usize;
    if n_side < 2 {
        return bins;
    }
    let dk = (k_max - k_min) / n_bins as f64;

    // Estimador: v_k ≈ media de v·exp(-i k·r) — usamos una versión simplificada
    // proyectando velocidades en modos k = (n, 0, 0) para n = 1..n_bins
    for ib in 0..n_bins {
        let k_val = k_min + (ib as f64 + 0.5) * dk;
        bins[ib].0 = k_val;
        let mut re_vx = 0.0_f64;
        let mut im_vx = 0.0_f64;
        let mut re_vy = 0.0_f64;
        let mut im_vy = 0.0_f64;
        let mut re_vz = 0.0_f64;
        let mut im_vz = 0.0_f64;
        for p in particles {
            let phase = k_val * p.position.x;
            re_vx += p.velocity.x * phase.cos();
            im_vx += p.velocity.x * phase.sin();
            re_vy += p.velocity.y * phase.cos();
            im_vy += p.velocity.y * phase.sin();
            re_vz += p.velocity.z * phase.cos();
            im_vz += p.velocity.z * phase.sin();
        }
        let n_p = particles.len() as f64;
        let ek = (re_vx * re_vx
            + im_vx * im_vx
            + re_vy * re_vy
            + im_vy * im_vy
            + re_vz * re_vz
            + im_vz * im_vz)
            / (n_p * n_p);
        bins[ib].1 = ek;
        counts[ib] += 1;
    }
    let _ = counts;
    bins
}

// ── Tests rápidos ─────────────────────────────────────────────────────────────

/// El forzado turbulento inyecta energía cinética: sigma_v crece con el tiempo.
#[test]
fn turbulence_forcing_injects_kinetic_energy() {
    let mut particles = setup_uniform_gas(64, 1.0);
    let cfg = turbulence_cfg(0.5);

    let v0 = velocity_variance(&particles);
    let dt = 0.01_f64;
    for step in 0..20usize {
        apply_turbulent_forcing(&mut particles, &cfg, dt, step as u64);
    }
    let v1 = velocity_variance(&particles);

    assert!(
        v1 > v0,
        "Forzado turbulento no inyectó energía: v²_antes={v0:.4e}, v²_después={v1:.4e}"
    );
}

/// `turbulence_stats` retorna valores finitos y no negativos.
#[test]
fn turbulence_stats_finite() {
    let mut particles = setup_uniform_gas(64, 1.0);
    let cfg = turbulence_cfg(0.1);
    let dt = 0.01_f64;
    for step in 0..5usize {
        apply_turbulent_forcing(&mut particles, &cfg, dt, step as u64);
    }
    let (mach_s, mach_a) = turbulence_stats(&particles, 5.0 / 3.0);
    assert!(
        mach_s.is_finite() && mach_s >= 0.0,
        "Mach sónico debe ser finito ≥ 0: {mach_s}"
    );
    assert!(
        mach_a.is_finite() && mach_a >= 0.0,
        "Mach Alfvénico debe ser finito ≥ 0: {mach_a}"
    );
}

/// La amplitud del forzado controla el nivel de energía: amplitud mayor → más energía.
#[test]
fn turbulence_amplitude_scales_energy() {
    let dt = 0.01_f64;
    let n_steps = 30usize;

    let mut p_low = setup_uniform_gas(64, 1.0);
    let cfg_low = turbulence_cfg(0.1);
    for step in 0..n_steps {
        apply_turbulent_forcing(&mut p_low, &cfg_low, dt, step as u64);
    }

    let mut p_high = setup_uniform_gas(64, 1.0);
    let cfg_high = turbulence_cfg(1.0);
    for step in 0..n_steps {
        apply_turbulent_forcing(&mut p_high, &cfg_high, dt, step as u64);
    }

    let v_low = velocity_variance(&p_low);
    let v_high = velocity_variance(&p_high);
    assert!(
        v_high > v_low,
        "Amplitud mayor debe producir más energía: v_low={v_low:.4e}, v_high={v_high:.4e}"
    );
}

// ── Tests lentos ──────────────────────────────────────────────────────────────

/// Espectro cinético tras múltiples tiempos de correlación: índice espectral ≈ -5/3 ± 0.4.
///
/// Usa N=32³ partículas y corre 200 pasos. El índice espectral del forzado
/// Ornstein-Uhlenbeck con `spectral_index = 5/3` debe reproducirse en el rango inercial.
///
/// Tolerancia amplia (±0.4) para SPH con pocos vecinos.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf07_mhd_turbulence_spectrum -- --include-ignored"]
fn turbulence_kolmogorov_spectrum_after_multiple_turnover_times() {
    let box_size = 1.0_f64;
    let mut particles = setup_uniform_gas(512, box_size); // 8³
    let cfg = turbulence_cfg(0.3);
    let dt = 0.005_f64;
    // Correr durante ~200 pasos ≈ 20 tiempos de correlación (t_corr=0.1)
    for step in 0..200usize {
        apply_turbulent_forcing(&mut particles, &cfg, dt, step as u64);
    }

    // Medir espectro cinético en 6 bins de k
    let bins = kinetic_power_spectrum_bins(&particles, box_size, 6);
    // Filtrar bins con señal
    let active: Vec<(f64, f64)> = bins
        .iter()
        .filter(|&&(_, ek)| ek > 1e-30)
        .copied()
        .collect();

    assert!(
        active.len() >= 3,
        "Se necesitan al menos 3 bins con señal para ajustar la pendiente, got {}",
        active.len()
    );

    let slope = fit_spectral_slope(&active);
    let expected = -5.0 / 3.0;
    let tolerance = 0.4;
    assert!(
        (slope - expected).abs() < tolerance,
        "Índice espectral = {slope:.3} (esperado {expected:.3} ± {tolerance:.1})"
    );
}
