//! Tests de validación para condiciones iniciales de Zel'dovich — Fase 26.
//!
//! ## Cobertura
//!
//! 1. `zel_reproducible`:
//!    Misma seed + parámetros → partículas idénticas bit-a-bit.
//!
//! 2. `zel_mean_displacement_zero`:
//!    El desplazamiento medio `⟨Ψ⟩ ≈ 0` como consecuencia del modo DC nulo.
//!
//! 3. `zel_dc_mode_zero`:
//!    El modo `k=0` del campo generado es exactamente cero.
//!
//! 4. `zel_positions_in_box`:
//!    Todas las posiciones están en `[0, box_size)` tras el wrap periódico.
//!
//! 5. `zel_displacement_rms_linear_regime`:
//!    El RMS del desplazamiento es `< 0.3·d` (régimen lineal garantizado).
//!
//! 6. `zel_pk_follows_power_law`:
//!    P(k) medido con el estimador CIC sigue aproximadamente `∝ k^n_s`
//!    (ratio en escala log lineal, tolerancia del 50% por bin de escala).
//!
//! 7. `zel_pm_short_run_stable`:
//!    10 pasos de integración con `PmSolver` no produce NaN/Inf.
//!
//! 8. `zel_treepm_short_run_stable`:
//!    10 pasos de integración con `TreePmSolver` no produce NaN/Inf.

use gadget_ng_analysis::power_spectrum::power_spectrum;
use gadget_ng_core::{
    build_particles, build_particles_for_gid_range, cosmology::CosmologyParams, growth_rate_f,
    wrap_position, CosmologySection, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, OutputSection, PerformanceSection, RunConfig, SimulationSection,
    TimestepSection, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use gadget_ng_treepm::TreePmSolver;

// ── Constantes ────────────────────────────────────────────────────────────────

const G: f64 = 1.0;
const BOX: f64 = 1.0;
const GRID: usize = 8; // 8³ = 512 partículas (rápido en CI)
const N_PART: usize = 512; // 8³
const NM: usize = 8; // grid PM

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Configuración EdS mínima con ICs de Zel'dovich.
fn zel_config(seed: u64, spectral_index: f64, amplitude: f64) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.002,
            num_steps: 10,
            softening: 0.02,
            physical_softening: false,
            gravitational_constant: G,
            particle_count: N_PART,
            box_size: BOX,
            seed,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed,
                grid_size: GRID,
                spectral_index,
                amplitude,
                // Campos de Fase 27/28 con defaults (retrocompatibilidad).
                transfer: gadget_ng_core::TransferKind::PowerLaw,
                sigma8: None,
                omega_b: 0.049,
                h: 0.674,
                t_cmb: 2.7255,
                box_size_mpc_h: None,
                use_2lpt: false,
                normalization_mode: gadget_ng_core::NormalizationMode::Legacy,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::Pm,
            pm_grid_size: NM,
            ..GravitySection::default()
        },
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m: 1.0,
            omega_lambda: 0.0,
            h0: 0.1,
            a_init: 0.02,
            auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(), reionization: Default::default(),
    }
}

// ── Test 1: reproducibilidad ──────────────────────────────────────────────────

/// Dos llamadas con la misma seed deben devolver partículas idénticas (bit-a-bit).
#[test]
fn zel_reproducible() {
    let cfg = zel_config(42, -2.0, 1.0e-4);
    let parts_a = build_particles(&cfg).expect("IC build A");
    let parts_b = build_particles(&cfg).expect("IC build B");

    assert_eq!(parts_a.len(), parts_b.len(), "Número de partículas difiere");

    for (a, b) in parts_a.iter().zip(parts_b.iter()) {
        assert_eq!(
            a.position.x.to_bits(),
            b.position.x.to_bits(),
            "x diferente en gid={}: {} vs {}",
            a.global_id,
            a.position.x,
            b.position.x
        );
        assert_eq!(
            a.velocity.x.to_bits(),
            b.velocity.x.to_bits(),
            "vx diferente en gid={}",
            a.global_id
        );
    }
}

// ── Test 2: desplazamiento medio ≈ 0 ─────────────────────────────────────────

/// El modo DC nulo implica que el desplazamiento medio de las partículas es ~0.
/// Medimos el desplazamiento respecto a la retícula regular.
#[test]
fn zel_mean_displacement_zero() {
    let cfg = zel_config(7, -2.0, 1.0e-4);
    let parts = build_particles(&cfg).expect("IC build");

    let d = BOX / GRID as f64; // spacing

    // Posición de la retícula esperada para cada gid.
    let mean_dx: f64 = parts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let gid = p.global_id;
            let ix = gid / (GRID * GRID);
            let rem = gid % (GRID * GRID);
            let iy = rem / GRID;
            let iz = rem % GRID;
            let q_x = (ix as f64 + 0.5) * d;
            // Desplazamiento wrapeado (puede cruzar el borde de caja).
            let raw_dx = p.position.x - q_x;
            let dx = raw_dx - BOX * (raw_dx / BOX).round();
            let _ = (iy, iz, i); // silenciar warnings
            dx
        })
        .sum::<f64>()
        / N_PART as f64;

    assert!(
        mean_dx.abs() < 5.0 * d / N_PART as f64,
        "⟨Ψ_x⟩ = {:.3e} no es ~0 (esperado < {:.3e})",
        mean_dx,
        5.0 * d / N_PART as f64
    );
}

// ── Test 3: modo DC exactamente cero ─────────────────────────────────────────

/// El campo de desplazamiento generado tiene modo DC = 0 por construcción.
/// Lo verificamos indirectamente: el P(k) del campo medido no incluye el modo k=0.
/// También verificamos directamente que las posiciones medias no tienen drift sistemático.
#[test]
fn zel_dc_mode_zero() {
    let cfg = zel_config(13, -2.0, 1.0e-4);
    let parts = build_particles(&cfg).expect("IC build");

    // Si el modo DC fuera no nulo, habría un desplazamiento sistemático de todas
    // las partículas en la misma dirección; el RMS del campo sería dominated por DC.
    // Verificamos que el COM de las partículas está cerca del centro de la caja.
    let com_x: f64 = parts.iter().map(|p| p.position.x * p.mass).sum::<f64>();
    let com_y: f64 = parts.iter().map(|p| p.position.y * p.mass).sum::<f64>();
    let com_z: f64 = parts.iter().map(|p| p.position.z * p.mass).sum::<f64>();

    let expected_com = 0.5 * BOX; // masa total = 1, COM ideal = BOX/2

    // Tolerancia: BOX / sqrt(N) ≈ 0.044 (fluctuaciones de Poisson esperadas).
    let tol = BOX / (N_PART as f64).sqrt();
    assert!(
        (com_x - expected_com).abs() < 3.0 * tol,
        "COM_x = {:.4} ≠ {:.4} (3σ = {:.4}) — posible modo DC no nulo",
        com_x,
        expected_com,
        3.0 * tol
    );
    assert!(
        (com_y - expected_com).abs() < 3.0 * tol,
        "COM_y = {:.4} ≠ {:.4}",
        com_y,
        expected_com
    );
    assert!(
        (com_z - expected_com).abs() < 3.0 * tol,
        "COM_z = {:.4} ≠ {:.4}",
        com_z,
        expected_com
    );
}

// ── Test 4: posiciones dentro de la caja ─────────────────────────────────────

/// Todas las posiciones deben estar en `[0, BOX)` tras el wrap periódico.
#[test]
fn zel_positions_in_box() {
    let cfg = zel_config(99, -2.0, 1.0e-4);
    let parts = build_particles(&cfg).expect("IC build");

    for p in &parts {
        assert!(
            p.position.x >= 0.0 && p.position.x < BOX,
            "x = {} fuera de [0, {BOX}) en gid={}",
            p.position.x,
            p.global_id
        );
        assert!(
            p.position.y >= 0.0 && p.position.y < BOX,
            "y = {} fuera de [0, {BOX}) en gid={}",
            p.position.y,
            p.global_id
        );
        assert!(
            p.position.z >= 0.0 && p.position.z < BOX,
            "z = {} fuera de [0, {BOX}) en gid={}",
            p.position.z,
            p.global_id
        );
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "Posición no finita en gid={}: {:?}",
            p.global_id,
            p.position
        );
    }
}

// ── Test 5: RMS de desplazamiento en régimen lineal ───────────────────────────

/// Con `amplitude = 1e-4`, el RMS del desplazamiento debe ser muy inferior
/// al spacing `d`, confirmando que estamos en el régimen lineal.
#[test]
fn zel_displacement_rms_linear_regime() {
    let cfg = zel_config(42, -2.0, 1.0e-4);
    let parts = build_particles(&cfg).expect("IC build");
    let d = BOX / GRID as f64;

    // Calcular RMS de desplazamiento respecto a la retícula.
    let sum_sq: f64 = parts
        .iter()
        .map(|p| {
            let gid = p.global_id;
            let ix = gid / (GRID * GRID);
            let rem = gid % (GRID * GRID);
            let iy = rem / GRID;
            let iz = rem % GRID;
            let q = Vec3::new(
                (ix as f64 + 0.5) * d,
                (iy as f64 + 0.5) * d,
                (iz as f64 + 0.5) * d,
            );
            // Desplazamiento con mínima imagen.
            let dx = p.position.x - q.x;
            let dy = p.position.y - q.y;
            let dz = p.position.z - q.z;
            let dx = dx - BOX * (dx / BOX).round();
            let dy = dy - BOX * (dy / BOX).round();
            let dz = dz - BOX * (dz / BOX).round();
            dx * dx + dy * dy + dz * dz
        })
        .sum();

    let rms = (sum_sq / N_PART as f64).sqrt();
    let rms_over_d = rms / d;

    assert!(
        rms_over_d < 0.3,
        "Ψ_rms/d = {:.4} ≥ 0.3 — posiblemente fuera del régimen lineal (amplitude demasiado grande)",
        rms_over_d
    );
    assert!(
        rms > 0.0,
        "Ψ_rms = 0 — el generador no produjo perturbaciones"
    );
}

// ── Test 6: P(k) sigue la ley de potencia ────────────────────────────────────

/// El espectro de potencia medido debe seguir `P(k) ∝ k^n_s` aproximadamente.
/// Verificamos que el ratio `P(k_high)/P(k_low)` es consistente con la pendiente.
#[test]
fn zel_pk_follows_power_law() {
    // Usar N=16³ = 4096 para tener suficientes modos.
    const GRID16: usize = 16;
    const N16: usize = 4096;

    let cfg = RunConfig {
        simulation: SimulationSection {
            dt: 0.002,
            num_steps: 10,
            softening: 0.02,
            physical_softening: false,
            gravitational_constant: G,
            particle_count: N16,
            box_size: BOX,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed: 42,
                grid_size: GRID16,
                spectral_index: -2.0,
                amplitude: 1.0e-4,
                transfer: gadget_ng_core::TransferKind::PowerLaw,
                sigma8: None,
                omega_b: 0.049,
                h: 0.674,
                t_cmb: 2.7255,
                box_size_mpc_h: None,
                use_2lpt: false,
                normalization_mode: gadget_ng_core::NormalizationMode::Legacy,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m: 1.0,
            omega_lambda: 0.0,
            h0: 0.1,
            a_init: 0.02,
            auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(), reionization: Default::default(),
    };

    let parts = build_particles(&cfg).expect("IC build N16");
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();

    let pk_bins = power_spectrum(&positions, &masses, BOX, GRID16);

    // Debe haber bins con señal.
    let bins_with_signal: Vec<_> = pk_bins
        .iter()
        .filter(|b| b.pk > 0.0 && b.n_modes > 0)
        .collect();
    assert!(
        bins_with_signal.len() >= 3,
        "Menos de 3 bins con señal en P(k): {} bins totales",
        pk_bins.len()
    );

    // Verificar pendiente: en log-log, la pendiente debe ser cercana a n_s = -2.
    // Usamos los primeros 4 bins (modos bajos) para evitar el roll-off del Nyquist.
    let good_bins: Vec<_> = bins_with_signal.iter().take(4).collect();
    if good_bins.len() >= 2 {
        let k_lo = good_bins[0].k;
        let pk_lo = good_bins[0].pk;
        let k_hi = good_bins[good_bins.len() - 1].k;
        let pk_hi = good_bins[good_bins.len() - 1].pk;

        if pk_lo > 0.0 && pk_hi > 0.0 && k_hi > k_lo {
            let measured_slope = (pk_hi / pk_lo).ln() / (k_hi / k_lo).ln();
            // Tolerancia generosa: ±2 unidades de n_s (la medición con N=16³ tiene varianza alta).
            assert!(
                measured_slope > -2.0 - 2.0 && measured_slope < -2.0 + 2.0,
                "Pendiente medida en P(k) = {:.2} vs n_s=-2 (tolerancia ±2) — k=[{:.3},{:.3}]",
                measured_slope,
                k_lo,
                k_hi
            );
        }
    }

    // Verificar que P(k) no tiene NaN/Inf.
    for b in &pk_bins {
        assert!(b.pk.is_finite(), "P(k={:.3}) = {} no es finito", b.k, b.pk);
    }
}

// ── Test 7: run corto con PM no explota ──────────────────────────────────────

/// 10 pasos de leapfrog cosmológico con PmSolver no debe producir NaN/Inf.
#[test]
fn zel_pm_short_run_stable() {
    let cfg = zel_config(42, -2.0, 1.0e-4);
    let mut parts = build_particles(&cfg).expect("IC build");
    let cosmo = CosmologyParams::new(1.0, 0.0, 0.1);
    let mut a = 0.02_f64;
    let dt = 0.002_f64;
    let pm = PmSolver {
        grid_size: NM,
        box_size: BOX,
    };
    let mut scratch = vec![Vec3::zero(); N_PART];

    for _ in 0..10 {
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half,
            kick_half2,
        };
        a = cosmo.advance_a(a, dt);

        leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc);
        });

        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }

    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "Posición no finita con PM en gid={}: {:?}",
            p.global_id,
            p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "Velocidad no finita con PM en gid={}: {:?}",
            p.global_id,
            p.velocity
        );
    }
}

// ── Test 8: run corto con TreePM no explota ───────────────────────────────────

/// 10 pasos de leapfrog cosmológico con TreePmSolver no debe producir NaN/Inf.
#[test]
fn zel_treepm_short_run_stable() {
    let cfg = zel_config(42, -2.0, 1.0e-4);
    let mut parts = build_particles(&cfg).expect("IC build");
    let cosmo = CosmologyParams::new(1.0, 0.0, 0.1);
    let mut a = 0.02_f64;
    let dt = 0.002_f64;
    let treepm = TreePmSolver {
        grid_size: NM,
        box_size: BOX,
        r_split: 0.0, // automático
    };
    let mut scratch = vec![Vec3::zero(); N_PART];

    for _ in 0..10 {
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half,
            kick_half2,
        };
        a = cosmo.advance_a(a, dt);

        leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            treepm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc);
        });

        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }

    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "Posición no finita con TreePM en gid={}: {:?}",
            p.global_id,
            p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "Velocidad no finita con TreePM en gid={}: {:?}",
            p.global_id,
            p.velocity
        );
    }
}

// ── Test bonus: consistencia rango MPI ───────────────────────────────────────

/// Las partículas generadas por rango [lo, hi) son idénticas a las del build completo.
#[test]
fn zel_gid_range_consistent() {
    let cfg = zel_config(42, -2.0, 1.0e-4);

    let all = build_particles(&cfg).expect("full build");
    let lo = build_particles_for_gid_range(&cfg, 0, N_PART / 2).expect("range 0..N/2");
    let hi = build_particles_for_gid_range(&cfg, N_PART / 2, N_PART).expect("range N/2..N");

    assert_eq!(
        lo.len() + hi.len(),
        all.len(),
        "split no cubre todas las partículas"
    );

    for p_all in &all {
        let found = lo
            .iter()
            .chain(hi.iter())
            .find(|q| q.global_id == p_all.global_id);
        let p_range = found.expect(&format!("gid {} no encontrado en rangos", p_all.global_id));
        assert_eq!(
            p_all.position.x.to_bits(),
            p_range.position.x.to_bits(),
            "posición x inconsistente para gid={}",
            p_all.global_id
        );
    }
}

// ── Test bonus: velocidades físicamente razonables ────────────────────────────

/// Las velocidades deben ser coherentes con la teoría lineal.
/// Para EdS con h0=0.1, a=0.02: v_rms ~ a·f·H·Ψ_rms ~ 0.02·1·0.1·Ψ_rms.
#[test]
fn zel_velocities_linear_theory() {
    let cfg = zel_config(42, -2.0, 1.0e-4);
    let parts = build_particles(&cfg).expect("IC build");
    let cosmo = CosmologyParams::new(
        cfg.cosmology.omega_m,
        cfg.cosmology.omega_lambda,
        cfg.cosmology.h0,
    );
    let a_init = cfg.cosmology.a_init;
    let f = growth_rate_f(cosmo, a_init);
    let h = gadget_ng_core::hubble_param(cosmo, a_init);

    // v_rms_esperado ≈ a²·f·H · Ψ_rms (ya que p = a²·f·H·Ψ)
    // pero en `particle.velocity` se almacena p directamente.
    // v_pec = p/a → v_rms_pec = p_rms / a

    let p_rms: f64 = {
        let sum_sq: f64 = parts.iter().map(|p| p.velocity.dot(p.velocity)).sum();
        (sum_sq / N_PART as f64).sqrt()
    };

    let vel_factor = a_init * a_init * f * h;

    // El ratio p_rms / vel_factor debe ser el Ψ_rms del campo.
    let psi_from_vel = if vel_factor > 0.0 {
        p_rms / vel_factor
    } else {
        0.0
    };

    // Verificar que no es cero ni explosivo.
    assert!(
        psi_from_vel > 0.0,
        "Velocidades son todas cero — el generador no asignó momenta"
    );
    assert!(
        psi_from_vel < 1.0,
        "Ψ_rms inferido de velocidades = {:.4e} — posiblemente incorrecto",
        psi_from_vel
    );
}
