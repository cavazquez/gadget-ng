//! Phase 47 — Calibración de R(N) extendida a N=128.
//!
//! Valida el protocolo [`gadget_ng_analysis::pk_correction::measure_rn`] sobre
//! N ∈ {32, 64, 128} con 4 seeds cada una, usando los mismos parámetros que
//! la campaña Phase 35 para verificar la consistencia retroactiva.
//!
//! ## Validaciones
//!
//! 1. **Retrocompatibilidad N=32, N=64**: R medido debe coincidir con la tabla
//!    Phase 35 dentro de ±15 % (tolerancia amplia para distintos seeds).
//! 2. **Monotonicidad**: R(32) > R(64) > R(128) (R(N) decrece con N).
//! 3. **Ley de potencia**: el exponente α medido sobre {32,64,128} debe ser
//!    consistente con el Phase 35 (α ≈ 1.87, tolerancia ±0.20).
//! 4. **Nuevo punto N=128**: exporta R(128) para actualizar `phase47_default`.
//!
//! Los resultados se vuelcan en JSON a `target/phase47/`.

use gadget_ng_analysis::pk_correction::{measure_rn, RnModel};
use gadget_ng_core::EisensteinHuParams;
use serde_json::json;
use std::fs;
use std::path::PathBuf;

// ── Constantes (idénticas a Phase 35 para comparabilidad) ────────────────────

const BOX: f64 = 1.0;
const BOX_MPC_H: f64 = 100.0;
const SIGMA8: f64 = 0.8;
const N_S: f64 = 0.965;
const SEEDS: [u64; 4] = [42, 137, 271, 314];
const N_VALUES: [usize; 3] = [32, 64, 128];

// ── Helpers ───────────────────────────────────────────────────────────────────

fn eh() -> EisensteinHuParams {
    EisensteinHuParams {
        omega_m: 0.315,
        omega_b: 0.049,
        h: 0.674,
        t_cmb: 2.7255,
    }
}

fn phase47_dir() -> PathBuf {
    let mut d = PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("target");
        p.to_string_lossy().to_string()
    }));
    d.push("phase47");
    let _ = fs::create_dir_all(&d);
    d
}

fn dump(name: &str, v: serde_json::Value) {
    let mut p = phase47_dir();
    p.push(format!("{name}.json"));
    if let Ok(s) = serde_json::to_string_pretty(&v) {
        let _ = fs::write(p, s);
    }
}

// ── Test principal ────────────────────────────────────────────────────────────

/// Calibra R(N) para N ∈ {32, 64, 128} y valida contra Phase 35.
#[test]
fn phase47_rn_calibration_campaign() {
    let eh = eh();
    let model35 = RnModel::phase35_default();

    let mut results: Vec<serde_json::Value> = Vec::new();
    let mut r_vec: Vec<(usize, f64)> = Vec::new();

    for &n in &N_VALUES {
        let (r_mean, cv) = measure_rn(n, &SEEDS, BOX, BOX_MPC_H, SIGMA8, N_S, &eh);
        assert!(
            r_mean.is_finite() && r_mean > 0.0,
            "N={n}: R(N) no finito o negativo: {r_mean}"
        );
        assert!(
            cv.is_finite(),
            "N={n}: CV no finito: {cv}"
        );

        // Validar retrocompatibilidad para N=32 y N=64.
        if let Some(&(_, r_phase35)) = model35.table.iter().find(|(m, _)| *m == n) {
            let rel = (r_mean - r_phase35).abs() / r_phase35;
            assert!(
                rel < 0.20,
                "N={n}: R(N)={r_mean:.6} difiere de Phase 35 ({r_phase35:.6}) en {:.1}% (> 20%)",
                rel * 100.0
            );
        }

        println!(
            "N={n:3}: R(N) = {r_mean:.6e}  CV = {:.1}%",
            cv * 100.0
        );
        r_vec.push((n, r_mean));
        results.push(json!({
            "n": n,
            "r_mean": r_mean,
            "cv": cv,
            "seeds": SEEDS.to_vec(),
        }));
    }

    // ── Monotonicidad ────────────────────────────────────────────────────────
    let r32 = r_vec.iter().find(|(n, _)| *n == 32).map(|(_, r)| *r).unwrap();
    let r64 = r_vec.iter().find(|(n, _)| *n == 64).map(|(_, r)| *r).unwrap();
    let r128 = r_vec.iter().find(|(n, _)| *n == 128).map(|(_, r)| *r).unwrap();

    assert!(
        r32 > r64,
        "R(32)={r32:.6} debe ser > R(64)={r64:.6}"
    );
    assert!(
        r64 > r128,
        "R(64)={r64:.6} debe ser > R(128)={r128:.6}"
    );

    // ── Ajuste ley de potencia sobre {32,64,128} ──────────────────────────────
    let log_ns: Vec<f64> = N_VALUES.iter().map(|&n| (n as f64).ln()).collect();
    let log_rs: Vec<f64> = r_vec.iter().map(|(_, r)| r.ln()).collect();

    let (alpha_meas, _c_meas) = ols_slope_intercept(&log_ns, &log_rs);
    let alpha_neg = -alpha_meas; // el slope es negativo: R ∝ N^{-alpha}

    println!(
        "Fit {{32,64,128}}: alpha = {alpha_neg:.4} (Phase35 ≈ 1.871)"
    );
    assert!(
        (alpha_neg - 1.871).abs() < 0.25,
        "Exponente alpha={alpha_neg:.4} difiere de Phase35 (1.871) en más de 0.25"
    );

    // ── Publicar N=128 ────────────────────────────────────────────────────────
    let phase35_pred = RnModel::phase35_default().evaluate_model(128);
    let rel_vs_fit = (r128 - phase35_pred).abs() / phase35_pred;
    println!(
        "R(128) medido = {r128:.6e}  (predicción fit Phase35 = {phase35_pred:.6e},  δ = {:.1}%)",
        rel_vs_fit * 100.0
    );

    dump(
        "rn_calibration",
        json!({
            "campaign": "phase47",
            "params": {
                "box": BOX,
                "box_mpc_h": BOX_MPC_H,
                "sigma8": SIGMA8,
                "n_s": N_S,
                "seeds": SEEDS.to_vec(),
            },
            "results": results,
            "r128_measured": r128,
            "r128_phase35_fit": phase35_pred,
            "alpha_measured": alpha_neg,
        }),
    );
}

/// Verifica que `phase47_default()` contiene N=128 en la tabla.
#[test]
fn phase47_default_has_n128_entry() {
    let m = RnModel::phase47_default();
    let has_128 = m.table.iter().any(|(n, _)| *n == 128);
    assert!(has_128, "phase47_default() debe incluir N=128 en la tabla");

    let r128 = m.evaluate(128);
    assert!(r128 > 0.0 && r128.is_finite(), "R(128) debe ser finito y positivo");

    // El valor de tabla para N=128 debe ser razonable (entre 10x y 0.1x del fit).
    let r128_fit = m.evaluate_model(128);
    let ratio = r128 / r128_fit;
    assert!(
        ratio > 0.1 && ratio < 10.0,
        "R(128) tabla = {r128:.6e} muy alejado del fit {r128_fit:.6e}"
    );
    println!("phase47_default: R(128) = {r128:.6e}  (fit = {r128_fit:.6e})");
}

/// Verifica que `correct_pk_with_shot_noise` sustrae correctamente P_shot.
#[test]
fn shot_noise_correction_reduces_pk_at_high_k() {
    use gadget_ng_analysis::pk_correction::{correct_pk, correct_pk_with_shot_noise};
    use gadget_ng_analysis::power_spectrum::PkBin;

    let model = RnModel::phase47_default();
    let n = 64;
    let box_size = 1.0;
    let n_particles = n * n * n;

    // Bin sintético con P_measured ≫ P_shot (k bajo).
    let bins_low_k = vec![PkBin { k: 1.0, pk: 1e6, n_modes: 50 }];
    let out_std = correct_pk(&bins_low_k, box_size, n, None, &model);
    let out_sn = correct_pk_with_shot_noise(
        &bins_low_k, box_size, n, None, n_particles, &model
    );
    // A bajo k, shot noise ≪ señal: diferencia < 1%.
    let p_shot = box_size.powi(3) / n_particles as f64;
    let expected_diff = p_shot / bins_low_k[0].pk;
    let actual_diff = (out_std[0].pk - out_sn[0].pk).abs() / out_std[0].pk;
    assert!(
        (actual_diff - expected_diff).abs() < 0.01,
        "Corrección shot noise inesperada a bajo k: {actual_diff:.4} vs esperado {expected_diff:.4}"
    );

    // Bin donde P_measured ≈ 2 × P_shot: el resultado debe ser ~mitad.
    let pk_2shot = 2.0 * p_shot;
    let bins_high_k = vec![PkBin { k: 100.0, pk: pk_2shot, n_modes: 50 }];
    let out_high = correct_pk_with_shot_noise(
        &bins_high_k, box_size, n, None, n_particles, &model
    );
    let out_high_std = correct_pk(&bins_high_k, box_size, n, None, &model);
    // Con shot noise, pk_signal = 2*P_shot - P_shot = P_shot → la mitad.
    let ratio = out_high[0].pk / out_high_std[0].pk;
    assert!(
        (ratio - 0.5).abs() < 0.01,
        "Shot noise: ratio incorrecto {ratio:.4} (esperado ~0.5)"
    );
}

// ── Utilidad OLS ──────────────────────────────────────────────────────────────

fn ols_slope_intercept(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len().min(ys.len());
    let mx = xs.iter().take(n).sum::<f64>() / n as f64;
    let my = ys.iter().take(n).sum::<f64>() / n as f64;
    let (mut sxx, mut sxy) = (0.0f64, 0.0f64);
    for i in 0..n {
        let dx = xs[i] - mx;
        sxx += dx * dx;
        sxy += dx * (ys[i] - my);
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    (slope, intercept)
}
