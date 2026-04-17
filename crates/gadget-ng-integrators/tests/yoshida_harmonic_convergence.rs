//! Convergencia temporal del integrador Yoshida 4º orden en un oscilador armónico 1D.
//!
//! El oscilador `d²x/dt² = -k·x/m` con `k = m = 1` tiene solución periódica
//! exacta y energía conservada `E = ½ v² + ½ x²`. Integrando sobre un tiempo
//! final fijo `t_f` con distintos pasos `dt`, el error máximo de energía debe
//! escalar como `dt^p` donde `p` es el orden del integrador:
//!
//! - `p ≈ 2` para leapfrog KDK (Yoshida 1990, Hairer+ 2006 §II.5)
//! - `p ≈ 4` para Yoshida 4º orden (Yoshida 1990)
//!
//! Este test:
//! 1. Integra en un barrido `dt ∈ {0.2, 0.1, 0.05, 0.025}` hasta `t_f = 10·2π`
//! 2. Mide `max_t |E(t) - E₀|` para ambos integradores
//! 3. Ajusta la pendiente `log(err) ~ p·log(dt)` por mínimos cuadrados
//! 4. Vuelca los resultados a `experiments/nbody/phase6_higher_order_integrator/results/harmonic_convergence.csv`
//!
//! Aserciones:
//! - Pendiente KDK en `[1.7, 2.3]`
//! - Pendiente Yoshida en `[3.5, 4.5]`
//! - Error Yoshida al dt menor al menos 100× más pequeño que el de KDK
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_integrators::{leapfrog_kdk_step, yoshida4_kdk_step};
use std::fs;
use std::path::PathBuf;

const T_FINAL: f64 = 10.0 * std::f64::consts::TAU;
const DTS: &[f64] = &[0.2, 0.1, 0.05, 0.025];

fn force_harmonic(parts: &[Particle], acc: &mut [Vec3]) {
    acc[0] = -parts[0].position;
}

fn energy(p: &Particle) -> f64 {
    0.5 * p.velocity.dot(p.velocity) + 0.5 * p.position.dot(p.position)
}

fn run_sweep<F>(mut one_step: F) -> Vec<(f64, f64)>
where
    F: FnMut(&mut [Particle], f64, &mut [Vec3]),
{
    let mut out = Vec::new();
    for &dt in DTS {
        let n_steps = (T_FINAL / dt).round() as u64;
        let mut parts = vec![Particle::new(
            0,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.3, 0.0),
        )];
        let mut scratch = vec![Vec3::zero(); 1];
        let e0 = energy(&parts[0]);
        let mut err_max: f64 = 0.0;
        for _ in 0..n_steps {
            one_step(&mut parts, dt, &mut scratch);
            err_max = err_max.max((energy(&parts[0]) - e0).abs() / e0.abs());
        }
        out.push((dt, err_max));
    }
    out
}

fn fit_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let xs: Vec<f64> = points.iter().map(|(dt, _)| dt.ln()).collect();
    let ys: Vec<f64> = points.iter().map(|(_, e)| e.max(1e-30).ln()).collect();
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        num += (x - mx) * (y - my);
        den += (x - mx).powi(2);
    }
    num / den
}

fn results_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("../..")
        .join("experiments/nbody/phase6_higher_order_integrator/results")
        .canonicalize()
        .unwrap_or_else(|_| {
            manifest
                .join("../..")
                .join("experiments/nbody/phase6_higher_order_integrator/results")
        })
}

#[test]
fn yoshida4_harmonic_order_four() {
    let kdk = run_sweep(|p, dt, s| leapfrog_kdk_step(p, dt, s, force_harmonic));
    let yos = run_sweep(|p, dt, s| yoshida4_kdk_step(p, dt, s, force_harmonic));

    let slope_kdk = fit_slope(&kdk);
    let slope_yos = fit_slope(&yos);

    println!("harmonic KDK    : {:?} slope={:.3}", kdk, slope_kdk);
    println!("harmonic Yoshida: {:?} slope={:.3}", yos, slope_yos);

    let dir = results_dir();
    let _ = fs::create_dir_all(&dir);
    let csv_path = dir.join("harmonic_convergence.csv");
    let mut csv = String::from("system,integrator,dt,err_rel_max,fitted_slope\n");
    for (dt, err) in &kdk {
        csv.push_str(&format!("harmonic,leapfrog,{dt},{err},{slope_kdk}\n"));
    }
    for (dt, err) in &yos {
        csv.push_str(&format!("harmonic,yoshida4,{dt},{err},{slope_yos}\n"));
    }
    let _ = fs::write(&csv_path, csv);

    assert!(
        (1.7..=2.3).contains(&slope_kdk),
        "pendiente KDK esperada ≈2, got {slope_kdk}"
    );
    assert!(
        (3.5..=4.5).contains(&slope_yos),
        "pendiente Yoshida esperada ≈4, got {slope_yos}"
    );

    let err_kdk_min = kdk.last().unwrap().1;
    let err_yos_min = yos.last().unwrap().1;
    assert!(
        err_yos_min * 100.0 < err_kdk_min,
        "Yoshida debería ser ≥100× mejor que KDK al dt menor: KDK={err_kdk_min}, Yos={err_yos_min}"
    );
}
