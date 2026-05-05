//! Validación analítica: pancake de Zel'dovich 1D (pre-caústica).
//!
//! Mapeo Lagrangiano-Euleriano:
//!   x(q, a) = q - (delta0 / k) * D(a) * sin(k q),   k = 2π/L
//! con densidad analítica:
//!   rho(q, a) = rho0 / (1 - delta0 * D(a) * cos(k q))
//!
//! La caústica ocurre en `delta0 * D(a) = 1`.

use std::f64::consts::PI;

fn pancake_positions(n: usize, box_size: f64, delta0: f64, d_growth: f64) -> (Vec<f64>, Vec<f64>) {
    let k = 2.0 * PI / box_size;
    let dq = box_size / n as f64;
    let mut q = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let qi = (i as f64 + 0.5) * dq;
        let xi = qi - (delta0 / k) * d_growth * (k * qi).sin();
        q.push(qi);
        x.push(xi.rem_euclid(box_size));
    }
    (q, x)
}

fn analytic_rho(q: f64, box_size: f64, delta0: f64, d_growth: f64, rho0: f64) -> f64 {
    let k = 2.0 * PI / box_size;
    rho0 / (1.0 - delta0 * d_growth * (k * q).cos())
}

fn numerical_rho_from_spacing(x_sorted: &[f64], mass: f64, box_size: f64) -> Vec<f64> {
    let n = x_sorted.len();
    let mut rho = vec![0.0; n];
    for i in 0..n {
        let im1 = if i == 0 { n - 1 } else { i - 1 };
        let ip1 = if i + 1 == n { 0 } else { i + 1 };
        let mut dx = x_sorted[ip1] - x_sorted[im1];
        if dx <= 0.0 {
            dx += box_size;
        }
        // Derivada centrada: dx/dq ≈ (x_{i+1}-x_{i-1})/(2 dq)
        // rho ≈ mass / local_spacing
        let local_spacing = 0.5 * dx;
        rho[i] = mass / local_spacing.max(1e-14);
    }
    rho
}

fn rms_rel_err(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()).max(1);
    let s = (0..n)
        .map(|i| {
            let den = b[i].abs().max(1e-12);
            let r = (a[i] - b[i]) / den;
            r * r
        })
        .sum::<f64>();
    (s / n as f64).sqrt()
}

#[test]
fn pancake_precaustic_density_matches_analytic() {
    let n = 512usize;
    let box_size = 1.0_f64;
    let delta0 = 0.5_f64;
    let d_growth = 1.0_f64; // pre-caústica: delta0*D = 0.5
    let rho0 = n as f64 / box_size;
    let mass = 1.0_f64;

    let (q, mut x) = pancake_positions(n, box_size, delta0, d_growth);
    // La perturbación es monótona pre-caústica; ordenar x y reordenar q por consistencia.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
    x = idx.iter().map(|&i| x[i]).collect();
    let q_sorted: Vec<f64> = idx.iter().map(|&i| q[i]).collect();

    let rho_num = numerical_rho_from_spacing(&x, mass, box_size);
    let rho_ref: Vec<f64> = q_sorted
        .iter()
        .map(|&qi| analytic_rho(qi, box_size, delta0, d_growth, rho0))
        .collect();
    let err = rms_rel_err(&rho_num, &rho_ref);
    assert!(err < 0.03, "RMS relativo rho num vs analítico = {err:.4}");
}

#[test]
fn pancake_convergence_vs_resolution() {
    let box_size = 1.0_f64;
    let delta0 = 0.5_f64;
    let d_growth = 1.0_f64;
    let mut errs = Vec::new();
    for &n in &[128usize, 256, 512] {
        let rho0 = n as f64 / box_size;
        let mass = 1.0_f64;
        let (q, mut x) = pancake_positions(n, box_size, delta0, d_growth);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
        x = idx.iter().map(|&i| x[i]).collect();
        let q_sorted: Vec<f64> = idx.iter().map(|&i| q[i]).collect();
        let rho_num = numerical_rho_from_spacing(&x, mass, box_size);
        let rho_ref: Vec<f64> = q_sorted
            .iter()
            .map(|&qi| analytic_rho(qi, box_size, delta0, d_growth, rho0))
            .collect();
        errs.push(rms_rel_err(&rho_num, &rho_ref));
    }
    assert!(
        errs[2] < errs[1] && errs[1] < errs[0],
        "Convergencia N falló: errs={errs:?}"
    );
}

fn integrate_positions_in_time(
    n: usize,
    box_size: f64,
    delta0: f64,
    t0: f64,
    t1: f64,
    dt: f64,
) -> Vec<f64> {
    let k = 2.0 * PI / box_size;
    let dq = box_size / n as f64;
    let c: Vec<f64> = (0..n)
        .map(|i| {
            let q = (i as f64 + 0.5) * dq;
            -(delta0 / k) * (k * q).sin()
        })
        .collect();
    let mut x: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) * dq).collect();
    let mut t = t0;
    while t < t1 - 1e-15 {
        let h = (t1 - t).min(dt);
        // EdS: a(t) = t^(2/3), da/dt = (2/3) t^(-1/3)
        let da_dt = (2.0 / 3.0) * t.powf(-1.0 / 3.0);
        for i in 0..n {
            x[i] = (x[i] + c[i] * da_dt * h).rem_euclid(box_size);
        }
        t += h;
    }
    x
}

#[test]
fn pancake_convergence_vs_dt() {
    let n = 256usize;
    let box_size = 1.0_f64;
    let delta0 = 0.5_f64;
    let t0 = 1.0_f64;
    let t1 = 8.0_f64; // a: 1 -> 4 ; pre-caústica si delta0*a < 1 => delta0<0.25.
    // Para evitar caústica durante la integración temporal, usamos delta0 pequeño.
    let delta0 = delta0 * 0.2;

    let x_ref = integrate_positions_in_time(n, box_size, delta0, t0, t1, 1e-4);
    let e = |dt: f64| -> f64 {
        let x = integrate_positions_in_time(n, box_size, delta0, t0, t1, dt);
        let s = x
            .iter()
            .zip(x_ref.iter())
            .map(|(a, b)| {
                let mut d = (a - b).abs();
                if d > 0.5 * box_size {
                    d = box_size - d;
                }
                d * d
            })
            .sum::<f64>();
        (s / n as f64).sqrt()
    };

    let e1 = e(0.1);
    let e2 = e(0.05);
    let e3 = e(0.025);
    assert!(
        e3 < e2 && e2 < e1,
        "Convergencia dt falló: [{e1:.3e}, {e2:.3e}, {e3:.3e}]"
    );
}
