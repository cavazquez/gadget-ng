//! PF-04 — PM periódico: convergencia con N_mesh
//!
//! Verifica que la fuerza calculada por el solver PM converge a la solución
//! analítica conforme crece el tamaño de la malla.
//!
//! ## Metodología
//!
//! Se usa una densidad sinusoidal ρ(x) = ρ₀ + A·sin(2π·x/L) cuya solución
//! de Poisson es exactamente conocida:
//!
//! ```text
//! φ(x) = -G·A·L² / (4π²) · sin(2π·x/L)
//! f_x(x) = G·A·L / (2π) · cos(2π·x/L)
//! ```
//!
//! Se mide el error relativo de la fuerza PM respecto al valor analítico y
//! se verifica que decrece con N_mesh.
//!
//! ## Tests rápidos (sin `#[ignore]`)
//!
//! Verifican que `solve_forces` produce fuerzas finitas y simétricas.
//!
//! ## Tests lentos (`#[ignore]`)
//!
//! Barren N_mesh ∈ {8, 16, 32} y verifican que el error decrece.
//!
//! ```bash
//! cargo test -p gadget-ng-physics --release --test pf04_pm_mesh_convergence -- --include-ignored
//! ```

use gadget_ng_pm::fft_poisson::solve_forces;
use std::f64::consts::PI;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Construye una grilla de densidad sinusoidal ρ(x) = ρ₀ + A·sin(2π·x/L).
fn sinusoidal_density(nm: usize, rho0: f64, amplitude: f64) -> Vec<f64> {
    let mut grid = vec![0.0_f64; nm * nm * nm];
    for ix in 0..nm {
        let x = (ix as f64 + 0.5) / nm as f64; // ∈ (0, 1) normalizado
        let rho = rho0 + amplitude * (2.0 * PI * x).sin();
        for iy in 0..nm {
            for iz in 0..nm {
                let idx = ix * nm * nm + iy * nm + iz;
                grid[idx] = rho;
            }
        }
    }
    grid
}

/// Error RMS de la fuerza PM en x respecto al valor analítico.
fn force_rms_error(nm: usize, amplitude: f64, g: f64, box_size: f64) -> f64 {
    let density = sinusoidal_density(nm, 1.0, amplitude);
    let forces = solve_forces(&density, g, nm, box_size);
    let fx = &forces[0];

    let mut sq_err = 0.0_f64;
    let mut n_pts = 0usize;

    for ix in 0..nm {
        let x = (ix as f64 + 0.5) / nm as f64 * box_size;
        // Solución analítica: f_x = G·A·L / (2π) · cos(2π·x/L)
        let f_ana = g * amplitude * box_size / (2.0 * PI) * (2.0 * PI * x / box_size).cos();

        // Promedio sobre iy, iz
        let f_pm: f64 = (0..nm * nm).map(|j| fx[ix * nm * nm + j]).sum::<f64>() / (nm * nm) as f64;

        if f_ana.abs() > 1e-10 {
            let rel = (f_pm - f_ana) / f_ana.abs();
            sq_err += rel * rel;
            n_pts += 1;
        }
    }

    if n_pts == 0 {
        return 1.0;
    }
    (sq_err / n_pts as f64).sqrt()
}

// ── Tests rápidos ─────────────────────────────────────────────────────────────

/// `solve_forces` con densidad uniforme produce fuerzas nulas (∇·const = 0).
#[test]
fn pm_uniform_density_zero_force() {
    let nm = 8usize;
    let density = vec![1.0_f64; nm * nm * nm];
    let forces = solve_forces(&density, 1.0, nm, 1.0);
    for i in 0..forces[0].len() {
        let fx = forces[0][i].abs();
        let fy = forces[1][i].abs();
        let fz = forces[2][i].abs();
        assert!(
            fx < 1e-10 && fy < 1e-10 && fz < 1e-10,
            "Fuerza no nula para densidad uniforme: fx={fx:.3e}, fy={fy:.3e}, fz={fz:.3e}"
        );
    }
}

/// `solve_forces` produce fuerzas finitas para densidad sinusoidal.
#[test]
fn pm_sinusoidal_density_finite_forces() {
    let nm = 16usize; // usar nm más grande para mayor precisión
    let density = sinusoidal_density(nm, 1.0, 0.5);
    let forces = solve_forces(&density, 1.0, nm, 1.0);
    for &f in &forces[0] {
        assert!(f.is_finite(), "Fuerza no finita: {f}");
    }
    // Con densidad sinusoidal en x, debe haber alguna fuerza en el dominio
    let max_fx = forces[0].iter().map(|f| f.abs()).fold(0.0_f64, f64::max);
    let max_fy = forces[1].iter().map(|f| f.abs()).fold(0.0_f64, f64::max);
    let max_fz = forces[2].iter().map(|f| f.abs()).fold(0.0_f64, f64::max);
    // Las fuerzas deben ser no nulas (cualquier componente)
    assert!(
        max_fx > 1e-12 || max_fy > 1e-12 || max_fz > 1e-12,
        "solve_forces debe dar fuerzas no nulas para densidad sinusoidal: fx={max_fx:.3e}, fy={max_fy:.3e}, fz={max_fz:.3e}"
    );
}

/// La fuerza PM tiene simetría antisimétrica para densidad sinusoidal.
#[test]
fn pm_force_antisymmetric() {
    let nm = 8usize;
    let density = sinusoidal_density(nm, 0.0, 1.0);
    let forces = solve_forces(&density, 1.0, nm, 1.0);
    let fx = &forces[0];
    // f(ix) ≈ -f(nm - ix - 1) para una sinusoidal
    let nm2 = nm / 2;
    let mut max_asym = 0.0_f64;
    for ix in 0..nm2 {
        let f1: f64 = (0..nm * nm).map(|j| fx[ix * nm * nm + j]).sum::<f64>() / (nm * nm) as f64;
        let ix2 = nm - ix - 1;
        let f2: f64 = (0..nm * nm).map(|j| fx[ix2 * nm * nm + j]).sum::<f64>() / (nm * nm) as f64;
        if f1.abs() > 1e-10 {
            max_asym = max_asym.max(((f1 + f2) / f1.abs()).abs());
        }
    }
    // Tolerancia amplia (la grilla introduce errores de discretización)
    assert!(
        max_asym < 0.5,
        "Asimetría de la fuerza PM demasiado grande: {max_asym:.3}"
    );
}

// ── Tests lentos ──────────────────────────────────────────────────────────────

/// La fuerza PM converge al valor analítico al aumentar N_mesh.
///
/// Se espera que el error disminuya con nm. Para el modo de Nyquist (k=2π/L),
/// el error PM decrece aproximadamente como 1/nm².
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf04_pm_mesh_convergence -- --include-ignored"]
fn pm_force_error_decreases_with_nmesh() {
    let g = 1.0_f64;
    let box_size = 1.0_f64;
    let amplitude = 0.5_f64;

    let nm_values = [8, 16, 32];
    let errors: Vec<f64> = nm_values
        .iter()
        .map(|&nm| force_rms_error(nm, amplitude, g, box_size))
        .collect();

    println!("PM convergencia:");
    for (nm, err) in nm_values.iter().zip(errors.iter()) {
        println!("  N_mesh={nm:3}: RMS error = {err:.4}");
    }

    // El error debe decrecer al aumentar N_mesh
    for i in 1..errors.len() {
        assert!(
            errors[i] <= errors[i - 1] * 1.1, // tolerancia 10% para variaciones numéricas
            "Error PM no decreció al aumentar N_mesh: err[{}]={:.4} > err[{}]={:.4}",
            nm_values[i],
            errors[i],
            nm_values[i - 1],
            errors[i - 1]
        );
    }

    // El error a N_mesh=32 debe ser < 20% (tolerancia para modo k=1)
    assert!(
        errors[2] < 0.20,
        "Error PM a N_mesh=32 demasiado alto: {:.4} (tolerancia 20%)",
        errors[2]
    );
}

/// Error relativo < 5% para N_mesh = 32 en modo k=1.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf04_pm_mesh_convergence -- --include-ignored"]
fn pm_force_error_lt_5pct_nmesh32() {
    let err = force_rms_error(32, 0.5, 1.0, 1.0);
    println!("PM N_mesh=32: RMS error = {err:.4}");
    assert!(err < 0.05, "Error PM a N_mesh=32: {err:.4} (tolerancia 5%)");
}
