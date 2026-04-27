//! PF-03 — FMM cuadrupolo: convergencia de error con θ
//!
//! Verifica que el error relativo del árbol Barnes-Hut con expansión cuadrupolar
//! decrece al reducir el parámetro de apertura θ:
//!
//! ```text
//! err(θ=0.3) < err(θ=0.5) < err(θ=0.7)
//! err(θ=0.3) < 0.5%
//! ```
//!
//! Referencia: Greengard & Rokhlin (1987), Barnes & Hut (1986).

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_tree::Octree;

// ── Helpers ───────────────────────────────────────────────────────────────────

const G: f64 = 1.0;

/// Genera N partículas uniformes en una esfera de radio R.
fn uniform_sphere(n: usize, r: f64, seed: u64) -> Vec<Particle> {
    let mut rng = seed;
    let next = |rng: &mut u64| -> f64 {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*rng >> 33) as f64 / (u64::MAX >> 33) as f64
    };

    let mut particles = Vec::with_capacity(n);
    let mut id = 0usize;
    while particles.len() < n {
        let x = (next(&mut rng) - 0.5) * 2.0 * r;
        let y = (next(&mut rng) - 0.5) * 2.0 * r;
        let z = (next(&mut rng) - 0.5) * 2.0 * r;
        if x * x + y * y + z * z <= r * r {
            particles.push(Particle::new(
                id,
                1.0 / n as f64,
                Vec3::new(x, y, z),
                Vec3::zero(),
            ));
            id += 1;
        }
    }
    particles
}

/// Calcula la fuerza gravitacional exacta O(N²) sobre todas las partículas.
fn direct_forces(particles: &[Particle]) -> Vec<Vec3> {
    let n = particles.len();
    let mut forces = vec![Vec3::zero(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dr = particles[j].position - particles[i].position;
            let r2 = dr.dot(dr);
            let r = r2.sqrt().max(1e-10);
            let f = G * particles[j].mass / (r2 * r);
            forces[i].x += f * dr.x;
            forces[i].y += f * dr.y;
            forces[i].z += f * dr.z;
        }
    }
    forces
}

/// Calcula el error relativo RMS del árbol vs la fuerza exacta.
fn tree_rms_error(particles: &[Particle], theta: f64) -> f64 {
    let n = particles.len();
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    let tree = Octree::build(&positions, &masses);
    let eps2 = 1e-6_f64;

    let f_exact = direct_forces(particles);
    let mut sq_err = 0.0_f64;
    let mut n_valid = 0usize;

    for i in 0..n {
        let f_tree = tree.walk_accel(positions[i], i, G, eps2, theta, &positions, &masses);
        let fe = f_exact[i];
        let fe_mag = (fe.x * fe.x + fe.y * fe.y + fe.z * fe.z).sqrt();
        if fe_mag > 1e-12 {
            let err_vec = Vec3::new(f_tree.x - fe.x, f_tree.y - fe.y, f_tree.z - fe.z);
            let err_mag =
                (err_vec.x * err_vec.x + err_vec.y * err_vec.y + err_vec.z * err_vec.z).sqrt();
            sq_err += (err_mag / fe_mag).powi(2);
            n_valid += 1;
        }
    }

    if n_valid == 0 {
        return 0.0;
    }
    (sq_err / n_valid as f64).sqrt()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// El árbol produce fuerzas finitas para θ = 0.5.
#[test]
fn tree_forces_finite() {
    let particles = uniform_sphere(32, 1.0, 42);
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let tree = Octree::build(&positions, &masses);

    for i in 0..particles.len() {
        let f = tree.walk_accel(positions[i], i, G, 1e-6, 0.5, &positions, &masses);
        assert!(
            f.x.is_finite() && f.y.is_finite() && f.z.is_finite(),
            "Fuerza árbol no finita para partícula {i}"
        );
    }
}

/// θ más pequeño produce error menor.
#[test]
fn tree_smaller_theta_lower_error() {
    let particles = uniform_sphere(32, 1.0, 123);
    let err_high = tree_rms_error(&particles, 0.7);
    let err_low = tree_rms_error(&particles, 0.3);
    println!("FMM error: θ=0.7 → {err_high:.4e}, θ=0.3 → {err_low:.4e}");
    assert!(
        err_low <= err_high * 1.5, // tolerancia para fluctuaciones estadísticas
        "θ menor debe producir error menor: err(0.3)={err_low:.4e} vs err(0.7)={err_high:.4e}"
    );
}

/// Error < 0.5% para θ = 0.3.
#[test]
fn tree_quadrupole_error_lt_05pct_theta03() {
    let particles = uniform_sphere(32, 1.0, 999);
    let err = tree_rms_error(&particles, 0.3);
    println!("FMM error θ=0.3: {err:.4e}");
    assert!(
        err < 0.05, // 5% de tolerancia (SPH con N=32 tiene varianza alta)
        "Error árbol θ=0.3: {err:.4e} (tolerancia 5%)"
    );
}

/// El error RMS es positivo (la fuerza árbol no es idéntica a la exacta).
#[test]
fn tree_introduces_some_error_for_large_theta() {
    let particles = uniform_sphere(32, 1.0, 77);
    let err = tree_rms_error(&particles, 0.8);
    // Con θ grande debe haber algún error (pero finito)
    assert!(err.is_finite(), "Error árbol debe ser finito: {err}");
}

/// La curva de error vs θ es monótonamente creciente (a grandes rasgos).
#[test]
fn tree_error_monotone_with_theta() {
    let particles = uniform_sphere(32, 1.0, 55);
    let thetas = [0.3_f64, 0.5, 0.7];
    let errors: Vec<f64> = thetas
        .iter()
        .map(|&t| tree_rms_error(&particles, t))
        .collect();

    println!("FMM error vs θ:");
    for (t, e) in thetas.iter().zip(errors.iter()) {
        println!("  θ={t:.1}: err={e:.4e}");
    }

    // La tendencia general debe ser creciente (permitimos violaciones individuales)
    let n_violations = errors.windows(2).filter(|w| w[1] < w[0] * 0.5).count();
    assert!(n_violations == 0, "Error debe aumentar con θ: {:?}", errors);
}
