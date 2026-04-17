//! Validación de Yoshida 4º orden en la órbita de Kepler circular.
//!
//! El Sol (M=1, G=1) fijo en el origen y un planeta ligero en órbita circular
//! de radio r=1 forman un sistema integrable con cantidades conservadas:
//!
//! - Energía específica `E = ½v² - GM/r = −½`
//! - Momento angular específico `L = |r × v| = 1`
//! - Período `T = 2π`
//!
//! Para 1 período con `dt = T/200`, Yoshida 4º orden debe satisfacer:
//! - `|ΔE/E₀|` < 1e-9 (KDK típico: ~1e-4 a ~1e-5 en las mismas condiciones)
//! - `|ΔL/L₀|` < 1e-10
//! - Cierre orbital: `|r(T) − r(0)|` < 1e-4 (KDK típico: ~1e-2)
//!
//! Adicionalmente, ejecuta un barrido dt en 10 períodos y escribe a
//! `experiments/nbody/phase6_higher_order_integrator/results/kepler_convergence.csv`.
//!
//! Nota sobre Runge–Lenz: para una órbita circular exacta `|A|` ≈ 0 numéricamente
//! ruidoso (precesión → desviación acumulativa). Usamos cierre orbital como
//! métrica estructural equivalente, más robusta y físicamente equivalente
//! para órbitas cerradas.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_integrators::{leapfrog_kdk_step, yoshida4_kdk_step};
use std::fs;
use std::path::PathBuf;

const G: f64 = 1.0;
const M_SUN: f64 = 1.0;
const R0: f64 = 1.0;
const T_PERIOD: f64 = std::f64::consts::TAU;

fn force_kepler(parts: &[Particle], acc: &mut [Vec3]) {
    let r_vec = parts[0].position;
    let r = r_vec.norm();
    let r3 = r * r * r;
    acc[0] = r_vec * (-G * M_SUN / r3);
}

fn energy(p: &Particle) -> f64 {
    0.5 * p.velocity.dot(p.velocity) - G * M_SUN / p.position.norm()
}

fn angular_momentum_z(p: &Particle) -> f64 {
    p.position.x * p.velocity.y - p.position.y * p.velocity.x
}

fn initial_particle() -> Particle {
    Particle::new(
        0,
        1e-10,
        Vec3::new(R0, 0.0, 0.0),
        Vec3::new(0.0, (G * M_SUN / R0).sqrt(), 0.0),
    )
}

fn run_orbit<F>(dt: f64, n_periods: u32, mut one_step: F) -> (f64, f64, f64)
where
    F: FnMut(&mut [Particle], f64, &mut [Vec3]),
{
    let mut parts = vec![initial_particle()];
    let mut scratch = vec![Vec3::zero(); 1];
    let e0 = energy(&parts[0]);
    let l0 = angular_momentum_z(&parts[0]);
    let r0 = parts[0].position;

    let n_steps = ((T_PERIOD * n_periods as f64) / dt).round() as u64;
    for _ in 0..n_steps {
        one_step(&mut parts, dt, &mut scratch);
    }
    let de_rel = ((energy(&parts[0]) - e0) / e0).abs();
    let dl_rel = ((angular_momentum_z(&parts[0]) - l0) / l0).abs();
    let closure = (parts[0].position - r0).norm();
    (de_rel, dl_rel, closure)
}

fn results_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("../..")
        .join("experiments/nbody/phase6_higher_order_integrator/results")
}

#[test]
fn yoshida4_kepler_one_period_conservation() {
    let dt = T_PERIOD / 200.0;
    let (de_kdk, dl_kdk, close_kdk) =
        run_orbit(dt, 1, |p, dt, s| leapfrog_kdk_step(p, dt, s, force_kepler));
    let (de_yos, dl_yos, close_yos) = run_orbit(dt, 1, |p, dt, s| {
        yoshida4_kdk_step(p, dt, s, force_kepler)
    });

    println!("Kepler 1 período @ dt=T/200:");
    println!(
        "  KDK    : |ΔE/E|={de_kdk:.3e}  |ΔL/L|={dl_kdk:.3e}  |Δr|={close_kdk:.3e}"
    );
    println!(
        "  Yoshida: |ΔE/E|={de_yos:.3e}  |ΔL/L|={dl_yos:.3e}  |Δr|={close_yos:.3e}"
    );

    // En órbita circular exacta, |ΔE/E| de KDK ya satura a precisión de máquina
    // (~1e-14) a dt=T/200, por lo que no discrimina. El cierre orbital (error de
    // fase) sí discrimina claramente y es la métrica estructural robusta.
    assert!(
        de_yos < 1e-9,
        "|ΔE/E| Yoshida debe ser < 1e-9, got {de_yos:.3e}"
    );
    assert!(
        dl_yos < 1e-10,
        "|ΔL/L| Yoshida debe ser < 1e-10, got {dl_yos:.3e}"
    );
    assert!(
        close_yos < 1e-4,
        "Cierre orbital Yoshida debe ser < 1e-4, got {close_yos:.3e}"
    );
    assert!(
        close_yos * 100.0 < close_kdk,
        "Yoshida debería cerrar órbita ≥100× mejor que KDK: KDK={close_kdk:.3e}, Yos={close_yos:.3e}"
    );
}

#[test]
fn yoshida4_kepler_dt_sweep() {
    let dts = [
        T_PERIOD / 50.0,
        T_PERIOD / 100.0,
        T_PERIOD / 200.0,
        T_PERIOD / 400.0,
    ];
    let n_periods: u32 = 10;
    let dir = results_dir();
    let _ = fs::create_dir_all(&dir);
    let mut csv = String::from(
        "system,integrator,dt,n_periods,dE_rel_final,dL_rel_final,closure\n",
    );
    for &dt in &dts {
        let (de_k, dl_k, cl_k) = run_orbit(dt, n_periods, |p, dt, s| {
            leapfrog_kdk_step(p, dt, s, force_kepler)
        });
        let (de_y, dl_y, cl_y) = run_orbit(dt, n_periods, |p, dt, s| {
            yoshida4_kdk_step(p, dt, s, force_kepler)
        });
        println!(
            "dt={dt:.4e} KDK: |ΔE/E|={de_k:.3e} |ΔL/L|={dl_k:.3e} close={cl_k:.3e}"
        );
        println!(
            "dt={dt:.4e} YOS: |ΔE/E|={de_y:.3e} |ΔL/L|={dl_y:.3e} close={cl_y:.3e}"
        );
        csv.push_str(&format!(
            "kepler,leapfrog,{dt},{n_periods},{de_k},{dl_k},{cl_k}\n"
        ));
        csv.push_str(&format!(
            "kepler,yoshida4,{dt},{n_periods},{de_y},{dl_y},{cl_y}\n"
        ));
    }
    let _ = fs::write(dir.join("kepler_convergence.csv"), csv);
}
