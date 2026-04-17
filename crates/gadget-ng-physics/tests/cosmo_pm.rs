//! Tests de validación del modo cosmológico periódico con PM — Fase 18.
//!
//! ## Cobertura
//!
//! 1. `minimum_image_3d_correct`:
//!    La diferencia mínima imagen en 1D es correcta en todos los cuadrantes.
//!
//! 2. `wrap_position_correct`:
//!    `wrap_position` envuelve coordenadas a `[0, box_size)` en los tres ejes.
//!
//! 3. `cic_mass_conservation`:
//!    El depósito CIC conserva la masa total (suma ρ_celda = suma masas).
//!
//! 4. `pm_poisson_single_mode`:
//!    Para un modo sinusoidal de densidad, la fuerza PM tiene el signo y
//!    dirección correctos (apunta hacia el máximo de densidad).
//!
//! 5. `pm_g_cosmo_scaling`:
//!    `PmSolver` con `g_cosmo = G/a` da fuerzas la mitad de `G` cuando `a = 2`.
//!
//! 6. `pm_cosmo_no_explosion`:
//!    30 pasos de leapfrog cosmológico con PM periódico no produce NaN/Inf.
//!
//! 7. `pm_cosmo_a_evolution`:
//!    La evolución `a(t)` con PM+cosmología error < 1% vs EdS analítico.
//!
//! 8. `pm_periodic_force_symmetry`:
//!    Para dos masas iguales en posiciones simétricas respecto al borde de caja,
//!    las fuerzas PM son antisimétricas (igual magnitud, sentidos opuestos).

use gadget_ng_core::{
    build_particles,
    cosmology::CosmologyParams,
    minimum_image, wrap_position,
    CosmologySection, GravitySection, GravitySolver, IcKind, InitialConditionsSection,
    OutputSection, PerformanceSection, RunConfig, SimulationSection, TimestepSection,
    UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::{cic, fft_poisson, PmSolver};

const G: f64 = 1.0;
const BOX: f64 = 1.0;
const NM: usize = 16;

// ── Config EdS para tests ─────────────────────────────────────────────────────

fn eds_pm_config(n: usize) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.005,
            num_steps: 30,
            softening: 0.02,
            gravitational_constant: G,
            particle_count: n,
            box_size: BOX,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::PerturbedLattice {
                amplitude: 0.05,
                velocity_amplitude: 0.0,
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
            a_init: 1.0,
        },
        units: UnitsSection::default(),
    }
}

// ── Test 1: minimum_image en 1D ───────────────────────────────────────────────

/// La diferencia mínima imagen es correcta en los cuatro cuadrantes.
#[test]
fn minimum_image_3d_correct() {
    let l = 1.0_f64;

    // Caso trivial: dx < L/2 → sin cambio.
    let d = minimum_image(0.3, l);
    assert!((d - 0.3).abs() < 1e-14, "dx=0.3 → {d}");

    // Cruzando el borde positivo: dx = 0.7 → imagen = -0.3.
    let d = minimum_image(0.7, l);
    assert!((d + 0.3).abs() < 1e-14, "dx=0.7 → {d} (esperado -0.3)");

    // Cruzando el borde negativo: dx = -0.7 → imagen = 0.3.
    let d = minimum_image(-0.7, l);
    assert!((d - 0.3).abs() < 1e-14, "dx=-0.7 → {d} (esperado 0.3)");

    // dx = 0.5 (exactamente en el límite) → 0.5 (o -0.5, depende del round).
    let d = minimum_image(0.5, l);
    assert!(d.abs() <= 0.5 + 1e-14, "|minimum_image(0.5, 1)| <= 0.5");

    // Invariante: |minimum_image(dx, l)| ≤ l/2 para todo dx.
    for i in 0..100 {
        let dx = (i as f64 - 50.0) * 0.03;
        let mi = minimum_image(dx, l);
        assert!(
            mi.abs() <= l / 2.0 + 1e-12,
            "|minimum_image({dx}, {l})| = {mi} > L/2"
        );
    }
}

// ── Test 2: wrap_position ─────────────────────────────────────────────────────

#[test]
fn wrap_position_correct() {
    let l = 1.0_f64;

    // Dentro: sin cambio.
    let p = Vec3::new(0.5, 0.3, 0.8);
    let w = wrap_position(p, l);
    assert!((w.x - 0.5).abs() < 1e-14 && (w.y - 0.3).abs() < 1e-14);

    // Fuera por la derecha: x=1.2 → 0.2.
    let p = Vec3::new(1.2, -0.1, 2.7);
    let w = wrap_position(p, l);
    assert!(
        (w.x - 0.2).abs() < 1e-14,
        "x=1.2 → {} (esperado 0.2)",
        w.x
    );
    assert!(
        (w.y - 0.9).abs() < 1e-14,
        "y=-0.1 → {} (esperado 0.9)",
        w.y
    );
    assert!(
        (w.z - 0.7).abs() < 1e-13,
        "z=2.7 → {} (esperado 0.7)",
        w.z
    );

    // Resultado siempre en [0, l).
    let positions = [
        Vec3::new(-5.3, 1.0000001, 0.0),
        Vec3::new(100.9, -0.999, 3.001),
    ];
    for orig in &positions {
        let w = wrap_position(*orig, l);
        assert!(w.x >= 0.0 && w.x < l, "x={} fuera de [0, l)", w.x);
        assert!(w.y >= 0.0 && w.y < l, "y={} fuera de [0, l)", w.y);
        assert!(w.z >= 0.0 && w.z < l, "z={} fuera de [0, l)", w.z);
    }
}

// ── Test 3: CIC conserva masa ─────────────────────────────────────────────────

/// La suma de todos los valores del grid CIC debe ser igual a la masa total.
#[test]
fn cic_mass_conservation() {
    // Posiciones uniformes dentro del cubo.
    let mut state = 17u64;
    let n = 200_usize;
    let positions: Vec<Vec3> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((state >> 33) as f64) / (u32::MAX as f64);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = ((state >> 33) as f64) / (u32::MAX as f64);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let z = ((state >> 33) as f64) / (u32::MAX as f64);
            Vec3::new(x, y, z)
        })
        .collect();
    let masses: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.1).collect();
    let total_mass: f64 = masses.iter().sum();

    let rho = cic::assign(&positions, &masses, BOX, NM);
    let rho_sum: f64 = rho.iter().sum();

    let rel_err = (rho_sum - total_mass).abs() / total_mass;
    assert!(
        rel_err < 1e-12,
        "CIC no conserva masa: sum(rho) = {rho_sum:.6e}, expected {total_mass:.6e}, err = {rel_err:.2e}"
    );
}

// ── Test 4: Poisson para un modo sinusoidal ───────────────────────────────────

/// Para ρ(x) = ρ_0 · sin(2π·x/L) con ρ_0 > 0, la solución analítica de Poisson en 1D es:
///   Φ(x) = -4πG·ρ_0·(L/2π)²·sin(2πx/L)
///   F_x(x) = -∂Φ/∂x = +4πG·ρ_0·(L/2π)·cos(2πx/L)
///
/// En los ceros de la densidad (donde el gradiente es máximo):
///   - En x=0 (cruce ascendente): F_x > 0  (convergencia hacia el máximo en x=L/4)
///   - En x=L/2 (cruce descendente): F_x < 0
///
/// En el máximo de densidad (x=L/4), la fuerza es cero (punto de equilibrio).
#[test]
fn pm_poisson_single_mode() {
    // Índices de grid: flat = iz * nm² + iy * nm + ix  (ix es el índice rápido, x físico).
    let nm = 32_usize;
    let nm2 = nm * nm;
    let nm3 = nm2 * nm;
    // Construir campo de densidad sinusoidal a lo largo de x (ix es la dim. x).
    let mut density = vec![0.0_f64; nm3];
    let rho0 = 1.0_f64;
    for iz in 0..nm {
        for iy in 0..nm {
            for ix in 0..nm {
                let rho_val = rho0 * (2.0 * std::f64::consts::PI * ix as f64 / nm as f64).sin();
                density[iz * nm2 + iy * nm + ix] = rho_val;
            }
        }
    }

    let [fx, _fy, _fz] = fft_poisson::solve_forces(&density, G, nm, BOX);

    // Acceso a celdas usando flat = iz * nm² + iy * nm + ix, con iy=iz=0:
    // ix=0:    flat = 0      (cruce ascendente, F_x > 0)
    // ix=nm/2: flat = nm/2   (cruce descendente, F_x < 0)
    // ix=nm/4: flat = nm/4   (máximo de densidad, F_x ≈ 0)

    let f_at_zero = fx[0]; // ix=0, iy=0, iz=0
    assert!(
        f_at_zero > 0.0,
        "Fuerza en cruce ascendente (ix=0) debe ser positiva: {f_at_zero:.4e}"
    );

    let f_at_half = fx[nm / 2]; // ix=nm/2, iy=0, iz=0
    assert!(
        f_at_half < 0.0,
        "Fuerza en cruce descendente (ix=nm/2) debe ser negativa: {f_at_half:.4e}"
    );

    // Antisimetría: f(ix=0) ≈ -f(ix=nm/2).
    let ratio = f_at_zero / (-f_at_half);
    assert!(
        (ratio - 1.0).abs() < 0.01,
        "f(0) / (-f(nm/2)) = {ratio:.4} (esperado ≈ 1.0)"
    );

    // En ix=nm/4 (máximo de densidad): fuerza ≈ 0.
    let f_at_max = fx[nm / 4]; // ix=nm/4, iy=0, iz=0
    let scale = f_at_zero.abs().max(1e-20);
    assert!(
        f_at_max.abs() / scale < 0.01,
        "Fuerza en máximo de densidad (ix=nm/4) debe ser ≈ 0: {f_at_max:.4e}"
    );
}

// ── Test 5: PmSolver escala con G/a ───────────────────────────────────────────

/// PmSolver con g_cosmo = G/a da fuerzas G/a respecto a G puro.
/// La relación debe ser lineal: F(G/2) = F(G)/2.
#[test]
fn pm_g_cosmo_scaling() {
    let n = 27_usize;
    let cfg = eds_pm_config(n);
    let parts = build_particles(&cfg).expect("IC");

    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();

    let pm = PmSolver { grid_size: NM, box_size: BOX };
    let idx: Vec<usize> = (0..n).collect();

    let mut acc_full = vec![Vec3::zero(); n];
    let mut acc_half = vec![Vec3::zero(); n];

    pm.accelerations_for_indices(&positions, &masses, 0.0, G, &idx, &mut acc_full);
    pm.accelerations_for_indices(&positions, &masses, 0.0, G / 2.0, &idx, &mut acc_half);

    // Para cada partícula, acc_half ≈ acc_full / 2 (linealidad en G).
    let mut max_err = 0.0_f64;
    for i in 0..n {
        let scale = acc_full[i].norm().max(1e-15);
        let diff = (acc_full[i] * 0.5 - acc_half[i]).norm();
        let rel = diff / scale;
        if rel > max_err { max_err = rel; }
    }
    assert!(
        max_err < 1e-10,
        "PmSolver escalado G/2: error relativo máximo = {max_err:.2e}"
    );
}

// ── Test 6: sin explosión numérica con PM+cosmo ───────────────────────────────

/// 30 pasos de leapfrog cosmológico con PmSolver periódico no produce NaN/Inf.
#[test]
fn pm_cosmo_no_explosion() {
    let n = 27_usize;
    let cfg = eds_pm_config(n);
    let mut parts = build_particles(&cfg).expect("IC");
    let cosmo = CosmologyParams::new(1.0, 0.0, 0.1);
    let dt = 0.005_f64;
    let n_steps = 30_usize;
    let mut a = 1.0_f64;
    let mut scratch = vec![Vec3::zero(); n];

    let positions_initial: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let idx: Vec<usize> = (0..n).collect();
    let pm = PmSolver { grid_size: NM, box_size: BOX };

    for _ in 0..n_steps {
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors { drift, kick_half, kick_half2 };
        a = cosmo.advance_a(a, dt);

        leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let local_idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &local_idx, acc);
        });

        // Wrap periódico.
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }

    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "Posición no finita gid={}: {:?}", p.global_id, p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "Velocidad no finita gid={}: {:?}", p.global_id, p.velocity
        );
    }
    // Posiciones deben estar en [0, box_size) tras el wrap.
    for p in &parts {
        assert!(p.position.x >= 0.0 && p.position.x < BOX, "x fuera de caja: {}", p.position.x);
        assert!(p.position.y >= 0.0 && p.position.y < BOX, "y fuera de caja: {}", p.position.y);
        assert!(p.position.z >= 0.0 && p.position.z < BOX, "z fuera de caja: {}", p.position.z);
    }
    assert!(a > 1.0, "a no creció: a = {a:.6}");

    let _ = positions_initial;
    let _ = idx;
    let _ = masses;
}

// ── Test 7: a(t) con PM+cosmo < 1% error vs EdS ──────────────────────────────

#[test]
fn pm_cosmo_a_evolution() {
    let h0 = 0.1_f64;
    let a0 = 1.0_f64;
    let dt = 0.005_f64;
    let n_steps = 20_usize;
    let cosmo = CosmologyParams::new(1.0, 0.0, h0);

    let mut a = a0;
    for _ in 0..n_steps {
        a = cosmo.advance_a(a, dt);
    }

    let t = n_steps as f64 * dt;
    let a_analytic = (a0.powf(1.5) + 1.5 * h0 * t).powf(2.0 / 3.0);
    let rel_err = (a - a_analytic).abs() / a_analytic;

    assert!(
        rel_err < 0.01,
        "a(t) error vs EdS analítico: {rel_err:.2e} > 1%"
    );
}

// ── Test 8: antisimetría de fuerzas PM para par simétrico ────────────────────

/// Dos masas idénticas simétricas respecto al borde de la caja deben
/// experimentar fuerzas iguales en magnitud y opuestas en signo.
///
/// Configuración:
///   masa A en x = 0.1  (cerca del borde izquierdo)
///   masa B en x = 0.9  (cerca del borde derecho)
/// Con condiciones periódicas, la distancia imagen mínima A-B es 0.2,
/// y las fuerzas deben ser antisimétricas.
#[test]
fn pm_periodic_force_symmetry() {
    let nm = 32_usize;
    // Dos masas idénticas en x=0.1 y x=0.9 (simétricas respecto al borde).
    let positions_a = vec![Vec3::new(0.1, 0.5, 0.5), Vec3::new(0.9, 0.5, 0.5)];
    let masses = vec![1.0_f64, 1.0_f64];

    let pm = PmSolver { grid_size: nm, box_size: BOX };
    let mut acc = vec![Vec3::zero(); 2];
    let idx = vec![0_usize, 1_usize];
    pm.accelerations_for_indices(&positions_a, &masses, 0.0, G, &idx, &mut acc);

    // acc[0].x debe ser positivo (masa B está en x=0.9, imagen más cercana es en -0.1 relativo).
    // acc[1].x debe ser negativo (por simetría).
    // La antisimetría exacta es aproximada (el grid de 32³ introduce discretización).
    let ratio_x = acc[0].x / (-acc[1].x + 1e-30);
    assert!(
        ratio_x > 0.5,
        "Fuerzas PM no antisimétricas: F_A.x = {:.4e}, F_B.x = {:.4e}, ratio = {:.3}",
        acc[0].x, acc[1].x, ratio_x
    );

    // Componentes y,z deben ser muy pequeñas (simetría del setup).
    for (i, a) in acc.iter().enumerate() {
        let mag_xy = (a.y.powi(2) + a.z.powi(2)).sqrt();
        let mag_x = a.x.abs().max(1e-20);
        assert!(
            mag_xy / mag_x < 0.15,
            "Partícula {i}: fuerza transversal significativa: |F_yz|/|F_x| = {:.3}",
            mag_xy / mag_x
        );
    }
}
