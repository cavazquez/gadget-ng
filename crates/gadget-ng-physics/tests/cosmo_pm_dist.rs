//! Tests de correctitud del path PM distribuido (Fase 19).
//!
//! Validan que el pipeline distribuido (deposit_local → allreduce → solve → interpolate)
//! produce resultados idénticos al PM serial, que la masa se conserva bajo la reducción,
//! y que `allreduce_sum_f64_slice` en SerialRuntime es un no-op.

use gadget_ng_core::{CosmologyParams, Vec3};
use gadget_ng_parallel::SerialRuntime;
use gadget_ng_parallel::ParallelRuntime;
use gadget_ng_pm::{cic, distributed as pm_dist, fft_poisson};

// ── Test 1: allreduce_sum_f64_slice en serial es identidad ───────────────────

#[test]
fn allreduce_sum_slice_is_noop_in_serial() {
    let rt = SerialRuntime;
    let original = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let mut buf = original.clone();
    rt.allreduce_sum_f64_slice(&mut buf);
    // En serial no hay otros rangos: el buffer debe quedar sin cambios.
    for (a, b) in buf.iter().zip(original.iter()) {
        assert!(
            (a - b).abs() < 1e-15,
            "allreduce serial modificó el buffer: {a} != {b}"
        );
    }
}

// ── Test 2: deposit_local produce el mismo resultado que cic::assign ─────────

#[test]
fn deposit_local_matches_full_assign() {
    let nm = 16usize;
    let box_size = 1.0_f64;
    let positions: Vec<Vec3> = (0..50)
        .map(|i| {
            let t = i as f64 / 50.0;
            Vec3::new(
                (t * 7.3).rem_euclid(1.0),
                (t * 3.7 + 0.1).rem_euclid(1.0),
                (t * 5.1 + 0.5).rem_euclid(1.0),
            )
        })
        .collect();
    let masses = vec![0.02_f64; 50];

    let via_deposit = pm_dist::deposit_local(&positions, &masses, box_size, nm);
    let via_assign = cic::assign(&positions, &masses, box_size, nm);

    assert_eq!(via_deposit.len(), nm * nm * nm);
    for (i, (a, b)) in via_deposit.iter().zip(via_assign.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-15,
            "celda {i}: deposit_local={a:.15e}, assign={b:.15e}"
        );
    }
}

// ── Test 3: conservación de masa tras reducción simulada entre dos "ranks" ───

#[test]
fn distributed_pm_mass_conservation() {
    let nm = 16usize;
    let box_size = 1.0_f64;

    // Simulamos que el rank 0 tiene estas partículas:
    let pos_r0 = vec![
        Vec3::new(0.1, 0.2, 0.3),
        Vec3::new(0.9, 0.9, 0.9),
        Vec3::new(0.5, 0.0, 0.5),
    ];
    let mass_r0 = vec![1.0_f64, 2.0, 0.5];

    // Y el rank 1 tiene estas:
    let pos_r1 = vec![
        Vec3::new(0.3, 0.7, 0.1),
        Vec3::new(0.6, 0.4, 0.8),
        Vec3::new(0.15, 0.85, 0.45),
        Vec3::new(0.0, 0.0, 0.0),
    ];
    let mass_r1 = vec![3.0_f64, 0.5, 1.5, 0.25];

    let mut grid_r0 = pm_dist::deposit_local(&pos_r0, &mass_r0, box_size, nm);
    let grid_r1 = pm_dist::deposit_local(&pos_r1, &mass_r1, box_size, nm);

    // Simulamos allreduce_sum_f64_slice sumando los grids.
    for (a, b) in grid_r0.iter_mut().zip(grid_r1.iter()) {
        *a += b;
    }

    let total_mass_grid: f64 = grid_r0.iter().sum();
    let total_mass_particles: f64 = mass_r0.iter().chain(mass_r1.iter()).sum();

    assert!(
        (total_mass_grid - total_mass_particles).abs() < 1e-12,
        "conservación de masa fallida: grid={total_mass_grid:.12e}, partículas={total_mass_particles:.12e}"
    );
}

// ── Test 4: fuerzas PM distribuidas ≡ PM serial ──────────────────────────────

#[test]
fn distributed_forces_match_serial_pm() {
    let nm = 16usize;
    let box_size = 1.0_f64;
    let g = 1.0_f64;

    let positions: Vec<Vec3> = vec![
        Vec3::new(0.1, 0.2, 0.3),
        Vec3::new(0.8, 0.5, 0.7),
        Vec3::new(0.4, 0.4, 0.4),
        Vec3::new(0.9, 0.1, 0.6),
    ];
    let masses = vec![1.0_f64, 2.0, 1.5, 0.5];

    // Path serial clásico: deposit todas las partículas, solve, interpolate.
    let rho_serial = cic::assign(&positions, &masses, box_size, nm);
    let [fx_s, fy_s, fz_s] = fft_poisson::solve_forces(&rho_serial, g, nm, box_size);
    let acc_serial = cic::interpolate(&fx_s, &fy_s, &fz_s, &positions, box_size, nm);

    // Path distribuido (simulando 2 ranks: primeras 2 partículas vs últimas 2).
    let pos_r0 = &positions[..2];
    let mass_r0 = &masses[..2];
    let pos_r1 = &positions[2..];
    let mass_r1 = &masses[2..];

    let mut grid_r0 = pm_dist::deposit_local(pos_r0, mass_r0, box_size, nm);
    let grid_r1 = pm_dist::deposit_local(pos_r1, mass_r1, box_size, nm);

    // Simula allreduce_sum.
    for (a, b) in grid_r0.iter_mut().zip(grid_r1.iter()) {
        *a += b;
    }

    let [fx_d, fy_d, fz_d] = pm_dist::forces_from_global_density(&grid_r0, g, nm, box_size);

    // Cada rank interpola solo sus partículas.
    let acc_r0 = pm_dist::interpolate_local(pos_r0, &fx_d, &fy_d, &fz_d, nm, box_size);
    let acc_r1 = pm_dist::interpolate_local(pos_r1, &fx_d, &fy_d, &fz_d, nm, box_size);

    // Combinar resultados.
    let acc_dist: Vec<Vec3> = acc_r0.into_iter().chain(acc_r1.into_iter()).collect();

    // Las fuerzas distribuidas deben coincidir bit a bit con el resultado serial.
    for (i, (s, d)) in acc_serial.iter().zip(acc_dist.iter()).enumerate() {
        let err = ((s.x - d.x).powi(2) + (s.y - d.y).powi(2) + (s.z - d.z).powi(2)).sqrt();
        assert!(
            err < 1e-12,
            "partícula {i}: serial=({:.6e},{:.6e},{:.6e}) dist=({:.6e},{:.6e},{:.6e}) err={err:.2e}",
            s.x, s.y, s.z, d.x, d.y, d.z
        );
    }
}

// ── Test 5: partícula en borde del slab deposita correctamente (periódico) ───

#[test]
fn distributed_border_particle_deposit() {
    // Una partícula exactamente en x=0 (borde izquierdo del grid):
    // con CIC periódico, su masa se reparte entre las celdas iz=0 e iz=nm-1
    // en la dirección z (o x según convención). Verificamos que la masa
    // total se conserva.
    let nm = 8usize;
    let box_size = 1.0_f64;
    let positions = vec![Vec3::new(0.0, 0.0, 0.0)];
    let masses = vec![1.0_f64];

    let grid = pm_dist::deposit_local(&positions, &masses, box_size, nm);
    let total: f64 = grid.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-12,
        "masa en borde: total={total:.12e}"
    );
}

// ── Test 6: run corto EdS con PM distribuido no explota ──────────────────────

#[test]
fn distributed_pm_no_explosion_eds() {
    use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
    use gadget_ng_core::Particle;

    let nm = 8usize;
    let box_size = 1.0_f64;
    let g = 43.009_f64;  // G en unidades N-body típicas

    let cosmo = CosmologyParams::new(1.0, 0.0, 70.0);
    let mut a = 0.02_f64;
    let dt = 0.005_f64;
    let num_steps = 20;

    // N=16 partículas en red casi uniforme + perturbaciones.
    let n = 16usize;
    let mut particles: Vec<Particle> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let perturb = 0.02 * (t * 13.7 + 1.0).sin();
            Particle {
                global_id: i,
                mass: 1.0 / n as f64,
                position: Vec3::new(
                    (t + perturb).rem_euclid(1.0),
                    (t * 2.3 + perturb * 0.5).rem_euclid(1.0),
                    (t * 1.7 + perturb * 0.3).rem_euclid(1.0),
                ),
                velocity: Vec3::new(0.0, 0.0, 0.0),
                acceleration: Vec3::zero(),
            }
        })
        .collect();

    let mut scratch: Vec<Vec3> = vec![Vec3::zero(); n];

    for _step in 0..num_steps {
        let g_cosmo = g / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors { drift, kick_half, kick_half2 };

        leapfrog_cosmo_kdk_step(&mut particles, cf, &mut scratch, |parts, acc| {
            let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
            let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
            let mut density = pm_dist::deposit_local(&pos, &mass, box_size, nm);
            // En serial, allreduce es no-op; aquí verificamos solo el pipeline.
            let rt = SerialRuntime;
            rt.allreduce_sum_f64_slice(&mut density);
            let [fx, fy, fz] = pm_dist::forces_from_global_density(&density, g_cosmo, nm, box_size);
            let accels = pm_dist::interpolate_local(&pos, &fx, &fy, &fz, nm, box_size);
            for (a, v) in acc.iter_mut().zip(accels.iter()) {
                *a = *v;
            }
        });

        // Wrap periódico.
        for p in particles.iter_mut() {
            p.position.x = p.position.x.rem_euclid(box_size);
            p.position.y = p.position.y.rem_euclid(box_size);
            p.position.z = p.position.z.rem_euclid(box_size);
        }

        a = cosmo.advance_a(a, dt);

        // Verificar estabilidad: sin NaN ni Inf.
        for (i, p) in particles.iter().enumerate() {
            assert!(
                p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
                "partícula {i}: posición no finita en paso {_step}, a={a:.4}"
            );
            assert!(
                p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
                "partícula {i}: velocidad no finita en paso {_step}, a={a:.4}"
            );
        }
    }
}

// ── Test 7: solve de Poisson distribuido para un modo sinusoidal conocido ────

#[test]
fn distributed_poisson_sanity_sinusoidal_mode() {
    // Modo k=(1,0,0): ρ(x) = ρ_mean + A·cos(2π x / L).
    // La fuerza esperada: Fx ∝ +sin(2πx/L) [apuntando hacia el mínimo de densidad].
    // En x=L/4 (mínimo de densidad): fuerza positiva (+x).
    // En x=3L/4 (mínimo de densidad): fuerza negativa (-x).
    let nm = 32usize;
    let box_size = 1.0_f64;
    let g = 1.0_f64;

    // Construimos la densidad analíticamente con el indexado correcto de fft_poisson.
    // Convención: flat_idx = iz * nm² + iy * nm + ix (ix varía más rápido).
    let nm2 = nm * nm;
    let mut density = vec![0.0_f64; nm * nm * nm];
    let amplitude = 1.0_f64;
    let rho_mean = 1.0_f64;
    for iz in 0..nm {
        for iy in 0..nm {
            for ix in 0..nm {
                let x = ix as f64 / nm as f64;
                density[iz * nm2 + iy * nm + ix] =
                    rho_mean + amplitude * (2.0 * std::f64::consts::PI * x).cos();
            }
        }
    }

    // Simula el pipeline distribuido: deposita + reduce (trivial: un solo "rank") + solve.
    let rt = SerialRuntime;
    rt.allreduce_sum_f64_slice(&mut density);  // no-op en serial
    let [fx, _fy, _fz] = pm_dist::forces_from_global_density(&density, g, nm, box_size);

    // En ix=nm/4 (x≈L/4, cruce de densidad en descenso): fuerza debería ser ≈0
    // pues cos(2π·1/4) = 0 → punto de inflexión, pero la fuerza es ∝ sin(2πx/L).
    // En ix=nm/4: sin(2π/4) = 1 → fuerza positiva (máxima).
    // En ix=3nm/4: sin(2π·3/4) = -1 → fuerza negativa (mínima).
    let ix_quarter = nm / 4;       // sin = +1, fuerza máxima positiva
    let ix_3quarter = 3 * nm / 4;  // sin = -1, fuerza máxima negativa

    let iy = 0usize;
    let iz = 0usize;
    let fx_quarter = fx[iz * nm2 + iy * nm + ix_quarter];
    let fx_3quarter = fx[iz * nm2 + iy * nm + ix_3quarter];

    // ρ(x) = A·cos(2πx/L) → Φ(x) ∝ -cos(2πx/L) → F_x = -∂Φ/∂x ∝ -sin(2πx/L).
    // En x=L/4: F_x ∝ -sin(π/2) = -1 → fuerza NEGATIVA (apunta hacia max de densidad en x=0).
    // En x=3L/4: F_x ∝ -sin(3π/2) = +1 → fuerza POSITIVA (apunta hacia max en x=1=0 via PBC).
    assert!(
        fx_quarter < 0.0,
        "fuerza en x=L/4 debería ser negativa (hacia max en x=0): fx={fx_quarter:.6e}"
    );
    assert!(
        fx_3quarter > 0.0,
        "fuerza en x=3L/4 debería ser positiva (hacia max en x=L via PBC): fx={fx_3quarter:.6e}"
    );
    // Antisimetría: |fx(nm/4)| ≈ |fx(3nm/4)|.
    let ratio = (-fx_quarter) / fx_3quarter;
    assert!(
        (ratio - 1.0).abs() < 0.01,
        "las fuerzas deben ser antimétricas: ratio={ratio:.4}"
    );
}
