//! Fase 20: Tests de correctitud del PM slab distribuido.
//!
//! Cubre:
//! 1. `slab_layout_covers_all_planes` — SlabLayout correcto para distintos nm/P
//! 2. `deposit_slab_mass_conservation` — masa total en buffer extendido = suma de masas
//! 3. `density_halo_exchange_conserves_mass` — tras exchange, masa total en slab = correcto (P=1 no-op)
//! 4. `border_particle_deposit_correct` — partícula en z=slab_lo deposita correctamente al ghost right
//! 5. `alltoall_transpose_roundtrip` — fwd + bwd transpose = identidad (P=1)
//! 6. `slab_solve_matches_serial_pm` — solve distribuido ≡ PM serial para N pequeño
//! 7. `slab_poisson_sanity_mode` — modo sinusoidal: fuerza tiene signo correcto
//! 8. `distributed_pm_no_explosion_slab` — run corto EdS + slab sin NaN/Inf

use gadget_ng_core::Vec3;
use gadget_ng_parallel::SerialRuntime;
use gadget_ng_pm::{
    cic, fft_poisson,
    slab_fft::{solve_forces_slab, SlabLayout},
    slab_pm,
};
use std::f64::consts::PI;

// ── Utilidades ────────────────────────────────────────────────────────────────

fn uniform_grid_positions(n_side: usize, box_size: f64) -> Vec<Vec3> {
    let dx = box_size / n_side as f64;
    let mut pos = Vec::new();
    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                pos.push(Vec3::new(
                    (ix as f64 + 0.5) * dx,
                    (iy as f64 + 0.5) * dx,
                    (iz as f64 + 0.5) * dx,
                ));
            }
        }
    }
    pos
}

// ── Test 1: SlabLayout cubre todos los planos ─────────────────────────────────

#[test]
fn slab_layout_covers_all_planes() {
    for nm in [8, 16, 32] {
        for n_ranks in [1, 2, 4, 8] {
            if nm % n_ranks != 0 {
                continue;
            }
            let mut covered = vec![false; nm];
            for r in 0..n_ranks {
                let layout = SlabLayout::new(nm, r, n_ranks);
                assert_eq!(layout.nz_local, nm / n_ranks);
                assert_eq!(layout.z_lo_idx, r * layout.nz_local);
                for iz in layout.z_lo_idx..layout.z_lo_idx + layout.nz_local {
                    assert!(
                        !covered[iz],
                        "plano {iz} cubierto dos veces (nm={nm} P={n_ranks})"
                    );
                    covered[iz] = true;
                }
            }
            assert!(
                covered.iter().all(|&c| c),
                "no todos los planos cubiertos nm={nm} P={n_ranks}"
            );
        }
    }
}

// ── Test 2: Conservación de masa en depósito slab ─────────────────────────────

#[test]
fn deposit_slab_mass_conservation() {
    let nm = 8usize;
    let layout = SlabLayout::new(nm, 0, 1); // P=1 = grid completo
    let box_size = 1.0;
    let pos = uniform_grid_positions(4, box_size); // 64 partículas
    let masses = vec![2.5_f64; pos.len()];

    let density = slab_pm::deposit_slab_extended(&pos, &masses, &layout, box_size);
    let total: f64 = density.iter().sum();
    let expected: f64 = masses.iter().sum();

    assert!(
        (total - expected).abs() < 1e-10,
        "conservación de masa fallida: {total:.6} vs {expected:.6}"
    );
}

// ── Test 3: Intercambio de halos conserva masa (P=1 no-op) ───────────────────

#[test]
fn density_halo_exchange_conserves_mass_serial() {
    let nm = 8usize;
    let layout = SlabLayout::new(nm, 0, 1);
    let box_size = 1.0;
    let pos = uniform_grid_positions(4, box_size);
    let masses = vec![1.0_f64; pos.len()];

    let mut density = slab_pm::deposit_slab_extended(&pos, &masses, &layout, box_size);
    let mass_before: f64 = density.iter().sum();

    // P=1: exchange_density_halos_z es no-op
    slab_pm::exchange_density_halos_z(&mut density, &layout, &SerialRuntime);
    let mass_after: f64 = density.iter().sum();

    assert!(
        (mass_after - mass_before).abs() < 1e-10,
        "exchange modificó masa: antes={mass_before:.6} después={mass_after:.6}"
    );
}

// ── Test 4: Partícula en borde Z deposita en ghost right ──────────────────────

#[test]
fn border_particle_deposit_correct() {
    // P=2 simulado: rank 0 con nz_local=4 de nm=8
    let nm = 8usize;
    let layout = SlabLayout::new(nm, 0, 2); // rank 0: z planos 0..3
    let box_size = 1.0_f64;
    let nz = layout.nz_local; // 4

    // Partícula justo en el borde z del slab de rank 0.
    // Coordenada z ~ nz * cell_size - epsilon para que iz0 = nz-1.
    let cell = box_size / nm as f64;
    let z_border = (nz as f64 - 0.01) * cell; // iz0 = nz-1, contribución CIC a iz1 = nz (ghost)
    let pos = vec![Vec3::new(0.5 * cell, 0.5 * cell, z_border)];
    let masses = vec![1.0_f64];

    let density = slab_pm::deposit_slab_extended(&pos, &masses, &layout, box_size);
    let nm2 = nm * nm;

    // Verificar que el ghost right (plano nz_local) tiene alguna masa.
    let ghost_mass: f64 = density[nz * nm2..(nz + 1) * nm2].iter().sum();
    let owned_mass: f64 = density[0..nz * nm2].iter().sum();
    let total = ghost_mass + owned_mass;

    assert!(
        ghost_mass > 0.0,
        "partícula en borde no depositó en ghost right: ghost_mass={ghost_mass:.6}"
    );
    assert!(
        (total - 1.0_f64).abs() < 1e-10,
        "masa total incorrecta: {total:.6}"
    );
}

// ── Test 5: Transpose fwd+bwd = identidad (P=1) via uniformidad ──────────────

/// El roundtrip correcto del transpose requiere que `solve_forces_slab` con P=1
/// reproduzca exactamente la solución serial. Una densidad uniforme debe dar
/// fuerzas ≈ 0 en todo el grid (verificación de que el transpose no introduce error).
#[test]
fn alltoall_transpose_roundtrip_p1() {
    let nm = 8usize;
    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;
    let box_size = 1.0_f64;
    let g = 1.0_f64;

    // Densidad uniforme → fuerzas = 0 (k=0 mode suprimido).
    let density = vec![1.0_f64; nm * nm * nm];
    let [fx, fy, fz] = solve_forces_slab(&density, &layout, g, box_size, None, &rt);

    for i in 0..nm * nm * nm {
        assert!(
            fx[i].abs() < 1e-10 && fy[i].abs() < 1e-10 && fz[i].abs() < 1e-10,
            "densidad uniforme debe dar fuerza≈0 en celda {i}: fx={:.2e} fy={:.2e} fz={:.2e}",
            fx[i],
            fy[i],
            fz[i]
        );
    }

    // Densidad sinusoidal → solve correcto implica transpose correcto.
    let mut density2 = vec![0.0_f64; nm * nm * nm];
    let nm2 = nm * nm;
    for iz in 0..nm {
        for iy in 0..nm {
            for ix in 0..nm {
                let x = ix as f64 / nm as f64;
                density2[iz * nm2 + iy * nm + ix] = 1.0 + (2.0 * PI * x).cos();
            }
        }
    }
    let [fx2_slab, _, _] = solve_forces_slab(&density2, &layout, g, box_size, None, &rt);
    let [fx2_ref, _, _] = fft_poisson::solve_forces(&density2, g, nm, box_size);

    let max_err = fx2_slab
        .iter()
        .zip(fx2_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < 1e-10,
        "transpose roundtrip: max error vs serial = {max_err:.2e}"
    );
}

// ── Test 6: Solve slab P=1 idéntico a PM serial ───────────────────────────────

#[test]
fn slab_solve_matches_serial_pm() {
    let nm = 8usize;
    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;
    let box_size = 1.0_f64;
    let g = 1.0_f64;

    let pos = uniform_grid_positions(4, box_size); // 64 partículas
    let masses = vec![1.0_f64; pos.len()];

    let density = cic::assign(&pos, &masses, box_size, nm);

    let [fx_s, fy_s, fz_s] = fft_poisson::solve_forces(&density, g, nm, box_size);
    let [fx_d, fy_d, fz_d] = solve_forces_slab(&density, &layout, g, box_size, None, &rt);

    let tol = 1e-10;
    for i in 0..nm * nm * nm {
        assert!(
            (fx_s[i] - fx_d[i]).abs() < tol,
            "fx slab != serial en {i}: slab={:.6e} serial={:.6e}",
            fx_d[i],
            fx_s[i]
        );
        assert!((fy_s[i] - fy_d[i]).abs() < tol, "fy slab != serial en {i}");
        assert!((fz_s[i] - fz_d[i]).abs() < tol, "fz slab != serial en {i}");
    }
}

// ── Test 7: Modo Poisson sinusoidal — fuerza con signo correcto ───────────────

#[test]
fn slab_poisson_sanity_sinusoidal_mode() {
    // ρ(x) = ρ_mean + A·cos(2πx/L)
    // Potencial: Φ̂(k=1) = -4πG·ρ̂(k=1)/k² = -4πG·(A/2)/(2π/L)²
    // Fuerza Fx = -∂Φ/∂x → F_x(x=L/4) ∝ -sin(2π·L/4/L) = -1 < 0 (apunta hacia x=0)
    let nm = 16usize;
    let nm2 = nm * nm;
    let nm3 = nm2 * nm;
    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;
    let box_size = 1.0_f64;
    let g = 1.0_f64;

    // Construir densidad sinusoidal en x con indexado correcto iz*nm²+iy*nm+ix.
    let mut density = vec![0.0_f64; nm3];
    let rho_mean = 1.0_f64;
    let amp = 0.5_f64;
    for iz in 0..nm {
        for iy in 0..nm {
            for ix in 0..nm {
                let x = ix as f64 / nm as f64;
                density[iz * nm2 + iy * nm + ix] = rho_mean + amp * (2.0 * PI * x).cos();
            }
        }
    }

    let [fx, _, _] = solve_forces_slab(&density, &layout, g, box_size, None, &rt);

    // En x = L/4 (índice nm/4): F_x debe ser negativo (fuerza hacia x=0 donde hay más masa).
    let ix_quarter = nm / 4;
    let flat_quarter = 0 * nm2 + 0 * nm + ix_quarter; // iz=0, iy=0
    assert!(
        fx[flat_quarter] < 0.0,
        "F_x en x=L/4 debe ser negativo, got {:.6e}",
        fx[flat_quarter]
    );

    // En x = 3L/4 (índice 3nm/4): F_x debe ser positivo (fuerza hacia x=L).
    let ix_3quarter = 3 * nm / 4;
    let flat_3q = 0 * nm2 + 0 * nm + ix_3quarter;
    assert!(
        fx[flat_3q] > 0.0,
        "F_x en x=3L/4 debe ser positivo, got {:.6e}",
        fx[flat_3q]
    );
}

// ── Test 8: Run corto EdS sin NaN/Inf ─────────────────────────────────────────

#[test]
fn distributed_pm_no_explosion_slab() {
    use gadget_ng_core::CosmologySection;

    // Construcción de partículas en malla perturbada.
    let n_side = 4usize;
    let n = n_side * n_side * n_side; // 64
    let box_size = 1.0_f64;
    let mass = 1.0_f64 / n as f64;

    let _ = CosmologySection::default(); // verificar que el tipo existe

    let pos = uniform_grid_positions(n_side, box_size);
    let masses = vec![mass; n];

    let nm = 8usize;
    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;
    let g = 4.302e-3; // G en unidades prácticas (aproximación)

    // Simular 3 pasos del pipeline slab completo.
    let mut positions = pos.clone();
    let mut velocities = vec![Vec3::new(0.0, 0.0, 0.0); n];
    let dt = 0.01_f64;

    for _step in 0..3 {
        let density = slab_pm::deposit_slab_extended(&positions, &masses, &layout, box_size);
        // Para P=1, exchange es no-op.
        let mut density_ext = density;
        slab_pm::exchange_density_halos_z(&mut density_ext, &layout, &rt);

        let mut forces = slab_pm::forces_from_slab(&density_ext, &layout, g, box_size, None, &rt);
        slab_pm::exchange_force_halos_z(&mut forces, &layout, &rt);

        let accels = slab_pm::interpolate_slab_local(&positions, &forces, &layout, box_size);

        // Integración leapfrog simple.
        for i in 0..n {
            velocities[i] = Vec3::new(
                velocities[i].x + accels[i].x * dt,
                velocities[i].y + accels[i].y * dt,
                velocities[i].z + accels[i].z * dt,
            );
            positions[i] = Vec3::new(
                (positions[i].x + velocities[i].x * dt).rem_euclid(box_size),
                (positions[i].y + velocities[i].y * dt).rem_euclid(box_size),
                (positions[i].z + velocities[i].z * dt).rem_euclid(box_size),
            );

            // Verificar que no hay explosión numérica.
            assert!(
                positions[i].x.is_finite()
                    && positions[i].y.is_finite()
                    && positions[i].z.is_finite(),
                "posición NaN/Inf en partícula {i} paso {_step}"
            );
            assert!(
                velocities[i].x.is_finite()
                    && velocities[i].y.is_finite()
                    && velocities[i].z.is_finite(),
                "velocidad NaN/Inf en partícula {i} paso {_step}"
            );
        }
    }
}
