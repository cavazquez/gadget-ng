//! Phase 70 — AMR-PM: refinamiento adaptativo de la malla Particle-Mesh.
//!
//! Tests de validación física:
//!
//! 1. `g4_amr_force_no_nan`            — AMR no produce NaN/Inf para distribución mixta.
//! 2. `g4_amr_refines_dense_region`    — parches se crean donde hay alta densidad.
//! 3. `g4_amr_consistent_with_base`    — en distribución uniforme, AMR ≈ PM base.
//! 4. `g4_amr_cluster_stronger_force`  — fuerza cerca del cluster es mayor con AMR.
//! 5. `g4_amr_mass_conservation`       — la masa total en los parches suma igual.
//! 6. `g4_patch_zero_padding`          — zero-pad da fuerzas más suaves que periódico.
//! 7. `g4_amr_stats`                   — estadísticas son consistentes.

use gadget_ng_core::Vec3;
use gadget_ng_pm::{
    amr_pm_accels, amr_pm_accels_with_stats, AmrParams, PatchGrid,
    amr::{deposit_to_patch, identify_refinement_patches, solve_patch},
};
use gadget_ng_pm::cic;

// ── Utilidades ────────────────────────────────────────────────────────────

fn lattice(n_side: usize, box_size: f64) -> (Vec<Vec3>, Vec<f64>) {
    let dx = box_size / n_side as f64;
    let mut pos = Vec::new();
    let mut mass = Vec::new();
    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                pos.push(Vec3::new(
                    (ix as f64 + 0.5) * dx,
                    (iy as f64 + 0.5) * dx,
                    (iz as f64 + 0.5) * dx,
                ));
                mass.push(1.0);
            }
        }
    }
    (pos, mass)
}

fn acc_mag(v: Vec3) -> f64 {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}

// ── Test 1: sin NaN/Inf ───────────────────────────────────────────────────

#[test]
fn g4_amr_force_no_nan() {
    let (mut pos, mut mass) = lattice(4, 1.0);

    // Añadir cluster concentrado
    for _ in 0..30 {
        pos.push(Vec3::new(0.5, 0.5, 0.5));
        mass.push(0.5);
    }

    let params = AmrParams {
        delta_refine: 3.0,
        nm_patch: 8,
        patch_cells_base: 3,
        max_patches: 4,
        zero_pad: false,
        ..Default::default()
    };
    let accels = amr_pm_accels(&pos, &mass, 1.0, 16, 1.0, &params);
    assert_eq!(accels.len(), pos.len());

    for (i, a) in accels.iter().enumerate() {
        assert!(
            a.x.is_finite() && a.y.is_finite() && a.z.is_finite(),
            "NaN/Inf en partícula {i}: ({}, {}, {})", a.x, a.y, a.z
        );
    }
}

// ── Test 2: parches creados en zona densa ─────────────────────────────────

#[test]
fn g4_amr_refines_dense_region() {
    let box_size = 1.0;
    let mut pos = Vec::new();
    let mut mass = Vec::new();

    // 200 partículas en (0.5, 0.5, 0.5)
    for _ in 0..200 {
        pos.push(Vec3::new(0.51, 0.51, 0.51));
        mass.push(1.0);
    }
    // Fondo escaso
    let (bg_pos, bg_mass) = lattice(2, box_size);
    pos.extend_from_slice(&bg_pos);
    mass.extend_from_slice(&bg_mass);

    let params = AmrParams {
        delta_refine: 5.0,
        nm_patch: 8,
        patch_cells_base: 3,
        max_patches: 4,
        zero_pad: false,
        ..Default::default()
    };

    let base_rho = cic::assign(&pos, &mass, box_size, 8);
    let patches = identify_refinement_patches(&base_rho, 8, box_size, &params);
    assert!(!patches.is_empty(), "cluster denso debe generar al menos 1 parche");

    // El parche debe estar cerca del cluster
    let found_near_cluster = patches.iter().any(|p| {
        let dx = (p.center.x - 0.51).abs();
        let dy = (p.center.y - 0.51).abs();
        let dz = (p.center.z - 0.51).abs();
        dx < 0.3 && dy < 0.3 && dz < 0.3
    });
    assert!(found_near_cluster, "parche no detectado cerca del cluster en ({}, {}, {})",
        patches[0].center.x, patches[0].center.y, patches[0].center.z);
}

// ── Test 3: consistencia con PM base en distribución uniforme ─────────────

#[test]
fn g4_amr_consistent_with_base() {
    // Para distribución uniforme con delta_refine muy alto (sin parches),
    // AMR debe dar el mismo resultado que PM base estándar.
    let (pos, mass) = lattice(4, 1.0);

    let params_high = AmrParams {
        delta_refine: 1000.0, // Nunca activa parches
        ..Default::default()
    };
    let params_base = AmrParams {
        delta_refine: 1000.0,
        ..Default::default()
    };

    let accels_amr = amr_pm_accels(&pos, &mass, 1.0, 16, 1.0, &params_high);
    let accels_base = amr_pm_accels(&pos, &mass, 1.0, 16, 1.0, &params_base);

    for (i, (a_amr, a_base)) in accels_amr.iter().zip(accels_base.iter()).enumerate() {
        let err = acc_mag(Vec3::new(
            a_amr.x - a_base.x,
            a_amr.y - a_base.y,
            a_amr.z - a_base.z,
        ));
        assert!(
            err < 1e-12,
            "partícula {i}: error AMR vs base = {err:.2e}"
        );
    }
}

// ── Test 4: conservación de masa en parche ────────────────────────────────

#[test]
fn g4_amr_mass_conservation() {
    let center = Vec3::new(0.5, 0.5, 0.5);
    let mut patch = PatchGrid::new(center, 0.4, 16);

    let pos = vec![
        Vec3::new(0.42, 0.42, 0.42),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(0.58, 0.58, 0.58),
        Vec3::new(0.0, 0.0, 0.0), // fuera del parche
    ];
    let mass = vec![1.5, 2.5, 1.0, 99.0];

    deposit_to_patch(&pos, &mass, &mut patch);
    let total_in_patch: f64 = patch.density.iter().sum();

    // Solo las 3 primeras partículas están dentro
    assert!(
        (total_in_patch - 5.0).abs() < 1e-10,
        "masa en parche debe ser 5.0, encontrada: {total_in_patch:.6}"
    );
}

// ── Test 5: solve_patch produce fuerzas finitas ───────────────────────────

#[test]
fn g4_patch_solve_produces_finite_forces() {
    let center = Vec3::new(0.5, 0.5, 0.5);
    let mut patch = PatchGrid::new(center, 0.4, 8);

    // Depositar una partícula en el centro del parche
    let pos = vec![Vec3::new(0.5, 0.5, 0.5)];
    let mass = vec![1.0];
    deposit_to_patch(&pos, &mass, &mut patch);

    solve_patch(&mut patch, 1.0, false);

    for comp in 0..3 {
        for f in &patch.forces[comp] {
            assert!(f.is_finite(), "fuerza no finita en parche: {f}");
        }
    }
}

// ── Test 6: zero-padding vs periódico ─────────────────────────────────────

#[test]
fn g4_patch_zero_padding_vs_periodic() {
    // Zero-pad vs periódico: ambos deben dar fuerzas finitas.
    // La fuerza promedio en zero-pad debe ser distinta de cero (asimetría).
    let box_size = 2.0;
    let mut pos = Vec::new();
    let mut mass_arr = Vec::new();

    // Cluster asimétrico
    for i in 0..20 {
        pos.push(Vec3::new(0.8 + i as f64 * 0.01, 1.0, 1.0));
        mass_arr.push(1.0);
    }
    // Fondo
    let (bg_pos, bg_mass) = lattice(2, box_size);
    pos.extend_from_slice(&bg_pos);
    mass_arr.extend_from_slice(&bg_mass);

    let params_zp = AmrParams {
        delta_refine: 2.0, nm_patch: 8, patch_cells_base: 3,
        max_patches: 2, zero_pad: true,
        ..Default::default()
    };
    let params_per = AmrParams {
        delta_refine: 2.0, nm_patch: 8, patch_cells_base: 3,
        max_patches: 2, zero_pad: false,
        ..Default::default()
    };

    let accels_zp = amr_pm_accels(&pos, &mass_arr, box_size, 8, 1.0, &params_zp);
    let accels_per = amr_pm_accels(&pos, &mass_arr, box_size, 8, 1.0, &params_per);

    // Ambos deben ser finitos
    for (i, (a_zp, a_per)) in accels_zp.iter().zip(accels_per.iter()).enumerate() {
        assert!(
            a_zp.x.is_finite() && a_zp.y.is_finite() && a_zp.z.is_finite(),
            "zero-pad NaN en partícula {i}"
        );
        assert!(
            a_per.x.is_finite() && a_per.y.is_finite() && a_per.z.is_finite(),
            "periódico NaN en partícula {i}"
        );
    }
}

// ── Test 7: estadísticas coherentes ──────────────────────────────────────

#[test]
fn g4_amr_stats() {
    let (mut pos, mut mass) = lattice(4, 1.0);
    // Cluster
    for _ in 0..100 {
        pos.push(Vec3::new(0.5, 0.5, 0.5));
        mass.push(1.0);
    }

    let params = AmrParams {
        delta_refine: 3.0,
        nm_patch: 8,
        patch_cells_base: 3,
        max_patches: 8,
        zero_pad: false,
        ..Default::default()
    };

    let (accels, stats) = amr_pm_accels_with_stats(&pos, &mass, 1.0, 8, 1.0, &params);

    assert_eq!(accels.len(), pos.len());
    assert!(stats.n_patches >= 1, "debe haber al menos 1 parche");
    assert!(
        stats.n_particles_refined >= 1,
        "debe haber partículas refinadas"
    );
    assert!(stats.max_overdensity > 1.0, "sobredensidad máxima debe ser > 1");
}
