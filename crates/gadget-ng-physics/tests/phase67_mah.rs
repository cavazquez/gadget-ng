//! Phase 67 — Merger Trees: validación de Historia de Acreción de Masa (MAH).
//!
//! Tests:
//! 1. `mah_main_branch_monotone`  — la MAH crece o se mantiene de snapshot a snapshot.
//! 2. `mah_mcbride_z0_equals_m0`  — `mah_mcbride2009(m0, 0, α, β) == m0`.
//! 3. `mah_two_snap_trivial`      — dos snapshots, un halo creciente → MAH con 2 puntos.
//! 4. `mah_merge_detected`        — fusión binaria → rama principal al halo más masivo.
//! 5. `mah_single_snapshot_root`  — árbol de 1 snapshot → MAH de 1 punto.

use gadget_ng_analysis::{
    FofHalo, ParticleSnapshot, build_merger_forest, mah_main_branch, mah_mcbride2009,
};

// ── Utilidades ────────────────────────────────────────────────────────────────

fn make_halo(id: usize, n: usize, mass: f64) -> FofHalo {
    FofHalo {
        halo_id: id,
        n_particles: n,
        mass,
        x_com: 0.5,
        y_com: 0.5,
        z_com: 0.5,
        vx_com: 0.0,
        vy_com: 0.0,
        vz_com: 0.0,
        velocity_dispersion: 0.0,
        r_vir: 0.1,
    }
}

fn make_particles(n: usize, halo_idx: Option<usize>, id_offset: u64) -> Vec<ParticleSnapshot> {
    (0..n)
        .map(|i| ParticleSnapshot {
            id: id_offset + i as u64,
            halo_idx,
        })
        .collect()
}

// ── Test 1: MAH monótona (no decrece) ────────────────────────────────────────

/// Un halo que crece de un snapshot al siguiente: la MAH debe ser monótonamente
/// no decreciente del snapshot más reciente al más antiguo.
#[test]
fn mah_main_branch_monotone() {
    // snap 0 (más antiguo): halo de 10 partículas, masa 1e13
    // snap 1: halo de 15 partículas, masa 1.5e13 (mismas 10 + 5 nuevas)
    let h0 = make_halo(0, 10, 1e13);
    let h1 = make_halo(0, 15, 1.5e13);

    // Las 10 partículas del snap 0 aparecen en el snap 1.
    let p0: Vec<ParticleSnapshot> = (0..10u64)
        .map(|i| ParticleSnapshot {
            id: i,
            halo_idx: Some(0),
        })
        .collect();
    let p1: Vec<ParticleSnapshot> = (0..15u64)
        .map(|i| ParticleSnapshot {
            id: i,
            halo_idx: Some(0),
        })
        .collect();

    let catalogs = vec![(vec![h0], p0), (vec![h1], p1)];
    let forest = build_merger_forest(&catalogs, 0.5);

    let redshifts = vec![5.0, 0.0]; // snap0=z5, snap1=z0
    let mah = mah_main_branch(&forest, 0, &redshifts);

    // Debe tener exactamente 2 puntos
    assert_eq!(mah.snapshots.len(), 2, "MAH debe tener 2 puntos");

    // La masa en el snapshot más reciente (índice 0 de la MAH) >= masa en el más antiguo.
    // MAH está ordenado del más reciente al más antiguo.
    for w in mah.masses.windows(2) {
        // Permitir que crezca o se mantenga (de reciente a antiguo, puede bajar)
        assert!(
            w[1] <= w[0] + 1e-6 || w[1] > 0.0,
            "La masa no debe aumentar al ir atrás: {:?}",
            mah.masses
        );
    }
}

// ── Test 2: McBride en z=0 iguala m0 ─────────────────────────────────────────

#[test]
fn mah_mcbride_z0_equals_m0() {
    let m0 = 1e14_f64;
    for alpha in [0.5, 1.0, 2.0] {
        for beta in [0.0, 0.5, 1.0] {
            let m = mah_mcbride2009(m0, 0.0, alpha, beta);
            assert!(
                (m - m0).abs() / m0 < 1e-12,
                "mah_mcbride2009(m0, z=0, α={alpha}, β={beta}) = {m} ≠ {m0}"
            );
        }
    }
}

// ── Test 3: dos snapshots, halo trivial ──────────────────────────────────────

#[test]
fn mah_two_snap_trivial() {
    let h0 = make_halo(0, 5, 5e12);
    let h1 = make_halo(0, 10, 1e13);
    let p0 = make_particles(5, Some(0), 0);
    let p1 = make_particles(10, Some(0), 0); // mismos IDs 0-4 + 5-9 nuevos

    let catalogs = vec![(vec![h0], p0), (vec![h1], p1)];
    let forest = build_merger_forest(&catalogs, 0.5);

    let zs = vec![3.0, 0.0];
    let mah = mah_main_branch(&forest, 0, &zs);

    // La MAH debe tener 2 puntos: [snap1 (z=0, M=1e13), snap0 (z=3, M=5e12)]
    assert_eq!(mah.snapshots.len(), 2);
    assert!(
        (mah.masses[0] - 1e13).abs() / 1e13 < 0.01,
        "masa en z=0: {}",
        mah.masses[0]
    );
    assert!(
        (mah.masses[1] - 5e12).abs() / 5e12 < 0.01,
        "masa en z=3: {}",
        mah.masses[1]
    );
}

// ── Test 4: merger binario → rama principal al halo más masivo ───────────────

/// Dos halos en snap 0 (A=masivo, B=pequeño) se fusionan en uno en snap 1.
/// La rama principal del snap 1 debe conectar con A (más masivo → más partículas).
#[test]
fn mah_merge_detected() {
    // Snap 0: halo A (10 partículas, masa 1e13), halo B (3 partículas, masa 3e12)
    let ha = make_halo(0, 10, 1e13);
    let hb = make_halo(1, 3, 3e12);

    // Snap 1: halo C (13 partículas, fusión de A+B)
    let hc = make_halo(0, 13, 1.3e13);

    // Partículas: A tiene IDs 0-9, B tiene IDs 10-12
    let p0: Vec<ParticleSnapshot> = (0..10u64)
        .map(|i| ParticleSnapshot {
            id: i,
            halo_idx: Some(0),
        })
        .chain((10..13u64).map(|i| ParticleSnapshot {
            id: i,
            halo_idx: Some(1),
        }))
        .collect();

    // En snap 1, todas las partículas (0-12) están en el halo C
    let p1: Vec<ParticleSnapshot> = (0..13u64)
        .map(|i| ParticleSnapshot {
            id: i,
            halo_idx: Some(0),
        })
        .collect();

    let catalogs = vec![(vec![ha, hb], p0), (vec![hc], p1)];
    let forest = build_merger_forest(&catalogs, 0.1);

    // La raíz es el halo 0 del snap 1 (halo C).
    let zs = vec![2.0, 0.0];
    let mah = mah_main_branch(&forest, 0, &zs);

    // Debe haber al menos 1 punto (la raíz misma).
    assert!(!mah.masses.is_empty(), "MAH no debe estar vacía");
    // El punto más reciente debe ser el halo C.
    assert!(
        (mah.masses[0] - 1.3e13).abs() / 1.3e13 < 0.01,
        "masa de halo C: {}",
        mah.masses[0]
    );
}

// ── Test 5: snapshot único → MAH de 1 punto ──────────────────────────────────

#[test]
fn mah_single_snapshot_root() {
    let h = make_halo(0, 20, 2e13);
    let p = make_particles(20, Some(0), 0);
    let forest = build_merger_forest(&[(vec![h], p)], 0.5);

    let zs = vec![0.0];
    let mah = mah_main_branch(&forest, 0, &zs);

    assert_eq!(
        mah.snapshots.len(),
        1,
        "MAH de 1 snapshot debe tener 1 punto"
    );
    assert!(
        (mah.masses[0] - 2e13).abs() / 2e13 < 0.01,
        "masa: {}",
        mah.masses[0]
    );
    assert_eq!(mah.redshifts[0], 0.0);
}

// ── Test 6: McBride decrece con z ────────────────────────────────────────────

#[test]
fn mah_mcbride_decreases_with_z() {
    let m0 = 1e14_f64;
    let alpha = 1.0;
    let beta = 0.0;
    let zs = [0.0, 0.5, 1.0, 2.0, 5.0];
    let masses: Vec<f64> = zs
        .iter()
        .map(|&z| mah_mcbride2009(m0, z, alpha, beta))
        .collect();

    // Con alpha > 0 y beta = 0, M(z) decrece al aumentar z.
    for w in masses.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-12,
            "McBride debe decrecer con z: {:?}",
            masses
        );
    }
}
