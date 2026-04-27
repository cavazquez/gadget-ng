//! Phase 62 — Merger trees de halos FoF (single-pass).
//!
//! Verifica la construcción del merger tree con:
//! - Halo que crece sin fusiones (progenitor único).
//! - Fusión binaria de dos halos en un paso.
//! - Roundtrip JSON de MergerForest.

use gadget_ng_analysis::{FofHalo, MergerForest, ParticleSnapshot, build_merger_forest};

fn skip() -> bool {
    std::env::var("PHASE62_SKIP")
        .map(|v| v == "1")
        .unwrap_or(false)
}

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

/// Construye partículas de snapshot con IDs consecutivos, todas en el halo h_idx.
fn parts_in_halo(n: usize, halo_idx: usize, id_offset: u64) -> Vec<ParticleSnapshot> {
    (0..n)
        .map(|i| ParticleSnapshot {
            id: id_offset + i as u64,
            halo_idx: Some(halo_idx),
        })
        .collect()
}

/// Construye partículas sin halo (campo).
fn parts_field(n: usize, id_offset: u64) -> Vec<ParticleSnapshot> {
    (0..n)
        .map(|i| ParticleSnapshot {
            id: id_offset + i as u64,
            halo_idx: None,
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

/// Halo con progenitor único: las mismas partículas aparecen en dos snapshots
/// consecutivos, masas crecientes. El nodo del snapshot 0 debe apuntar al 1.
#[test]
fn phase62_trivial_no_mergers() {
    if skip() {
        return;
    }

    // Snapshot 0: halo_0 con partículas 0..9
    let halos_0 = vec![make_halo(0, 10, 1e13)];
    let parts_0 = parts_in_halo(10, 0, 0);

    // Snapshot 1: halo_0 con las mismas partículas (creció un poco, ids iguales)
    let halos_1 = vec![make_halo(0, 10, 1.5e13)];
    let parts_1 = parts_in_halo(10, 0, 0);

    let catalogs = vec![(halos_0, parts_0), (halos_1, parts_1)];

    let forest = build_merger_forest(&catalogs, 0.3);

    assert_eq!(
        forest.nodes.len(),
        2,
        "deben existir 2 nodos (1 por snapshot)"
    );
    assert_eq!(forest.roots, vec![0], "raíz debe ser halo_0 del snapshot 1");

    // El nodo del snapshot 0 debe tener progenitor = halo_0 del snapshot 1.
    let node_s0 = forest.nodes.iter().find(|n| n.snapshot == 0).unwrap();
    assert_eq!(
        node_s0.prog_main_id,
        Some(0),
        "progenitor principal del snap_0 debe ser halo 0 del snap_1"
    );
    assert!(
        node_s0.merger_ids.is_empty(),
        "no debe haber mergers secundarios"
    );
}

/// Fusión binaria: dos halos en snapshot 0 se fusionan en uno en snapshot 1.
#[test]
fn phase62_binary_merger() {
    if skip() {
        return;
    }

    // Snapshot 0: dos halos
    let halos_0 = vec![
        make_halo(0, 10, 1e13), // halo_0: partículas 0..9
        make_halo(1, 8, 8e12),  // halo_1: partículas 10..17
    ];
    let mut parts_0 = parts_in_halo(10, 0, 0);
    parts_0.extend(parts_in_halo(8, 1, 10));

    // Snapshot 1: un único halo que contiene TODAS las partículas.
    let halos_1 = vec![make_halo(0, 18, 1.8e13)];
    let mut parts_1 = parts_in_halo(10, 0, 0); // partículas del halo_0 original
    parts_1.extend(parts_in_halo(8, 0, 10)); // partículas del halo_1 original

    let catalogs = vec![(halos_0, parts_0), (halos_1, parts_1)];

    let forest = build_merger_forest(&catalogs, 0.1);

    assert_eq!(forest.nodes.len(), 3, "3 nodos: 2 en snap_0 + 1 en snap_1");
    assert_eq!(forest.roots, vec![0], "raíz = halo_0 del snap_1");

    // Ambos halos del snapshot 0 deben apuntar al halo_0 del snapshot 1.
    for node in forest.nodes.iter().filter(|n| n.snapshot == 0) {
        assert_eq!(
            node.prog_main_id,
            Some(0),
            "halo {} del snap_0 debe converger al halo_0 del snap_1",
            node.halo_id
        );
    }
}

/// Roundtrip JSON: serializar y deserializar MergerForest preserva todos los campos.
#[test]
fn phase62_roundtrip_json() {
    if skip() {
        return;
    }

    let halos_0 = vec![make_halo(0, 5, 5e12)];
    let parts_0 = parts_in_halo(5, 0, 0);
    let halos_1 = vec![make_halo(0, 5, 6e12)];
    let parts_1 = parts_in_halo(5, 0, 0);

    let catalogs = vec![(halos_0, parts_0), (halos_1, parts_1)];
    let forest = build_merger_forest(&catalogs, 0.3);

    let json = serde_json::to_string_pretty(&forest).expect("serialización no debe fallar");
    assert!(
        json.contains("\"snapshot\""),
        "JSON debe contener campo 'snapshot'"
    );
    assert!(
        json.contains("\"mass_msun_h\""),
        "JSON debe contener campo 'mass_msun_h'"
    );

    let forest2: MergerForest =
        serde_json::from_str(&json).expect("deserialización no debe fallar");
    assert_eq!(forest.nodes.len(), forest2.nodes.len());
    assert_eq!(forest.roots, forest2.roots);

    let n0 = &forest.nodes[0];
    let n0_2 = &forest2.nodes[0];
    assert_eq!(n0.snapshot, n0_2.snapshot);
    assert_eq!(n0.halo_id, n0_2.halo_id);
    assert!((n0.mass_msun_h - n0_2.mass_msun_h).abs() < 1e-3);
}

/// Con una sola entrada (un snapshot) la forest tiene raíces pero sin progenitores.
#[test]
fn phase62_single_snapshot_no_progenitors() {
    if skip() {
        return;
    }

    let halos = vec![make_halo(0, 10, 1e14), make_halo(1, 5, 5e13)];
    let mut parts = parts_in_halo(10, 0, 0);
    parts.extend(parts_in_halo(5, 1, 10));

    let forest = build_merger_forest(&[(halos, parts)], 0.3);

    assert_eq!(forest.nodes.len(), 2);
    for node in &forest.nodes {
        assert!(
            node.prog_main_id.is_none(),
            "sin snapshots previos, no hay progenitores"
        );
    }
    // Las raíces son los 2 halos del único snapshot.
    assert_eq!(forest.roots.len(), 2);
}
