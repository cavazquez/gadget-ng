//! Phase 107 — Merger trees con FoF real
//!
//! Verifica que:
//! 1. `find_halos_with_membership` retorna membresía correcta por partícula.
//! 2. `particle_snapshots_from_catalog` asigna halo_idx por proximidad al COM.
//! 3. `build_merger_forest` detecta correctamente mergers entre halos consecutivos
//!    cuando se usa membresía real (halo_idx != None).

use gadget_ng_analysis::{
    build_merger_forest, find_halos_with_membership, particle_snapshots_from_catalog,
    FofHalo, ParticleSnapshot,
};
use gadget_ng_core::Vec3;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_cluster(center: Vec3, n: usize, spread: f64) -> Vec<Vec3> {
    let mut positions = Vec::new();
    for i in 0..n {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / n as f64;
        positions.push(Vec3::new(
            center.x + spread * angle.cos(),
            center.y + spread * angle.sin(),
            center.z,
        ));
    }
    positions
}

fn make_fof_halo(halo_id: usize, x: f64, y: f64, z: f64, n: usize, r_vir: f64) -> FofHalo {
    FofHalo {
        halo_id,
        n_particles: n,
        mass: n as f64,
        x_com: x, y_com: y, z_com: z,
        vx_com: 0.0, vy_com: 0.0, vz_com: 0.0,
        velocity_dispersion: 0.0,
        r_vir,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// find_halos_with_membership retorna membresía correcta para 2 halos separados.
#[test]
fn membership_two_separated_clusters() {
    let box_size = 100.0;
    // Cluster A: 20 partículas alrededor de (20, 20, 20)
    let cluster_a = make_cluster(Vec3::new(20.0, 20.0, 20.0), 20, 0.3);
    // Cluster B: 10 partículas alrededor de (80, 80, 80)
    let cluster_b = make_cluster(Vec3::new(80.0, 80.0, 80.0), 10, 0.3);

    let mut positions: Vec<Vec3> = cluster_a.iter().chain(cluster_b.iter()).cloned().collect();
    // Agregar 5 partículas de campo lejos de todo
    for i in 0..5 {
        positions.push(Vec3::new(50.0 + i as f64 * 5.0, 50.0, 10.0));
    }
    let n = positions.len();
    let velocities = vec![Vec3::zero(); n];
    let masses = vec![1.0f64; n];

    let (halos, membership) = find_halos_with_membership(
        &positions, &velocities, &masses, box_size, 0.2, 5, 0.0,
    );

    assert!(halos.len() >= 2, "debe haber al menos 2 halos: {}", halos.len());
    assert_eq!(membership.len(), n, "membresía debe tener mismo tamaño que partículas");

    // Las primeras 20 partículas deben pertenecer a algún halo
    let halo_a_count = membership[..20].iter().filter(|m| m.is_some()).count();
    assert!(halo_a_count >= 15, "al menos 15/20 partículas del cluster A en un halo: {halo_a_count}");

    // Las siguientes 10 partículas también deben pertenecer a algún halo
    let halo_b_count = membership[20..30].iter().filter(|m| m.is_some()).count();
    assert!(halo_b_count >= 8, "al menos 8/10 partículas del cluster B en un halo: {halo_b_count}");

    // Los halos deben ser grupos distintos
    if halo_a_count > 0 && halo_b_count > 0 {
        let idx_a = membership[0].unwrap();
        let idx_b = membership[20].unwrap();
        assert_ne!(idx_a, idx_b, "clusters A y B deben estar en halos distintos");
    }
}

/// particle_snapshots_from_catalog asigna halo_idx por proximidad.
#[test]
fn catalog_proximity_assigns_halo_idx() {
    let box_size = 100.0;
    // Halo en (10, 10, 10) con r_vir = 5.0
    let halos = vec![make_fof_halo(0, 10.0, 10.0, 10.0, 100, 5.0)];

    let positions = vec![
        Vec3::new(10.5, 10.5, 10.5), // dentro del halo (dist ≈ 0.87 < 5.0)
        Vec3::new(12.0, 10.0, 10.0), // dentro del halo (dist = 2.0 < 5.0)
        Vec3::new(90.0, 90.0, 90.0), // campo (lejos)
    ];
    let ids = vec![0u64, 1, 2];

    let snaps = particle_snapshots_from_catalog(&positions, &ids, &halos, box_size);
    assert_eq!(snaps.len(), 3);
    assert_eq!(snaps[0].halo_idx, Some(0), "partícula 0 debe estar en halo 0");
    assert_eq!(snaps[1].halo_idx, Some(0), "partícula 1 debe estar en halo 0");
    assert_eq!(snaps[2].halo_idx, None, "partícula 2 debe ser campo");
}

/// build_merger_forest detecta merger cuando 2 halos se fusionan en el siguiente snapshot.
#[test]
fn merger_forest_detects_fusion() {
    // Snapshot 0: 2 halos separados, 8 partículas cada uno
    let h0_s0 = make_fof_halo(0, 10.0, 10.0, 10.0, 8, 3.0);
    let h1_s0 = make_fof_halo(1, 80.0, 80.0, 80.0, 8, 3.0);

    // IDs 0-7 en halo 0, IDs 8-15 en halo 1
    let parts_s0: Vec<ParticleSnapshot> = (0u64..8)
        .map(|id| ParticleSnapshot { id, halo_idx: Some(0) })
        .chain((8u64..16).map(|id| ParticleSnapshot { id, halo_idx: Some(1) }))
        .collect();

    // Snapshot 1: los dos halos se fusionaron en uno solo (recibe IDs 0-15)
    let h0_s1 = make_fof_halo(0, 45.0, 45.0, 45.0, 16, 5.0);

    // Todas las partículas del snapshot anterior ahora están en el mismo halo
    let parts_s1: Vec<ParticleSnapshot> = (0u64..16)
        .map(|id| ParticleSnapshot { id, halo_idx: Some(0) })
        .collect();

    let catalogs = vec![
        (vec![h0_s0, h1_s0], parts_s0),
        (vec![h0_s1], parts_s1),
    ];

    let forest = build_merger_forest(&catalogs, 0.1);

    // El snapshot 0 debe tener 2 nodos
    let snap0_nodes: Vec<_> = forest.nodes.iter().filter(|n| n.snapshot == 0).collect();
    assert_eq!(snap0_nodes.len(), 2, "snapshot 0 debe tener 2 halos");

    // Al menos un nodo del snapshot 0 debe tener prog_main_id apuntando al halo 0 del snapshot 1
    let has_connection = snap0_nodes.iter().any(|n| n.prog_main_id.is_some());
    assert!(has_connection, "al menos un halo del snapshot 0 debe tener progenitor en snapshot 1");
}

/// Snapshot sin halos produce forest vacío.
#[test]
fn empty_catalogs_produce_empty_forest() {
    let catalogs: Vec<(Vec<FofHalo>, Vec<ParticleSnapshot>)> = Vec::new();
    let forest = build_merger_forest(&catalogs, 0.1);
    assert!(forest.nodes.is_empty());
    assert!(forest.roots.is_empty());
}

/// Un solo snapshot produce nodos sin progenitores.
#[test]
fn single_snapshot_no_progenitors() {
    let h = make_fof_halo(0, 50.0, 50.0, 50.0, 10, 5.0);
    let parts: Vec<ParticleSnapshot> = (0u64..10)
        .map(|id| ParticleSnapshot { id, halo_idx: Some(0) })
        .collect();
    let catalogs = vec![(vec![h], parts)];
    let forest = build_merger_forest(&catalogs, 0.1);
    assert_eq!(forest.nodes.len(), 1);
    assert_eq!(forest.nodes[0].prog_main_id, None);
    assert!(forest.nodes[0].merger_ids.is_empty());
}

/// find_halos_with_membership funciona con 0 partículas.
#[test]
fn membership_empty_particles() {
    let (halos, membership) = find_halos_with_membership(&[], &[], &[], 10.0, 0.2, 5, 0.0);
    assert!(halos.is_empty());
    assert!(membership.is_empty());
}
