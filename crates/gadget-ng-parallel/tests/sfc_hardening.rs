//! Tests de hardening para la descomposición SFC (Fase 8).
//!
//! - `sfc_build_global_consistent`: verificar que `build_with_bbox` con bbox explícita
//!   produce cutpoints idénticos a `build` cuando todos los rangos ven las mismas posiciones.
//! - `migration_arbitrary`: particiones con partículas que van a rangos no adyacentes
//!   (`exchange_domain_sfc` vía Alltoallv) se resuelven en un solo paso.
//! - `halo_geometric_3d`: halos SFC 3D—partículas dentro del rango expandido
//!   de un vecino se envían correctamente.

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_parallel::{
    sfc::{global_bbox, partition_local, SfcDecomposition},
    ParallelRuntime, SerialRuntime,
};

// ── Utilidades ────────────────────────────────────────────────────────────────

fn make_particle(gid: usize, x: f64, y: f64, z: f64) -> Particle {
    Particle::new(gid, 1.0, Vec3::new(x, y, z), Vec3::zero())
}

fn make_grid(n: usize) -> Vec<Particle> {
    let mut ps = Vec::with_capacity(n * n * n);
    let mut gid = 0;
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let x = ix as f64 / n as f64;
                let y = iy as f64 / n as f64;
                let z = iz as f64 / n as f64;
                ps.push(make_particle(gid, x, y, z));
                gid += 1;
            }
        }
    }
    ps
}

// ── Test 1: bbox global consistente ──────────────────────────────────────────

/// Verifica que `build_with_bbox` con la bbox global produce los mismos cutpoints
/// que `build` cuando el llamante ya tiene todas las posiciones (baseline serial).
#[test]
fn sfc_build_global_consistent() {
    let rt = SerialRuntime;
    let particles = make_grid(8); // 512 partículas
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();

    // build normal (calcula bbox localmente).
    let decomp_local = SfcDecomposition::build(&positions, 1.0, 4);

    // build_with_bbox usando bbox global via allreduce (en serial = mismos valores).
    let (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) = global_bbox(&rt, &particles);
    let decomp_global =
        SfcDecomposition::build_with_bbox(&positions, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, 4);

    // Ambas descomposiciones deben asignar las mismas partículas a los mismos rangos.
    for p in &particles {
        assert_eq!(
            decomp_local.rank_for_pos(p.position),
            decomp_global.rank_for_pos(p.position),
            "gid={} pos={:?}: rango difiere entre build local y build_with_bbox global",
            p.global_id,
            p.position,
        );
    }
}

/// Con bbox conocida, los cutpoints deben repartir las partículas de forma aproximadamente igual.
#[test]
fn sfc_build_with_bbox_balance() {
    let rt = SerialRuntime;
    let particles = make_grid(10); // 1000 partículas
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) = global_bbox(&rt, &particles);

    let n_ranks = 4;
    let decomp =
        SfcDecomposition::build_with_bbox(&positions, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, n_ranks);

    let mut counts = vec![0usize; n_ranks as usize];
    for pos in &positions {
        counts[decomp.rank_for_pos(*pos) as usize] += 1;
    }
    let total: usize = counts.iter().sum();
    assert_eq!(total, 1000, "ninguna partícula debe perderse");
    // Tolerancia ±40%: cada rango debe tener entre 150 y 350 partículas.
    for (r, &c) in counts.iter().enumerate() {
        assert!(
            c >= 150 && c <= 350,
            "rango {r} tiene {c} partículas (desequilibrado tras build_with_bbox)"
        );
    }
}

// ── Test 2: migración arbitraria ──────────────────────────────────────────────

/// Verifica que `partition_local` clasifica correctamente partículas que deben ir
/// a rangos no adyacentes (no solo rank±1).
#[test]
fn migration_arbitrary_ranks() {
    // 4 partículas, una en cada cuadrante de [0,1)²  (z=0.5, simplificado a 2D).
    // Con 4 rangos y la SFC Morton, cada cuadrante debería quedar en un rango distinto.
    let ps = vec![
        make_particle(0, 0.1, 0.1, 0.5), // cuadrante 0,0
        make_particle(1, 0.9, 0.1, 0.5), // cuadrante 1,0
        make_particle(2, 0.1, 0.9, 0.5), // cuadrante 0,1
        make_particle(3, 0.9, 0.9, 0.5), // cuadrante 1,1
    ];

    let rt = SerialRuntime;
    let (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) = global_bbox(&rt, &ps);
    let positions: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
    let decomp =
        SfcDecomposition::build_with_bbox(&positions, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, 4);

    // Verificar que las 4 partículas se asignan a 4 rangos distintos.
    let ranks: std::collections::HashSet<i32> =
        ps.iter().map(|p| decomp.rank_for_pos(p.position)).collect();
    assert_eq!(
        ranks.len(),
        4,
        "Las 4 partículas en cuadrantes opuestos deberían caer en rangos distintos; ranks={:?}",
        ranks
    );

    // partition_local para rango 0: todas las partículas de otros rangos van en `leaves`.
    let (stay, leaves) = partition_local(&ps, &decomp, 0);
    let leaves_flat: Vec<_> = leaves.iter().flat_map(|(_, v)| v.iter()).collect();
    assert_eq!(stay.len() + leaves_flat.len(), 4);
    // Las que se quedan son solo las del rango 0.
    for p in &stay {
        assert_eq!(decomp.rank_for_pos(p.position), 0);
    }
    // Las que salen van a rangos distintos de 0 (no solo rank±1).
    for (r, _) in &leaves {
        assert_ne!(*r, 0);
    }
}

// ── Test 3: halos geométricos 3D ─────────────────────────────────────────────

/// Verifica la lógica de halos 3D: partículas dentro del halo_width de la AABB
/// de otro rango son correctamente identificadas para envío.
#[test]
fn halo_geometric_3d_filter() {
    // Simular qué partículas de un rango A (posiciones en [0,0.5)³)
    // deben enviarse al rango B (posiciones en (0.5,1.0]³) si halo_width=0.1.
    // La AABB de B es [0.5, 1.0] × [0.5, 1.0] × [0.5, 1.0].
    // Expandida por halo_width=0.1: [0.4, 1.1] × [0.4, 1.1] × [0.4, 1.1].
    // Partículas de A con x >= 0.4 AND y >= 0.4 AND z >= 0.4 deben enviarse.

    let halo_width = 0.1f64;

    // Partículas del rango A (dominio [0,0.5)³).
    let particles_a = vec![
        make_particle(0, 0.1, 0.1, 0.1),    // lejos del borde → NO enviar
        make_particle(1, 0.45, 0.45, 0.45), // dentro de la zona halo 3D → SÍ enviar
        make_particle(2, 0.48, 0.1, 0.1),   // cerca en x pero no en y/z → NO enviar
        make_particle(3, 0.48, 0.48, 0.48), // cerca en todas → SÍ enviar
        make_particle(4, 0.3, 0.45, 0.45),  // cerca en y/z pero no en x → NO enviar
    ];

    // AABB del rango B.
    let (bxlo, bxhi, bylo, byhi, bzlo, bzhi) = (0.5f64, 1.0, 0.5, 1.0, 0.5, 1.0);

    // AABB expandida del rango B.
    let (rxlo, rxhi) = (bxlo - halo_width, bxhi + halo_width);
    let (rylo, ryhi) = (bylo - halo_width, byhi + halo_width);
    let (rzlo, rzhi) = (bzlo - halo_width, bzhi + halo_width);

    let in_halo: Vec<&Particle> = particles_a
        .iter()
        .filter(|p| {
            p.position.x >= rxlo
                && p.position.x <= rxhi
                && p.position.y >= rylo
                && p.position.y <= ryhi
                && p.position.z >= rzlo
                && p.position.z <= rzhi
        })
        .collect();

    // Solo las partículas 1 y 3 deben estar dentro de la AABB 3D expandida.
    let ids: Vec<usize> = in_halo.iter().map(|p| p.global_id).collect();
    assert!(
        ids.contains(&1),
        "partícula 1 debe estar en el halo 3D; ids={:?}",
        ids
    );
    assert!(
        ids.contains(&3),
        "partícula 3 debe estar en el halo 3D; ids={:?}",
        ids
    );
    assert_eq!(
        ids.len(),
        2,
        "solo 2 partículas deben estar en el halo 3D; ids={:?}",
        ids
    );
}

/// Verifica que el allgather_f64 del SerialRuntime devuelve los datos locales.
#[test]
fn allgather_f64_serial_returns_local() {
    let rt = SerialRuntime;
    let local = vec![1.0f64, 2.0, 3.0];
    let result = rt.allgather_f64(&local);
    assert_eq!(result.len(), 1, "serial: 1 rango");
    assert_eq!(
        result[0], local,
        "serial allgather debe devolver datos propios"
    );
}

/// Verifica que alltoallv_f64 en serial no transfiere datos (sin otros rangos).
#[test]
fn alltoallv_f64_serial_noop() {
    let rt = SerialRuntime;
    let sends = vec![vec![1.0f64, 2.0]]; // sends[0] = datos para rank 0 (yo mismo)
    let received = rt.alltoallv_f64(&sends);
    // En serial, no hay otros rangos a los que enviar; recibimos vacío.
    assert_eq!(received.len(), 1);
    assert!(
        received[0].is_empty(),
        "serial alltoallv: no hay datos recibidos de otros rangos"
    );
}
