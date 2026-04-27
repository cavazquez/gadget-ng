//! Validación del protocolo LET (Locally Essential Tree).
//!
//! - `let_nodes_cover_all_mass`: verifica que la suma de masas exportadas en LET
//!   corresponde a la masa total del árbol para receptores remotos.
//! - `let_force_consistent`: verifica que la fuerza calculada con árbol local +
//!   nodos LET remotos coincide (dentro de tolerancia) con la fuerza del árbol
//!   global para un sistema de partículas sencillo.
//! - `export_let_prunes_subtrees`: verifica que nodos muy lejanos se exportan
//!   completos sin abrir sus subárboles (poda).

use gadget_ng_core::{Vec3, pairwise_accel_plummer};
use gadget_ng_tree::{Octree, RMN_FLOATS, accel_from_let, pack_let_nodes, unpack_let_nodes};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn uniform_positions(n: usize, seed: u64) -> Vec<Vec3> {
    let mut xs = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = ((state >> 33) as f64) / (u32::MAX as f64);
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let y = ((state >> 33) as f64) / (u32::MAX as f64);
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let z = ((state >> 33) as f64) / (u32::MAX as f64);
        xs.push(Vec3::new(x, y, z));
    }
    xs
}

// ── Test 1: cobertura de masa ─────────────────────────────────────────────────

/// Verifica que los nodos LET exportados para un receptor remoto cubren toda la masa.
#[test]
fn let_nodes_cover_all_mass() {
    let pos: Vec<Vec3> = uniform_positions(100, 42);
    let masses = vec![1.0f64; 100];
    let tree = Octree::build(&pos, &masses);

    let total_mass: f64 = masses.iter().sum();

    // AABB remoto muy alejado (no intersecta el dominio local).
    let target_aabb = [10.0f64, 11.0, 10.0, 11.0, 10.0, 11.0];
    let theta = 0.5;
    let let_nodes = tree.export_let(target_aabb, theta);

    let let_mass: f64 = let_nodes.iter().map(|n| n.mass).sum();
    let rel_err = (let_mass - total_mass).abs() / total_mass;
    assert!(
        rel_err < 1e-10,
        "masa LET exportada = {let_mass:.6e}, total = {total_mass:.6e}, err_rel = {rel_err:.2e}"
    );
}

// ── Test 2: serialización round-trip ─────────────────────────────────────────

/// Verifica que pack_let_nodes → unpack_let_nodes es lossless.
#[test]
fn let_wire_roundtrip() {
    let pos: Vec<Vec3> = uniform_positions(50, 7);
    let masses = vec![1.0f64; 50];
    let tree = Octree::build(&pos, &masses);

    let target_aabb = [5.0f64, 6.0, 5.0, 6.0, 5.0, 6.0];
    let nodes = tree.export_let(target_aabb, 0.5);
    assert!(!nodes.is_empty(), "debe exportar al menos un nodo");

    let packed = pack_let_nodes(&nodes);
    assert_eq!(packed.len(), nodes.len() * RMN_FLOATS);

    let unpacked = unpack_let_nodes(&packed);
    assert_eq!(unpacked.len(), nodes.len());

    for (orig, rec) in nodes.iter().zip(unpacked.iter()) {
        assert!((orig.com.x - rec.com.x).abs() < 1e-14, "com.x no coincide");
        assert!((orig.mass - rec.mass).abs() < 1e-14, "mass no coincide");
        assert_eq!(orig.quad, rec.quad, "quad no coincide");
        assert_eq!(orig.oct, rec.oct, "oct no coincide");
        assert!(
            (orig.half_size - rec.half_size).abs() < 1e-14,
            "half_size no coincide"
        );
    }
}

// ── Test 3: fuerza LET vs referencia directa ──────────────────────────────────

/// Verifica que la fuerza desde nodos LET coincide con la referencia de fuerza directa
/// cuando el conjunto de partículas fuente está suficientemente lejos del evaluador.
///
/// Setup: partículas fuente en [0, 0.5]³, evaluador en x = 5 (lejos).
/// La fuerza LET debe aproximar la fuerza directa dentro del error multipolar.
#[test]
fn let_force_matches_direct_far_field() {
    // Partículas fuente: 64 partículas en [0, 0.5]³.
    let source_pos: Vec<Vec3> = uniform_positions(64, 42)
        .into_iter()
        .map(|p| Vec3::new(p.x * 0.5, p.y * 0.5, p.z * 0.5))
        .collect();
    let source_masses = vec![1.0f64; 64];
    let tree = Octree::build(&source_pos, &source_masses);

    let g = 1.0f64;
    let eps2 = 0.01f64 * 0.01;
    let theta = 0.5f64;

    // Evaluador muy lejos (x = 5.0).
    let eval_pos = Vec3::new(5.0, 0.25, 0.25);

    // AABB "del receptor": solo un punto muy pequeño alrededor del evaluador.
    let target_aabb = [4.9f64, 5.1, 0.2, 0.3, 0.2, 0.3];
    let let_nodes = tree.export_let(target_aabb, theta);

    // Fuerza desde nodos LET.
    let a_let = accel_from_let(eval_pos, &let_nodes, g, eps2);

    // Fuerza directa de referencia (sin árbol, sin MAC).
    let mut a_direct = Vec3::zero();
    for (j, &pos_j) in source_pos.iter().enumerate() {
        a_direct += pairwise_accel_plummer(eval_pos, source_masses[j], pos_j, g, eps2);
    }

    // Error relativo: para campo lejano bien separado, el error debe ser < 1% con θ=0.5.
    let a_mag = a_direct.norm();
    let err = (a_let - a_direct).norm() / a_mag.max(1e-300);
    assert!(
        err < 0.02,
        "error LET vs directo = {err:.4} (> 2%); |a_direct| = {a_mag:.3e}, |a_let| = {:.3e}",
        a_let.norm()
    );
}

// ── Test 4: poda de subárbol ──────────────────────────────────────────────────

/// Verifica que para un receptor muy lejano, el árbol exporta pocos nodos (poda efectiva).
#[test]
fn export_let_prunes_subtrees() {
    let pos = uniform_positions(200, 42);
    let masses = vec![1.0f64; 200];
    let tree = Octree::build(&pos, &masses);

    // Receptor muy lejano (debería bastar con 1-3 nodos de alto nivel).
    let far_aabb = [100.0f64, 101.0, 100.0, 101.0, 100.0, 101.0];
    let near_aabb = [0.3f64, 0.7, 0.3, 0.7, 0.3, 0.7]; // receptor cercano

    let let_far = tree.export_let(far_aabb, 0.5);
    let let_near = tree.export_let(near_aabb, 0.5);

    // El LET para un receptor lejano debe ser mucho más compacto.
    assert!(
        let_far.len() < let_near.len(),
        "LET lejano ({} nodos) debe ser más compacto que cercano ({} nodos)",
        let_far.len(),
        let_near.len()
    );
    // Para un receptor muy lejano con θ=0.5, esperamos muy pocos nodos (≤ 10).
    assert!(
        let_far.len() <= 10,
        "LET muy lejano debería tener ≤ 10 nodos; tiene {}",
        let_far.len()
    );
}
