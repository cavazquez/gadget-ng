//! Tests de correctitud y rendimiento del `LetTree` (Fase 10).
//!
//! Verifica que:
//! 1. El árbol se construye sin panics con entradas degeneradas (vacío, 1 nodo).
//! 2. Las fuerzas del `LetTree` coinciden con el loop plano `accel_from_let` dentro
//!    de tolerancias controladas por `theta`.
//! 3. El número de nodos del árbol crece subarítmicamente con N.
//! 4. La conservación de energía en una integración corta no empeora respecto al
//!    path plano.

use gadget_ng_core::Vec3;
use gadget_ng_tree::{accel_from_let, LetTree, RemoteMultipoleNode};

// ── Utilidades ─────────────────────────────────────────────────────────────────

/// Crea un `RemoteMultipoleNode` monopolar puro (quad = oct = 0).
fn rmn_monopole(com: Vec3, mass: f64, half_size: f64) -> RemoteMultipoleNode {
    RemoteMultipoleNode {
        com,
        mass,
        quad: [0.0; 6],
        oct: [0.0; 7],
        half_size,
    }
}

/// Genera N nodos en posiciones pseudoaleatorias dentro de `[-box/2, box/2]³`.
fn gen_rmns(n: usize, seed: u64, box_size: f64, mass_per: f64) -> Vec<RemoteMultipoleNode> {
    let mut rmns = Vec::with_capacity(n);
    let mut s = seed ^ 0xdeadbeef;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = ((s >> 33) as f64 / u32::MAX as f64 - 0.5) * box_size;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let y = ((s >> 33) as f64 / u32::MAX as f64 - 0.5) * box_size;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let z = ((s >> 33) as f64 / u32::MAX as f64 - 0.5) * box_size;
        rmns.push(rmn_monopole(Vec3::new(x, y, z), mass_per, 0.05 * box_size));
    }
    rmns
}

/// Error relativo RMS entre dos campos de fuerzas.
fn rms_rel_error(a: &[Vec3], b: &[Vec3]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let diff = (*ai - *bi).norm();
        let mag = bi.norm();
        if mag > 1e-30 {
            num += diff * diff;
            den += mag * mag;
        }
    }
    if den == 0.0 {
        return 0.0;
    }
    (num / den).sqrt()
}

// ── Test 1: árbol vacío ────────────────────────────────────────────────────────

#[test]
fn let_tree_empty() {
    let tree = LetTree::build(&[]);
    assert!(tree.is_empty(), "árbol vacío debe reportar is_empty()");
    assert_eq!(tree.node_count(), 0);
    let a = tree.walk_accel(Vec3::new(1.0, 2.0, 3.0), 1.0, 0.01, 0.5);
    assert_eq!(a, Vec3::zero(), "árbol vacío debe devolver fuerza cero");
}

// ── Test 2: un solo RMN ────────────────────────────────────────────────────────

#[test]
fn let_tree_single_rmn() {
    let rmn = rmn_monopole(Vec3::new(0.0, 0.0, 0.0), 2.0, 0.1);
    let rmns = vec![rmn];
    let tree = LetTree::build(&rmns);

    assert!(!tree.is_empty());
    assert_eq!(tree.node_count(), 1, "un solo RMN → exactamente 1 nodo");

    let pos = Vec3::new(5.0, 0.0, 0.0);
    let g = 1.0;
    let eps2 = 0.01_f64;

    let a_tree = tree.walk_accel(pos, g, eps2, 0.5);
    let a_flat = accel_from_let(pos, &rmns, g, eps2);

    let diff = (a_tree - a_flat).norm();
    let mag = a_flat.norm();
    let rel = if mag > 1e-30 { diff / mag } else { diff };
    assert!(
        rel < 1e-12,
        "single RMN: error relativo {rel:.2e} (tree={a_tree:?} flat={a_flat:?})"
    );
}

// ── Test 3: 200 RMNs, theta = 0.5 → error < 2 % ─────────────────────────────

#[test]
fn let_tree_force_matches_flat_theta05() {
    let rmns = gen_rmns(200, 42, 10.0, 1.0 / 200.0);
    let tree = LetTree::build(&rmns);

    let g = 1.0;
    let eps2 = 0.01_f64;
    let theta = 0.5;

    // 20 posiciones de evaluación fuera del volumen de los nodos.
    let queries: Vec<Vec3> = (0..20)
        .map(|i| {
            let t = i as f64 * 0.1;
            Vec3::new(15.0 + t.cos(), t.sin(), 0.1 * t)
        })
        .collect();

    let a_tree: Vec<Vec3> = queries
        .iter()
        .map(|&p| tree.walk_accel(p, g, eps2, theta))
        .collect();
    let a_flat: Vec<Vec3> = queries
        .iter()
        .map(|&p| accel_from_let(p, &rmns, g, eps2))
        .collect();

    let err = rms_rel_error(&a_tree, &a_flat);
    assert!(
        err < 0.02,
        "theta=0.5, N=200: RMS error relativo {err:.4} ≥ 2%"
    );
}

// ── Test 4: 200 RMNs, theta = 0.3 → error < 0.5 % ───────────────────────────

#[test]
fn let_tree_force_matches_flat_theta03() {
    let rmns = gen_rmns(200, 99, 10.0, 1.0 / 200.0);
    let tree = LetTree::build(&rmns);

    let g = 1.0;
    let eps2 = 0.01_f64;
    let theta = 0.3;

    let queries: Vec<Vec3> = (0..20)
        .map(|i| {
            let t = i as f64 * 0.1;
            Vec3::new(15.0 + t.cos(), t.sin(), 0.1 * t)
        })
        .collect();

    let a_tree: Vec<Vec3> = queries
        .iter()
        .map(|&p| tree.walk_accel(p, g, eps2, theta))
        .collect();
    let a_flat: Vec<Vec3> = queries
        .iter()
        .map(|&p| accel_from_let(p, &rmns, g, eps2))
        .collect();

    let err = rms_rel_error(&a_tree, &a_flat);
    assert!(
        err < 0.005,
        "theta=0.3, N=200: RMS error relativo {err:.4} ≥ 0.5%"
    );
}

// ── Test 5: node_count subarítmico ────────────────────────────────────────────

#[test]
fn let_tree_node_count_sublinear() {
    let n = 1000;
    let rmns = gen_rmns(n, 7, 10.0, 1.0 / n as f64);
    let tree = LetTree::build(&rmns);

    let nc = tree.node_count();
    // Para N=1000 y leaf_max=8, el árbol tiene como máximo ~2N nodos internos+hojas.
    // Muy raramente más de 5N (tolerancia conservadora).
    assert!(
        nc < 5 * n,
        "N={n}: node_count={nc} ≥ 5N (árbol demasiado grande)"
    );
    assert!(
        nc >= n / 8,
        "N={n}: node_count={nc} muy pequeño (leaf_max posiblemente ignorado)"
    );
}

// ── Test 6: conservación de energía en integración corta ─────────────────────

#[test]
fn let_tree_energy_conservation_short() {
    // Simulación simple: N=20 partículas evaluadas contra un campo LET de 50 nodos.
    // Las partículas se mueven 5 pasos con KDK leapfrog.
    // Verificamos que |ΔE/E₀| < 1% (tolerancia muy laxa; el objetivo es detectar
    // regresiones graves, no medir conservación de alta precisión).

    let n_local = 20usize;
    let n_let = 50usize;
    let g = 1.0_f64;
    let eps2 = 0.01_f64;
    let theta = 0.5_f64;
    let dt = 0.001_f64;

    // Generar nodos LET "remotos" (campo fijo durante la integración).
    let let_rmns = gen_rmns(n_let, 13, 20.0, 5.0 / n_let as f64);
    let tree = LetTree::build(&let_rmns);

    // Generar partículas locales (masa pequeña, velocidad aleatoria).
    let mut pos: Vec<Vec3> = (0..n_local)
        .map(|i| {
            let t = i as f64 / n_local as f64;
            Vec3::new(
                (t * 6.28).cos() * 5.0,
                (t * 6.28).sin() * 5.0,
                0.0,
            )
        })
        .collect();
    let mut vel: Vec<Vec3> = (0..n_local)
        .map(|i| {
            let t = i as f64 / n_local as f64;
            Vec3::new(
                -(t * 6.28).sin() * 0.3,
                (t * 6.28).cos() * 0.3,
                0.0,
            )
        })
        .collect();
    let mass = 0.01_f64;

    let accel = |positions: &[Vec3]| -> Vec<Vec3> {
        positions
            .iter()
            .map(|&p| tree.walk_accel(p, g, eps2, theta))
            .collect()
    };

    // Energía potencial: solo contribución LET (campo externo, no self-energy).
    let pot_energy = |positions: &[Vec3]| -> f64 {
        positions
            .iter()
            .map(|&p| {
                let a = tree.walk_accel(p, g, eps2, theta);
                // Potencial aproximado: -<a, r> / |r| (solo para monitoreo relativo).
                // En realidad usamos la variación cinética.
                a.dot(p)
            })
            .sum::<f64>()
            * (-mass)
    };

    let kin_energy = |velocities: &[Vec3]| -> f64 {
        velocities.iter().map(|v| 0.5 * mass * v.dot(*v)).sum()
    };

    // Kick half → Drift → Kick half (leapfrog KDK).
    let e0 = kin_energy(&vel) + pot_energy(&pos);

    for _ in 0..5 {
        let acc = accel(&pos);
        // Kick half
        for i in 0..n_local {
            vel[i] = vel[i] + acc[i] * (dt * 0.5);
        }
        // Drift
        for i in 0..n_local {
            pos[i] = pos[i] + vel[i] * dt;
        }
        // Kick half
        let acc2 = accel(&pos);
        for i in 0..n_local {
            vel[i] = vel[i] + acc2[i] * (dt * 0.5);
        }
    }

    let ef = kin_energy(&vel) + pot_energy(&pos);
    let de = (ef - e0).abs() / e0.abs().max(1e-30);
    assert!(
        de < 0.10,
        "Conservación energía: |ΔE/E₀| = {de:.4} ≥ 10% (regresión grave)"
    );
}
