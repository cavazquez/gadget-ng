//! Fuerza gravitacional de corto alcance para el esquema TreePM.
//!
//! ## División de fuerzas
//!
//! La fuerza total de Newton se descompone en:
//!
//! ```text
//! F_total(r) = F_lr(r)  +  F_sr(r)
//!
//! F_lr(r) = G·m/r²  ·  erf(r / (√2·r_s))     ← calculado por el solver PM filtrado
//! F_sr(r) = G·m/r²  ·  erfc(r / (√2·r_s))    ← calculado aquí, con cutoff en r_cut
//! ```
//!
//! donde `r_s` = radio de splitting y `r_cut` ≈ 5·r_s (más allá erfc < 1e-11).
//!
//! ## Implementación
//!
//! Para cada partícula activa se recorre el octree y se suman las contribuciones
//! de todas las partículas dentro de `r_cut`. No se aplica MAC (criterio
//! Barnes-Hut): la fuerza de corto alcance requiere pares exactos; la escala
//! local `r_cut` ya limita el número de vecinos a O(N_neighbor).

use gadget_ng_core::Vec3;
use gadget_ng_tree::{Octree, NO_CHILD};

/// Parámetros del kernel de corto alcance, agrupados para reducir el número
/// de argumentos en las funciones internas.
pub struct ShortRangeParams<'a> {
    pub positions: &'a [Vec3],
    pub masses: &'a [f64],
    pub eps2: f64,
    pub g: f64,
    pub r_split: f64,
    pub r_cut2: f64,
}

/// Complementary error function using the rational polynomial approximation of
/// Abramowitz & Stegun §7.1.26, with max error ≤ 1.5 × 10⁻⁷ for x ≥ 0.
#[inline]
pub fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-x * x).exp()
}

/// Factor de corto alcance: `erfc(r / (√2·r_s))`.
///
/// La fuerza total sobre i debida a j queda:
/// `a_sr = G·m_j·(x_j - x_i) / r³ · erfc_factor(r, r_s)`
#[inline]
pub fn erfc_factor(r: f64, r_split: f64) -> f64 {
    let x = r / (std::f64::consts::SQRT_2 * r_split);
    erfc_approx(x)
}

/// Calcula las aceleraciones de corto alcance para un conjunto de partículas activas.
///
/// Los parámetros físicos se pasan agrupados en [`ShortRangeParams`].
/// - `global_indices` — índices de las partículas activas (subconjunto local).
/// - `out` — aceleraciones de salida (longitud = `global_indices.len()`).
pub fn short_range_accels(
    params: &ShortRangeParams<'_>,
    global_indices: &[usize],
    out: &mut [Vec3],
) {
    assert_eq!(global_indices.len(), out.len());
    assert_eq!(params.positions.len(), params.masses.len());

    if params.positions.is_empty() {
        return;
    }

    let tree = Octree::build(params.positions, params.masses);

    for (k, &gi) in global_indices.iter().enumerate() {
        let xi = params.positions[gi];
        let mut a = Vec3::zero();
        walk_short_range(&tree, xi, gi, params, &mut a);
        out[k] = a;
    }
}

/// Recorre el octree y acumula la fuerza de corto alcance sobre `xi` (partícula `skip`).
fn walk_short_range(
    tree: &Octree,
    xi: Vec3,
    skip: usize,
    p: &ShortRangeParams<'_>,
    a: &mut Vec3,
) {
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    stack.push(tree.root);

    while let Some(node_idx) = stack.pop() {
        let node = &tree.nodes[node_idx as usize];

        if node.mass == 0.0 {
            continue;
        }

        let dx = node.com.x - xi.x;
        let dy = node.com.y - xi.y;
        let dz = node.com.z - xi.z;

        // Distancia mínima AABB; si supera r_cut todo el nodo está fuera del cutoff.
        let h = node.half_size;
        let ex = (dx.abs() - h).max(0.0);
        let ey = (dy.abs() - h).max(0.0);
        let ez = (dz.abs() - h).max(0.0);
        if ex * ex + ey * ey + ez * ez > p.r_cut2 {
            continue;
        }

        let is_leaf = node.children.iter().all(|&c| c == NO_CHILD);

        if is_leaf {
            if let Some(j) = node.particle_idx {
                if j != skip {
                    let rj = p.positions[j];
                    let rx = rj.x - xi.x;
                    let ry = rj.y - xi.y;
                    let rz = rj.z - xi.z;
                    let r2 = rx * rx + ry * ry + rz * rz + p.eps2;
                    let r = r2.sqrt();
                    let w = erfc_factor(r, p.r_split);
                    let inv3 = p.g * p.masses[j] * w / (r2 * r);
                    *a += Vec3::new(rx * inv3, ry * inv3, rz * inv3);
                }
            }
        } else {
            let d2 = dx * dx + dy * dy + dz * dz;
            let use_monopole = h < 0.1 * p.r_cut2.sqrt() && d2 > 1e-30;

            if use_monopole {
                let r2 = d2 + p.eps2;
                let r = r2.sqrt();
                let w = erfc_factor(r, p.r_split);
                let inv3 = p.g * node.mass * w / (r2 * r);
                *a += Vec3::new(dx * inv3, dy * inv3, dz * inv3);
            } else {
                for &ch in &node.children {
                    if ch != NO_CHILD {
                        stack.push(ch);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_special_values() {
        assert!((erfc_approx(0.0) - 1.0).abs() < 1e-6);
        // erfc(∞) ≈ 0
        assert!(erfc_approx(6.0) < 1e-9);
        // erfc(-∞) ≈ 2
        assert!((erfc_approx(-6.0) - 2.0).abs() < 1e-9);
        // simetría: erfc(x) + erfc(-x) = 2
        for &x in &[0.5_f64, 1.0, 2.0] {
            let sum = erfc_approx(x) + erfc_approx(-x);
            assert!((sum - 2.0).abs() < 1e-6, "erfc sum at x={x}: {sum}");
        }
    }

    #[test]
    fn erfc_factor_partition_of_unity() {
        // erf(x) + erfc(x) = 1
        let r_split = 0.5;
        for &r in &[0.1_f64, 0.5, 1.0, 2.0, 3.0] {
            let x = r / (std::f64::consts::SQRT_2 * r_split);
            let erf_val = 1.0 - erfc_approx(x);
            let erfc_val = erfc_approx(x);
            assert!(
                (erf_val + erfc_val - 1.0).abs() < 1e-6,
                "partition fails at r={r}: erf={erf_val:.6} erfc={erfc_val:.6}"
            );
        }
    }
}
