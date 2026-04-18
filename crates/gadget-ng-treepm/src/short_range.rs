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
//!
//! ## Periodicidad
//!
//! Las variantes `_periodic` aplican `minimum_image` en cada distancia
//! partícula–nodo y partícula–partícula, garantizando correctitud en cajas periódicas.
//! La distancia mínima a una AABB periódica también se calcula con wrap.

use gadget_ng_core::Vec3;
use gadget_ng_tree::{Octree, NO_CHILD};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

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

// ── Funciones de periodicidad ──────────────────────────────────────────────────

/// Aplica la convención de imagen mínima a un desplazamiento escalar.
///
/// Devuelve `dx'` tal que `|dx'| ≤ box_size / 2`, eligiendo la imagen
/// periódica más cercana del origen.
#[inline]
pub fn minimum_image(dx: f64, box_size: f64) -> f64 {
    let half = box_size * 0.5;
    let mut d = dx - box_size * (dx / box_size).round();
    // Corrección de borde por redondeo en límites exactos de ±half.
    if d > half {
        d -= box_size;
    } else if d < -half {
        d += box_size;
    }
    d
}

/// Distancia mínima al cuadrado entre el punto `xi` y la AABB periódica de un
/// nodo del octree con centro `com` y semilado `half_size`, en una caja periódica
/// de longitud `box_size`.
///
/// La AABB puede solapar el borde de la caja (wrap periódico); se considera
/// la imagen más cercana del nodo respecto a `xi`.
#[inline]
pub fn min_dist2_to_aabb_periodic(xi: Vec3, com: Vec3, half_size: f64, box_size: f64) -> f64 {
    // Desplazamiento del CoM al punto xi con minimum_image.
    let dxc = minimum_image(com.x - xi.x, box_size);
    let dyc = minimum_image(com.y - xi.y, box_size);
    let dzc = minimum_image(com.z - xi.z, box_size);

    // Distancia mínima de xi al segmento [-half, half] centrado en el CoM
    // usando la imagen mínima del desplazamiento.
    let ex = (dxc.abs() - half_size).max(0.0);
    let ey = (dyc.abs() - half_size).max(0.0);
    let ez = (dzc.abs() - half_size).max(0.0);
    ex * ex + ey * ey + ez * ez
}

// ── Complementary error function ──────────────────────────────────────────────

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

// ── Walk aperiódico (original) ─────────────────────────────────────────────────

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

    #[cfg(not(feature = "rayon"))]
    {
        for (k, &gi) in global_indices.iter().enumerate() {
            let xi = params.positions[gi];
            let mut a = Vec3::zero();
            walk_short_range(&tree, xi, gi, params, &mut a);
            out[k] = a;
        }
    }
    #[cfg(feature = "rayon")]
    {
        out.par_iter_mut()
            .zip(global_indices.par_iter())
            .for_each(|(a, &gi)| {
                let xi = params.positions[gi];
                let mut acc = Vec3::zero();
                walk_short_range(&tree, xi, gi, params, &mut acc);
                *a = acc;
            });
    }
}

/// Recorre el octree y acumula la fuerza de corto alcance sobre `xi` (partícula `skip`).
fn walk_short_range(tree: &Octree, xi: Vec3, skip: usize, p: &ShortRangeParams<'_>, a: &mut Vec3) {
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

// ── Walk periódico (Fase 21) ───────────────────────────────────────────────────

/// Parámetros extendidos para el walk periódico de corto alcance.
///
/// Extiende [`ShortRangeParams`] con `box_size` para aplicar minimum_image.
pub struct ShortRangeParamsPeriodic<'a> {
    pub positions: &'a [Vec3],
    pub masses: &'a [f64],
    pub eps2: f64,
    pub g: f64,
    pub r_split: f64,
    pub r_cut2: f64,
    pub box_size: f64,
}

/// Versión periódica de [`short_range_accels`].
///
/// Aplica `minimum_image` en cada distancia partícula–nodo y partícula–partícula.
/// El árbol se construye con todas las posiciones dadas (locales + halos).
///
/// - `n_local` — número de partículas locales (índices 0..n_local).
///   Las aceleraciones de salida corresponden exactamente a estas partículas.
/// - `out` — longitud = `n_local`.
pub fn short_range_accels_periodic(
    params: &ShortRangeParamsPeriodic<'_>,
    n_local: usize,
    out: &mut [Vec3],
) {
    assert_eq!(out.len(), n_local);
    assert!(n_local <= params.positions.len());
    assert_eq!(params.positions.len(), params.masses.len());

    if params.positions.is_empty() || n_local == 0 {
        return;
    }

    let tree = Octree::build(params.positions, params.masses);

    #[cfg(not(feature = "rayon"))]
    {
        for li in 0..n_local {
            let xi = params.positions[li];
            let mut a = Vec3::zero();
            walk_short_range_periodic(&tree, xi, li, params, &mut a);
            out[li] = a;
        }
    }
    #[cfg(feature = "rayon")]
    {
        out.par_iter_mut().enumerate().for_each(|(li, a)| {
            let xi = params.positions[li];
            let mut acc = Vec3::zero();
            walk_short_range_periodic(&tree, xi, li, params, &mut acc);
            *a = acc;
        });
    }
}

/// Walk periódico: aplica `minimum_image` en todas las distancias.
///
/// - La AABB de cada nodo usa [`min_dist2_to_aabb_periodic`] para el culling.
/// - Las distancias CoM–partícula usan `minimum_image` en cada componente.
/// - Las distancias par hoja–partícula usan `minimum_image` en cada componente.
fn walk_short_range_periodic(
    tree: &Octree,
    xi: Vec3,
    skip: usize,
    p: &ShortRangeParamsPeriodic<'_>,
    a: &mut Vec3,
) {
    let box_size = p.box_size;
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    stack.push(tree.root);

    while let Some(node_idx) = stack.pop() {
        let node = &tree.nodes[node_idx as usize];

        if node.mass == 0.0 {
            continue;
        }

        // Distancia mínima periódica a la AABB del nodo.
        let dist2_aabb = min_dist2_to_aabb_periodic(xi, node.com, node.half_size, box_size);
        if dist2_aabb > p.r_cut2 {
            continue;
        }

        let is_leaf = node.children.iter().all(|&c| c == NO_CHILD);

        if is_leaf {
            if let Some(j) = node.particle_idx {
                if j != skip {
                    let rj = p.positions[j];
                    let rx = minimum_image(rj.x - xi.x, box_size);
                    let ry = minimum_image(rj.y - xi.y, box_size);
                    let rz = minimum_image(rj.z - xi.z, box_size);
                    let r2 = rx * rx + ry * ry + rz * rz + p.eps2;
                    if r2 - p.eps2 > p.r_cut2 {
                        continue; // fuera del cutoff con minimum_image
                    }
                    let r = r2.sqrt();
                    let w = erfc_factor(r, p.r_split);
                    let inv3 = p.g * p.masses[j] * w / (r2 * r);
                    *a += Vec3::new(rx * inv3, ry * inv3, rz * inv3);
                }
            }
        } else {
            // Distancia CoM con minimum_image para el monopolo.
            let dx = minimum_image(node.com.x - xi.x, box_size);
            let dy = minimum_image(node.com.y - xi.y, box_size);
            let dz = minimum_image(node.com.z - xi.z, box_size);
            let d2 = dx * dx + dy * dy + dz * dz;
            let use_monopole = node.half_size < 0.1 * p.r_cut2.sqrt() && d2 > 1e-30;

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

// ── Tests ─────────────────────────────────────────────────────────────────────

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

    #[test]
    fn minimum_image_basic() {
        let box_size = 1.0_f64;
        // Desplazamiento pequeño positivo: sin cambio.
        assert!((minimum_image(0.3, box_size) - 0.3).abs() < 1e-14);
        // Desplazamiento pequeño negativo: sin cambio.
        assert!((minimum_image(-0.3, box_size) + 0.3).abs() < 1e-14);
        // Desplazamiento > 0.5: se mapea a -0.3.
        let d = minimum_image(0.7, box_size);
        assert!((d + 0.3).abs() < 1e-12, "minimum_image(0.7, 1.0) debe ser -0.3, got {d}");
        // Desplazamiento < -0.5: se mapea a +0.3.
        let d2 = minimum_image(-0.7, box_size);
        assert!((d2 - 0.3).abs() < 1e-12, "minimum_image(-0.7, 1.0) debe ser 0.3, got {d2}");
        // Exactamente en el borde: |d| = 0.5.
        let d3 = minimum_image(0.5, box_size);
        assert!(d3.abs() <= 0.5 + 1e-12, "minimum_image(0.5, 1.0) debe ser ≤0.5, got {d3}");
    }

    #[test]
    fn minimum_image_symmetry() {
        let box_size = 2.0_f64;
        for dx in [-1.5, -0.8, 0.0, 0.3, 0.9, 1.1, 1.9_f64] {
            let d = minimum_image(dx, box_size);
            assert!(
                d.abs() <= box_size / 2.0 + 1e-12,
                "minimum_image({dx}, {box_size}) = {d} fuera de [-L/2, L/2]"
            );
        }
    }

    #[test]
    fn min_dist2_to_aabb_periodic_trivial() {
        // Punto en el interior de la AABB: distancia debe ser 0.
        let xi = Vec3::new(0.5, 0.5, 0.5);
        let com = Vec3::new(0.5, 0.5, 0.5);
        let d2 = min_dist2_to_aabb_periodic(xi, com, 0.3, 1.0);
        assert!(d2 < 1e-14, "punto dentro de AABB: d2={d2}");
    }

    #[test]
    fn min_dist2_to_aabb_periodic_wrap() {
        // xi en z=0.05, nodo con CoM en z=0.95, half=0.04, box=1.0.
        // La distancia directa |0.95-0.05|=0.9 (fuera del cutoff).
        // La imagen periódica: |0.95-0.05-1.0|=|−0.1|=0.1; la AABB
        // tiene half=0.04, así que la distancia mínima es 0.1-0.04=0.06.
        let xi = Vec3::new(0.0, 0.0, 0.05);
        let com = Vec3::new(0.0, 0.0, 0.95);
        let d2 = min_dist2_to_aabb_periodic(xi, com, 0.04, 1.0);
        let expected = (0.06_f64) * (0.06_f64); // ≈ 0.0036
        assert!(
            (d2 - expected).abs() < 1e-10,
            "d2={d2:.6e} vs esperado={expected:.6e}"
        );
    }
}
