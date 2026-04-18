//! Geometría periódica 3D para el halo volumétrico del árbol de corto alcance (Fase 22).
//!
//! Proporciona:
//! - [`Aabb3`] — AABB rectangular genérica (semiejes distintos en x, y, z)
//! - [`min_dist2_to_aabb_3d_periodic`] — distancia mínima al cuadrado periódica a AABB rectangular
//! - [`compute_aabb_3d`] — AABB real de un slice de partículas
//! - [`aabb_to_f64`] / [`f64_to_aabb`] — serialización para allgather
//!
//! ## Por qué se necesita este módulo
//!
//! La función existente `min_dist2_to_aabb_periodic` en `gadget-ng-treepm` solo acepta
//! AABBs cúbicas (un único `half_size`). Para verificar si una partícula pertenece al
//! halo de dominio de otro rank (cuyo AABB puede ser un paralelepípedo rectangular),
//! se necesita la variante con semiejes distintos en cada eje.
//!
//! ## Correctitud periódica
//!
//! El criterio de pertenencia al halo de rank r para una partícula p es:
//!
//! ```text
//! min_dist2_to_aabb_3d_periodic(p.position, aabb_r, box_size) < halo_width²
//! ```
//!
//! Esto usa `minimum_image` componente a componente sobre el vector CoM→p,
//! eligiendo la imagen periódica del dominio del rank más cercana a p.
//! Es correcto para todos los casos de borde, incluyendo interacciones diagonales
//! y combinaciones (x+y, x+z, y+z, x+y+z).

use gadget_ng_core::Particle;

// ── Tipo AABB ─────────────────────────────────────────────────────────────────

/// AABB (Axis-Aligned Bounding Box) rectangular 3D.
///
/// `lo[k]` = mínimo en el eje k, `hi[k]` = máximo en el eje k.
/// Para una caja periódica, los valores deben estar en `[0, box_size)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb3 {
    /// Esquina mínima: [xlo, ylo, zlo]
    pub lo: [f64; 3],
    /// Esquina máxima: [xhi, yhi, zhi]
    pub hi: [f64; 3],
}

impl Aabb3 {
    /// AABB vacía (infinitos invertidos); valor seguro para folding con min/max.
    pub fn empty() -> Self {
        Self {
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
        }
    }

    /// Devuelve el CoM (centro geométrico) del AABB.
    pub fn center(&self) -> [f64; 3] {
        [
            (self.lo[0] + self.hi[0]) * 0.5,
            (self.lo[1] + self.hi[1]) * 0.5,
            (self.lo[2] + self.hi[2]) * 0.5,
        ]
    }

    /// Devuelve los semiejes (half-extents) en cada eje.
    pub fn half_extents(&self) -> [f64; 3] {
        [
            (self.hi[0] - self.lo[0]) * 0.5,
            (self.hi[1] - self.lo[1]) * 0.5,
            (self.hi[2] - self.lo[2]) * 0.5,
        ]
    }

    /// Devuelve `true` si el AABB es válido (lo ≤ hi en todos los ejes).
    pub fn is_valid(&self) -> bool {
        self.lo[0] <= self.hi[0] && self.lo[1] <= self.hi[1] && self.lo[2] <= self.hi[2]
    }
}

// ── Serialización para allgather ──────────────────────────────────────────────

/// Serializa un `Aabb3` como 6 f64 en el orden `[xlo, xhi, ylo, yhi, zlo, zhi]`.
///
/// Compatible con el formato usado por `compute_aabb` en `mpi_rt.rs`.
pub fn aabb_to_f64(aabb: &Aabb3) -> [f64; 6] {
    [
        aabb.lo[0], aabb.hi[0], // xlo, xhi
        aabb.lo[1], aabb.hi[1], // ylo, yhi
        aabb.lo[2], aabb.hi[2], // zlo, zhi
    ]
}

/// Deserializa 6 f64 en el orden `[xlo, xhi, ylo, yhi, zlo, zhi]` a un `Aabb3`.
///
/// Devuelve `None` si el slice tiene menos de 6 elementos.
pub fn f64_to_aabb(v: &[f64]) -> Option<Aabb3> {
    if v.len() < 6 {
        return None;
    }
    Some(Aabb3 {
        lo: [v[0], v[2], v[4]],
        hi: [v[1], v[3], v[5]],
    })
}

// ── AABB real de partículas ───────────────────────────────────────────────────

/// Calcula el AABB ajustado de un slice de partículas.
///
/// Para un slice vacío devuelve `Aabb3::empty()` (infinitos invertidos),
/// que es el valor neutral para operaciones allreduce min/max.
pub fn compute_aabb_3d(particles: &[Particle]) -> Aabb3 {
    if particles.is_empty() {
        return Aabb3::empty();
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in particles {
        let pos = [p.position.x, p.position.y, p.position.z];
        for k in 0..3 {
            if pos[k] < lo[k] {
                lo[k] = pos[k];
            }
            if pos[k] > hi[k] {
                hi[k] = pos[k];
            }
        }
    }
    Aabb3 { lo, hi }
}

// ── Distancia mínima periódica punto–AABB 3D ─────────────────────────────────

/// Aplica la convención de imagen mínima a un desplazamiento escalar.
///
/// Mismo algoritmo que `minimum_image` en `gadget-ng-treepm/src/short_range.rs`,
/// duplicado aquí para evitar una dependencia circular entre crates.
#[inline]
pub fn minimum_image_scalar(dx: f64, box_size: f64) -> f64 {
    let half = box_size * 0.5;
    let mut d = dx - box_size * (dx / box_size).round();
    if d > half {
        d -= box_size;
    } else if d < -half {
        d += box_size;
    }
    d
}

/// Distancia mínima al cuadrado desde el punto `p` hasta la AABB rectangular
/// `aabb` en una caja periódica de longitud `box_size`.
///
/// ## Algoritmo
///
/// 1. Calcula el vector CoM → p con `minimum_image` en cada componente,
///    eligiendo la copia periódica del AABB más cercana a p.
/// 2. Para cada eje k, la distancia al segmento `[-half_k, half_k]` centrado
///    en el CoM (imagen mínima) es `max(|d_k| - half_k, 0)`.
/// 3. Devuelve la suma de cuadrados de esas excedentes.
///
/// ## Casos especiales
///
/// - Si p está dentro del AABB (o en su frontera): devuelve 0.
/// - Si el AABB es inválido (vacío): devuelve `f64::INFINITY`.
///
/// ## Correctitud periódica
///
/// Cubre los 26 imágenes periódicas del AABB (todas las combinaciones de wrap
/// en ±1 en cada eje), eligiendo la más cercana a p componente a componente.
/// Este criterio es necesario y suficiente para distancias de la norma ‖·‖₂.
#[inline]
pub fn min_dist2_to_aabb_3d_periodic(p: [f64; 3], aabb: &Aabb3, box_size: f64) -> f64 {
    if !aabb.is_valid() {
        return f64::INFINITY;
    }
    let com = aabb.center();
    let half = aabb.half_extents();

    let mut dist2 = 0.0;
    for k in 0..3 {
        let d = minimum_image_scalar(com[k] - p[k], box_size);
        let excess = (d.abs() - half[k]).max(0.0);
        dist2 += excess * excess;
    }
    dist2
}

/// Comprueba si la partícula en posición `pos` pertenece al halo de un dominio
/// con AABB `aabb`, con ancho de halo `halo_width`, en caja periódica `box_size`.
///
/// Equivalente a `min_dist2_to_aabb_3d_periodic(pos, aabb, box_size) < halo_width²`.
#[inline]
pub fn is_in_periodic_halo(pos: [f64; 3], aabb: &Aabb3, halo_width: f64, box_size: f64) -> bool {
    min_dist2_to_aabb_3d_periodic(pos, aabb, box_size) < halo_width * halo_width
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_aabb(xlo: f64, xhi: f64, ylo: f64, yhi: f64, zlo: f64, zhi: f64) -> Aabb3 {
        Aabb3 {
            lo: [xlo, ylo, zlo],
            hi: [xhi, yhi, zhi],
        }
    }

    #[test]
    fn minimum_image_scalar_basic() {
        let b = 1.0_f64;
        assert!((minimum_image_scalar(0.3, b) - 0.3).abs() < 1e-14);
        assert!((minimum_image_scalar(-0.3, b) + 0.3).abs() < 1e-14);
        let d = minimum_image_scalar(0.7, b);
        assert!((d + 0.3).abs() < 1e-12, "0.7 → -0.3, got {d}");
    }

    #[test]
    fn aabb_center_and_half_extents() {
        let aabb = make_aabb(0.0, 0.5, 0.1, 0.3, 0.2, 0.8);
        let c = aabb.center();
        let h = aabb.half_extents();
        assert!((c[0] - 0.25).abs() < 1e-14);
        assert!((c[1] - 0.20).abs() < 1e-14);
        assert!((c[2] - 0.50).abs() < 1e-14);
        assert!((h[0] - 0.25).abs() < 1e-14);
        assert!((h[1] - 0.10).abs() < 1e-14);
        assert!((h[2] - 0.30).abs() < 1e-14);
    }

    #[test]
    fn aabb_serde_roundtrip() {
        let aabb = make_aabb(0.1, 0.6, 0.2, 0.7, 0.3, 0.8);
        let serialized = aabb_to_f64(&aabb);
        let deserialized = f64_to_aabb(&serialized).unwrap();
        for k in 0..3 {
            assert!((deserialized.lo[k] - aabb.lo[k]).abs() < 1e-14);
            assert!((deserialized.hi[k] - aabb.hi[k]).abs() < 1e-14);
        }
    }

    #[test]
    fn point_inside_aabb_gives_zero_dist() {
        let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
        let p = [0.25, 0.25, 0.25];
        assert!(min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0) < 1e-14);
    }

    #[test]
    fn point_outside_non_periodic_gives_correct_dist() {
        // p a distancia 0.1 del borde xhi=0.5, lejos de bordes periódicos.
        let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
        let p = [0.6, 0.25, 0.25]; // dx_excess = 0.1
        let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
        assert!((d2 - 0.01).abs() < 1e-12, "d2={d2}");
    }

    #[test]
    fn periodic_x_border() {
        // p en x=0.95, AABB con xlo=0.0 (caja L=1).
        // min_image(0.25 - 0.95, 1) = min_image(-0.7, 1) = 0.3 → pero CoM es 0.25
        // min_image(0.25 - 0.95) = -0.7+1=0.3 → |0.3|-0.25=0.05 → dist_x=0.05
        let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
        let p = [0.95, 0.25, 0.25]; // x=0.95, distancia periódica al borde xlo=0: 0.05
        let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
        let expected = 0.05_f64 * 0.05;
        assert!(
            (d2 - expected).abs() < 1e-10,
            "dist_x periódica: d2={d2:.6e} vs {expected:.6e}"
        );
    }

    #[test]
    fn periodic_y_border() {
        let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
        let p = [0.25, 0.95, 0.25]; // y=0.95, distancia periódica al borde ylo=0: 0.05
        let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
        let expected = 0.05_f64 * 0.05;
        assert!(
            (d2 - expected).abs() < 1e-10,
            "dist_y periódica: d2={d2:.6e} vs {expected:.6e}"
        );
    }

    #[test]
    fn periodic_z_border() {
        let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
        let p = [0.25, 0.25, 0.95]; // z=0.95, distancia periódica al borde zlo=0: 0.05
        let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
        let expected = 0.05_f64 * 0.05;
        assert!(
            (d2 - expected).abs() < 1e-10,
            "dist_z periódica: d2={d2:.6e} vs {expected:.6e}"
        );
    }

    #[test]
    fn periodic_diagonal_xyz() {
        // p en (0.95, 0.95, 0.95), AABB = [0,0.5)³.
        // CoM = (0.25, 0.25, 0.25), half = (0.25, 0.25, 0.25).
        // min_image(0.25-0.95, 1) = min_image(-0.7, 1) = 0.3 por cada eje.
        // excess_k = |0.3| - 0.25 = 0.05 en cada eje.
        // dist2 = 3 * 0.05² = 0.0075.
        let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
        let p = [0.95, 0.95, 0.95];
        let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
        let expected = 3.0 * 0.05_f64 * 0.05;
        assert!(
            (d2 - expected).abs() < 1e-10,
            "diagonal periódico: d2={d2:.6e} vs {expected:.6e}"
        );
        // sqrt(d2) ≈ 0.0866 < r_cut=0.1 → debe estar en el halo
        assert!(
            is_in_periodic_halo(p, &aabb, 0.1, 1.0),
            "partícula diagonal periódica debe estar en halo con r_cut=0.1"
        );
    }

    #[test]
    fn halo_1d_z_misses_diagonal() {
        // Demostración del gap del halo 1D-z para descomposición en octantes.
        // Rank 0: [0,0.5)³  →  my_z_lo=0, my_z_hi=0.5, halo_width=0.1
        // Partícula en (0.95, 0.95, 0.95): z=0.95.
        // Criterio 1D: z < my_z_lo + halo_width = 0.1 → EXCLUIDA (z=0.95 > 0.1)
        // Y también: z > my_z_hi - halo_width = 0.4 → para enviar al vecino derecho
        // El rank 1 enviaría a rank 0 partículas con z < z_lo_rank1 + halo_width.
        // Con descomposición octante, z_lo_rank1 = 0.5, halo = 0.1 → z < 0.6.
        // La partícula tiene z=0.95 > 0.6 → NO enviada por halo 1D. ✗
        let my_z_lo = 0.0_f64;
        let halo_width = 0.1_f64;
        let z_particle = 0.95_f64;
        // Criterio 1D: z ∈ [z_lo - halo_width, z_lo) → NOT included from "left"
        // Rank vecino (z_lo'=0.5): envía si z < z_lo' + halo_width = 0.6
        let z_lo_neighbor = 0.5_f64;
        let included_by_1d = z_particle < z_lo_neighbor + halo_width;
        assert!(
            !included_by_1d,
            "halo 1D-z debe EXCLUIR partícula diagonal z=0.95 (criterio z < {:.2}): got included={}",
            z_lo_neighbor + halo_width,
            included_by_1d
        );

        // Pero la distancia 3D periódica es < r_cut → debería interactuar.
        let aabb_rank0 = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
        let p = [0.95, 0.95, 0.95];
        let d = min_dist2_to_aabb_3d_periodic(p, &aabb_rank0, 1.0).sqrt();
        assert!(
            d < halo_width,
            "distancia 3D periódica={d:.4} debe ser < r_cut={halo_width} para confirmar la interacción"
        );
    }

    #[test]
    fn halo_3d_catches_diagonal() {
        // El halo 3D periódico SÍ incluye la partícula diagonal.
        let aabb_rank0 = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
        let p = [0.95, 0.95, 0.95];
        let r_cut = 0.1_f64;
        assert!(
            is_in_periodic_halo(p, &aabb_rank0, r_cut, 1.0),
            "halo 3D debe INCLUIR partícula diagonal (0.95,0.95,0.95) con r_cut={r_cut}"
        );
    }

    #[test]
    fn invalid_aabb_gives_infinity() {
        let aabb = Aabb3::empty();
        let p = [0.5, 0.5, 0.5];
        assert!(min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0).is_infinite());
    }

    #[test]
    fn compute_aabb_3d_empty() {
        let aabb = compute_aabb_3d(&[]);
        assert!(!aabb.is_valid());
    }
}
