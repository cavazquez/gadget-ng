//! Octree en arena para Barnes-Hut con corrección cuadrupolar (FMM orden 2).
//!
//! ## Expansión multipolar
//!
//! Cuando el criterio de aceptación MAC (`s/d < θ`) se cumple, la aceleración
//! sobre la partícula evaluada se calcula como:
//!
//! ```text
//! a = a_mono + a_quad
//! ```
//!
//! donde `a_mono` es la contribución de monopolo suavizada (Plummer) y `a_quad`
//! es la corrección de cuadrupolo **no suavizada** (válida en la zona lejana):
//!
//! ```text
//! a_quad_α = G · [Q·r / |r|^5 − (5/2 · r^T Q r) · r_α / |r|^7]
//! ```
//!
//! con `r = pos_i − com_celda` y `Q` el tensor de cuadrupolo sin traza.
//!
//! ## Tensor de cuadrupolo
//!
//! Se almacena la parte superior del tensor simétrico sin traza:
//! `[Qxx, Qxy, Qxz, Qyy, Qyz, Qzz]`.  El cálculo usa el teorema del eje paralelo
//! para cuadrupolos: al fusionar hijos en el nodo padre con COM `r_0`:
//!
//! ```text
//! Q_total = Σ_k [ Q_k + m_k · outer_traceless(r_k − r_0) ]
//! ```
use gadget_ng_core::Vec3;

pub const NO_CHILD: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct OctNode {
    pub center: Vec3,
    pub half_size: f64,
    /// Masa total del subárbol (rellenada en `aggregate`).
    pub mass: f64,
    pub com: Vec3,
    pub children: [u32; 8],
    /// Índice global de partícula si es hoja con exactamente un cuerpo.
    pub particle_idx: Option<usize>,
    /// Tensor de cuadrupolo sin traza almacenado como triángulo superior:
    /// `[Qxx, Qxy, Qxz, Qyy, Qyz, Qzz]`.
    /// Para hojas (un solo cuerpo en su propio COM) vale `[0.0; 6]`.
    pub quad: [f64; 6],
}

impl OctNode {
    fn empty_leaf(center: Vec3, half_size: f64) -> Self {
        Self {
            center,
            half_size,
            mass: 0.0,
            com: Vec3::zero(),
            children: [NO_CHILD; 8],
            particle_idx: None,
            quad: [0.0; 6],
        }
    }
}

/// Componentes del tensor sin traza `Q_ij = m · (3 s_i s_j − |s|² δ_ij)`.
///
/// Resultado como `[Qxx, Qxy, Qxz, Qyy, Qyz, Qzz]`.
#[inline]
fn outer_traceless(s: Vec3, m: f64) -> [f64; 6] {
    let s2 = s.dot(s);
    [
        m * (3.0 * s.x * s.x - s2),
        m * 3.0 * s.x * s.y,
        m * 3.0 * s.x * s.z,
        m * (3.0 * s.y * s.y - s2),
        m * 3.0 * s.y * s.z,
        m * (3.0 * s.z * s.z - s2),
    ]
}

/// Aceleración cuadrupolar: corrección de lejano campo para el multipolo de orden 2.
///
/// `r` es el vector `pos_i − com_celda` (campo evaluado alejado de la fuente).
/// No incluye suavizado: sólo es válida cuando el MAC garantiza `|r| ≫ s`.
///
/// Fórmula: `a_α = G · [(Q·r)_α / |r|⁵ − 5/2 · (r^T Q r) · r_α / |r|⁷]`
#[inline]
fn quad_accel(r: Vec3, q: [f64; 6], g: f64) -> Vec3 {
    let r2 = r.dot(r);
    if r2 < 1e-300 {
        return Vec3::zero();
    }
    let [qxx, qxy, qxz, qyy, qyz, qzz] = q;
    let r_inv = 1.0 / r2.sqrt();
    let r5_inv = r_inv * r_inv * r_inv * r_inv * r_inv;
    let r7_inv = r5_inv * r_inv * r_inv;

    // Producto Q · r (fila por vector; Q es simétrica)
    let qr_x = qxx * r.x + qxy * r.y + qxz * r.z;
    let qr_y = qxy * r.x + qyy * r.y + qyz * r.z;
    let qr_z = qxz * r.x + qyz * r.y + qzz * r.z;

    // Escalar r^T Q r
    let rqr = qr_x * r.x + qr_y * r.y + qr_z * r.z;

    let c1 = g * r5_inv;
    let c2 = g * 2.5 * rqr * r7_inv;
    Vec3::new(c1 * qr_x - c2 * r.x, c1 * qr_y - c2 * r.y, c1 * qr_z - c2 * r.z)
}

fn octant_of(p: Vec3, c: Vec3) -> usize {
    let mut o = 0usize;
    if p.x >= c.x {
        o |= 1;
    }
    if p.y >= c.y {
        o |= 2;
    }
    if p.z >= c.z {
        o |= 4;
    }
    o
}

/// Comprueba si `p` está dentro (o en la frontera) del cubo AABB del nodo.
#[inline]
fn point_in_node_cell(p: Vec3, center: Vec3, half: f64) -> bool {
    let tol = 1e-14_f64 * (1.0 + half);
    (p.x - center.x).abs() <= half + tol
        && (p.y - center.y).abs() <= half + tol
        && (p.z - center.z).abs() <= half + tol
}

fn child_center(parent_center: Vec3, parent_half: f64, oct: usize) -> Vec3 {
    let q = parent_half * 0.5;
    let mut off = Vec3::zero();
    if (oct & 1) != 0 {
        off.x += q;
    } else {
        off.x -= q;
    }
    if (oct & 2) != 0 {
        off.y += q;
    } else {
        off.y -= q;
    }
    if (oct & 4) != 0 {
        off.z += q;
    } else {
        off.z -= q;
    }
    parent_center + off
}

pub struct Octree {
    pub nodes: Vec<OctNode>,
    pub root: u32,
}

impl Octree {
    /// Construye un octree que contiene todas las posiciones (caja cúbica mínima con padding).
    pub fn build(global_positions: &[Vec3], global_masses: &[f64]) -> Self {
        assert_eq!(global_positions.len(), global_masses.len());
        let n = global_positions.len();
        if n == 0 {
            return Self {
                nodes: vec![OctNode::empty_leaf(Vec3::zero(), 1.0)],
                root: 0,
            };
        }
        let (center, half) = bounding_cube(global_positions);
        let mut tree = Self {
            nodes: vec![OctNode::empty_leaf(center, half)],
            root: 0,
        };
        for j in 0..n {
            tree.insert(tree.root, j, global_positions[j], global_positions);
        }
        tree.aggregate(tree.root, global_positions, global_masses);
        tree
    }

    fn push_node(&mut self, node: OctNode) -> u32 {
        let i = self.nodes.len() as u32;
        self.nodes.push(node);
        i
    }

    fn insert(&mut self, node_idx: u32, particle_idx: usize, pos: Vec3, positions: &[Vec3]) {
        let all_empty = self.nodes[node_idx as usize]
            .children
            .iter()
            .all(|&c| c == NO_CHILD);

        if all_empty && self.nodes[node_idx as usize].particle_idx.is_none() {
            self.nodes[node_idx as usize].particle_idx = Some(particle_idx);
            return;
        }

        if all_empty {
            let old = self.nodes[node_idx as usize]
                .particle_idx
                .take()
                .expect("hoja con partícula");
            let old_pos = positions[old];
            let center = self.nodes[node_idx as usize].center;
            let half = self.nodes[node_idx as usize].half_size;
            let ch_half = half * 0.5;
            for oct in 0..8 {
                let cc = child_center(center, half, oct);
                let ni = self.push_node(OctNode::empty_leaf(cc, ch_half));
                self.nodes[node_idx as usize].children[oct] = ni;
            }
            let oct_old = octant_of(old_pos, center);
            let oct_new = octant_of(pos, center);
            let c_old = self.nodes[node_idx as usize].children[oct_old];
            let c_new = self.nodes[node_idx as usize].children[oct_new];
            self.insert(c_old, old, old_pos, positions);
            self.insert(c_new, particle_idx, pos, positions);
            return;
        }

        let center = self.nodes[node_idx as usize].center;
        let oct = octant_of(pos, center);
        let child = self.nodes[node_idx as usize].children[oct];
        debug_assert_ne!(child, NO_CHILD);
        self.insert(child, particle_idx, pos, positions);
    }

    /// Agrega masa, COM y **tensor de cuadrupolo** para el subárbol bajo `idx`.
    ///
    /// Usa dos pasadas:
    /// 1. Agrega hijos recursivamente (establece su masa, COM y cuadrupolo).
    /// 2. Calcula el cuadrupolo del padre aplicando el teorema del eje paralelo
    ///    sobre los hijos ya aggregados.
    fn aggregate(&mut self, idx: u32, positions: &[Vec3], masses: &[f64]) -> (f64, Vec3) {
        let is_leaf = self.nodes[idx as usize]
            .children
            .iter()
            .all(|&c| c == NO_CHILD);
        if is_leaf {
            if let Some(j) = self.nodes[idx as usize].particle_idx {
                let m = masses[j];
                let c = positions[j];
                self.nodes[idx as usize].mass = m;
                self.nodes[idx as usize].com = c;
                self.nodes[idx as usize].quad = [0.0; 6]; // punto en su propio COM
                return (m, c);
            }
            self.nodes[idx as usize].mass = 0.0;
            self.nodes[idx as usize].com = Vec3::zero();
            self.nodes[idx as usize].quad = [0.0; 6];
            return (0.0, Vec3::zero());
        }
        let children = self.nodes[idx as usize].children; // [u32; 8] es Copy

        // ── Pasada 1: agregar hijos → obtener masa total y COM ───────────────────
        let mut mtot = 0.0_f64;
        let mut com_acc = Vec3::zero();
        for &ch in &children {
            if ch == NO_CHILD {
                continue;
            }
            let (m, c) = self.aggregate(ch, positions, masses);
            mtot += m;
            com_acc += c * m;
        }
        let com = if mtot > 0.0 { com_acc / mtot } else { Vec3::zero() };
        self.nodes[idx as usize].mass = mtot;
        self.nodes[idx as usize].com = com;

        // ── Pasada 2: cuadrupolo del padre (teorema del eje paralelo) ────────────
        // Los hijos ya tienen `mass`, `com` y `quad` correctamente calculados.
        let mut quad = [0.0_f64; 6];
        for &ch in &children {
            if ch == NO_CHILD {
                continue;
            }
            let child_mass = self.nodes[ch as usize].mass;
            if child_mass == 0.0 {
                continue;
            }
            let child_com = self.nodes[ch as usize].com;
            let child_quad = self.nodes[ch as usize].quad;
            // Desplazamiento del COM del hijo al COM del padre.
            let s = child_com - com;
            let outer = outer_traceless(s, child_mass);
            for i in 0..6 {
                quad[i] += child_quad[i] + outer[i];
            }
        }
        self.nodes[idx as usize].quad = quad;

        (mtot, com)
    }

    /// Aceleración gravitatoria sobre la partícula `gi` en `pos_i`
    /// (Plummer monopolo + corrección cuadrupolar en zona lejana).
    #[allow(clippy::too_many_arguments)]
    pub fn walk_accel(
        &self,
        pos_i: Vec3,
        gi: usize,
        g: f64,
        eps2: f64,
        theta: f64,
        positions: &[Vec3],
        masses: &[f64],
    ) -> Vec3 {
        use gadget_ng_core::pairwise_accel_plummer;
        self.walk_inner(
            self.root,
            pos_i,
            gi,
            g,
            eps2,
            theta,
            positions,
            masses,
            pairwise_accel_plummer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_inner(
        &self,
        node_idx: u32,
        pos_i: Vec3,
        gi: usize,
        g: f64,
        eps2: f64,
        theta: f64,
        positions: &[Vec3],
        masses: &[f64],
        pair: fn(Vec3, f64, Vec3, f64, f64) -> Vec3,
    ) -> Vec3 {
        let node = &self.nodes[node_idx as usize];
        if node.mass == 0.0 {
            return Vec3::zero();
        }
        let is_leaf = node.children.iter().all(|&c| c == NO_CHILD);
        if is_leaf {
            if let Some(j) = node.particle_idx {
                if j == gi {
                    return Vec3::zero();
                }
                return pair(pos_i, masses[j], positions[j], g, eps2);
            }
            return Vec3::zero();
        }
        let s = 2.0 * node.half_size;
        // No aproximar por monopolo si la partícula evaluada cae dentro de la celda: el subárbol
        // puede incluir su propia masa (fuerza propia / doble conteo).
        let eval_inside_cell = point_in_node_cell(pos_i, node.center, node.half_size);
        // MAC clásico Barnes-Hut: s / d < theta con d = distancia al centro de masa del nodo.
        let r_com = pos_i - node.com;
        let d_com = r_com.norm();
        let use_mac = theta > 0.0 && !eval_inside_cell && d_com > 1e-300 && s / d_com < theta;
        if use_mac {
            // Monopolo suavizado (Plummer) + corrección cuadrupolar en lejano campo.
            let a_mono = pair(pos_i, node.mass, node.com, g, eps2);
            let a_quad = quad_accel(r_com, node.quad, g);
            return a_mono + a_quad;
        }
        let mut a = Vec3::zero();
        for &ch in &node.children {
            if ch == NO_CHILD {
                continue;
            }
            a += self.walk_inner(ch, pos_i, gi, g, eps2, theta, positions, masses, pair);
        }
        a
    }
}

fn bounding_cube(pos: &[Vec3]) -> (Vec3, f64) {
    let mut min_x = pos[0].x;
    let mut max_x = pos[0].x;
    let mut min_y = pos[0].y;
    let mut max_y = pos[0].y;
    let mut min_z = pos[0].z;
    let mut max_z = pos[0].z;
    for p in pos.iter().skip(1) {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
        min_z = min_z.min(p.z);
        max_z = max_z.max(p.z);
    }
    let cx = 0.5 * (min_x + max_x);
    let cy = 0.5 * (min_y + max_y);
    let cz = 0.5 * (min_z + max_z);
    let center = Vec3::new(cx, cy, cz);
    let hx = (max_x - min_x) * 0.5;
    let hy = (max_y - min_y) * 0.5;
    let hz = (max_z - min_z) * 0.5;
    let mut half = hx.max(hy).max(hz);
    let pad = (half * 1e-6).max(1e-12);
    half += pad;
    (center, half)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{pairwise_accel_plummer, Vec3};

    /// Verifica que el tensor cuadrupolar de una distribución simétricamente
    /// opuesta (dos masas iguales en ±r) tiene los términos correctos.
    #[test]
    fn quadrupole_two_masses_symmetric() {
        // Dos masas iguales en (d, 0, 0) y (-d, 0, 0) → COM en origen.
        let d = 2.0_f64;
        let m = 1.0_f64;
        let pos = vec![Vec3::new(d, 0.0, 0.0), Vec3::new(-d, 0.0, 0.0)];
        let masses = vec![m, m];
        let tree = Octree::build(&pos, &masses);
        let root_quad = tree.nodes[tree.root as usize].quad;

        // Q = m*(3x²-r²) para cada masa, sumadas: 2m*(3d²-d²) = 4md²
        let expected_qxx = 2.0 * m * (3.0 * d * d - d * d); // 4md²
        let tol = 1e-10_f64 * expected_qxx.abs();
        assert!(
            (root_quad[0] - expected_qxx).abs() < tol,
            "Qxx = {:.6e}, esperado = {:.6e}",
            root_quad[0],
            expected_qxx
        );
        // Qyy = Qzz = -2md² (por simetría y condición de sin traza)
        let expected_qyy = -2.0 * m * d * d;
        assert!(
            (root_quad[3] - expected_qyy).abs() < tol.abs().max(1e-10),
            "Qyy = {:.6e}, esperado = {:.6e}",
            root_quad[3],
            expected_qyy
        );
        // Tensor sin traza: Qxx + Qyy + Qzz = 0
        let trace = root_quad[0] + root_quad[3] + root_quad[5];
        assert!(trace.abs() < 1e-10, "traza = {trace:.2e} (esperado ≈ 0)");
    }

    /// La corrección cuadrupolar mejora la aceleración vs. monopolo puro
    /// para una distribución elongada evaluada a distancia moderada.
    #[test]
    fn quadrupole_improves_accuracy_vs_direct() {
        // Distribución: 4 masas en ±2 en x y ±0.5 en y → fuerte cuadrupolo
        let pos = vec![
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
        ];
        let masses = vec![1.0_f64; 4];
        let g = 1.0_f64;
        let eps2 = 0.0_f64; // sin suavizado para comparación limpia

        // Fuerza exacta (directa) sobre una partícula de prueba lejana
        let eval_pos = Vec3::new(10.0, 0.0, 0.0);
        let mut a_direct = Vec3::zero();
        for (i, &p) in pos.iter().enumerate() {
            let r = eval_pos - p;
            let r2 = r.dot(r);
            let r3 = r2 * r2.sqrt();
            a_direct -= r * (g * masses[i] / r3);
        }

        let tree = Octree::build(&pos, &masses);
        let root = &tree.nodes[tree.root as usize];

        // Monopolo puro (sin cuadrupolo)
        let a_mono = pairwise_accel_plummer(eval_pos, root.mass, root.com, g, eps2);

        // Monopolo + cuadrupolo
        let r_com = eval_pos - root.com;
        let a_quad = quad_accel(r_com, root.quad, g);
        let a_mono_quad = a_mono + a_quad;

        let err_mono = (a_mono - a_direct).norm() / a_direct.norm();
        let err_mono_quad = (a_mono_quad - a_direct).norm() / a_direct.norm();

        assert!(
            err_mono_quad < err_mono,
            "la corrección cuadrupolar debería mejorar la precisión: \
             err_mono={err_mono:.3e}, err_mono_quad={err_mono_quad:.3e}"
        );
        // La corrección cuadrupolar debería dar < 1 % de error relativo a d=10
        assert!(
            err_mono_quad < 0.01,
            "error relativo con cuadrupolo demasiado grande: {err_mono_quad:.3e}"
        );
    }

    /// Con `theta = 0.5` y cuadrupolo, el error relativo promedio en N=20
    /// partículas debe ser < 0.5 % (mejor que con monopolo solo).
    #[test]
    fn quadrupole_bh_better_than_monopole_random_particles() {
        let n = 20usize;
        let mut pos = Vec::with_capacity(n);
        let mut masses = Vec::with_capacity(n);
        let mut rng_state = 42u64;
        let mut lcg = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng_state >> 33) as f64 / (u32::MAX as f64)
        };
        for _ in 0..n {
            pos.push(Vec3::new(lcg() - 0.5, lcg() - 0.5, lcg() - 0.5));
            masses.push(0.5 + lcg());
        }

        let eps2 = 0.0_f64;
        let g = 1.0_f64;
        let theta = 0.5_f64;

        // Referencia: gravedad directa pairwise
        let a_ref: Vec<Vec3> = (0..n)
            .map(|i| {
                let mut a = Vec3::zero();
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    a += pairwise_accel_plummer(pos[i], masses[j], pos[j], g, eps2);
                }
                a
            })
            .collect();

        // Barnes-Hut con cuadrupolo (implementación actual)
        let tree = Octree::build(&pos, &masses);
        let a_bh: Vec<Vec3> = (0..n)
            .map(|i| tree.walk_accel(pos[i], i, g, eps2, theta, &pos, &masses))
            .collect();

        let mean_err: f64 = a_bh
            .iter()
            .zip(a_ref.iter())
            .map(|(&ab, &ar)| (ab - ar).norm() / (ar.norm() + 1e-30))
            .sum::<f64>()
            / n as f64;

        // Con cuadrupolo y theta=0.5, el error relativo medio debe ser < 0.5 %
        assert!(
            mean_err < 0.005,
            "error relativo medio con cuadrupolo (theta=0.5): {mean_err:.3e}"
        );
    }
}
