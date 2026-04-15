//! Octree en arena para Barnes-Hut (3D).
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
        }
    }
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
                return (m, c);
            }
            self.nodes[idx as usize].mass = 0.0;
            self.nodes[idx as usize].com = Vec3::zero();
            return (0.0, Vec3::zero());
        }
        let children = self.nodes[idx as usize].children;
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
        let com = if mtot > 0.0 {
            com_acc / mtot
        } else {
            Vec3::zero()
        };
        self.nodes[idx as usize].mass = mtot;
        self.nodes[idx as usize].com = com;
        (mtot, com)
    }

    /// Aceleración gravitatoria sobre la partícula `gi` en `pos_i` (Plummer + Barnes-Hut).
    #[allow(clippy::too_many_arguments)] // API explícita para el tree walk sin struct auxiliar en el MVP.
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
            return pair(pos_i, node.mass, node.com, g, eps2);
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
