//! LET-tree: octree espacial sobre `RemoteMultipoleNode` importados.
//!
//! Reduce el costo de `apply_let` de O(N_local × N_let) a O(N_local log N_let)
//! mediante un octree Barnes-Hut sobre los nodos LET recibidos por este rango.
//!
//! ## Algoritmo
//!
//! 1. **Build** (`LetTree::build`): construye un octree top-down sobre los COMs de
//!    los `RemoteMultipoleNode`s. Los nodos internos almacenan multipolos agregados
//!    (M2M para monopolo+cuadrupolo exacto; octupolo con traslación de término
//!    monopolar). Las hojas aplican cada RMN individualmente.
//!
//! 2. **Walk** (`LetTree::walk_accel`): para cada partícula local, recorre el
//!    árbol con el MAC geométrico `2·half_size / d < θ`. Si MAC pasa: aplica el
//!    multipolo agregado. Si no: desciende a hijos. Las hojas siempre aplican
//!    cada RMN individualmente con mono+quad+oct exacto.
//!
//! ## Precisión
//!
//! - Monopolo y cuadrupolo en nodos internos: M2M exacto via `outer_traceless`.
//! - Octupolo en nodos internos: suma más traslación del término monopolar
//!   (`m·TF3(s)`). Los términos cruzados quad×s se omiten (error O((s/d)⁴)
//!   relativo al monopolo, menor que la truncación del cuadrupolo).
//! - Hojas: aplicación exacta (mono+quad+oct del RMN original sin modificación).

use crate::octree::{
    oct_accel_softened, outer3_tf, outer_traceless, quad_accel_softened, RemoteMultipoleNode,
};
#[cfg(feature = "simd")]
use crate::rmn_soa::RmnSoa;
use gadget_ng_core::Vec3;

// ── Profiling de hojas (atómicos globales) ────────────────────────────────────

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static LT_PROF_ACTIVE: AtomicBool = AtomicBool::new(false);
static LT_LEAF_CALLS: AtomicU64 = AtomicU64::new(0);
static LT_LEAF_RMN_COUNT: AtomicU64 = AtomicU64::new(0);
/// Número de llamadas a `apply_leaf_soa_4xi` (Fase 16)
static LT_TILE_CALLS: AtomicU64 = AtomicU64::new(0);
/// Suma de `tile_size` en cada llamada tileada (partículas-i evaluadas en modo tileado)
static LT_TILE_I_COUNT: AtomicU64 = AtomicU64::new(0);

/// Reinicia los contadores de profiling de hojas (todos los threads).
pub fn let_tree_prof_begin() {
    LT_LEAF_CALLS.store(0, Ordering::Relaxed);
    LT_LEAF_RMN_COUNT.store(0, Ordering::Relaxed);
    LT_TILE_CALLS.store(0, Ordering::Relaxed);
    LT_TILE_I_COUNT.store(0, Ordering::Relaxed);
    LT_PROF_ACTIVE.store(true, Ordering::Release);
}

/// Lee y desactiva los contadores de profiling.
/// Devuelve `(leaf_calls, leaf_rmn_count)`.
pub fn let_tree_prof_end() -> (u64, u64) {
    LT_PROF_ACTIVE.store(false, Ordering::Release);
    let calls = LT_LEAF_CALLS.load(Ordering::Relaxed);
    let rmn = LT_LEAF_RMN_COUNT.load(Ordering::Relaxed);
    (calls, rmn)
}

/// Lee y devuelve los contadores de profiling tileados (Fase 16).
/// Devuelve `(tile_calls, tile_i_count)`.
pub fn let_tree_tile_prof_read() -> (u64, u64) {
    let calls = LT_TILE_CALLS.load(Ordering::Relaxed);
    let i_cnt = LT_TILE_I_COUNT.load(Ordering::Relaxed);
    (calls, i_cnt)
}

/// Valor centinela: sin hijo en esta dirección.
const NO_LET_CHILD: u32 = u32::MAX;

/// Número máximo de RMNs por hoja (por defecto).
pub const DEFAULT_LEAF_MAX: usize = 8;

/// Nodo del `LetTree`.
#[derive(Clone, Debug)]
pub struct LetNode {
    /// Centro espacial de la celda (para subdivisión en octantes).
    pub center: Vec3,
    /// Half-size espacial (MAC: `2·half_size / d < θ`).
    pub half_size: f64,
    /// Centro de masa agregado de todos los RMNs del subárbol.
    pub com: Vec3,
    /// Masa total del subárbol.
    pub mass: f64,
    /// Cuadrupolo agregado con traslación M2M exacta.
    pub quad: [f64; 6],
    /// Octupolo agregado (términos mono+oct de cada hijo; traslación M2M aproximada).
    pub oct: [f64; 7],
    /// Índices de hijos (NO_LET_CHILD si es hoja o si el octante está vacío).
    pub children: [u32; 8],
    /// Inicio en `LetTree::leaf_storage` (solo válido cuando `leaf_count > 0`).
    pub leaf_start: u32,
    /// Número de RMNs almacenados en la hoja (`> 0` ↔ es hoja).
    pub leaf_count: u32,
}

// ── LetTree ────────────────────────────────────────────────────────────────────

/// Octree de Barnes-Hut sobre nodos `RemoteMultipoleNode` importados.
///
/// Se construye una vez por evaluación de fuerza con los nodos LET recibidos.
/// Es de solo lectura durante `walk_accel`, por lo que es seguro de usar con Rayon.
pub struct LetTree {
    nodes: Vec<LetNode>,
    root: u32,
    /// Almacenamiento AoS de RMNs en hojas (siempre presente; usado durante build
    /// y como fuente para `leaf_soa`; el walk lo lee directamente sin simd).
    #[allow(dead_code)]
    leaf_storage: Vec<RemoteMultipoleNode>,
    /// Almacenamiento SoA de RMNs en hojas — activo con feature `simd`.
    /// Permite kernel fusionado con auto-vectorización AVX2 en `apply_leaf_soa`.
    #[cfg(feature = "simd")]
    leaf_soa: RmnSoa,
}

unsafe impl Sync for LetTree {}
unsafe impl Send for LetTree {}

impl LetTree {
    /// Construye con `DEFAULT_LEAF_MAX` RMNs por hoja.
    pub fn build(rmns: &[RemoteMultipoleNode]) -> Self {
        Self::build_with_leaf_max(rmns, DEFAULT_LEAF_MAX)
    }

    /// Construye con un `leaf_max` personalizado (mínimo 1).
    pub fn build_with_leaf_max(rmns: &[RemoteMultipoleNode], leaf_max: usize) -> Self {
        if rmns.is_empty() {
            return Self {
                nodes: Vec::new(),
                root: NO_LET_CHILD,
                leaf_storage: Vec::new(),
                #[cfg(feature = "simd")]
                leaf_soa: RmnSoa::default(),
            };
        }

        let (center, half_size) = bounding_cube_of_coms(rmns);
        let indices: Vec<usize> = (0..rmns.len()).collect();

        let mut ctx = BuildCtx {
            rmns,
            nodes: Vec::with_capacity(rmns.len() * 2),
            leaf_storage: Vec::with_capacity(rmns.len()),
            leaf_max: leaf_max.max(1),
        };
        let root = ctx.build_node(&indices, center, half_size, 0);

        // Con feature "simd" construimos el SoA de leaf_storage para el kernel
        // fusionado con auto-vectorización AVX2.
        #[cfg(feature = "simd")]
        let leaf_soa = RmnSoa::from_slice(&ctx.leaf_storage);

        Self {
            nodes: ctx.nodes,
            root,
            leaf_storage: ctx.leaf_storage,
            #[cfg(feature = "simd")]
            leaf_soa,
        }
    }

    /// Número total de nodos (internos + hojas).
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// `true` si el árbol está vacío (sin nodos LET importados).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Aceleración gravitacional sobre `pos_i` procedente de todos los nodos LET.
    ///
    /// Usa MAC geométrico `2·half_size / d < theta` en cada nodo interno.
    pub fn walk_accel(&self, pos_i: Vec3, g: f64, eps2: f64, theta: f64) -> Vec3 {
        if self.root == NO_LET_CHILD {
            return Vec3::zero();
        }
        self.walk_inner(self.root, pos_i, g, eps2, theta)
    }

    fn walk_inner(&self, node_idx: u32, pos_i: Vec3, g: f64, eps2: f64, theta: f64) -> Vec3 {
        let node = &self.nodes[node_idx as usize];
        if node.mass == 0.0 {
            return Vec3::zero();
        }

        // Nodo hoja: aplicar cada RMN individualmente (exacto).
        if node.leaf_count > 0 {
            #[cfg(feature = "simd")]
            return self.apply_leaf_soa(node, pos_i, g, eps2);
            #[cfg(not(feature = "simd"))]
            return self.apply_leaf(node, pos_i, g, eps2);
        }

        // Criterio MAC geométrico.
        let r = pos_i - node.com;
        let d = r.norm();
        if d > 1e-300 && 2.0 * node.half_size / d < theta {
            // Usar multipolo agregado (mono + quad + oct).
            let a_mono = accel_mono_softened(pos_i, node.mass, node.com, g, eps2);
            let a_quad = quad_accel_softened(r, node.quad, g, eps2);
            let a_oct = oct_accel_softened(r, node.oct, g, eps2);
            return a_mono + a_quad + a_oct;
        }

        // MAC falla: abrir hijos.
        let mut acc = Vec3::zero();
        for &ch in &node.children {
            if ch != NO_LET_CHILD {
                acc += self.walk_inner(ch, pos_i, g, eps2, theta);
            }
        }
        acc
    }

    /// Path AoS: utilizado sin feature `simd` y como referencia de validación.
    #[cfg_attr(feature = "simd", allow(dead_code))]
    #[inline]
    fn apply_leaf(&self, node: &LetNode, pos_i: Vec3, g: f64, eps2: f64) -> Vec3 {
        if LT_PROF_ACTIVE.load(Ordering::Relaxed) {
            LT_LEAF_CALLS.fetch_add(1, Ordering::Relaxed);
            LT_LEAF_RMN_COUNT.fetch_add(node.leaf_count as u64, Ordering::Relaxed);
        }

        let mut acc = Vec3::zero();
        let start = node.leaf_start as usize;
        let end = start + node.leaf_count as usize;
        for rmn in &self.leaf_storage[start..end] {
            if rmn.mass == 0.0 {
                continue;
            }
            let r = pos_i - rmn.com;
            acc += accel_mono_softened(pos_i, rmn.mass, rmn.com, g, eps2);
            acc += quad_accel_softened(r, rmn.quad, g, eps2);
            acc += oct_accel_softened(r, rmn.oct, g, eps2);
        }
        acc
    }

    /// Path SoA con kernel fusionado (feature `simd`): mono+quad+oct en un solo
    /// recorrido con una sola llamada a `sqrt` por nodo j.
    /// Auto-vectorizable con AVX2+FMA en el loop monopolar.
    #[cfg(feature = "simd")]
    #[inline]
    fn apply_leaf_soa(&self, node: &LetNode, pos_i: Vec3, g: f64, eps2: f64) -> Vec3 {
        if LT_PROF_ACTIVE.load(Ordering::Relaxed) {
            LT_LEAF_CALLS.fetch_add(1, Ordering::Relaxed);
            LT_LEAF_RMN_COUNT.fetch_add(node.leaf_count as u64, Ordering::Relaxed);
        }

        let start = node.leaf_start as usize;
        let len = node.leaf_count as usize;
        self.leaf_soa.accel_range(pos_i, start, len, g, eps2)
    }

    // ── Fase 16: walk tileado 4×N_i ──────────────────────────────────────────

    /// **Fase 16 — Walk tileado**: calcula la aceleración gravitacional sobre
    /// `tile_size` (1-4) partículas simultáneamente contra todos los nodos LET.
    ///
    /// Usa **MAC conservativo**: un nodo se abre si CUALQUIERA de las `tile_size`
    /// partículas válidas no satisface el criterio. El SFC ordering garantiza que
    /// `pos[0..tile_size]` son espacialmente próximas, minimizando el overhead.
    ///
    /// Solo disponible con feature `simd` (requiere `RmnSoa::accel_range_4xi`).
    #[cfg(feature = "simd")]
    pub fn walk_accel_4xi(
        &self,
        pos: [Vec3; 4],
        tile_size: usize,
        g: f64,
        eps2: f64,
        theta: f64,
    ) -> [Vec3; 4] {
        if self.root == NO_LET_CHILD {
            return [Vec3::zero(); 4];
        }
        let mut acc = [Vec3::zero(); 4];
        self.walk_inner_4xi(self.root, &pos, tile_size, g, eps2, theta, &mut acc);
        acc
    }

    /// Recursión interna del walk tileado para el nodo `node_idx`.
    #[cfg(feature = "simd")]
    fn walk_inner_4xi(
        &self,
        node_idx: u32,
        pos: &[Vec3; 4],
        tile_size: usize,
        g: f64,
        eps2: f64,
        theta: f64,
        acc: &mut [Vec3; 4],
    ) {
        let node = &self.nodes[node_idx as usize];
        if node.mass == 0.0 {
            return;
        }

        // Hoja: aplicar todos los RMNs a las tile_size partículas con kernel 4xi.
        if node.leaf_count > 0 {
            self.apply_leaf_soa_4xi(node, pos, tile_size, g, eps2, acc);
            return;
        }

        // MAC conservativo: abrir el nodo si CUALQUIER partícula válida falla.
        let all_pass = pos[..tile_size].iter().all(|p| {
            let d = (*p - node.com).norm();
            d > 1e-300 && 2.0 * node.half_size / d < theta
        });

        if all_pass {
            // Aplicar multipolo agregado a cada partícula válida (escalar — raro).
            for k in 0..tile_size {
                let r = pos[k] - node.com;
                acc[k] += accel_mono_softened(pos[k], node.mass, node.com, g, eps2);
                acc[k] += quad_accel_softened(r, node.quad, g, eps2);
                acc[k] += oct_accel_softened(r, node.oct, g, eps2);
            }
        } else {
            // MAC falla: descender a hijos (todos bajo el mismo tile).
            for &ch in &node.children {
                if ch != NO_LET_CHILD {
                    self.walk_inner_4xi(ch, pos, tile_size, g, eps2, theta, acc);
                }
            }
        }
    }

    /// Aplica el leaf SoA a `tile_size` partículas simultáneas via `accel_range_4xi`.
    ///
    /// Registra profiling de tiles (contadores atómicos).
    #[cfg(feature = "simd")]
    #[inline]
    fn apply_leaf_soa_4xi(
        &self,
        node: &LetNode,
        pos: &[Vec3; 4],
        tile_size: usize,
        g: f64,
        eps2: f64,
        acc: &mut [Vec3; 4],
    ) {
        if LT_PROF_ACTIVE.load(Ordering::Relaxed) {
            LT_TILE_CALLS.fetch_add(1, Ordering::Relaxed);
            LT_TILE_I_COUNT.fetch_add(tile_size as u64, Ordering::Relaxed);
        }

        let start = node.leaf_start as usize;
        let len = node.leaf_count as usize;
        let result = self
            .leaf_soa
            .accel_range_4xi(pos, start, len, g, eps2, tile_size);
        for k in 0..tile_size {
            acc[k] += result[k];
        }
    }
}

// ── Build (top-down) ──────────────────────────────────────────────────────────

struct BuildCtx<'a> {
    rmns: &'a [RemoteMultipoleNode],
    nodes: Vec<LetNode>,
    leaf_storage: Vec<RemoteMultipoleNode>,
    leaf_max: usize,
}

impl<'a> BuildCtx<'a> {
    /// Construye el subárbol que cubre `indices` y devuelve el índice del nodo raíz creado.
    ///
    /// `depth` limita la recursión a 32 niveles (guarda contra puntos degenerados colocalizados).
    fn build_node(&mut self, indices: &[usize], center: Vec3, half_size: f64, depth: u32) -> u32 {
        let (com, mass, quad, oct) = aggregate_multipoles(self.rmns, indices);

        let node_idx = self.nodes.len() as u32;

        // ── Condición de hoja ────────────────────────────────────────────────
        // - ≤ leaf_max nodos, o
        // - profundidad máxima (degenerado: todos los COMs en el mismo punto), o
        // - celda microscópica (evita subdivisión infinita en f64).
        if indices.len() <= self.leaf_max || depth >= 32 || half_size < 1e-12 {
            let leaf_start = self.leaf_storage.len() as u32;
            for &i in indices {
                self.leaf_storage.push(self.rmns[i]);
            }
            self.nodes.push(LetNode {
                center,
                half_size,
                com,
                mass,
                quad,
                oct,
                children: [NO_LET_CHILD; 8],
                leaf_start,
                leaf_count: indices.len() as u32,
            });
            return node_idx;
        }

        // ── Nodo interno ─────────────────────────────────────────────────────
        // Reservar el slot (se completarán los hijos después de la recursión).
        self.nodes.push(LetNode {
            center,
            half_size,
            com,
            mass,
            quad,
            oct,
            children: [NO_LET_CHILD; 8],
            leaf_start: 0,
            leaf_count: 0,
        });

        // Partición en 8 octantes.
        let mut groups: [Vec<usize>; 8] = Default::default();
        for &i in indices {
            let oct = octant_of(self.rmns[i].com, center);
            groups[oct].push(i);
        }

        // Detección de caso degenerado: todos caen en el mismo octante → hoja.
        let non_empty = groups.iter().filter(|g| !g.is_empty()).count();
        if non_empty == 1 {
            // Forzar hoja: infinita subdivisión evitada.
            let leaf_start = self.leaf_storage.len() as u32;
            for &i in indices {
                self.leaf_storage.push(self.rmns[i]);
            }
            let node = &mut self.nodes[node_idx as usize];
            node.leaf_start = leaf_start;
            node.leaf_count = indices.len() as u32;
            return node_idx;
        }

        let child_half = half_size * 0.5;
        #[allow(clippy::needless_range_loop)]
        for oct in 0..8usize {
            if groups[oct].is_empty() {
                continue;
            }
            let child_center = child_center_of(center, child_half, oct);
            let child_idx = self.build_node(&groups[oct], child_center, child_half, depth + 1);
            self.nodes[node_idx as usize].children[oct] = child_idx;
        }

        node_idx
    }
}

// ── Helpers geométricos ───────────────────────────────────────────────────────

/// Cubo delimitador de los COMs: devuelve (centro, half_size).
fn bounding_cube_of_coms(rmns: &[RemoteMultipoleNode]) -> (Vec3, f64) {
    let mut xlo = f64::INFINITY;
    let mut xhi = f64::NEG_INFINITY;
    let mut ylo = f64::INFINITY;
    let mut yhi = f64::NEG_INFINITY;
    let mut zlo = f64::INFINITY;
    let mut zhi = f64::NEG_INFINITY;
    for n in rmns {
        xlo = xlo.min(n.com.x);
        xhi = xhi.max(n.com.x);
        ylo = ylo.min(n.com.y);
        yhi = yhi.max(n.com.y);
        zlo = zlo.min(n.com.z);
        zhi = zhi.max(n.com.z);
    }
    let cx = (xlo + xhi) * 0.5;
    let cy = (ylo + yhi) * 0.5;
    let cz = (zlo + zhi) * 0.5;
    // Añadir pequeño margen para que ningún COM caiga exactamente en el borde.
    let half = ((xhi - xlo).max(yhi - ylo).max(zhi - zlo)) * 0.5 + 1e-10;
    (Vec3::new(cx, cy, cz), half)
}

/// Octante (0..8) de `pos` respecto a `center`.
#[inline]
fn octant_of(pos: Vec3, center: Vec3) -> usize {
    let mut o = 0usize;
    if pos.x >= center.x {
        o |= 1;
    }
    if pos.y >= center.y {
        o |= 2;
    }
    if pos.z >= center.z {
        o |= 4;
    }
    o
}

/// Centro del hijo en el octante `oct` (mismo convenio que `octree.rs`).
#[inline]
fn child_center_of(center: Vec3, child_half: f64, oct: usize) -> Vec3 {
    Vec3::new(
        center.x
            + if oct & 1 != 0 {
                child_half
            } else {
                -child_half
            },
        center.y
            + if oct & 2 != 0 {
                child_half
            } else {
                -child_half
            },
        center.z
            + if oct & 4 != 0 {
                child_half
            } else {
                -child_half
            },
    )
}

// ── Agregación de multipolos (M2M) ────────────────────────────────────────────

/// Calcula los multipolos agregados de un subconjunto de RMNs.
///
/// ### Exacto
/// - Masa: suma directa.
/// - COM: media ponderada por masa.
/// - Cuadrupolo: M2M via `outer_traceless(s, m)` donde `s = com_j - com_agg`.
///
/// ### Aproximado
/// - Octupolo: suma más traslación del término monopolar `m·TF3(s)`.
///   Se omiten los términos cruzados cuadrupolo×desplazamiento; introduce un
///   error O((s/d)⁴) relativo al monopolo.
fn aggregate_multipoles(
    rmns: &[RemoteMultipoleNode],
    indices: &[usize],
) -> (Vec3, f64, [f64; 6], [f64; 7]) {
    let mass: f64 = indices.iter().map(|&i| rmns[i].mass).sum();

    let com = if mass > 0.0 {
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mut cz = 0.0f64;
        for &i in indices {
            let n = &rmns[i];
            cx += n.mass * n.com.x;
            cy += n.mass * n.com.y;
            cz += n.mass * n.com.z;
        }
        Vec3::new(cx / mass, cy / mass, cz / mass)
    } else {
        Vec3::zero()
    };

    // Cuadrupolo M2M exacto:
    // Q_agg[ab] = Σ_j [ Q_j[ab] + m_j · outer_traceless(s_j)[ab] ]
    // donde s_j = com_j − com_agg.
    let mut quad = [0.0f64; 6];
    for &i in indices {
        let n = &rmns[i];
        for (k, q) in quad.iter_mut().enumerate() {
            *q += n.quad[k];
        }
        if n.mass != 0.0 {
            let s = n.com - com;
            let dq = outer_traceless(s, n.mass);
            for (k, q) in quad.iter_mut().enumerate() {
                *q += dq[k];
            }
        }
    }

    // Octupolo M2M aproximado:
    // O_agg = Σ_j [ O_j + m_j · TF3(s_j) ]
    // (falta el término cruzado Q_j × s_j, error O(s/d)^4).
    let mut oct = [0.0f64; 7];
    for &i in indices {
        let n = &rmns[i];
        for (k, o) in oct.iter_mut().enumerate() {
            *o += n.oct[k];
        }
        if n.mass != 0.0 {
            let s = n.com - com;
            let do_ = outer3_tf(s, n.mass);
            for (k, o) in oct.iter_mut().enumerate() {
                *o += do_[k];
            }
        }
    }

    (com, mass, quad, oct)
}

// ── Fuerza monopolar ──────────────────────────────────────────────────────────

/// Aceleración monopolar Plummer suavizada sobre `pos_i` por una masa `m` en `com`.
#[inline]
fn accel_mono_softened(pos_i: Vec3, m: f64, com: Vec3, g: f64, eps2: f64) -> Vec3 {
    let r = pos_i - com;
    let r2 = r.dot(r) + eps2;
    if r2 < 1e-300 {
        return Vec3::zero();
    }
    let r_inv = 1.0 / r2.sqrt();
    let r3_inv = r_inv * r_inv * r_inv;
    Vec3::new(
        -g * m * r3_inv * r.x,
        -g * m * r3_inv * r.y,
        -g * m * r3_inv * r.z,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::{accel_from_let, RemoteMultipoleNode};
    use gadget_ng_core::Vec3;

    fn dummy_rmn(com: Vec3, mass: f64) -> RemoteMultipoleNode {
        RemoteMultipoleNode {
            com,
            mass,
            quad: [0.0; 6],
            oct: [0.0; 7],
            half_size: 0.1,
        }
    }

    #[test]
    fn let_tree_empty_build_and_walk() {
        let tree = LetTree::build(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.node_count(), 0);
        let a = tree.walk_accel(Vec3::new(1.0, 0.0, 0.0), 1.0, 0.01, 0.5);
        assert_eq!(a, Vec3::zero());
    }

    #[test]
    fn let_tree_single_node() {
        let rmn = dummy_rmn(Vec3::new(0.0, 0.0, 0.0), 1.0);
        let rmns = vec![rmn];
        let tree = LetTree::build(&rmns);
        assert!(!tree.is_empty());
        let pos = Vec3::new(5.0, 0.0, 0.0);
        let a_tree = tree.walk_accel(pos, 1.0, 0.01, 0.5);
        let a_flat = accel_from_let(pos, &rmns, 1.0, 0.01);
        let diff = (a_tree - a_flat).norm();
        assert!(
            diff < 1e-12,
            "Single RMN: tree={a_tree:?} flat={a_flat:?} diff={diff}"
        );
    }
}
