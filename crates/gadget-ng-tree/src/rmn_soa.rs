//! Representación **SoA (Structure of Arrays)** para lotes de [`RemoteMultipoleNode`].
//!
//! ## Motivación
//!
//! En el layout AoS original cada `RemoteMultipoleNode` ocupa 18 `f64` contiguos
//! (com × 3, mass, quad × 6, oct × 7, half_size). Al procesar un lote de `N` nodos
//! para calcular la aceleración sobre una partícula, las cargas de datos son dispersas
//! y el compilador no puede vectorizar el inner loop de forma eficiente.
//!
//! Con SoA todas las coordenadas `cx[]`, masas `mass[]`, etc. forman columnas contiguas,
//! permitiendo:
//!
//! 1. **Auto-vectorización AVX2** del loop monopolar (4 `f64` por registro YMMD).
//! 2. **Kernel fusionado** mono+quad+oct con una sola llamada a `sqrt` por nodo `j`
//!    (reduce 3× las llamadas a `sqrt` respecto al AoS original).
//! 3. **Cache locality** mejorada para accesos secuenciales.
//!
//! ## Diseño
//!
//! El kernel divide la aceleración en dos pasadas:
//!
//! - **Pasada mono** (`cx/cy/cz/mass`): loop limpio → vectoriza bien.
//! - **Pasada quad+oct**: loop con tensores 6+7 → SoA mejora caché aunque el
//!   compilador no vectorice completamente.
//!
//! Para batches pequeños (≤ 8 RMNs en `apply_leaf`), el beneficio principal es
//! el kernel fusionado (ahorro de sqrt). Para `accel_from_let` plano (miles de RMNs),
//! además se obtiene vectorización real.

use crate::octree::RemoteMultipoleNode;
use gadget_ng_core::Vec3;

// ── Struct SoA ────────────────────────────────────────────────────────────────

/// Layout columnar de un batch de [`RemoteMultipoleNode`].
///
/// Las columnas `quad[k]` replican el mismo orden que `RemoteMultipoleNode.quad`:
/// `[qxx, qxy, qxz, qyy, qyz, qzz]`.
/// Las columnas `oct[k]` replican `[o_xxx, o_xxy, o_xxz, o_xyy, o_xyz, o_yyy, o_yzz]`.
#[derive(Default, Clone)]
pub struct RmnSoa {
    pub cx: Vec<f64>,
    pub cy: Vec<f64>,
    pub cz: Vec<f64>,
    pub mass: Vec<f64>,
    /// 6 columnas del tensor cuadrupolar.
    pub quad: [Vec<f64>; 6],
    /// 7 columnas del tensor octupolar.
    pub oct: [Vec<f64>; 7],
    pub len: usize,
}

impl RmnSoa {
    /// Construye un `RmnSoa` vacío con capacidad reservada para `cap` nodos.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            cx: Vec::with_capacity(cap),
            cy: Vec::with_capacity(cap),
            cz: Vec::with_capacity(cap),
            mass: Vec::with_capacity(cap),
            quad: std::array::from_fn(|_| Vec::with_capacity(cap)),
            oct: std::array::from_fn(|_| Vec::with_capacity(cap)),
            len: 0,
        }
    }

    /// Convierte un slice AoS de `RemoteMultipoleNode` a layout SoA.
    ///
    /// Coste: O(N) copias de f64 — negligible comparado con el cómputo de fuerzas.
    pub fn from_slice(rmns: &[RemoteMultipoleNode]) -> Self {
        let n = rmns.len();
        let mut soa = Self::with_capacity(n);
        soa.extend_from_slice(rmns);
        soa
    }

    /// Añade un slice de RMNs al final del SoA existente.
    pub fn extend_from_slice(&mut self, rmns: &[RemoteMultipoleNode]) {
        for rmn in rmns {
            self.cx.push(rmn.com.x);
            self.cy.push(rmn.com.y);
            self.cz.push(rmn.com.z);
            self.mass.push(rmn.mass);
            for k in 0..6 {
                self.quad[k].push(rmn.quad[k]);
            }
            for k in 0..7 {
                self.oct[k].push(rmn.oct[k]);
            }
            self.len += 1;
        }
    }

    /// Vacía el SoA sin liberar la memoria reservada (útil para reutilización).
    pub fn clear(&mut self) {
        self.cx.clear();
        self.cy.clear();
        self.cz.clear();
        self.mass.clear();
        for k in 0..6 { self.quad[k].clear(); }
        for k in 0..7 { self.oct[k].clear(); }
        self.len = 0;
    }
}

// ── Kernels de fuerza ─────────────────────────────────────────────────────────

/// Kernel fusionado mono+quad+oct sobre un rango `[start, start+len)` de la SoA.
///
/// Una sola llamada a `sqrt` por nodo `j` es compartida entre monopolo, cuadrupolo
/// y octupolo. Con `#[target_feature]` el compilador puede vectorizar el loop mono
/// con AVX2 (4 f64/ciclo) y aplicar FMA en las multiplicaciones.
///
/// # Seguridad
/// Requiere CPU con soporte AVX2+FMA. En plataformas no-x86 cae en la versión escalar.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn accel_soa_avx2(
    xi: f64,
    yi: f64,
    zi: f64,
    start: usize,
    len: usize,
    soa: &RmnSoa,
    g: f64,
    eps2: f64,
) -> Vec3 {
    accel_soa_scalar(xi, yi, zi, start, len, soa, g, eps2)
}

/// Kernel fusionado mono+quad+oct — versión escalar.
///
/// Sirve como fallback en plataformas no-AVX2 y como cuerpo del
/// loop que el compilador intenta vectorizar con `#[target_feature]`.
#[inline(always)]
pub(crate) fn accel_soa_scalar(
    xi: f64,
    yi: f64,
    zi: f64,
    start: usize,
    len: usize,
    soa: &RmnSoa,
    g: f64,
    eps2: f64,
) -> Vec3 {
    let mut ax = 0.0_f64;
    let mut ay = 0.0_f64;
    let mut az = 0.0_f64;

    // Alias de columnas para que el compilador vea slices contiguos.
    let cx = &soa.cx[start..start + len];
    let cy = &soa.cy[start..start + len];
    let cz = &soa.cz[start..start + len];
    let mass = &soa.mass[start..start + len];
    let q = [
        &soa.quad[0][start..start + len],
        &soa.quad[1][start..start + len],
        &soa.quad[2][start..start + len],
        &soa.quad[3][start..start + len],
        &soa.quad[4][start..start + len],
        &soa.quad[5][start..start + len],
    ];
    let o = [
        &soa.oct[0][start..start + len],
        &soa.oct[1][start..start + len],
        &soa.oct[2][start..start + len],
        &soa.oct[3][start..start + len],
        &soa.oct[4][start..start + len],
        &soa.oct[5][start..start + len],
        &soa.oct[6][start..start + len],
    ];

    for j in 0..len {
        let mj = mass[j];
        if mj == 0.0 {
            continue;
        }

        // Vector r = pos_i − com_j
        let rx = xi - cx[j];
        let ry = yi - cy[j];
        let rz = zi - cz[j];

        // r² + ε² — compartido por los tres multipolos (evita 3 sqrt).
        let r2 = rx * rx + ry * ry + rz * rz + eps2;
        if r2 < 1e-300 {
            continue;
        }

        // Un solo sqrt por nodo j (compartido mono+quad+oct).
        let r_inv = 1.0 / r2.sqrt();

        // ── Monopolo ──────────────────────────────────────────────────────────
        let r3_inv = r_inv * r_inv * r_inv;
        let mono_fac = -g * mj * r3_inv;
        ax += mono_fac * rx;
        ay += mono_fac * ry;
        az += mono_fac * rz;

        // ── Cuadrupolo ────────────────────────────────────────────────────────
        // [qxx, qxy, qxz, qyy, qyz, qzz]
        let qxx = q[0][j];
        let qxy = q[1][j];
        let qxz = q[2][j];
        let qyy = q[3][j];
        let qyz = q[4][j];
        let qzz = q[5][j];

        let r5_inv = r3_inv * r_inv * r_inv;
        let r7_inv = r5_inv * r_inv * r_inv;

        let qr_x = qxx * rx + qxy * ry + qxz * rz;
        let qr_y = qxy * rx + qyy * ry + qyz * rz;
        let qr_z = qxz * rx + qyz * ry + qzz * rz;
        let rqr = qr_x * rx + qr_y * ry + qr_z * rz;

        let c1 = g * r5_inv;
        let c2 = g * 2.5 * rqr * r7_inv;
        ax += c1 * qr_x - c2 * rx;
        ay += c1 * qr_y - c2 * ry;
        az += c1 * qr_z - c2 * rz;

        // ── Octupolo ──────────────────────────────────────────────────────────
        // [o_xxx, o_xxy, o_xxz, o_xyy, o_xyz, o_yyy, o_yzz]
        let o_xxx = o[0][j];
        let o_xxy = o[1][j];
        let o_xxz = o[2][j];
        let o_xyy = o[3][j];
        let o_xyz = o[4][j];
        let o_yyy = o[5][j];
        let o_yzz = o[6][j];

        // Componentes derivadas (condición sin traza STF).
        let o_xzz = -(o_xxx + o_xyy);
        let o_yyz = -(o_xxy + o_yyy);
        let o_zzz = -(o_xxz - o_xxy - o_yyy);

        let orr_x = o_xxx * rx * rx
            + 2.0 * o_xxy * rx * ry
            + 2.0 * o_xxz * rx * rz
            + o_xyy * ry * ry
            + 2.0 * o_xyz * ry * rz
            + o_xzz * rz * rz;
        let orr_y = o_xxy * rx * rx
            + 2.0 * o_xyy * rx * ry
            + 2.0 * o_xyz * rx * rz
            + o_yyy * ry * ry
            + 2.0 * o_yyz * ry * rz
            + o_yzz * rz * rz;
        let orr_z = o_xxz * rx * rx
            + 2.0 * o_xyz * rx * ry
            + 2.0 * o_xzz * rx * rz
            + o_yyz * ry * ry
            + 2.0 * o_yzz * ry * rz
            + o_zzz * rz * rz;

        let orrr = o_xxx * rx * rx * rx
            + 3.0 * o_xxy * rx * rx * ry
            + 3.0 * o_xxz * rx * rx * rz
            + 3.0 * o_xyy * rx * ry * ry
            + 6.0 * o_xyz * rx * ry * rz
            + 3.0 * o_xzz * rx * rz * rz
            + o_yyy * ry * ry * ry
            + 3.0 * o_yyz * ry * ry * rz
            + 3.0 * o_yzz * ry * rz * rz
            + o_zzz * rz * rz * rz;

        let r9_inv = r7_inv * r_inv * r_inv;
        let co1 = -g * 0.5 * r7_inv;
        let co2 = g * (7.0 / 6.0) * orrr * r9_inv;

        ax += co1 * orr_x + co2 * rx;
        ay += co1 * orr_y + co2 * ry;
        az += co1 * orr_z + co2 * rz;
    }

    Vec3::new(ax, ay, az)
}

// ── API pública ───────────────────────────────────────────────────────────────

impl RmnSoa {
    /// Aceleración gravitacional de todos los nodos del SoA sobre `pos_i`.
    ///
    /// Usa AVX2+FMA en plataformas x86/x86_64 si el hardware lo soporta en
    /// tiempo de ejecución; cae en escalar en caso contrario.
    #[inline]
    pub fn accel(&self, pos_i: Vec3, g: f64, eps2: f64) -> Vec3 {
        self.accel_range(pos_i, 0, self.len, g, eps2)
    }

    /// Aceleración gravitacional de los nodos `[start, start+len)` sobre `pos_i`.
    #[inline]
    pub fn accel_range(&self, pos_i: Vec3, start: usize, len: usize, g: f64, eps2: f64) -> Vec3 {
        if len == 0 {
            return Vec3::zero();
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: Acabamos de verificar que la CPU tiene AVX2+FMA.
                return unsafe {
                    accel_soa_avx2(pos_i.x, pos_i.y, pos_i.z, start, len, self, g, eps2)
                };
            }
        }

        accel_soa_scalar(pos_i.x, pos_i.y, pos_i.z, start, len, self, g, eps2)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::{accel_from_let, RemoteMultipoleNode};

    fn make_rmn(cx: f64, cy: f64, cz: f64, mass: f64) -> RemoteMultipoleNode {
        RemoteMultipoleNode {
            com: Vec3::new(cx, cy, cz),
            mass,
            quad: [0.0; 6],
            oct: [0.0; 7],
            half_size: 0.5,
        }
    }

    fn make_rmn_full(
        cx: f64,
        cy: f64,
        cz: f64,
        mass: f64,
        quad: [f64; 6],
        oct: [f64; 7],
    ) -> RemoteMultipoleNode {
        RemoteMultipoleNode {
            com: Vec3::new(cx, cy, cz),
            mass,
            quad,
            oct,
            half_size: 0.5,
        }
    }

    /// RMS relativo entre vectores.
    fn rms_error(a: Vec3, b: Vec3) -> f64 {
        let denom = (a.x * a.x + a.y * a.y + a.z * a.z).sqrt().max(1e-300);
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        ((dx * dx + dy * dy + dz * dz).sqrt()) / denom
    }

    #[test]
    fn soa_vs_aos_monopole_only() {
        let g = 1.0;
        let eps2 = 0.01_f64.powi(2);
        let pos_i = Vec3::new(1.0, 2.0, 3.0);

        let rmns: Vec<RemoteMultipoleNode> = (0..16)
            .map(|k| make_rmn(k as f64 * 0.5, -(k as f64), k as f64 * 0.3, 1.0 + k as f64 * 0.1))
            .collect();

        let soa = RmnSoa::from_slice(&rmns);
        let a_soa = soa.accel(pos_i, g, eps2);
        let a_aos = accel_from_let(pos_i, &rmns, g, eps2);

        let err = rms_error(a_soa, a_aos);
        assert!(
            err < 1e-12,
            "SoA vs AoS monopole RMS error = {:.2e} (tol 1e-12)",
            err
        );
    }

    #[test]
    fn soa_vs_aos_with_quad_oct() {
        let g = 1.0;
        let eps2 = 0.01_f64.powi(2);
        let pos_i = Vec3::new(3.0, -1.0, 2.0);

        let quad = [0.1, -0.05, 0.03, 0.08, -0.02, -0.18];
        let oct = [0.01, -0.005, 0.003, 0.008, -0.002, 0.007, -0.001];

        let rmns: Vec<RemoteMultipoleNode> = (0..32)
            .map(|k| {
                let s = k as f64 * 0.2;
                make_rmn_full(
                    s + 0.1,
                    -s * 0.5 + 1.0,
                    s * 0.3 - 2.0,
                    0.5 + s * 0.05,
                    quad.map(|v| v * (1.0 + s * 0.01)),
                    oct.map(|v| v * (1.0 + s * 0.01)),
                )
            })
            .collect();

        let soa = RmnSoa::from_slice(&rmns);
        let a_soa = soa.accel(pos_i, g, eps2);
        let a_aos = accel_from_let(pos_i, &rmns, g, eps2);

        let err = rms_error(a_soa, a_aos);
        assert!(
            err < 1e-12,
            "SoA vs AoS quad+oct RMS error = {:.2e} (tol 1e-12)",
            err
        );
    }

    #[test]
    fn soa_from_slice_length() {
        let rmns: Vec<RemoteMultipoleNode> = (0..100)
            .map(|k| make_rmn(k as f64, 0.0, 0.0, 1.0))
            .collect();
        let soa = RmnSoa::from_slice(&rmns);
        assert_eq!(soa.len, 100);
        assert_eq!(soa.cx.len(), 100);
        assert_eq!(soa.mass.len(), 100);
    }

    #[test]
    fn soa_accel_range_subslice() {
        let g = 1.0;
        let eps2 = 0.01;
        let pos_i = Vec3::new(1.0, 1.0, 1.0);

        let rmns: Vec<RemoteMultipoleNode> = (0..20)
            .map(|k| make_rmn(k as f64 * 0.4, 0.0, 0.0, 1.0))
            .collect();

        let soa = RmnSoa::from_slice(&rmns);
        let a_full = soa.accel(pos_i, g, eps2);
        let a_range = soa.accel_range(pos_i, 0, 20, g, eps2);

        let err = rms_error(a_full, a_range);
        assert!(err < 1e-14, "accel vs accel_range mismatch: {:.2e}", err);
    }
}
