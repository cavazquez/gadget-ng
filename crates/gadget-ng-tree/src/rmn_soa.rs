#![allow(unused_unsafe)]
//! Representación **SoA (Structure of Arrays)** para lotes de [`RemoteMultipoleNode`].
#![allow(clippy::too_many_arguments)]
//!
//! ## Motivación
//!
//! En el layout AoS original cada `RemoteMultipoleNode` ocupa [`crate::RMN_FLOATS`] `f64` contiguos.
//! Con SoA todas las coordenadas `cx[]`, masas `mass[]`, etc. forman columnas
//! contiguas, mejorando cache locality y habilitando SIMD real.
//!
//! ## Kernels disponibles
//!
//! ### Fase 14 — Kernel fusionado auto-vectorizable (`accel_soa_scalar`)
//!
//! - Loop mono+quad+oct con 1 sqrt por nodo j (3× menos que AoS).
//! - Con `#[target_feature(avx2,fma)]` el compilador intenta auto-vectorizar.
//! - El loop fusionado con 17 arrays y condicionales limita la vectorización
//!   efectiva a registros `xmm` (SSE2), no `ymm` (AVX2 completo).
//!
//! ### Fase 15 — Kernel two-pass con intrinsics AVX2 explícitos
//!
//! Estrategia para obtener SIMD real garantizado:
//!
//! 1. **Pass 1 (`mono_pass_avx2`)**: procesa **4 f64 por iteración** con `__m256d`.
//!    Calcula `r_inv[j]` para cada nodo j y lo almacena en un buffer de stack.
//!    Acumula la contribución monopolar. Usa `vmulpd`, `vaddpd`, `vfmadd*`,
//!    `vsqrtpd` con registros `ymm` garantizados.
//!
//! 2. **Pass 2 (`quad_oct_pass_scalar`)**: usa el `r_inv[j]` almacenado para
//!    calcular quad+oct **sin ningún `sqrt` adicional**. El número total de
//!    llamadas a `sqrt` es N — igual que el kernel fusionado de Fase 14 —
//!    pero el loop monopolar es ahora completamente vectorizado en AVX2.
//!
//! El procesado se hace en **chunks de `RINV_CHUNK = 256` elementos** para que
//! el buffer `r_inv` (256 × 8 = 2 KB) quepa en L1 y evitar cualquier
//! allocación dinámica.
//!
//! ## Despacho en tiempo de ejecución
//!
//! - AVX2 + FMA detectados en runtime → `accel_range_p15` (two-pass)
//! - Fallback → `accel_soa_scalar` (fused, Phase 14)

use crate::hexadecapole::{hex_accel_from_r2s, hex_accel_softened};
use crate::octree::RemoteMultipoleNode;
use gadget_ng_core::Vec3;

// ── Constantes ────────────────────────────────────────────────────────────────

/// Tamaño del chunk para el buffer `r_inv` de stack.
/// 256 × 8 bytes = 2 KiB → cabe holgadamente en L1 (≥32 KiB en x86 moderno).
const RINV_CHUNK: usize = 256;

// ── Struct SoA ────────────────────────────────────────────────────────────────

/// Layout columnar de un batch de [`RemoteMultipoleNode`].
///
/// - `quad[k]`: columnas del tensor cuadrupolar en orden
///   `[qxx, qxy, qxz, qyy, qyz, qzz]`.
/// - `oct[k]`: columnas del tensor octupolar en orden
///   `[o_xxx, o_xxy, o_xxz, o_xyy, o_xyz, o_yyy, o_yzz]`.
/// - `hex[k]`: 15 columnas del hexadecapolo STF (multigrados nx,ny,nz).
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
    /// 15 columnas del hexadecapolo STF.
    pub hex: [Vec<f64>; 15],
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
            hex: std::array::from_fn(|_| Vec::with_capacity(cap)),
            len: 0,
        }
    }

    /// Convierte un slice AoS de `RemoteMultipoleNode` a layout SoA.
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
            for k in 0..15 {
                self.hex[k].push(rmn.hex[k]);
            }
            self.len += 1;
        }
    }

    /// Vacía el SoA sin liberar la memoria reservada.
    pub fn clear(&mut self) {
        self.cx.clear();
        self.cy.clear();
        self.cz.clear();
        self.mass.clear();
        for k in 0..6 {
            self.quad[k].clear();
        }
        for k in 0..7 {
            self.oct[k].clear();
        }
        for k in 0..15 {
            self.hex[k].clear();
        }
        self.len = 0;
    }
}

// ── Fase 14: kernel fusionado (fallback / no-AVX2) ────────────────────────────

/// Kernel fusionado mono+quad+oct — versión escalar con 1 sqrt por nodo j.
///
/// Sirve como fallback en plataformas sin AVX2/FMA y como referencia.
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
    let h: [&[f64]; 15] = std::array::from_fn(|k| &soa.hex[k][start..start + len]);

    for j in 0..len {
        let mj = mass[j];
        if mj == 0.0 {
            continue;
        }
        let rx = xi - cx[j];
        let ry = yi - cy[j];
        let rz = zi - cz[j];
        let r2 = rx * rx + ry * ry + rz * rz + eps2;
        if r2 < 1e-300 {
            continue;
        }
        let r_inv = 1.0 / r2.sqrt();

        // Monopolo
        let r3_inv = r_inv * r_inv * r_inv;
        let mono_fac = -g * mj * r3_inv;
        ax += mono_fac * rx;
        ay += mono_fac * ry;
        az += mono_fac * rz;

        // Cuadrupolo
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

        // Octupolo
        let o_xxx = o[0][j];
        let o_xxy = o[1][j];
        let o_xxz = o[2][j];
        let o_xyy = o[3][j];
        let o_xyz = o[4][j];
        let o_yyy = o[5][j];
        let o_yzz = o[6][j];
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

        // Hexadecapolo
        let mut harr = [0.0_f64; 15];
        for t in 0..15 {
            harr[t] = h[t][j];
        }
        let a_hex = hex_accel_softened(Vec3::new(rx, ry, rz), &harr, g, eps2);
        ax += a_hex.x;
        ay += a_hex.y;
        az += a_hex.z;
    }

    Vec3::new(ax, ay, az)
}

/// Wrapper Fase 14: añade el atributo `target_feature` para que LLVM pueda
/// intentar auto-vectorizar el loop (produce xmm/SSE2 en la práctica).
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

// ── Fase 15: intrinsics AVX2 explícitos ──────────────────────────────────────

/// Suma horizontal de un registro `__m256d` (4 × f64 → un f64).
///
/// Equivalente a `v[0]+v[1]+v[2]+v[3]` usando instrucciones SSE/AVX.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hadd_m256d(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    // [v0,v1,v2,v3]
    let hi128 = _mm256_extractf128_pd(v, 1); // [v2, v3]
    let lo128 = _mm256_castpd256_pd128(v); // [v0, v1]
    let s128 = _mm_add_pd(lo128, hi128); // [v0+v2, v1+v3]
    let s128h = _mm_unpackhi_pd(s128, s128); // [v1+v3, v1+v3]
    let total = _mm_add_pd(s128, s128h); // [v0+v1+v2+v3, ...]
    _mm_cvtsd_f64(total)
}

/// **Fase 15 — Pass 1 (AVX2 explícito)**: kernel monopolar con intrinsics `__m256d`.
///
/// Procesa **4 elementos por iteración** usando registros `ymm` reales.
/// Instrucciones garantizadas: `vsqrtpd`, `vmulpd`, `vaddpd`, `vfmadd213pd`.
///
/// Además almacena `r_inv[j]` en `r_inv_out[0..len]` para que el Pass 2
/// pueda calcular quad+oct sin ningún `sqrt` extra.
///
/// # Semántica
///
/// Calcula la contribución **monopolar solamente** de nodos `[0, len)` del
/// batch (coordenadas en `cx/cy/cz`, masas en `mass`).
///
/// # Seguridad
/// Requiere que el hardware soporte AVX2 y FMA (verificado antes de llamar).
/// Los slices `cx`, `cy`, `cz`, `mass` deben tener longitud ≥ `len`.
/// El slice `r_inv_out` debe tener longitud exactamente `len`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn mono_pass_avx2(
    xi: f64,
    yi: f64,
    zi: f64,
    cx: &[f64],
    cy: &[f64],
    cz: &[f64],
    mass: &[f64],
    neg_g: f64,
    eps2: f64,
    r_inv_out: &mut [f64],
) -> (f64, f64, f64) {
    use std::arch::x86_64::*;

    let n = cx.len();
    debug_assert_eq!(n, r_inv_out.len());

    // Registros de broadcast (4 copias del escalar en ymm)
    let xi4 = unsafe { _mm256_set1_pd(xi) };
    let yi4 = unsafe { _mm256_set1_pd(yi) };
    let zi4 = unsafe { _mm256_set1_pd(zi) };
    let eps24 = unsafe { _mm256_set1_pd(eps2) };
    let ng4 = unsafe { _mm256_set1_pd(neg_g) }; // −g
    let ones = unsafe { _mm256_set1_pd(1.0) };

    // Acumuladores de aceleración (4 lanes en paralelo)
    let mut ax4 = unsafe { _mm256_setzero_pd() };
    let mut ay4 = unsafe { _mm256_setzero_pd() };
    let mut az4 = unsafe { _mm256_setzero_pd() };

    let chunks = n / 4;
    let tail_start = chunks * 4;

    // ── Loop vectorizado: 4 elementos por iteración ─────────────────────────────────────
    for i in 0..chunks {
        let base = i * 4;

        // Carga 4 coordenadas de los nodos LET (acceso secuencial, cache-friendly)
        let cxj = unsafe { _mm256_loadu_pd(cx.as_ptr().add(base)) };
        let cyj = unsafe { _mm256_loadu_pd(cy.as_ptr().add(base)) };
        let czj = unsafe { _mm256_loadu_pd(cz.as_ptr().add(base)) };
        let mj = unsafe { _mm256_loadu_pd(mass.as_ptr().add(base)) };

        // r = pos_i − com_j  (vectorizado para las 4 lanes)
        let rx = unsafe { _mm256_sub_pd(xi4, cxj) };
        let ry = unsafe { _mm256_sub_pd(yi4, cyj) };
        let rz = unsafe { _mm256_sub_pd(zi4, czj) };

        // r² + ε² usando FMA: rz*rz + (ry*ry + (rx*rx + eps2))
        let r2 = unsafe {
            _mm256_fmadd_pd(
                rz,
                rz,
                _mm256_fmadd_pd(ry, ry, _mm256_fmadd_pd(rx, rx, eps24)),
            )
        };

        // r_inv = 1 / sqrt(r²+ε²)   → instrucción vsqrtpd + vdivpd
        let r_inv = unsafe { _mm256_div_pd(ones, _mm256_sqrt_pd(r2)) };

        // Almacena r_inv para que Pass 2 compute quad+oct sin más sqrt
        unsafe { _mm256_storeu_pd(r_inv_out.as_mut_ptr().add(base), r_inv) };

        // r3_inv = r_inv³ = r_inv * r_inv * r_inv
        let r_inv2 = unsafe { _mm256_mul_pd(r_inv, r_inv) };
        let r3_inv = unsafe { _mm256_mul_pd(r_inv2, r_inv) };

        // factor = −g * mj * r3_inv  (−g ya está en neg_g)
        let factor = unsafe { _mm256_mul_pd(ng4, _mm256_mul_pd(mj, r3_inv)) };

        // ax += factor * rx   (FMA: factor*rx + ax4)
        ax4 = unsafe { _mm256_fmadd_pd(factor, rx, ax4) };
        ay4 = unsafe { _mm256_fmadd_pd(factor, ry, ay4) };
        az4 = unsafe { _mm256_fmadd_pd(factor, rz, az4) };
    }

    // Reducción horizontal: suma los 4 lanes de cada acumulador
    let mut ax = unsafe { hadd_m256d(ax4) };
    let mut ay = unsafe { hadd_m256d(ay4) };
    let mut az = unsafe { hadd_m256d(az4) };

    // ── Tail escalar: elementos restantes (0–3) ───────────────────────────────
    for j in tail_start..n {
        let rx = xi - cx[j];
        let ry = yi - cy[j];
        let rz = zi - cz[j];
        let r2 = rx * rx + ry * ry + rz * rz + eps2;
        let r_inv_j = 1.0 / r2.sqrt();
        r_inv_out[j] = r_inv_j;
        let r3_inv = r_inv_j * r_inv_j * r_inv_j;
        let fac = neg_g * mass[j] * r3_inv;
        ax += fac * rx;
        ay += fac * ry;
        az += fac * rz;
    }

    (ax, ay, az)
}

/// **Fase 15 — Pass 2 (escalar)**: cuadrupolo + octupolo usando `r_inv` pre-computado.
///
/// No llama a `sqrt` — usa los valores `r_inv[j]` almacenados por [`mono_pass_avx2`].
/// El número total de `sqrt` del kernel completo (Pass1 + Pass2) sigue siendo N,
/// igual que el kernel fusionado de Fase 14, pero el Pass1 los calcula 4 en paralelo.
///
/// `r_inv_buf[j]` corresponde al elemento `start+j` del SoA global.
#[inline(always)]
fn quad_oct_pass_scalar(
    xi: f64,
    yi: f64,
    zi: f64,
    start: usize,
    len: usize,
    soa: &RmnSoa,
    r_inv_buf: &[f64],
    g: f64,
) -> (f64, f64, f64) {
    let mut ax = 0.0_f64;
    let mut ay = 0.0_f64;
    let mut az = 0.0_f64;

    let cx = &soa.cx[start..start + len];
    let cy = &soa.cy[start..start + len];
    let cz = &soa.cz[start..start + len];
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
    let h: [&[f64]; 15] = std::array::from_fn(|k| &soa.hex[k][start..start + len]);

    for j in 0..len {
        // r_inv pre-computado en Pass1; si ≈0 el nodo tenía r²≈0 → skip
        let r_inv = r_inv_buf[j];
        // r_inv es siempre finito (eps2 > 0 garantiza r2 ≥ eps2);
        // si quad y oct son todos cero (nodo vacío) las contribuciones son cero igualmente.

        let rx = xi - cx[j];
        let ry = yi - cy[j];
        let rz = zi - cz[j];

        // Potencias de r_inv derivadas (sin sqrt)
        let r_inv2 = r_inv * r_inv;
        let r3_inv = r_inv2 * r_inv;
        let r5_inv = r3_inv * r_inv2;
        let r7_inv = r5_inv * r_inv2;

        // ── Cuadrupolo ────────────────────────────────────────────────────────
        let qxx = q[0][j];
        let qxy = q[1][j];
        let qxz = q[2][j];
        let qyy = q[3][j];
        let qyz = q[4][j];
        let qzz = q[5][j];

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
        let o_xxx = o[0][j];
        let o_xxy = o[1][j];
        let o_xxz = o[2][j];
        let o_xyy = o[3][j];
        let o_xyz = o[4][j];
        let o_yyy = o[5][j];
        let o_yzz = o[6][j];
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

        let r9_inv = r7_inv * r_inv2;
        let co1 = -g * 0.5 * r7_inv;
        let co2 = g * (7.0 / 6.0) * orrr * r9_inv;
        ax += co1 * orr_x + co2 * rx;
        ay += co1 * orr_y + co2 * ry;
        az += co1 * orr_z + co2 * rz;

        let r2_eff = 1.0 / (r_inv * r_inv);
        let mut harr = [0.0_f64; 15];
        for t in 0..15 {
            harr[t] = h[t][j];
        }
        let a_hex = hex_accel_from_r2s(Vec3::new(rx, ry, rz), r2_eff, &harr, g);
        ax += a_hex.x;
        ay += a_hex.y;
        az += a_hex.z;
    }

    (ax, ay, az)
}

/// **Fase 15 — kernel two-pass completo** para un sub-rango `[start, start+len)`.
///
/// Procesa en chunks de `RINV_CHUNK` elementos para que el buffer `r_inv`
/// de 2 KiB quede en L1 y no requiera ninguna allocación dinámica.
///
/// # Seguridad
/// Requiere AVX2+FMA en el hardware (verificado por el llamador).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn accel_p15_avx2_range(
    xi: f64,
    yi: f64,
    zi: f64,
    start: usize,
    len: usize,
    soa: &RmnSoa,
    g: f64,
    eps2: f64,
) -> Vec3 {
    let mut total_ax = 0.0_f64;
    let mut total_ay = 0.0_f64;
    let mut total_az = 0.0_f64;

    // Buffer r_inv en stack: 256 × 8 = 2 KiB
    let mut r_inv_buf = [0.0_f64; RINV_CHUNK];

    let neg_g = -g;
    let mut chunk_start = start;
    let full_end = start + len;

    while chunk_start < full_end {
        let chunk_end = (chunk_start + RINV_CHUNK).min(full_end);
        let chunk_len = chunk_end - chunk_start;

        let cx = &soa.cx[chunk_start..chunk_end];
        let cy = &soa.cy[chunk_start..chunk_end];
        let cz = &soa.cz[chunk_start..chunk_end];
        let mass = &soa.mass[chunk_start..chunk_end];
        let rinv_slice = &mut r_inv_buf[..chunk_len];

        // Pass 1: AVX2 monopolar + almacena r_inv
        let (ax1, ay1, az1) =
            unsafe { mono_pass_avx2(xi, yi, zi, cx, cy, cz, mass, neg_g, eps2, rinv_slice) };
        total_ax += ax1;
        total_ay += ay1;
        total_az += az1;

        // Pass 2: cuadrupolo + octupolo sin sqrt (usa r_inv almacenado)
        let (ax2, ay2, az2) =
            quad_oct_pass_scalar(xi, yi, zi, chunk_start, chunk_len, soa, rinv_slice, g);
        total_ax += ax2;
        total_ay += ay2;
        total_az += az2;

        chunk_start += RINV_CHUNK;
    }

    Vec3::new(total_ax, total_ay, total_az)
}

// ── API pública ───────────────────────────────────────────────────────────────

impl RmnSoa {
    /// Aceleración gravitacional de todos los nodos del SoA sobre `pos_i`.
    ///
    /// Despacha al kernel AVX2 explícito (Fase 15) si el hardware lo soporta;
    /// en caso contrario usa el kernel fusionado escalar (Fase 14).
    #[inline]
    pub fn accel(&self, pos_i: Vec3, g: f64, eps2: f64) -> Vec3 {
        self.accel_range(pos_i, 0, self.len, g, eps2)
    }

    /// Aceleración gravitacional de los nodos `[start, start+len)` sobre `pos_i`.
    ///
    /// Despacho en tiempo de ejecución:
    /// - AVX2 + FMA disponibles → `accel_p15_avx2_range` (intrinsics explícitos)
    /// - Fallback → `accel_soa_scalar` (kernel fusionado Fase 14)
    #[inline]
    pub fn accel_range(&self, pos_i: Vec3, start: usize, len: usize, g: f64, eps2: f64) -> Vec3 {
        if len == 0 {
            return Vec3::zero();
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: verificamos AVX2+FMA en runtime.
                return unsafe {
                    accel_p15_avx2_range(pos_i.x, pos_i.y, pos_i.z, start, len, self, g, eps2)
                };
            }
        }

        accel_soa_scalar(pos_i.x, pos_i.y, pos_i.z, start, len, self, g, eps2)
    }

    /// Kernel Fase 14 (fusionado con target_feature, sin intrinsics explícitos).
    /// Disponible para benchmarks comparativos directos.
    #[inline]
    pub fn accel_range_p14(
        &self,
        pos_i: Vec3,
        start: usize,
        len: usize,
        g: f64,
        eps2: f64,
    ) -> Vec3 {
        if len == 0 {
            return Vec3::zero();
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe {
                    accel_soa_avx2(pos_i.x, pos_i.y, pos_i.z, start, len, self, g, eps2)
                };
            }
        }
        accel_soa_scalar(pos_i.x, pos_i.y, pos_i.z, start, len, self, g, eps2)
    }
}

// ── Fase 16: kernel tileado 4×N_i ────────────────────────────────────────────

/// Tamaño del chunk para el buffer `r_inv` en el kernel tileado.
/// 64 RMNs × 4 partículas × 8 bytes = 2 KiB → cabe en L1.
const RINV_CHUNK_4XI: usize = 64;

/// **Fase 16 — Pass 1 (AVX2 tileado)**: monopolar para **4 partículas × 1 RMN por iteración**.
///
/// Invierte el rol respecto a [`mono_pass_avx2`] (P15):
/// - P15: 1 partícula × 4 RMNs por iteración SIMD
/// - P16: **4 partículas × 1 RMN por iteración SIMD** ← este kernel
///
/// Cada registor `ymm` lleva valores para 4 partículas distintas contra el mismo
/// nodo j. Con batch_avg≈3.7 se obtienen ~3.7 iteraciones SIMD completas por
/// llamada de hoja, frente a ~0.9 de P15.
///
/// Para cada RMN j:
/// - `broadcast(cx[j])` → `[cx[j], cx[j], cx[j], cx[j]]`
/// - `load xi4` → `[xi[0], xi[1], xi[2], xi[3]]`
/// - `rx4 = xi4 - cxj`, `r2_4`, `r_inv4` → `vsqrtpd ymm`, etc.
/// - Almacena `r_inv4` en `r_inv_out[j]` para Pass 2.
/// - Acumula `ax4 += factor4 * rx4` (FMA).
///
/// No requiere reducción horizontal al finalizar: cada lane k del acumulador
/// contiene el resultado para la partícula k.
///
/// # Seguridad
/// Requiere AVX2 + FMA (verificado por el llamador).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn mono_pass_avx2_4xi(
    xi: &[f64; 4],
    yi: &[f64; 4],
    zi: &[f64; 4],
    cx: &[f64],
    cy: &[f64],
    cz: &[f64],
    mass: &[f64],
    neg_g: f64,
    eps2: f64,
    r_inv_out: &mut [[f64; 4]],
) -> ([f64; 4], [f64; 4], [f64; 4]) {
    use std::arch::x86_64::*;

    let n = cx.len();
    debug_assert_eq!(n, r_inv_out.len());

    // Cargar posiciones de las 4 partículas en registros ymm
    let xi4 = unsafe { _mm256_loadu_pd(xi.as_ptr()) }; // [xi[0], xi[1], xi[2], xi[3]]
    let yi4 = unsafe { _mm256_loadu_pd(yi.as_ptr()) };
    let zi4 = unsafe { _mm256_loadu_pd(zi.as_ptr()) };
    let eps24 = unsafe { _mm256_set1_pd(eps2) };
    let ones = unsafe { _mm256_set1_pd(1.0) };

    // Acumuladores: cada lane k acumula para partícula k
    let mut ax4 = unsafe { _mm256_setzero_pd() };
    let mut ay4 = unsafe { _mm256_setzero_pd() };
    let mut az4 = unsafe { _mm256_setzero_pd() };

    // Loop sobre RMNs (1 por iteración SIMD — 4 partículas en paralelo)
    for j in 0..n {
        // Broadcast coordenadas del nodo j → mismo valor en las 4 lanes
        let cxj = unsafe { _mm256_set1_pd(cx[j]) };
        let cyj = unsafe { _mm256_set1_pd(cy[j]) };
        let czj = unsafe { _mm256_set1_pd(cz[j]) };
        let mj_ng = unsafe { _mm256_set1_pd(neg_g * mass[j]) };

        // r = pos_i - com_j para las 4 partículas simultáneamente
        let rx = unsafe { _mm256_sub_pd(xi4, cxj) };
        let ry = unsafe { _mm256_sub_pd(yi4, cyj) };
        let rz = unsafe { _mm256_sub_pd(zi4, czj) };

        // r² + ε² via FMA: rz²+ry²+rx²+eps2
        let r2 = unsafe {
            _mm256_fmadd_pd(
                rz,
                rz,
                _mm256_fmadd_pd(ry, ry, _mm256_fmadd_pd(rx, rx, eps24)),
            )
        };

        // r_inv = 1 / sqrt(r²+ε²) para las 4 partículas (vsqrtpd ymm)
        let r_inv = unsafe { _mm256_div_pd(ones, _mm256_sqrt_pd(r2)) };

        // Almacenar r_inv para las 4 partículas → Pass 2 (quad+oct sin sqrt)
        unsafe { _mm256_storeu_pd(r_inv_out[j].as_mut_ptr(), r_inv) };

        // r3_inv = r_inv³ = r_inv * r_inv * r_inv
        let r_inv2 = unsafe { _mm256_mul_pd(r_inv, r_inv) };
        let r3_inv = unsafe { _mm256_mul_pd(r_inv2, r_inv) };

        // factor = neg_g * mass[j] * r3_inv  (ya tiene el signo correcto)
        let factor = unsafe { _mm256_mul_pd(mj_ng, r3_inv) };

        // acc += factor * r  (FMA: factor*r + acc)
        ax4 = unsafe { _mm256_fmadd_pd(factor, rx, ax4) };
        ay4 = unsafe { _mm256_fmadd_pd(factor, ry, ay4) };
        az4 = unsafe { _mm256_fmadd_pd(factor, rz, az4) };
    }

    // Extraer resultados: cada lane k contiene la aceleración monopolar de partícula k
    let mut ax_out = [0.0_f64; 4];
    let mut ay_out = [0.0_f64; 4];
    let mut az_out = [0.0_f64; 4];
    unsafe { _mm256_storeu_pd(ax_out.as_mut_ptr(), ax4) };
    unsafe { _mm256_storeu_pd(ay_out.as_mut_ptr(), ay4) };
    unsafe { _mm256_storeu_pd(az_out.as_mut_ptr(), az4) };

    (ax_out, ay_out, az_out)
}

/// **Fase 16 — Pass 2 (escalar tileado)**: cuadrupolo + octupolo para 4 partículas.
///
/// Usa `r_inv_buf[j][k]` pre-computado en Pass 1 para evitar llamadas a `sqrt`.
/// El número total de `sqrt` = N (idéntico a P15, pero calculadas en P1 con AVX2).
#[inline(always)]
fn quad_oct_pass_scalar_4xi(
    xi: &[f64; 4],
    yi: &[f64; 4],
    zi: &[f64; 4],
    start: usize,
    len: usize,
    soa: &RmnSoa,
    r_inv_buf: &[[f64; 4]],
    g: f64,
    tile_size: usize,
) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let mut ax = [0.0_f64; 4];
    let mut ay = [0.0_f64; 4];
    let mut az = [0.0_f64; 4];

    let cx = &soa.cx[start..start + len];
    let cy = &soa.cy[start..start + len];
    let cz = &soa.cz[start..start + len];
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
    let h: [&[f64]; 15] = std::array::from_fn(|t| &soa.hex[t][start..start + len]);

    for j in 0..len {
        let rinv4 = &r_inv_buf[j]; // [r_inv para particula 0,1,2,3]

        for k in 0..tile_size {
            let r_inv = rinv4[k];
            let rx = xi[k] - cx[j];
            let ry = yi[k] - cy[j];
            let rz = zi[k] - cz[j];

            let r_inv2 = r_inv * r_inv;
            let r3_inv = r_inv2 * r_inv;
            let r5_inv = r3_inv * r_inv2;
            let r7_inv = r5_inv * r_inv2;

            // Cuadrupolo
            let qxx = q[0][j];
            let qxy = q[1][j];
            let qxz = q[2][j];
            let qyy = q[3][j];
            let qyz = q[4][j];
            let qzz = q[5][j];
            let qr_x = qxx * rx + qxy * ry + qxz * rz;
            let qr_y = qxy * rx + qyy * ry + qyz * rz;
            let qr_z = qxz * rx + qyz * ry + qzz * rz;
            let rqr = qr_x * rx + qr_y * ry + qr_z * rz;
            let c1 = g * r5_inv;
            let c2 = g * 2.5 * rqr * r7_inv;
            ax[k] += c1 * qr_x - c2 * rx;
            ay[k] += c1 * qr_y - c2 * ry;
            az[k] += c1 * qr_z - c2 * rz;

            // Octupolo
            let o_xxx = o[0][j];
            let o_xxy = o[1][j];
            let o_xxz = o[2][j];
            let o_xyy = o[3][j];
            let o_xyz = o[4][j];
            let o_yyy = o[5][j];
            let o_yzz = o[6][j];
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
            let r9_inv = r7_inv * r_inv2;
            let co1 = -g * 0.5 * r7_inv;
            let co2 = g * (7.0 / 6.0) * orrr * r9_inv;
            ax[k] += co1 * orr_x + co2 * rx;
            ay[k] += co1 * orr_y + co2 * ry;
            az[k] += co1 * orr_z + co2 * rz;

            let r2_eff = 1.0 / (r_inv * r_inv);
            let mut harr = [0.0_f64; 15];
            for t in 0..15 {
                harr[t] = h[t][j];
            }
            let a_hex = hex_accel_from_r2s(Vec3::new(rx, ry, rz), r2_eff, &harr, g);
            ax[k] += a_hex.x;
            ay[k] += a_hex.y;
            az[k] += a_hex.z;
        }
    }

    (ax, ay, az)
}

/// **Fase 16 — kernel two-pass tileado** para 4 partículas y sub-rango `[start, start+len)`.
///
/// Procesa en chunks de `RINV_CHUNK_4XI` RMNs. Buffer r_inv: 64×4×8 = 2 KiB (stack).
///
/// # Seguridad
/// Requiere AVX2+FMA (verificado por el llamador). `tile_size ∈ [1,4]`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn accel_p16_avx2_range_4xi(
    xi: &[f64; 4],
    yi: &[f64; 4],
    zi: &[f64; 4],
    start: usize,
    len: usize,
    soa: &RmnSoa,
    g: f64,
    eps2: f64,
    tile_size: usize,
) -> [Vec3; 4] {
    let mut total_ax = [0.0_f64; 4];
    let mut total_ay = [0.0_f64; 4];
    let mut total_az = [0.0_f64; 4];

    // Buffer r_inv en stack: 64 RMNs × 4 partículas × 8 = 2 KiB
    let mut r_inv_buf = [[0.0_f64; 4]; RINV_CHUNK_4XI];

    let neg_g = -g;
    let mut chunk_start = start;
    let full_end = start + len;

    while chunk_start < full_end {
        let chunk_end = (chunk_start + RINV_CHUNK_4XI).min(full_end);
        let chunk_len = chunk_end - chunk_start;

        let cx = &soa.cx[chunk_start..chunk_end];
        let cy = &soa.cy[chunk_start..chunk_end];
        let cz = &soa.cz[chunk_start..chunk_end];
        let mass = &soa.mass[chunk_start..chunk_end];
        let rinv_slice = &mut r_inv_buf[..chunk_len];

        // Pass 1: AVX2 monopolar tileado — 4 partículas × chunk_len RMNs
        let (ax1, ay1, az1) =
            unsafe { mono_pass_avx2_4xi(xi, yi, zi, cx, cy, cz, mass, neg_g, eps2, rinv_slice) };

        // Pass 2: quad+oct escalar tileado — sin sqrt adicionales
        let (ax2, ay2, az2) = quad_oct_pass_scalar_4xi(
            xi,
            yi,
            zi,
            chunk_start,
            chunk_len,
            soa,
            rinv_slice,
            g,
            tile_size,
        );

        for k in 0..tile_size {
            total_ax[k] += ax1[k] + ax2[k];
            total_ay[k] += ay1[k] + ay2[k];
            total_az[k] += az1[k] + az2[k];
        }

        chunk_start += RINV_CHUNK_4XI;
    }

    let mut result = [Vec3::zero(); 4];
    for k in 0..tile_size {
        result[k] = Vec3::new(total_ax[k], total_ay[k], total_az[k]);
    }
    result
}

impl RmnSoa {
    /// **Fase 16 — API tileada**: aceleración de `[start, start+len)` sobre 4 partículas.
    ///
    /// `pos[0..tile_size]` son posiciones válidas; `pos[tile_size..4]` se ignoran.
    /// Devuelve `[Vec3; 4]` donde solo las primeras `tile_size` son significativas.
    ///
    /// Despacha a AVX2 4xi si hardware lo soporta; fallback: `tile_size` llamadas a `accel_range`.
    #[inline]
    pub fn accel_range_4xi(
        &self,
        pos: &[Vec3; 4],
        start: usize,
        len: usize,
        g: f64,
        eps2: f64,
        tile_size: usize,
    ) -> [Vec3; 4] {
        debug_assert!((1..=4).contains(&tile_size));
        if len == 0 {
            return [Vec3::zero(); 4];
        }

        // Extraer arrays de posición (layout contiguo para cargar en ymm)
        let mut xi = [0.0_f64; 4];
        let mut yi = [0.0_f64; 4];
        let mut zi = [0.0_f64; 4];
        for k in 0..tile_size {
            xi[k] = pos[k].x;
            yi[k] = pos[k].y;
            zi[k] = pos[k].z;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: verificamos AVX2+FMA en runtime.
                return unsafe {
                    accel_p16_avx2_range_4xi(&xi, &yi, &zi, start, len, self, g, eps2, tile_size)
                };
            }
        }

        // Fallback escalar: tile_size llamadas individuales
        let mut result = [Vec3::zero(); 4];
        for k in 0..tile_size {
            result[k] = accel_soa_scalar(pos[k].x, pos[k].y, pos[k].z, start, len, self, g, eps2);
        }
        result
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::{RemoteMultipoleNode, accel_from_let};

    fn make_rmn(cx: f64, cy: f64, cz: f64, mass: f64) -> RemoteMultipoleNode {
        RemoteMultipoleNode {
            com: Vec3::new(cx, cy, cz),
            mass,
            quad: [0.0; 6],
            oct: [0.0; 7],
            hex: [0.0; 15],
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
            hex: [0.0; 15],
            half_size: 0.5,
        }
    }

    fn rms_error(a: Vec3, b: Vec3) -> f64 {
        let denom = (a.x * a.x + a.y * a.y + a.z * a.z).sqrt().max(1e-300);
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        ((dx * dx + dy * dy + dz * dz).sqrt()) / denom
    }

    // ── Tests heredados de Fase 14 (regresión) ─────────────────────────────────

    #[test]
    fn soa_vs_aos_monopole_only() {
        let g = 1.0;
        let eps2 = 0.01_f64.powi(2);
        let pos_i = Vec3::new(1.0, 2.0, 3.0);
        let rmns: Vec<RemoteMultipoleNode> = (0..16)
            .map(|k| {
                make_rmn(
                    k as f64 * 0.5,
                    -(k as f64),
                    k as f64 * 0.3,
                    1.0 + k as f64 * 0.1,
                )
            })
            .collect();
        let soa = RmnSoa::from_slice(&rmns);
        let a_soa = soa.accel(pos_i, g, eps2);
        let a_aos = accel_from_let(pos_i, &rmns, g, eps2);
        let err = rms_error(a_soa, a_aos);
        assert!(err < 1e-12, "SoA vs AoS monopole RMS = {:.2e}", err);
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
        assert!(err < 1e-12, "SoA vs AoS quad+oct RMS = {:.2e}", err);
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
        assert!(err < 1e-14, "accel vs accel_range: {:.2e}", err);
    }

    // ── Tests nuevos Fase 15 ───────────────────────────────────────────────────

    /// P15 vs escalar para varios N incluyendo non-múltiplos de 4 (tail handling)
    #[test]
    fn p15_vs_scalar_various_n() {
        let g = 1.0;
        let eps2 = 0.04_f64; // sqrt(eps2) = 0.2
        let pos_i = Vec3::new(5.0, -3.0, 2.0);
        let quad = [0.05, -0.02, 0.01, 0.04, -0.01, -0.09];
        let oct = [0.005, -0.003, 0.001, 0.004, -0.001, 0.003, -0.0005];

        for &n in &[1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 32, 64, 256, 257] {
            let rmns: Vec<RemoteMultipoleNode> = (0..n)
                .map(|k| {
                    let t = k as f64 * 0.1;
                    make_rmn_full(
                        (t * 1.5).sin() * 4.0 + 2.0,
                        (t * 0.9).cos() * 3.0 - 1.0,
                        t * 0.2 - 5.0,
                        0.8 + (t * 0.3).sin().abs() * 0.4,
                        quad.map(|v| v * (1.0 + t * 0.02)),
                        oct.map(|v| v * (1.0 + t * 0.02)),
                    )
                })
                .collect();

            let soa = RmnSoa::from_slice(&rmns);
            let a_p15 = soa.accel_range(pos_i, 0, n, g, eps2);
            let a_scalar = accel_soa_scalar(pos_i.x, pos_i.y, pos_i.z, 0, n, &soa, g, eps2);

            let err = rms_error(a_p15, a_scalar);
            assert!(
                err < 1e-12,
                "P15 vs escalar para N={n}: RMS = {err:.3e} (tol 1e-12)"
            );
        }
    }

    /// P15 vs AoS reference para N=500 con quad+oct (test de física completo)
    #[test]
    fn p15_vs_aos_full_physics_n500() {
        let g = 1.0;
        let eps2 = 0.01_f64.powi(2);
        let pos_i = Vec3::new(7.0, -2.0, 4.0);
        let quad = [0.08, -0.04, 0.02, 0.06, -0.015, -0.14];
        let oct = [0.008, -0.004, 0.002, 0.006, -0.002, 0.005, -0.001];

        let rmns: Vec<RemoteMultipoleNode> = (0..500)
            .map(|k| {
                let t = k as f64 * 0.03;
                make_rmn_full(
                    (t * 1.3).sin() * 6.0,
                    (t * 0.7).cos() * 4.0,
                    t * 0.12 - 3.0,
                    0.5 + (t * 0.4).sin().abs() * 0.5,
                    quad.map(|v| v * (1.0 + t * 0.005)),
                    oct.map(|v| v * (1.0 + t * 0.005)),
                )
            })
            .collect();

        let soa = RmnSoa::from_slice(&rmns);
        let a_p15 = soa.accel(pos_i, g, eps2);
        let a_aos = accel_from_let(pos_i, &rmns, g, eps2);

        let err = rms_error(a_p15, a_aos);
        assert!(
            err < 1e-12,
            "P15 vs AoS N=500 quad+oct: RMS = {err:.3e} (tol 1e-12)"
        );
    }

    /// Exactamente N=4 — primer chunk completo sin tail
    #[test]
    fn p15_n4_no_tail() {
        let g = 1.0;
        let eps2 = 0.01;
        let pos_i = Vec3::new(1.0, 0.0, 0.0);
        let rmns: Vec<RemoteMultipoleNode> = vec![
            make_rmn(2.0, 0.0, 0.0, 1.0),
            make_rmn(0.0, 2.0, 0.0, 1.0),
            make_rmn(-2.0, 0.0, 0.0, 1.0),
            make_rmn(0.0, -2.0, 0.0, 1.0),
        ];
        let soa = RmnSoa::from_slice(&rmns);
        let a_p15 = soa.accel(pos_i, g, eps2);
        let a_ref = accel_soa_scalar(pos_i.x, pos_i.y, pos_i.z, 0, 4, &soa, g, eps2);
        let err = rms_error(a_p15, a_ref);
        assert!(err < 1e-14, "N=4 no-tail: RMS = {err:.3e}");
    }

    /// Chunk boundary: N=256 (exactamente RINV_CHUNK, sin segundo chunk)
    #[test]
    fn p15_chunk_boundary_n256() {
        let g = 1.0;
        let eps2 = 0.04;
        let pos_i = Vec3::new(3.0, 1.0, -2.0);
        let rmns: Vec<RemoteMultipoleNode> = (0..256)
            .map(|k| make_rmn(k as f64 * 0.02 - 2.5, (k as f64 * 0.1).sin(), 0.5, 1.0))
            .collect();
        let soa = RmnSoa::from_slice(&rmns);
        let a_p15 = soa.accel(pos_i, g, eps2);
        let a_ref = accel_soa_scalar(pos_i.x, pos_i.y, pos_i.z, 0, 256, &soa, g, eps2);
        let err = rms_error(a_p15, a_ref);
        assert!(err < 1e-12, "N=256 chunk boundary: RMS = {err:.3e}");
    }

    /// N=257 — dos chunks (256+1), verificando que el segundo chunk/tail funciona
    #[test]
    fn p15_two_chunks_n257() {
        let g = 1.0;
        let eps2 = 0.04;
        let pos_i = Vec3::new(-1.0, 2.0, 0.5);
        let rmns: Vec<RemoteMultipoleNode> = (0..257)
            .map(|k| make_rmn(k as f64 * 0.02 - 2.5, (k as f64 * 0.08).cos(), -0.5, 1.0))
            .collect();
        let soa = RmnSoa::from_slice(&rmns);
        let a_p15 = soa.accel(pos_i, g, eps2);
        let a_ref = accel_soa_scalar(pos_i.x, pos_i.y, pos_i.z, 0, 257, &soa, g, eps2);
        let err = rms_error(a_p15, a_ref);
        assert!(err < 1e-12, "N=257 two chunks: RMS = {err:.3e}");
    }

    /// accel_range con start > 0 (sub-rango no trivial)
    #[test]
    fn p15_accel_range_nonzero_start() {
        let g = 1.0;
        let eps2 = 0.01;
        let pos_i = Vec3::new(0.0, 5.0, 0.0);
        let rmns: Vec<RemoteMultipoleNode> = (0..32)
            .map(|k| make_rmn(k as f64 * 0.3, 0.0, 0.0, 1.0))
            .collect();
        let soa = RmnSoa::from_slice(&rmns);

        // Comparar sub-rango [8, 24)
        let a_p15 = soa.accel_range(pos_i, 8, 16, g, eps2);
        let a_ref = accel_soa_scalar(pos_i.x, pos_i.y, pos_i.z, 8, 16, &soa, g, eps2);
        let err = rms_error(a_p15, a_ref);
        assert!(err < 1e-12, "accel_range start=8 len=16: RMS = {err:.3e}");
    }

    /// P15 vs P14 (ambos deben dar < 1e-12 de diferencia en física de doble precisión)
    #[test]
    fn p15_vs_p14_rms() {
        let g = 1.0;
        let eps2 = 0.01_f64.powi(2);
        let pos_i = Vec3::new(4.0, -2.0, 1.5);
        let quad = [0.07, -0.03, 0.01, 0.05, -0.01, -0.12];
        let oct = [0.006, -0.003, 0.001, 0.005, -0.001, 0.004, -0.001];
        let rmns: Vec<RemoteMultipoleNode> = (0..200)
            .map(|k| {
                let t = k as f64 * 0.05;
                make_rmn_full(
                    (t * 1.1).sin() * 5.0,
                    (t * 0.8).cos() * 3.0,
                    t * 0.1 - 5.0,
                    1.0 + (t * 0.2).sin().abs() * 0.3,
                    quad.map(|v| v * (1.0 + t * 0.01)),
                    oct.map(|v| v * (1.0 + t * 0.01)),
                )
            })
            .collect();

        let soa = RmnSoa::from_slice(&rmns);
        let a_p15 = soa.accel_range(pos_i, 0, 200, g, eps2);
        let a_p14 = soa.accel_range_p14(pos_i, 0, 200, g, eps2);

        let err = rms_error(a_p15, a_p14);
        assert!(
            err < 1e-12,
            "P15 vs P14 N=200 quad+oct: RMS = {err:.3e} (tol 1e-12)"
        );
    }

    // ── Tests Fase 16 ────────────────────────────────────────────────────────

    fn rms_error_4xi(result: &[Vec3; 4], ref_: &[Vec3; 4], tile_size: usize) -> f64 {
        let mut sum = 0.0;
        for k in 0..tile_size {
            let d = result[k] - ref_[k];
            sum += d.x * d.x + d.y * d.y + d.z * d.z;
        }
        (sum / (tile_size as f64)).sqrt()
    }

    /// Compara `accel_range_4xi` (tile completo, N variado) contra 4 llamadas
    /// individuales a `accel_soa_scalar`.
    #[test]
    fn p16_4xi_vs_scalar_various_n() {
        let g = 1.0;
        let eps2 = 0.01;
        let positions = [
            Vec3::new(0.5, -1.0, 2.0),
            Vec3::new(-2.0, 0.3, 1.5),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-0.5, 2.5, 0.0),
        ];
        for &n in &[1usize, 4, 8, 16, 17, 64, 65, 128] {
            let rmns: Vec<RemoteMultipoleNode> = (0..n)
                .map(|k| make_rmn(k as f64 * 0.1 - 5.0, (k as f64 * 0.3).sin(), 0.5, 1.0))
                .collect();
            let soa = RmnSoa::from_slice(&rmns);

            let result = soa.accel_range_4xi(&positions, 0, n, g, eps2, 4);

            let mut ref_ = [Vec3::zero(); 4];
            for k in 0..4 {
                ref_[k] = accel_soa_scalar(
                    positions[k].x,
                    positions[k].y,
                    positions[k].z,
                    0,
                    n,
                    &soa,
                    g,
                    eps2,
                );
            }
            let err = rms_error_4xi(&result, &ref_, 4);
            assert!(err < 1e-12, "N={n}: RMS 4xi vs scalar = {err:.3e}");
        }
    }

    /// Verifica que el manejo de tail (tile_size < 4) produce resultados correctos
    /// para las partículas válidas.
    #[test]
    fn p16_4xi_tail_handling() {
        let g = 1.0;
        let eps2 = 0.04;
        let n = 12;
        let rmns: Vec<RemoteMultipoleNode> = (0..n)
            .map(|k| make_rmn(k as f64 * 0.4 - 2.0, -0.5, 0.8, 2.0))
            .collect();
        let soa = RmnSoa::from_slice(&rmns);

        let positions_full = [
            Vec3::new(3.0, -1.0, 0.0),
            Vec3::new(-1.0, 2.0, 1.0),
            Vec3::new(0.0, 0.0, 4.0),
            Vec3::new(1.5, -2.0, -1.0),
        ];

        for tile_size in 1..=3 {
            let result = soa.accel_range_4xi(&positions_full, 0, n, g, eps2, tile_size);

            let mut ref_ = [Vec3::zero(); 4];
            for k in 0..tile_size {
                ref_[k] = accel_soa_scalar(
                    positions_full[k].x,
                    positions_full[k].y,
                    positions_full[k].z,
                    0,
                    n,
                    &soa,
                    g,
                    eps2,
                );
            }
            let err = rms_error_4xi(&result, &ref_, tile_size);
            assert!(
                err < 1e-12,
                "tile_size={tile_size}: RMS 4xi vs scalar = {err:.3e}"
            );
        }
    }

    /// Compara `accel_range_4xi` (P16) contra 4 llamadas a `accel_range` (P15)
    /// con quad+oct completos. RMS debe ser < 1e-12.
    #[test]
    fn p16_4xi_vs_p15_rms() {
        let g = 1.0;
        let eps2 = 0.01_f64.powi(2);
        let positions = [
            Vec3::new(4.0, -2.0, 1.5),
            Vec3::new(-1.0, 3.0, 0.5),
            Vec3::new(2.5, 2.5, -2.5),
            Vec3::new(0.1, -0.1, 5.0),
        ];
        let quad = [0.07, -0.03, 0.01, 0.05, -0.01, -0.12];
        let oct = [0.006, -0.003, 0.001, 0.005, -0.001, 0.004, -0.001];
        let rmns: Vec<RemoteMultipoleNode> = (0..200)
            .map(|k| {
                let t = k as f64 * 0.05;
                make_rmn_full(
                    (t * 1.1).sin() * 5.0,
                    (t * 0.8).cos() * 3.0,
                    t * 0.1 - 5.0,
                    1.0 + (t * 0.2).sin().abs() * 0.3,
                    quad.map(|v| v * (1.0 + t * 0.01)),
                    oct.map(|v| v * (1.0 + t * 0.01)),
                )
            })
            .collect();
        let soa = RmnSoa::from_slice(&rmns);

        let result_4xi = soa.accel_range_4xi(&positions, 0, 200, g, eps2, 4);

        let mut ref_p15 = [Vec3::zero(); 4];
        for k in 0..4 {
            ref_p15[k] = soa.accel_range(positions[k], 0, 200, g, eps2);
        }

        let err = rms_error_4xi(&result_4xi, &ref_p15, 4);
        assert!(err < 1e-12, "P16 vs P15 N=200 quad+oct: RMS = {err:.3e}");
    }
}
