//! Hexadecapolo (orden multipolar l = 4): tensor cartesiano totalmente simétrico y sin traza.
//!
//! Se almacenan las **15** componentes compactas del tensor simétrico de rango 4 en el orden
//! de grados `(nx, ny, nz)` en `(x, y, z)` con `nx + ny + nz = 4` (convención tipo Gaussian /
//! PAMoC). Las restricciones STF (traza nula en todo par de índices) ya están incorporadas en la
//! construcción tipo Stone/Buckingham (`Φ` a partir de momentos primitivos; véase ecuación (7.12)
//! en la notación de momentos sin traza).
//!
//! La corrección de aceleración usa el factor **`1/4!`** de la expansión de `|r-r'|⁻¹`
//! (`ψ ⊃ −(G/4!) Φ_{ijkl} ∂_i∂_j∂_k∂_l |r|⁻¹` ⇒ `a_α` con el mismo prefactor frente a `Φ ∂_α ∂⁴|r|⁻¹`).
//!
//! Las derivadas explícitas están en `hex_dt_patterns.rs`.
use crate::hex_dt_patterns::{eval_dt_x, eval_dt_y, eval_dt_z};
use gadget_ng_core::Vec3;

/// Prefactor `1/4!` del potencial multipolar de orden 4 (expansión de `|r-r'|⁻¹`).
const L4_FACT: f64 = 1.0 / 24.0;

/// Grados `x^nx y^ny z^nz` con `nx + ny + nz = 4` (15 multiconjuntos).
pub(crate) const HEX_DEGS: [(u8, u8, u8); 15] = [
    (4, 0, 0),
    (3, 1, 0),
    (3, 0, 1),
    (2, 2, 0),
    (2, 1, 1),
    (2, 0, 2),
    (1, 3, 0),
    (1, 2, 1),
    (1, 1, 2),
    (1, 0, 3),
    (0, 4, 0),
    (0, 3, 1),
    (0, 2, 2),
    (0, 1, 3),
    (0, 0, 4),
];

/// `pat[i][j][k][l]` aplanado en orden C (i más lento): clase 0..14 de `∂_x T_{ijkl}`.
const HEX_PAT_FLAT: [u8; 81] = [
    0, 1, 2, 1, 3, 4, 2, 4, 5, 1, 3, 4, 3, 6, 7, 4, 7, 8, 2, 4, 5, 4, 7, 8, 5, 8, 9, 1, 3, 4, 3, 6,
    7, 4, 7, 8, 3, 6, 7, 6, 10, 11, 7, 11, 12, 4, 7, 8, 7, 11, 12, 8, 12, 13, 2, 4, 5, 4, 7, 8, 5,
    8, 9, 4, 7, 8, 7, 11, 12, 8, 12, 13, 5, 8, 9, 8, 12, 13, 9, 13, 14,
];

#[inline(always)]
fn delta(a: usize, b: usize) -> f64 {
    if a == b { 1.0 } else { 0.0 }
}

#[inline(always)]
fn primitive_m2(v: [f64; 3], m: f64, a: usize, b: usize) -> f64 {
    m * v[a] * v[b]
}

/// Índices representativos para una partición `(nx,ny,nz)`.
#[inline(always)]
fn rep_four_indices(nx: usize, ny: usize, nz: usize) -> (usize, usize, usize, usize) {
    let mut idx = [0usize; 4];
    let mut p = 0usize;
    for _ in 0..nx {
        idx[p] = 0;
        p += 1;
    }
    for _ in 0..ny {
        idx[p] = 1;
        p += 1;
    }
    for _ in 0..nz {
        idx[p] = 2;
        p += 1;
    }
    (idx[0], idx[1], idx[2], idx[3])
}

#[inline(always)]
pub(crate) fn hex_slot(nx: usize, ny: usize, nz: usize) -> usize {
    HEX_DEGS
        .iter()
        .position(|&(a, b, c)| (a as usize, b as usize, c as usize) == (nx, ny, nz))
        .expect("hex_slot: grados inválidos")
}

/// Componente `Φ_{ijkl}` a partir del arreglo compacto `hex`.
#[inline(always)]
pub(crate) fn phi_ijkl(hex: &[f64; 15], i: usize, j: usize, k: usize, l: usize) -> f64 {
    let mut c = [0u8; 3];
    c[i] += 1;
    c[j] += 1;
    c[k] += 1;
    c[l] += 1;
    hex[hex_slot(c[0] as usize, c[1] as usize, c[2] as usize)]
}

/// Tensor STF hexadecapolar para una masa puntual en `s` (teorema del eje paralelo: término `m · Φ(s)`).
///
/// Usa la forma traceless de Stone (7.12) para momentos primitivos de orden ≤ 4.
pub(crate) fn outer4_tf(s: Vec3, m: f64) -> [f64; 15] {
    let v = [s.x, s.y, s.z];
    let s2 = s.dot(s);
    let s4 = s2 * s2;

    let mut out = [0.0_f64; 15];
    for (slot, &(nx, ny, nz)) in HEX_DEGS.iter().enumerate() {
        let (i, j, k, l) = rep_four_indices(nx as usize, ny as usize, nz as usize);
        let mijkl = m * v[i] * v[j] * v[k] * v[l];
        let mut s6 = 0.0_f64;
        s6 += primitive_m2(v, m, i, j) * delta(k, l);
        s6 += primitive_m2(v, m, i, k) * delta(j, l);
        s6 += primitive_m2(v, m, i, l) * delta(j, k);
        s6 += primitive_m2(v, m, j, k) * delta(i, l);
        s6 += primitive_m2(v, m, j, l) * delta(i, k);
        s6 += primitive_m2(v, m, k, l) * delta(i, j);
        let ddd = delta(i, j) * delta(k, l) + delta(i, k) * delta(j, l) + delta(i, l) * delta(j, k);
        out[slot] = (35.0 / 8.0) * mijkl - (5.0 / 8.0) * s2 * s6 + (1.0 / 8.0) * s4 * m * ddd;
    }
    out
}

/// Pesos `W[p] = Σ_{ijkl: pat=p} Φ_{ijkl}` para factorizar `∂_α T`.
#[inline]
fn hex_weights(hex: &[f64; 15]) -> [f64; 15] {
    let mut w = [0.0_f64; 15];
    for i in 0..3usize {
        for j in 0..3usize {
            for k in 0..3usize {
                for l in 0..3usize {
                    let p = HEX_PAT_FLAT[i * 27 + j * 9 + k * 3 + l] as usize;
                    w[p] += phi_ijkl(hex, i, j, k, l);
                }
            }
        }
    }
    w
}

/// Aceleración hexadecapolar (bare): `a_α = −G Σ Φ_{ijkl} ∂_α ∂_i∂_j∂_k∂_l |r|⁻¹`.
pub(crate) fn hex_accel(r: Vec3, hex: &[f64; 15], g: f64) -> Vec3 {
    let r2 = r.dot(r);
    if r2 < 1e-300 {
        return Vec3::zero();
    }
    let rx = r.x;
    let ry = r.y;
    let rz = r.z;
    let w = hex_weights(hex);
    let mut ax = 0.0_f64;
    let mut ay = 0.0_f64;
    let mut az = 0.0_f64;
    for p in 0..15usize {
        ax += eval_dt_x(p, rx, ry, rz, r2) * w[p];
        ay += eval_dt_y(p, rx, ry, rz, r2) * w[p];
        az += eval_dt_z(p, rx, ry, rz, r2) * w[p];
    }
    Vec3::new(-g * L4_FACT * ax, -g * L4_FACT * ay, -g * L4_FACT * az)
}

/// Versión para cuando `r2_eff` ya incluye el softening (`|r|²+ε²`), p. ej. `(r_inv)⁻²`
/// del paso monopolar SoA (sin volver a sumar ε²).
pub(crate) fn hex_accel_from_r2s(r: Vec3, r2_eff: f64, hex: &[f64; 15], g: f64) -> Vec3 {
    if r2_eff < 1e-300 {
        return Vec3::zero();
    }
    let rx = r.x;
    let ry = r.y;
    let rz = r.z;
    let w = hex_weights(hex);
    let mut ax = 0.0_f64;
    let mut ay = 0.0_f64;
    let mut az = 0.0_f64;
    for p in 0..15usize {
        ax += eval_dt_x(p, rx, ry, rz, r2_eff) * w[p];
        ay += eval_dt_y(p, rx, ry, rz, r2_eff) * w[p];
        az += eval_dt_z(p, rx, ry, rz, r2_eff) * w[p];
    }
    Vec3::new(-g * L4_FACT * ax, -g * L4_FACT * ay, -g * L4_FACT * az)
}

/// Igual que [`hex_accel`] pero con `|r|² → |r|² + ε²` en los denominadores tipo `|r|⁻¹₁₁`
/// (coherente con quad/oct suavizados en `octree`).
pub(crate) fn hex_accel_softened(r: Vec3, hex: &[f64; 15], g: f64, eps2: f64) -> Vec3 {
    let r2 = r.dot(r) + eps2;
    if r2 < 1e-300 {
        return Vec3::zero();
    }
    let rx = r.x;
    let ry = r.y;
    let rz = r.z;
    let w = hex_weights(hex);
    let mut ax = 0.0_f64;
    let mut ay = 0.0_f64;
    let mut az = 0.0_f64;
    for p in 0..15usize {
        ax += eval_dt_x(p, rx, ry, rz, r2) * w[p];
        ay += eval_dt_y(p, rx, ry, rz, r2) * w[p];
        az += eval_dt_z(p, rx, ry, rz, r2) * w[p];
    }
    Vec3::new(-g * L4_FACT * ax, -g * L4_FACT * ay, -g * L4_FACT * az)
}
