//! Pesos STF para hexadecapolo en kernel GPU (misma convención que `gadget-ng-tree`).

/// Grados `x^nx y^ny z^nz` con `nx + ny + nz = 4`.
const HEX_DEGS: [(u8, u8, u8); 15] = [
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
fn hex_slot(nx: usize, ny: usize, nz: usize) -> usize {
    HEX_DEGS
        .iter()
        .position(|&(a, b, c)| (a as usize, b as usize, c as usize) == (nx, ny, nz))
        .expect("hex_slot: grados inválidos")
}

#[inline(always)]
fn phi_ijkl(hex: &[f64; 15], i: usize, j: usize, k: usize, l: usize) -> f64 {
    let mut c = [0u8; 3];
    c[i] += 1;
    c[j] += 1;
    c[k] += 1;
    c[l] += 1;
    hex[hex_slot(c[0] as usize, c[1] as usize, c[2] as usize)]
}

/// Pesos `W[p] = Σ_{ijkl: pat=p} Φ_{ijkl}` para el kernel WGSL hexadecapolar.
#[inline]
pub fn hex_pattern_weights(hex: &[f64; 15]) -> [f64; 15] {
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
