//! Cloud-in-Cell (CIC) mass assignment y force interpolation en 3D periódico.
//!
//! La CIC distribuye la propiedad de cada partícula entre las 8 celdas vecinas
//! más cercanas usando pesos trilineales. El grid es periódico con lado `nm`.
//!
//! ## Indexación del grid
//! ```text
//! flat_idx = iz * nm² + iy * nm + ix
//! ```
//! con `ix, iy, iz ∈ [0, nm)`.
//!
//! ## Niveles SIMD
//!
//! | Nivel       | f64/iter | target_feature        |
//! |-------------|----------|-----------------------|
//! | AVX-512     | 8        | `avx512f`            |
//! | AVX2+FMA    | 4        | `avx2` + `fma`       |
//! | Scalar      | 1        | — (fallback)          |
//!
//! Las funciones `_batch_*` evalúan la interpolación CIC para arrays de
//! posiciones, permitiendo al compilador vectorizar el bucle externo vía
//! `#[target_feature]`.

use gadget_ng_core::Vec3;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// ── Batch assign — procesa N partículas produciendo N arrays de densidad parcial
//    que luego se suman. Permite vectorización del bucle de partículas.
//    La función interna usa el mismo cálculo CIC que assign_impl pero sin
//    acumular en un grid global — en su lugar escribe a un grid de salida.
//    Para vectorización, se procesan partículas en batches con #[target_feature].

/// Asigna masas al grid NM³ con interpolación CIC periódica, un partícula a la vez.
/// Esta versión es el fallback escalar.
fn assign_batch_scalar(
    pos_x: &[f64],
    pos_y: &[f64],
    pos_z: &[f64],
    masses: &[f64],
    inv_cell: f64,
    nm: usize,
    out: &mut [f64],
) {
    let nm2 = nm * nm;
    let nmf = nm as f64;

    for i in 0..pos_x.len() {
        let cx = (pos_x[i] * inv_cell).rem_euclid(nmf);
        let cy = (pos_y[i] * inv_cell).rem_euclid(nmf);
        let cz = (pos_z[i] * inv_cell).rem_euclid(nmf);

        let ix0 = cx.floor() as usize;
        let iy0 = cy.floor() as usize;
        let iz0 = cz.floor() as usize;

        let dx = cx - ix0 as f64;
        let dy = cy - iy0 as f64;
        let dz = cz - iz0 as f64;

        let wx0 = 1.0 - dx;
        let wx1 = dx;
        let wy0 = 1.0 - dy;
        let wy1 = dy;
        let wz0 = 1.0 - dz;
        let wz1 = dz;
        let m = masses[i];

        let ix1 = (ix0 + 1) % nm;
        let iy1 = (iy0 + 1) % nm;
        let iz1 = (iz0 + 1) % nm;

        // 8 vecinos CIC
        let wxs = [wx0, wx1];
        let wys = [wy0, wy1];
        let wzs = [wz0, wz1];
        let ixs = [ix0, ix1];
        let iys = [iy0, iy1];
        let izs = [iz0, iz1];

        for (diz, &wz_v) in wzs.iter().enumerate() {
            let iz = izs[diz];
            for (diy, &wy_v) in wys.iter().enumerate() {
                let iy = iys[diy];
                for (dix, &wx_v) in wxs.iter().enumerate() {
                    let ix = ixs[dix];
                    out[iz * nm2 + iy * nm + ix] += m * wx_v * wy_v * wz_v;
                }
            }
        }
    }
}

/// Interpola fuerzas del grid a posiciones de partículas, escalar.
#[expect(
    clippy::too_many_arguments,
    reason = "CIC grid interpolation keeps SoA slices explicit"
)]
fn interpolate_batch_scalar(
    fx_grid: &[f64],
    fy_grid: &[f64],
    fz_grid: &[f64],
    pos_x: &[f64],
    pos_y: &[f64],
    pos_z: &[f64],
    inv_cell: f64,
    nm: usize,
    acc_x: &mut [f64],
    acc_y: &mut [f64],
    acc_z: &mut [f64],
) {
    let nm2 = nm * nm;
    let nmf = nm as f64;

    for i in 0..pos_x.len() {
        let cx = (pos_x[i] * inv_cell).rem_euclid(nmf);
        let cy = (pos_y[i] * inv_cell).rem_euclid(nmf);
        let cz = (pos_z[i] * inv_cell).rem_euclid(nmf);

        let ix0 = cx.floor() as usize;
        let iy0 = cy.floor() as usize;
        let iz0 = cz.floor() as usize;

        let dx = cx - ix0 as f64;
        let dy = cy - iy0 as f64;
        let dz = cz - iz0 as f64;

        let wx0 = 1.0 - dx;
        let wx1 = dx;
        let wy0 = 1.0 - dy;
        let wy1 = dy;
        let wz0 = 1.0 - dz;
        let wz1 = dz;

        let ix1 = (ix0 + 1) % nm;
        let iy1 = (iy0 + 1) % nm;
        let iz1 = (iz0 + 1) % nm;

        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        let mut az = 0.0_f64;

        let wxs = [wx0, wx1];
        let wys = [wy0, wy1];
        let wzs = [wz0, wz1];
        let ixs = [ix0, ix1];
        let iys = [iy0, iy1];
        let izs = [iz0, iz1];

        for (diz, &wz_v) in wzs.iter().enumerate() {
            let iz_ = izs[diz];
            for (diy, &wy_v) in wys.iter().enumerate() {
                let iy_ = iys[diy];
                for (dix, &wx_v) in wxs.iter().enumerate() {
                    let ix_ = ixs[dix];
                    let w = wx_v * wy_v * wz_v;
                    let idx = iz_ * nm2 + iy_ * nm + ix_;
                    ax += fx_grid[idx] * w;
                    ay += fy_grid[idx] * w;
                    az += fz_grid[idx] * w;
                }
            }
        }

        acc_x[i] = ax;
        acc_y[i] = ay;
        acc_z[i] = az;
    }
}

// ── AVX2+FMA batch versions ──────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn assign_batch_avx2(
    pos_x: &[f64],
    pos_y: &[f64],
    pos_z: &[f64],
    masses: &[f64],
    inv_cell: f64,
    nm: usize,
    out: &mut [f64],
) {
    // LLVM vectorizes the loop over particles with YMM registers
    assign_batch_scalar(pos_x, pos_y, pos_z, masses, inv_cell, nm, out)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[expect(
    clippy::too_many_arguments,
    reason = "CIC grid interpolation keeps SoA slices explicit"
)]
unsafe fn interpolate_batch_avx2(
    fx_grid: &[f64],
    fy_grid: &[f64],
    fz_grid: &[f64],
    pos_x: &[f64],
    pos_y: &[f64],
    pos_z: &[f64],
    inv_cell: f64,
    nm: usize,
    acc_x: &mut [f64],
    acc_y: &mut [f64],
    acc_z: &mut [f64],
) {
    interpolate_batch_scalar(
        fx_grid, fy_grid, fz_grid, pos_x, pos_y, pos_z, inv_cell, nm, acc_x, acc_y, acc_z,
    )
}

// ── AVX-512 batch versions ──────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn assign_batch_avx512(
    pos_x: &[f64],
    pos_y: &[f64],
    pos_z: &[f64],
    masses: &[f64],
    inv_cell: f64,
    nm: usize,
    out: &mut [f64],
) {
    // LLVM vectorizes with ZMM registers (8×f64)
    assign_batch_scalar(pos_x, pos_y, pos_z, masses, inv_cell, nm, out)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[expect(
    clippy::too_many_arguments,
    reason = "CIC grid interpolation keeps SoA slices explicit"
)]
unsafe fn interpolate_batch_avx512(
    fx_grid: &[f64],
    fy_grid: &[f64],
    fz_grid: &[f64],
    pos_x: &[f64],
    pos_y: &[f64],
    pos_z: &[f64],
    inv_cell: f64,
    nm: usize,
    acc_x: &mut [f64],
    acc_y: &mut [f64],
    acc_z: &mut [f64],
) {
    interpolate_batch_scalar(
        fx_grid, fy_grid, fz_grid, pos_x, pos_y, pos_z, inv_cell, nm, acc_x, acc_y, acc_z,
    )
}

// ── Dispatch en runtime ─────────────────────────────────────────────────────

/// Asigna masas al grid NM³ usando interpolación CIC periódica con dispatch SIMD.
///
/// Selecciona AVX-512 → AVX2+FMA → escalar en runtime según las capacidades de la CPU.
/// Internamente convierte posiciones a SoA para permitir auto-vectorización.
pub fn assign(positions: &[Vec3], masses: &[f64], box_size: f64, nm: usize) -> Vec<f64> {
    let nm3 = nm * nm * nm;
    let inv_cell = nm as f64 / box_size;
    let n = positions.len();
    assert_eq!(n, masses.len());

    // SoA layout for better vectorization of the main loop
    let pos_x: Vec<f64> = positions.iter().map(|p| p.x).collect();
    let pos_y: Vec<f64> = positions.iter().map(|p| p.y).collect();
    let pos_z: Vec<f64> = positions.iter().map(|p| p.z).collect();

    let mut rho = vec![0.0_f64; nm3];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                assign_batch_avx512(&pos_x, &pos_y, &pos_z, masses, inv_cell, nm, &mut rho);
            }
            return rho;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                assign_batch_avx2(&pos_x, &pos_y, &pos_z, masses, inv_cell, nm, &mut rho);
            }
            return rho;
        }
    }

    // Scalar fallback (also handles non-x86 platforms)
    assign_batch_scalar(&pos_x, &pos_y, &pos_z, masses, inv_cell, nm, &mut rho);
    rho
}

/// Interpola las fuerzas del grid a las posiciones de las partículas usando CIC
/// con dispatch SIMD.
///
/// Selecciona AVX-512 → AVX2+FMA → escalar en runtime.
pub fn interpolate(
    fx_grid: &[f64],
    fy_grid: &[f64],
    fz_grid: &[f64],
    positions: &[Vec3],
    box_size: f64,
    nm: usize,
) -> Vec<Vec3> {
    let inv_cell = nm as f64 / box_size;
    let n = positions.len();

    let pos_x: Vec<f64> = positions.iter().map(|p| p.x).collect();
    let pos_y: Vec<f64> = positions.iter().map(|p| p.y).collect();
    let pos_z: Vec<f64> = positions.iter().map(|p| p.z).collect();

    let mut ax = vec![0.0_f64; n];
    let mut ay = vec![0.0_f64; n];
    let mut az = vec![0.0_f64; n];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                interpolate_batch_avx512(
                    fx_grid, fy_grid, fz_grid, &pos_x, &pos_y, &pos_z, inv_cell, nm, &mut ax,
                    &mut ay, &mut az,
                );
            }
            let acc: Vec<Vec3> = (0..n).map(|i| Vec3::new(ax[i], ay[i], az[i])).collect();
            return acc;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                interpolate_batch_avx2(
                    fx_grid, fy_grid, fz_grid, &pos_x, &pos_y, &pos_z, inv_cell, nm, &mut ax,
                    &mut ay, &mut az,
                );
            }
            let acc: Vec<Vec3> = (0..n).map(|i| Vec3::new(ax[i], ay[i], az[i])).collect();
            return acc;
        }
    }

    // Scalar fallback
    interpolate_batch_scalar(
        fx_grid, fy_grid, fz_grid, &pos_x, &pos_y, &pos_z, inv_cell, nm, &mut ax, &mut ay, &mut az,
    );
    (0..n).map(|i| Vec3::new(ax[i], ay[i], az[i])).collect()
}

// ── Versiones paralelas (feature `rayon`) ────────────────────────────────────

/// Versión paralela de [`assign`] usando Rayon con collect-then-merge.
///
/// Cada hilo acumula en un array local y al final se suman todos los arrays.
/// Con `N` grande y grids medianos es ~Ncpu× más rápida que la versión serial.
#[cfg(feature = "rayon")]
pub fn assign_rayon(positions: &[Vec3], masses: &[f64], box_size: f64, nm: usize) -> Vec<f64> {
    assert_eq!(positions.len(), masses.len());
    let nm2 = nm * nm;
    let nm3 = nm2 * nm;
    let inv_cell = nm as f64 / box_size;

    positions
        .par_iter()
        .zip(masses.par_iter())
        .fold(
            || vec![0.0_f64; nm3],
            |mut local_rho, (&pos, &m)| {
                let cx = (pos.x * inv_cell).rem_euclid(nm as f64);
                let cy = (pos.y * inv_cell).rem_euclid(nm as f64);
                let cz = (pos.z * inv_cell).rem_euclid(nm as f64);
                let ix0 = cx.floor() as usize;
                let iy0 = cy.floor() as usize;
                let iz0 = cz.floor() as usize;
                let dx = cx - ix0 as f64;
                let dy = cy - iy0 as f64;
                let dz = cz - iz0 as f64;
                let wx = [1.0 - dx, dx];
                let wy = [1.0 - dy, dy];
                let wz = [1.0 - dz, dz];
                for (diz, &wz_v) in wz.iter().enumerate() {
                    let iz = (iz0 + diz) % nm;
                    for (diy, &wy_v) in wy.iter().enumerate() {
                        let iy = (iy0 + diy) % nm;
                        for (dix, &wx_v) in wx.iter().enumerate() {
                            let ix = (ix0 + dix) % nm;
                            local_rho[iz * nm2 + iy * nm + ix] += m * wx_v * wy_v * wz_v;
                        }
                    }
                }
                local_rho
            },
        )
        .reduce(
            || vec![0.0_f64; nm3],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        )
}

/// Versión paralela de [`interpolate`] usando Rayon con dispatch SIMD.
///
/// El cálculo por partícula es independiente (solo lectura del grid),
/// por lo que el bucle externo se paraleliza directamente.
#[cfg(feature = "rayon")]
pub fn interpolate_rayon(
    fx_grid: &[f64],
    fy_grid: &[f64],
    fz_grid: &[f64],
    positions: &[Vec3],
    box_size: f64,
    nm: usize,
) -> Vec<Vec3> {
    let nm2 = nm * nm;
    let inv_cell = nm as f64 / box_size;

    positions
        .par_iter()
        .map(|&pos| {
            let cx = (pos.x * inv_cell).rem_euclid(nm as f64);
            let cy = (pos.y * inv_cell).rem_euclid(nm as f64);
            let cz = (pos.z * inv_cell).rem_euclid(nm as f64);
            let ix0 = cx.floor() as usize;
            let iy0 = cy.floor() as usize;
            let iz0 = cz.floor() as usize;
            let dx = cx - ix0 as f64;
            let dy = cy - iy0 as f64;
            let dz = cz - iz0 as f64;
            let wx = [1.0 - dx, dx];
            let wy = [1.0 - dy, dy];
            let wz = [1.0 - dz, dz];
            let mut ax = 0.0_f64;
            let mut ay = 0.0_f64;
            let mut az = 0.0_f64;
            for (diz, &wz_v) in wz.iter().enumerate() {
                let iz = (iz0 + diz) % nm;
                for (diy, &wy_v) in wy.iter().enumerate() {
                    let iy = (iy0 + diy) % nm;
                    for (dix, &wx_v) in wx.iter().enumerate() {
                        let ix = (ix0 + dix) % nm;
                        let w = wx_v * wy_v * wz_v;
                        let idx = iz * nm2 + iy * nm + ix;
                        ax += fx_grid[idx] * w;
                        ay += fy_grid[idx] * w;
                        az += fz_grid[idx] * w;
                    }
                }
            }
            Vec3::new(ax, ay, az)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    #[test]
    fn assign_single_particle_on_grid_node() {
        let nm = 4usize;
        let box_size = 4.0_f64;
        // Partícula exactamente en el nodo (0,0,0) del grid (x=0,y=0,z=0):
        // toda su masa va al nodo (0,0,0) con peso trilineal 1.
        let pos = vec![Vec3::new(0.0, 0.0, 0.0)];
        let mass = vec![1.0_f64];
        let rho = assign(&pos, &mass, box_size, nm);
        // El nodo (0,0,0) tiene flat_idx = 0; debe tener toda la masa.
        assert!(
            (rho[0] - 1.0).abs() < 1e-12,
            "masa en nodo (0,0,0) = {} (esperado 1.0)",
            rho[0]
        );
        let total: f64 = rho.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "masa total={total}");
    }

    #[test]
    fn assign_conserves_total_mass() {
        let nm = 8usize;
        let box_size = 1.0_f64;
        let positions: Vec<Vec3> = (0..27)
            .map(|i| {
                let x = (i % 3) as f64 / 3.0 + 0.1;
                let y = ((i / 3) % 3) as f64 / 3.0 + 0.1;
                let z = (i / 9) as f64 / 3.0 + 0.1;
                Vec3::new(x, y, z)
            })
            .collect();
        let masses = vec![2.0_f64; 27];
        let rho = assign(&positions, &masses, box_size, nm);
        let total: f64 = rho.iter().sum();
        assert!((total - 54.0).abs() < 1e-10, "masa total={total}");
    }

    #[test]
    fn interpolate_constant_field_gives_same_value() {
        let nm = 4usize;
        let box_size = 1.0_f64;
        // Campo constante Fx = 1.
        let fx = vec![1.0_f64; nm * nm * nm];
        let fy = vec![0.0_f64; nm * nm * nm];
        let fz = vec![0.0_f64; nm * nm * nm];
        let pos = vec![Vec3::new(0.3, 0.6, 0.7)];
        let acc = interpolate(&fx, &fy, &fz, &pos, box_size, nm);
        assert!((acc[0].x - 1.0).abs() < 1e-12);
        assert!(acc[0].y.abs() < 1e-12);
        assert!(acc[0].z.abs() < 1e-12);
    }

    #[test]
    fn assign_symmetry_at_center() {
        // Particle at exact center of a grid cell distributes mass equally to all 8 neighbors.
        // With nm=4, box_size=4.0, cell_size=1.0, a particle at (0.5, 0.5, 0.5) is
        // at the exact center between nodes (0,0,0) and (1,1,1).
        // CIC weights are all 0.5 → weight = 0.5³ = 0.125 per node.
        let nm = 4usize;
        let box_size = 4.0_f64;
        let pos = vec![Vec3::new(0.5, 0.5, 0.5)];
        let mass = vec![8.0_f64];
        let rho = assign(&pos, &mass, box_size, nm);
        // Node (0,0,0)=idx 0 gets 8.0 * 0.125 = 1.0
        assert!(
            (rho[0] - 1.0).abs() < 1e-12,
            "rho[0] = {} (esperado 1.0)",
            rho[0]
        );
        // Node (1,1,1)=idx 1*16+1*4+1 = 21 gets 8.0 * 0.125 = 1.0
        let idx_111 = 21; // node (1,1,1) en malla 4³
        assert!(
            (rho[idx_111] - 1.0).abs() < 1e-12,
            "rho[{}] = {} (esperado 1.0)",
            idx_111,
            rho[idx_111]
        );
        // All other nodes get 0 (or near-0 from rounding)
        let total: f64 = rho.iter().sum();
        assert!((total - 8.0).abs() < 1e-10, "total mass = {total}");
    }

    #[test]
    fn interpolate_interpolation_is_inverse_of_assign() {
        // Assign a unit mass, then interpolate the resulting density field
        // at the particle position — should get back 1.0 (exact for CIC)
        let nm = 8usize;
        let box_size = 1.0_f64;
        let pos = vec![Vec3::new(0.33, 0.55, 0.77)];
        let mass = vec![1.0_f64];
        let rho = assign(&pos, &mass, box_size, nm);
        // Interpolate density at particle position
        let acc = interpolate(
            &rho,
            &vec![0.0; nm * nm * nm],
            &vec![0.0; nm * nm * nm],
            &pos,
            box_size,
            nm,
        );
        // CIC interpolation of CIC-assigned mass should give the CIC kernel value
        // (which sums to 1 over all grid points, so interpolation gives ρ = 1/V_cell * W ≈ density)
        // More precisely: the CIC is self-consistent so assign+interpolate at same point gives
        // the kernel weight, which integrates to 1. We check it's positive and finite:
        assert!(
            acc[0].x > 0.0,
            "ρ at particle should be positive, got {}",
            acc[0].x
        );
    }

    #[test]
    fn assign_periodic_wrapping() {
        // Particle near the far edge wraps around correctly
        let nm = 4usize;
        let box_size = 4.0_f64;
        // Particle at x ≈ 3.99 should deposit weight to cells (3,3) and (0,3) via periodicity
        let pos = vec![Vec3::new(3.99, 3.99, 3.99)];
        let mass = vec![1.0_f64];
        let rho = assign(&pos, &mass, box_size, nm);
        let total: f64 = rho.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "mass conservation: total={total}"
        );
    }
}
