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

use gadget_ng_core::Vec3;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Asigna masas al grid NM³ usando interpolación CIC periódica.
///
/// - `positions` — posiciones de todas las partículas (coordenadas en `[0, box_size)`).
/// - `masses` — masas correspondientes.
/// - `box_size` — longitud del cubo periódico.
/// - `nm` — número de celdas por lado.
///
/// Devuelve el array de densidad plano de longitud `nm³`. Las unidades son
/// masa / celda (no masa / volumen); el solver Poisson añade el factor de
/// volumen de celda cuando construye el Green's function.
pub fn assign(positions: &[Vec3], masses: &[f64], box_size: f64, nm: usize) -> Vec<f64> {
    assert_eq!(positions.len(), masses.len());
    let nm2 = nm * nm;
    let nm3 = nm2 * nm;
    let mut rho = vec![0.0_f64; nm3];

    let inv_cell = nm as f64 / box_size; // conversión posición → unidades de celda

    for (&pos, &m) in positions.iter().zip(masses.iter()) {
        // Coordenada en unidades de celda (periódica).
        let cx = (pos.x * inv_cell).rem_euclid(nm as f64);
        let cy = (pos.y * inv_cell).rem_euclid(nm as f64);
        let cz = (pos.z * inv_cell).rem_euclid(nm as f64);

        // Celda base (entero por debajo).
        let ix0 = cx.floor() as usize;
        let iy0 = cy.floor() as usize;
        let iz0 = cz.floor() as usize;

        // Desplazamiento dentro de la celda [0, 1).
        let dx = cx - ix0 as f64;
        let dy = cy - iy0 as f64;
        let dz = cz - iz0 as f64;

        // Pesos CIC: (1-d) para la celda base, d para la siguiente.
        let wx = [1.0 - dx, dx];
        let wy = [1.0 - dy, dy];
        let wz = [1.0 - dz, dz];

        // Distribuir masa a las 8 celdas vecinas con condiciones periódicas.
        for (diz, &wz_v) in wz.iter().enumerate() {
            let iz = (iz0 + diz) % nm;
            for (diy, &wy_v) in wy.iter().enumerate() {
                let iy = (iy0 + diy) % nm;
                for (dix, &wx_v) in wx.iter().enumerate() {
                    let ix = (ix0 + dix) % nm;
                    rho[iz * nm2 + iy * nm + ix] += m * wx_v * wy_v * wz_v;
                }
            }
        }
    }

    rho
}

/// Interpola las fuerzas del grid a las posiciones de las partículas usando CIC.
///
/// - `fx_grid`, `fy_grid`, `fz_grid` — componentes de fuerza en el grid (planos, longitud `nm³`).
/// - `positions` — posiciones de las partículas.
/// - `box_size` — longitud del cubo periódico.
/// - `nm` — número de celdas por lado.
///
/// Devuelve las aceleraciones interpoladas (una por partícula). Se usa la misma
/// interpolación bilineal que en `assign` para garantizar la reciprocidad acción-reacción.
pub fn interpolate(
    fx_grid: &[f64],
    fy_grid: &[f64],
    fz_grid: &[f64],
    positions: &[Vec3],
    box_size: f64,
    nm: usize,
) -> Vec<Vec3> {
    let nm2 = nm * nm;
    let inv_cell = nm as f64 / box_size;
    let mut acc = vec![Vec3::zero(); positions.len()];

    for (i, &pos) in positions.iter().enumerate() {
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

        acc[i] = Vec3::new(ax, ay, az);
    }

    acc
}

// ── Versiones paralelas (feature `rayon`) ────────────────────────────────────

/// Versión paralela de [`assign`] usando Rayon.
///
/// Cada hilo acumula en un array local y al final se suman todos.
/// Con `N` grande y grids medianos es ~Ncpu× más rápida que la versión serial.
/// Con la feature `rayon` desactivada, `assign` ya es el punto de entrada.
#[cfg(feature = "rayon")]
pub fn assign_rayon(positions: &[Vec3], masses: &[f64], box_size: f64, nm: usize) -> Vec<f64> {
    assert_eq!(positions.len(), masses.len());
    let nm2 = nm * nm;
    let nm3 = nm2 * nm;
    let inv_cell = nm as f64 / box_size;

    // Cada hilo acumula en un rho local; luego sumamos todos los arrays.
    let rho = positions
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
        );
    rho
}

/// Versión paralela de [`interpolate`] usando Rayon.
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
}
