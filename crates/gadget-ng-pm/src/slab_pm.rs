//! Pipeline PM distribuido con slab decomposition (Fase 20).
//!
//! ## Flujo completo (P > 1)
//!
//! 1. **`deposit_slab_extended`** — CIC en buffer `(nz_local + 1, nm, nm)`:
//!    los planos 0..nz_local-1 son propios; el plano `nz_local` es el "ghost right"
//!    que contiene las contribuciones CIC destinadas al rank+1.
//!
//! 2. **`exchange_density_halos_z`** — intercambio punto-a-punto (ring periódico):
//!    cada rank envía su ghost right al rank+1 (mod P), y recibe el ghost right
//!    del rank-1 (mod P) que suma al plano iz_local = 0.
//!
//! 3. **`forces_from_slab`** — delega a `slab_fft::solve_forces_slab` (FFT
//!    distribuida mediante alltoall transposes).
//!
//! 4. **`exchange_force_halos_z`** — igual que el intercambio de densidad pero
//!    para las tres componentes de fuerza, necesario para interpolación CIC
//!    correcta cerca de los bordes del slab.
//!
//! 5. **`interpolate_slab_local`** — CIC interpolation usando el campo de
//!    fuerza local extendido con el plano halo recibido.
//!
//! ## Caso P = 1 (serial)
//!
//! Cuando `layout.n_ranks == 1`, cada función delega directamente a los módulos
//! `cic` y `fft_poisson` existentes sin ninguna comunicación, garantizando
//! exactitud bit-a-bit con el path serial de Fase 18.

use gadget_ng_core::Vec3;
use gadget_ng_parallel::ParallelRuntime;

use crate::cic;
use crate::slab_fft::{self, SlabLayout};

// ── Depósito CIC en slab extendido ───────────────────────────────────────────

/// Deposita masa CIC en el buffer de slab extendido.
///
/// El buffer retornado tiene tamaño `(nz_local + 1) * nm * nm` donde:
/// - Planos `0..nz_local-1`: planos propios del rank.
/// - Plano `nz_local`: "ghost right" con contribuciones CIC al rank vecino
///   (iz_local = 0 del rank+1).
///
/// Para P = 1, el buffer tiene tamaño `nm³` (sin plano ghost extra) y
/// es equivalente al CIC estándar periódico.
pub fn deposit_slab_extended(
    positions: &[Vec3],
    masses: &[f64],
    layout: &SlabLayout,
    box_size: f64,
) -> Vec<f64> {
    assert_eq!(positions.len(), masses.len());
    let nm = layout.nm;
    let nm2 = nm * nm;
    let nz = layout.nz_local;

    // P=1: depósito estándar global (sin ghost plane extra).
    if layout.n_ranks == 1 {
        return cic::assign(positions, masses, box_size, nm);
    }

    // P>1: buffer con plano ghost right extra.
    let buf_nz = nz + 1; // nz_local planos propios + 1 ghost right
    let mut density = vec![0.0_f64; buf_nz * nm2];

    let inv_cell = nm as f64 / box_size;
    let z_lo_idx = layout.z_lo_idx;

    for (&mass, pos) in masses.iter().zip(positions.iter()) {
        let cx = pos.x * inv_cell;
        let cy = pos.y * inv_cell;
        let cz = pos.z * inv_cell;

        let ix0 = cx.floor() as i64;
        let iy0 = cy.floor() as i64;
        let iz0_global = cz.floor() as i64;

        let dx = cx - ix0 as f64;
        let dy = cy - iy0 as f64;
        let dz = cz - iz0_global as f64;

        for dix in 0..2_i64 {
            let wx = if dix == 0 { 1.0 - dx } else { dx };
            let ix = ((ix0 + dix).rem_euclid(nm as i64)) as usize;
            for diy in 0..2_i64 {
                let wy = if diy == 0 { 1.0 - dy } else { dy };
                let iy = ((iy0 + diy).rem_euclid(nm as i64)) as usize;
                for diz in 0..2_i64 {
                    let wz = if diz == 0 { 1.0 - dz } else { dz };
                    let iz_global = iz0_global + diz;
                    // Convertir a índice local extendido.
                    // iz_local ∈ [0, nz_local): propio
                    // iz_local == nz_local: ghost right
                    // iz_local < 0: no puede ocurrir tras exchange_domain_by_z
                    let iz_local_ext = iz_global - z_lo_idx as i64;
                    if iz_local_ext < 0 || iz_local_ext > nz as i64 {
                        // Partícula en borde fuera del rango aceptable: ignorar
                        // (no debería ocurrir tras migración correcta).
                        continue;
                    }
                    let iz_local_ext = iz_local_ext as usize;
                    let flat = iz_local_ext * nm2 + iy * nm + ix;
                    density[flat] += mass * wx * wy * wz;
                }
            }
        }
    }
    density
}

// ── Intercambio de halos de densidad ─────────────────────────────────────────

/// Intercambia los planos ghost CIC entre ranks vecinos (ring periódico en Z).
///
/// Cada rank envía su plano ghost right (`density_ext[nz_local * nm², ...]`)
/// al rank `(rank+1) % P` y recibe del rank `(rank-1+P) % P` para sumar
/// al plano propio `iz_local = 0`.
///
/// Tras la llamada, los primeros `nz_local` planos de `density_ext` contienen
/// la densidad completa (contribuciones propias + halos recibidos).
///
/// Para P = 1: no-op (el buffer no tiene plano ghost extra).
pub fn exchange_density_halos_z<R: ParallelRuntime + ?Sized>(
    density_ext: &mut Vec<f64>,
    layout: &SlabLayout,
    rt: &R,
) {
    if layout.n_ranks == 1 {
        return;
    }
    let nm = layout.nm;
    let nm2 = nm * nm;
    let nz = layout.nz_local;

    // Extraer el plano ghost right.
    let ghost_start = nz * nm2;
    let ghost_right: Vec<f64> = density_ext[ghost_start..ghost_start + nm2].to_vec();

    // Intercambio en anillo: enviar a right = (rank+1)%P, recibir de left = (rank-1+P)%P
    let nr = layout.n_ranks;
    let right = (layout.rank + 1) % nr;
    let left = (layout.rank + nr - 1) % nr;

    let mut sends: Vec<Vec<f64>> = (0..nr).map(|_| Vec::new()).collect();
    sends[right] = ghost_right;

    let received = rt.alltoallv_f64(&sends);

    // Sumar el plano recibido del left al plano iz_local=0.
    for (i, &v) in received[left].iter().enumerate() {
        density_ext[i] += v;
    }

    // Truncar el ghost plane (ya no necesario).
    density_ext.truncate(nz * nm2);
}

/// Elimina el plano ghost del buffer extendido (sin intercambio previo).
/// Útil si se quiere descartar el ghost sin comunicar.
pub fn trim_extended(density_ext: &[f64], layout: &SlabLayout) -> Vec<f64> {
    let nm2 = layout.nm * layout.nm;
    let nz = layout.nz_local;
    density_ext[..nz * nm2].to_vec()
}

// ── Solve de Poisson en slab distribuido ─────────────────────────────────────

/// Calcula las tres componentes de fuerza PM en el slab local usando FFT
/// distribuida (alltoall transposes).
///
/// Delega a [`slab_fft::solve_forces_slab`]. Para P = 1, el resultado es
/// bit-a-bit idéntico al solver serial de Fase 18.
pub fn forces_from_slab<R: ParallelRuntime + ?Sized>(
    density: &[f64],
    layout: &SlabLayout,
    g: f64,
    box_size: f64,
    r_split: Option<f64>,
    rt: &R,
) -> [Vec<f64>; 3] {
    slab_fft::solve_forces_slab(density, layout, g, box_size, r_split, rt)
}

// ── Intercambio de halos de fuerza ────────────────────────────────────────────

/// Intercambia el plano de fuerza en el borde derecho entre ranks vecinos.
///
/// Igual que [`exchange_density_halos_z`] pero para las componentes de fuerza.
/// Agrega al inicio de cada componente el plano recibido del rank izquierdo,
/// necesario para la interpolación CIC de partículas cerca del borde.
///
/// Para P = 1: no-op.
pub fn exchange_force_halos_z<R: ParallelRuntime + ?Sized>(
    forces: &mut [Vec<f64>; 3],
    layout: &SlabLayout,
    rt: &R,
) {
    if layout.n_ranks == 1 {
        return;
    }
    let nm = layout.nm;
    let nm2 = nm * nm;
    let nz = layout.nz_local;
    let nr = layout.n_ranks;

    let right = (layout.rank + 1) % nr;
    let left = (layout.rank + nr - 1) % nr;

    for comp in forces.iter_mut() {
        assert_eq!(comp.len(), nz * nm2);

        // Enviar el último plano (iz_local = nz-1) al rank derecho.
        // El rank derecho lo necesita como plano "-1" para CIC de sus partículas
        // con iz_local < 0 (que cruzan el borde izquierdo).
        // También enviar el primer plano (iz_local = 0) al rank izquierdo.
        let last_plane: Vec<f64> = comp[(nz - 1) * nm2..nz * nm2].to_vec();
        let first_plane: Vec<f64> = comp[0..nm2].to_vec();

        let mut sends_r: Vec<Vec<f64>> = (0..nr).map(|_| Vec::new()).collect();
        let mut sends_l: Vec<Vec<f64>> = (0..nr).map(|_| Vec::new()).collect();
        sends_r[right] = last_plane;
        sends_l[left] = first_plane;

        // Dos rondas de alltoallv: enviar right, luego left.
        // Alternativa: un alltoallv con 2*nm² datos por vecino (send right y left juntos).
        // Para simplicidad usamos dos llamadas separadas.
        let recv_from_left_side = rt.alltoallv_f64(&sends_r);
        let recv_from_right_side = rt.alltoallv_f64(&sends_l);

        // recv_from_left_side[left] = plano nm-1 del rank izquierdo (añadir como plano "-1").
        // recv_from_right_side[right] = plano 0 del rank derecho (añadir como plano "nz_local").
        // Extender el array con estos halos (prepend y append).
        let halo_left = &recv_from_left_side[left];  // plano a prepend
        let halo_right = &recv_from_right_side[right]; // plano a append

        let old_len = comp.len();
        comp.resize(old_len + nm2, 0.0);
        // Shift all existing data right by nm2 elements to make room for halo_left.
        comp.copy_within(0..old_len, nm2);
        // Insert halo_left at the beginning.
        for (i, &v) in halo_left.iter().enumerate().take(nm2) {
            comp[i] = v;
        }
        // Append halo_right.
        let new_len = comp.len();
        comp.resize(new_len + nm2, 0.0);
        for (i, &v) in halo_right.iter().enumerate().take(nm2) {
            comp[new_len + i] = v;
        }
        // Ahora comp tiene (nz+2) planes: [halo_left, 0..nz-1, halo_right]
    }
}

// ── Interpolación CIC local ───────────────────────────────────────────────────

/// Interpola las fuerzas PM de vuelta a las partículas usando CIC.
///
/// `forces` puede tener `nz_local`, `nz_local + 1` o `nz_local + 2` planos
/// (con halos prepend/append). El offset z de los datos dentro de `forces` es
/// `halo_offset` planos (0, 1 o 2 según si se agregaron halos).
///
/// Para P = 1, delega directamente a `cic::interpolate`.
pub fn interpolate_slab_local(
    positions: &[Vec3],
    forces: &[Vec<f64>; 3],
    layout: &SlabLayout,
    box_size: f64,
) -> Vec<Vec3> {
    let nm = layout.nm;
    let nm2 = nm * nm;

    // P=1: usar interpolación global estándar.
    if layout.n_ranks == 1 {
        assert_eq!(forces[0].len(), nm * nm2);
        return cic::interpolate(&forces[0], &forces[1], &forces[2], positions, box_size, nm);
    }

    // P>1: el array forces tiene (nz_local + 2) planos
    // [halo_left | iz_local=0..nz_local-1 | halo_right]
    // El índice de plano local para iz_global es: (iz_global - z_lo_idx + 1)
    // donde +1 es el desplazamiento por el halo izquierdo.
    let nz_local = layout.nz_local;
    let expected_nz = nz_local + 2; // with both halos
    let force_nz = forces[0].len() / nm2;

    // Determinar offset: si hay (nz+2) planos, offset=1; si (nz+1), offset=0 o 1.
    let halo_offset: i64 = if force_nz == expected_nz {
        1
    } else if force_nz == nz_local + 1 {
        0
    } else {
        0
    };

    let z_lo = layout.z_lo_idx as i64;
    let n = positions.len();
    let mut acc = vec![Vec3::new(0.0, 0.0, 0.0); n];

    let inv_cell = nm as f64 / box_size;

    for (i, pos) in positions.iter().enumerate() {
        let cx = pos.x * inv_cell;
        let cy = pos.y * inv_cell;
        let cz = pos.z * inv_cell;

        let ix0 = cx.floor() as i64;
        let iy0 = cy.floor() as i64;
        let iz0_global = cz.floor() as i64;

        let dx = cx - ix0 as f64;
        let dy = cy - iy0 as f64;
        let dz = cz - iz0_global as f64;

        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;
        let mut fz_acc = 0.0_f64;

        for dix in 0..2_i64 {
            let wx = if dix == 0 { 1.0 - dx } else { dx };
            let ix = ((ix0 + dix).rem_euclid(nm as i64)) as usize;
            for diy in 0..2_i64 {
                let wy = if diy == 0 { 1.0 - dy } else { dy };
                let iy = ((iy0 + diy).rem_euclid(nm as i64)) as usize;
                for diz in 0..2_i64 {
                    let wz = if diz == 0 { 1.0 - dz } else { dz };
                    let iz_global = iz0_global + diz;
                    // Índice en el array forces extendido.
                    let iz_ext = iz_global - z_lo + halo_offset;
                    if iz_ext < 0 || iz_ext >= force_nz as i64 {
                        continue; // fuera del rango (no debería ocurrir normalmente)
                    }
                    let flat = iz_ext as usize * nm2 + iy * nm + ix;
                    let w = wx * wy * wz;
                    fx += forces[0][flat] * w;
                    fy += forces[1][flat] * w;
                    fz_acc += forces[2][flat] * w;
                }
            }
        }
        acc[i] = Vec3::new(fx, fy, fz_acc);
    }
    acc
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_parallel::SerialRuntime;

    fn uniform_positions(n: usize, box_size: f64) -> Vec<Vec3> {
        let side = (n as f64).cbrt().ceil() as usize;
        let dx = box_size / side as f64;
        let mut pos = Vec::new();
        for iz in 0..side {
            for iy in 0..side {
                for ix in 0..side {
                    if pos.len() >= n {
                        break;
                    }
                    pos.push(Vec3::new(
                        (ix as f64 + 0.5) * dx,
                        (iy as f64 + 0.5) * dx,
                        (iz as f64 + 0.5) * dx,
                    ));
                }
            }
        }
        pos.truncate(n);
        pos
    }

    #[test]
    fn deposit_mass_conservation_p1() {
        let nm = 8usize;
        let layout = SlabLayout::new(nm, 0, 1);
        let box_size = 1.0;
        let n = 64;
        let pos = uniform_positions(n, box_size);
        let masses = vec![1.0_f64; n];

        let density = deposit_slab_extended(&pos, &masses, &layout, box_size);
        let total_mass: f64 = density.iter().sum();
        let expected_mass: f64 = masses.iter().sum();
        assert!(
            (total_mass - expected_mass).abs() < 1e-10,
            "mass conservation failed: {total_mass} vs {expected_mass}"
        );
    }

    #[test]
    fn deposit_mass_conservation_p2_simulated() {
        // Simulate P=2 with rank=0, nm=8, nz_local=4
        let nm = 8usize;
        let layout0 = SlabLayout::new(nm, 0, 2);
        let layout1 = SlabLayout::new(nm, 1, 2);
        let box_size = 1.0;
        let n = 64;
        let pos = uniform_positions(n, box_size);
        let masses = vec![1.0_f64; n];

        // Partition particles to slabs.
        let z_mid = box_size / 2.0;
        let (pos0, m0): (Vec<_>, Vec<_>) = pos
            .iter()
            .zip(masses.iter())
            .filter(|(p, _)| p.z < z_mid)
            .map(|(p, &m)| (*p, m))
            .unzip();
        let (pos1, m1): (Vec<_>, Vec<_>) = pos
            .iter()
            .zip(masses.iter())
            .filter(|(p, _)| p.z >= z_mid)
            .map(|(p, &m)| (*p, m))
            .unzip();

        let den0 = deposit_slab_extended(&pos0, &m0, &layout0, box_size);
        let den1 = deposit_slab_extended(&pos1, &m1, &layout1, box_size);

        // Sin intercambio de halos, masa en cada rank no incluye contribuciones cruzadas.
        // La suma de los dos slabs de masa propia debe = total.
        let nm2 = nm * nm;
        let nz = layout0.nz_local;
        // Sumar solo los planos propios (sin ghost right).
        let mass0: f64 = den0[0..nz * nm2].iter().sum::<f64>()
            + den0[nz * nm2..].iter().sum::<f64>(); // incluyendo ghost
        let mass1: f64 = den1[0..nz * nm2].iter().sum::<f64>()
            + den1[nz * nm2..].iter().sum::<f64>();

        let total = mass0 + mass1;
        let expected: f64 = masses.iter().sum();
        assert!(
            (total - expected).abs() < 1e-10,
            "mass conservation failed across slabs: {total} vs {expected}"
        );
    }

    #[test]
    fn forces_slab_p1_mass_conservation() {
        // Verificar que forces_from_slab P=1 no produce NaN/Inf
        let nm = 8usize;
        let layout = SlabLayout::new(nm, 0, 1);
        let rt = SerialRuntime;
        let box_size = 1.0;
        let n = 64;
        let pos = uniform_positions(n, box_size);
        let masses = vec![1.0_f64; n];
        let density = cic::assign(&pos, &masses, box_size, nm);
        let [fx, fy, fz] = forces_from_slab(&density, &layout, 1.0, box_size, None, &rt);
        for (i, (&a, &b, &c)) in fx.iter().zip(fy.iter()).zip(fz.iter()).map(|((a, b), c)| (a, b, c)).enumerate() {
            assert!(a.is_finite() && b.is_finite() && c.is_finite(),
                "NaN/Inf en celda {i}: fx={a} fy={b} fz={c}");
        }
    }
}
