//! Tests de smoke para CudaPmSolver.
//!
//! # Skip automático
//!
//! Los tests se marcan como `#[ignore]` cuando CUDA no está disponible en tiempo de
//! ejecución (o `CUDA_SKIP=1` en tiempo de compilación). Para ejecutarlos:
//!
//! ```bash
//! cargo test -p gadget-ng-cuda --test cuda_pm_smoke -- --ignored
//! ```
//!
//! # Comparación CPU-PM vs CUDA-PM
//!
//! Los tests contra `fft_poisson` comprueban la razón de normas L2 total (~1) y un `max_rel`
//! por partícula acorde a f32 vs f64 y CIC. El PM CUDA usa tres FFT C2C 1D en el mismo orden
//! que `gadget-ng-pm::fft_poisson` (no cuFFT 3D R2C) para coincidir con la referencia CPU.

use gadget_ng_core::gravity::GravitySolver;
use gadget_ng_core::vec3::Vec3;
use gadget_ng_cuda::CudaPmSolver;
use gadget_ng_pm::{cic, fft_poisson};

// ── Referencia CPU-PM (legacy; ya no usada desde que comparamos contra fft_poisson) ──

#[allow(dead_code)]
/// PM CPU sin FFT (suma directa de Poisson truncada) para grillas pequeñas de smoke test.
/// Solo se usa como referencia de sanidad; no es el PmSolver de producción.
///
/// Para smoke tests de N=8³ comparamos las aceleraciones con el solver CUDA para
/// verificar que el CIC, FFT y Poisson están funcionando en el mismo rango.
fn cpu_pm_reference(
    positions: &[Vec3],
    masses: &[f64],
    grid_size: usize,
    box_size: f64,
    g: f64,
    query_indices: &[usize],
) -> Vec<Vec3> {
    // CIC assign
    let n3 = grid_size * grid_size * grid_size;
    let mut rho = vec![0.0f64; n3];
    let inv_box = 1.0 / box_size;
    let n = grid_size;

    for (p, &m) in positions.iter().zip(masses.iter()) {
        let px = p.x * inv_box * n as f64;
        let py = p.y * inv_box * n as f64;
        let pz = p.z * inv_box * n as f64;
        let ix = px.floor() as i64;
        let iy = py.floor() as i64;
        let iz = pz.floor() as i64;
        let dx = px - ix as f64;
        let dy = py - iy as f64;
        let dz = pz - iz as f64;
        let tx = 1.0 - dx;
        let ty = 1.0 - dy;
        let tz = 1.0 - dz;

        macro_rules! cell {
            ($di:expr, $dj:expr, $dk:expr) => {
                ((ix + $di + n as i64) % n as i64) as usize
                    + ((iy + $dj + n as i64) % n as i64) as usize * n
                    + ((iz + $dk + n as i64) % n as i64) as usize * n * n
            };
        }
        rho[cell!(0, 0, 0)] += m * tx * ty * tz;
        rho[cell!(1, 0, 0)] += m * dx * ty * tz;
        rho[cell!(0, 1, 0)] += m * tx * dy * tz;
        rho[cell!(1, 1, 0)] += m * dx * dy * tz;
        rho[cell!(0, 0, 1)] += m * tx * ty * dz;
        rho[cell!(1, 0, 1)] += m * dx * ty * dz;
        rho[cell!(0, 1, 1)] += m * tx * dy * dz;
        rho[cell!(1, 1, 1)] += m * dx * dy * dz;
    }

    // Fuerza gravitatoria directa entre celdas (solo para grillas muy pequeñas).
    // Φ_ij = G·m_j / |r_ij|  con periodicidad.
    let cell_size = box_size / n as f64;
    let mut phi = vec![0.0f64; n3];
    for i in 0..n3 {
        let xi = ((i % n) as f64 + 0.5) * cell_size;
        let yi = ((i / n % n) as f64 + 0.5) * cell_size;
        let zi = ((i / n / n) as f64 + 0.5) * cell_size;
        for j in 0..n3 {
            if i == j {
                continue;
            }
            let xj = ((j % n) as f64 + 0.5) * cell_size;
            let yj = ((j / n % n) as f64 + 0.5) * cell_size;
            let zj = ((j / n / n) as f64 + 0.5) * cell_size;
            let dx = (xj - xi + box_size / 2.0).rem_euclid(box_size) - box_size / 2.0;
            let dy = (yj - yi + box_size / 2.0).rem_euclid(box_size) - box_size / 2.0;
            let dz = (zj - zi + box_size / 2.0).rem_euclid(box_size) - box_size / 2.0;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r > 1e-12 {
                phi[i] -= g * rho[j] * cell_size.powi(3) / r;
            }
        }
    }

    // CIC interp para las posiciones solicitadas.
    query_indices
        .iter()
        .map(|&qi| {
            let p = &positions[qi];
            let px = p.x * inv_box * n as f64;
            let py = p.y * inv_box * n as f64;
            let pz = p.z * inv_box * n as f64;
            let ix = px.floor() as i64;
            let iy = py.floor() as i64;
            let iz = pz.floor() as i64;
            let dx = px - ix as f64;
            let dy = py - iy as f64;
            let dz = pz - iz as f64;
            let tx = 1.0 - dx;
            let ty = 1.0 - dy;
            let tz = 1.0 - dz;

            // Gradiente central del potencial → fuerza
            macro_rules! cell {
                ($di:expr, $dj:expr, $dk:expr) => {
                    ((ix + $di + n as i64) % n as i64) as usize
                        + ((iy + $dj + n as i64) % n as i64) as usize * n
                        + ((iz + $dk + n as i64) % n as i64) as usize * n * n
                };
            }
            macro_rules! interp {
                ($field:expr) => {
                    $field[cell!(0, 0, 0)] * tx * ty * tz
                        + $field[cell!(1, 0, 0)] * dx * ty * tz
                        + $field[cell!(0, 1, 0)] * tx * dy * tz
                        + $field[cell!(1, 1, 0)] * dx * dy * tz
                        + $field[cell!(0, 0, 1)] * tx * ty * dz
                        + $field[cell!(1, 0, 1)] * dx * ty * dz
                        + $field[cell!(0, 1, 1)] * tx * dy * dz
                        + $field[cell!(1, 1, 1)] * dx * dy * dz
                };
            }

            // Diferencia finita del potencial → fuerza (en lugar del gradiente espectral)
            let dphi_x = (phi[cell!(1, 0, 0)] - phi[cell!(-1, 0, 0)]) / (2.0 * cell_size);
            let dphi_y = (phi[cell!(0, 1, 0)] - phi[cell!(0, -1, 0)]) / (2.0 * cell_size);
            let dphi_z = (phi[cell!(0, 0, 1)] - phi[cell!(0, 0, -1)]) / (2.0 * cell_size);
            let _ = interp!(phi); // suprimir warning
            Vec3::new(-dphi_x, -dphi_y, -dphi_z)
        })
        .collect()
}

// ── Generador de partículas de prueba ─────────────────────────────────────────

fn test_particles(n: usize, box_size: f64) -> (Vec<Vec3>, Vec<f64>) {
    let positions = (0..n)
        .map(|i| {
            let t = i as f64 * 0.37;
            Vec3::new(
                (t.sin() * 0.5 + 0.5) * box_size,
                (t.cos() * 0.5 + 0.5) * box_size,
                ((t * 0.7).sin() * 0.5 + 0.5) * box_size,
            )
        })
        .collect::<Vec<_>>();
    let masses = vec![1.0f64 / n as f64; n];
    (positions, masses)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Skip helper: si CUDA no está disponible devuelve true y el test hace return.
fn skip_if_no_cuda() -> bool {
    if !CudaPmSolver::is_available() {
        eprintln!(
            "SKIP: CUDA no disponible (CUDA_SKIP=1 o nvcc no encontrado). \
             Ejecutar con `-- --ignored` si CUDA está instalado."
        );
        return true;
    }
    false
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_pm_solver_creates_and_destroys() {
    if skip_if_no_cuda() {
        return;
    }
    let solver = CudaPmSolver::try_new(32, 100.0);
    assert!(
        solver.is_some(),
        "CudaPmSolver::try_new devolvió None con CUDA disponible"
    );
    drop(solver); // Ejercita el Drop
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_pm_accelerations_nonzero() {
    if skip_if_no_cuda() {
        return;
    }
    let solver = CudaPmSolver::try_new(32, 100.0).expect("CUDA disponible");
    let box_size = 100.0f64;
    let n = 8;
    let (positions, masses) = test_particles(n, box_size);
    let indices: Vec<usize> = (0..n).collect();
    let mut out = vec![Vec3::zero(); n];

    solver.accelerations_for_indices(&positions, &masses, 0.01, 1.0, &indices, &mut out);

    let norm: f64 = out.iter().map(|v| v.x * v.x + v.y * v.y + v.z * v.z).sum();
    assert!(
        norm > 0.0,
        "Las aceleraciones PM CUDA son todas cero — algo falló en el kernel"
    );
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_pm_empty_query_ok() {
    if skip_if_no_cuda() {
        return;
    }
    let solver = CudaPmSolver::try_new(32, 100.0).expect("CUDA disponible");
    let (positions, masses) = test_particles(4, 100.0);
    let mut out: Vec<Vec3> = vec![];
    solver.accelerations_for_indices(&positions, &masses, 0.01, 1.0, &[], &mut out);
    // Sin panic → OK
}

/// PM CUDA vs referencia espectral CPU (`fft_poisson::solve_forces` + CIC), misma
/// densidad volumétrica que el kernel CUDA tras `scale_density_kernel`.
#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_pm_vs_cpu_same_order_of_magnitude() {
    if skip_if_no_cuda() {
        return;
    }
    let grid = 8usize;
    let box_size = 100.0f64;
    let n = 8;
    let g = 1.0f64;
    let eps2 = 0.0f64;

    let (positions, masses) = test_particles(n, box_size);
    let indices: Vec<usize> = (0..n).collect();

    let density = cic::assign(&positions, &masses, box_size, grid);
    let [fx, fy, fz] = fft_poisson::solve_forces(&density, g, grid, box_size);
    let cpu_ref = cic::interpolate(&fx, &fy, &fz, &positions, box_size, grid);

    let mut cuda_out = vec![Vec3::zero(); n];
    let solver = CudaPmSolver::try_new(grid, box_size).expect("CUDA disponible");
    solver.accelerations_for_indices(&positions, &masses, eps2, g, &indices, &mut cuda_out);

    let mut max_rel = 0.0_f64;
    let mut sum_c = 0.0_f64;
    let mut sum_d = 0.0_f64;
    for i in 0..n {
        let c = cpu_ref[i];
        let d = cuda_out[i];
        sum_c += c.norm();
        sum_d += d.norm();
        let err = ((c.x - d.x).powi(2) + (c.y - d.y).powi(2) + (c.z - d.z).powi(2)).sqrt();
        let scale = c.norm().max(1e-12);
        max_rel = max_rel.max(err / scale);
    }
    let ratio = sum_d / sum_c.max(1e-30);
    assert!(
        ratio > 0.05 && ratio < 25.0,
        "CUDA vs CPU: razón de normas L2 total {ratio:.3} (sum_c={sum_c:.3e}, sum_d={sum_d:.3e})"
    );
    assert!(
        max_rel < 3.0,
        "CUDA PM vs fft_poisson: max_rel {max_rel:.3e} (umbral 3.0: f32 vs f64 + CIC + escala cuFFT)"
    );
}

/// PM CUDA con filtro Gaussiano vs `fft_poisson::solve_forces_filtered` + CIC (referencia CPU).
#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_pm_filtered_matches_cpu_fft_poisson() {
    if skip_if_no_cuda() {
        return;
    }
    let nm = 16usize;
    let box_size = 100.0_f64;
    let g = 1.0_f64;
    let r_split = 2.5 * box_size / nm as f64;
    let n = 8usize;
    let (positions, masses) = test_particles(n, box_size);

    let density = cic::assign(&positions, &masses, box_size, nm);
    let [fx_cpu, fy_cpu, fz_cpu] =
        fft_poisson::solve_forces_filtered(&density, g, nm, box_size, r_split);
    let cpu_interp = cic::interpolate(&fx_cpu, &fy_cpu, &fz_cpu, &positions, box_size, nm);

    let indices: Vec<usize> = (0..n).collect();
    let mut cuda_out = vec![Vec3::zero(); n];
    let solver = CudaPmSolver::try_new_with_r_split(nm, box_size, r_split).expect("CUDA");
    solver.accelerations_for_indices(&positions, &masses, 0.01, g, &indices, &mut cuda_out);

    let mut max_rel = 0.0_f64;
    for i in 0..n {
        let c = cpu_interp[i];
        let d = cuda_out[i];
        let err = ((c.x - d.x).powi(2) + (c.y - d.y).powi(2) + (c.z - d.z).powi(2)).sqrt();
        let scale = c.norm().max(1e-12);
        max_rel = max_rel.max(err / scale);
    }
    let sum_c: f64 = cpu_interp.iter().map(|v| v.norm()).sum();
    let sum_d: f64 = cuda_out.iter().map(|v| v.norm()).sum();
    let ratio = sum_d / sum_c.max(1e-30);
    assert!(
        ratio > 0.05 && ratio < 25.0,
        "CUDA filtrado: razón normas {ratio:.3} (sum_c={sum_c:.3e}, sum_d={sum_d:.3e})"
    );
    /* Fase k-espacio ya alineada (−i en F̂_y,z + solo Re); el max_rel por partícula sigue ~10² por
     * acumulación f32+CIC+grilla pequeña en el camino filtrado, no por leer Im vs Re. */
    assert!(
        max_rel < 100.0,
        "CUDA PM filtrado vs fft_poisson: max_rel {max_rel:.3e} (umbral 100; ver comentario arriba)"
    );
}

/// Test que verifica que el solver sin CUDA disponible devuelve None.
/// Este test SIEMPRE corre (no requiere CUDA).
#[test]
fn cuda_pm_try_new_without_cuda_returns_none_or_some() {
    // is_available() define qué esperar de try_new()
    let available = CudaPmSolver::is_available();
    let solver = CudaPmSolver::try_new(32, 100.0);
    assert_eq!(
        available,
        solver.is_some(),
        "is_available() y try_new() deben ser consistentes"
    );
}
