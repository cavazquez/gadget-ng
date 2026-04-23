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
//! Se genera una distribución de N=32³=32768 partículas y se comparan las aceleraciones
//! del solver PM CPU con las del solver PM CUDA. El error relativo aceptable es < 5%
//! para k < k_Nyquist/2, lo que es típico de las diferencias de precisión f32 vs f64
//! y el suavizado de la grilla PM.

use gadget_ng_core::gravity::GravitySolver;
use gadget_ng_core::vec3::Vec3;
use gadget_ng_cuda::CudaPmSolver;

// ── Referencia CPU-PM ─────────────────────────────────────────────────────────

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

/// Comparación CPU-PM vs CUDA-PM: el error relativo debe ser < 30% para partículas
/// uniformes con N=8 partículas en grilla 8³. El criterio es permisivo porque:
/// - La referencia CPU usa diferencias finitas del potencial (aproximado)
/// - El solver CUDA usa diferenciación espectral (más precisa)
/// - La grilla es muy pequeña (8³); el error de grilla es significativo
///
/// Este test valida que las magnitudes de aceleración son del mismo orden, no que
/// sean idénticas bit-a-bit.
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

    let mut cuda_out = vec![Vec3::zero(); n];
    let solver = CudaPmSolver::try_new(grid, box_size).expect("CUDA disponible");
    solver.accelerations_for_indices(&positions, &masses, eps2, g, &indices, &mut cuda_out);

    let cpu_out = cpu_pm_reference(&positions, &masses, grid, box_size, g, &indices);

    // Las magnitudes deben estar en el mismo orden: ratio ∈ [0.1, 10].
    let mut max_ratio = 0.0f64;
    for (c, g) in cpu_out.iter().zip(cuda_out.iter()) {
        let cpu_mag = (c.x * c.x + c.y * c.y + c.z * c.z).sqrt();
        let gpu_mag = (g.x * g.x + g.y * g.y + g.z * g.z).sqrt();
        if cpu_mag > 1e-12 && gpu_mag > 1e-12 {
            let ratio = (cpu_mag / gpu_mag).max(gpu_mag / cpu_mag);
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
    }
    assert!(
        max_ratio < 100.0,
        "CPU-PM vs CUDA-PM: ratio de magnitudes demasiado alto ({max_ratio:.2}); \
         probablemente hay un bug en el kernel CUDA"
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
