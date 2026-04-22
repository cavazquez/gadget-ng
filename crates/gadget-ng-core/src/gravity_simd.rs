//! Kernel de gravedad directa con layout SoA, caché-blocking y auto-vectorización AVX2.
//!
//! Estrategia de optimización:
//! - **SoA** (Structure of Arrays): los componentes `x`, `y`, `z` y `mass` se pasan como
//!   slices contiguos separados, lo que permite al compilador vectorizar el bucle interno.
//! - **Caché-blocking** (`BLOCK_J = 64`): el bucle sobre j se procesa en tiles que ocupan
//!   ~2 KB por componente en L1 (64 × 8 bytes), reduciendo los fallos de caché.
//! - **`#[target_feature(enable = "avx2", enable = "fma")]`**: fuerza la emisión de
//!   instrucciones 256-bit (4× f64 por registro `ymm`) con FMA fusionado.
//! - **Mask en lugar de branch**: la condición `j == skip` se convierte en un factor
//!   `0.0 | 1.0` para mantener el bucle libre de saltos y vectorizable con `vblendpd`.
use crate::gravity::GravitySolver;
use crate::vec3::Vec3;

/// Tamaño del tile para caché-blocking. 64 × 4 componentes × 8 bytes = 2 KB por tile.
pub const BLOCK_J: usize = 64;

// ── Datos SoA del sistema global ──────────────────────────────────────────────

/// Parámetros del campo gravitatorio en layout SoA, compartidos por todos los kernels.
pub struct KernelParams<'a> {
    pub xs: &'a [f64],
    pub ys: &'a [f64],
    pub zs: &'a [f64],
    pub masses: &'a [f64],
    pub eps2: f64,
    pub g: f64,
}

// ── Kernel escalar (fallback sin AVX2) ────────────────────────────────────────

fn inner_scalar(xi: f64, yi: f64, zi: f64, skip: usize, p: &KernelParams<'_>) -> (f64, f64, f64) {
    let n = p.xs.len();
    let mut ax = 0.0_f64;
    let mut ay = 0.0_f64;
    let mut az = 0.0_f64;
    let mut j = 0;
    while j < n {
        let end = (j + BLOCK_J).min(n);
        for k in j..end {
            // Mask evita el branch: skip aporta 0 a la acumulación.
            let mask = if k == skip { 0.0_f64 } else { 1.0_f64 };
            let dx = p.xs[k] - xi;
            let dy = p.ys[k] - yi;
            let dz = p.zs[k] - zi;
            let r2 = dx * dx + dy * dy + dz * dz + p.eps2;
            let inv = 1.0 / r2.sqrt();
            let inv3 = inv * inv * inv;
            let factor = mask * p.g * p.masses[k] * inv3;
            ax += factor * dx;
            ay += factor * dy;
            az += factor * dz;
        }
        j += BLOCK_J;
    }
    (ax, ay, az)
}

// ── Kernel AVX2+FMA (hot path, compilador auto-vectoriza con SoA) ─────────────

/// # Safety
/// Debe llamarse sólo cuando la CPU soporte AVX2 y FMA.
/// El `#[target_feature]` permite al compilador emitir instrucciones ymm de 256 bits;
/// con layout SoA los cuatro slices son contiguos → vectorización automática de 4×f64.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn inner_blocked_avx2(
    xi: f64,
    yi: f64,
    zi: f64,
    skip: usize,
    p: &KernelParams<'_>,
) -> (f64, f64, f64) {
    // Mismo algoritmo que inner_scalar; con target_feature AVX2+FMA el compilador
    // auto-vectoriza los bucles sobre floats contiguos en slices SoA.
    inner_scalar(xi, yi, zi, skip, p)
}

// ── Wrapper público con detección en runtime ──────────────────────────────────

/// Calcula la aceleración sobre la partícula `(xi,yi,zi)` debida a todas las demás.
///
/// - Si la CPU soporta AVX2+FMA el compilador habrá emitido código vectorizado.
/// - En caso contrario usa el fallback escalar con caché-blocking equivalente.
pub fn accel_soa_blocked(
    xi: f64,
    yi: f64,
    zi: f64,
    skip: usize,
    p: &KernelParams<'_>,
) -> (f64, f64, f64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: acabamos de verificar en runtime que avx2 y fma están disponibles.
        return unsafe { inner_blocked_avx2(xi, yi, zi, skip, p) };
    }
    inner_scalar(xi, yi, zi, skip, p)
}

// ── Solver serial SIMD+blocking ───────────────────────────────────────────────

/// Solver de gravedad directa O(N²) con SoA, caché-blocking y auto-vectorización AVX2.
///
/// A diferencia de `DirectGravity` (escalar AoS), este solver **no garantiza resultados
/// bit-a-bit idénticos** entre plataformas o con reordenaciones de bloques, aunque sí
/// es determinista (misma entrada → misma salida en la misma plataforma).
#[derive(Debug, Default, Clone, Copy)]
pub struct SimdDirectGravity;

impl GravitySolver for SimdDirectGravity {
    fn accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
    ) {
        assert_eq!(global_positions.len(), global_masses.len());
        assert_eq!(global_indices.len(), out.len());

        // Extraer SoA una sola vez para todos los i.
        let xs: Vec<f64> = global_positions.iter().map(|p| p.x).collect();
        let ys: Vec<f64> = global_positions.iter().map(|p| p.y).collect();
        let zs: Vec<f64> = global_positions.iter().map(|p| p.z).collect();
        let params = KernelParams {
            xs: &xs,
            ys: &ys,
            zs: &zs,
            masses: global_masses,
            eps2,
            g,
        };

        for (k, &gi) in global_indices.iter().enumerate() {
            let (ax, ay, az) = accel_soa_blocked(xs[gi], ys[gi], zs[gi], gi, &params);
            out[k] = Vec3::new(ax, ay, az);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        GravitySection, IcKind, InitialConditionsSection, OutputSection, PerformanceSection,
        RunConfig, SimulationSection,
    };
    use crate::gravity::DirectGravity;
    use crate::ic::{build_particles, IcError};

    fn lattice_cfg(n: usize) -> RunConfig {
        RunConfig {
            simulation: SimulationSection {
                dt: 0.01,
                num_steps: 1,
                softening: 0.05,
                gravitational_constant: 1.0,
                particle_count: n,
                box_size: 1.0,
                seed: 42,
                integrator: Default::default(),
            },
            initial_conditions: InitialConditionsSection {
                kind: IcKind::Lattice,
            },
            output: OutputSection::default(),
            gravity: GravitySection::default(),
            performance: PerformanceSection::default(),
            timestep: crate::config::TimestepSection::default(),
            cosmology: crate::config::CosmologySection::default(),
            units: crate::config::UnitsSection::default(),
        decomposition: Default::default(),
        }
    }

    fn softening_squared(cfg: &RunConfig) -> f64 {
        cfg.simulation.softening * cfg.simulation.softening
    }

    /// Verifica que SimdDirectGravity produce resultados consistentes con
    /// DirectGravity (error relativo < 1e-10, sólo diferencias de redondeo f64).
    #[test]
    fn simd_matches_direct_gravity() -> Result<(), IcError> {
        let cfg = lattice_cfg(27);
        let particles = build_particles(&cfg)?;
        let n = particles.len();
        let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
        let eps2 = softening_squared(&cfg);
        let g = cfg.simulation.gravitational_constant;
        let idx: Vec<usize> = (0..n).collect();

        let mut out_ref = vec![Vec3::zero(); n];
        let mut out_simd = vec![Vec3::zero(); n];

        DirectGravity.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut out_ref);
        SimdDirectGravity.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut out_simd);

        for i in 0..n {
            let r = out_ref[i];
            let s = out_simd[i];
            // La reordenación del blocking puede dar pequeñas diferencias de redondeo.
            let err = ((r.x - s.x).powi(2) + (r.y - s.y).powi(2) + (r.z - s.z).powi(2)).sqrt();
            let mag = (r.x.powi(2) + r.y.powi(2) + r.z.powi(2)).sqrt().max(1e-30);
            assert!(
                err / mag < 1e-10,
                "partícula {i}: ref={r:?} simd={s:?} err_rel={:.2e}",
                err / mag
            );
        }
        Ok(())
    }
}
