//! TreePM de corto alcance distribuido — Fase 21.
//!
//! Implementa el árbol de corto alcance con descomposición en slab z y halos
//! de partículas periódicos, reutilizando la infraestructura de slab PM de Fase 20.
//!
//! ## Estrategia (Opción C — ghost particles por z-slab)
//!
//! Cada rank posee partículas propias en z ∈ [z_lo, z_hi). Para calcular las
//! fuerzas de corto alcance sin allgather global:
//!
//! 1. Intercambiar halos SR con rank vecinos: se solicitan partículas de
//!    z ∈ [z_lo − r_cut, z_lo) ∪ [z_hi, z_hi + r_cut) con wrap periódico.
//! 2. Construir árbol local con partículas propias + halos recibidos.
//! 3. Calcular fuerzas SR (kernel erfc + minimum_image) solo sobre partículas propias.
//!
//! ## Limitación arquitectónica documentada
//!
//! El halo es **1D en z**: las interacciones de corto alcance que cruzan
//! fronteras x,y entre slabs de distintos ranks no están cubiertas.
//! Para un TreePM distribuido serio tipo GADGET se necesitaría halo volumétrico SFC 3D.
//! Esta limitación se documenta explícitamente en el reporte de Fase 21.

use gadget_ng_core::{Particle, Vec3};

use crate::short_range::{ShortRangeParamsPeriodic};
use crate::short_range;

// ── Parámetros ─────────────────────────────────────────────────────────────────

/// Parámetros para el árbol de corto alcance distribuido en slab z.
///
/// `local_particles` = partículas propias del rank (índices 0..n_local).
/// `halo_particles`  = partículas ghost recibidas de ranks vecinos.
/// Las fuerzas se calculan solo sobre `local_particles`.
pub struct SlabShortRangeParams<'a> {
    pub local_particles: &'a [Particle],
    pub halo_particles: &'a [Particle],
    pub eps2: f64,
    /// Constante gravitacional efectiva (usar `g_cosmo = G/a` en cosmología).
    pub g: f64,
    pub r_split: f64,
    pub box_size: f64,
}

// ── Función principal ──────────────────────────────────────────────────────────

/// Calcula las aceleraciones de corto alcance distribuidas en slab z.
///
/// Construye un árbol con `local + halo` y computa fuerzas SR solo sobre
/// las `n_local = local_particles.len()` partículas propias, usando
/// `minimum_image` periódico en todas las distancias.
///
/// El kernel es `erfc(r / (√2·r_s))`, consistente con el PM largo alcance
/// que usa el filtro Gaussiano `exp(-k²r_s²/2)` en k-space.
pub fn short_range_accels_slab(params: &SlabShortRangeParams<'_>, out: &mut [Vec3]) {
    let n_local = params.local_particles.len();
    assert_eq!(out.len(), n_local);

    if n_local == 0 {
        return;
    }

    let r_s = params.r_split;
    let r_cut = 5.0 * r_s;
    let r_cut2 = r_cut * r_cut;

    // Construir arrays combinados: local primero, luego halos.
    // Los índices 0..n_local corresponden a las partículas propias.
    let n_halo = params.halo_particles.len();
    let n_total = n_local + n_halo;

    let mut all_pos: Vec<Vec3> = Vec::with_capacity(n_total);
    let mut all_mass: Vec<f64> = Vec::with_capacity(n_total);

    for p in params.local_particles {
        all_pos.push(p.position);
        all_mass.push(p.mass);
    }
    for p in params.halo_particles {
        all_pos.push(p.position);
        all_mass.push(p.mass);
    }

    let sr_params = ShortRangeParamsPeriodic {
        positions: &all_pos,
        masses: &all_mass,
        eps2: params.eps2,
        g: params.g,
        r_split: r_s,
        r_cut2,
        box_size: params.box_size,
    };

    short_range::short_range_accels_periodic(&sr_params, n_local, out);
}

/// Estadísticas del halo de corto alcance intercambiado.
#[derive(Debug, Clone, Copy, Default)]
pub struct HaloStats {
    /// Número total de partículas halo recibidas de todos los vecinos.
    pub n_particles: usize,
    /// Bytes totales comunicados (estimación: n_particles × sizeof(Particle)).
    pub bytes: usize,
}

/// Radio de cutoff efectivo de corto alcance: `r_cut = 5 × r_split`.
pub fn effective_r_cut(r_split: f64) -> f64 {
    5.0 * r_split
}

/// Calcula estadísticas del halo dados los parámetros de slab.
pub fn halo_stats(halo_particles: &[Particle]) -> HaloStats {
    let n = halo_particles.len();
    HaloStats {
        n_particles: n,
        bytes: n * std::mem::size_of::<Particle>(),
    }
}

// ── Tests internos ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn make_particle(id: usize, pos: Vec3, mass: f64) -> Particle {
        Particle {
            position: pos,
            velocity: Vec3::zero(),
            acceleration: Vec3::zero(),
            mass,
            global_id: id,
        }
    }

    #[test]
    fn no_particles_no_crash() {
        let params = SlabShortRangeParams {
            local_particles: &[],
            halo_particles: &[],
            eps2: 0.01,
            g: 1.0,
            r_split: 0.1,
            box_size: 1.0,
        };
        let mut out: Vec<Vec3> = vec![];
        short_range_accels_slab(&params, &mut out);
    }

    #[test]
    fn single_local_no_halo_zero_force() {
        // Una sola partícula, sin halos → fuerza = 0.
        let p = make_particle(0, Vec3::new(0.5, 0.5, 0.5), 1.0);
        let params = SlabShortRangeParams {
            local_particles: &[p],
            halo_particles: &[],
            eps2: 1e-4,
            g: 1.0,
            r_split: 0.1,
            box_size: 1.0,
        };
        let mut out = vec![Vec3::zero()];
        short_range_accels_slab(&params, &mut out);
        assert!(
            out[0].x.abs() < 1e-14 && out[0].y.abs() < 1e-14 && out[0].z.abs() < 1e-14,
            "fuerza no nula con una sola partícula: {:?}", out[0]
        );
    }

    #[test]
    fn halo_particle_contributes_force() {
        // Partícula local en z=0.1, halo en z=0.2 (dentro de r_cut=0.5).
        // Fuerza SR debe ser no nula.
        let local = make_particle(0, Vec3::new(0.5, 0.5, 0.1), 1.0);
        let halo  = make_particle(1, Vec3::new(0.5, 0.5, 0.2), 1.0);
        let params = SlabShortRangeParams {
            local_particles: &[local],
            halo_particles: &[halo],
            eps2: 1e-6,
            g: 1.0,
            r_split: 0.1,
            box_size: 1.0,
        };
        let mut out = vec![Vec3::zero()];
        short_range_accels_slab(&params, &mut out);
        // La fuerza debe apuntar en +z (hacia el halo).
        assert!(
            out[0].z > 0.0,
            "fuerza SR esperada en +z, got {:?}", out[0]
        );
    }

    #[test]
    fn periodic_halo_force_via_minimum_image() {
        // Partícula local en z=0.05 (rank 0), partícula halo en z=0.95 (rank P-1).
        // Sin minimum_image: distancia = 0.9 (fuera de r_cut=0.5 → fuerza ≈ 0).
        // Con minimum_image: distancia = 0.1 (dentro → fuerza ≠ 0).
        let local = make_particle(0, Vec3::new(0.5, 0.5, 0.05), 1.0);
        let halo  = make_particle(1, Vec3::new(0.5, 0.5, 0.95), 1.0);
        let r_split = 0.1_f64;
        let r_cut = 5.0 * r_split; // 0.5

        let params = SlabShortRangeParams {
            local_particles: &[local],
            halo_particles: &[halo],
            eps2: 1e-6,
            g: 1.0,
            r_split,
            box_size: 1.0,
        };
        let mut out = vec![Vec3::zero()];
        short_range_accels_slab(&params, &mut out);

        // La fuerza debe ser no nula porque minimum_image da distancia 0.1 < r_cut=0.5.
        let fmag = (out[0].x * out[0].x + out[0].y * out[0].y + out[0].z * out[0].z).sqrt();
        assert!(
            fmag > 0.0,
            "fuerza SR periódica debe ser no nula (distancia min-image=0.1 < r_cut={r_cut}), got fmag={fmag}"
        );
        // La fuerza debe apuntar en -z (hacia el halo a través del borde periódico).
        assert!(
            out[0].z < 0.0,
            "fuerza periódica debe apuntar en -z (imagen más cercana en z=-0.05), got {:?}", out[0]
        );
    }

    #[test]
    fn halo_stats_correct() {
        let p = make_particle(0, Vec3::zero(), 1.0);
        let halos = vec![p; 10];
        let stats = halo_stats(&halos);
        assert_eq!(stats.n_particles, 10);
        assert_eq!(stats.bytes, 10 * std::mem::size_of::<Particle>());
    }
}
