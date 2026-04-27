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

use std::collections::HashMap;

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_parallel::ParallelRuntime;
use gadget_ng_pm::{SlabLayout, slab_pm};

use crate::short_range;
use crate::short_range::ShortRangeParamsPeriodic;

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

// ── Corto alcance sobre dominio SFC (Fase 23) ─────────────────────────────────

/// Parámetros para el árbol de corto alcance distribuido en dominio 3D/SFC.
///
/// Idéntico en estructura a [`SlabShortRangeParams`], pero los `halo_particles`
/// fueron obtenidos mediante `exchange_halos_3d_periodic` sobre un dominio SFC
/// (no mediante `exchange_halos_by_z_periodic` sobre un slab-z).
///
/// La diferencia arquitectónica es upstream: este path garantiza cobertura
/// geométrica completa en x, y, z y diagonales periódicas, independientemente
/// de cómo estén distribuidas las partículas en los ranks.
pub struct SfcShortRangeParams<'a> {
    /// Partículas propias del rank (índices 0..n_local). Sus fuerzas SR se acumulan.
    pub local_particles: &'a [Particle],
    /// Partículas ghost recibidas de ranks vecinos mediante `exchange_halos_3d_periodic`.
    /// Solo contribuyen a las fuerzas de las partículas locales (lectura).
    pub halo_particles: &'a [Particle],
    pub eps2: f64,
    /// Constante gravitacional efectiva (usar `g_cosmo = G/a` en cosmología).
    pub g: f64,
    pub r_split: f64,
    pub box_size: f64,
}

/// Calcula las aceleraciones de corto alcance sobre dominio 3D/SFC.
///
/// Construye un árbol con `local + halo` y computa fuerzas SR solo sobre las
/// `n_local = local_particles.len()` partículas propias, usando `minimum_image`
/// periódico en todas las distancias.
///
/// El kernel es `erfc(r / (√2·r_s))`, idéntico al de [`short_range_accels_slab`].
/// La diferencia está en que los halos fueron reunidos con halo volumétrico 3D
/// periódico, garantizando cobertura correcta para cualquier descomposición de dominio.
pub fn short_range_accels_sfc(params: &SfcShortRangeParams<'_>, out: &mut [Vec3]) {
    // Delegamos al mismo kernel periódico: la diferencia es cómo se obtuvieron los halos.
    let slab_params = SlabShortRangeParams {
        local_particles: params.local_particles,
        halo_particles: params.halo_particles,
        eps2: params.eps2,
        g: params.g,
        r_split: params.r_split,
        box_size: params.box_size,
    };
    short_range_accels_slab(&slab_params, out);
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

// ── Scatter/Gather PM ↔ SFC (Fase 24) ─────────────────────────────────────────

/// Estadísticas del ciclo scatter/gather PM (Fase 24).
#[derive(Debug, Clone, Copy, Default)]
pub struct PmScatterStats {
    /// Partículas enviadas al scatter PM (= partículas locales SFC).
    pub scatter_particles: usize,
    /// Bytes totales enviados en el scatter (5 × f64 × scatter_particles).
    pub scatter_bytes: usize,
    /// Partículas recibidas en el gather (= scatter_particles).
    pub gather_particles: usize,
    /// Bytes totales recibidos en el gather (4 × f64 × gather_particles).
    pub gather_bytes: usize,
    /// Tiempo del scatter alltoallv (ns).
    pub scatter_ns: u64,
    /// Tiempo del gather alltoallv (ns).
    pub gather_ns: u64,
    /// Tiempo del PM solve en slab (deposit + FFT + interp) (ns).
    pub pm_solve_ns: u64,
}

/// Calcula las aceleraciones PM de largo alcance para partículas en dominio SFC
/// mediante un protocolo scatter/gather mínimo hacia slabs PM.
///
/// ## Protocolo
///
/// ### Scatter (SFC → slab PM)
/// Cada rank SFC enruta cada partícula al rank PM dueño de la celda CIC `iz0_global`:
/// ```text
/// iz0_global = floor(pos.z * nm / box_size) mod nm
/// target_rank = iz0_global / nz_local
/// mensaje: [gid_as_f64, x, y, z, mass]  →  40 bytes/partícula
/// ```
///
/// ### PM solve (slab rank)
/// Desempaqueta `(gid, pos, mass)` → llama al pipeline PM existente:
/// `deposit_slab_extended → exchange_density_halos_z → forces_from_slab →
/// exchange_force_halos_z → interpolate_slab_local`.
/// El mecanismo ghost-right de `deposit_slab_extended` ya maneja partículas
/// en el borde derecho del slab (CIC `iz0+1 = z_lo + nz_local`), sin cambiar la API.
///
/// ### Gather (slab PM → SFC)
/// Cada slab PM envía fuerzas de vuelta al source rank:
/// ```text
/// mensaje: [gid_as_f64, ax, ay, az]  →  32 bytes/partícula
/// ```
///
/// ## Resultado
/// - `Vec<Vec3>`: aceleraciones PM indexadas igual que `local` (por posición).
/// - `PmScatterStats`: métricas de scatter, gather y PM solve.
///
/// ## Correctitud
/// - Equivalente física al path `clone + exchange_domain_by_z + PM + exchange_domain_sfc` de Fase 23.
/// - Para P=1: `alltoallv_f64` es una copia local; resultado bit-a-bit idéntico.
/// - Partículas en borde periódico Z (`iz0 = nm-1`, `iz0+1 → 0`): ghost-right + `exchange_density_halos_z` existente.
pub fn pm_scatter_gather_accels<R: ParallelRuntime + ?Sized>(
    local: &[Particle],
    layout: &SlabLayout,
    g: f64,
    r_split: f64,
    box_size: f64,
    rt: &R,
) -> (Vec<Vec3>, PmScatterStats) {
    use std::time::Instant;

    let n_local = local.len();
    let size = rt.size() as usize;
    let nm = layout.nm;
    let nz = layout.nz_local;

    // ── Shortcut P=1: sin alltoallv ─────────────────────────────────────────
    //
    // Con un solo rank, todos los `alltoallv_f64` son no-ops en el runtime
    // serial. En lugar de enrutar a través del protocolo scatter/gather,
    // llamamos directamente al pipeline PM existente sobre las partículas
    // locales. El resultado es bit-a-bit idéntico al path de Fase 23 en P=1.
    if size == 1 {
        let t_pm = Instant::now();
        let positions: Vec<Vec3> = local.iter().map(|p| p.position).collect();
        let masses: Vec<f64> = local.iter().map(|p| p.mass).collect();
        let mut density_ext = slab_pm::deposit_slab_extended(&positions, &masses, layout, box_size);
        slab_pm::exchange_density_halos_z(&mut density_ext, layout, rt);
        let mut forces =
            slab_pm::forces_from_slab(&density_ext, layout, g, box_size, Some(r_split), rt);
        slab_pm::exchange_force_halos_z(&mut forces, layout, rt);
        let acc_pm = slab_pm::interpolate_slab_local(&positions, &forces, layout, box_size);
        let pm_solve_ns = t_pm.elapsed().as_nanos() as u64;
        let stats = PmScatterStats {
            scatter_particles: n_local,
            scatter_bytes: n_local * 5 * 8,
            gather_particles: n_local,
            gather_bytes: n_local * 4 * 8,
            pm_solve_ns,
            ..Default::default()
        };
        return (acc_pm, stats);
    }

    // ── 1. SCATTER: empaquetar (gid, pos, mass) por rank PM destino ─────────
    //
    // Formato por partícula: [gid_bits, x, y, z, mass] = 5 × f64 = 40 bytes.
    // Usamos f64::from_bits(gid as u64) para empaquetar el ID entero sin
    // estructuras adicionales; se recupera con data.to_bits() as usize.
    //
    // Routing: target_rank = iz0_global / nz_local
    // donde iz0_global = floor(pos.z * nm / box_size).rem_euclid(nm)
    // Esto garantiza que iz0_global cae en [0, nm) y la partícula va al
    // rank que posee su celda CIC izquierda, exactamente como exchange_domain_by_z.

    let mut scatter_sends: Vec<Vec<f64>> = vec![Vec::new(); size];
    for p in local.iter() {
        let iz0 = (p.position.z * nm as f64 / box_size).floor() as i64;
        let iz0 = iz0.rem_euclid(nm as i64) as usize;
        let target = (iz0 / nz).min(size - 1);
        let buf = &mut scatter_sends[target];
        buf.push(f64::from_bits(p.global_id as u64));
        buf.push(p.position.x);
        buf.push(p.position.y);
        buf.push(p.position.z);
        buf.push(p.mass);
    }

    let scatter_bytes: usize = scatter_sends.iter().map(|v| v.len() * 8).sum();

    let t_scatter = Instant::now();
    let scatter_recv = rt.alltoallv_f64(&scatter_sends);
    let scatter_ns = t_scatter.elapsed().as_nanos() as u64;

    // ── 2. PM SOLVE: deposit → FFT → interpolate en este slab rank ──────────
    //
    // Desempaquetar partículas recibidas, recordando de qué source rank vinieron
    // para poder devolver las fuerzas al destino correcto.

    let mut slab_pos: Vec<Vec3> = Vec::new();
    let mut slab_mass: Vec<f64> = Vec::new();
    let mut slab_src: Vec<usize> = Vec::new();
    let mut slab_gids: Vec<u64> = Vec::new();

    for (src_rank, data) in scatter_recv.iter().enumerate() {
        let n = data.len() / 5;
        for i in 0..n {
            let base = i * 5;
            slab_gids.push(data[base].to_bits());
            slab_pos.push(Vec3::new(data[base + 1], data[base + 2], data[base + 3]));
            slab_mass.push(data[base + 4]);
            slab_src.push(src_rank);
        }
    }

    let t_pm = Instant::now();

    let mut density_ext = slab_pm::deposit_slab_extended(&slab_pos, &slab_mass, layout, box_size);
    slab_pm::exchange_density_halos_z(&mut density_ext, layout, rt);
    let mut forces =
        slab_pm::forces_from_slab(&density_ext, layout, g, box_size, Some(r_split), rt);
    slab_pm::exchange_force_halos_z(&mut forces, layout, rt);
    let acc_slab = slab_pm::interpolate_slab_local(&slab_pos, &forces, layout, box_size);

    let pm_solve_ns = t_pm.elapsed().as_nanos() as u64;

    // ── 3. GATHER: empaquetar (gid, acc) de vuelta al source rank ───────────
    //
    // Formato: [gid_bits, ax, ay, az] = 4 × f64 = 32 bytes/partícula.

    let mut gather_sends: Vec<Vec<f64>> = vec![Vec::new(); size];
    for (i, &src) in slab_src.iter().enumerate() {
        let buf = &mut gather_sends[src];
        buf.push(f64::from_bits(slab_gids[i]));
        buf.push(acc_slab[i].x);
        buf.push(acc_slab[i].y);
        buf.push(acc_slab[i].z);
    }

    let gather_bytes: usize = gather_sends.iter().map(|v| v.len() * 8).sum();

    let t_gather = Instant::now();
    let gather_recv = rt.alltoallv_f64(&gather_sends);
    let gather_ns = t_gather.elapsed().as_nanos() as u64;

    // ── 4. RECONSTRUIR acc_pm por global_id ─────────────────────────────────
    //
    // Construir mapa global_id → acc_pm desde los datos recibidos.

    let mut lr_map: HashMap<usize, Vec3> = HashMap::with_capacity(n_local);
    for data in gather_recv.iter() {
        let n = data.len() / 4;
        for i in 0..n {
            let base = i * 4;
            let gid = data[base].to_bits() as usize;
            let acc = Vec3::new(data[base + 1], data[base + 2], data[base + 3]);
            lr_map.insert(gid, acc);
        }
    }

    let acc_pm: Vec<Vec3> = local
        .iter()
        .map(|p| lr_map.get(&p.global_id).copied().unwrap_or(Vec3::zero()))
        .collect();

    let stats = PmScatterStats {
        scatter_particles: n_local,
        scatter_bytes,
        gather_particles: n_local,
        gather_bytes,
        scatter_ns,
        gather_ns,
        pm_solve_ns,
    };

    (acc_pm, stats)
}

/// Calcula estadísticas del halo dados los parámetros de slab.
pub fn halo_stats(halo_particles: &[Particle]) -> HaloStats {
    let n = halo_particles.len();
    HaloStats {
        n_particles: n,
        bytes: std::mem::size_of_val(halo_particles),
    }
}

// ── Tests internos ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn make_particle(id: usize, pos: Vec3, mass: f64) -> Particle {
        Particle::new(id, mass, pos, Vec3::zero())
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
            "fuerza no nula con una sola partícula: {:?}",
            out[0]
        );
    }

    #[test]
    fn halo_particle_contributes_force() {
        // Partícula local en z=0.1, halo en z=0.2 (dentro de r_cut=0.5).
        // Fuerza SR debe ser no nula.
        let local = make_particle(0, Vec3::new(0.5, 0.5, 0.1), 1.0);
        let halo = make_particle(1, Vec3::new(0.5, 0.5, 0.2), 1.0);
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
        assert!(out[0].z > 0.0, "fuerza SR esperada en +z, got {:?}", out[0]);
    }

    #[test]
    fn periodic_halo_force_via_minimum_image() {
        // Partícula local en z=0.05 (rank 0), partícula halo en z=0.95 (rank P-1).
        // Sin minimum_image: distancia = 0.9 (fuera de r_cut=0.5 → fuerza ≈ 0).
        // Con minimum_image: distancia = 0.1 (dentro → fuerza ≠ 0).
        let local = make_particle(0, Vec3::new(0.5, 0.5, 0.05), 1.0);
        let halo = make_particle(1, Vec3::new(0.5, 0.5, 0.95), 1.0);
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
            "fuerza periódica debe apuntar en -z (imagen más cercana en z=-0.05), got {:?}",
            out[0]
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
