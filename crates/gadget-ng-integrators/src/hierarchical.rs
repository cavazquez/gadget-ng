//! Block timesteps jerárquicos al estilo GADGET-4.
//!
//! ## Esquema de bins
//!
//! El tiempo base `dt_base` se subdivide en `2^max_level` sub-pasos finos (`fine_dt`).
//! Cada partícula tiene un nivel `k ∈ [0, max_level]` con paso individual:
//!
//! ```text
//! Nivel 0 → dt_base         (la más "lenta")
//! Nivel k → dt_base / 2^k
//! Nivel max_level → fine_dt (la más "rápida")
//! ```
//!
//! ## Algoritmo KDK start/end correcto
//!
//! Para cada sub-paso fino `s` ∈ [0, 2^max_level):
//!
//! 1. **START kick** para partículas cuyo paso individual *comienza* en `t = s·fine_dt`,
//!    es decir, `s % stride(k) == 0` donde `stride(k) = 2^(max_level - k)`.
//!    ```text
//!    v_i += a_i * (dt_i / 2)
//!    ```
//!
//! 2. **Drift** de *todas* las partículas:
//!    ```text
//!    x_i += v_i * fine_dt
//!    ```
//!
//! 3. **END kick** para partículas cuyo paso individual *termina* en `t = (s+1)·fine_dt`,
//!    es decir, `(s+1) % stride(k) == 0`. Se evalúan fuerzas en la nueva posición:
//!    ```text
//!    compute a_new
//!    v_i += a_new * (dt_i / 2)
//!    a_i = a_new
//!    reasignar bin con criterio de Aarseth
//!    ```
//!
//! Con todas las partículas en nivel 0 (stride = n_fine) el algoritmo es idéntico
//! al KDK global: START en s=0, drift acumulado = dt_base, END en s=n_fine-1.
use gadget_ng_core::{Particle, Vec3};

/// Estado de bins por partícula. Se mantiene fuera de `Particle` para no contaminar
/// `PartialEq`, `Serialize` y demás derives del struct de core.
#[derive(Debug, Clone)]
pub struct HierarchicalState {
    /// Nivel de cada partícula local: `dt_i = dt_base / 2^levels[i]`.
    pub levels: Vec<u32>,
}

impl HierarchicalState {
    /// Crea el estado inicial con todas las partículas en nivel 0 (paso completo).
    pub fn new(n: usize) -> Self {
        Self { levels: vec![0; n] }
    }

    /// Asigna los bins iniciales en base a las aceleraciones ya calculadas.
    pub fn init_from_accels(
        &mut self,
        particles: &[Particle],
        eps2: f64,
        dt_base: f64,
        eta: f64,
        max_level: u32,
    ) {
        assert_eq!(self.levels.len(), particles.len());
        for (i, p) in particles.iter().enumerate() {
            let acc_mag = p.acceleration.dot(p.acceleration).sqrt();
            self.levels[i] = aarseth_bin(acc_mag, eps2, dt_base, eta, max_level);
        }
    }
}

/// Calcula el nivel de bin óptimo según el criterio de Aarseth.
///
/// `dt_courant = eta * sqrt(softening / |a|)`, cuantizado a la potencia de 2 más
/// cercana (por abajo) en el rango `[0, max_level]`.
///
/// - Con `acc_mag == 0` devuelve 0 (paso completo).
/// - Con `eta` suficientemente grande (`dt_courant >= dt_base`) devuelve 0.
/// - Con `eta` suficientemente pequeño devuelve `max_level` (paso más fino).
pub fn aarseth_bin(acc_mag: f64, eps2: f64, dt_base: f64, eta: f64, max_level: u32) -> u32 {
    if acc_mag <= 0.0 || eta <= 0.0 {
        return 0;
    }
    let eps = eps2.sqrt().max(f64::EPSILON);
    let dt_courant = eta * (eps / acc_mag).sqrt();
    let fine_dt = dt_base / (1u64 << max_level) as f64;
    // Cuantizar a la potencia de 2 que no excede dt_courant.
    let dt_i = dt_courant.clamp(fine_dt, dt_base);
    // floor(log2(dt_base / dt_i)) = número de halvings necesarios.
    ((dt_base / dt_i).log2().floor() as u32).min(max_level)
}

/// Realiza un paso completo del sistema (`dt_base`) usando block timesteps jerárquicos.
///
/// # Argumentos
/// - `particles` — estado de todas las partículas locales (modificado in-place).
/// - `state` — niveles de bin por partícula; reasignados tras cada END kick.
/// - `dt_base` — paso base del sistema.
/// - `eps2` — cuadrado del softening de Plummer.
/// - `eta` — parámetro de Aarseth.
/// - `max_level` — máximo nivel de subdivisión.
/// - `compute` — cierre `FnMut(&[Particle], &[usize], &mut [Vec3])`.
///   Rellena `out[j]` con la aceleración de `particles[active_local[j]]`.
///   Los índices son **locales** (posición en `particles`), no `global_id`.
pub fn hierarchical_kdk_step(
    particles: &mut [Particle],
    state: &mut HierarchicalState,
    dt_base: f64,
    eps2: f64,
    eta: f64,
    max_level: u32,
    mut compute: impl FnMut(&[Particle], &[usize], &mut [Vec3]),
) {
    assert_eq!(particles.len(), state.levels.len());

    let n_fine = 1u64 << max_level; // 2^max_level sub-pasos
    let fine_dt = dt_base / n_fine as f64;
    let n = particles.len();

    // Buffer de aceleraciones; sólo usamos los primeros `active.len()` elementos.
    let mut acc_buf = vec![Vec3::zero(); n];

    for s in 0..n_fine {
        // ── 1. START kick para partículas que comienzan su paso en t = s·fine_dt ──
        // Condición: s % stride(k) == 0, stride(k) = 2^(max_level - k).
        for (p, &lvl) in particles.iter_mut().zip(state.levels.iter()) {
            let stride = 1u64 << (max_level - lvl);
            if s % stride == 0 {
                let dt_i = dt_base / (1u64 << lvl) as f64;
                p.velocity += p.acceleration * (0.5 * dt_i);
            }
        }

        // ── 2. Drift de TODAS las partículas ─────────────────────────────────────
        for p in particles.iter_mut() {
            p.position += p.velocity * fine_dt;
        }

        // ── 3. END kick para partículas cuyo paso termina en t = (s+1)·fine_dt ───
        // Condición: (s+1) % stride(k) == 0.
        // Primero recogemos los índices activos con sus strides ANTES de re-binning.
        let end_active: Vec<usize> = (0..n)
            .filter(|&i| {
                let stride = 1u64 << (max_level - state.levels[i]);
                (s + 1) % stride == 0
            })
            .collect();

        if !end_active.is_empty() {
            // Calcular aceleraciones en las nuevas posiciones para los activos.
            compute(particles, &end_active, &mut acc_buf[..end_active.len()]);

            for (j, &i) in end_active.iter().enumerate() {
                let dt_i = dt_base / (1u64 << state.levels[i]) as f64;
                let a_new = acc_buf[j];
                particles[i].velocity += a_new * (0.5 * dt_i);
                particles[i].acceleration = a_new;
                // Reasignar bin con la nueva aceleración.
                let acc_mag = a_new.dot(a_new).sqrt();
                state.levels[i] = aarseth_bin(acc_mag, eps2, dt_base, eta, max_level);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    // ── Tests de utilidades ───────────────────────────────────────────────────

    #[test]
    fn aarseth_bin_zero_accel_returns_level_0() {
        assert_eq!(aarseth_bin(0.0, 0.0025, 0.1, 0.025, 6), 0);
    }

    #[test]
    fn aarseth_bin_zero_eta_returns_level_0() {
        assert_eq!(aarseth_bin(1.0, 0.0025, 0.1, 0.0, 6), 0);
    }

    #[test]
    fn aarseth_bin_clamps_to_max_level() {
        // Aceleración enorme → dt_courant muy pequeño → nivel máximo.
        let level = aarseth_bin(1e10, 0.0025, 0.1, 0.025, 6);
        assert_eq!(level, 6);
    }

    #[test]
    fn aarseth_bin_large_eta_returns_level_0() {
        // eta muy grande → dt_courant > dt_base → clamped → nivel 0.
        let level = aarseth_bin(1.0, 1.0, 0.1, 1000.0, 6);
        assert_eq!(level, 0);
    }

    #[test]
    fn hierarchical_state_new_all_zero() {
        let hs = HierarchicalState::new(5);
        assert!(hs.levels.iter().all(|&l| l == 0));
    }

    // ── Test de stride ────────────────────────────────────────────────────────

    #[test]
    fn stride_level0_only_fires_at_s0() {
        // Nivel 0, max_level=3: stride=8, activo en s=0,8,...
        // START: s%8==0 → solo s=0 dentro de [0,7]
        let max_level = 3u32;
        let n_fine = 1u64 << max_level;
        let level = 0u32;
        let stride = 1u64 << (max_level - level); // 8
        let active_starts: Vec<u64> = (0..n_fine).filter(|&s| s % stride == 0).collect();
        assert_eq!(active_starts, vec![0]);
        // END: (s+1)%8==0 → s=7
        let active_ends: Vec<u64> = (0..n_fine).filter(|&s| (s + 1) % stride == 0).collect();
        assert_eq!(active_ends, vec![7]);
    }

    #[test]
    fn stride_max_level_fires_every_substep() {
        let max_level = 3u32;
        let n_fine = 1u64 << max_level;
        let stride = 1u64 << (max_level - max_level); // 1
                                                      // START: every s
        let starts: Vec<u64> = (0..n_fine).filter(|&s| s % stride == 0).collect();
        assert_eq!(starts.len(), n_fine as usize);
        // END: every s
        let ends: Vec<u64> = (0..n_fine).filter(|&s| (s + 1) % stride == 0).collect();
        assert_eq!(ends.len(), n_fine as usize);
    }

    // ── Test de equivalencia con KDK global ──────────────────────────────────

    /// Con todas las partículas en nivel 0 el algoritmo jerárquico debe ser
    /// equivalente al KDK global (oscilador armónico: posición idéntica).
    #[test]
    fn hierarchical_level0_matches_uniform_leapfrog() {
        let k_spring = 1.0_f64;
        let dt = 0.01_f64;
        let max_level = 4u32; // 16 sub-pasos
                              // eta muy grande → bins siempre en nivel 0.
        let eta_large = 1000.0_f64;
        let eps2 = 1.0_f64;

        let make_particle =
            || Particle::new(0, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.2, 0.0));

        // ── Referencia: KDK global ───────────────────────────────────────────
        let mut p_ref = vec![make_particle()];
        let mut scratch = vec![Vec3::zero()];
        for _ in 0..50 {
            crate::leapfrog_kdk_step(&mut p_ref, dt, &mut scratch, |ps, acc| {
                acc[0] = -k_spring * ps[0].position;
            });
        }

        // ── Jerárquico nivel 0 ───────────────────────────────────────────────
        let mut p_hier = vec![make_particle()];
        // Inicializar aceleración (necesaria para el primer START kick).
        p_hier[0].acceleration = -k_spring * p_hier[0].position;
        let mut state = HierarchicalState::new(1);
        // Forzar nivel 0 y que no cambie (eta_large → siempre nivel 0).
        for _ in 0..50 {
            hierarchical_kdk_step(
                &mut p_hier,
                &mut state,
                dt,
                eps2,
                eta_large,
                max_level,
                |ps, _idx, out| {
                    out[0] = -k_spring * ps[0].position;
                },
            );
        }

        let dx = (p_ref[0].position.x - p_hier[0].position.x).abs();
        assert!(
            dx < 1e-12,
            "pos_ref={:.12} pos_hier={:.12} diff={dx:.2e}",
            p_ref[0].position.x,
            p_hier[0].position.x
        );
    }
}
