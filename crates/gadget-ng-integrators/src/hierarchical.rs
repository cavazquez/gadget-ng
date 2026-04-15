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
//! ## Algoritmo KDK con predictor de posición para inactivas
//!
//! Para cada sub-paso fino `s` ∈ [0, 2^max_level):
//!
//! 1. **START kick** para partículas cuyo paso individual *comienza* en `t = s·fine_dt`
//!    (`s % stride(k) == 0`, `stride(k) = 2^(max_level - k)`):
//!    ```text
//!    elapsed[i] = 0          // reinicia contador desde último sync
//!    v_i += a_i * (dt_i / 2)
//!    ```
//!
//! 2. **Drift** de *todas* las partículas (primer orden):
//!    ```text
//!    x_i += v_i * fine_dt
//!    elapsed[i] += 1
//!    ```
//!
//! 3. **Predictor + END kick** para partículas cuyo paso termina en `t = (s+1)·fine_dt`
//!    (`(s+1) % stride(k) == 0`):
//!    ```text
//!    // Corrección de segundo orden para partículas INACTIVAS antes de evaluar fuerzas:
//!    Δx_j = 0.5 * a_j * (elapsed[j] * fine_dt)²   (j ∉ end_active)
//!    x_j += Δx_j
//!    compute a_new  (usa posiciones predichas de inactivas)
//!    x_j -= Δx_j   (restaurar posición real)
//!    // END kick y rebinning para activas:
//!    v_i += a_new * (dt_i / 2)
//!    a_i = a_new
//!    elapsed[i] = 0
//!    reasignar bin con criterio de Aarseth
//!    ```
//!
//! La corrección `Δx_j` es el predictor de Störmer para partículas inactivas:
//! sus posiciones se mejoran de O(Δt²) a O(Δt³) para la evaluación de fuerzas,
//! sin modificar la integración simpléctica de las partículas activas.
use gadget_ng_core::{Particle, Vec3};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Estado de bins por partícula. Se mantiene fuera de `Particle` para no contaminar
/// `PartialEq` y demás derives del struct de core.
///
/// Se puede serializar a JSON para persistir el estado entre reinicios de simulación.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalState {
    /// Nivel de cada partícula local: `dt_i = dt_base / 2^levels[i]`.
    pub levels: Vec<u32>,
    /// Número de sub-pasos finos transcurridos desde el último kick (START o END).
    /// Se usa para el predictor de posición de partículas inactivas.
    pub elapsed: Vec<u64>,
}

impl HierarchicalState {
    /// Crea el estado inicial con todas las partículas en nivel 0 (paso completo).
    pub fn new(n: usize) -> Self {
        Self {
            levels: vec![0; n],
            elapsed: vec![0; n],
        }
    }

    /// Serializa el estado a `<dir>/hierarchical_state.json`.
    ///
    /// El directorio debe existir; normalmente es el directorio de un snapshot.
    /// Guardar el estado permite reanudar una simulación jerárquica sin perder
    /// la información de bins y contadores de tiempo por partícula.
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        let path = dir.join("hierarchical_state.json");
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Carga el estado desde `<dir>/hierarchical_state.json`.
    pub fn load(dir: &Path) -> std::io::Result<Self> {
        let path = dir.join("hierarchical_state.json");
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
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
/// ## Predictor de posición para partículas inactivas
///
/// Antes de evaluar fuerzas en cada END kick, las posiciones de las partículas
/// **inactivas** se mejoran temporalmente con el predictor de Störmer:
///
/// ```text
/// Δx_j = 0.5 * a_j * (elapsed[j] * fine_dt)²
/// ```
///
/// donde `elapsed[j]` es el número de sub-pasos finos desde el último kick de
/// la partícula j, y `a_j` es la aceleración en su último sync-point.
/// Las posiciones predichas se usan **solo** para la evaluación de fuerzas
/// y se restauran inmediatamente después, preservando la integración simpléctica.
///
/// # Argumentos
/// - `particles` — estado de todas las partículas locales (modificado in-place).
/// - `state` — niveles de bin y contadores de tiempo; gestionado internamente.
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
    assert_eq!(particles.len(), state.elapsed.len());

    let n_fine = 1u64 << max_level; // 2^max_level sub-pasos
    let fine_dt = dt_base / n_fine as f64;
    let n = particles.len();

    // Buffer de aceleraciones; sólo usamos los primeros `active.len()` elementos.
    let mut acc_buf = vec![Vec3::zero(); n];
    // Buffer de correcciones de posición para predictor inactivo.
    let mut pred_corr = vec![Vec3::zero(); n];

    for s in 0..n_fine {
        // ── 1. START kick para partículas que comienzan su paso en t = s·fine_dt ──
        // Condición: s % stride(k) == 0, stride(k) = 2^(max_level - k).
        for (i, (p, &lvl)) in particles.iter_mut().zip(state.levels.iter()).enumerate() {
            let stride = 1u64 << (max_level - lvl);
            if s % stride == 0 {
                state.elapsed[i] = 0; // reiniciar contador desde este sync
                let dt_i = dt_base / (1u64 << lvl) as f64;
                p.velocity += p.acceleration * (0.5 * dt_i);
            }
        }

        // ── 2. Drift de TODAS las partículas (primer orden) ──────────────────────
        for p in particles.iter_mut() {
            p.position += p.velocity * fine_dt;
        }
        for e in state.elapsed.iter_mut() {
            *e += 1;
        }

        // ── 3. Predictor + END kick ───────────────────────────────────────────────
        // Condición: (s+1) % stride(k) == 0.
        let end_active: Vec<usize> = (0..n)
            .filter(|&i| {
                let stride = 1u64 << (max_level - state.levels[i]);
                (s + 1) % stride == 0
            })
            .collect();

        if !end_active.is_empty() {
            // Aplicar predictor de segundo orden a partículas INACTIVAS:
            // sus posiciones se mejoran temporalmente antes de la evaluación de fuerzas.
            // Para activas (elapsed == stride → elapsed_t == dt_i), no se modifica nada
            // porque su posición ya es la correcta del leapfrog.
            for i in 0..n {
                let stride_i = 1u64 << (max_level - state.levels[i]);
                let el = state.elapsed[i];
                if el > 0 && el < stride_i {
                    // Partícula inactiva: elapsed < stride (no está en end_active)
                    let elapsed_t = el as f64 * fine_dt;
                    let corr = particles[i].acceleration * (0.5 * elapsed_t * elapsed_t);
                    particles[i].position += corr;
                    pred_corr[i] = corr;
                }
                // Activas (elapsed == stride): su posición ya es correcta; pred_corr[i] = Vec3::zero() (default)
            }

            // Calcular aceleraciones con posiciones predichas.
            compute(particles, &end_active, &mut acc_buf[..end_active.len()]);

            // Restaurar posiciones reales de las inactivas.
            for i in 0..n {
                particles[i].position -= pred_corr[i];
                pred_corr[i] = Vec3::zero(); // limpiar para siguiente sub-paso
            }

            // END kick, actualizar aceleración y reasignar bin.
            for (j, &i) in end_active.iter().enumerate() {
                let dt_i = dt_base / (1u64 << state.levels[i]) as f64;
                let a_new = acc_buf[j];
                particles[i].velocity += a_new * (0.5 * dt_i);
                particles[i].acceleration = a_new;
                state.elapsed[i] = 0; // reiniciar tras END kick
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
        assert!(hs.elapsed.iter().all(|&e| e == 0));
    }

    #[test]
    fn hierarchical_state_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut hs = HierarchicalState::new(4);
        hs.levels = vec![0, 1, 2, 3];
        hs.elapsed = vec![0, 3, 7, 1];
        hs.save(dir.path()).unwrap();

        let hs2 = HierarchicalState::load(dir.path()).unwrap();
        assert_eq!(hs.levels, hs2.levels);
        assert_eq!(hs.elapsed, hs2.elapsed);
    }

    // ── Tests de stride ───────────────────────────────────────────────────────

    #[test]
    fn stride_level0_only_fires_at_s0() {
        let max_level = 3u32;
        let n_fine = 1u64 << max_level;
        let level = 0u32;
        let stride = 1u64 << (max_level - level); // 8
        let active_starts: Vec<u64> = (0..n_fine).filter(|&s| s % stride == 0).collect();
        assert_eq!(active_starts, vec![0]);
        let active_ends: Vec<u64> = (0..n_fine).filter(|&s| (s + 1) % stride == 0).collect();
        assert_eq!(active_ends, vec![7]);
    }

    #[test]
    fn stride_max_level_fires_every_substep() {
        let max_level = 3u32;
        let n_fine = 1u64 << max_level;
        let stride = 1u64 << (max_level - max_level); // 1
        let starts: Vec<u64> = (0..n_fine).filter(|&s| s % stride == 0).collect();
        assert_eq!(starts.len(), n_fine as usize);
        let ends: Vec<u64> = (0..n_fine).filter(|&s| (s + 1) % stride == 0).collect();
        assert_eq!(ends.len(), n_fine as usize);
    }

    // ── Test de conservación de energía con nivel 0 ──────────────────────────

    /// Con todas las partículas en nivel 0 el integrador jerárquico debe conservar
    /// la energía del oscilador armónico dentro del 0.1 %.
    #[test]
    fn hierarchical_level0_energy_conserved() {
        let k_spring = 1.0_f64;
        let dt = 0.01_f64;
        let max_level = 4u32;
        let eta_large = 1000.0_f64;
        let eps2 = 1.0_f64;

        let mut p = vec![Particle::new(
            0,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.2, 0.0),
        )];
        p[0].acceleration = -k_spring * p[0].position;

        let energy = |pp: &[Particle]| {
            0.5 * pp[0].velocity.dot(pp[0].velocity)
                + 0.5 * k_spring * pp[0].position.dot(pp[0].position)
        };
        let e0 = energy(&p);

        let mut state = HierarchicalState::new(1);
        for _ in 0..200 {
            hierarchical_kdk_step(
                &mut p,
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

        let e_final = energy(&p);
        let rel_err = ((e_final - e0) / e0).abs();
        assert!(
            rel_err < 1e-3,
            "deriva de energía demasiado grande: |ΔE/E₀| = {rel_err:.2e} (> 0.1 %)"
        );
    }
}
