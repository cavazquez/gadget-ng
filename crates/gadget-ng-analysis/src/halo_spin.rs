//! Parámetro de spin λ de halos FoF (Phase 72).
//!
//! ## Definiciones
//!
//! **Peebles (1971)**:
//! ```text
//! λ = |L| / (M × V_vir × R_vir)
//! ```
//! donde `V_vir = sqrt(G×M/R_vir)` y `R_vir ≈ R_200`.
//!
//! **Bullock et al. (2001)** (versión simplificada, más común en la literatura moderna):
//! ```text
//! λ' = |L| / (sqrt(2) × M × V_vir × R_vir)
//! ```
//!
//! El momento angular `L` se calcula respecto al centro de masa:
//! ```text
//! L = Σ_i m_i × (r_i - r_com) × (v_i - v_com)
//! ```
//!
//! ## Referencia
//!
//! Peebles (1971), A&A 11, 377; Bullock et al. (2001), ApJ 555, 240.

use gadget_ng_core::Vec3;

const G_INTERNAL: f64 = 4.302e-3; // pc M_sun⁻¹ (km/s)² — en unidades de kpc·(km/s)²/M_sun = 4.302e-6

/// Resultado de spin para un halo.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HaloSpin {
    /// Masa total del halo (unidades de la simulación).
    pub mass: f64,
    /// Radio virial R_200 (unidades de la simulación).
    pub r200: f64,
    /// Centro de masa (x,y,z).
    pub pos_com: [f64; 3],
    /// Velocidad del centro de masa (vx,vy,vz).
    pub vel_com: [f64; 3],
    /// Momento angular total L = (Lx, Ly, Lz).
    pub angular_momentum: [f64; 3],
    /// Módulo |L|.
    pub l_mag: f64,
    /// Parámetro de spin Peebles λ.
    pub lambda_peebles: f64,
    /// Parámetro de spin Bullock λ'.
    pub lambda_bullock: f64,
}

/// Parámetros para el cálculo de spin.
#[derive(Debug, Clone)]
pub struct SpinParams {
    /// Constante gravitacional G en unidades internas de la simulación.
    /// Valor por defecto: 4.302e-3 (kpc, km/s, M_sun).
    pub g_newton: f64,
    /// Factor de sobredensidad para R_vir (por defecto 200 × ρ_crit).
    pub delta_vir: f64,
    /// Densidad crítica ρ_crit en unidades internas.
    pub rho_crit: f64,
}

impl Default for SpinParams {
    fn default() -> Self {
        Self {
            g_newton: G_INTERNAL,
            delta_vir: 200.0,
            rho_crit: 2.775e11, // M_sun/Mpc³ para H0=100 h km/s/Mpc
        }
    }
}

/// Calcula el parámetro de spin λ para un halo dado su lista de partículas.
///
/// # Parámetros
/// - `positions`  — posiciones de las partículas del halo.
/// - `velocities` — velocidades de las partículas del halo.
/// - `masses`     — masas de las partículas del halo.
/// - `params`     — parámetros del cálculo.
///
/// # Retorna
/// `None` si el halo no tiene partículas o tiene masa cero.
pub fn halo_spin(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    params: &SpinParams,
) -> Option<HaloSpin> {
    let n = positions.len();
    if n == 0 || masses.is_empty() {
        return None;
    }

    let mass_total: f64 = masses.iter().sum();
    if mass_total <= 0.0 {
        return None;
    }

    // ── Centro de masa ────────────────────────────────────────────────────
    let pos_com = center_of_mass(positions, masses, mass_total);
    let vel_com = velocity_center(velocities, masses, mass_total);

    // ── Momento angular L = Σ m_i × (r_i - r_com) × (v_i - v_com) ───────
    let mut lx = 0.0f64;
    let mut ly = 0.0f64;
    let mut lz = 0.0f64;

    for (i, (&pos, &vel)) in positions.iter().zip(velocities.iter()).enumerate() {
        let m = if i < masses.len() {
            masses[i]
        } else {
            masses[0]
        };
        let rx = pos.x - pos_com[0];
        let ry = pos.y - pos_com[1];
        let rz = pos.z - pos_com[2];
        let vx = vel.x - vel_com[0];
        let vy = vel.y - vel_com[1];
        let vz = vel.z - vel_com[2];
        // L += m × (r × v)
        lx += m * (ry * vz - rz * vy);
        ly += m * (rz * vx - rx * vz);
        lz += m * (rx * vy - ry * vx);
    }

    let l_mag = (lx * lx + ly * ly + lz * lz).sqrt();

    // ── Radio virial R_200 = (3M / (4π × Δ_vir × ρ_crit))^(1/3) ─────────
    let r200 = r200_from_mass(mass_total, params);

    // ── Velocidad virial V_vir = sqrt(G × M / R_200) ─────────────────────
    let v_vir = if r200 > 0.0 {
        (params.g_newton * mass_total / r200).sqrt()
    } else {
        1.0
    };

    // ── Parámetros de spin ────────────────────────────────────────────────
    let denom = mass_total * v_vir * r200;
    let lambda_peebles = if denom > 0.0 { l_mag / denom } else { 0.0 };
    let lambda_bullock = lambda_peebles / std::f64::consts::SQRT_2;

    Some(HaloSpin {
        mass: mass_total,
        r200,
        pos_com,
        vel_com,
        angular_momentum: [lx, ly, lz],
        l_mag,
        lambda_peebles,
        lambda_bullock,
    })
}

/// Calcula el spin para múltiples halos dados como índices de partículas.
///
/// # Parámetros
/// - `positions`   — posiciones de TODAS las partículas.
/// - `velocities`  — velocidades de TODAS las partículas.
/// - `masses`      — masas de TODAS las partículas.
/// - `halo_ids`    — para cada halo, lista de índices de partículas.
/// - `params`      — parámetros del cálculo.
pub fn compute_halo_spins(
    positions: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    halo_ids: &[Vec<usize>],
    params: &SpinParams,
) -> Vec<Option<HaloSpin>> {
    halo_ids
        .iter()
        .map(|ids| {
            let pos: Vec<Vec3> = ids.iter().map(|&i| positions[i]).collect();
            let vel: Vec<Vec3> = ids.iter().map(|&i| velocities[i]).collect();
            let mass: Vec<f64> = ids.iter().map(|&i| masses[i]).collect();
            halo_spin(&pos, &vel, &mass, params)
        })
        .collect()
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn center_of_mass(positions: &[Vec3], masses: &[f64], total_mass: f64) -> [f64; 3] {
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    let mut cz = 0.0f64;
    for (i, &pos) in positions.iter().enumerate() {
        let m = if i < masses.len() {
            masses[i]
        } else {
            masses[0]
        };
        cx += m * pos.x;
        cy += m * pos.y;
        cz += m * pos.z;
    }
    [cx / total_mass, cy / total_mass, cz / total_mass]
}

fn velocity_center(velocities: &[Vec3], masses: &[f64], total_mass: f64) -> [f64; 3] {
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    let mut vz = 0.0f64;
    for (i, &vel) in velocities.iter().enumerate() {
        let m = if i < masses.len() {
            masses[i]
        } else {
            masses[0]
        };
        vx += m * vel.x;
        vy += m * vel.y;
        vz += m * vel.z;
    }
    [vx / total_mass, vy / total_mass, vz / total_mass]
}

fn r200_from_mass(mass: f64, params: &SpinParams) -> f64 {
    let rho_thresh = params.delta_vir * params.rho_crit;
    if rho_thresh <= 0.0 {
        return 0.0;
    }
    (3.0 * mass / (4.0 * std::f64::consts::PI * rho_thresh)).cbrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn make_ring(n: usize, r: f64, v: f64, m: f64) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>) {
        let mut pos = Vec::new();
        let mut vel = Vec::new();
        let masses = vec![m; n];
        for i in 0..n {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            pos.push(Vec3::new(r * theta.cos(), r * theta.sin(), 0.0));
            // Velocidad tangencial para órbita circular: v = sqrt(GM/r)
            vel.push(Vec3::new(-v * theta.sin(), v * theta.cos(), 0.0));
        }
        (pos, vel, masses)
    }

    #[test]
    fn spin_ring_positive() {
        // Un anillo de partículas con velocidad circular debe tener λ > 0
        let (pos, vel, mass) = make_ring(16, 10.0, 5.0, 1e10);
        let params = SpinParams::default();
        let spin = halo_spin(&pos, &vel, &mass, &params).unwrap();
        assert!(
            spin.lambda_peebles > 0.0,
            "λ debe ser positivo: {}",
            spin.lambda_peebles
        );
        assert!(spin.lambda_bullock > 0.0, "λ' debe ser positivo");
        assert!(spin.l_mag > 0.0, "|L| debe ser positivo");
    }

    #[test]
    fn spin_static_halo_zero() {
        // Halo sin velocidades → L = 0 → λ = 0
        let pos = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let vel = vec![Vec3::new(0.0, 0.0, 0.0); 3];
        let masses = vec![1e10; 3];
        let params = SpinParams::default();
        let spin = halo_spin(&pos, &vel, &masses, &params).unwrap();
        assert!(
            spin.l_mag < 1e-10,
            "L debe ser 0 para halo estático: {}",
            spin.l_mag
        );
        assert!(
            spin.lambda_peebles < 1e-10,
            "λ debe ser 0: {}",
            spin.lambda_peebles
        );
    }

    #[test]
    fn spin_empty_returns_none() {
        let params = SpinParams::default();
        let result = halo_spin(&[], &[], &[], &params);
        assert!(result.is_none());
    }

    #[test]
    fn lambda_bullock_smaller_than_peebles() {
        // λ_Bullock = λ_Peebles / sqrt(2) < λ_Peebles
        let (pos, vel, mass) = make_ring(8, 5.0, 3.0, 1e10);
        let params = SpinParams::default();
        let spin = halo_spin(&pos, &vel, &mass, &params).unwrap();
        assert!(
            spin.lambda_bullock < spin.lambda_peebles,
            "λ' debe ser menor que λ: {} vs {}",
            spin.lambda_bullock,
            spin.lambda_peebles
        );
        let ratio = spin.lambda_peebles / spin.lambda_bullock;
        assert!(
            (ratio - std::f64::consts::SQRT_2).abs() < 1e-10,
            "ratio λ/λ' debe ser sqrt(2): {ratio}"
        );
    }

    #[test]
    fn center_of_mass_symmetric() {
        // Distribución simétrica → COM en origen
        let pos = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let masses = vec![1.0, 1.0];
        let com = center_of_mass(&pos, &masses, 2.0);
        assert!(com[0].abs() < 1e-15, "COM.x debe ser 0: {}", com[0]);
    }

    #[test]
    fn spin_angular_momentum_direction() {
        // Rotación en plano XY → L debe apuntar en +Z
        let (pos, vel, mass) = make_ring(8, 5.0, 3.0, 1e10);
        let params = SpinParams::default();
        let spin = halo_spin(&pos, &vel, &mass, &params).unwrap();
        // Lz debe dominar
        assert!(
            spin.angular_momentum[2] > 0.0,
            "Lz debe ser positivo para rotación antihoraria: {}",
            spin.angular_momentum[2]
        );
        let lz_frac = spin.angular_momentum[2].abs() / spin.l_mag;
        assert!(lz_frac > 0.99, "L debe apuntar casi en Z: {lz_frac}");
    }

    #[test]
    fn compute_halo_spins_multi() {
        let (pos1, vel1, mass1) = make_ring(8, 5.0, 3.0, 1e10);
        let (pos2, vel2, mass2) = make_ring(4, 10.0, 2.0, 1e11);
        let n1 = pos1.len();
        let n2 = pos2.len();
        let all_pos: Vec<Vec3> = pos1.into_iter().chain(pos2).collect();
        let all_vel: Vec<Vec3> = vel1.into_iter().chain(vel2).collect();
        let all_mass: Vec<f64> = mass1.into_iter().chain(mass2).collect();
        let halo_ids = vec![
            (0..n1).collect::<Vec<_>>(),
            (n1..n1 + n2).collect::<Vec<_>>(),
        ];
        let params = SpinParams::default();
        let spins = compute_halo_spins(&all_pos, &all_vel, &all_mass, &halo_ids, &params);
        assert_eq!(spins.len(), 2);
        assert!(spins[0].is_some());
        assert!(spins[1].is_some());
    }
}
