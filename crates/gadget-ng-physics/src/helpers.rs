//! Shared test utilities for physics validation tests.
//!
//! Provides common setup functions (cosmological configs, ICs, direct forces)
//! to reduce code duplication across ~140 test files.

use gadget_ng_core::{
    CosmologySection, GravitySection, IcKind, InitialConditionsSection, OutputSection, Particle,
    PerformanceSection, RunConfig, SimulationSection, TimestepSection, UnitsSection, Vec3,
};

// ── Deterministic RNG ─────────────────────────────────────────────────────────

/// Simple LCG random number generator (deterministic, reproducible).
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as f64 / u32::MAX as f64
    }
    /// Standard normal via Box–Muller.
    fn gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Config helpers ────────────────────────────────────────────────────────────

/// LCDM Plummer-like config: Ωm=0.31, ΩΛ=0.69, h=0.677, box=100 Mpc/h.
pub fn lcdm_config() -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 100,
            softening: 0.05,
            physical_softening: false,
            gravitational_constant: 1.0,
            particle_count: 1000,
            box_size: 100.0,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Plummer { a: 1.0 },
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m: 0.31,
            omega_lambda: 0.69,
            h0: 0.677,
            a_init: 1.0,
            auto_g: false,
            w0: -1.0,
            wa: 0.0,
            m_nu_ev: 0.0,
            neutrino_hierarchy: Default::default(),
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(),
        reionization: Default::default(),
        mhd: Default::default(),
        turbulence: Default::default(),
        two_fluid: Default::default(),
        sidm: Default::default(),
        modified_gravity: Default::default(),
    }
}

/// EdS config: Ωm=1.0, ΩΛ=0.0, h=0.673, box=1.0 (unit box).
pub fn eds_config() -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.01,
            num_steps: 100,
            softening: 0.02,
            physical_softening: false,
            gravitational_constant: 1.0,
            particle_count: 64,
            box_size: 1.0,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::PerturbedLattice {
                amplitude: 0.05,
                velocity_amplitude: 0.0,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m: 1.0,
            omega_lambda: 0.0,
            h0: 0.673,
            a_init: 1.0,
            auto_g: false,
            w0: -1.0,
            wa: 0.0,
            m_nu_ev: 0.0,
            neutrino_hierarchy: Default::default(),
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(),
        reionization: Default::default(),
        mhd: Default::default(),
        turbulence: Default::default(),
        two_fluid: Default::default(),
        sidm: Default::default(),
        modified_gravity: Default::default(),
    }
}

// ── IC generators ─────────────────────────────────────────────────────────────

/// Create N DM particles in a Plummer sphere with given total mass and scale radius.
///
/// Uses inverse transform sampling for the radial profile and Maxwellian velocities
/// with σ = √(G·M / (6·a)).
pub fn plummer_sphere(n: usize, m_total: f64, r_scale: f64) -> Vec<Particle> {
    let mut rng = Lcg(12345);
    let m_part = m_total / n as f64;
    let sigma = (m_total / (6.0 * r_scale)).sqrt();

    (0..n)
        .map(|i| {
            let x = rng.next_f64().max(1e-15);
            let r = r_scale / (x.powf(-2.0 / 3.0) - 1.0).sqrt();

            let cos_theta = 2.0 * rng.next_f64() - 1.0;
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            let phi = 2.0 * std::f64::consts::PI * rng.next_f64();

            let pos = Vec3::new(
                r * sin_theta * phi.cos(),
                r * sin_theta * phi.sin(),
                r * cos_theta,
            );

            let vel = Vec3::new(
                sigma * rng.gauss(),
                sigma * rng.gauss(),
                sigma * rng.gauss(),
            );

            Particle::new(i, m_part, pos, vel)
        })
        .collect()
}

// ── Direct force computation ──────────────────────────────────────────────────

/// Direct O(N²) softened Newtonian acceleration.
///
/// acc_i = G · Σ_{j≠i} m_j · (r_j − r_i) / (|r_j − r_i|² + ε²)^(3/2)
pub fn direct_accel(positions: &[Vec3], masses: &[f64], eps2: f64, g: f64) -> Vec<Vec3> {
    let n = positions.len();
    let mut acc = vec![Vec3::zero(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dr = positions[j] - positions[i];
            let r2 = dr.dot(dr) + eps2;
            let inv_r3 = r2.powf(-1.5);
            acc[i] += dr * (g * masses[j] * inv_r3);
        }
    }
    acc
}

/// Direct O(N²) for a subset of particles (`indices` into positions/masses).
pub fn direct_accel_for_indices(
    positions: &[Vec3],
    masses: &[f64],
    eps2: f64,
    g: f64,
    indices: &[usize],
) -> Vec<Vec3> {
    let n = positions.len();
    indices
        .iter()
        .map(|&i| {
            let mut a = Vec3::zero();
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dr = positions[j] - positions[i];
                let r2 = dr.dot(dr) + eps2;
                let inv_r3 = r2.powf(-1.5);
                a += dr * (g * masses[j] * inv_r3);
            }
            a
        })
        .collect()
}

// ── Kepler ICs ────────────────────────────────────────────────────────────────

/// Two-body Kepler initial conditions: m1 at (−r,0,0), m2 at (+r,0,0).
///
/// Velocities are set for a circular orbit (G=1). Returns `(particles, period)`
/// where `period = 2π·√((2r)³ / (m1+m2))`.
pub fn kepler_two_body(m1: f64, m2: f64, r: f64) -> (Vec<Particle>, f64) {
    let g = 1.0;
    let total_mass = m1 + m2;
    let separation = 2.0 * r;
    let v_circ = (g * total_mass / separation).sqrt();
    let period = 2.0 * std::f64::consts::PI * (separation.powi(3) / (g * total_mass)).sqrt();
    let vy1 = m2 / total_mass * v_circ;
    let vy2 = -m1 / total_mass * v_circ;

    let p0 = Particle::new(0, m1, Vec3::new(-r, 0.0, 0.0), Vec3::new(0.0, vy1, 0.0));
    let p1 = Particle::new(1, m2, Vec3::new(r, 0.0, 0.0), Vec3::new(0.0, vy2, 0.0));
    (vec![p0, p1], period)
}
