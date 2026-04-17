use crate::config::{IcKind, RunConfig};
use crate::particle::Particle;
use crate::vec3::Vec3;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IcError {
    #[error("particle_count must be > 0")]
    ZeroParticles,
    #[error("two_body initial conditions require particle_count == 2")]
    TwoBodyCount,
    #[error("lattice ic requires a perfect cube particle_count = n^3, got {0}")]
    LatticeNotCube(usize),
}

fn hash_u64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

fn perturb(seed: u64, gid: usize, axis: u8) -> f64 {
    let h = hash_u64(seed.wrapping_add(gid as u64).wrapping_add(axis as u64));
    let u = (h as f64) / (u64::MAX as f64);
    (u - 0.5) * 1.0e-6
}

/// Condiciones iniciales deterministas por índice global (reproducibles MPI/serial).
pub fn build_particles(cfg: &RunConfig) -> Result<Vec<Particle>, IcError> {
    let n = cfg.simulation.particle_count;
    build_particles_for_gid_range(cfg, 0, n)
}

/// Partículas con `global_id ∈ [lo, hi)` (medio abierto), orden creciente por `global_id`.
pub fn build_particles_for_gid_range(
    cfg: &RunConfig,
    lo: usize,
    hi: usize,
) -> Result<Vec<Particle>, IcError> {
    let n = cfg.simulation.particle_count;
    if n == 0 {
        return Err(IcError::ZeroParticles);
    }
    if lo > hi || hi > n {
        return Ok(Vec::new());
    }
    let seed = cfg.simulation.seed;
    let box_size = cfg.simulation.box_size;
    let g = cfg.simulation.gravitational_constant;

    match cfg.initial_conditions.kind {
        IcKind::Lattice => {
            let side = (n as f64).cbrt().round() as usize;
            if side * side * side != n {
                return Err(IcError::LatticeNotCube(n));
            }
            let mut out = Vec::new();
            let spacing = box_size / side as f64;
            for gid in lo..hi {
                let ix = gid / (side * side);
                let rem = gid % (side * side);
                let iy = rem / side;
                let iz = rem % side;
                let x = (ix as f64 + 0.5) * spacing + perturb(seed, gid, 0);
                let y = (iy as f64 + 0.5) * spacing + perturb(seed, gid, 1);
                let z = (iz as f64 + 0.5) * spacing + perturb(seed, gid, 2);
                let pos = Vec3::new(x, y, z);
                let mass = 1.0 / n as f64;
                out.push(Particle::new(gid, mass, pos, Vec3::zero()));
            }
            let _ = g;
            Ok(out)
        }
        IcKind::TwoBody {
            mass1,
            mass2,
            separation,
        } => {
            if n != 2 {
                return Err(IcError::TwoBodyCount);
            }
            let eps = cfg.simulation.softening;
            let half = separation * 0.5;
            let m1 = mass1;
            let m2 = mass2;
            let x1 = Vec3::new(-half, 0.0, 0.0);
            let x2 = Vec3::new(half, 0.0, 0.0);
            let com = (m1 * x1 + m2 * x2) / (m1 + m2);
            let p1 = x1 - com;
            let p2 = x2 - com;
            let d = separation;
            let d2_eps = d * d + eps * eps;
            let v_rel = (g * (m1 + m2) * d * d / d2_eps.powf(1.5)).sqrt();
            let v1 = Vec3::new(0.0, v_rel * (m2 / (m1 + m2)), 0.0);
            let v2 = Vec3::new(0.0, -v_rel * (m1 / (m1 + m2)), 0.0);
            let mut full = vec![Particle::new(0, m1, p1, v1), Particle::new(1, m2, p2, v2)];
            full.retain(|p| p.global_id >= lo && p.global_id < hi);
            Ok(full)
        }
        IcKind::Plummer { a } => plummer_ics(n, a, seed, g, lo, hi),
        IcKind::UniformSphere { r } => uniform_sphere_ics(n, r, seed, lo, hi),
    }
}

// ── Esfera de Plummer ─────────────────────────────────────────────────────────

/// Genera ICs de Plummer deterministas para `gid ∈ [lo, hi)`.
///
/// Algoritmo:
/// 1. Posiciones: inversión numérica de la CDF de masa M(<r) de Plummer.
/// 2. Velocidades: Gaussianas isótropas σ_local(r) = σ_0 / (1+r²/a²)^{1/4}.
/// 3. Virial scaling global: escala v para que T = |W|/2.
///
/// El generador LCG es reproducible: se avanza hasta gid para ser consistente
/// en modo MPI (cada rango genera solo su segmento).
fn plummer_ics(
    n: usize,
    a: f64,
    seed: u64,
    g: f64,
    lo: usize,
    hi: usize,
) -> Result<Vec<Particle>, IcError> {
    if n == 0 {
        return Err(IcError::ZeroParticles);
    }
    let m_each = 1.0 / n as f64; // masa total = 1
    let m_tot = 1.0_f64;
    let sigma0 = (g * m_tot / (6.0 * a)).sqrt(); // dispersión de Jeans global

    // Generamos TODAS las partículas (para el virial scaling global).
    // Cada gid → estado LCG bien definido.
    let mut particles: Vec<Particle> = (0..n)
        .map(|gid| {
            let mut lcg = LcgState(seed.wrapping_add((gid as u64).wrapping_mul(0x9e3779b97f4a7c15)));
            // Radio (inversión de CDF)
            let r = plummer_cdf_inv(lcg.next_f64(), a);
            // Posición aleatoria en la esfera
            let (px, py, pz) = uniform_sphere_point(&mut lcg, r);
            // Velocidad Gaussiana isótropa con σ local
            let sigma_loc = sigma0 / (1.0 + (r / a) * (r / a)).sqrt().powf(0.25);
            let vx = sigma_loc * gauss_bm(&mut lcg);
            let vy = sigma_loc * gauss_bm(&mut lcg);
            let vz = sigma_loc * gauss_bm(&mut lcg);
            Particle::new(gid, m_each, Vec3::new(px, py, pz), Vec3::new(vx, vy, vz))
        })
        .collect();

    // Corregir COM y VCM.
    let (com, vcm) = center_of_mass_vel(&particles);
    for p in &mut particles {
        p.position -= com;
        p.velocity -= vcm;
    }

    // Virial scaling: calcular T y W y escalar v.
    let t = kinetic_energy(&particles);
    let w = plummer_potential_energy(m_tot, a, g); // fórmula analítica
    if t > 0.0 && w < 0.0 {
        let scale = (0.5 * w.abs() / t).sqrt();
        for p in &mut particles {
            p.velocity *= scale;
        }
    }

    // Devolver solo el rango [lo, hi).
    particles.retain(|p| p.global_id >= lo && p.global_id < hi);
    Ok(particles)
}

/// Inversión numérica de CDF de Plummer: resuelve u = x³/(x²+1)^{3/2} con x = r/a.
fn plummer_cdf_inv(u: f64, a: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = 1000.0 * a;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let x = mid / a;
        let cdf = x * x * x / (x * x + 1.0).powf(1.5);
        if cdf < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Punto uniformemente distribuido en la esfera de radio `r`.
fn uniform_sphere_point(lcg: &mut LcgState, r: f64) -> (f64, f64, f64) {
    let cos_t = 2.0 * lcg.next_f64() - 1.0;
    let sin_t = (1.0 - cos_t * cos_t).sqrt();
    let phi = 2.0 * std::f64::consts::PI * lcg.next_f64();
    (r * sin_t * phi.cos(), r * sin_t * phi.sin(), r * cos_t)
}

/// Gaussiana estándar via Box-Muller.
fn gauss_bm(lcg: &mut LcgState) -> f64 {
    let u1 = lcg.next_f64().max(1e-300);
    let u2 = lcg.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn center_of_mass_vel(ps: &[Particle]) -> (Vec3, Vec3) {
    let m_tot: f64 = ps.iter().map(|p| p.mass).sum();
    let com = ps
        .iter()
        .map(|p| p.position * p.mass)
        .fold(Vec3::zero(), |a, b| a + b)
        / m_tot;
    let vcm = ps
        .iter()
        .map(|p| p.velocity * p.mass)
        .fold(Vec3::zero(), |a, b| a + b)
        / m_tot;
    (com, vcm)
}

fn kinetic_energy(ps: &[Particle]) -> f64 {
    ps.iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity))
        .sum()
}

/// Energía potencial analítica de la esfera de Plummer: W = -3πGM²/(32a).
fn plummer_potential_energy(m: f64, a: f64, g: f64) -> f64 {
    -3.0 * std::f64::consts::PI * g * m * m / (32.0 * a)
}

// ── Esfera uniforme (colapso frío) ────────────────────────────────────────────

/// Genera N partículas distribuidas uniformemente en una esfera sólida de radio `r`,
/// todas con velocidad cero (benchmark de colapso frío). El COM se corrige a cero.
///
/// Algoritmo: rejection sampling en el cubo [-r, r]³; determinista por `seed` y `gid`.
/// Compatible con `build_particles_for_gid_range` para MPI.
///
/// Nota: usa un generador LCG independiente con salida en [0, 1) para una distribución
/// uniforme no sesgada en la esfera completa.
fn uniform_sphere_ics(
    n: usize,
    r: f64,
    seed: u64,
    lo: usize,
    hi: usize,
) -> Result<Vec<Particle>, IcError> {
    if n == 0 {
        return Err(IcError::ZeroParticles);
    }
    let m_each = 1.0 / n as f64;

    // LCG con salida correcta en [0, 1) usando los 53 bits superiores.
    // Es independiente de `LcgState` para no alterar las ICs de Plummer.
    let lcg_next = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
    };

    // Generar TODAS las N posiciones para poder corregir el COM global.
    let mut particles: Vec<Particle> = (0..n)
        .map(|gid| {
            // Semilla única por partícula y reproducible en rangos MPI.
            let mut s = seed
                .wrapping_add(gid as u64)
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add(0xbf58476d1ce4e5b9);
            // Rejection sampling: genera puntos en [-r,r]³ hasta caer dentro de la esfera.
            let pos = loop {
                let x = (lcg_next(&mut s) * 2.0 - 1.0) * r;
                let y = (lcg_next(&mut s) * 2.0 - 1.0) * r;
                let z = (lcg_next(&mut s) * 2.0 - 1.0) * r;
                if x * x + y * y + z * z <= r * r {
                    break Vec3::new(x, y, z);
                }
            };
            Particle::new(gid, m_each, pos, Vec3::zero())
        })
        .collect();

    // Corregir el centro de masa para que esté en el origen.
    let (com, _vcm) = center_of_mass_vel(&particles);
    for p in &mut particles {
        p.position -= com;
    }

    // Devolver solo el rango [lo, hi).
    particles.retain(|p| p.global_id >= lo && p.global_id < hi);
    Ok(particles)
}

/// Generador LCG rápido y reproducible.
struct LcgState(u64);

impl LcgState {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as f64 / u32::MAX as f64
    }
}
