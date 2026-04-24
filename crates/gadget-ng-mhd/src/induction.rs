//! Ecuación de inducción SPH: `dB/dt = ∇×(v×B)` (Phase 123).
//!
//! ## Formulación
//!
//! En la discretización SPH de Morris & Monaghan (1997), la ecuación de inducción
//! para la partícula `i` es:
//!
//! ```text
//! dB_i/dt = Σ_j (m_j/ρ_j) [(B_i·∇W_ij) v_ij - (v_ij·∇W_ij) B_i]
//! ```
//!
//! donde `v_ij = v_i - v_j` y `∇W_ij` es el gradiente del kernel SPH.
//!
//! Esta formulación preserva `∇·B = 0` mejor que la formulación no conservativa
//! y es simétrica respecto al intercambio i↔j.
//!
//! ## Referencia
//!
//! Morris & Monaghan (1997), J. Comput. Phys. 136, 41–60.
//! Price & Monaghan (2005), MNRAS 364, 384–406.

use gadget_ng_core::{BFieldKind, MhdSection, Particle, ParticleType, Vec3};
use crate::MU0;

/// Gradiente del kernel SPH cúbico (en 3D): `∇W(r, h)`.
///
/// Devuelve el gradiente evaluado en la dirección `r_ij = r_j - r_i`.
fn kernel_gradient(r_vec: Vec3, h: f64) -> Vec3 {
    let r2 = r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z;
    let r = r2.sqrt();
    if r < 1e-10 || h <= 0.0 { return Vec3::zero(); }
    let q = r / h;
    // Derivada del kernel cúbico B-spline
    let dw_dq = if q < 1.0 {
        let norm = 8.0 / (std::f64::consts::PI * h * h * h);
        norm * (-6.0 * q + 9.0 * q * q) // d/dq [1 - 6q² + 6q³] / norm_h
    } else if q < 2.0 {
        let norm = 8.0 / (std::f64::consts::PI * h * h * h);
        norm * (-6.0 * (2.0 - q).powi(2)) / 4.0
    } else {
        0.0
    };
    let dw_dr = dw_dq / h;
    // ∇W = (dW/dr) * r_hat = (dW/dr) * r_vec/r
    Vec3 {
        x: dw_dr * r_vec.x / r,
        y: dw_dr * r_vec.y / r,
        z: dw_dr * r_vec.z / r,
    }
}

/// Avanza el campo magnético de todas las partículas de gas un paso `dt`
/// usando la ecuación de inducción SPH (Phase 123).
///
/// La densidad SPH se aproxima como `ρ ≈ mass/h³` (consistente con el resto del código).
pub fn advance_induction(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    let mut db = vec![Vec3::zero(); n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas { continue; }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let rho_i = (particles[i].mass / (h_i * h_i * h_i)).max(1e-30);
        let b_i = particles[i].b_field;
        let v_i = particles[i].velocity;

        for j in 0..n {
            if i == j { continue; }
            if particles[j].ptype != ParticleType::Gas { continue; }

            let h_j = particles[j].smoothing_length.max(1e-10);
            let rho_j = (particles[j].mass / (h_j * h_j * h_j)).max(1e-30);
            let v_j = particles[j].velocity;
            let b_j = particles[j].b_field;

            let r_ij = Vec3 {
                x: particles[j].position.x - particles[i].position.x,
                y: particles[j].position.y - particles[i].position.y,
                z: particles[j].position.z - particles[i].position.z,
            };

            // ∇W evaluado en r_ij con h promedio
            let h_ij = 0.5 * (h_i + h_j);
            let grad_w = kernel_gradient(r_ij, h_ij);

            // v_ij = v_i - v_j
            let v_ij = Vec3 { x: v_i.x - v_j.x, y: v_i.y - v_j.y, z: v_i.z - v_j.z };
            let b_ij = Vec3 { x: b_i.x - b_j.x, y: b_i.y - b_j.y, z: b_i.z - b_j.z };

            // Contribución de j a dB_i/dt (formulación simétrica):
            // dB_i/dt += (m_j/ρ_j) [(B_ij·∇W_ij) v_ij - (v_ij·∇W_ij) B_ij]
            let b_dot_grad = b_ij.x * grad_w.x + b_ij.y * grad_w.y + b_ij.z * grad_w.z;
            let v_dot_grad = v_ij.x * grad_w.x + v_ij.y * grad_w.y + v_ij.z * grad_w.z;
            let factor = particles[j].mass / rho_j;

            db[i].x += factor * (b_dot_grad * v_ij.x - v_dot_grad * b_ij.x);
            db[i].y += factor * (b_dot_grad * v_ij.y - v_dot_grad * b_ij.y);
            db[i].z += factor * (b_dot_grad * v_ij.z - v_dot_grad * b_ij.z);
        }
        // Factor de normalización para consistencia con SPH estándar
        let _ = rho_i; // rho_i usada para verificación futura (supresión de warning)
    }

    // Integración explícita de Euler
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas { continue; }
        particles[i].b_field.x += db[i].x * dt;
        particles[i].b_field.y += db[i].y * dt;
        particles[i].b_field.z += db[i].z * dt;
    }
}

/// Inicializa el campo magnético de las partículas de gas según `cfg.b0_kind` (Phase 127).
///
/// - `Uniform`:  B = b0_uniform para todas las partículas de gas.
/// - `Random`:   B = amplitud aleatoria con |B| ≈ |b0_uniform| (usando global_id como semilla).
/// - `Spiral`:   B = B0 × (sin(2πy/L), cos(2πx/L), 0).
/// - `None`:     no-op.
///
/// `box_size` se usa para normalizar la posición en el modo espiral.
pub fn init_b_field(particles: &mut [Particle], cfg: &MhdSection, box_size: f64) {
    let b0 = Vec3::new(cfg.b0_uniform[0], cfg.b0_uniform[1], cfg.b0_uniform[2]);
    let b_mag = (b0.x * b0.x + b0.y * b0.y + b0.z * b0.z).sqrt();
    let l = box_size.max(1e-10);

    for p in particles.iter_mut() {
        if p.ptype != ParticleType::Gas { continue; }
        p.b_field = match cfg.b0_kind {
            BFieldKind::None => Vec3::zero(),
            BFieldKind::Uniform => b0,
            BFieldKind::Random => {
                // LCG con global_id como semilla — reproducible y sin dependencia externa
                let seed = p.global_id as u64;
                let rx = lcg_uniform(seed);
                let ry = lcg_uniform(seed.wrapping_add(1));
                let rz = lcg_uniform(seed.wrapping_add(2));
                let norm = (rx * rx + ry * ry + rz * rz).sqrt().max(1e-10);
                Vec3::new(rx / norm * b_mag, ry / norm * b_mag, rz / norm * b_mag)
            }
            BFieldKind::Spiral => {
                let x = p.position.x / l;
                let y = p.position.y / l;
                let two_pi = 2.0 * std::f64::consts::PI;
                Vec3::new(
                    b_mag * (two_pi * y).sin(),
                    b_mag * (two_pi * x).cos(),
                    0.0,
                )
            }
        };
    }
}

/// Genera un número pseudo-aleatorio ∈ [−1, 1] a partir de una semilla u64.
#[inline]
fn lcg_uniform(seed: u64) -> f64 {
    let s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (s >> 33) as u32;
    (bits as f64 / u32::MAX as f64) * 2.0 - 1.0
}

/// Calcula el paso de tiempo máximo permitido por el criterio CFL de Alfvén (Phase 127).
///
/// `dt_A = cfl × min_i(h_i) / max_i(v_{A,i})`
///
/// donde `v_{A,i} = |B_i| / sqrt(μ₀ ρ_i)`.
///
/// Retorna `f64::INFINITY` si no hay partículas de gas o si todos los B son cero.
pub fn alfven_dt(particles: &[Particle], cfl: f64) -> f64 {
    let mut h_min = f64::INFINITY;
    let mut v_a_max = 0.0_f64;

    for p in particles.iter() {
        if p.ptype != ParticleType::Gas { continue; }
        let h = p.smoothing_length.max(1e-10);
        let rho = (p.mass / (h * h * h)).max(1e-30);
        let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
        let v_a = (b2 / (MU0 * rho)).sqrt();

        h_min = h_min.min(h);
        v_a_max = v_a_max.max(v_a);
    }

    if v_a_max < 1e-30 || !h_min.is_finite() {
        return f64::INFINITY;
    }
    cfl * h_min / v_a_max
}

/// Aplica resistividad numérica artificial al campo magnético (Phase 135).
///
/// Suaviza las discontinuidades de B usando el esquema de Price (2008):
///
/// ```text
/// (∂B_i/∂t)_η = η_art × Σ_j m_j/ρ_j × (B_j − B_i) × 2|∇W_ij|/|r_ij|
/// ```
///
/// donde `η_art = alpha_b × h_i × v_sig` y `v_sig = |v_ij|` es la señal de velocidad relativa.
///
/// Con `alpha_b = 0.0` es un no-op.
#[allow(clippy::needless_range_loop)]
pub fn apply_artificial_resistivity(particles: &mut [Particle], alpha_b: f64, dt: f64) {
    if alpha_b <= 0.0 { return; }
    let n = particles.len();
    if n == 0 { return; }

    let mut db = vec![Vec3::zero(); n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas { continue; }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let b_i = particles[i].b_field;
        let vel_i = particles[i].velocity;

        for j in 0..n {
            if i == j { continue; }
            if particles[j].ptype != ParticleType::Gas { continue; }

            let dx = particles[j].position.x - pos_i.x;
            let dy = particles[j].position.y - pos_i.y;
            let dz = particles[j].position.z - pos_i.z;
            let r2 = dx*dx + dy*dy + dz*dz;
            let r = r2.sqrt();
            if r < 1e-14 { continue; }

            let h_j = particles[j].smoothing_length.max(1e-10);
            let h_avg = 0.5 * (h_i + h_j);
            if r > 2.0 * h_avg { continue; }

            // Velocidad de señal: diferencia relativa de velocidades
            let dvx = particles[j].velocity.x - vel_i.x;
            let dvy = particles[j].velocity.y - vel_i.y;
            let dvz = particles[j].velocity.z - vel_i.z;
            let v_sig = (dvx*dvx + dvy*dvy + dvz*dvz).sqrt();

            // Resistividad artificial: η_art = alpha_b × h_i × v_sig
            let eta_art = alpha_b * h_i * v_sig;

            // Gradiente del kernel: |∇W| ≈ |dW/dr| ≈ 1/(h⁴) simplificado
            let q = r / h_avg;
            let dw_dr = if q <= 1.0 {
                -3.0 * q * (2.0 - q) / (h_avg.powi(4))
            } else if q <= 2.0 {
                -3.0 * (2.0 - q).powi(2) / (2.0 * h_avg.powi(4))
            } else {
                0.0
            };
            let grad_w_mag = dw_dr.abs();

            let rho_j = (particles[j].mass / (h_j * h_j * h_j)).max(1e-30);
            let factor = eta_art * particles[j].mass / rho_j * 2.0 * grad_w_mag / r;

            let db_x = factor * (particles[j].b_field.x - b_i.x);
            let db_y = factor * (particles[j].b_field.y - b_i.y);
            let db_z = factor * (particles[j].b_field.z - b_i.z);

            db[i].x += db_x;
            db[i].y += db_y;
            db[i].z += db_z;
        }
    }

    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].b_field.x += db[i].x * dt;
            particles[i].b_field.y += db[i].y * dt;
            particles[i].b_field.z += db[i].z * dt;
        }
    }
}
