use crate::vec3::Vec3;
use serde::{Deserialize, Serialize};

/// Tipo de partícula: materia oscura, gas SPH o estrella.
///
/// `DarkMatter` (default) → campos SPH (`internal_energy`, `smoothing_length`) son `0.0`.
/// `Gas` → partícula termodinámica; usa `internal_energy` y `smoothing_length`.
/// `Star` → partícula estelar (Phase 109): gravedad sí, SPH no; tiene `metallicity` y `stellar_age`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ParticleType {
    #[default]
    DarkMatter,
    Gas,
    /// Partícula estelar formada por spawning desde gas (Phase 109).
    Star,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particle {
    /// Índice global único en [0, N) usado para orden estable MPI/serial.
    pub global_id: usize,
    pub mass: f64,
    pub position: Vec3,
    pub velocity: Vec3,
    #[serde(skip)]
    pub acceleration: Vec3,
    /// Tipo de partícula (Phase 66). Default: `DarkMatter`.
    #[serde(default)]
    pub ptype: ParticleType,
    /// Energía interna específica u [unidades internas]. `0.0` para DM.
    #[serde(default)]
    pub internal_energy: f64,
    /// Radio de suavizado SPH h_sml [unidades internas]. `0.0` para DM.
    #[serde(default)]
    pub smoothing_length: f64,
    /// Fracción de masa en metales Z ∈ [0, 1] (Phase 109). `0.0` para DM primordial.
    #[serde(default)]
    pub metallicity: f64,
    /// Edad de la partícula estelar en Gyr desde su formación (Phase 109).
    /// `0.0` para DM y gas.
    #[serde(default)]
    pub stellar_age: f64,
    /// Energía interna de la componente fría del ISM [unidades internas] (Phase 114).
    /// Usado en el modelo multifase de Springel & Hernquist (2003).
    /// `0.0` para DM, estrellas y gas sin modelo ISM.
    #[serde(default)]
    pub u_cold: f64,
    /// Energía de rayos cósmicos específica [(km/s)²] (Phase 117).
    /// `0.0` para DM y estrellas.
    #[serde(default)]
    pub cr_energy: f64,
}

impl Particle {
    pub fn new(global_id: usize, mass: f64, position: Vec3, velocity: Vec3) -> Self {
        Self {
            global_id,
            mass,
            position,
            velocity,
            acceleration: Vec3::zero(),
            ptype: ParticleType::DarkMatter,
            internal_energy: 0.0,
            smoothing_length: 0.0,
            metallicity: 0.0,
            stellar_age: 0.0,
            u_cold: 0.0,
            cr_energy: 0.0,
        }
    }

    /// Crea una partícula de gas SPH.
    pub fn new_gas(
        global_id: usize,
        mass: f64,
        position: Vec3,
        velocity: Vec3,
        internal_energy: f64,
        smoothing_length: f64,
    ) -> Self {
        Self {
            global_id,
            mass,
            position,
            velocity,
            acceleration: Vec3::zero(),
            ptype: ParticleType::Gas,
            internal_energy,
            smoothing_length,
            metallicity: 0.0,
            stellar_age: 0.0,
            u_cold: 0.0,
            cr_energy: 0.0,
        }
    }

    /// Crea una partícula estelar (Phase 109).
    ///
    /// Las estrellas participan en gravedad pero no en SPH.
    /// Heredan `metallicity` del gas del que se formaron y registran `stellar_age = 0`.
    pub fn new_star(
        global_id: usize,
        mass: f64,
        position: Vec3,
        velocity: Vec3,
        metallicity: f64,
    ) -> Self {
        Self {
            global_id,
            mass,
            position,
            velocity,
            acceleration: Vec3::zero(),
            ptype: ParticleType::Star,
            internal_energy: 0.0,
            smoothing_length: 0.0,
            metallicity,
            stellar_age: 0.0,
            u_cold: 0.0,
            cr_energy: 0.0,
        }
    }

    #[inline]
    pub fn is_gas(&self) -> bool {
        self.ptype == ParticleType::Gas
    }

    /// Retorna `true` si la partícula es una estrella (Phase 109).
    #[inline]
    pub fn is_star(&self) -> bool {
        self.ptype == ParticleType::Star
    }
}
