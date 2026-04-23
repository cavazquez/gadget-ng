use crate::vec3::Vec3;
use serde::{Deserialize, Serialize};

/// Tipo de partícula: materia oscura o gas SPH.
///
/// `DarkMatter` (default) → campos SPH (`internal_energy`, `smoothing_length`) son `0.0`.
/// `Gas` → partícula termodinámica; usa `internal_energy` y `smoothing_length`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ParticleType {
    #[default]
    DarkMatter,
    Gas,
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
        }
    }

    #[inline]
    pub fn is_gas(&self) -> bool {
        self.ptype == ParticleType::Gas
    }
}
