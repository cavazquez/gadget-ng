use crate::vec3::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particle {
    /// Índice global único en [0, N) usado para orden estable MPI/serial.
    pub global_id: usize,
    pub mass: f64,
    pub position: Vec3,
    pub velocity: Vec3,
    #[serde(skip)]
    pub acceleration: Vec3,
}

impl Particle {
    pub fn new(global_id: usize, mass: f64, position: Vec3, velocity: Vec3) -> Self {
        Self {
            global_id,
            mass,
            position,
            velocity,
            acceleration: Vec3::zero(),
        }
    }
}
