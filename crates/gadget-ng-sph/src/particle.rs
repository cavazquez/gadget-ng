//! Tipos de partícula extendidos para SPH.
use gadget_ng_core::Vec3;
use serde::{Deserialize, Serialize};

/// Tipo de partícula: materia oscura o gas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ParticleType {
    #[default]
    DarkMatter,
    Gas,
}

/// Datos termodinámicos de una partícula de gas.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GasData {
    /// Energía interna específica (por unidad de masa) [km²/s²].
    pub u: f64,
    /// Densidad SPH estimada localmente [M_sun / kpc³ o unidades internas].
    pub rho: f64,
    /// Presión P = (γ-1) ρ u.
    pub pressure: f64,
    /// Radio de suavizado SPH h_sml.
    pub h_sml: f64,
    /// Aceleración SPH (excluye gravedad).
    pub acc_sph: Vec3,
    /// Derivada de la energía interna du/dt.
    pub du_dt: f64,
}

impl GasData {
    /// Inicializa con energía interna `u0` y radio de suavizado inicial `h0`.
    pub fn new(u0: f64, h0: f64) -> Self {
        Self {
            u: u0,
            rho: 0.0,
            pressure: 0.0,
            h_sml: h0,
            acc_sph: Vec3::zero(),
            du_dt: 0.0,
        }
    }
}

/// Partícula SPH completa: propiedades dinámicas + datos de gas.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SphParticle {
    pub global_id: usize,
    pub ptype: ParticleType,
    pub mass: f64,
    pub position: Vec3,
    pub velocity: Vec3,
    #[serde(skip)]
    pub acceleration: Vec3,
    /// `Some(...)` sólo cuando `ptype == Gas`.
    pub gas: Option<GasData>,
}

impl SphParticle {
    pub fn new_dm(id: usize, mass: f64, pos: Vec3, vel: Vec3) -> Self {
        Self {
            global_id: id,
            ptype: ParticleType::DarkMatter,
            mass,
            position: pos,
            velocity: vel,
            acceleration: Vec3::zero(),
            gas: None,
        }
    }

    pub fn new_gas(id: usize, mass: f64, pos: Vec3, vel: Vec3, u0: f64, h0: f64) -> Self {
        Self {
            global_id: id,
            ptype: ParticleType::Gas,
            mass,
            position: pos,
            velocity: vel,
            acceleration: Vec3::zero(),
            gas: Some(GasData::new(u0, h0)),
        }
    }
}
