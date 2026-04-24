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
    /// Presión P = (γ-1) ρ u  o  P = A ρ^γ  (depende del integrador activo).
    pub pressure: f64,
    /// Radio de suavizado SPH h_sml.
    pub h_sml: f64,
    /// Aceleración SPH (excluye gravedad).
    pub acc_sph: Vec3,
    /// Derivada de la energía interna du/dt (formulación de energía clásica).
    pub du_dt: f64,

    // ── Formulación de entropía (Springel & Hernquist 2002) ──────────────────
    /// Función entrópica A_i = P_i / ρ_i^γ = (γ-1) u_i / ρ_i^(γ-1).
    /// Calculada en `compute_density`; evoluciona con `da_dt`.
    pub entropy: f64,
    /// dA_i/dt por calentamiento viscoso (sólo flujos compresivos).
    pub da_dt: f64,

    // ── Limitador de Balsara (Balsara 1995) ──────────────────────────────────
    /// Factor Balsara f_i = |∇·v_i| / (|∇·v_i| + |∇×v_i| + ε c_s/h) ∈ [0,1].
    /// Suprime la viscosidad artificial en flujos de cizallamiento.
    pub balsara: f64,

    // ── Condición de Courant hidrodinámica ───────────────────────────────────
    /// Velocidad de señal máxima v_sig_max sobre todos los vecinos j.
    /// Determina dt_i^(hyd) = C_courant · h_i / max_vsig.
    pub max_vsig: f64,
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
            entropy: 0.0,
            da_dt: 0.0,
            balsara: 1.0,
            max_vsig: 0.0,
        }
    }

    /// Velocidad del sonido local: c_s = sqrt(γ P / ρ).
    #[inline]
    pub fn sound_speed(&self, gamma: f64) -> f64 {
        if self.rho > 0.0 {
            (gamma * self.pressure / self.rho).sqrt()
        } else {
            0.0
        }
    }

    /// Inicializa la función entrópica a partir de `u` y `rho` ya calculados.
    ///
    /// Debe llamarse después de `compute_density`.
    #[inline]
    pub fn init_entropy(&mut self, gamma: f64) {
        if self.rho > 0.0 {
            self.entropy = (gamma - 1.0) * self.u / self.rho.powf(gamma - 1.0);
        }
    }

    /// Actualiza `u` y `pressure` a partir de `entropy` y `rho` actuales.
    #[inline]
    pub fn sync_from_entropy(&mut self, gamma: f64) {
        self.pressure = self.entropy * self.rho.powf(gamma);
        if self.rho > 0.0 {
            self.u = self.entropy * self.rho.powf(gamma - 1.0) / (gamma - 1.0);
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
