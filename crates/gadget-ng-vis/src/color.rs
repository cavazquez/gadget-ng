//! Mapas de color para partículas.

use gadget_ng_core::Vec3;

/// Modo de coloración de partículas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorMode {
    /// Color uniforme blanco.
    #[default]
    White,
    /// Color por velocidad (escalar = norma).
    Velocity,
    /// Color por densidad SPH (valor pasado externamente).
    Density,
}

/// Convierte un valor en [0, 1] a color RGB usando el mapa "Viridis" simplificado
/// (azul → verde → amarillo).
pub fn viridis(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    // Tres segmentos lineales: azul→verde (0..0.5), verde→amarillo (0.5..1)
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (0.0, s, 1.0 - s)
    } else {
        let s = (t - 0.5) * 2.0;
        (s, 1.0, 0.0)
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// Devuelve el color [R, G, B] para una partícula dado su color-mode y el valor escalar.
///
/// - `mode == White`: siempre blanco.
/// - `mode == Velocity`: colorea por `value / value_max`.
/// - `mode == Density`: ídem.
pub fn particle_color(mode: ColorMode, value: f64, value_max: f64) -> [u8; 3] {
    match mode {
        ColorMode::White => [255, 255, 255],
        ColorMode::Velocity | ColorMode::Density => {
            let t = if value_max > 0.0 {
                value / value_max
            } else {
                0.0
            };
            viridis(t)
        }
    }
}

/// Calcula las velocidades escalares (norma) de cada partícula desde Vec3.
pub fn velocity_scalars(velocities: &[Vec3]) -> Vec<f64> {
    velocities.iter().map(|v| v.norm()).collect()
}
