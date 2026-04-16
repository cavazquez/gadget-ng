//! Renderer principal que combina proyección, coloración y canvas CPU.
//!
//! `Renderer` es el punto de entrada público para crear frames de visualización.
//! Intenta usar wgpu si está disponible; cae en el backend CPU si no hay GPU.

use crate::canvas::CpuCanvas;
use crate::color::{velocity_scalars, ColorMode};
use crate::export::save_png;
use crate::projection::Projection;
use gadget_ng_core::Vec3;
use std::path::Path;

/// Configuración del renderer.
#[derive(Clone, Debug)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub projection: Projection,
    pub color_mode: ColorMode,
    pub box_size: f64,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 800,
            projection: Projection::XY,
            color_mode: ColorMode::Velocity,
            box_size: 1.0,
        }
    }
}

/// Renderer de partículas N-body a imagen PNG.
///
/// Renderiza partículas como puntos de 1 px coloreados según `color_mode`.
/// Llama a [`Renderer::render_frame`] para generar un frame y
/// [`Renderer::save_frame`] para guardarlo a disco.
pub struct Renderer {
    config: RendererConfig,
    canvas: CpuCanvas,
}

impl Renderer {
    /// Crea un nuevo renderer con la configuración dada.
    pub fn new(config: RendererConfig) -> Self {
        let canvas = CpuCanvas::new(config.width, config.height);
        Self { config, canvas }
    }

    /// Renderiza las partículas sobre el canvas interno.
    ///
    /// Los `scalars` se calculan automáticamente a partir del `color_mode`:
    /// - `Velocity`: norma del vector velocidad.
    /// - `Density` / `White`: zeros (el color se pasa externo si se desea via `render_frame_with_scalars`).
    pub fn render_frame(&mut self, positions: &[Vec3], velocities: &[Vec3]) {
        self.canvas.clear();
        let scalars = match self.config.color_mode {
            ColorMode::Velocity => velocity_scalars(velocities),
            _ => vec![0.0_f64; positions.len()],
        };
        let scalar_max = scalars.iter().cloned().fold(0.0_f64, f64::max);
        self.canvas.render(
            positions,
            &scalars,
            scalar_max,
            &self.config.projection,
            self.config.color_mode,
            self.config.box_size,
        );
    }

    /// Renderiza con escalares externos (e.g., densidad SPH ya calculada).
    pub fn render_frame_with_scalars(&mut self, positions: &[Vec3], scalars: &[f64]) {
        self.canvas.clear();
        let scalar_max = scalars.iter().cloned().fold(0.0_f64, f64::max);
        self.canvas.render(
            positions,
            scalars,
            scalar_max,
            &self.config.projection,
            self.config.color_mode,
            self.config.box_size,
        );
    }

    /// Guarda el frame actual como PNG en `path`.
    pub fn save_frame(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        save_png(&self.canvas, path)
    }

    /// Acceso al canvas (útil para tests).
    pub fn canvas(&self) -> &CpuCanvas {
        &self.canvas
    }

    pub fn config(&self) -> &RendererConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;
    use tempfile::tempdir;

    fn random_particles(n: usize, box_size: f64) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut rng = 0xdeadbeef_u64;
        let mut lcg = || -> f64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f64 / u32::MAX as f64
        };
        let pos: Vec<Vec3> = (0..n)
            .map(|_| Vec3::new(lcg() * box_size, lcg() * box_size, lcg() * box_size))
            .collect();
        let vel: Vec<Vec3> = (0..n)
            .map(|_| Vec3::new(lcg() - 0.5, lcg() - 0.5, lcg() - 0.5))
            .collect();
        (pos, vel)
    }

    #[test]
    fn renderer_draws_particles() {
        let cfg = RendererConfig {
            width: 128,
            height: 128,
            box_size: 1.0,
            projection: Projection::XY,
            color_mode: ColorMode::White,
        };
        let mut r = Renderer::new(cfg);
        let (pos, vel) = random_particles(100, 1.0);
        r.render_frame(&pos, &vel);
        assert!(r.canvas().non_black_pixels() > 0);
    }

    #[test]
    fn renderer_saves_png() {
        let dir = tempdir().unwrap();
        let cfg = RendererConfig {
            width: 64,
            height: 64,
            box_size: 1.0,
            ..Default::default()
        };
        let mut r = Renderer::new(cfg);
        let (pos, vel) = random_particles(50, 1.0);
        r.render_frame(&pos, &vel);
        let path = dir.path().join("frame.png");
        r.save_frame(&path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn velocity_color_mode_produces_colored_output() {
        let cfg = RendererConfig {
            width: 64,
            height: 64,
            box_size: 1.0,
            projection: Projection::XY,
            color_mode: ColorMode::Velocity,
        };
        let mut r = Renderer::new(cfg);
        let pos = vec![Vec3::new(0.5, 0.5, 0.0)];
        let vel = vec![Vec3::new(1.0, 0.0, 0.0)]; // v != 0 → color != negro
        r.render_frame(&pos, &vel);
        assert!(r.canvas().non_black_pixels() > 0);
    }
}
