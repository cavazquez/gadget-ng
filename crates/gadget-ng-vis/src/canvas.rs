//! Canvas de píxeles en CPU para renderizado de partículas sin GPU.
//!
//! El `CpuCanvas` acumula partículas como puntos de 1 píxel sobre un buffer RGBA.
//! Este backend es portable (sin requerir GPU) y se usa como fallback y para export PNG.

use crate::color::{particle_color, ColorMode};
use crate::projection::Projection;
use gadget_ng_core::Vec3;

/// Canvas RGBA en CPU.
pub struct CpuCanvas {
    pub width:  u32,
    pub height: u32,
    /// Buffer RGBA lineal (row-major, y=0 arriba).
    pub data: Vec<u8>,
}

impl CpuCanvas {
    /// Crea un canvas con fondo negro.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; (width * height * 4) as usize],
        }
    }

    /// Limpia el canvas a negro.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Dibuja una lista de partículas sobre el canvas.
    ///
    /// # Parámetros
    /// - `positions`: posiciones 3D de las partículas.
    /// - `scalars`: valores escalares usados para colorear (uno por partícula).
    /// - `scalar_max`: valor máximo de referencia para normalización de color.
    /// - `proj`: modo de proyección.
    /// - `color_mode`: cómo mapear `scalars` a color.
    /// - `box_size`: tamaño del dominio cúbico (para normalizar la proyección).
    pub fn render(
        &mut self,
        positions: &[Vec3],
        scalars: &[f64],
        scalar_max: f64,
        proj: &Projection,
        color_mode: ColorMode,
        box_size: f64,
    ) {
        // Bounding box del dominio proyectado.
        let box_min = (0.0, 0.0);
        let box_max = (box_size, box_size);

        for (i, &pos) in positions.iter().enumerate() {
            let (xw, yw) = proj.project(pos);
            let Some((px, py)) = proj.world_to_pixel(xw, yw, box_min, box_max, self.width, self.height)
            else {
                continue;
            };
            let scalar = scalars.get(i).copied().unwrap_or(0.0);
            let [r, g, b] = particle_color(color_mode, scalar, scalar_max);
            let idx = ((py * self.width + px) * 4) as usize;
            // Saturar: si ya hay un píxel brillante no lo oscurecer.
            self.data[idx]     = self.data[idx].max(r);
            self.data[idx + 1] = self.data[idx + 1].max(g);
            self.data[idx + 2] = self.data[idx + 2].max(b);
            self.data[idx + 3] = 255;
        }
    }

    /// Número de píxeles no negros (útil para tests).
    pub fn non_black_pixels(&self) -> usize {
        self.data.chunks(4).filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    #[test]
    fn render_single_particle_visible() {
        let mut canvas = CpuCanvas::new(64, 64);
        let pos = vec![Vec3::new(5.0, 5.0, 0.0)];
        canvas.render(&pos, &[0.0], 1.0, &Projection::XY, ColorMode::White, 10.0);
        assert!(canvas.non_black_pixels() > 0, "no se dibujó ningún píxel");
    }

    #[test]
    fn render_outside_box_clipped() {
        let mut canvas = CpuCanvas::new(64, 64);
        let pos = vec![Vec3::new(20.0, 20.0, 0.0)]; // fuera de box_size=10
        canvas.render(&pos, &[0.0], 1.0, &Projection::XY, ColorMode::White, 10.0);
        assert_eq!(canvas.non_black_pixels(), 0, "se dibujó partícula fuera del dominio");
    }
}
