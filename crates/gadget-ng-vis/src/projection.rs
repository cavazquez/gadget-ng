//! Proyecciones 3D → 2D para visualización de partículas.

use gadget_ng_core::Vec3;

/// Modo de proyección.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Projection {
    /// Proyección ortográfica XY (ignora Z).
    #[default]
    XY,
    /// Proyección ortográfica XZ (ignora Y).
    XZ,
    /// Proyección ortográfica YZ (ignora X).
    YZ,
    /// Proyección en perspectiva simple (cámara en z = -fov, mira hacia +z).
    Perspective {
        /// Distancia focal (distancia de la cámara al plano de proyección).
        fov: f64,
        /// Distancia de la cámara al centro de la escena.
        camera_z: f64,
    },
}

impl Projection {
    /// Proyecta `pos` al plano 2D.  Retorna `(x, y)` en coordenadas de mundo (sin normalizar).
    pub fn project(&self, pos: Vec3) -> (f64, f64) {
        match self {
            Projection::XY => (pos.x, pos.y),
            Projection::XZ => (pos.x, pos.z),
            Projection::YZ => (pos.y, pos.z),
            Projection::Perspective { fov, camera_z } => {
                let dz = camera_z - pos.z;
                if dz.abs() < 1e-10 {
                    (pos.x, pos.y)
                } else {
                    let scale = fov / dz;
                    (pos.x * scale, pos.y * scale)
                }
            }
        }
    }

    /// Convierte coordenadas de mundo (xw, yw) a píxeles de imagen `(px, py)`.
    ///
    /// `box_min` y `box_max` son los límites del dominio a mapear.
    /// `width` y `height` son las dimensiones de la imagen en píxeles.
    pub fn world_to_pixel(
        &self,
        xw: f64,
        yw: f64,
        box_min: (f64, f64),
        box_max: (f64, f64),
        width: u32,
        height: u32,
    ) -> Option<(u32, u32)> {
        let (x0, y0) = box_min;
        let (x1, y1) = box_max;
        let dx = x1 - x0;
        let dy = y1 - y0;
        if dx <= 0.0 || dy <= 0.0 { return None; }
        let px = ((xw - x0) / dx * width as f64) as i64;
        let py = ((yw - y0) / dy * height as f64) as i64;
        if px < 0 || py < 0 || px >= width as i64 || py >= height as i64 {
            return None;
        }
        // Y invertida (imagen: y=0 arriba, mundo: y=0 abajo).
        let py_img = height as i64 - 1 - py;
        Some((px as u32, py_img as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    #[test]
    fn xy_projection_drops_z() {
        let proj = Projection::XY;
        let (x, y) = proj.project(Vec3::new(1.0, 2.0, 99.0));
        assert_eq!(x, 1.0);
        assert_eq!(y, 2.0);
    }

    #[test]
    fn world_to_pixel_center() {
        let proj = Projection::XY;
        let (px, py) = proj
            .world_to_pixel(0.5, 0.5, (0.0, 0.0), (1.0, 1.0), 100, 100)
            .unwrap();
        assert_eq!(px, 50);
        assert_eq!(py, 49); // y invertida
    }
}
