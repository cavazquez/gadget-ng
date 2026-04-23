//! Renderizado PPM (Portable Pixel Map) sin dependencias externas.
//!
//! El formato PPM binario (P6) es trivial de escribir:
//! ```text
//! P6\n
//! <width> <height>\n
//! 255\n
//! <raw RGB bytes en orden raster>
//! ```
//!
//! Sin dependencias externas (`png`, `image`, etc.): ideal para entornos sin X11
//! o donde el binario final debe ser mínimo.
//!
//! ## Uso
//!
//! ```rust
//! use gadget_ng_vis::{render_ppm, write_ppm};
//! use gadget_ng_core::Vec3;
//!
//! let positions = vec![Vec3::new(25.0, 25.0, 0.0), Vec3::new(75.0, 75.0, 0.0)];
//! let pixels = render_ppm(&positions, 100.0, 128, 128);
//! // write_ppm(std::path::Path::new("/tmp/frame.ppm"), &pixels, 128, 128).unwrap();
//! ```

use gadget_ng_core::Vec3;
use std::io::{self, Write};
use std::path::Path;

/// Renderiza partículas como imagen PPM (proyección ortográfica en el plano XY).
///
/// # Parámetros
/// - `positions`: posiciones de las partículas (cualquier unidad, pero consistente con `box_size`).
/// - `box_size`: tamaño de la caja. Las posiciones deben estar en `[0, box_size)`.
/// - `width`, `height`: dimensiones en píxeles.
///
/// # Retorna
/// Vector de bytes RGB planos (3 × width × height bytes),
/// en orden raster de arriba-izquierda a abajo-derecha.
/// Listo para pasarse directamente a `write_ppm`.
///
/// La intensidad de cada pixel se satura a blanco si hay ≥1 partícula;
/// el fondo es negro.
pub fn render_ppm(positions: &[Vec3], box_size: f64, width: usize, height: usize) -> Vec<u8> {
    let n_pixels = width * height;
    let mut pixels = vec![0u8; n_pixels * 3];

    if box_size <= 0.0 || width == 0 || height == 0 {
        return pixels;
    }

    let scale_x = width as f64 / box_size;
    let scale_y = height as f64 / box_size;

    for p in positions {
        // Proyección XY; Y invertido para convención raster (Y=0 arriba).
        let ix = (p.x * scale_x).floor() as isize;
        let iy = (height as f64 - 1.0 - p.y * scale_y).floor() as isize;

        if ix >= 0 && ix < width as isize && iy >= 0 && iy < height as isize {
            let idx = (iy as usize * width + ix as usize) * 3;
            pixels[idx] = 255;
            pixels[idx + 1] = 255;
            pixels[idx + 2] = 255;
        }
    }

    pixels
}

/// Escribe un buffer RGB en formato PPM binario (P6) a un archivo.
///
/// # Parámetros
/// - `path`: ruta del archivo de salida (se creará o sobreescribirá).
/// - `pixels`: buffer RGB plano de 3 × width × height bytes.
/// - `width`, `height`: dimensiones en píxeles.
///
/// # Errores
/// Propaga errores de I/O del sistema de archivos.
pub fn write_ppm(path: &Path, pixels: &[u8], width: usize, height: usize) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = io::BufWriter::new(file);

    // Cabecera PPM P6.
    write!(writer, "P6\n{width} {height}\n255\n")?;
    writer.write_all(pixels)?;
    writer.flush()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    #[test]
    fn ppm_empty_is_black() {
        let pixels = render_ppm(&[], 100.0, 32, 32);
        assert_eq!(pixels.len(), 32 * 32 * 3);
        assert!(
            pixels.iter().all(|&b| b == 0),
            "imagen vacía debe ser negra"
        );
    }

    #[test]
    fn ppm_particle_at_origin_is_white() {
        let pos = vec![Vec3::new(0.0, 0.0, 0.0)];
        let pixels = render_ppm(&pos, 100.0, 32, 32);
        // Pixel (0, 0) debe ser blanco (raster: arriba-izquierda).
        // Y=0 → iy = 31 (invertido), ix=0.
        let idx = 31 * 32 * 3;
        assert_eq!(pixels[idx], 255);
        assert_eq!(pixels[idx + 1], 255);
        assert_eq!(pixels[idx + 2], 255);
    }

    #[test]
    fn ppm_size_correct() {
        let pos = vec![Vec3::new(50.0, 50.0, 0.0)];
        let w = 64usize;
        let h = 48usize;
        let pixels = render_ppm(&pos, 100.0, w, h);
        assert_eq!(pixels.len(), w * h * 3);
    }

    #[test]
    fn ppm_write_and_read_back() {
        use std::io::Read;

        let pos = vec![Vec3::new(10.0, 20.0, 0.0)];
        let w = 32usize;
        let h = 32usize;
        let pixels = render_ppm(&pos, 100.0, w, h);

        let tmp = std::env::temp_dir().join("gadget_ng_ppm_test.ppm");
        write_ppm(&tmp, &pixels, w, h).expect("escritura PPM no debe fallar");

        let mut buf = Vec::new();
        std::fs::File::open(&tmp)
            .expect("debe poder abrir el PPM escrito")
            .read_to_end(&mut buf)
            .unwrap();

        // Verificar cabecera "P6"
        assert!(buf.starts_with(b"P6\n"), "cabecera PPM debe empezar con P6");
        // Verificar tamaño total: cabecera + datos
        assert!(
            buf.len() > w * h * 3,
            "archivo debe contener cabecera + datos"
        );

        // Limpiar
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn ppm_particle_out_of_bounds_ignored() {
        let pos = vec![
            Vec3::new(150.0, 50.0, 0.0), // fuera (x > box_size)
            Vec3::new(-1.0, 50.0, 0.0),  // fuera (x < 0)
            Vec3::new(50.0, 50.0, 0.0),  // dentro
        ];
        let pixels = render_ppm(&pos, 100.0, 32, 32);
        let n_white = pixels.chunks(3).filter(|c| c[0] == 255).count();
        assert_eq!(
            n_white, 1,
            "solo 1 partícula dentro de la caja debe aparecer"
        );
    }
}
