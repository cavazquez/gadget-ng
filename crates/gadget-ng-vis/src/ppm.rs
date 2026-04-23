//! Renderizado PPM (Portable Pixel Map) y PNG sin dependencias pesadas.
//!
//! El formato PPM binario (P6) es trivial de escribir:
//! ```text
//! P6\n
//! <width> <height>\n
//! 255\n
//! <raw RGB bytes en orden raster>
//! ```
//!
//! Para PNG se usa la crate `png` (ya dependencia de `gadget-ng-vis`).
//!
//! ## Funciones disponibles
//!
//! - [`render_ppm`]: proyección XY, puntos blancos (retrocompatible).
//! - [`render_ppm_projection`]: cualquier proyección (XY, XZ, YZ), puntos blancos.
//! - [`render_density_ppm`]: mapa de densidad logarítmica + colormap Viridis.
//! - [`write_ppm`]: escribe buffer RGB en formato PPM P6.
//! - [`write_png`]: escribe buffer RGB en formato PNG.

use gadget_ng_core::Vec3;
use std::io::{self, Write};
use std::path::Path;

use crate::{color::viridis, projection::Projection};

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

// ── Nuevas funciones Phase 64 ─────────────────────────────────────────────────

/// Renderiza partículas como imagen PPM con proyección configurable.
///
/// Igual que [`render_ppm`] pero permite elegir el plano de proyección:
/// - `Projection::XY`: usa coordenadas (x, y) — equivalente a `render_ppm`.
/// - `Projection::XZ`: usa coordenadas (x, z).
/// - `Projection::YZ`: usa coordenadas (y, z).
///
/// Las partículas fuera del rango `[0, box_size)` en ambos ejes se ignoran.
pub fn render_ppm_projection(
    positions: &[Vec3],
    box_size: f64,
    width: usize,
    height: usize,
    proj: Projection,
) -> Vec<u8> {
    let n_pixels = width * height;
    let mut pixels = vec![0u8; n_pixels * 3];

    if box_size <= 0.0 || width == 0 || height == 0 {
        return pixels;
    }

    let scale_x = width as f64 / box_size;
    let scale_y = height as f64 / box_size;

    for p in positions {
        let (px_world, py_world) = proj.project(*p);

        let ix = (px_world * scale_x).floor() as isize;
        // Y invertido: coordenada-mundo y=0 en la parte inferior; raster y=0 arriba.
        let iy = (height as f64 - 1.0 - py_world * scale_y).floor() as isize;

        if ix >= 0 && ix < width as isize && iy >= 0 && iy < height as isize {
            let idx = (iy as usize * width + ix as usize) * 3;
            pixels[idx] = 255;
            pixels[idx + 1] = 255;
            pixels[idx + 2] = 255;
        }
    }

    pixels
}

/// Renderiza un mapa de densidad proyectada en escala logarítmica con colormap Viridis.
///
/// Cada pixel acumula el número de partículas proyectadas en él.
/// El color se asigna según `log10(1 + count)` normalizado al máximo del frame,
/// usando el colormap Viridis (azul=vacío → amarillo=denso).
///
/// Mismo sistema de proyección configurable que [`render_ppm_projection`].
pub fn render_density_ppm(
    positions: &[Vec3],
    box_size: f64,
    width: usize,
    height: usize,
    proj: Projection,
) -> Vec<u8> {
    let n_pixels = width * height;
    let mut counts = vec![0u32; n_pixels];

    if box_size > 0.0 && width > 0 && height > 0 {
        let scale_x = width as f64 / box_size;
        let scale_y = height as f64 / box_size;

        for p in positions {
            let (px_world, py_world) = proj.project(*p);
            let ix = (px_world * scale_x).floor() as isize;
            let iy = (height as f64 - 1.0 - py_world * scale_y).floor() as isize;

            if ix >= 0 && ix < width as isize && iy >= 0 && iy < height as isize {
                counts[iy as usize * width + ix as usize] += 1;
            }
        }
    }

    // Normalizar en escala log10.
    let max_log = counts
        .iter()
        .map(|&c| (1.0 + c as f64).log10())
        .fold(0.0f64, f64::max);

    let mut pixels = vec![0u8; n_pixels * 3];
    for (i, &c) in counts.iter().enumerate() {
        let t = if max_log > 0.0 {
            (1.0 + c as f64).log10() / max_log
        } else {
            0.0
        };
        let [r, g, b] = viridis(t.clamp(0.0, 1.0));
        pixels[i * 3] = r;
        pixels[i * 3 + 1] = g;
        pixels[i * 3 + 2] = b;
    }

    pixels
}

/// Escribe un buffer RGB en formato PNG usando la crate `png`.
///
/// El buffer debe tener exactamente `width × height × 3` bytes.
///
/// # Errores
/// Propaga errores de I/O. Si `png::Encoder` falla, convierte el error a `io::Error`.
pub fn write_png(path: &Path, pixels: &[u8], width: usize, height: usize) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(writer, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);

    let mut png_writer = encoder
        .write_header()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    png_writer
        .write_image_data(pixels)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    Ok(())
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
