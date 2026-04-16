//! Exportación de frames a PNG.

use crate::canvas::CpuCanvas;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Guarda el canvas como PNG en `path`.
///
/// El nombre sugerido para un frame de simulación es `snap_{step:06}.png`.
pub fn save_png(canvas: &CpuCanvas, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, canvas.width, canvas.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&canvas.data)?;
    Ok(())
}

/// Genera el nombre de archivo para el frame `step` dentro de `dir`.
pub fn frame_path(dir: &Path, step: u64) -> std::path::PathBuf {
    dir.join(format!("snap_{step:06}.png"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::CpuCanvas;
    use tempfile::tempdir;

    #[test]
    fn save_and_file_exists() {
        let dir = tempdir().unwrap();
        let path = frame_path(dir.path(), 42);
        let canvas = CpuCanvas::new(32, 32);
        save_png(&canvas, &path).expect("save_png falló");
        assert!(path.exists(), "el archivo PNG no fue creado");
        assert!(path.metadata().unwrap().len() > 0, "PNG vacío");
    }
}
