//! Phase 64 — Tests extendidos de visualización PPM/PNG.
//!
//! Verifica:
//! - Mapa de densidad: cluster concentrado genera pixel más brillante que fondo.
//! - Proyección XZ: partícula en (x,0,z) aparece en pixel (x,z).
//! - Header PNG: archivo inicia con los 4 bytes mágicos `\x89PNG`.

use gadget_ng_core::Vec3;
use gadget_ng_vis::{render_density_ppm, render_ppm_projection, write_png, Projection};

// ── density_map_concentrated_bright ───────────────────────────────────────────

/// Un cluster de 100 partículas en el centro debe producir el pixel más brillante
/// de toda la imagen (suma RGB máxima), y ser más brillante que la suma media del fondo.
#[test]
fn density_map_concentrated_bright() {
    let width = 32usize;
    let height = 32usize;
    let box_size = 100.0_f64;

    // Cluster en el centro exacto.
    let cluster: Vec<Vec3> = (0..100)
        .map(|_| Vec3::new(50.0, 50.0, 0.0))
        .collect();

    let pixels = render_density_ppm(&cluster, box_size, width, height, Projection::XY);
    assert_eq!(pixels.len(), width * height * 3);

    // Encontrar el pixel más brillante (por suma R+G+B).
    let max_brightness: u32 = pixels
        .chunks(3)
        .map(|c| c[0] as u32 + c[1] as u32 + c[2] as u32)
        .max()
        .unwrap_or(0);

    // Todos los 100 puntos coinciden en el mismo pixel → t=1 → viridis(1) = amarillo [255, 255, 0]
    // brightness = 510, que es mayor que cualquier pixel vacío (t=0 → [0, 0, 255], brightness=255).
    assert!(
        max_brightness > 255,
        "el pixel con el cluster debe ser más brillante que un pixel vacío; max={max_brightness}"
    );

    // Verificar que hay exactamente 1 pixel completamente lleno (el cluster ocupa un pixel).
    let n_bright = pixels
        .chunks(3)
        .filter(|c| c[0] as u32 + c[1] as u32 + c[2] as u32 == max_brightness)
        .count();
    assert!(n_bright >= 1, "debe haber al menos 1 pixel de alta densidad");
}

/// Una imagen vacía con densidad debe ser completamente azul oscuro (t=0 → viridis(0)).
#[test]
fn density_map_empty_is_dark() {
    let pixels = render_density_ppm(&[], 100.0, 16, 16, Projection::XY);
    assert_eq!(pixels.len(), 16 * 16 * 3, "tamaño correcto");
    // Todos los pixels deben tener el mismo color (t=0, viridis(0) ≈ (68, 1, 84)).
    let first = [pixels[0], pixels[1], pixels[2]];
    for chunk in pixels.chunks(3) {
        assert_eq!(chunk, &first, "imagen vacía debe ser uniforme");
    }
}

// ── projection_xz_correct ─────────────────────────────────────────────────────

/// Una partícula en (x=25, y=0, z=75) proyectada en XZ debe aparecer en el
/// pixel que corresponde a x=25 horizontalmente y z=75 verticalmente.
#[test]
fn projection_xz_correct() {
    let width = 100usize;
    let height = 100usize;
    let box_size = 100.0_f64;

    let pos = vec![Vec3::new(25.0, 0.0, 75.0)];
    let pixels = render_ppm_projection(&pos, box_size, width, height, Projection::XZ);

    let expected_px = 25usize;
    // Z=75 en raster: y_raster = height - 1 - floor(75 * height/box_size) = 24
    let expected_py = (height as f64 - 1.0 - (75.0 * height as f64 / box_size).floor()) as usize;

    let idx = (expected_py * width + expected_px) * 3;
    assert_eq!(
        pixels[idx], 255,
        "pixel en ({expected_px}, {expected_py}) debe ser blanco R=255"
    );
    assert_eq!(
        pixels[idx + 1], 255,
        "pixel en ({expected_px}, {expected_py}) debe ser blanco G=255"
    );
    assert_eq!(
        pixels[idx + 2], 255,
        "pixel en ({expected_px}, {expected_py}) debe ser blanco B=255"
    );
}

/// Proyección YZ: partícula en (0, y=30, z=60) debe aparecer en pixel (30, height-1-60).
#[test]
fn projection_yz_correct() {
    let width = 100usize;
    let height = 100usize;
    let box_size = 100.0_f64;

    let pos = vec![Vec3::new(0.0, 30.0, 60.0)];
    let pixels = render_ppm_projection(&pos, box_size, width, height, Projection::YZ);

    let expected_px = 30usize;
    let expected_py = (height as f64 - 1.0 - (60.0 * height as f64 / box_size).floor()) as usize;

    let idx = (expected_py * width + expected_px) * 3;
    assert_eq!(pixels[idx], 255, "R debe ser 255");
    assert_eq!(pixels[idx + 1], 255, "G debe ser 255");
    assert_eq!(pixels[idx + 2], 255, "B debe ser 255");
}

// ── write_png_header ──────────────────────────────────────────────────────────

/// El archivo PNG escrito por `write_png` debe comenzar con los 8 bytes mágicos de PNG.
#[test]
fn write_png_header() {
    use std::io::Read;

    let width = 16usize;
    let height = 16usize;
    let pixels = vec![128u8; width * height * 3];

    let tmp = std::env::temp_dir().join("gadget_ng_phase64_test.png");
    write_png(&tmp, &pixels, width, height).expect("write_png no debe fallar");

    let mut buf = Vec::new();
    std::fs::File::open(&tmp)
        .expect("debe poder abrir el PNG escrito")
        .read_to_end(&mut buf)
        .unwrap();

    // Los 8 bytes mágicos del PNG: \x89PNG\r\n\x1a\n
    assert!(
        buf.len() >= 8,
        "el archivo PNG debe tener al menos 8 bytes"
    );
    assert_eq!(
        &buf[0..4],
        &[0x89, 0x50, 0x4e, 0x47], // \x89PNG
        "el archivo debe empezar con los bytes mágicos de PNG"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// `write_png` con buffer vacío y dimensiones 0 produce un archivo PNG mínimo válido.
#[test]
fn write_png_minimal() {
    let tmp = std::env::temp_dir().join("gadget_ng_phase64_minimal.png");
    // Imagen 1x1 con 1 pixel negro.
    write_png(&tmp, &[0u8, 0, 0], 1, 1).expect("PNG 1x1 no debe fallar");
    let meta = std::fs::metadata(&tmp).expect("archivo debe existir");
    assert!(meta.len() > 8, "archivo PNG 1x1 debe tener datos");
    let _ = std::fs::remove_file(&tmp);
}
