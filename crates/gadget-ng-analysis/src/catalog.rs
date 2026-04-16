//! Escritura y lectura de catálogos de halos y power spectrum en JSONL.

use crate::fof::FofHalo;
use crate::power_spectrum::PkBin;
use std::io::{self, Write};
use std::path::Path;
use std::{fs, io::BufRead};

// ── Catálogo de halos ─────────────────────────────────────────────────────────

/// Escribe un catálogo de halos en `<dir>/halo_catalog.jsonl`.
///
/// Cada línea es un JSON de `FofHalo`.
pub fn write_halo_catalog(dir: &Path, halos: &[FofHalo]) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join("halo_catalog.jsonl");
    let mut f = fs::File::create(&path)?;
    for h in halos {
        let line =         serde_json::to_string(h).map_err(io::Error::other)?;
        writeln!(f, "{line}")?;
    }
    Ok(())
}

/// Lee un catálogo de halos desde `<dir>/halo_catalog.jsonl`.
pub fn read_halo_catalog(dir: &Path) -> io::Result<Vec<FofHalo>> {
    let path = dir.join("halo_catalog.jsonl");
    let file = fs::File::open(&path)?;
    io::BufReader::new(file)
        .lines()
        .map(|l| {
            let line = l?;
            serde_json::from_str(&line).map_err(io::Error::other)
        })
        .collect()
}

// ── Power spectrum ─────────────────────────────────────────────────────────────

/// Escribe el power spectrum en `<dir>/power_spectrum.jsonl`.
///
/// Cada línea es un JSON de `PkBin`.
pub fn write_power_spectrum(dir: &Path, bins: &[PkBin]) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join("power_spectrum.jsonl");
    let mut f = fs::File::create(&path)?;
    for b in bins {
        let line =         serde_json::to_string(b).map_err(io::Error::other)?;
        writeln!(f, "{line}")?;
    }
    Ok(())
}

/// Lee el power spectrum desde `<dir>/power_spectrum.jsonl`.
pub fn read_power_spectrum(dir: &Path) -> io::Result<Vec<PkBin>> {
    let path = dir.join("power_spectrum.jsonl");
    let file = fs::File::open(&path)?;
    io::BufReader::new(file)
        .lines()
        .map(|l| {
            let line = l?;
            serde_json::from_str(&line).map_err(io::Error::other)
        })
        .collect()
}

// ── Función conveniente "analizar snapshot" ────────────────────────────────────

use gadget_ng_core::{Particle, Vec3};

/// Parámetros para el análisis de un snapshot.
#[derive(Debug, Clone)]
pub struct AnalysisParams {
    /// Tamaño de la caja (unidades internas).
    pub box_size: f64,
    /// Parámetro de enlace FoF (default 0.2).
    pub b: f64,
    /// Número mínimo de partículas por halo (default 20).
    pub min_particles: usize,
    /// Densidad crítica en unidades internas (0 → r_vir = r_max).
    pub rho_crit: f64,
    /// Resolución del grid para P(k) (default 64).
    pub pk_mesh: usize,
}

impl Default for AnalysisParams {
    fn default() -> Self {
        Self {
            box_size:      1.0,
            b:             0.2,
            min_particles: 20,
            rho_crit:      0.0,
            pk_mesh:       64,
        }
    }
}

/// Resultado del análisis de un snapshot.
pub struct AnalysisResult {
    pub halos:           Vec<FofHalo>,
    pub power_spectrum:  Vec<PkBin>,
}

/// Analiza un snapshot: ejecuta FoF + P(k) y devuelve el resultado.
pub fn analyse(particles: &[Particle], params: &AnalysisParams) -> AnalysisResult {
    let positions:  Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let velocities: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let masses:     Vec<f64>  = particles.iter().map(|p| p.mass).collect();

    let halos = crate::fof::find_halos(
        &positions,
        &velocities,
        &masses,
        params.box_size,
        params.b,
        params.min_particles,
        params.rho_crit,
    );

    let power_spectrum = crate::power_spectrum::power_spectrum(
        &positions,
        &masses,
        params.box_size,
        params.pk_mesh,
    );

    AnalysisResult { halos, power_spectrum }
}
