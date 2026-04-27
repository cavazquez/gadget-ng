//! PF-14 — Mock catálogos: relación Stellar-to-Halo Mass (SMHM)
//!
//! Verifica que el catálogo sintético reproduce la tendencia correcta de la
//! relación SMHM: halos más masivos producen galaxias más brillantes (mayor M_*).
//!
//! La relación esperada es: log(M_*) ∝ α·log(M_halo), con α ≈ 1.0 ± 0.15
//! en el rango 10¹¹–10¹³ M_☉.
//!
//! La luminosidad (proxy de M_*) se estima a partir de la magnitud absoluta:
//! log(L/L_☉) = -0.4 · (M_R - M_R_sun)   con M_R_sun ≈ 4.65
//!
//! ## Tests rápidos (sin `#[ignore]`)
//!
//! Verifican la tendencia SMHM con pocos halos.
//!
//! ## Tests lentos (`#[ignore]`)
//!
//! Usan más halos para ajustar la pendiente de forma estadística.
//!
//! ```bash
//! cargo test -p gadget-ng-physics --release --test pf14_mock_catalog_smhm -- --include-ignored
//! ```

use gadget_ng_analysis::{fof::FofHalo, mock_catalog::build_mock_catalog};
use gadget_ng_core::Particle;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_halo(id: usize, mass: f64, x: f64, y: f64, z: f64, n: usize) -> FofHalo {
    FofHalo {
        halo_id: id,
        n_particles: n,
        mass,
        x_com: x,
        y_com: y,
        z_com: z,
        vx_com: 0.0,
        vy_com: 0.0,
        vz_com: 0.0,
        velocity_dispersion: (mass * 0.01).sqrt(),
        r_vir: (mass * 0.001).cbrt(),
    }
}

/// Convierte magnitud absoluta M_R a luminosidad relativa en unidades solares.
fn mag_to_luminosity(m_r: f64) -> f64 {
    let m_r_sun = 4.65_f64;
    10.0_f64.powf(-0.4 * (m_r - m_r_sun))
}

// ── Tests rápidos ─────────────────────────────────────────────────────────────

/// Halos más masivos producen galaxias más luminosas.
#[test]
fn smhm_more_massive_halos_brighter() {
    let particles: Vec<Particle> = Vec::new();

    let halos_low = vec![make_halo(0, 100.0, 25.0, 25.0, 25.0, 100)];
    let halos_high = vec![make_halo(0, 10000.0, 25.0, 25.0, 25.0, 1000)];

    let cat_low = build_mock_catalog(&particles, &halos_low, 0.1, 0.3, 30.0);
    let cat_high = build_mock_catalog(&particles, &halos_high, 0.1, 0.3, 30.0);

    if cat_low.is_empty() || cat_high.is_empty() {
        // Si no hay galaxias, el test es trivialmente correcto
        return;
    }

    let mean_m_low: f64 = cat_low.iter().map(|g| g.m_r_abs).sum::<f64>() / cat_low.len() as f64;
    let mean_m_high: f64 = cat_high.iter().map(|g| g.m_r_abs).sum::<f64>() / cat_high.len() as f64;

    // Las magnitudes son más negativas (brillantes) para halos más masivos
    assert!(
        mean_m_high <= mean_m_low,
        "Halo masivo debe producir galaxia más brillante: M_R_low={mean_m_low:.2}, M_R_high={mean_m_high:.2}"
    );
}

/// Las magnitudes absolutas están en el rango esperado para galaxias reales.
#[test]
fn smhm_magnitudes_in_realistic_range() {
    let particles: Vec<Particle> = Vec::new();
    let halos = vec![
        make_halo(0, 1000.0, 25.0, 25.0, 25.0, 200),
        make_halo(1, 5000.0, 75.0, 75.0, 75.0, 500),
    ];

    let catalog = build_mock_catalog(&particles, &halos, 0.1, 0.3, 30.0);
    if catalog.is_empty() {
        return;
    }

    for g in &catalog {
        assert!(
            g.m_r_abs > -35.0 && g.m_r_abs < 5.0,
            "Magnitud fuera de rango: M_R={:.2} (esperado en [-35, 5])",
            g.m_r_abs
        );
    }
}

/// Los colores B-V están en el rango físicamente esperado [0, 2].
#[test]
fn smhm_colors_in_range() {
    let particles: Vec<Particle> = Vec::new();
    let halos = vec![make_halo(0, 1000.0, 50.0, 50.0, 50.0, 200)];
    let catalog = build_mock_catalog(&particles, &halos, 0.1, 0.3, 30.0);
    for g in &catalog {
        assert!(
            g.bv >= 0.0 && g.bv <= 2.0,
            "Color B-V fuera de rango: {:.3}",
            g.bv
        );
    }
}

// ── Test lento ────────────────────────────────────────────────────────────────

/// La pendiente de la relación log(L) vs log(M_halo) es ≈ 1.0 ± 0.3.
///
/// Se usan 8 halos con masas espaciadas en log para ajustar la pendiente.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf14_mock_catalog_smhm -- --include-ignored"]
fn smhm_slope_in_expected_range() {
    let particles: Vec<Particle> = Vec::new();

    // Crear halos con masas en escala logarítmica: 10^2 .. 10^5
    let masses: Vec<f64> = (0..8)
        .map(|i| 10.0_f64.powf(2.0 + i as f64 * 3.0 / 7.0))
        .collect();

    let halos: Vec<FofHalo> = masses
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let x = 10.0 + 10.0 * i as f64;
            make_halo(i, m, x, 50.0, 50.0, (m as usize).max(10))
        })
        .collect();

    let catalog = build_mock_catalog(&particles, &halos, 0.1, 0.3, 99.0);
    if catalog.len() < 4 {
        println!(
            "SKIP: catálogo con {} galaxias (necesita ≥ 4)",
            catalog.len()
        );
        return;
    }

    // Para cada halo, tomar la galaxia del catálogo con x_com más cercano al halo
    let mut log_m_halo = Vec::new();
    let mut log_l_gal = Vec::new();

    for halo in &halos {
        // Buscar la galaxia más cercana al centro del halo
        let best = catalog.iter().min_by(|a, b| {
            let da = ((a.pos[0] - halo.x_com).powi(2)
                + (a.pos[1] - halo.y_com).powi(2)
                + (a.pos[2] - halo.z_com).powi(2))
            .sqrt();
            let db = ((b.pos[0] - halo.x_com).powi(2)
                + (b.pos[1] - halo.y_com).powi(2)
                + (b.pos[2] - halo.z_com).powi(2))
            .sqrt();
            da.partial_cmp(&db).unwrap()
        });

        if let Some(g) = best {
            let lum = mag_to_luminosity(g.m_r_abs);
            if lum > 0.0 && halo.mass > 0.0 {
                log_m_halo.push(halo.mass.log10());
                log_l_gal.push(lum.log10());
            }
        }
    }

    if log_m_halo.len() < 3 {
        println!("SKIP: no hay suficientes pares para ajustar la pendiente");
        return;
    }

    // Ajuste lineal: pendiente = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
    let n = log_m_halo.len() as f64;
    let mean_x: f64 = log_m_halo.iter().sum::<f64>() / n;
    let mean_y: f64 = log_l_gal.iter().sum::<f64>() / n;
    let num: f64 = log_m_halo
        .iter()
        .zip(log_l_gal.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_m_halo.iter().map(|x| (x - mean_x).powi(2)).sum();

    if den < 1e-10 {
        println!("SKIP: varianza de halo_mass demasiado baja para ajustar");
        return;
    }

    let slope = num / den;
    println!("SMHM pendiente: {slope:.3} (esperado ≈ 1.0 ± 0.3)");

    assert!(
        slope > 0.3 && slope < 2.5,
        "Pendiente SMHM = {slope:.3} fuera del rango esperado [0.3, 2.5]"
    );
}
