//! Mock catalogues con efectos de selección — Phase 154.
//!
//! ## Modelo
//!
//! Se generan catálogos galácticos sintéticos para comparar con surveys:
//!
//! 1. **Asignación stellar-to-halo mass** (SMHM): cada halo FoF de masa `M_h`
//!    recibe una galaxia de masa estelar `M_* = f_*(M_h)` usando la relación
//!    de Behroozi et al. (2013) simplificada:
//!
//!    ```text
//!    log10(M_*/M_h) = log10(ε·M1) - log10(M_h/M1) + α·log10(1 + M_h/M1) - δ
//!    ```
//!
//! 2. **Magnitud aparente**: se calcula a partir de la distancia luminosidad
//!    y una k-correction lineal con z.
//!
//! 3. **Selección por flujo**: corte en magnitud límite `m_lim`.
//!
//! 4. **C_l angular**: espectro de potencia angular `C_l` usando la proyección
//!    de Limber simplificada para catálogos 2D en la caja.
//!
//! ## Referencias
//!
//! - Behroozi, Wechsler & Conroy (2013) ApJ 770, 57.
//! - Peacock (1999) "Cosmological Physics", cap. 18.

use crate::fof::FofHalo;
use gadget_ng_core::Particle;

/// Galaxia en el mock catalogue (Phase 154).
#[derive(Debug, Clone, PartialEq)]
pub struct MockGalaxy {
    /// Posición comóvil [x, y, z] en unidades internas.
    pub pos: [f64; 3],
    /// Redshift observado (cosmológico; sin RSD en esta versión).
    pub z_obs: f64,
    /// Magnitud absoluta en banda R.
    pub m_r_abs: f64,
    /// Magnitud aparente en banda R (incluye distancia + k-correction).
    pub m_r_app: f64,
    /// Color B-V estimado desde la edad media del halo.
    pub bv: f64,
    /// SFR específica estimada [yr⁻¹].
    pub ssfr: f64,
    /// Metalicidad media del gas del halo.
    pub metallicity: f64,
    /// Masa del halo FoF en unidades internas.
    pub halo_mass: f64,
    /// Masa estelar estimada SMHM en unidades internas.
    pub stellar_mass: f64,
}

/// Relación SMHM simplificada de Behroozi+2013 (Phase 154).
///
/// Retorna `log10(M_*/M_h)`.
///
/// # Parámetros
/// - `log_mh`: `log10(M_h / M_pivot)` donde `M_pivot` es la masa de pivote
fn smhm_log_ratio(log_mh: f64) -> f64 {
    // Parámetros Behroozi+2013 simplificados a z=0
    const EPSILON: f64 = 0.023; // eficiencia de conversión pico
    const ALPHA: f64 = -1.412; // pendiente de baja masa
    const DELTA: f64 = 3.508; // contribución exponencial
    const GAMMA: f64 = 0.316; // forma de la parte exponencial

    // f(x) = -log10(10^(alpha*x) + 1) + delta * (log10(1 + exp(x)))^gamma / (1 + exp(10^(-x)))
    let x = log_mh;
    let f = -(10.0_f64.powf(ALPHA * x) + 1.0).log10()
        + DELTA
            * (1.0 + (x * std::f64::consts::LN_10).exp())
                .log10()
                .powf(GAMMA)
            / (1.0 + (-x * std::f64::consts::LN_10).exp() + 1.0);
    EPSILON.log10() + f
}

/// Distancia de luminosidad comóvil simplificada [Mpc/h] a redshift z (Phase 154).
///
/// Usa la aproximación de Pen (1999) para ΛCDM plano:
/// `d_L ≈ (c/H0) × (1+z) × [η(1) - η(a)]`
fn luminosity_distance(z: f64, omega_m: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let c_over_h0 = 2997.92; // c/H0 en Mpc/h (H0 = 100 km/s/Mpc)
    let n = 100_usize;
    let a_obs = 1.0 / (1.0 + z);
    let da = (1.0 - a_obs) / n as f64;
    let omega_lambda = 1.0 - omega_m;
    let mut integral = 0.0_f64;
    for i in 0..n {
        let a = a_obs + (i as f64 + 0.5) * da;
        let h_a = (omega_m / (a * a * a) + omega_lambda).sqrt();
        integral += da / (a * a * h_a);
    }
    c_over_h0 * (1.0 + z) * integral
}

/// Magnitud aparente en banda R a redshift z (Phase 154).
///
/// `m = M + 5 × log10(d_L / 10 pc) + K(z)`
/// con k-correction lineal `K(z) ≈ 1.8 × z` para galaxias tardías.
///
/// # Parámetros
/// - `m_abs`: magnitud absoluta en banda R
/// - `z`: redshift
/// - `omega_m`: Ω_m cosmológico
pub fn apparent_magnitude(m_abs: f64, z: f64, omega_m: f64) -> f64 {
    if z <= 0.0 {
        return m_abs;
    }
    let d_l_mpc = luminosity_distance(z, omega_m);
    let d_l_pc = d_l_mpc * 1.0e6;
    let mu = if d_l_pc > 0.0 {
        5.0 * (d_l_pc / 10.0).log10()
    } else {
        0.0
    };
    let k_corr = 1.8 * z;
    m_abs + mu + k_corr
}

/// Aplica el corte en magnitud límite (Phase 154).
///
/// Retorna `true` si la galaxia pasa el corte (es observable).
pub fn selection_flux_limit(m_r_apparent: f64, m_lim: f64) -> bool {
    m_r_apparent <= m_lim
}

/// Construye el mock catalogue a partir de halos FoF y partículas (Phase 154).
///
/// Cada halo recibe una galaxia central con masa estelar SMHM y propiedades
/// estimadas desde las partículas de gas/estrellas dentro del halo.
///
/// # Parámetros
/// - `particles`: slice de partículas de la simulación
/// - `halos`: halos FoF identificados (de `find_halos_combined`)
/// - `z_box`: redshift de la caja de simulación
/// - `omega_m`: parámetro cosmológico Ω_m
/// - `m_lim`: magnitud límite del survey (corte de flujo)
///
/// # Retorna
/// `Vec<MockGalaxy>` con solo las galaxias que pasan la selección.
pub fn build_mock_catalog(
    particles: &[Particle],
    halos: &[FofHalo],
    z_box: f64,
    omega_m: f64,
    m_lim: f64,
) -> Vec<MockGalaxy> {
    let mut catalog = Vec::with_capacity(halos.len());

    for halo in halos {
        let m_h = halo.mass;
        if m_h <= 0.0 {
            continue;
        }

        // Masa estelar SMHM
        const M_PIVOT: f64 = 1e12; // masa de pivote en M☉/h (unidades internas ~ 10¹⁰)
        let m_pivot_code = M_PIVOT / 1e10; // en unidades internas (M☉/h × 10⁻¹⁰)
        let log_mh = (m_h / m_pivot_code).log10();
        let log_ratio = smhm_log_ratio(log_mh);
        let stellar_mass = m_h * 10.0_f64.powf(log_ratio);

        // Propiedades del gas del halo — promedio simple sobre partículas cercanas al COM
        let com = [halo.x_com, halo.y_com, halo.z_com];
        let r_halo = halo.r_vir.max(0.1);
        let mut metal_sum = 0.0_f64;
        let mut metal_count = 0_usize;

        for p in particles {
            if !p.is_gas() {
                continue;
            }
            let dx = p.position.x - com[0];
            let dy = p.position.y - com[1];
            let dz = p.position.z - com[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < r_halo {
                metal_sum += p.metallicity;
                metal_count += 1;
            }
        }
        let metallicity = if metal_count > 0 {
            metal_sum / metal_count as f64
        } else {
            0.02
        };

        // Edad estimada del halo: usar un proxy simple (10 Gyr para halos masivos)
        let age_est = 3.0 + 7.0 * (m_h / m_pivot_code).min(1.0);
        let bv = 0.35 + 0.25 * (age_est + 0.01).log10() + 0.10 * metallicity.max(1e-3).log10();
        let bv = bv.clamp(-0.3, 1.5);

        // SFR específica (sSFR) — anticorrelacionada con masa
        let ssfr = 1.0e-10 * (m_h / m_pivot_code).powf(-0.5);

        // Magnitud absoluta en R desde la masa estelar
        // M_R ≈ M_R_sun - 2.5 × log10(L_R/L_sun)  con M_R_sun ≈ 4.42
        let l_r_lsun = stellar_mass * 1e10 * 0.9; // ~1 L☉/M☉ para edad ~5 Gyr
        let m_r_abs = if l_r_lsun > 0.0 {
            4.42 - 2.5 * l_r_lsun.log10()
        } else {
            99.0
        };

        let m_r_app = apparent_magnitude(m_r_abs, z_box, omega_m);
        if !selection_flux_limit(m_r_app, m_lim) {
            continue;
        }

        catalog.push(MockGalaxy {
            pos: [com[0], com[1], com[2]],
            z_obs: z_box,
            m_r_abs,
            m_r_app,
            bv,
            ssfr,
            metallicity,
            halo_mass: m_h,
            stellar_mass,
        });
    }

    catalog
}

/// Espectro de potencia angular C_l simplificado (Phase 154).
///
/// Proyecta las posiciones de las galaxias sobre una malla 2D (plano XY)
/// y calcula C_l via FFT anular.
///
/// # Parámetros
/// - `catalog`: galaxias del mock catalogue
/// - `l_max`: multipolo máximo
/// - `box_size`: tamaño de la caja en unidades internas
///
/// # Retorna
/// `Vec<f64>` de longitud `l_max + 1` con C_l para l = 0, 1, ..., l_max.
pub fn angular_power_spectrum_cl(catalog: &[MockGalaxy], l_max: usize, box_size: f64) -> Vec<f64> {
    if catalog.is_empty() || l_max == 0 {
        return vec![0.0; l_max + 1];
    }

    let n_mesh = (2 * l_max).max(16);
    let mut grid = vec![0.0_f64; n_mesh * n_mesh];

    for gal in catalog {
        let ix = ((gal.pos[0] / box_size * n_mesh as f64) as usize).min(n_mesh - 1);
        let iy = ((gal.pos[1] / box_size * n_mesh as f64) as usize).min(n_mesh - 1);
        grid[iy * n_mesh + ix] += 1.0;
    }

    // Overdensity: δ = (n - n_bar) / n_bar
    let n_bar = catalog.len() as f64 / (n_mesh * n_mesh) as f64;
    if n_bar <= 0.0 {
        return vec![0.0; l_max + 1];
    }
    for v in &mut grid {
        *v = (*v - n_bar) / n_bar;
    }

    // C_l via suma de |δ_lm|² — aproximación Fourier plana
    let mut cl = vec![0.0_f64; l_max + 1];
    let mut counts = vec![0_usize; l_max + 1];

    for ky in 0..n_mesh {
        for kx in 0..n_mesh {
            let lx = if kx <= n_mesh / 2 {
                kx as f64
            } else {
                kx as f64 - n_mesh as f64
            };
            let ly = if ky <= n_mesh / 2 {
                ky as f64
            } else {
                ky as f64 - n_mesh as f64
            };
            let l_val = (lx * lx + ly * ly).sqrt().round() as usize;
            if l_val > l_max {
                continue;
            }
            let v = grid[ky * n_mesh + kx];
            cl[l_val] += v * v;
            counts[l_val] += 1;
        }
    }

    for (c, n) in cl.iter_mut().zip(counts.iter()) {
        if *n > 0 {
            *c /= *n as f64;
        }
    }
    cl
}
