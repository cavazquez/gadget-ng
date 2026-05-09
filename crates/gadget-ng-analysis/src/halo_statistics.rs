//! Estadísticas de halos: función de masa de sub-halos (SHMF) y
//! relación concentración-masa.
//!
//! ## SHMF (Sub-Halo Mass Function)
//!
//! La función de masa de sub-halos `dn/d ln(m_sub)` cuantifica la distribución
//! de masa de los sub-halos dentro de halos padres. Para cada halo padre se
//! identifican los sub-halos dentro de su radio virial y se calcula:
//!
//! - `dn/d ln(m_sub)`: número diferencial de sub-halos por intervalo logarítmico
//!   de masa, normalizado por el número de padres.
//! - `N(> m_sub)`: número acumulado de sub-halos con masa mayor a `m_sub`
//!   sobre todos los padres.
//!
//! ## Relación concentración-masa
//!
//! La concentración NFW se define como `c_200 = r_200 / r_s`, donde `r_200`
//! es el radio virial y `r_s` es el radio de escala. Se utilizan las relaciones
//! teóricas de Duffy et al. (2008) y Ludlow et al. (2016) implementadas en
//! [`crate::nfw`].
//!
//! ## Unidades
//!
//! - Masas: M_sun/h
//! - Longitudes: Mpc/h

use crate::fof::FofHalo;
use crate::nfw;
use serde::{Deserialize, Serialize};

// ── Parámetros ────────────────────────────────────────────────────────────────

/// Parámetros para los cálculos de estadísticas de halos.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaloStatsParams {
    /// Número de bins logarítmicos de masa para la SHMF (default: 20).
    pub n_mass_bins: usize,
    /// Número de bins logarítmicos de masa para la relación c(M) (default: 15).
    pub n_conc_bins: usize,
    /// Redshift para las relaciones teóricas de concentración (default: 0.0).
    pub z: f64,
    /// Masa mínima para los bins de masa (default: 1e10 M_sun/h).
    pub m_min: f64,
    /// Masa máxima para los bins de masa (default: 1e15 M_sun/h).
    pub m_max: f64,
}

impl Default for HaloStatsParams {
    fn default() -> Self {
        Self {
            n_mass_bins: 20,
            n_conc_bins: 15,
            z: 0.0,
            m_min: 1e10,
            m_max: 1e15,
        }
    }
}

// ── SHMF ───────────────────────────────────────────────────────────────────────

/// Un bin de la función de masa de sub-halos (SHMF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShmfBin {
    /// Masa del sub-halo (media geométrica del bin) en M_sun/h.
    pub m_sub: f64,
    /// dn/d ln(m_sub): número diferencial de sub-halos por intervalo
    /// logarítmico de masa, normalizado por el número de padres.
    pub dn_dlnm: f64,
    /// N(> m_sub): número acumulado de sub-halos con masa > m_sub
    /// sobre todos los padres.
    pub n_cumulative: usize,
    /// Masa característica del padre (promedio sobre padres con sub-halos)
    /// en M_sun/h.
    pub parent_mass: f64,
}

/// Calcula la función de masa de sub-halos (SHMF).
///
/// Para cada halo padre, identifica sub-halos dentro de su radio virial
/// y computa `dn/d ln(m_sub)` y `N(> m_sub)`.
///
/// # Algoritmo
///
/// 1. Se ordenan los halos por masa descendente.
/// 2. Cada halo que no es sub-halo de otro es un "padre".
/// 3. Un halo `j` es sub-halo de `i` si `|r_j - r_i| < r_vir(i)` y
///    `m_j < m_i` y `j` no ha sido asignado previamente como sub-halo.
/// 4. Se calcula `dn/d ln(m_sub)` en bins logarítmicos de masa.
///
/// # Parámetros
///
/// - `halos`: catálogo de halos FoF.
/// - `params`: parámetros de binning y configuración.
pub fn subhalo_mass_function(halos: &[FofHalo], params: &HaloStatsParams) -> Vec<ShmfBin> {
    if halos.is_empty() {
        return Vec::new();
    }

    let mut sorted_indices: Vec<usize> = (0..halos.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        halos[b]
            .mass
            .partial_cmp(&halos[a].mass)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Para cada halo, encontrar su padre (el más masivo que lo contiene dentro de r_vir).
    let n = halos.len();
    let mut sub_of: Vec<Option<usize>> = vec![None; n];

    for &parent_idx in &sorted_indices {
        let parent = &halos[parent_idx];
        if parent.r_vir <= 0.0 {
            continue;
        }
        let px = parent.x_com;
        let py = parent.y_com;
        let pz = parent.z_com;
        let r2_max = parent.r_vir * parent.r_vir;

        for &child_idx in &sorted_indices {
            if child_idx == parent_idx {
                continue;
            }
            if halos[child_idx].mass >= parent.mass {
                continue;
            }
            if sub_of[child_idx].is_some() {
                continue;
            }
            let dx = halos[child_idx].x_com - px;
            let dy = halos[child_idx].y_com - py;
            let dz = halos[child_idx].z_com - pz;
            if dx * dx + dy * dy + dz * dz < r2_max {
                sub_of[child_idx] = Some(parent_idx);
            }
        }
    }

    // Padres: halos que no son sub-halos de ningún otro.
    let parent_indices: Vec<usize> = (0..n).filter(|&i| sub_of[i].is_none()).collect();

    // Recolectar sub-halos por padre.
    let mut parent_subs: std::collections::HashMap<usize, Vec<f64>> =
        std::collections::HashMap::new();
    for &pi in &parent_indices {
        parent_subs.insert(pi, Vec::new());
    }
    for child_idx in 0..n {
        if let Some(pi) = sub_of[child_idx]
            && let Some(v) = parent_subs.get_mut(&pi)
        {
            v.push(halos[child_idx].mass);
        }
    }

    let all_sub_masses: Vec<f64> = parent_subs
        .values()
        .flat_map(|v| v.iter().copied())
        .collect();

    if all_sub_masses.is_empty() {
        let max_halo_mass = halos
            .iter()
            .map(|h| h.mass)
            .fold(0.0_f64, f64::max)
            .max(params.m_min * 10.0);
        let m_lo = params.m_min;
        let m_hi = max_halo_mass;
        return build_shmf_bins(m_lo, m_hi, params.n_mass_bins, &parent_indices, halos, &[]);
    }

    let m_lo = all_sub_masses
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let m_hi = all_sub_masses
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    build_shmf_bins(
        m_lo,
        m_hi,
        params.n_mass_bins,
        &parent_indices,
        halos,
        &all_sub_masses,
    )
}

fn build_shmf_bins(
    m_lo: f64,
    m_hi: f64,
    n_bins: usize,
    parent_indices: &[usize],
    halos: &[FofHalo],
    all_sub_masses: &[f64],
) -> Vec<ShmfBin> {
    let n_bins = n_bins.max(1);
    let log_m_lo = m_lo.ln();
    let log_m_hi = m_hi.ln();
    let dlog_m = if log_m_hi > log_m_lo {
        (log_m_hi - log_m_lo) / n_bins as f64
    } else {
        1.0
    };

    let n_parents = parent_indices.len().max(1) as f64;

    // Contar sub-halos en cada bin.
    let mut counts = vec![0usize; n_bins];
    for &m in all_sub_masses {
        if m <= 0.0 || dlog_m <= 0.0 {
            continue;
        }
        let idx = ((m.ln() - log_m_lo) / dlog_m).floor() as usize;
        if idx < n_bins {
            counts[idx] += 1;
        }
    }

    // Construir bins.
    let mut bins: Vec<ShmfBin> = (0..n_bins)
        .map(|i| {
            let m_lo_i = (log_m_lo + i as f64 * dlog_m).exp();
            let m_hi_i = (log_m_lo + (i + 1) as f64 * dlog_m).exp();
            let m_sub = (m_lo_i * m_hi_i).sqrt();
            ShmfBin {
                m_sub,
                dn_dlnm: 0.0,
                n_cumulative: 0,
                parent_mass: 0.0,
            }
        })
        .collect();

    // N acumulado desde el bin más masivo.
    let mut cumul = 0usize;
    for i in (0..n_bins).rev() {
        cumul += counts[i];
        bins[i].n_cumulative = cumul;
    }

    // dn/d ln(m) = N_bin / (N_parents × d ln m).
    for i in 0..n_bins {
        bins[i].dn_dlnm = counts[i] as f64 / (n_parents * dlog_m);
    }

    // Masa promedio de los padres.
    let avg_parent_mass: f64 = if parent_indices.is_empty() {
        0.0
    } else {
        parent_indices
            .iter()
            .map(|&pi| halos[pi].mass)
            .sum::<f64>()
            / parent_indices.len() as f64
    };
    for bin in &mut bins {
        bin.parent_mass = avg_parent_mass;
    }

    bins
}

// ── Relación concentración-masa ────────────────────────────────────────────────

/// Un bin de la relación concentración-masa.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcentrationMassBin {
    /// Masa M_200 central del bin (media geométrica) en M_sun/h.
    pub m_200: f64,
    /// Concentración mediana c_200 = r_200 / r_s en este bin.
    pub c_200: f64,
    /// Número de halos en este bin.
    pub n_halos: usize,
}

/// Calcula la relación concentración-masa `c_200(M_200)` a partir de un
/// catálogo de halos, usando las relaciones teóricas NFW implementadas en
/// [`crate::nfw`].
///
/// Para cada halo se calcula `c_200` usando la relación de
/// Duffy et al. (2008) y se computa la mediana por bin logarítmico de `M_200`.
///
/// # Parámetros
///
/// - `halos`: catálogo de halos FoF (se usa `halo.mass` como `M_200`).
/// - `params`: parámetros de binning, redshift y configuración.
///
/// # Retorno
///
/// `Vec<ConcentrationMassBin>` con la mediana de `c_200` y el conteo por bin.
pub fn concentration_mass_relation(
    halos: &[FofHalo],
    params: &HaloStatsParams,
) -> Vec<ConcentrationMassBin> {
    if halos.is_empty() {
        return Vec::new();
    }

    let n_bins = params.n_conc_bins.max(1);
    let log_m_min = params.m_min.log10();
    let log_m_max = params.m_max.log10();
    let dlog_m = (log_m_max - log_m_min) / n_bins as f64;
    let z = params.z;

    // c_200 para cada halo usando Duffy+2008.
    let halo_c: Vec<f64> = halos
        .iter()
        .map(|h| nfw::concentration_duffy2008(h.mass, z))
        .collect();

    // Asignar halos a bins.
    let mut bin_c_values: Vec<Vec<f64>> = vec![Vec::new(); n_bins];

    for (i, h) in halos.iter().enumerate() {
        if h.mass <= 0.0 || dlog_m <= 0.0 {
            continue;
        }
        let log_m = h.mass.log10();
        let idx = ((log_m - log_m_min) / dlog_m).floor() as usize;
        if idx < n_bins {
            bin_c_values[idx].push(halo_c[i]);
        }
    }

    // Calcular mediana por bin.
    (0..n_bins)
        .map(|i| {
            let m_lo = 10_f64.powf(log_m_min + i as f64 * dlog_m);
            let m_hi = 10_f64.powf(log_m_min + (i + 1) as f64 * dlog_m);
            let m_200 = (m_lo * m_hi).sqrt();

            let n_halos = bin_c_values[i].len();
            let c_median = if n_halos == 0 {
                0.0
            } else {
                bin_c_values[i].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let len = bin_c_values[i].len();
                if len.is_multiple_of(2) {
                    (bin_c_values[i][len / 2 - 1] + bin_c_values[i][len / 2]) / 2.0
                } else {
                    bin_c_values[i][len / 2]
                }
            };

            ConcentrationMassBin {
                m_200,
                c_200: c_median,
                n_halos,
            }
        })
        .collect()
}

// ── Tests unitarios ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_halo(id: usize, mass: f64, x: f64, y: f64, z: f64, r_vir: f64) -> FofHalo {
        FofHalo {
            halo_id: id,
            n_particles: 100,
            mass,
            x_com: x,
            y_com: y,
            z_com: z,
            vx_com: 0.0,
            vy_com: 0.0,
            vz_com: 0.0,
            velocity_dispersion: 100.0,
            r_vir,
        }
    }

    #[test]
    fn shmf_empty_halos() {
        let halos: Vec<FofHalo> = Vec::new();
        let params = HaloStatsParams::default();
        let result = subhalo_mass_function(&halos, &params);
        assert!(result.is_empty(), "SHMF vacío para halos vacíos");
    }

    #[test]
    fn shmf_single_parent_with_subs() {
        // Un halo padre masivo (M=1e14) con 3 sub-halos dentro de r_vir.
        let parent = make_halo(0, 1e14, 0.0, 0.0, 0.0, 1.0);
        let sub1 = make_halo(1, 1e12, 0.1, 0.0, 0.0, 0.2);
        let sub2 = make_halo(2, 5e11, 0.0, 0.2, 0.0, 0.15);
        let sub3 = make_halo(3, 1e11, 0.05, 0.05, 0.05, 0.1);
        // Un halo lejano (fuera de r_vir del padre) — no es sub-halo.
        let isolated = make_halo(4, 2e13, 50.0, 50.0, 50.0, 0.5);
        let halos = vec![parent, sub1, sub2, sub3, isolated];

        let params = HaloStatsParams {
            m_min: 1e10,
            m_max: 1e15,
            n_mass_bins: 10,
            ..Default::default()
        };
        let result = subhalo_mass_function(&halos, &params);

        // El primer bin debe tener N_cumulative >= 2 (los sub-halos dentro del padre)
        let first_bin = result.first().expect("SHMF debe tener bins");
        assert!(
            first_bin.n_cumulative >= 2,
            "N_cumulative debe ser >= 2: {}",
            first_bin.n_cumulative
        );

        // Al menos un bin debe tener dn_dlnm > 0.
        let any_nonzero = result.iter().any(|b| b.dn_dlnm > 0.0);
        assert!(any_nonzero, "Al menos un bin debe tener dn_dlnm > 0");

        // parent_mass debe ser el promedio de los padres
        // (padre masivo + aislado ≈ (1e14 + 2e13)/2 = 6e13).
        assert!(
            first_bin.parent_mass > 0.0,
            "parent_mass debe ser positivo: {}",
            first_bin.parent_mass
        );
    }

    #[test]
    fn concentration_mass_relation_bins() {
        // Halos con masas típicas de grupo/cluster.
        let halos = vec![
            make_halo(0, 1e12, 0.0, 0.0, 0.0, 0.1),
            make_halo(1, 5e12, 1.0, 0.0, 0.0, 0.15),
            make_halo(2, 1e13, 2.0, 0.0, 0.0, 0.3),
            make_halo(3, 5e13, 3.0, 0.0, 0.0, 0.5),
            make_halo(4, 1e14, 4.0, 0.0, 0.0, 0.8),
        ];

        let params = HaloStatsParams {
            n_conc_bins: 5,
            m_min: 1e12,
            m_max: 1e14,
            ..Default::default()
        };
        let result = concentration_mass_relation(&halos, &params);

        assert_eq!(result.len(), 5, "Debe haber 5 bins");

        // Los bins con halos deben tener c_200 en rango físico [1, 30].
        for bin in &result {
            if bin.n_halos > 0 {
                assert!(
                    bin.c_200 > 1.0 && bin.c_200 < 30.0,
                    "c_200 fuera de rango físico: m={:.2e}, c={:.2}",
                    bin.m_200,
                    bin.c_200
                );
            }
        }

        // Total de halos en bins debe ser <= 5 (algunos pueden caer fuera del rango)
        let total: usize = result.iter().map(|b| b.n_halos).sum();
        assert!(total <= 5 && total > 0, "Debe haber halos en bins: total = {total}");
    }

    #[test]
    fn default_params() {
        let params = HaloStatsParams::default();
        assert_eq!(params.n_mass_bins, 20);
        assert_eq!(params.n_conc_bins, 15);
        assert_eq!(params.z, 0.0);
        assert!(
            params.m_min > 0.0 && params.m_max > params.m_min,
            "m_min y m_max deben ser válidos"
        );
    }
}