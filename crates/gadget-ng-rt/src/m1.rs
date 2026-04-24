//! Solver de transferencia radiativa con cierre M1 (Phase 81).
//!
//! ## El modelo M1
//!
//! El cierre M1 (Levermore 1984; Morel 2000) aproxima la ecuación de transferencia
//! radiativa completa trazando solo los dos primeros momentos del campo de radiación:
//!
//! - **Energía radiativa** E [erg/cm³] (momento 0).
//! - **Flujo radiativo** F [erg/cm²/s] (momento 1).
//!
//! Las ecuaciones de evolución son hiperbólicas:
//!
//! ```text
//! ∂E/∂t + ∇·F = η - cκ_abs E          (fuentes − absorción)
//! ∂F/∂t + c²∇·P = -cκ_abs F           (transporte − absorción)
//! ```
//!
//! donde `P = f × E` es el tensor de presión radiativa con el factor de Eddington
//! de cierre M1:
//!
//! ```text
//! f(ξ) = (3 + 4ξ²) / (5 + 2√(4 - 3ξ²))     ξ = |F| / (cE)
//! ```
//!
//! Para resolver las ecuaciones hiperbólicas se usa una velocidad de luz reducida
//! `c_red = c / c_red_factor` (Gnedin & Abel 2001) para evitar pasos de tiempo
//! prohibitivamente pequeños. Valores típicos: `c_red_factor = 100 - 1000`.
//!
//! ## Implementación
//!
//! Solver Godunov explícito de primer orden con esquema HLL (Harten-Lax-van Leer)
//! en una malla cartesiana regular. El acoplamiento con el gas se hace mediante
//! fuente implícita linealizada (splitting de operadores).
//!
//! ## Referencia
//!
//! Levermore (1984), J. Quant. Spectrosc. Radiat. Transf. 31, 149;
//! Morel (2000), J. Quant. Spectrosc. Radiat. Transf. 65, 769;
//! González et al. (2007), A&A 464, 429;
//! Rosdahl et al. (2013), MNRAS 436, 2188 (implementación RAMSES-RT).

// ── Constantes físicas ────────────────────────────────────────────────────

/// Velocidad de la luz en km/s.
pub const C_KMS: f64 = 2.998e5;

/// Factor de Eddington M1: f(ξ) = (3 + 4ξ²) / (5 + 2√(4 - 3ξ²)).
/// Asintótico: f→1/3 para ξ→0 (isótropo), f→1 para ξ→1 (streaming libre).
#[inline]
pub fn eddington_factor(xi: f64) -> f64 {
    let xi = xi.clamp(0.0, 1.0);
    let xi2 = xi * xi;
    let denom = 5.0 + 2.0 * (4.0 - 3.0 * xi2).max(0.0).sqrt();
    (3.0 + 4.0 * xi2) / denom
}

// ── Structs ───────────────────────────────────────────────────────────────

/// Campo de radiación en una malla cartesiana regular.
///
/// La malla tiene resolución `nx × ny × nz` celdas con espaciado uniforme `dx`.
/// Indexación: `[iz * ny * nx + iy * nx + ix]`.
#[derive(Debug, Clone)]
pub struct RadiationField {
    /// Densidad de energía radiativa E [unidades internas].
    pub energy_density: Vec<f64>,
    /// Flujo radiativo F = (Fx, Fy, Fz) [unidades internas].
    pub flux_x: Vec<f64>,
    pub flux_y: Vec<f64>,
    pub flux_z: Vec<f64>,
    /// Número de celdas en cada dimensión.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Espaciado de la malla [unidades físicas].
    pub dx: f64,
}

impl RadiationField {
    /// Crea un campo de radiación con energía uniforme y flujo cero.
    pub fn uniform(nx: usize, ny: usize, nz: usize, dx: f64, e0: f64) -> Self {
        let n3 = nx * ny * nz;
        Self {
            energy_density: vec![e0; n3],
            flux_x: vec![0.0; n3],
            flux_y: vec![0.0; n3],
            flux_z: vec![0.0; n3],
            nx, ny, nz, dx,
        }
    }

    /// Índice lineal (ix, iy, iz) → posición en el array.
    #[inline]
    pub fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz * self.ny * self.nx + iy * self.nx + ix
    }

    /// Número total de celdas.
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Energía total en el dominio (suma de E × dV).
    pub fn total_energy(&self, dv: f64) -> f64 {
        self.energy_density.iter().sum::<f64>() * dv
    }

    /// Parámetro de anisotropía ξ = |F| / (c_red × E) para cada celda.
    pub fn xi_field(&self, c_red: f64) -> Vec<f64> {
        let n = self.n_cells();
        (0..n)
            .map(|i| {
                let e = self.energy_density[i];
                if e < 1e-300 {
                    return 0.0;
                }
                let fx = self.flux_x[i];
                let fy = self.flux_y[i];
                let fz = self.flux_z[i];
                let f_mag = (fx * fx + fy * fy + fz * fz).sqrt();
                (f_mag / (c_red * e)).clamp(0.0, 1.0)
            })
            .collect()
    }
}

/// Parámetros del solver M1.
#[derive(Debug, Clone)]
pub struct M1Params {
    /// Factor de reducción de la velocidad de la luz: `c_red = c / c_red_factor`.
    pub c_red_factor: f64,
    /// Opacidad de absorción κ_abs [1/unidad_longitud].
    pub kappa_abs: f64,
    /// Opacidad de scattering κ_scat [1/unidad_longitud].
    pub kappa_scat: f64,
    /// Número de pasos del solver M1 por paso cosmológico.
    pub substeps: usize,
    /// Sección eficaz de polvo para fotocalentamiento (Phase 137). Default: `0.1`.
    pub sigma_dust: f64,
}

impl Default for M1Params {
    fn default() -> Self {
        Self {
            c_red_factor: 100.0,
            kappa_abs: 1.0,
            kappa_scat: 0.0,
            substeps: 5,
            sigma_dust: 0.1,
        }
    }
}

// ── Solver M1 ─────────────────────────────────────────────────────────────

/// Actualiza el campo de radiación por un paso de tiempo `dt` con el solver HLL M1.
///
/// Usa el método de splitting de operadores:
/// 1. Advección hiperbólica: transporte con HLL + flujo de Riemann M1.
/// 2. Fuentes: absorción implícita linealizada.
///
/// # Parámetros
/// - `rad`    — campo de radiación (modificado in-place).
/// - `dt`     — paso de tiempo.
/// - `params` — parámetros del solver (c_red, opacidades).
///
/// # CFL
/// Para estabilidad: `dt < dx / c_red`. La función aplica sub-stepping automático.
pub fn m1_update(rad: &mut RadiationField, dt: f64, params: &M1Params) {
    let c_red = C_KMS / params.c_red_factor;
    let dt_cfl = rad.dx / c_red * 0.5; // CFL con margen de seguridad
    let n_sub = ((dt / dt_cfl).ceil() as usize).max(1).max(params.substeps);
    let dt_sub = dt / n_sub as f64;

    for _ in 0..n_sub {
        m1_substep(rad, dt_sub, c_red, params.kappa_abs, params.kappa_scat);
    }
}

/// Un sub-paso del solver M1 con c_red y opacidades explícitas.
fn m1_substep(rad: &mut RadiationField, dt: f64, c_red: f64, kappa_abs: f64, kappa_scat: f64) {
    let nx = rad.nx;
    let ny = rad.ny;
    let nz = rad.nz;
    let n3 = nx * ny * nz;
    let dx = rad.dx;
    let kappa = kappa_abs + kappa_scat;

    let mut de = vec![0.0f64; n3];
    let mut dfx = vec![0.0f64; n3];
    let mut dfy = vec![0.0f64; n3];
    let mut dfz = vec![0.0f64; n3];

    // ── Advección en X ────────────────────────────────────────────────────
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let ixr = (ix + 1) % nx;
                let il = rad.idx(ix, iy, iz);
                let ir = rad.idx(ixr, iy, iz);
                let (fe, ffx) = hll_flux_x(
                    rad.energy_density[il], rad.flux_x[il], rad.flux_y[il], rad.flux_z[il],
                    rad.energy_density[ir], rad.flux_x[ir], rad.flux_y[ir], rad.flux_z[ir],
                    c_red,
                );
                let dtdx = dt / dx;
                de[il] -= dtdx * fe;
                dfx[il] -= dtdx * ffx;
                de[ir] += dtdx * fe;
                dfx[ir] += dtdx * ffx;
            }
        }
    }

    // ── Advección en Y ────────────────────────────────────────────────────
    for iz in 0..nz {
        for iy in 0..ny {
            let iyr = (iy + 1) % ny;
            for ix in 0..nx {
                let il = rad.idx(ix, iy, iz);
                let ir = rad.idx(ix, iyr, iz);
                let (fe, ffy) = hll_flux_x(
                    rad.energy_density[il], rad.flux_y[il], rad.flux_x[il], rad.flux_z[il],
                    rad.energy_density[ir], rad.flux_y[ir], rad.flux_x[ir], rad.flux_z[ir],
                    c_red,
                );
                let dtdx = dt / dx;
                de[il] -= dtdx * fe;
                dfy[il] -= dtdx * ffy;
                de[ir] += dtdx * fe;
                dfy[ir] += dtdx * ffy;
            }
        }
    }

    // ── Advección en Z ────────────────────────────────────────────────────
    for iz in 0..nz {
        let izr = (iz + 1) % nz;
        for iy in 0..ny {
            for ix in 0..nx {
                let il = rad.idx(ix, iy, iz);
                let ir = rad.idx(ix, iy, izr);
                let (fe, ffz) = hll_flux_x(
                    rad.energy_density[il], rad.flux_z[il], rad.flux_x[il], rad.flux_y[il],
                    rad.energy_density[ir], rad.flux_z[ir], rad.flux_x[ir], rad.flux_y[ir],
                    c_red,
                );
                let dtdx = dt / dx;
                de[il] -= dtdx * fe;
                dfz[il] -= dtdx * ffz;
                de[ir] += dtdx * fe;
                dfz[ir] += dtdx * ffz;
            }
        }
    }

    // ── Actualizar + fuente implícita ────────────────────────────────────
    let decay = (-c_red * kappa * dt).exp();
    for i in 0..n3 {
        let e_new = (rad.energy_density[i] + de[i]).max(0.0);
        rad.energy_density[i] = e_new * decay;
        rad.flux_x[i] = (rad.flux_x[i] + dfx[i]) * decay;
        rad.flux_y[i] = (rad.flux_y[i] + dfy[i]) * decay;
        rad.flux_z[i] = (rad.flux_z[i] + dfz[i]) * decay;
    }
}

/// Flujo HLL en la interfaz izquierda-derecha para la componente x.
///
/// Usa el aproximador de Riemann de Harten-Lax-van Leer.
/// Para el transporte a lo largo de x, el flujo de E es Fx,
/// y el flujo de Fx es c_red² × f_edd × E.
fn hll_flux_x(
    el: f64, fxl: f64, _fyl: f64, _fzl: f64,
    er: f64, fxr: f64, _fyr: f64, _fzr: f64,
    c_red: f64,
) -> (f64, f64) {
    let xi_l = (fxl.abs() / (c_red * el.max(1e-300))).clamp(0.0, 1.0);
    let xi_r = (fxr.abs() / (c_red * er.max(1e-300))).clamp(0.0, 1.0);
    let f_l = eddington_factor(xi_l);
    let f_r = eddington_factor(xi_r);

    // Velocidades de onda HLL
    let s_l = -c_red;
    let s_r = c_red;

    // Flujos físicos: para E → F_x; para F_x → c² × f × E
    let flux_e_l = fxl;
    let flux_e_r = fxr;
    let flux_fx_l = c_red * c_red * f_l * el;
    let flux_fx_r = c_red * c_red * f_r * er;

    // Flujo HLL
    let denom = s_r - s_l;
    let fe = if denom.abs() < 1e-30 {
        0.5 * (flux_e_l + flux_e_r)
    } else {
        (s_r * flux_e_l - s_l * flux_e_r + s_r * s_l * (er - el)) / denom
    };
    let ffx = if denom.abs() < 1e-30 {
        0.5 * (flux_fx_l + flux_fx_r)
    } else {
        (s_r * flux_fx_l - s_l * flux_fx_r + s_r * s_l * (fxr - fxl)) / denom
    };

    (fe, ffx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eddington_factor_isotropic() {
        // Para ξ=0 (campo isótropo), f = 1/3
        let f = eddington_factor(0.0);
        assert!((f - 1.0 / 3.0).abs() < 1e-10, "f(ξ=0) debe ser 1/3: {f}");
    }

    #[test]
    fn eddington_factor_streaming() {
        // Para ξ=1 (streaming libre), f = 1
        let f = eddington_factor(1.0);
        assert!((f - 1.0).abs() < 1e-10, "f(ξ=1) debe ser 1.0: {f}");
    }

    #[test]
    fn eddington_factor_monotone() {
        // f debe ser creciente en [0, 1]
        let vals: Vec<f64> = (0..=10).map(|i| eddington_factor(i as f64 / 10.0)).collect();
        for i in 1..vals.len() {
            assert!(vals[i] >= vals[i - 1], "f debe ser creciente: {vals:?}");
        }
    }

    #[test]
    fn radiation_field_uniform() {
        let rad = RadiationField::uniform(4, 4, 4, 1.0, 1.0);
        assert_eq!(rad.n_cells(), 64);
        assert!(rad.energy_density.iter().all(|&e| (e - 1.0).abs() < 1e-12));
        assert!(rad.flux_x.iter().all(|&f| f == 0.0));
    }

    #[test]
    fn radiation_field_total_energy() {
        let rad = RadiationField::uniform(4, 4, 4, 1.0, 2.0);
        let dv = 1.0_f64;
        let e_total = rad.total_energy(dv);
        assert!((e_total - 128.0).abs() < 1e-10, "E_total = 64 × 2 = 128: {e_total}");
    }

    #[test]
    fn m1_update_conserves_energy_vacuum() {
        // En vacío (sin absorción), la energía debe conservarse aproximadamente
        let mut rad = RadiationField::uniform(8, 8, 8, 1.0, 1.0);
        let params = M1Params { kappa_abs: 0.0, kappa_scat: 0.0, ..Default::default() };
        let dv = 1.0_f64;
        let e0 = rad.total_energy(dv);
        m1_update(&mut rad, 0.1, &params);
        let e1 = rad.total_energy(dv);
        // Con distribución uniforme, E se conserva exactamente (no hay flujo neto)
        assert!(
            (e1 - e0).abs() / e0 < 0.01,
            "Energía no conservada: ΔE/E = {:.4}", (e1 - e0).abs() / e0
        );
    }

    #[test]
    fn m1_update_absorption_decays() {
        // Con absorción, la energía debe decrecer
        let mut rad = RadiationField::uniform(4, 4, 4, 1.0, 1.0);
        let params = M1Params { kappa_abs: 1.0, kappa_scat: 0.0, c_red_factor: 10.0, substeps: 1 };
        let e0 = rad.total_energy(1.0);
        m1_update(&mut rad, 0.1, &params);
        let e1 = rad.total_energy(1.0);
        assert!(e1 < e0, "Con absorción, energía debe decrecer: E0={e0:.4}, E1={e1:.4}");
    }

    #[test]
    fn m1_update_energy_positive() {
        // La energía radiativa siempre debe ser no negativa
        let mut rad = RadiationField::uniform(4, 4, 4, 0.5, 1.0);
        // Campo con gradiente para activar flujo
        for ix in 0..4 {
            for iy in 0..4 {
                for iz in 0..4 {
                    let i = rad.idx(ix, iy, iz);
                    rad.flux_x[i] = (ix as f64 - 1.5) * 0.1;
                }
            }
        }
        let params = M1Params { kappa_abs: 0.1, ..Default::default() };
        m1_update(&mut rad, 0.01, &params);
        for (i, &e) in rad.energy_density.iter().enumerate() {
            assert!(e >= 0.0, "Energía negativa en celda {i}: {e}");
        }
    }

    #[test]
    fn hll_flux_symmetric() {
        // Flujo simétrico para estados iguales → solo difusión
        let c_red = 100.0;
        let (fe, ffx) = hll_flux_x(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, c_red);
        // Para E_l = E_r y F_l = F_r = 0, el flujo de E debe ser 0
        assert!(fe.abs() < 1e-10, "Flujo de E no nulo para estados iguales: {fe}");
        // El flujo de Fx es c² × f × E (puro tensor de presión)
        assert!(ffx.is_finite(), "Flujo de Fx no finito: {ffx}");
    }
}
