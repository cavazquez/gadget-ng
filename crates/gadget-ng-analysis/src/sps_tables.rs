//! Tablas de síntesis de población estelar BC03-lite — Phase 153.
//!
//! ## Modelo
//!
//! Se embebe una grilla bilineal 6×5 (edad × metalicidad) derivada de los
//! modelos de Bruzual & Charlot (2003) con IMF de Chabrier (2003).
//!
//! Las bandas disponibles son U, B, V, R, I (magnitudes absolutas de M☉).
//!
//! ## Uso
//!
//! ```rust
//! use gadget_ng_analysis::sps_tables::{sps_luminosity, Spsband};
//!
//! let l_b = sps_luminosity(2.0, 0.008, Spsband::B);
//! ```
//!
//! ## Referencia
//!
//! Bruzual & Charlot (2003) MNRAS 344, 1000.
//! Chabrier (2003) PASP 115, 763.

/// Banda fotométrica disponible en la grilla SPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spsband {
    /// Banda U (Johnson), λ_eff ≈ 3650Å.
    U,
    /// Banda B (Johnson), λ_eff ≈ 4440Å.
    B,
    /// Banda V (Johnson), λ_eff ≈ 5500Å.
    V,
    /// Banda R (Cousins), λ_eff ≈ 6400Å.
    R,
    /// Banda I (Cousins), λ_eff ≈ 8090Å.
    I,
}

/// Grilla SPS BC03-lite: luminosidad en cada banda en unidades de L☉/M☉.
///
/// Dimensiones: 6 edades × 5 metalicidades.
///
/// Edades [Gyr]: 0.01, 0.1, 0.5, 1.0, 5.0, 13.0
/// Metalicidades Z: 0.0004, 0.004, 0.008, 0.02, 0.05
#[derive(Debug, Clone)]
pub struct SpsGrid {
    /// Edades de la grilla en Gyr (longitud 6).
    pub ages_gyr: [f64; 6],
    /// Metalicidades Z (longitud 5).
    pub metals: [f64; 5],
    /// Luminosidad en banda U [L☉/M☉], indexada [edad][metal].
    pub l_u: [[f64; 5]; 6],
    /// Luminosidad en banda B [L☉/M☉], indexada [edad][metal].
    pub l_b: [[f64; 5]; 6],
    /// Luminosidad en banda V [L☉/M☉], indexada [edad][metal].
    pub l_v: [[f64; 5]; 6],
    /// Luminosidad en banda R [L☉/M☉], indexada [edad][metal].
    pub l_r: [[f64; 5]; 6],
    /// Luminosidad en banda I [L☉/M☉], indexada [edad][metal].
    pub l_i: [[f64; 5]; 6],
}

impl SpsGrid {
    /// Grilla BC03-lite por defecto (valores tabulados simplificados).
    ///
    /// Los valores son representativos del comportamiento cualitativo de BC03:
    /// galaxias jóvenes son más azules y luminosas; las viejas son más rojas.
    pub fn bc03_lite() -> Self {
        // Edades [Gyr]: 0.01, 0.1, 0.5, 1.0, 5.0, 13.0
        let ages_gyr = [0.01, 0.1, 0.5, 1.0, 5.0, 13.0];
        // Metalicidades: 0.0004 (sub-solar), 0.004, 0.008, 0.02 (solar), 0.05 (super-solar)
        let metals = [0.0004, 0.004, 0.008, 0.02, 0.05];

        // L_U [L☉/M☉] — mayor en poblaciones jóvenes, decrece rápido
        let l_u = [
            [120.0, 130.0, 140.0, 150.0, 160.0], // 0.01 Gyr
            [25.0, 28.0, 30.0, 33.0, 36.0],      // 0.1 Gyr
            [5.0, 6.0, 6.5, 7.0, 7.5],           // 0.5 Gyr
            [2.5, 3.0, 3.2, 3.5, 3.8],           // 1.0 Gyr
            [0.4, 0.5, 0.55, 0.6, 0.65],         // 5.0 Gyr
            [0.15, 0.18, 0.2, 0.22, 0.25],       // 13.0 Gyr
        ];

        // L_B [L☉/M☉]
        let l_b = [
            [80.0, 85.0, 90.0, 95.0, 100.0], // 0.01 Gyr
            [20.0, 22.0, 24.0, 26.0, 28.0],  // 0.1 Gyr
            [5.5, 6.2, 6.8, 7.2, 7.8],       // 0.5 Gyr
            [3.0, 3.5, 3.8, 4.0, 4.3],       // 1.0 Gyr
            [0.7, 0.8, 0.85, 0.9, 0.95],     // 5.0 Gyr
            [0.3, 0.35, 0.38, 0.4, 0.43],    // 13.0 Gyr
        ];

        // L_V [L☉/M☉]
        let l_v = [
            [50.0, 52.0, 55.0, 58.0, 62.0], // 0.01 Gyr
            [15.0, 17.0, 18.5, 20.0, 22.0], // 0.1 Gyr
            [5.0, 5.8, 6.3, 6.8, 7.4],      // 0.5 Gyr
            [3.2, 3.8, 4.1, 4.4, 4.8],      // 1.0 Gyr
            [1.0, 1.15, 1.25, 1.35, 1.45],  // 5.0 Gyr
            [0.55, 0.62, 0.67, 0.72, 0.78], // 13.0 Gyr
        ];

        // L_R [L☉/M☉]
        let l_r = [
            [35.0, 37.0, 40.0, 43.0, 46.0], // 0.01 Gyr
            [12.0, 13.5, 14.5, 15.5, 17.0], // 0.1 Gyr
            [4.8, 5.5, 6.0, 6.5, 7.0],      // 0.5 Gyr
            [3.5, 4.0, 4.3, 4.6, 5.0],      // 1.0 Gyr
            [1.3, 1.5, 1.6, 1.7, 1.85],     // 5.0 Gyr
            [0.75, 0.85, 0.9, 0.95, 1.02],  // 13.0 Gyr
        ];

        // L_I [L☉/M☉]
        let l_i = [
            [25.0, 27.0, 29.0, 31.0, 33.0], // 0.01 Gyr
            [10.0, 11.2, 12.0, 13.0, 14.0], // 0.1 Gyr
            [4.5, 5.1, 5.5, 6.0, 6.5],      // 0.5 Gyr
            [3.8, 4.3, 4.6, 5.0, 5.4],      // 1.0 Gyr
            [1.7, 1.95, 2.1, 2.25, 2.45],   // 5.0 Gyr
            [1.0, 1.15, 1.25, 1.35, 1.45],  // 13.0 Gyr
        ];

        Self {
            ages_gyr,
            metals,
            l_u,
            l_b,
            l_v,
            l_r,
            l_i,
        }
    }

    /// Interpolación bilineal en la grilla SPS (Phase 153).
    ///
    /// Retorna L/M [L☉/M☉] para la banda, edad y metalicidad dadas.
    pub fn interpolate(&self, age_gyr: f64, metallicity: f64, band: Spsband) -> f64 {
        let age_safe = age_gyr
            .max(self.ages_gyr[0])
            .min(*self.ages_gyr.last().unwrap());
        let z_safe = metallicity
            .max(self.metals[0])
            .min(*self.metals.last().unwrap());

        // Encontrar índices en la grilla
        let ia = self
            .ages_gyr
            .partition_point(|&a| a <= age_safe)
            .saturating_sub(1)
            .min(self.ages_gyr.len() - 2);
        let iz = self
            .metals
            .partition_point(|&m| m <= z_safe)
            .saturating_sub(1)
            .min(self.metals.len() - 2);

        // Fracciones de interpolación
        let fa = if self.ages_gyr[ia + 1] > self.ages_gyr[ia] {
            (age_safe - self.ages_gyr[ia]) / (self.ages_gyr[ia + 1] - self.ages_gyr[ia])
        } else {
            0.0
        }
        .clamp(0.0, 1.0);
        let fz = if self.metals[iz + 1] > self.metals[iz] {
            (z_safe - self.metals[iz]) / (self.metals[iz + 1] - self.metals[iz])
        } else {
            0.0
        }
        .clamp(0.0, 1.0);

        let table = match band {
            Spsband::U => &self.l_u,
            Spsband::B => &self.l_b,
            Spsband::V => &self.l_v,
            Spsband::R => &self.l_r,
            Spsband::I => &self.l_i,
        };

        // Interpolación bilineal: (1-fa)(1-fz)·v00 + fa(1-fz)·v10 + (1-fa)fz·v01 + fa·fz·v11
        let v00 = table[ia][iz];
        let v10 = table[ia + 1][iz];
        let v01 = table[ia][iz + 1];
        let v11 = table[ia + 1][iz + 1];
        (1.0 - fa) * (1.0 - fz) * v00
            + fa * (1.0 - fz) * v10
            + (1.0 - fa) * fz * v01
            + fa * fz * v11
    }
}

/// Luminosidad específica L/M en una banda usando la grilla BC03-lite (Phase 153).
///
/// # Parámetros
/// - `age_gyr`: edad estelar en Gyr (clampado a [0.01, 13] Gyr)
/// - `metallicity`: fracción de masa en metales Z (clampado a [0.0004, 0.05])
/// - `band`: banda fotométrica
///
/// # Retorna
/// L/M en unidades de L☉/M☉.
pub fn sps_luminosity(age_gyr: f64, metallicity: f64, band: Spsband) -> f64 {
    let grid = SpsGrid::bc03_lite();
    grid.interpolate(age_gyr, metallicity, band)
}
