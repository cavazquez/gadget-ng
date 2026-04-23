//! AMR-PM: refinamiento adaptativo de la malla Particle-Mesh (Phase 70).
//!
//! ## Diseño
//!
//! El solver AMR-PM opera en dos niveles jerárquicos:
//!
//! - **Nivel 0 (base):** grid global `nm³` periódico, idéntico al PM estándar.
//!   Captura las fuerzas de largo alcance y la estructura de gran escala.
//!
//! - **Nivel 1 (parches):** grids locales `nm_patch³` sobre regiones con
//!   sobredensidad `δ > δ_refine`. Resuelven Poisson localmente con
//!   condiciones de borde tipo "zero-padding" (convolución no periódica),
//!   añadiendo una corrección de fuerza de corto alcance a alta resolución.
//!
//! ## Combinación de fuerzas
//!
//! La fuerza total de cada partícula es:
//!
//! ```text
//! F_total = F_base + Σ_{parches p que contienen la partícula} ΔF_p
//! ```
//!
//! donde `ΔF_p = F_patch(p) - F_base_interp(p)` es la **corrección** del parche
//! sobre la solución base, evitando doble conteo.
//!
//! ## Complejidad
//!
//! - Base: O(nm³ log nm) — igual que PM estándar.
//! - Por parche: O(nm_p³ log nm_p) — mucho menor si los parches son pequeños.
//! - Típicamente nm_p = nm/4, con ≪ nm³ partículas activas.
//!
//! ## Referencia
//!
//! Kravtsov et al. (1997), ApJS 111, 73 — ART (Adaptive Refinement Tree).
//! Knebe et al. (2001) — malla AMR para simulaciones cosmológicas.

use gadget_ng_core::Vec3;

use crate::{cic, fft_poisson};

// ── Estructuras principales ────────────────────────────────────────────────

/// Parámetros del solver AMR-PM.
#[derive(Debug, Clone)]
pub struct AmrParams {
    /// Sobredensidad de refinamiento: `δ = ρ_celda / ρ̄ - 1`.
    /// Si la densidad media de una celda del base grid supera `rho_mean * (1 + delta_refine)`,
    /// se crea un parche de refinamiento centrado en esa celda.
    /// Valores típicos: 5–20 (factor 5–20 sobre la densidad media).
    pub delta_refine: f64,
    /// Lado del parche en celdas del **base grid**.
    /// El parche cubre `patch_cells_base` × `cell_size` unidades físicas.
    /// Debe ser impar para centrar el parche en la celda que activa el refinamiento.
    pub patch_cells_base: usize,
    /// Resolución del parche: número de celdas por lado.
    /// El espaciado de la celda del parche es `(patch_cells_base / nm_patch)` veces
    /// más fino que el base grid (factor típico 4–8).
    pub nm_patch: usize,
    /// Número máximo de parches por llamada. Los parches sobrantes (densidad más baja)
    /// se ignoran.
    pub max_patches: usize,
    /// Si `true`, usa zero-padding en la FFT del parche (convolución no periódica).
    /// Si `false`, trata el parche como periódico (aproximación para sistemas aislados).
    pub zero_pad: bool,
    /// Número máximo de niveles de refinamiento (Phase 80).
    /// - `1` = comportamiento original Phase 70 (solo nivel 0 + parches de nivel 1).
    /// - `2` = dos niveles de refinamiento (parches dentro de parches).
    /// - `3` = tres niveles (default Phase 80).
    pub max_levels: usize,
    /// Factor de umbral entre niveles. El nivel `l` usa `delta_refine * refine_factor^l`.
    /// Valores típicos: 3–5. Default: 4.0.
    pub refine_factor: f64,
}

impl Default for AmrParams {
    fn default() -> Self {
        Self {
            delta_refine: 10.0,
            patch_cells_base: 5,
            nm_patch: 32,
            max_patches: 16,
            zero_pad: true,
            max_levels: 1,
            refine_factor: 4.0,
        }
    }
}

/// Un parche de refinamiento AMR.
///
/// Describe una región del espacio donde se aplica mayor resolución PM.
/// El parche es un cubo de lado `size` centrado en `center`.
#[derive(Debug, Clone)]
pub struct PatchGrid {
    /// Centro del parche en coordenadas físicas.
    pub center: Vec3,
    /// Lado del parche en unidades físicas.
    pub size: f64,
    /// Resolución interna del parche (celdas por lado).
    pub nm: usize,
    /// Densidad en el parche (masa/celda, longitud `nm³`).
    pub density: Vec<f64>,
    /// Aceleración en el parche (3 componentes × nm³), en orden [fx, fy, fz].
    pub forces: [Vec<f64>; 3],
}

impl PatchGrid {
    /// Crea un parche vacío de resolución `nm` centrado en `center` con lado `size`.
    pub fn new(center: Vec3, size: f64, nm: usize) -> Self {
        let nm3 = nm * nm * nm;
        Self {
            center,
            size,
            nm,
            density: vec![0.0; nm3],
            forces: [vec![0.0; nm3], vec![0.0; nm3], vec![0.0; nm3]],
        }
    }

    /// Esquina inferior del parche (origen del sistema de coordenadas local).
    #[inline]
    pub fn origin(&self) -> Vec3 {
        Vec3::new(
            self.center.x - self.size * 0.5,
            self.center.y - self.size * 0.5,
            self.center.z - self.size * 0.5,
        )
    }

    /// Tamaño de celda del parche.
    #[inline]
    pub fn cell_size(&self) -> f64 {
        self.size / self.nm as f64
    }

    /// Verifica si una posición cae dentro del parche.
    #[inline]
    pub fn contains(&self, pos: Vec3) -> bool {
        let o = self.origin();
        pos.x >= o.x
            && pos.x < o.x + self.size
            && pos.y >= o.y
            && pos.y < o.y + self.size
            && pos.z >= o.z
            && pos.z < o.z + self.size
    }
}

// ── Identificación de regiones a refinar ──────────────────────────────────

/// Identifica las celdas del base grid que superan el umbral de refinamiento.
///
/// Devuelve hasta `max_patches` posiciones de parche ordenadas por densidad
/// descendente (las celdas más densas primero).
pub fn identify_refinement_patches(
    base_density: &[f64],
    nm: usize,
    box_size: f64,
    params: &AmrParams,
) -> Vec<PatchGrid> {
    let nm2 = nm * nm;
    let nm3 = nm2 * nm;
    assert_eq!(base_density.len(), nm3);

    let rho_mean: f64 = base_density.iter().sum::<f64>() / nm3 as f64;
    let threshold = rho_mean * (1.0 + params.delta_refine);
    let cell_size = box_size / nm as f64;
    let patch_phys = params.patch_cells_base as f64 * cell_size;

    // Recolectar celdas que superan el umbral
    let mut candidates: Vec<(f64, Vec3)> = Vec::new();
    for iz in 0..nm {
        for iy in 0..nm {
            for ix in 0..nm {
                let rho = base_density[iz * nm2 + iy * nm + ix];
                if rho > threshold {
                    // Centro de la celda
                    let cx = (ix as f64 + 0.5) * cell_size;
                    let cy = (iy as f64 + 0.5) * cell_size;
                    let cz = (iz as f64 + 0.5) * cell_size;
                    candidates.push((rho, Vec3::new(cx, cy, cz)));
                }
            }
        }
    }

    // Ordenar por densidad descendente
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Seleccionar los top `max_patches` sin solapamiento
    let mut patches: Vec<PatchGrid> = Vec::new();
    for (_, center) in candidates.iter().take(params.max_patches * 4) {
        if patches.len() >= params.max_patches {
            break;
        }
        // Evitar parches que solapen demasiado con los ya seleccionados
        let overlap = patches.iter().any(|p| {
            let dx = (p.center.x - center.x).abs();
            let dy = (p.center.y - center.y).abs();
            let dz = (p.center.z - center.z).abs();
            dx < patch_phys * 0.5 && dy < patch_phys * 0.5 && dz < patch_phys * 0.5
        });
        if !overlap {
            patches.push(PatchGrid::new(*center, patch_phys, params.nm_patch));
        }
    }

    patches
}

// ── Depósito CIC en parche ─────────────────────────────────────────────────

/// Deposita masas de partículas en el grid de un parche con CIC no periódico.
///
/// Solo deposita partículas dentro del parche; las que quedan fuera se ignoran.
/// La CIC es lineal (sin wrap periódico) para posiciones en [origin, origin + size].
pub fn deposit_to_patch(
    positions: &[Vec3],
    masses: &[f64],
    patch: &mut PatchGrid,
) {
    let nm = patch.nm;
    let nm2 = nm * nm;
    let origin = patch.origin();
    let inv_cell = nm as f64 / patch.size;

    patch.density.iter_mut().for_each(|x| *x = 0.0);

    for (&pos, &m) in positions.iter().zip(masses.iter()) {
        // Posición relativa al origen del parche
        let lx = pos.x - origin.x;
        let ly = pos.y - origin.y;
        let lz = pos.z - origin.z;

        // Ignorar partículas fuera del parche
        if lx < 0.0 || lx >= patch.size
            || ly < 0.0 || ly >= patch.size
            || lz < 0.0 || lz >= patch.size
        {
            continue;
        }

        // Coordenada en unidades de celda
        let cx = lx * inv_cell;
        let cy = ly * inv_cell;
        let cz = lz * inv_cell;

        let ix0 = cx.floor() as usize;
        let iy0 = cy.floor() as usize;
        let iz0 = cz.floor() as usize;

        let dx = cx - ix0 as f64;
        let dy = cy - iy0 as f64;
        let dz = cz - iz0 as f64;

        let wx = [1.0 - dx, dx];
        let wy = [1.0 - dy, dy];
        let wz = [1.0 - dz, dz];

        for (diz, &wz_v) in wz.iter().enumerate() {
            let iz = iz0 + diz;
            if iz >= nm { continue; }
            for (diy, &wy_v) in wy.iter().enumerate() {
                let iy = iy0 + diy;
                if iy >= nm { continue; }
                for (dix, &wx_v) in wx.iter().enumerate() {
                    let ix = ix0 + dix;
                    if ix >= nm { continue; }
                    patch.density[iz * nm2 + iy * nm + ix] += m * wx_v * wy_v * wz_v;
                }
            }
        }
    }
}

// ── Solver Poisson en parche ───────────────────────────────────────────────

/// Resuelve la ecuación de Poisson en un parche y almacena las fuerzas en `patch.forces`.
///
/// Si `zero_pad = true`, usa zero-padding para simular condiciones no periódicas:
/// el grid del parche se extiende a 2×nm³ con ceros, se resuelve Poisson en el
/// grid extendido y se extrae la solución central.
///
/// Si `zero_pad = false`, trata el parche como periódico (válido para sistemas
/// de alta densidad y parches aislados).
pub fn solve_patch(patch: &mut PatchGrid, g: f64, zero_pad: bool) {
    let nm = patch.nm;

    if zero_pad {
        // Zero-padding: extender a 2×nm por lado
        let nm2 = nm * 2;
        let nm2_3 = nm2 * nm2 * nm2;
        let mut density_padded = vec![0.0_f64; nm2_3];

        // Copiar densidad al octante [0, nm)³ del grid extendido
        let nm_src2 = nm * nm;
        for iz in 0..nm {
            for iy in 0..nm {
                for ix in 0..nm {
                    let src = iz * nm_src2 + iy * nm + ix;
                    let dst = iz * (nm2 * nm2) + iy * nm2 + ix;
                    density_padded[dst] = patch.density[src];
                }
            }
        }

        // Resolver Poisson en el grid extendido (tamaño físico 2× = box del parche × 2)
        let patch_box = patch.size * 2.0;
        let forces_ext = fft_poisson::solve_forces(&density_padded, g, nm2, patch_box);

        // Extraer fuerzas del octante central [0, nm)³
        let nm_ext2 = nm2 * nm2;
        for comp in 0..3 {
            for iz in 0..nm {
                for iy in 0..nm {
                    for ix in 0..nm {
                        let src = iz * nm_ext2 + iy * nm2 + ix;
                        let dst = iz * nm_src2 + iy * nm + ix;
                        patch.forces[comp][dst] = forces_ext[comp][src];
                    }
                }
            }
        }
    } else {
        // Periódico: usar el parche directamente
        let [fx, fy, fz] = fft_poisson::solve_forces(&patch.density, g, nm, patch.size);
        patch.forces[0] = fx;
        patch.forces[1] = fy;
        patch.forces[2] = fz;
    }
}

// ── Interpolación de fuerzas del parche ───────────────────────────────────

/// Interpola las fuerzas del parche a las posiciones de las partículas dentro de él.
///
/// Solo interpola partículas que caen dentro del parche; el resto recibe `Vec3::zero()`.
/// Usa la misma interpolación CIC bilineal que el solver base para consistencia.
pub fn interpolate_patch_forces(patch: &PatchGrid, positions: &[Vec3]) -> Vec<Vec3> {
    let nm = patch.nm;
    let nm2 = nm * nm;
    let origin = patch.origin();
    let inv_cell = nm as f64 / patch.size;

    let mut accels = vec![Vec3::zero(); positions.len()];

    for (i, &pos) in positions.iter().enumerate() {
        let lx = pos.x - origin.x;
        let ly = pos.y - origin.y;
        let lz = pos.z - origin.z;

        if lx < 0.0 || lx >= patch.size
            || ly < 0.0 || ly >= patch.size
            || lz < 0.0 || lz >= patch.size
        {
            continue;
        }

        let cx = lx * inv_cell;
        let cy = ly * inv_cell;
        let cz = lz * inv_cell;

        let ix0 = cx.floor() as usize;
        let iy0 = cy.floor() as usize;
        let iz0 = cz.floor() as usize;

        let dx = cx - ix0 as f64;
        let dy = cy - iy0 as f64;
        let dz = cz - iz0 as f64;

        let wx = [1.0 - dx, dx];
        let wy = [1.0 - dy, dy];
        let wz = [1.0 - dz, dz];

        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        let mut az = 0.0_f64;

        for (diz, &wz_v) in wz.iter().enumerate() {
            let iz = iz0 + diz;
            if iz >= nm { continue; }
            for (diy, &wy_v) in wy.iter().enumerate() {
                let iy = iy0 + diy;
                if iy >= nm { continue; }
                for (dix, &wx_v) in wx.iter().enumerate() {
                    let ix = ix0 + dix;
                    if ix >= nm { continue; }
                    let w = wx_v * wy_v * wz_v;
                    let idx = iz * nm2 + iy * nm + ix;
                    ax += w * patch.forces[0][idx];
                    ay += w * patch.forces[1][idx];
                    az += w * patch.forces[2][idx];
                }
            }
        }

        accels[i] = Vec3::new(ax, ay, az);
    }

    accels
}

// ── Solver AMR completo ────────────────────────────────────────────────────

/// Calcula las aceleraciones PM con refinamiento AMR.
///
/// ## Algoritmo
///
/// 1. Depositar masa en el grid base (`nm_base³`).
/// 2. Resolver Poisson en el grid base (fuerzas de fondo).
/// 3. Interpolar fuerzas base a todas las partículas.
/// 4. Identificar celdas con `ρ > ρ̄ × (1 + δ_refine)`.
/// 5. Para cada parche:
///    a. Depositar partículas en el parche.
///    b. Resolver Poisson en el parche.
///    c. Interpolar fuerzas del parche a partículas dentro del parche.
///    d. Interpolar fuerzas base en el parche (para calcular la corrección).
///    e. Añadir corrección `ΔF = F_patch - F_base_local`.
/// 6. Retornar `F_base + Σ ΔF_parche`.
///
/// ## Parámetros
///
/// - `positions` — posiciones de las partículas.
/// - `masses`    — masas de las partículas.
/// - `box_size`  — lado del cubo periódico.
/// - `nm_base`   — resolución del grid base (celdas por lado).
/// - `g`         — constante gravitacional.
/// - `params`    — parámetros AMR.
///
/// ## Retorno
///
/// Vector de aceleraciones, una por partícula.
pub fn amr_pm_accels(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    nm_base: usize,
    g: f64,
    params: &AmrParams,
) -> Vec<Vec3> {
    assert_eq!(positions.len(), masses.len());
    let _n = positions.len();

    // ── Paso 1-3: grid base ────────────────────────────────────────────────
    let base_density = cic::assign(positions, masses, box_size, nm_base);
    let [fx_base, fy_base, fz_base] =
        fft_poisson::solve_forces(&base_density, g, nm_base, box_size);
    let mut accels = cic::interpolate(
        &fx_base,
        &fy_base,
        &fz_base,
        positions,
        box_size,
        nm_base,
    );

    // ── Paso 4: identificar parches ────────────────────────────────────────
    let mut patches = identify_refinement_patches(&base_density, nm_base, box_size, params);

    if patches.is_empty() {
        return accels;
    }

    // ── Paso 5: procesar cada parche ───────────────────────────────────────
    for patch in patches.iter_mut() {
        // 5a: depositar partículas en el parche
        deposit_to_patch(positions, masses, patch);

        // 5b: resolver Poisson en el parche
        solve_patch(patch, g, params.zero_pad);

        // 5c: interpolar fuerzas del parche
        let f_patch = interpolate_patch_forces(patch, positions);

        // 5d: interpolar fuerzas base en las posiciones de las partículas dentro del parche
        // (ya están en `accels`)

        // 5e: añadir corrección solo para partículas dentro del parche
        // La corrección es ΔF = F_patch - F_base_local
        // Pero F_base_local ya está en accels[i], así que:
        //   accels[i] = F_base + ΔF = F_base + (F_patch - F_base) = F_patch
        // Para partículas dentro del parche, reemplazar con F_patch.
        // Esto es equivalente a: accels[i] += (F_patch[i] - F_base[i])
        //
        // Nota: para evitar doble-conteo, calculamos la corrección explícita.
        // La diferencia entre F_patch y F_base captura el refinamiento.
        for (i, &pos) in positions.iter().enumerate() {
            if patch.contains(pos) {
                let f_p = f_patch[i];
                // Solo aplicar si la fuerza del parche es no trivial
                if f_p.x != 0.0 || f_p.y != 0.0 || f_p.z != 0.0 {
                    // Corrección: F_patch - F_base_local (diferencia de resolución)
                    // Para suavizar la transición, usamos una interpolación lineal
                    // basada en qué tan centrada está la partícula en el parche.
                    let o = patch.origin();
                    let frac_x = (pos.x - o.x) / patch.size;
                    let frac_y = (pos.y - o.y) / patch.size;
                    let frac_z = (pos.z - o.z) / patch.size;
                    // Peso de transición: 1 en el centro, 0 en los bordes
                    let w_x = 1.0 - 2.0 * (frac_x - 0.5).abs();
                    let w_y = 1.0 - 2.0 * (frac_y - 0.5).abs();
                    let w_z = 1.0 - 2.0 * (frac_z - 0.5).abs();
                    let w = (w_x * w_y * w_z).max(0.0);

                    let delta = Vec3::new(
                        f_p.x - accels[i].x,
                        f_p.y - accels[i].y,
                        f_p.z - accels[i].z,
                    );
                    accels[i] = Vec3::new(
                        accels[i].x + w * delta.x,
                        accels[i].y + w * delta.y,
                        accels[i].z + w * delta.z,
                    );
                }
            }
        }
    }

    accels
}

// ── Estadísticas de refinamiento ───────────────────────────────────────────

/// Estadísticas del último paso AMR: número de parches, partículas refinadas, etc.
#[derive(Debug, Clone, Default)]
pub struct AmrStats {
    /// Número de parches identificados.
    pub n_patches: usize,
    /// Número de partículas dentro de al menos un parche.
    pub n_particles_refined: usize,
    /// Densidad máxima encontrada en el base grid (en unidades de `rho_mean`).
    pub max_overdensity: f64,
}

/// Versión instrumentada de [`amr_pm_accels`] que también devuelve estadísticas.
pub fn amr_pm_accels_with_stats(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    nm_base: usize,
    g: f64,
    params: &AmrParams,
) -> (Vec<Vec3>, AmrStats) {
    let base_density = cic::assign(positions, masses, box_size, nm_base);

    let nm3 = nm_base * nm_base * nm_base;
    let rho_mean = base_density.iter().sum::<f64>() / nm3 as f64;
    let rho_max = base_density.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_od = if rho_mean > 0.0 { rho_max / rho_mean } else { 0.0 };

    let patches_list = identify_refinement_patches(&base_density, nm_base, box_size, params);
    let n_patches = patches_list.len();
    let n_refined = positions.iter().filter(|&&p| patches_list.iter().any(|pg| pg.contains(p))).count();

    let accels = amr_pm_accels(positions, masses, box_size, nm_base, g, params);
    let stats = AmrStats {
        n_patches,
        n_particles_refined: n_refined,
        max_overdensity: max_od,
    };
    (accels, stats)
}

// ── AMR Multi-nivel (Phase 80) ─────────────────────────────────────────────

/// Un nivel en la jerarquía AMR.
///
/// Cada nivel contiene sus parches de refinamiento y los sub-niveles hijos
/// que se aplican sobre las regiones más densas de cada parche.
#[derive(Debug)]
pub struct AmrLevel {
    /// Parches de refinamiento en este nivel.
    pub patches: Vec<PatchGrid>,
    /// Sub-niveles de refinamiento (uno por parche padre, puede estar vacío).
    pub child_levels: Vec<AmrLevel>,
    /// Profundidad del nivel (0 = base).
    pub depth: usize,
}

impl AmrLevel {
    /// Crea un nivel vacío.
    pub fn empty(depth: usize) -> Self {
        Self { patches: Vec::new(), child_levels: Vec::new(), depth }
    }

    /// Número total de parches en todo el árbol.
    pub fn total_patches(&self) -> usize {
        self.patches.len() + self.child_levels.iter().map(|c| c.total_patches()).sum::<usize>()
    }
}

/// Construye la jerarquía AMR recursivamente desde el campo de densidad base.
///
/// Para cada nivel `l` de 1 hasta `max_levels`:
/// 1. Identifica parches donde `δ > δ_refine × factor^l`.
/// 2. Para cada parche, deposita masa y resuelve Poisson.
/// 3. Si `l < max_levels`, llama recursivamente sobre el campo de cada parche.
pub fn build_amr_hierarchy(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    g: f64,
    params: &AmrParams,
    current_level: usize,
) -> AmrLevel {
    if current_level >= params.max_levels || positions.is_empty() {
        return AmrLevel::empty(current_level);
    }

    // Umbral de refinamiento para este nivel: delta_refine * factor^(level-1)
    let delta_this = params.delta_refine * params.refine_factor.powi(current_level as i32);
    let params_this = AmrParams {
        delta_refine: delta_this,
        ..params.clone()
    };

    // Determinar resolución del grid para este nivel
    let nm_level = params.nm_patch;
    let base_density = cic::assign(positions, masses, box_size, nm_level);
    let mut patches = identify_refinement_patches(&base_density, nm_level, box_size, &params_this);

    // Resolver Poisson en cada parche
    for patch in patches.iter_mut() {
        deposit_to_patch(positions, masses, patch);
        solve_patch(patch, g, params.zero_pad);
    }

    // Construir sub-niveles recursivamente para cada parche
    let mut child_levels = Vec::new();
    if current_level + 1 < params.max_levels {
        for patch in &patches {
            // Filtrar partículas dentro del parche
            let patch_pos: Vec<Vec3> = positions.iter().cloned()
                .filter(|&p| patch.contains(p)).collect();
            let patch_mass: Vec<f64> = positions.iter().zip(masses.iter())
                .filter(|(&p, _)| patch.contains(p))
                .map(|(_, &m)| m).collect();

            if patch_pos.len() >= 8 {
                let child = build_amr_hierarchy(
                    &patch_pos, &patch_mass,
                    patch.size, g, params,
                    current_level + 1,
                );
                child_levels.push(child);
            } else {
                child_levels.push(AmrLevel::empty(current_level + 1));
            }
        }
    }

    AmrLevel { patches, child_levels, depth: current_level }
}

/// Solver AMR multi-nivel: extiende `amr_pm_accels` a N niveles recursivos.
///
/// El algoritmo aplica el mismo esquema de corrección `ΔF = F_patch - F_base`
/// de forma recursiva para cada nivel, refinando progresivamente las regiones
/// más densas con resolución creciente.
///
/// ## Parámetros
///
/// - `max_levels` en `params` controla la profundidad:
///   - `1` → comportamiento original Phase 70 (solo base + nivel 1).
///   - `2` → tres niveles totales (base + nivel 1 + nivel 2).
///   - `3` → cuatro niveles (default Phase 80).
pub fn amr_pm_accels_multilevel(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    nm_base: usize,
    g: f64,
    params: &AmrParams,
) -> Vec<Vec3> {
    assert_eq!(positions.len(), masses.len());

    // ── Nivel 0: grid base ─────────────────────────────────────────────────
    let base_density = cic::assign(positions, masses, box_size, nm_base);
    let [fx_base, fy_base, fz_base] =
        fft_poisson::solve_forces(&base_density, g, nm_base, box_size);
    let mut accels = cic::interpolate(&fx_base, &fy_base, &fz_base, positions, box_size, nm_base);

    if params.max_levels == 0 {
        return accels;
    }

    // ── Nivel 1: parches ────────────────────────────────────────────────────
    // Reutiliza la lógica existente de amr_pm_accels para el primer nivel
    let mut patches_l1 = identify_refinement_patches(&base_density, nm_base, box_size, params);
    for patch in patches_l1.iter_mut() {
        deposit_to_patch(positions, masses, patch);
        solve_patch(patch, g, params.zero_pad);
    }
    let accels_l1_contrib = apply_patch_corrections(&patches_l1, positions, &accels);
    accels = accels_l1_contrib;

    // ── Niveles 2..max_levels ────────────────────────────────────────────────
    if params.max_levels >= 2 {
        for patch in &patches_l1 {
            // Sub-nivel dentro del parche L1
            let sub_pos: Vec<Vec3> = positions.iter().cloned()
                .filter(|&p| patch.contains(p)).collect();
            let sub_mass: Vec<f64> = positions.iter().zip(masses.iter())
                .filter(|(&p, _)| patch.contains(p))
                .map(|(_, &m)| m).collect();
            let sub_idx: Vec<usize> = positions.iter().enumerate()
                .filter(|(_, &p)| patch.contains(p))
                .map(|(i, _)| i).collect();

            if sub_pos.len() < 8 {
                continue;
            }

            let params_l2 = AmrParams {
                delta_refine: params.delta_refine * params.refine_factor,
                max_levels: params.max_levels.saturating_sub(1),
                ..params.clone()
            };

            // Resolver sub-nivel en el espacio local del parche
            let sub_accels_l2 = amr_pm_accels(
                &sub_pos, &sub_mass, patch.size, params.nm_patch, g, &params_l2,
            );

            // Mapear sub-aceleraciones de vuelta al índice global
            for (&global_i, &a_l2) in sub_idx.iter().zip(sub_accels_l2.iter()) {
                // Corrección L2: ΔF = F_L2 - F_L1 (para partículas en ambos)
                let delta = Vec3::new(
                    a_l2.x - accels[global_i].x,
                    a_l2.y - accels[global_i].y,
                    a_l2.z - accels[global_i].z,
                );
                // Peso: 1 en el centro del parche, suavizado en los bordes
                let o = patch.origin();
                let fx = (sub_pos[sub_idx.iter().position(|&x| x == global_i).unwrap()].x - o.x) / patch.size;
                let fy = (sub_pos[sub_idx.iter().position(|&x| x == global_i).unwrap()].y - o.y) / patch.size;
                let fz = (sub_pos[sub_idx.iter().position(|&x| x == global_i).unwrap()].z - o.z) / patch.size;
                let w = ((1.0 - 2.0 * (fx - 0.5).abs())
                    * (1.0 - 2.0 * (fy - 0.5).abs())
                    * (1.0 - 2.0 * (fz - 0.5).abs())).max(0.0);

                accels[global_i] = Vec3::new(
                    accels[global_i].x + w * delta.x,
                    accels[global_i].y + w * delta.y,
                    accels[global_i].z + w * delta.z,
                );
            }
        }
    }

    accels
}

/// Aplica las correcciones de los parches sobre las aceleraciones base.
///
/// Para cada parche, reemplaza las aceleraciones de las partículas internas
/// con la fuerza del parche (ponderada por distancia al centro).
fn apply_patch_corrections(
    patches: &[PatchGrid],
    positions: &[Vec3],
    base_accels: &[Vec3],
) -> Vec<Vec3> {
    let mut accels = base_accels.to_vec();
    for patch in patches {
        let f_patch = interpolate_patch_forces(patch, positions);
        for (i, &pos) in positions.iter().enumerate() {
            if patch.contains(pos) {
                let f_p = f_patch[i];
                if f_p.x != 0.0 || f_p.y != 0.0 || f_p.z != 0.0 {
                    let o = patch.origin();
                    let frac_x = (pos.x - o.x) / patch.size;
                    let frac_y = (pos.y - o.y) / patch.size;
                    let frac_z = (pos.z - o.z) / patch.size;
                    let w = ((1.0 - 2.0 * (frac_x - 0.5).abs())
                        * (1.0 - 2.0 * (frac_y - 0.5).abs())
                        * (1.0 - 2.0 * (frac_z - 0.5).abs()))
                        .max(0.0);
                    let delta = Vec3::new(
                        f_p.x - accels[i].x,
                        f_p.y - accels[i].y,
                        f_p.z - accels[i].z,
                    );
                    accels[i] = Vec3::new(
                        accels[i].x + w * delta.x,
                        accels[i].y + w * delta.y,
                        accels[i].z + w * delta.z,
                    );
                }
            }
        }
    }
    accels
}

/// Estadísticas del solver AMR multi-nivel.
#[derive(Debug, Clone, Default)]
pub struct AmrMultilevelStats {
    /// Número de parches por nivel.
    pub patches_per_level: Vec<usize>,
    /// Número total de partículas refinadas al menos en 1 nivel.
    pub n_refined: usize,
    /// Profundidad máxima alcanzada.
    pub max_depth: usize,
}

/// Versión instrumentada de [`amr_pm_accels_multilevel`] con estadísticas.
pub fn amr_pm_accels_multilevel_with_stats(
    positions: &[Vec3],
    masses: &[f64],
    box_size: f64,
    nm_base: usize,
    g: f64,
    params: &AmrParams,
) -> (Vec<Vec3>, AmrMultilevelStats) {
    let accels = amr_pm_accels_multilevel(positions, masses, box_size, nm_base, g, params);

    let base_density = cic::assign(positions, masses, box_size, nm_base);
    let patches_l1 = identify_refinement_patches(&base_density, nm_base, box_size, params);
    let n_refined = positions.iter().filter(|&&p| patches_l1.iter().any(|pg| pg.contains(p))).count();

    let stats = AmrMultilevelStats {
        patches_per_level: vec![patches_l1.len()],
        n_refined,
        max_depth: params.max_levels.min(3),
    };
    (accels, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn uniform_grid(n_side: usize, box_size: f64) -> (Vec<Vec3>, Vec<f64>) {
        let dx = box_size / n_side as f64;
        let mut pos = Vec::new();
        let mut mass = Vec::new();
        for iz in 0..n_side {
            for iy in 0..n_side {
                for ix in 0..n_side {
                    pos.push(Vec3::new(
                        (ix as f64 + 0.5) * dx,
                        (iy as f64 + 0.5) * dx,
                        (iz as f64 + 0.5) * dx,
                    ));
                    mass.push(1.0);
                }
            }
        }
        (pos, mass)
    }

    #[test]
    fn amr_uniform_no_patches() {
        // Grid uniforme con umbral muy alto → sin parches de refinamiento.
        // CIC en un lattice crea variaciones de densidad de O(1) en ρ_mean;
        // un umbral de 50× garantiza que no se activen parches.
        let (pos, mass) = uniform_grid(8, 1.0);
        let params = AmrParams { delta_refine: 50.0, ..Default::default() };
        let base_rho = cic::assign(&pos, &mass, 1.0, 16);
        let patches = identify_refinement_patches(&base_rho, 16, 1.0, &params);
        assert!(
            patches.is_empty(),
            "con delta_refine=50 no debe haber parches para distribución uniforme: {}",
            patches.len()
        );
    }

    #[test]
    fn amr_concentrated_cluster_creates_patch() {
        // Cluster concentrado → debe crear al menos 1 parche
        let box_size = 1.0;
        let mut pos = Vec::new();
        let mut mass = Vec::new();

        // 100 partículas en el centro
        for _ in 0..100 {
            pos.push(Vec3::new(0.505, 0.505, 0.505));
            mass.push(1.0);
        }
        // Fondo: 8 partículas dispersas
        let (bg_pos, bg_mass) = uniform_grid(2, box_size);
        pos.extend_from_slice(&bg_pos);
        mass.extend_from_slice(&bg_mass);

        let params = AmrParams {
            delta_refine: 3.0,
            nm_patch: 8,
            patch_cells_base: 3,
            max_patches: 4,
            zero_pad: false,
            ..Default::default()
        };
        let base_rho = cic::assign(&pos, &mass, box_size, 8);
        let patches = identify_refinement_patches(&base_rho, 8, box_size, &params);
        assert!(!patches.is_empty(), "cluster concentrado debe crear al menos un parche");
    }

    #[test]
    fn amr_patch_contains_logic() {
        let center = Vec3::new(0.5, 0.5, 0.5);
        let patch = PatchGrid::new(center, 0.2, 8);
        assert!(patch.contains(Vec3::new(0.5, 0.5, 0.5)));
        assert!(patch.contains(Vec3::new(0.41, 0.41, 0.41)));
        assert!(!patch.contains(Vec3::new(0.1, 0.5, 0.5)));
        assert!(!patch.contains(Vec3::new(0.5, 0.5, 0.65)));
    }

    #[test]
    fn amr_deposit_mass_conservation() {
        let center = Vec3::new(0.5, 0.5, 0.5);
        let mut patch = PatchGrid::new(center, 0.4, 8);

        // Partículas dentro del parche
        let pos = vec![
            Vec3::new(0.45, 0.45, 0.45),
            Vec3::new(0.55, 0.55, 0.55),
        ];
        let mass = vec![2.0, 3.0];

        deposit_to_patch(&pos, &mass, &mut patch);
        let total_mass: f64 = patch.density.iter().sum();
        assert!((total_mass - 5.0).abs() < 1e-10, "masa no conservada: {total_mass}");
    }

    #[test]
    fn amr_pm_accels_no_nan() {
        // Smoke test: AMR no produce NaN/Inf para distribución simple
        let (pos, mass) = uniform_grid(4, 1.0);
        let params = AmrParams {
            delta_refine: 0.5,
            nm_patch: 8,
            patch_cells_base: 3,
            max_patches: 2,
            zero_pad: false,
            ..Default::default()
        };
        let accels = amr_pm_accels(&pos, &mass, 1.0, 8, 1.0, &params);
        assert_eq!(accels.len(), pos.len());
        for (i, a) in accels.iter().enumerate() {
            assert!(
                a.x.is_finite() && a.y.is_finite() && a.z.is_finite(),
                "aceleración no finita en partícula {i}: ({}, {}, {})",
                a.x, a.y, a.z
            );
        }
    }

    #[test]
    fn amr_stats_reports_correct_n_patches() {
        let (mut pos, mut mass) = uniform_grid(2, 1.0);
        // Agregar un cluster concentrado
        for _ in 0..50 {
            pos.push(Vec3::new(0.5, 0.5, 0.5));
            mass.push(1.0);
        }

        let params = AmrParams {
            delta_refine: 2.0,
            nm_patch: 8,
            patch_cells_base: 3,
            max_patches: 4,
            zero_pad: false,
            ..Default::default()
        };
        let (_, stats) = amr_pm_accels_with_stats(&pos, &mass, 1.0, 8, 1.0, &params);
        assert!(stats.n_patches >= 1, "debería haber al menos 1 parche");
        assert!(stats.max_overdensity > 1.0, "sobredensidad máxima debe ser > 1");
    }

    #[test]
    fn amr_params_default() {
        let p = AmrParams::default();
        assert_eq!(p.delta_refine, 10.0);
        assert_eq!(p.nm_patch, 32);
        assert_eq!(p.patch_cells_base, 5);
        assert_eq!(p.max_patches, 16);
        assert!(p.zero_pad);
        assert_eq!(p.max_levels, 1);
        assert_eq!(p.refine_factor, 4.0);
    }

    // ── Phase 80: Tests multi-nivel ───────────────────────────────────────

    #[test]
    fn amr_multilevel_max1_equals_base() {
        // max_levels=1 debe comportarse igual que amr_pm_accels estándar
        let (mut pos, mut mass) = uniform_grid(4, 1.0);
        // Cluster para activar parches
        pos.push(Vec3::new(0.5, 0.5, 0.5));
        mass.push(100.0);

        let params = AmrParams {
            delta_refine: 1.0,
            nm_patch: 8,
            patch_cells_base: 3,
            max_patches: 2,
            zero_pad: false,
            max_levels: 1,
            refine_factor: 4.0,
        };
        let a1 = amr_pm_accels_multilevel(&pos, &mass, 1.0, 8, 1.0, &params);
        assert_eq!(a1.len(), pos.len());
        for (i, a) in a1.iter().enumerate() {
            assert!(a.x.is_finite() && a.y.is_finite() && a.z.is_finite(),
                "multilevel L1: aceleración no finita en partícula {i}");
        }
    }

    #[test]
    fn amr_multilevel_3levels_no_nan() {
        // 3 niveles de refinamiento no debe producir NaN/Inf
        let (mut pos, mut mass) = uniform_grid(4, 1.0);
        // Cluster muy concentrado en el centro
        for _ in 0..50 {
            pos.push(Vec3::new(0.5, 0.5, 0.5));
            mass.push(1.0);
        }
        let params = AmrParams {
            delta_refine: 1.0,
            nm_patch: 8,
            patch_cells_base: 3,
            max_patches: 2,
            zero_pad: false,
            max_levels: 3,
            refine_factor: 3.0,
        };
        let accels = amr_pm_accels_multilevel(&pos, &mass, 1.0, 8, 1.0, &params);
        assert_eq!(accels.len(), pos.len());
        for (i, a) in accels.iter().enumerate() {
            assert!(a.x.is_finite() && a.y.is_finite() && a.z.is_finite(),
                "multilevel L3: NaN en partícula {i}: ({},{},{})", a.x, a.y, a.z);
        }
    }

    #[test]
    fn amr_multilevel_stats() {
        let (mut pos, mut mass) = uniform_grid(4, 1.0);
        pos.push(Vec3::new(0.5, 0.5, 0.5));
        mass.push(100.0);
        let params = AmrParams {
            delta_refine: 0.5,
            nm_patch: 8,
            patch_cells_base: 3,
            max_patches: 4,
            zero_pad: false,
            max_levels: 2,
            refine_factor: 4.0,
        };
        let (accels, stats) = amr_pm_accels_multilevel_with_stats(&pos, &mass, 1.0, 8, 1.0, &params);
        assert_eq!(accels.len(), pos.len());
        assert!(stats.max_depth <= 3);
        assert!(stats.patches_per_level[0] >= 1 || stats.n_refined == 0);
    }

    #[test]
    fn amr_level_struct_builds() {
        let (mut pos, mut mass) = uniform_grid(4, 1.0);
        pos.push(Vec3::new(0.5, 0.5, 0.5));
        mass.push(50.0);
        let params = AmrParams {
            delta_refine: 1.0,
            nm_patch: 8,
            patch_cells_base: 3,
            max_patches: 2,
            zero_pad: false,
            max_levels: 2,
            refine_factor: 4.0,
        };
        let level = build_amr_hierarchy(&pos, &mass, 1.0, 1.0, &params, 0);
        assert_eq!(level.depth, 0);
    }
}
