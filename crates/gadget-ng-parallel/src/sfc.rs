//! Curvas SFC (Space-Filling Curves) 3D para partición de dominio.
//!
//! Implementa dos curvas seleccionables mediante [`SfcKind`]:
//!
//! - **Morton** (Z-order, default): 21 bits/eje, 63 bits totales. Implementación
//!   clásica con "magic bits". Retrocompatible con Fases 8-12.
//!
//! - **Hilbert** (Peano-Hilbert 3D): mismo rango de claves que Morton (63 bits),
//!   mejor localidad espacial — puntos vecinos en 3D tienden a tener claves más
//!   cercanas que con Morton. Usa el algoritmo de Skilling (2004).
//!
//! # Uso
//!
//! ```rust
//! use gadget_ng_parallel::sfc::{morton3, hilbert3, SfcDecomposition};
//! use gadget_ng_core::{Vec3, SfcKind};
//!
//! // Clave Morton de una posición normalizada [0,1)³
//! let km = morton3(0.25, 0.5, 0.75);
//! // Clave Hilbert de la misma posición
//! let kh = hilbert3(0.25, 0.5, 0.75);
//!
//! // Partición con Morton (default)
//! let positions = vec![Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.8, 0.8, 0.8)];
//! let decomp_m = SfcDecomposition::build(&positions, 1.0, 2);
//! // Partición con Hilbert
//! let decomp_h = SfcDecomposition::build_with_kind(&positions, 1.0, 2, SfcKind::Hilbert);
//! ```

use gadget_ng_core::{Particle, SfcKind, Vec3};

// ── Morton code (Z-order) 63 bits ─────────────────────────────────────────────

/// Separa los 21 bits inferiores de `x` insertando dos ceros entre cada bit.
///
/// Esto permite intercalar (interleave) tres enteros de 21 bits en 63 bits.
/// Implementación estándar "magic bits" para Z-order 3D.
#[inline]
pub fn spread_bits(mut x: u64) -> u64 {
    x &= 0x001f_ffff; // máscara 21 bits
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8)) & 0x100f00f00f00f00f;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3;
    x = (x | (x << 2)) & 0x1249249249249249;
    x
}

/// Computa el código de Morton 3D de una posición normalizada `(x, y, z) ∈ [0,1)`.
///
/// Cada coordenada se discretiza en 21 bits (2^21 = 2_097_152 niveles) antes
/// de intercalar. El resultado tiene 63 bits de información.
#[inline]
pub fn morton3(x: f64, y: f64, z: f64) -> u64 {
    const SCALE: f64 = (1u64 << 21) as f64; // 2^21
    let ix = (x.clamp(0.0, 1.0 - f64::EPSILON) * SCALE) as u64;
    let iy = (y.clamp(0.0, 1.0 - f64::EPSILON) * SCALE) as u64;
    let iz = (z.clamp(0.0, 1.0 - f64::EPSILON) * SCALE) as u64;
    spread_bits(ix) | (spread_bits(iy) << 1) | (spread_bits(iz) << 2)
}

/// Código Morton de una partícula dada la bounding box del dominio.
#[inline]
pub fn particle_morton(
    pos: Vec3,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
    z_lo: f64,
    z_hi: f64,
) -> u64 {
    let lx = (x_hi - x_lo).max(f64::EPSILON);
    let ly = (y_hi - y_lo).max(f64::EPSILON);
    let lz = (z_hi - z_lo).max(f64::EPSILON);
    morton3(
        (pos.x - x_lo) / lx,
        (pos.y - y_lo) / ly,
        (pos.z - z_lo) / lz,
    )
}

// ── Hilbert curve (Peano-Hilbert 3D) ─────────────────────────────────────────

/// Computa el índice de Hilbert 3D de una posición normalizada `(x, y, z) ∈ [0,1)`.
///
/// Usa 21 bits por eje (igual que Morton), produciendo un índice de 63 bits.
/// Implementación basada en el algoritmo de Skilling (2004):
/// "Programming the Hilbert Curve", AIP Conference Proceedings 707, pp. 381-387.
///
/// La curva de Hilbert tiene mejor localidad espacial que Morton: puntos cercanos
/// en 3D tienden a recibir claves más cercanas, lo que reduce el volumen LET en
/// descomposiciones de dominio N-body.
#[inline]
pub fn hilbert3(x: f64, y: f64, z: f64) -> u64 {
    const BITS: u32 = 21;
    const SCALE: f64 = (1u64 << BITS) as f64;
    let ix = (x.clamp(0.0, 1.0 - f64::EPSILON) * SCALE) as u32;
    let iy = (y.clamp(0.0, 1.0 - f64::EPSILON) * SCALE) as u32;
    let iz = (z.clamp(0.0, 1.0 - f64::EPSILON) * SCALE) as u32;
    coords_to_hilbert(ix, iy, iz, BITS)
}

/// Algoritmo de Skilling (2004): transforma coordenadas enteras en p bits por
/// eje a índice Hilbert de 3p bits.
///
/// Referencia: John Skilling, "Programming the Hilbert Curve",
/// AIP Conf. Proc. 707, 381 (2004). https://doi.org/10.1063/1.1751381
///
/// Implementa `AxesToTranspose` — convierte coordenadas espaciales al índice
/// Hilbert en representación "transpuesta" y lo empaqueta en u64.
///
/// # Verificación para p=1
/// La curva de Hilbert visita las 8 celdas del cubo unidad en el orden:
/// (0,0,0)=0, (0,0,1)=1, (0,1,1)=2, (0,1,0)=3, (1,1,0)=4, (1,1,1)=5,
/// (1,0,1)=6, (1,0,0)=7 — un camino Hamiltoniano válido donde cada paso
/// cambia exactamente una coordenada.
fn coords_to_hilbert(ix: u32, iy: u32, iz: u32, p: u32) -> u64 {
    debug_assert!(p > 0 && p <= 21, "p debe estar en [1,21]");

    let mut x = [ix, iy, iz];
    let n = 3usize;
    let m = 1u32 << (p - 1);

    // ── Paso 1: "Inverse undo excess work" (Skilling 2004, AxesToTranspose) ──
    // Transforma las coordenadas espaciales al espacio de transposición Hilbert.
    // Q va desde m=2^(p-1) descendiendo hasta 2 (mientras Q > 1).
    {
        let mut q = m;
        while q > 1 {
            let p_mask = q - 1;
            // Recorrer ejes en orden inverso (n-1 downto 0)
            let mut i = n;
            while i > 0 {
                i -= 1;
                if (x[i] & q) != 0 {
                    x[0] ^= p_mask; // invert
                } else {
                    // swap
                    let t = (x[0] ^ x[i]) & p_mask;
                    x[0] ^= t;
                    x[i] ^= t;
                }
            }
            q >>= 1;
        }
    }

    // ── Paso 2: Gray encode ──────────────────────────────────────────────────
    // Cada elemento se XOR con el anterior: x[i] ^= x[i-1].
    // (No es Gray decode; encode y decode son operaciones diferentes.)
    for i in 1..n {
        x[i] ^= x[i - 1];
    }

    // ── Paso 3: Corrección XOR adicional (Skilling 2004) ─────────────────────
    {
        let mut q = m;
        let mut t: u32 = 0;
        while q > 1 {
            if (x[n - 1] & q) != 0 {
                t ^= q - 1;
            }
            q >>= 1;
        }
        for xi in x.iter_mut() {
            *xi ^= t;
        }
    }

    // ── Paso 4: Empaquetar en u64 (representación transpuesta) ───────────────
    // X[j][k] codifica el bit k*n+j del índice Hilbert (k=0 → bits más significativos).
    // Empaquetado: para k ascendente (0 → p-1), luego j ascendente (0 → n-1):
    //   h = X[0][0], X[1][0], X[2][0], X[0][1], X[1][1], X[2][1], ..., X[2][p-1]
    // MSB de h = X[0][0].
    let mut h = 0u64;
    for k in 0..p {
        for xj in x.iter().take(n) {
            h = (h << 1) | ((*xj >> k) & 1) as u64;
        }
    }
    h
}

/// Código Hilbert de una partícula dada la bounding box del dominio.
#[inline]
pub fn particle_hilbert(
    pos: Vec3,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
    z_lo: f64,
    z_hi: f64,
) -> u64 {
    let lx = (x_hi - x_lo).max(f64::EPSILON);
    let ly = (y_hi - y_lo).max(f64::EPSILON);
    let lz = (z_hi - z_lo).max(f64::EPSILON);
    hilbert3(
        (pos.x - x_lo) / lx,
        (pos.y - y_lo) / ly,
        (pos.z - z_lo) / lz,
    )
}

// ── SfcDecomposition ──────────────────────────────────────────────────────────

/// Descomposición de dominio basada en curva SFC (Morton Z-order o Hilbert 3D).
///
/// El dominio se parte en `n_ranks` segmentos de igual número de partículas
/// a lo largo de la curva. Para el balanceo dinámico, `build` recalcula los
/// límites en cada paso.
#[derive(Debug, Clone)]
pub struct SfcDecomposition {
    /// Bounding box del dominio.
    pub x_lo: f64,
    pub x_hi: f64,
    pub y_lo: f64,
    pub y_hi: f64,
    pub z_lo: f64,
    pub z_hi: f64,
    /// Puntos de corte (cutpoints.len() == n_ranks - 1).
    /// `rank r` posee partículas con clave SFC en `[cutpoints[r-1], cutpoints[r])`.
    /// `rank 0`  posee claves `[0, cutpoints[0])`.
    /// `rank n-1` posee claves `[cutpoints[n-2], u64::MAX]`.
    cutpoints: Vec<u64>,
    /// Número de rangos.
    n_ranks: i32,
    /// Curva SFC usada para generar las claves.
    pub kind: SfcKind,
}

impl SfcDecomposition {
    /// Construye la descomposición SFC a partir de un conjunto de posiciones.
    ///
    /// Usa Morton Z-order (default retrocompatible). Para Hilbert usar
    /// [`build_with_kind`] o [`build_with_bbox_and_kind`].
    ///
    /// La bounding box se calcula directamente desde `positions`.
    ///
    /// **Nota MPI:** en modo multirank, usa [`build_with_bbox`] pasando la bounding
    /// box global (obtenida vía `allreduce_min/max` o [`global_bbox`]) para que todos
    /// los rangos produzcan exactamente los mismos cutpoints.
    pub fn build(positions: &[Vec3], box_size: f64, n_ranks: i32) -> Self {
        Self::build_with_kind(positions, box_size, n_ranks, SfcKind::Morton)
    }

    /// Construye la descomposición SFC con una curva específica.
    pub fn build_with_kind(positions: &[Vec3], box_size: f64, n_ranks: i32, kind: SfcKind) -> Self {
        let n_ranks = n_ranks.max(1);
        if positions.is_empty() {
            return Self {
                x_lo: 0.0,
                x_hi: box_size,
                y_lo: 0.0,
                y_hi: box_size,
                z_lo: 0.0,
                z_hi: box_size,
                cutpoints: vec![u64::MAX / n_ranks as u64; (n_ranks - 1) as usize],
                n_ranks,
                kind,
            };
        }
        let x_lo = positions.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let x_hi = positions
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let y_lo = positions.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let y_hi = positions
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);
        let z_lo = positions.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
        let z_hi = positions
            .iter()
            .map(|p| p.z)
            .fold(f64::NEG_INFINITY, f64::max);
        Self::build_with_bbox_and_kind(positions, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, n_ranks, kind)
    }

    /// Construye la descomposición SFC con bounding box explícita (correcta en modo MPI).
    ///
    /// Usa Morton Z-order (retrocompatible con Fases 8-12).
    /// Para Hilbert usar [`build_with_bbox_and_kind`].
    ///
    /// En modo MPI, todos los rangos deben llamar a esta función con la **bbox global**
    /// (obtenida por `allreduce`), garantizando cutpoints idénticos en todos los rangos.
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use gadget_ng_parallel::sfc::{SfcDecomposition, global_bbox};
    /// # use gadget_ng_parallel::SerialRuntime;
    /// # use gadget_ng_core::{Particle, Vec3};
    /// let rt = SerialRuntime;
    /// let particles = vec![
    ///     Particle::new(0, 1.0, Vec3::new(0.1, 0.2, 0.3), Vec3::zero()),
    ///     Particle::new(1, 1.0, Vec3::new(0.8, 0.7, 0.6), Vec3::zero()),
    /// ];
    /// let (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) = global_bbox(&rt, &particles);
    /// let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    /// let decomp = SfcDecomposition::build_with_bbox(
    ///     &positions, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, 2,
    /// );
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_bbox(
        positions: &[Vec3],
        x_lo: f64,
        x_hi: f64,
        y_lo: f64,
        y_hi: f64,
        z_lo: f64,
        z_hi: f64,
        n_ranks: i32,
    ) -> Self {
        Self::build_with_bbox_and_kind(
            positions,
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            z_lo,
            z_hi,
            n_ranks,
            SfcKind::Morton,
        )
    }

    /// Construye la descomposición SFC con bounding box explícita y curva configurable.
    ///
    /// Es la función base que usan todas las demás variantes.
    /// `kind` selecciona entre Morton Z-order o Peano-Hilbert 3D.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_bbox_and_kind(
        positions: &[Vec3],
        x_lo: f64,
        x_hi: f64,
        y_lo: f64,
        y_hi: f64,
        z_lo: f64,
        z_hi: f64,
        n_ranks: i32,
        kind: SfcKind,
    ) -> Self {
        let n_ranks = n_ranks.max(1);
        if positions.is_empty() {
            return Self {
                x_lo,
                x_hi,
                y_lo,
                y_hi,
                z_lo,
                z_hi,
                cutpoints: vec![u64::MAX / n_ranks as u64; (n_ranks - 1) as usize],
                n_ranks,
                kind,
            };
        }

        let key_fn: Box<dyn Fn(Vec3) -> u64> = match kind {
            SfcKind::Morton => {
                Box::new(move |p: Vec3| particle_morton(p, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi))
            }
            SfcKind::Hilbert => {
                Box::new(move |p: Vec3| particle_hilbert(p, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi))
            }
        };

        let mut keys: Vec<u64> = positions.iter().map(|p| key_fn(*p)).collect();
        keys.sort_unstable();

        let n = keys.len();
        let mut cutpoints = Vec::with_capacity((n_ranks - 1) as usize);
        for r in 1..n_ranks {
            let idx = (r as usize * n) / n_ranks as usize;
            let idx = idx.min(n - 1);
            cutpoints.push(keys[idx]);
        }

        Self {
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            z_lo,
            z_hi,
            cutpoints,
            n_ranks,
            kind,
        }
    }

    /// Construye la descomposición SFC con **balanceo por coste ponderado**.
    ///
    /// En lugar de dividir el dominio en segmentos de igual número de partículas
    /// (cuantiles de conteo), divide en segmentos de **igual suma de pesos** acumulados
    /// a lo largo de la curva SFC (cuantiles de prefix-sum).
    ///
    /// ## Parámetros
    /// - `positions` — posiciones locales (todas las partículas del rank local).
    /// - `weights` — coste por partícula (p. ej. `opened_nodes` del walk BH).
    ///   Debe tener la misma longitud que `positions`. Valores ≤ 0 se tratan como 1.
    /// - Los demás parámetros tienen el mismo significado que en [`build_with_bbox_and_kind`].
    ///
    /// ## Invariante MPI
    /// Todos los ranks deben ver exactamente la **misma** lista ordenada de
    /// `(key, weight)` globales. El caller es responsable de pasar los datos globales
    /// (ya reunidos con allgather o equivalente) en `positions` y `weights`.
    #[allow(clippy::too_many_arguments)]
    pub fn build_weighted(
        positions: &[Vec3],
        weights: &[f64],
        x_lo: f64,
        x_hi: f64,
        y_lo: f64,
        y_hi: f64,
        z_lo: f64,
        z_hi: f64,
        n_ranks: i32,
        kind: SfcKind,
    ) -> Self {
        assert_eq!(
            positions.len(),
            weights.len(),
            "build_weighted: positions.len() != weights.len()"
        );
        let n_ranks = n_ranks.max(1);
        let n = positions.len();
        if n == 0 {
            return Self {
                x_lo,
                x_hi,
                y_lo,
                y_hi,
                z_lo,
                z_hi,
                cutpoints: vec![u64::MAX / n_ranks as u64; (n_ranks - 1) as usize],
                n_ranks,
                kind,
            };
        }

        let key_fn: Box<dyn Fn(Vec3) -> u64> = match kind {
            SfcKind::Morton => {
                Box::new(move |p: Vec3| particle_morton(p, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi))
            }
            SfcKind::Hilbert => {
                Box::new(move |p: Vec3| particle_hilbert(p, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi))
            }
        };

        // Generar pares (clave, peso) y ordenar por clave SFC.
        let mut kw: Vec<(u64, f64)> = positions
            .iter()
            .zip(weights.iter())
            .map(|(&p, &w)| (key_fn(p), w.max(1.0)))
            .collect();
        kw.sort_unstable_by_key(|&(k, _)| k);

        let total_weight: f64 = kw.iter().map(|&(_, w)| w).sum();
        let target_per_rank = total_weight / n_ranks as f64;

        // Calcular cutpoints por prefix-sum de pesos.
        let mut cutpoints = Vec::with_capacity((n_ranks - 1) as usize);
        let mut acc = 0.0_f64;
        let mut rank_idx = 1_i32;
        for i in 0..n {
            acc += kw[i].1;
            if acc >= target_per_rank * rank_idx as f64 && rank_idx < n_ranks {
                cutpoints.push(kw[i.min(n - 1)].0);
                rank_idx += 1;
                if rank_idx >= n_ranks {
                    break;
                }
            }
        }
        // Rellenar cutpoints faltantes si no se alcanzaron todos los umbrales.
        while cutpoints.len() < (n_ranks - 1) as usize {
            cutpoints.push(kw[n - 1].0);
        }

        Self {
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            z_lo,
            z_hi,
            cutpoints,
            n_ranks,
            kind,
        }
    }

    /// Devuelve el rango propietario de una clave SFC dada.
    #[inline]
    pub fn rank_for(&self, key: u64) -> i32 {
        match self.cutpoints.binary_search(&key) {
            Ok(pos) => (pos + 1).min(self.n_ranks as usize - 1) as i32,
            Err(pos) => pos.min(self.n_ranks as usize - 1) as i32,
        }
    }

    /// Devuelve el rango propietario de una posición.
    /// Despacha a Morton o Hilbert según `self.kind`.
    #[inline]
    pub fn rank_for_pos(&self, pos: Vec3) -> i32 {
        let key = match self.kind {
            SfcKind::Morton => particle_morton(
                pos, self.x_lo, self.x_hi, self.y_lo, self.y_hi, self.z_lo, self.z_hi,
            ),
            SfcKind::Hilbert => particle_hilbert(
                pos, self.x_lo, self.x_hi, self.y_lo, self.y_hi, self.z_lo, self.z_hi,
            ),
        };
        self.rank_for(key)
    }

    /// Ancho de halo en unidades de la bounding box (usado para intercambio de halos).
    ///
    /// Para SFC el halo no es un simple slab, sino la región cercana al borde
    /// de cada segmento. Aquí usamos una estimación conservadora: `halo_factor`
    /// veces el lado promedio del segmento de cada rango.
    pub fn halo_width(&self, halo_factor: f64) -> f64 {
        let lx = (self.x_hi - self.x_lo).max(f64::EPSILON);
        // Estimar el ancho de un segmento promedio como lx / n_ranks.
        halo_factor * lx / self.n_ranks as f64
    }

    /// Número de rangos.
    pub fn n_ranks(&self) -> i32 {
        self.n_ranks
    }
}

// ── Utilidades de migración ───────────────────────────────────────────────────

/// Separa `local` en dos vectores: partículas que permanecen y partículas que
/// deben enviarse a otros rangos, dado el rango actual `my_rank`.
///
/// Devuelve `(stays, leaves)` donde `leaves[r]` contiene las partículas
/// que van al rango `r` (con `r != my_rank`).
pub fn partition_local(
    local: &[Particle],
    decomp: &SfcDecomposition,
    my_rank: i32,
) -> (Vec<Particle>, Vec<(i32, Vec<Particle>)>) {
    let n_ranks = decomp.n_ranks();
    let mut buckets: Vec<Vec<Particle>> = (0..n_ranks as usize).map(|_| Vec::new()).collect();
    for p in local {
        let r = decomp.rank_for_pos(p.position);
        buckets[r as usize].push(p.clone());
    }
    let stays = std::mem::take(&mut buckets[my_rank as usize]);
    let leaves: Vec<(i32, Vec<Particle>)> = buckets
        .into_iter()
        .enumerate()
        .filter(|(r, v)| *r != my_rank as usize && !v.is_empty())
        .map(|(r, v)| (r as i32, v))
        .collect();
    (stays, leaves)
}

// ── Bounding box global usando allreduce ──────────────────────────────────────

/// Calcula la bounding box global de un conjunto de posiciones locales usando
/// los métodos `allreduce_min/max_f64` del runtime.
pub fn global_bbox<R: crate::ParallelRuntime + ?Sized>(
    rt: &R,
    local: &[Particle],
) -> (f64, f64, f64, f64, f64, f64) {
    let (xl, xh, yl, yh, zl, zh) = if local.is_empty() {
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        )
    } else {
        (
            local
                .iter()
                .map(|p| p.position.x)
                .fold(f64::INFINITY, f64::min),
            local
                .iter()
                .map(|p| p.position.x)
                .fold(f64::NEG_INFINITY, f64::max),
            local
                .iter()
                .map(|p| p.position.y)
                .fold(f64::INFINITY, f64::min),
            local
                .iter()
                .map(|p| p.position.y)
                .fold(f64::NEG_INFINITY, f64::max),
            local
                .iter()
                .map(|p| p.position.z)
                .fold(f64::INFINITY, f64::min),
            local
                .iter()
                .map(|p| p.position.z)
                .fold(f64::NEG_INFINITY, f64::max),
        )
    };
    (
        rt.allreduce_min_f64(xl),
        rt.allreduce_max_f64(xh),
        rt.allreduce_min_f64(yl),
        rt.allreduce_max_f64(yh),
        rt.allreduce_min_f64(zl),
        rt.allreduce_max_f64(zh),
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tests Morton ──────────────────────────────────────────────────────────

    #[test]
    fn morton_zero_maps_to_zero() {
        assert_eq!(morton3(0.0, 0.0, 0.0), 0);
    }

    #[test]
    fn morton_order_preserves_locality() {
        // Puntos en el mismo octante 1/8 del cubo deben tener claves menores
        // que puntos en el octante opuesto.
        let a = morton3(0.1, 0.1, 0.1);
        let b = morton3(0.15, 0.12, 0.11); // cerca de a
        let c = morton3(0.9, 0.9, 0.9); // lejos de a
                                        // Ambos a y b están en el octante inferior; c está en el superior.
                                        // Sus códigos morton deben estar en rangos distintos.
        assert!(a < c, "morton(0.1,0.1,0.1) debe ser < morton(0.9,0.9,0.9)");
        assert!(
            b < c,
            "morton(0.15,0.12,0.11) debe ser < morton(0.9,0.9,0.9)"
        );
    }

    #[test]
    fn spread_bits_and_back() {
        // spread_bits(x) intercala x correctamente: todos los bits pares
        for x in [0u64, 1, 42, 1048575, (1 << 21) - 1] {
            let s = spread_bits(x);
            // Verificar que ningún bit impar está activo (posiciones 1, 3, 5, ...).
            let even_mask: u64 = 0x1249249249249249; // bits en posiciones 0,3,6,...
            assert_eq!(
                s & !even_mask,
                0,
                "spread_bits({x}) tiene bits en posición incorrecta"
            );
        }
    }

    #[test]
    fn sfc_decomp_all_particles_assigned() {
        let mut positions = Vec::new();
        for i in 0..100 {
            positions.push(Vec3::new((i % 10) as f64 * 0.1, (i / 10) as f64 * 0.1, 0.5));
        }
        let decomp = SfcDecomposition::build(&positions, 1.0, 4);
        for pos in &positions {
            let r = decomp.rank_for_pos(*pos);
            assert!(r >= 0 && r < 4, "rango {r} fuera de rango [0,4)");
        }
    }

    #[test]
    fn sfc_decomp_roughly_balanced() {
        // 1000 partículas en lattice cúbico → ~250 por rango con 4 rangos.
        let mut positions = Vec::new();
        for ix in 0..10usize {
            for iy in 0..10 {
                for iz in 0..10 {
                    positions.push(Vec3::new(
                        ix as f64 / 10.0,
                        iy as f64 / 10.0,
                        iz as f64 / 10.0,
                    ));
                }
            }
        }
        let decomp = SfcDecomposition::build(&positions, 1.0, 4);
        let mut counts = [0usize; 4];
        for pos in &positions {
            counts[decomp.rank_for_pos(*pos) as usize] += 1;
        }
        let total: usize = counts.iter().sum();
        assert_eq!(total, 1000);
        // Cada rango debe tener entre 150 y 350 partículas (±40%).
        for (r, &c) in counts.iter().enumerate() {
            assert!(
                c >= 150 && c <= 350,
                "rango {r} tiene {c} partículas (desequilibrado)"
            );
        }
    }

    #[test]
    fn sfc_decomp_single_rank() {
        let positions = vec![Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.1, 0.2, 0.3)];
        let decomp = SfcDecomposition::build(&positions, 1.0, 1);
        for pos in &positions {
            assert_eq!(decomp.rank_for_pos(*pos), 0);
        }
    }

    // ── Tests Hilbert ─────────────────────────────────────────────────────────

    #[test]
    fn hilbert_zero_maps_to_zero() {
        assert_eq!(hilbert3(0.0, 0.0, 0.0), 0);
    }

    #[test]
    fn hilbert_keys_in_valid_range() {
        // Todas las claves deben estar en [0, 2^63 - 1] (63 bits = 3 ejes × 21 bits).
        let max_valid = (1u64 << 63) - 1;
        let test_points = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.9999, 0.9999, 0.9999),
            (0.1, 0.9, 0.5),
            (0.25, 0.75, 0.25),
        ];
        for (x, y, z) in test_points {
            let k = hilbert3(x, y, z);
            assert!(k <= max_valid, "hilbert3({x},{y},{z}) = {k} excede 63 bits");
        }
    }

    #[test]
    fn hilbert_preserves_locality_basic() {
        // Puntos en el mismo octante deben tener claves más cercanas entre sí
        // que respecto a puntos en el octante opuesto.
        // La curva de Hilbert garantiza esto por construcción.
        let near_a = hilbert3(0.1, 0.1, 0.1);
        let near_b = hilbert3(0.15, 0.12, 0.11);
        let far_c = hilbert3(0.85, 0.88, 0.90);

        let dist_near = near_a.abs_diff(near_b);
        let dist_far_a = near_a.abs_diff(far_c);
        let dist_far_b = near_b.abs_diff(far_c);

        assert!(
            dist_near < dist_far_a,
            "Hilbert: dist(near_a,near_b)={dist_near} debe ser < dist(near_a,far)={dist_far_a}"
        );
        assert!(
            dist_near < dist_far_b,
            "Hilbert: dist(near_a,near_b)={dist_near} debe ser < dist(near_b,far)={dist_far_b}"
        );
    }

    #[test]
    fn hilbert_unique_keys_for_distinct_points() {
        // Puntos distintos en un grid 4×4×4 deben producir claves distintas.
        let mut keys = Vec::new();
        for ix in 0..4usize {
            for iy in 0..4 {
                for iz in 0..4 {
                    let x = ix as f64 / 4.0 + 0.1;
                    let y = iy as f64 / 4.0 + 0.1;
                    let z = iz as f64 / 4.0 + 0.1;
                    keys.push(hilbert3(x, y, z));
                }
            }
        }
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(
            keys.len(),
            64,
            "64 puntos distintos deben producir 64 claves distintas"
        );
    }

    #[test]
    fn sfc_hilbert_roughly_balanced() {
        // 1000 partículas en lattice cúbico → ~250 por rango con 4 rangos (Hilbert).
        let mut positions = Vec::new();
        for ix in 0..10usize {
            for iy in 0..10 {
                for iz in 0..10 {
                    positions.push(Vec3::new(
                        ix as f64 / 10.0,
                        iy as f64 / 10.0,
                        iz as f64 / 10.0,
                    ));
                }
            }
        }
        let decomp = SfcDecomposition::build_with_kind(&positions, 1.0, 4, SfcKind::Hilbert);
        let mut counts = [0usize; 4];
        for pos in &positions {
            counts[decomp.rank_for_pos(*pos) as usize] += 1;
        }
        let total: usize = counts.iter().sum();
        assert_eq!(total, 1000);
        for (r, &c) in counts.iter().enumerate() {
            assert!(
                c >= 150 && c <= 350,
                "Hilbert rango {r} tiene {c} partículas (desequilibrado)"
            );
        }
    }

    #[test]
    fn hilbert_near_origin_small_key() {
        // Puntos en el primer octante (x,y,z << 0.5) deben tener claves
        // Hilbert menores que puntos en el último octante.
        // Propiedad: la curva de Hilbert comienza en (0,0,0) y los puntos
        // cercanos al origen tienen claves pequeñas.
        let h0 = hilbert3(0.0, 0.0, 0.0);
        assert_eq!(h0, 0, "hilbert3(0,0,0) debe ser 0");

        // Puntos en el cuadrante inferior-izquierdo deben tener claves pequeñas.
        let h_small_1 = hilbert3(0.05, 0.05, 0.05);
        let h_small_2 = hilbert3(0.01, 0.02, 0.03);
        let h_large = hilbert3(0.95, 0.95, 0.95);

        // La diferencia entre puntos cerca del origen debe ser mucho menor
        // que la diferencia de puntos lejanos entre sí.
        let diff_near_origin = h_small_1.abs_diff(h_small_2);
        let diff_far = h_small_1.abs_diff(h_large);

        assert!(
            diff_near_origin < diff_far,
            "Puntos cercanos al origen deben tener claves más parecidas. \
             diff_near={diff_near_origin}, diff_far={diff_far}"
        );
    }

    #[test]
    fn hilbert_different_from_morton() {
        // Hilbert y Morton deben producir ordenamientos distintos para verificar
        // que ambas implementaciones son genuinamente diferentes.
        let test_points = [
            (0.3, 0.7, 0.1),
            (0.8, 0.2, 0.9),
            (0.5, 0.5, 0.5),
            (0.1, 0.9, 0.4),
        ];
        let same_count = test_points
            .iter()
            .filter(|&&(x, y, z)| hilbert3(x, y, z) == morton3(x, y, z))
            .count();
        assert!(
            same_count < test_points.len(),
            "Hilbert y Morton deben producir ordenamientos distintos"
        );
    }
}
