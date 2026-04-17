//! FFT 3D distribuida mediante descomposición de slabs en el eje Z (Fase 20).
//!
//! ## Arquitectura
//!
//! Con slabs en Z (eje más lento del indexado `flat = iz*nm² + iy*nm + ix`):
//! - Rank `r` posee los planos `iz ∈ [r·nz_local, (r+1)·nz_local)`.
//! - Las pasadas X e Y de la FFT son completamente locales.
//! - La pasada Z requiere un **alltoall transpose** para que cada rank tenga
//!   todas las `kz` de un subconjunto de pencils `(ky,kx)`.
//!
//! ## Pipeline
//!
//! ```text
//! ρ_slab[(iz_local, iy, ix)]
//!   → fft_xy_local        (X e Y FFT, sin comm)
//!   → alltoall_transpose_fwd  (slab (kz_local, ky, kx) → pencil (p_local, kz_all))
//!   → fft_z_pencils        (Z FFT sobre nm²/P pencils locales)
//!   → apply_poisson_kernel_pencils   (kernel k-space por pencil)
//!   → ifft_z_pencils
//!   → alltoall_transpose_bwd  (pencil → slab)
//!   → ifft_xy_local
//! ```
//!
//! ## Comunicación
//!
//! Cada alltoall transfiere `nz_local × nk_local × 2` f64 a cada vecino, donde
//! `nz_local = nm/P` y `nk_local = nm²/P`. Total por rank y alltoall: `nm³/P` f64.
//! Esto es **P× menos** que el `allreduce` de Fase 19 (`nm³` f64 por rank).

use gadget_ng_parallel::ParallelRuntime;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

// ── Layout de slabs ───────────────────────────────────────────────────────────

/// Descripción de la descomposición de slabs para un rank dado.
///
/// Rank `r` posee los planos Z globales `[z_lo_idx, z_lo_idx + nz_local)`.
#[derive(Debug, Clone, Copy)]
pub struct SlabLayout {
    /// Número de celdas por lado del grid.
    pub nm: usize,
    /// Planos Z locales (= `nm / n_ranks`).
    pub nz_local: usize,
    /// Índice Z global del primer plano de este rank (`rank * nz_local`).
    pub z_lo_idx: usize,
    pub rank: usize,
    pub n_ranks: usize,
}

impl SlabLayout {
    /// Construye el layout para un rank dado. Requiere `nm % n_ranks == 0`.
    pub fn new(nm: usize, rank: usize, n_ranks: usize) -> Self {
        assert!(
            nm % n_ranks == 0,
            "pm_slab requiere pm_grid_size ({nm}) divisible por n_ranks ({n_ranks})"
        );
        let nz_local = nm / n_ranks;
        SlabLayout {
            nm,
            nz_local,
            z_lo_idx: rank * nz_local,
            rank,
            n_ranks,
        }
    }

    /// Número de pencils (ky,kx) por rank para el solve distribuido.
    pub fn nk_local(&self) -> usize {
        self.nm * self.nm / self.n_ranks
    }

    /// Slab z-bounds en unidades de celdas.
    pub fn z_cell_bounds(&self) -> (usize, usize) {
        (self.z_lo_idx, self.z_lo_idx + self.nz_local)
    }

    /// Tamaño del buffer de slab local: `nz_local * nm * nm`.
    pub fn slab_len(&self) -> usize {
        self.nz_local * self.nm * self.nm
    }

    /// Tamaño del buffer de pencils: `nk_local * nm`.
    pub fn pencil_len(&self) -> usize {
        self.nk_local() * self.nm
    }
}

// ── Pasadas locales de FFT ────────────────────────────────────────────────────

/// Aplica FFT en X e Y al slab local `data[(iz_local, iy, ix)]` in-place.
///
/// El buffer tiene tamaño `nz_local * nm * nm`. Tras la llamada los datos
/// están en k-space para las dimensiones X e Y, con Z todavía en espacio real.
pub fn fft_xy_local(
    data: &mut [Complex<f64>],
    layout: &SlabLayout,
    fft: &Arc<dyn rustfft::Fft<f64>>,
) {
    let nm = layout.nm;
    let nm2 = nm * nm;
    let nz = layout.nz_local;
    assert_eq!(data.len(), nz * nm2);

    // Pasada X: filas contiguas en memoria.
    for iz in 0..nz {
        for iy in 0..nm {
            let start = iz * nm2 + iy * nm;
            fft.process(&mut data[start..start + nm]);
        }
    }
    // Pasada Y: columnas (no contiguas, necesita buffer temporal).
    let mut tmp = vec![Complex::new(0.0, 0.0); nm];
    for iz in 0..nz {
        for ix in 0..nm {
            for iy in 0..nm {
                tmp[iy] = data[iz * nm2 + iy * nm + ix];
            }
            fft.process(&mut tmp);
            for iy in 0..nm {
                data[iz * nm2 + iy * nm + ix] = tmp[iy];
            }
        }
    }
}

/// Aplica IFFT en X e Y al slab local. Sin normalización (se aplica fuera).
pub fn ifft_xy_local(
    data: &mut [Complex<f64>],
    layout: &SlabLayout,
    ifft: &Arc<dyn rustfft::Fft<f64>>,
) {
    let nm = layout.nm;
    let nm2 = nm * nm;
    let nz = layout.nz_local;
    assert_eq!(data.len(), nz * nm2);

    // Pasada Y primero (orden inverso respecto a forward).
    let mut tmp = vec![Complex::new(0.0, 0.0); nm];
    for iz in 0..nz {
        for ix in 0..nm {
            for iy in 0..nm {
                tmp[iy] = data[iz * nm2 + iy * nm + ix];
            }
            ifft.process(&mut tmp);
            for iy in 0..nm {
                data[iz * nm2 + iy * nm + ix] = tmp[iy];
            }
        }
    }
    // Pasada X.
    for iz in 0..nz {
        for iy in 0..nm {
            let start = iz * nm2 + iy * nm;
            ifft.process(&mut data[start..start + nm]);
        }
    }
}

/// Aplica FFT-Z a los `nk_local` pencils locales de longitud `nm`.
///
/// Buffer `pencils[(p_local, kz)]` con `p_local ∈ [0, nk_local)` y `kz ∈ [0, nm)`.
pub fn fft_z_pencils(
    pencils: &mut [Complex<f64>],
    layout: &SlabLayout,
    fft: &Arc<dyn rustfft::Fft<f64>>,
) {
    let nm = layout.nm;
    let nk = layout.nk_local();
    assert_eq!(pencils.len(), nk * nm);
    for p in 0..nk {
        fft.process(&mut pencils[p * nm..(p + 1) * nm]);
    }
}

/// Aplica IFFT-Z a los pencils locales.
pub fn ifft_z_pencils(
    pencils: &mut [Complex<f64>],
    layout: &SlabLayout,
    ifft: &Arc<dyn rustfft::Fft<f64>>,
) {
    let nm = layout.nm;
    let nk = layout.nk_local();
    for p in 0..nk {
        ifft.process(&mut pencils[p * nm..(p + 1) * nm]);
    }
}

// ── Alltoall transposes ───────────────────────────────────────────────────────

/// Desempaqueta `Vec<f64>` a `Vec<Complex<f64>>`.
fn unpack_complex(data: &[f64]) -> Vec<Complex<f64>> {
    assert_eq!(data.len() % 2, 0);
    data.chunks_exact(2)
        .map(|ch| Complex::new(ch[0], ch[1]))
        .collect()
}

/// Transpose forward: slab `(kz_local, ky, kx)` → pencil `(p_local, kz_all)`.
///
/// Cada rank `r` envía a rank `s` el bloque de datos correspondiente a los
/// `nk_local` pencils de `s` para sus propios `nz_local` planos kz.
///
/// Retorna `pencils[(p_local, kz)]` con todos los kz (0..nm).
///
/// Para `n_ranks == 1` (serial): rearregla los datos localmente (sin comm).
pub fn alltoall_transpose_fwd<R: ParallelRuntime + ?Sized>(
    data: &[Complex<f64>],
    layout: &SlabLayout,
    rt: &R,
) -> Vec<Complex<f64>> {
    let nm = layout.nm;
    let nm2 = nm * nm;
    let nz = layout.nz_local;
    let nk = layout.nk_local(); // nm² / n_ranks
    let nr = layout.n_ranks;
    assert_eq!(data.len(), nz * nm2);

    if nr == 1 {
        // P=1: rearregla en memoria: (iz, iy*nm+ix) → (iy*nm+ix, iz)
        // nk_local = nm², nz = nm → pencil[p * nm + kz] = data[kz * nm² + p]
        let mut pencils = vec![Complex::new(0.0, 0.0); nm2 * nm];
        for kz in 0..nm {
            for p in 0..nm2 {
                pencils[p * nm + kz] = data[kz * nm2 + p];
            }
        }
        return pencils;
    }

    // Pack sends: sends[s] contains data for pencils of rank s at our local kz planes.
    // For rank s: pencils [s*nk_local, (s+1)*nk_local) → iy in [s*nk/nm..] and ix.
    let mut sends: Vec<Vec<f64>> = (0..nr).map(|_| Vec::with_capacity(nz * nk * 2)).collect();
    for s in 0..nr {
        let p_lo = s * nk;
        let p_hi = p_lo + nk;
        for iz_local in 0..nz {
            for p_global in p_lo..p_hi {
                let iy = p_global / nm;
                let ix = p_global % nm;
                let c = data[iz_local * nm2 + iy * nm + ix];
                sends[s].push(c.re);
                sends[s].push(c.im);
            }
        }
    }

    let received = rt.alltoallv_f64(&sends);

    // Assemble pencil buffer.
    // received[r_src][iz_local_r_src * nk * 2 + p_local * 2 + 0/1] = complex
    let mut pencils = vec![Complex::new(0.0, 0.0); nk * nm];
    for r_src in 0..nr {
        let data_src = unpack_complex(&received[r_src]);
        // data_src[iz_local_r_src * nk + p_local] = value at (kz_global = r_src*nz + iz_local_r_src, pencil p_local)
        for iz_local_r_src in 0..nz {
            let kz_global = r_src * nz + iz_local_r_src;
            for p_local in 0..nk {
                pencils[p_local * nm + kz_global] =
                    data_src[iz_local_r_src * nk + p_local];
            }
        }
    }
    pencils
}

/// Transpose backward: pencil `(p_local, kz_all)` → slab `(kz_local, ky, kx)`.
///
/// Operación inversa de [`alltoall_transpose_fwd`].
pub fn alltoall_transpose_bwd<R: ParallelRuntime + ?Sized>(
    pencils: &[Complex<f64>],
    layout: &SlabLayout,
    rt: &R,
) -> Vec<Complex<f64>> {
    let nm = layout.nm;
    let nm2 = nm * nm;
    let nz = layout.nz_local;
    let nk = layout.nk_local();
    let nr = layout.n_ranks;
    assert_eq!(pencils.len(), nk * nm);

    if nr == 1 {
        // P=1: rearregla: (p, kz) → (kz, p)
        let mut slab = vec![Complex::new(0.0, 0.0); nm2 * nm];
        for kz in 0..nm {
            for p in 0..nm2 {
                slab[kz * nm2 + p] = pencils[p * nm + kz];
            }
        }
        return slab;
    }

    // Pack: sends[r_dst] = nz values per pencil for r_dst's kz range.
    let mut sends: Vec<Vec<f64>> = (0..nr).map(|_| Vec::with_capacity(nz * nk * 2)).collect();
    for r_dst in 0..nr {
        for iz_local_dst in 0..nz {
            let kz = r_dst * nz + iz_local_dst;
            for p_local in 0..nk {
                let c = pencils[p_local * nm + kz];
                sends[r_dst].push(c.re);
                sends[r_dst].push(c.im);
            }
        }
    }

    let received = rt.alltoallv_f64(&sends);

    // Reassemble slab: received[r_src] contains our kz range for pencils of r_src.
    let mut slab = vec![Complex::new(0.0, 0.0); nz * nm2];
    for r_src in 0..nr {
        let data_src = unpack_complex(&received[r_src]);
        let p_lo = r_src * nk;
        // data_src[iz_local_here * nk + p_local_at_r_src]
        for iz_local in 0..nz {
            for p_local_at_r_src in 0..nk {
                let p_global = p_lo + p_local_at_r_src;
                let iy = p_global / nm;
                let ix = p_global % nm;
                slab[iz_local * nm2 + iy * nm + ix] =
                    data_src[iz_local * nk + p_local_at_r_src];
            }
        }
    }
    slab
}

// ── Kernel de Poisson en pencil-space ────────────────────────────────────────

/// Convierte índice DFT j → número de onda entero.
#[inline]
fn freq_index(j: usize, nm: usize) -> i64 {
    if j <= nm / 2 {
        j as i64
    } else {
        j as i64 - nm as i64
    }
}

/// Aplica el kernel de Poisson periódico a los pencils locales,
/// produciendo tres pencil arrays de fuerza (F̂_x, F̂_y, F̂_z).
///
/// `pencils[(p_local, kz)]` contiene `ρ̂(kz, ky, kx)` para los pencils
/// locales después del forward transpose y Z-FFT.
///
/// Retorna `[fx_pencils, fy_pencils, fz_pencils]` en pencil layout.
pub fn apply_poisson_kernel_pencils(
    rho_pencils: &[Complex<f64>],
    layout: &SlabLayout,
    g: f64,
    box_size: f64,
    r_split: Option<f64>,
) -> [Vec<Complex<f64>>; 3] {
    let nm = layout.nm;
    let nk = layout.nk_local();
    let rank = layout.rank;
    let nz = layout.nz_local;
    assert_eq!(rho_pencils.len(), nk * nm);

    let dk = 2.0 * std::f64::consts::PI / box_size;
    let four_pi_g = 4.0 * std::f64::consts::PI * g;

    let mut fx_p = vec![Complex::new(0.0, 0.0); nk * nm];
    let mut fy_p = vec![Complex::new(0.0, 0.0); nk * nm];
    let mut fz_p = vec![Complex::new(0.0, 0.0); nk * nm];

    for p_local in 0..nk {
        let p_global = rank * nk + p_local;
        let iy = p_global / nm;
        let ix = p_global % nm;
        let kx = dk * freq_index(ix, nm) as f64;
        let ky = dk * freq_index(iy, nm) as f64;

        // For n_ranks > 1, kz iterates over ALL kz (0..nm) after the transpose.
        // For n_ranks == 1, same (nz == nm).
        let kz_count = if layout.n_ranks == 1 { nm } else { nm };
        for kz_idx in 0..kz_count {
            let kz = dk * freq_index(kz_idx, nm) as f64;
            let k2 = kx * kx + ky * ky + kz * kz;
            let flat = p_local * nm + kz_idx;
            if k2 < 1e-30 {
                // DC mode: zero force.
                continue;
            }
            let filter = if let Some(r_s) = r_split {
                (-0.5 * k2 * r_s * r_s).exp()
            } else {
                1.0
            };
            let phi_k = rho_pencils[flat] * (-four_pi_g * filter / k2);
            // F̂_α = -i kα Φ̂  →  F̂.re = kα Φ̂.im,  F̂.im = -kα Φ̂.re
            fx_p[flat] = Complex::new(kx * phi_k.im, -kx * phi_k.re);
            fy_p[flat] = Complex::new(ky * phi_k.im, -ky * phi_k.re);
            fz_p[flat] = Complex::new(kz * phi_k.im, -kz * phi_k.re);
        }
        let _ = nz; // suppress unused warning
    }
    [fx_p, fy_p, fz_p]
}

// ── Pipeline completo de solve distribuido ────────────────────────────────────

/// Resuelve la ecuación de Poisson de forma distribuida usando slab decomposition.
///
/// Recibe la densidad local del slab `density_slab[(iz_local, iy, ix)]` con
/// `iz_local ∈ [0, nz_local)`, la densidad **ya global** (tras el intercambio
/// de halos) en el slab de este rank.
///
/// Retorna `[fx_slab, fy_slab, fz_slab]` en el mismo layout de slab.
///
/// Para `n_ranks == 1`, delega a `fft_poisson::solve_forces` directamente.
pub fn solve_forces_slab<R: ParallelRuntime + ?Sized>(
    density_slab: &[f64],
    layout: &SlabLayout,
    g: f64,
    box_size: f64,
    r_split: Option<f64>,
    rt: &R,
) -> [Vec<f64>; 3] {
    let nm = layout.nm;

    // P=1: delegar al solver serial existente para exactitud bit-a-bit.
    if layout.n_ranks == 1 {
        assert_eq!(density_slab.len(), nm * nm * nm);
        return if let Some(rs) = r_split {
            crate::fft_poisson::solve_forces_filtered(density_slab, g, nm, box_size, rs)
        } else {
            crate::fft_poisson::solve_forces(density_slab, g, nm, box_size)
        };
    }

    let nm2 = nm * nm;
    let nz = layout.nz_local;
    assert_eq!(density_slab.len(), nz * nm2);

    let cell_vol = (box_size / nm as f64).powi(3);
    let rho_scale = 1.0 / cell_vol;

    let mut planner = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(nm);
    let fft_inv = planner.plan_fft_inverse(nm);

    // Convertir densidad a complejo con escala volumétrica.
    let mut rho_c: Vec<Complex<f64>> = density_slab
        .iter()
        .map(|&r| Complex::new(r * rho_scale, 0.0))
        .collect();

    // FFT X e Y local.
    fft_xy_local(&mut rho_c, layout, &fft_fwd);

    // Transpose forward: (kz_local, ky, kx) → pencil (p_local, kz).
    let mut rho_pencils = alltoall_transpose_fwd(&rho_c, layout, rt);
    drop(rho_c);

    // FFT Z de los pencils locales.
    fft_z_pencils(&mut rho_pencils, layout, &fft_fwd);

    // Kernel de Poisson: produce [fx_pencils, fy_pencils, fz_pencils].
    let [mut fx_p, mut fy_p, mut fz_p] =
        apply_poisson_kernel_pencils(&rho_pencils, layout, g, box_size, r_split);
    drop(rho_pencils);

    // IFFT Z de cada componente de fuerza.
    ifft_z_pencils(&mut fx_p, layout, &fft_inv);
    ifft_z_pencils(&mut fy_p, layout, &fft_inv);
    ifft_z_pencils(&mut fz_p, layout, &fft_inv);

    // Transpose backward para cada componente → slab layout.
    let fx_slab_c = alltoall_transpose_bwd(&fx_p, layout, rt);
    let fy_slab_c = alltoall_transpose_bwd(&fy_p, layout, rt);
    let fz_slab_c = alltoall_transpose_bwd(&fz_p, layout, rt);
    drop(fx_p);
    drop(fy_p);
    drop(fz_p);

    // IFFT X e Y de cada componente.
    let mut fx_c = fx_slab_c;
    let mut fy_c = fy_slab_c;
    let mut fz_c = fz_slab_c;
    ifft_xy_local(&mut fx_c, layout, &fft_inv);
    ifft_xy_local(&mut fy_c, layout, &fft_inv);
    ifft_xy_local(&mut fz_c, layout, &fft_inv);

    // Normalización 1/nm³ y extracción de parte real.
    let nm3 = nm * nm * nm;
    let norm = 1.0 / nm3 as f64;
    let fx: Vec<f64> = fx_c.iter().map(|c| c.re * norm).collect();
    let fy: Vec<f64> = fy_c.iter().map(|c| c.re * norm).collect();
    let fz: Vec<f64> = fz_c.iter().map(|c| c.re * norm).collect();

    [fx, fy, fz]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_parallel::SerialRuntime;

    fn make_layout_p1(nm: usize) -> SlabLayout {
        SlabLayout::new(nm, 0, 1)
    }

    #[test]
    fn slab_layout_basic() {
        let l = SlabLayout::new(16, 0, 4);
        assert_eq!(l.nz_local, 4);
        assert_eq!(l.z_lo_idx, 0);
        assert_eq!(l.nk_local(), 16 * 16 / 4);

        let l2 = SlabLayout::new(16, 3, 4);
        assert_eq!(l2.z_lo_idx, 12);
    }

    #[test]
    #[should_panic]
    fn slab_layout_bad_divisibility() {
        let _ = SlabLayout::new(15, 0, 4); // 15 % 4 != 0
    }

    #[test]
    fn fft_xy_roundtrip() {
        let nm = 4usize;
        let layout = make_layout_p1(nm);
        let n = nm * nm * nm;
        let original: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut data = original.clone();

        let mut planner = FftPlanner::new();
        let fft_fwd = Arc::new(planner.plan_fft_forward(nm));
        let fft_inv = Arc::new(planner.plan_fft_inverse(nm));

        fft_xy_local(&mut data, &layout, &fft_fwd);
        ifft_xy_local(&mut data, &layout, &fft_inv);

        let norm = 1.0 / (nm * nm) as f64;
        for (i, (d, o)) in data.iter().zip(original.iter()).enumerate() {
            let err = (d.re * norm - o.re).abs();
            assert!(err < 1e-10, "fft_xy roundtrip error en {i}: {err}");
        }
    }

    #[test]
    fn alltoall_transpose_roundtrip_p1() {
        let nm = 8usize;
        let layout = make_layout_p1(nm);
        let rt = SerialRuntime;
        let n = nm * nm * nm;
        let data: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(i as f64, -(i as f64)))
            .collect();

        let pencils = alltoall_transpose_fwd(&data, &layout, &rt);
        let recovered = alltoall_transpose_bwd(&pencils, &layout, &rt);

        assert_eq!(recovered.len(), data.len());
        for (i, (a, b)) in recovered.iter().zip(data.iter()).enumerate() {
            assert!(
                (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
                "transpose roundtrip error en {i}: got ({},{}) expected ({},{})",
                a.re, a.im, b.re, b.im
            );
        }
    }

    #[test]
    fn solve_forces_slab_p1_matches_serial() {
        use crate::fft_poisson;
        let nm = 8usize;
        let layout = make_layout_p1(nm);
        let rt = SerialRuntime;
        let box_size = 1.0_f64;
        let g = 1.0_f64;

        // Densidad sinusoidal en x.
        let nm2 = nm * nm;
        let mut density = vec![0.0_f64; nm * nm2];
        for iz in 0..nm {
            for iy in 0..nm {
                for ix in 0..nm {
                    let x = ix as f64 / nm as f64;
                    density[iz * nm2 + iy * nm + ix] =
                        1.0 + (2.0 * std::f64::consts::PI * x).cos();
                }
            }
        }

        let [fx_s, fy_s, fz_s] = fft_poisson::solve_forces(&density, g, nm, box_size);
        let [fx_d, fy_d, fz_d] = solve_forces_slab(&density, &layout, g, box_size, None, &rt);

        assert_eq!(fx_s.len(), fx_d.len());
        for i in 0..fx_s.len() {
            assert!(
                (fx_s[i] - fx_d[i]).abs() < 1e-10,
                "fx slab != serial en {i}: {:.6e} vs {:.6e}",
                fx_d[i], fx_s[i]
            );
            assert!(
                (fy_s[i] - fy_d[i]).abs() < 1e-10,
                "fy slab != serial en {i}"
            );
            assert!(
                (fz_s[i] - fz_d[i]).abs() < 1e-10,
                "fz slab != serial en {i}"
            );
        }
    }
}
