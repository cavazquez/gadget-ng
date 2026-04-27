//! FFT 3D distribuida mediante descomposición pencil 2D (Brecha GADGET-4: P ≤ nm²).
#![allow(clippy::needless_range_loop)]
//!
//! ## Motivación
//!
//! La FFT slab 1D distribuye planos Z entre P ranks, imponiendo la restricción
//! P ≤ nm (P procesos × 1 plano mínimo = nm planos totales).
//!
//! La descomposición pencil 2D usa una malla de procesos Py × Pz = P donde
//! cada rank (ry, rz) posee una submatriz 2D del grid: ny_local planos Y ×
//! nz_local planos Z, con el eje X completo. Esto permite P ≤ nm² (e.g.
//! Py = Pz = √P con Py, Pz ≤ nm).
//!
//! ## Layout inicial (input)
//!
//! El rank (ry, rz) posee `density_2d[ny_local][nz_local][nm]` donde:
//! - `ny_local = nm / py`
//! - `nz_local = nm / pz`
//! - El eje X tiene extensión completa nm.
//!
//! ## Pipeline con dos alltoalls
//!
//! ```text
//! density_2d[ny_local, nz_local, nm]
//!   → FFT-X local
//!   → alltoall_z_fwd  (Y-group: Pz ranks con mismo ry)
//!       [ny_local, nm_kz, nkx] donde nkx = nm/Pz
//!   → FFT-Z local
//!   → alltoall_y_fwd  (Z-group: Py ranks con mismo rz)
//!       [nm, kz_local, nkx] donde kz_local = nm/Py
//!   → FFT-Y local
//!   → apply_poisson_kernel
//!   → IFFT-Y
//!   → alltoall_y_bwd
//!   → IFFT-Z
//!   → alltoall_z_bwd
//!   → IFFT-X
//! ```
//!
//! ## Comunicación
//!
//! Primer alltoall (Z-group, Pz ranks): transfiere `ny_local × nz_local × nkx × 2` f64
//! por rank por vecino.
//! Segundo alltoall (Y-group, Py ranks): transfiere `nm × nkz_local × nkx × 2` f64
//! por rank por vecino.
//! Total por rank: 2 × nm³/P f64 (idéntico a un solo alltoall de slab 1D).

use gadget_ng_parallel::ParallelRuntime;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;

// ── Layout pencil 2D ──────────────────────────────────────────────────────────

/// Descripción de la descomposición pencil 2D para un rank dado.
///
/// Malla de procesos Py × Pz = P total. Rank global = ry * pz + rz.
#[derive(Debug, Clone, Copy)]
pub struct PencilLayout2D {
    /// Número de celdas por lado del grid cúbico.
    pub nm: usize,
    /// Filas de la malla de procesos (dimensión Y).
    pub py: usize,
    /// Columnas de la malla de procesos (dimensión Z).
    pub pz: usize,
    /// Índice de fila de este rank: `rank / pz`.
    pub rank_y: usize,
    /// Índice de columna de este rank: `rank % pz`.
    pub rank_z: usize,
    /// Planos Y locales: `nm / py`.
    pub ny_local: usize,
    /// Planos Z locales: `nm / pz`.
    pub nz_local: usize,
    pub rank: usize,
    pub n_ranks: usize,
}

impl PencilLayout2D {
    /// Construye el layout para un rank dado en la malla Py × Pz.
    ///
    /// Requisitos: `nm % py == 0` y `nm % pz == 0` y `py * pz == n_ranks`.
    pub fn new(nm: usize, rank: usize, py: usize, pz: usize) -> Self {
        let n_ranks = py * pz;
        assert!(
            nm.is_multiple_of(py),
            "pencil_2d: nm ({nm}) no es divisible por py ({py})"
        );
        assert!(
            nm.is_multiple_of(pz),
            "pencil_2d: nm ({nm}) no es divisible por pz ({pz})"
        );
        assert!(
            rank < n_ranks,
            "pencil_2d: rank ({rank}) ≥ n_ranks ({n_ranks})"
        );
        let rank_y = rank / pz;
        let rank_z = rank % pz;
        Self {
            nm,
            py,
            pz,
            rank_y,
            rank_z,
            ny_local: nm / py,
            nz_local: nm / pz,
            rank,
            n_ranks,
        }
    }

    /// Factoriza P en (Py, Pz) eligiendo Pz lo más cercano posible a √P con Pz ≤ nm.
    ///
    /// Invariante: py * pz == n_ranks, nm % py == 0, nm % pz == 0.
    /// Si no hay factorización válida, devuelve (1, n_ranks) (equivalente a slab).
    pub fn factorize(nm: usize, n_ranks: usize) -> (usize, usize) {
        if n_ranks == 1 || n_ranks <= nm {
            // Slab 1D: Py=1, Pz=P
            return (1, n_ranks);
        }
        // Buscar Pz = mayor divisor de n_ranks con Pz ≤ nm y nm % Pz == 0.
        let mut best_pz = 1usize;
        for pz in (1..=nm.min(n_ranks)).rev() {
            if n_ranks.is_multiple_of(pz) && nm.is_multiple_of(pz) {
                let py = n_ranks / pz;
                if nm.is_multiple_of(py) {
                    best_pz = pz;
                    break;
                }
            }
        }
        (n_ranks / best_pz, best_pz)
    }

    /// Tamaño del buffer 2D slab: `ny_local * nz_local * nm`.
    pub fn slab2d_len(&self) -> usize {
        self.ny_local * self.nz_local * self.nm
    }

    /// Número de kx locales por rank: `nm / pz`.
    pub fn nkx_local(&self) -> usize {
        self.nm / self.pz
    }

    /// Número de kz locales por rank (tras el alltoall Y): `nm / py`.
    pub fn nkz_local(&self) -> usize {
        self.nm / self.py
    }

    /// Rango global del rank (ry_dest, rank_z) en la misma columna Z.
    /// Usado para alltoall dentro del Y-group (filas = misma ry).
    #[inline]
    pub fn y_group_rank(&self, rz_dest: usize) -> usize {
        self.rank_y * self.pz + rz_dest
    }

    /// Rango global del rank (ry_dest, rank_z) en la misma columna.
    /// Usado para alltoall dentro del Z-group (columnas = misma rz).
    #[inline]
    pub fn z_group_rank(&self, ry_dest: usize) -> usize {
        ry_dest * self.pz + self.rank_z
    }
}

// ── Pasadas locales de FFT ────────────────────────────────────────────────────

/// FFT-X en el buffer 2D slab `data[ny_local][nz_local][nm]` in-place.
fn fft_x_local_2d(
    data: &mut [Complex<f64>],
    layout: &PencilLayout2D,
    fft: &Arc<dyn rustfft::Fft<f64>>,
) {
    let nm = layout.nm;
    // Para cada (iy_local, iz_local), aplica FFT sobre las nm celdas en X.
    for row in data.chunks_exact_mut(nm) {
        fft.process(row);
    }
}

/// FFT-Z en el buffer `[ny_local][nm_kz][nkx_local]` in-place.
///
/// Tras el primer alltoall (Z), cada rank tiene ny_local filas Y × nm_kz=nm
/// valores Z × nkx_local valores kx. Para cada (iy_local, kx_local), aplica
/// FFT sobre los nm kz valores contiguos en Z.
fn fft_z_local_2d(
    data: &mut [Complex<f64>],
    layout: &PencilLayout2D,
    fft: &Arc<dyn rustfft::Fft<f64>>,
) {
    let nm = layout.nm;
    let nkx = layout.nkx_local();
    let ny = layout.ny_local;
    // Indexado: data[iy][iz][kx_local]
    // Para FFT-Z, el eje Z es el eje medio (tamaño nm), no contiguo en memoria.
    // Necesitamos extraer pencils en Z.
    let mut pencil = vec![Complex::new(0.0, 0.0); nm];
    for iy in 0..ny {
        for kx in 0..nkx {
            for iz in 0..nm {
                pencil[iz] = data[iy * nm * nkx + iz * nkx + kx];
            }
            fft.process(&mut pencil);
            for iz in 0..nm {
                data[iy * nm * nkx + iz * nkx + kx] = pencil[iz];
            }
        }
    }
}

/// FFT-Y en el buffer `[nm_y][kz_local][nkx_local]` in-place.
///
/// Tras el segundo alltoall (Y), cada rank tiene nm Y-valores × nkz_local
/// valores kz × nkx_local valores kx. Para cada (kz_local, kx_local), aplica
/// FFT sobre los nm ky valores contiguos en Y.
fn fft_y_local_2d(
    data: &mut [Complex<f64>],
    layout: &PencilLayout2D,
    fft: &Arc<dyn rustfft::Fft<f64>>,
) {
    let nm = layout.nm;
    let nkz = layout.nkz_local();
    let nkx = layout.nkx_local();
    // Indexado: data[iy][kz_local][kx_local] con iy el eje más lento.
    // Para FFT-Y, el eje Y es el más lento (tamaño nm), no contiguo.
    let mut pencil = vec![Complex::new(0.0, 0.0); nm];
    for kz in 0..nkz {
        for kx in 0..nkx {
            for iy in 0..nm {
                pencil[iy] = data[iy * nkz * nkx + kz * nkx + kx];
            }
            fft.process(&mut pencil);
            for iy in 0..nm {
                data[iy * nkz * nkx + kz * nkx + kx] = pencil[iy];
            }
        }
    }
}

// ── Alltoalls ─────────────────────────────────────────────────────────────────

/// Primer alltoall (Z-direction): redistribuye kx entre ranks del Y-group.
///
/// Antes: `data[ny_local][nz_local][nm_kx]` (todos los kx, pocos kz).
/// Después: `data_out[ny_local][nm_kz][nkx_local]` (pocos kx, todos los kz).
///
/// Comunica solo dentro del Y-group (Pz ranks con mismo rank_y) usando subcomunicador.
fn alltoall_z_fwd<R: ParallelRuntime + ?Sized>(
    data: &[Complex<f64>],
    layout: &PencilLayout2D,
    rt: &R,
) -> Vec<Complex<f64>> {
    let nm = layout.nm;
    let ny = layout.ny_local;
    let nz = layout.nz_local;
    let nkx = layout.nkx_local(); // nm / pz
    let pz = layout.pz;

    // Construir sends indexados por rank dentro del Y-group (0..pz).
    let mut sends: Vec<Vec<f64>> = (0..pz).map(|_| Vec::new()).collect();
    for rz_dest in 0..pz {
        let kx_lo = rz_dest * nkx;
        let kx_hi = kx_lo + nkx;
        let mut buf = Vec::with_capacity(ny * nz * nkx * 2);
        for iy in 0..ny {
            for iz in 0..nz {
                for kx in kx_lo..kx_hi {
                    let c = data[iy * nz * nm + iz * nm + kx];
                    buf.push(c.re);
                    buf.push(c.im);
                }
            }
        }
        sends[rz_dest] = buf;
    }

    // Color del Y-group: todos los ranks con el mismo rank_y comparten este color.
    let color = layout.rank_y as i32;
    let received = rt.alltoallv_f64_subgroup(&sends, color);

    // Ensamblar: received[rz_src] contiene datos de (ry, rz_src).
    // Layout salida: data_out[iy][kz_global][kx_local].
    let mut out = vec![Complex::new(0.0, 0.0); ny * nm * nkx];
    for rz_src in 0..pz {
        let raw = &received[rz_src];
        if raw.is_empty() {
            continue;
        }
        let kz_lo = rz_src * nz;
        let mut idx = 0;
        for iy in 0..ny {
            for iz_local in 0..nz {
                let kz_global = kz_lo + iz_local;
                for kx in 0..nkx {
                    let re = raw[idx];
                    let im = raw[idx + 1];
                    idx += 2;
                    out[iy * nm * nkx + kz_global * nkx + kx] = Complex::new(re, im);
                }
            }
        }
    }
    out
}

/// Segundo alltoall (Y-direction): redistribuye Y entre ranks del Z-group.
///
/// Antes: `data[ny_local][nm_kz][nkx_local]` (todos los kz, pocos Y reales).
/// Después: `data_out[nm_y][nkz_local][nkx_local]` (todos los Y reales, pocos kz).
///
/// Comunica solo dentro del Z-group (Py ranks con mismo rank_z) usando subcomunicador.
fn alltoall_y_fwd<R: ParallelRuntime + ?Sized>(
    data: &[Complex<f64>],
    layout: &PencilLayout2D,
    rt: &R,
) -> Vec<Complex<f64>> {
    let nm = layout.nm;
    let ny = layout.ny_local;
    let nkz_local = layout.nkz_local(); // nm / py
    let nkx = layout.nkx_local();
    let py = layout.py;

    // Construir sends indexados por rank dentro del Z-group (0..py).
    let mut sends: Vec<Vec<f64>> = (0..py).map(|_| Vec::new()).collect();
    for ry_dest in 0..py {
        let kz_lo = ry_dest * nkz_local;
        let kz_hi = kz_lo + nkz_local;
        let mut buf = Vec::with_capacity(ny * nkz_local * nkx * 2);
        for iy in 0..ny {
            for kz in kz_lo..kz_hi {
                for kx in 0..nkx {
                    let c = data[iy * nm * nkx + kz * nkx + kx];
                    buf.push(c.re);
                    buf.push(c.im);
                }
            }
        }
        sends[ry_dest] = buf;
    }

    // Color del Z-group: offset por pz para no colisionar con colores del Y-group.
    let color = layout.pz as i32 + layout.rank_z as i32;
    let received = rt.alltoallv_f64_subgroup(&sends, color);

    // Ensamblar: received[ry_src] contiene datos de (ry_src, rz).
    // Layout salida: data_out[iy_global][kz_local][kx_local].
    let ny_src = nm / py;
    let mut out = vec![Complex::new(0.0, 0.0); nm * nkz_local * nkx];
    for ry_src in 0..py {
        let raw = &received[ry_src];
        if raw.is_empty() {
            continue;
        }
        let iy_lo = ry_src * ny_src;
        let mut idx = 0;
        for iy_local in 0..ny_src {
            let iy_global = iy_lo + iy_local;
            for kz in 0..nkz_local {
                for kx in 0..nkx {
                    let re = raw[idx];
                    let im = raw[idx + 1];
                    idx += 2;
                    out[iy_global * nkz_local * nkx + kz * nkx + kx] = Complex::new(re, im);
                }
            }
        }
    }
    out
}

/// Alltoall Y inverso: de `[nm_y][nkz_local][nkx_local]` → `[ny_local][nm_kz][nkx_local]`.
fn alltoall_y_bwd<R: ParallelRuntime + ?Sized>(
    data: &[Complex<f64>],
    layout: &PencilLayout2D,
    rt: &R,
) -> Vec<Complex<f64>> {
    let nm = layout.nm;
    let ny = layout.ny_local;
    let nkz_local = layout.nkz_local();
    let nkx = layout.nkx_local();
    let py = layout.py;

    // Repartir iy_global a sus ranks propietarios en el Z-group (0..py).
    let mut sends: Vec<Vec<f64>> = (0..py).map(|_| Vec::new()).collect();
    for ry_dest in 0..py {
        let iy_lo = ry_dest * ny;
        let iy_hi = iy_lo + ny;
        let mut buf = Vec::with_capacity(ny * nkz_local * nkx * 2);
        for iy in iy_lo..iy_hi {
            for kz in 0..nkz_local {
                for kx in 0..nkx {
                    let c = data[iy * nkz_local * nkx + kz * nkx + kx];
                    buf.push(c.re);
                    buf.push(c.im);
                }
            }
        }
        sends[ry_dest] = buf;
    }

    let color = layout.pz as i32 + layout.rank_z as i32;
    let received = rt.alltoallv_f64_subgroup(&sends, color);

    // Reensamblar: received[ry_src] contiene los kz ∈ [ry_src*nkz_local, ...).
    let mut out = vec![Complex::new(0.0, 0.0); ny * nm * nkx];
    for ry_src in 0..py {
        let raw = &received[ry_src];
        if raw.is_empty() {
            continue;
        }
        let kz_lo = ry_src * nkz_local;
        let mut idx = 0;
        for iy in 0..ny {
            for kz_local in 0..nkz_local {
                let kz_global = kz_lo + kz_local;
                for kx in 0..nkx {
                    let re = raw[idx];
                    let im = raw[idx + 1];
                    idx += 2;
                    out[iy * nm * nkx + kz_global * nkx + kx] = Complex::new(re, im);
                }
            }
        }
    }
    out
}

/// Alltoall Z inverso: de `[ny_local][nm_kz][nkx_local]` → `[ny_local][nz_local][nm]`.
fn alltoall_z_bwd<R: ParallelRuntime + ?Sized>(
    data: &[Complex<f64>],
    layout: &PencilLayout2D,
    rt: &R,
) -> Vec<Complex<f64>> {
    let nm = layout.nm;
    let ny = layout.ny_local;
    let nz = layout.nz_local;
    let nkx = layout.nkx_local();
    let pz = layout.pz;

    // Repartir kz a sus ranks propietarios en el Y-group (0..pz).
    let mut sends: Vec<Vec<f64>> = (0..pz).map(|_| Vec::new()).collect();
    for rz_dest in 0..pz {
        let kz_lo = rz_dest * nz;
        let kz_hi = kz_lo + nz;
        let mut buf = Vec::with_capacity(ny * nz * nkx * 2);
        for iy in 0..ny {
            for kz in kz_lo..kz_hi {
                for kx in 0..nkx {
                    let c = data[iy * nm * nkx + kz * nkx + kx];
                    buf.push(c.re);
                    buf.push(c.im);
                }
            }
        }
        sends[rz_dest] = buf;
    }

    let color = layout.rank_y as i32;
    let received = rt.alltoallv_f64_subgroup(&sends, color);

    // Reensamblar: received[rz_src] contiene los kx ∈ [rz_src*nkx, ...).
    let mut out = vec![Complex::new(0.0, 0.0); ny * nz * nm];
    for rz_src in 0..pz {
        let raw = &received[rz_src];
        if raw.is_empty() {
            continue;
        }
        let kx_lo = rz_src * nkx;
        let mut idx = 0;
        for iy in 0..ny {
            for iz in 0..nz {
                for kx_local in 0..nkx {
                    let kx_global = kx_lo + kx_local;
                    let re = raw[idx];
                    let im = raw[idx + 1];
                    idx += 2;
                    out[iy * nz * nm + iz * nm + kx_global] = Complex::new(re, im);
                }
            }
        }
    }
    out
}

// ── Kernel de Poisson en k-espacio ────────────────────────────────────────────

/// Aplica el kernel de Poisson (y fuerza) al buffer pencil 2D en k-espacio.
///
/// Cada rank (ry, rz) tiene `data[nm_y][nkz_local][nkx_local]` y conoce:
/// - ky ∈ [0, nm) (dimensión Y completa, eje más lento)
/// - kz ∈ [rank_y * nkz_local, (rank_y + 1) * nkz_local)
/// - kx ∈ [rank_z * nkx_local, (rank_z + 1) * nkx_local)
///
/// Devuelve [fx, fy, fz] arrays en k-espacio (misma distribución).
fn apply_poisson_kernel_pencil2d(
    rho_pencils: &[Complex<f64>],
    layout: &PencilLayout2D,
    g: f64,
    box_size: f64,
    r_split: Option<f64>,
) -> [Vec<Complex<f64>>; 3] {
    let nm = layout.nm;
    let nkz = layout.nkz_local();
    let nkx = layout.nkx_local();
    let kz_lo = layout.rank_y * nkz;
    let kx_lo = layout.rank_z * nkx;
    let dk = std::f64::consts::TAU / box_size;
    let len = nm * nkz * nkx;
    let mut fx = vec![Complex::new(0.0, 0.0); len];
    let mut fy = vec![Complex::new(0.0, 0.0); len];
    let mut fz = vec![Complex::new(0.0, 0.0); len];

    // Prefactor de Poisson: -G / k² (en cada celda k-space).
    let poisson_pre = -g * (box_size * box_size * box_size) / (nm * nm * nm) as f64;

    for iy in 0..nm {
        let ky = if iy <= nm / 2 {
            iy as f64
        } else {
            iy as f64 - nm as f64
        };
        for iz in 0..nkz {
            let kz_global = kz_lo + iz;
            let kz = if kz_global <= nm / 2 {
                kz_global as f64
            } else {
                kz_global as f64 - nm as f64
            };
            for ix in 0..nkx {
                let kx_global = kx_lo + ix;
                let kx = if kx_global <= nm / 2 {
                    kx_global as f64
                } else {
                    kx_global as f64 - nm as f64
                };
                let idx = iy * nkz * nkx + iz * nkx + ix;
                let k2_grid = kx * kx + ky * ky + kz * kz;
                if k2_grid == 0.0 {
                    fx[idx] = Complex::new(0.0, 0.0);
                    fy[idx] = Complex::new(0.0, 0.0);
                    fz[idx] = Complex::new(0.0, 0.0);
                    continue;
                }
                let k2_phys = k2_grid * dk * dk;
                // Kernel Poisson estándar: φ̂ = G ρ̂ / k²
                let phi_hat = rho_pencils[idx] * (poisson_pre / k2_phys);

                // Filtro de largo alcance (TreePM): F(k) = exp(-k² r_split² / 2).
                let filter = if let Some(rs) = r_split {
                    (-k2_phys * rs * rs * 0.5).exp()
                } else {
                    1.0
                };

                // Fuerza: F̂_x = i kx φ̂, etc. (con derivada espectral).
                let i_kx = Complex::new(0.0, kx * dk) * filter;
                let i_ky = Complex::new(0.0, ky * dk) * filter;
                let i_kz = Complex::new(0.0, kz * dk) * filter;
                fx[idx] = phi_hat * i_kx;
                fy[idx] = phi_hat * i_ky;
                fz[idx] = phi_hat * i_kz;
            }
        }
    }
    [fx, fy, fz]
}

// ── Función principal ─────────────────────────────────────────────────────────

/// Calcula las fuerzas PM usando FFT pencil 2D distribuida.
///
/// ## Argumentos
///
/// - `density_2d`: densidad en layout 2D slab `[ny_local][nz_local][nm]` (real, `f64`).
/// - `layout`: descripción del layout pencil 2D para este rank.
/// - `g`: constante gravitacional efectiva.
/// - `box_size`: tamaño del cubo periódico (mismas unidades que las posiciones).
/// - `r_split`: radio de separación TreePM (Gaussiano). `None` → PM completo.
/// - `rt`: runtime paralelo.
///
/// ## Retorno
///
/// `[fx, fy, fz]`: componentes de fuerza en layout 2D slab `[ny_local][nz_local][nm]`.
pub fn solve_forces_pencil2d<R: ParallelRuntime + ?Sized>(
    density_2d: &[f64],
    layout: &PencilLayout2D,
    g: f64,
    box_size: f64,
    r_split: Option<f64>,
    rt: &R,
) -> [Vec<f64>; 3] {
    let nm = layout.nm;
    let ny = layout.ny_local;
    let nz = layout.nz_local;

    // Normalizar densidad a densidad volumétrica.
    let cell_vol = box_size * box_size * box_size / (nm * nm * nm) as f64;
    let len_2d = ny * nz * nm;
    assert_eq!(
        density_2d.len(),
        len_2d,
        "density_2d tiene longitud incorrecta"
    );

    let mut planner = FftPlanner::new();
    let fft_x = planner.plan_fft_forward(nm);
    let fft_z = planner.plan_fft_forward(nm);
    let fft_y = planner.plan_fft_forward(nm);
    let ifft_x: Arc<dyn rustfft::Fft<f64>> = planner.plan_fft_inverse(nm);
    let ifft_z: Arc<dyn rustfft::Fft<f64>> = planner.plan_fft_inverse(nm);
    let ifft_y: Arc<dyn rustfft::Fft<f64>> = planner.plan_fft_inverse(nm);

    // Convertir densidad real a complejo.
    let mut rho: Vec<Complex<f64>> = density_2d
        .iter()
        .map(|&d| Complex::new(d * cell_vol, 0.0))
        .collect();

    // Caso P=1: todas las dimensiones son locales, sin comunicación.
    if layout.n_ranks == 1 {
        return solve_serial_3d(rho, nm, g, box_size, r_split, &mut planner);
    }

    // Paso 1: FFT-X local.
    fft_x_local_2d(&mut rho, layout, &fft_x);

    // Paso 2: Alltoall-Z (Y-group) → [ny_local][nm_kz][nkx_local].
    let mut rho_z = alltoall_z_fwd(&rho, layout, rt);
    drop(rho);

    // Paso 3: FFT-Z local.
    fft_z_local_2d(&mut rho_z, layout, &fft_z);

    // Paso 4: Alltoall-Y (Z-group) → [nm_y][nkz_local][nkx_local].
    let mut rho_y = alltoall_y_fwd(&rho_z, layout, rt);
    drop(rho_z);

    // Paso 5: FFT-Y local.
    fft_y_local_2d(&mut rho_y, layout, &fft_y);

    // Paso 6: Kernel de Poisson → [fx, fy, fz] en k-espacio.
    let [mut fx_k, mut fy_k, mut fz_k] =
        apply_poisson_kernel_pencil2d(&rho_y, layout, g, box_size, r_split);

    // Paso 7: IFFT-Y sobre cada componente.
    for arr in [&mut fx_k, &mut fy_k, &mut fz_k].iter_mut() {
        fft_y_local_2d(arr, layout, &ifft_y);
    }

    // Paso 8: Alltoall-Y inverso → [ny_local][nm_kz][nkx_local].
    let fx_zy = alltoall_y_bwd(&fx_k, layout, rt);
    let fy_zy = alltoall_y_bwd(&fy_k, layout, rt);
    let fz_zy = alltoall_y_bwd(&fz_k, layout, rt);
    drop(fx_k);
    drop(fy_k);
    drop(fz_k);

    // Paso 9: IFFT-Z.
    let mut fx_z = fx_zy;
    let mut fy_z = fy_zy;
    let mut fz_z = fz_zy;
    for arr in [&mut fx_z, &mut fy_z, &mut fz_z].iter_mut() {
        fft_z_local_2d(arr, layout, &ifft_z);
    }

    // Paso 10: Alltoall-Z inverso → [ny_local][nz_local][nm].
    let fx_2d = alltoall_z_bwd(&fx_z, layout, rt);
    let fy_2d = alltoall_z_bwd(&fy_z, layout, rt);
    let fz_2d = alltoall_z_bwd(&fz_z, layout, rt);
    drop(fx_z);
    drop(fy_z);
    drop(fz_z);

    // Paso 11: IFFT-X + normalización (1/nm³ de las 3 IFFTs).
    let norm = 1.0 / (nm * nm * nm) as f64;

    let extract_real =
        |mut arr: Vec<Complex<f64>>, ifft: &Arc<dyn rustfft::Fft<f64>>| -> Vec<f64> {
            // IFFT-X sobre cada fila (iy_local, iz_local).
            for row in arr.chunks_exact_mut(nm) {
                ifft.process(row);
            }
            arr.iter().map(|c| c.re * norm).collect()
        };

    let fx_out = extract_real(fx_2d, &ifft_x);
    let fy_out = extract_real(fy_2d, &ifft_x);
    let fz_out = extract_real(fz_2d, &ifft_x);

    [fx_out, fy_out, fz_out]
}

/// Caso P=1: FFT 3D serial directa (reusa el mismo layout pero sin alltoalls).
fn solve_serial_3d(
    rho: Vec<Complex<f64>>,
    nm: usize,
    g: f64,
    box_size: f64,
    r_split: Option<f64>,
    _planner: &mut FftPlanner<f64>,
) -> [Vec<f64>; 3] {
    // Delega al solver serial de fft_poisson cuando P=1.
    use crate::fft_poisson::{solve_forces, solve_forces_filtered};
    // Extraer densidad real del buffer complejo.
    let density: Vec<f64> = rho.iter().map(|c| c.re).collect();
    if let Some(rs) = r_split {
        solve_forces_filtered(&density, g, nm, box_size, rs)
    } else {
        solve_forces(&density, g, nm, box_size)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_parallel::SerialRuntime;

    /// Con P=1, pencil 2D (Py=1,Pz=1) debe ser equivalente al PM serial.
    #[test]
    fn pencil2d_p1_zero_force_uniform() {
        let rt = SerialRuntime;
        let nm = 8usize;
        let layout = PencilLayout2D::new(nm, 0, 1, 1);
        let density = vec![1.0_f64; nm * nm * nm];
        let [fx, fy, fz] = solve_forces_pencil2d(&density, &layout, 1.0, 1.0, None, &rt);
        let max_f = fx
            .iter()
            .chain(fy.iter())
            .chain(fz.iter())
            .map(|&v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_f < 1e-10,
            "densidad uniforme → fuerza cero; max_f = {max_f:.2e}"
        );
    }

    /// `factorize` devuelve (1, P) para P ≤ nm.
    #[test]
    fn factorize_small_p_is_slab() {
        let (py, pz) = PencilLayout2D::factorize(8, 4);
        assert_eq!(py, 1);
        assert_eq!(pz, 4);
    }

    /// `factorize` para P > nm devuelve un layout válido.
    #[test]
    fn factorize_large_p_valid() {
        let nm = 8;
        let p = 16;
        let (py, pz) = PencilLayout2D::factorize(nm, p);
        assert_eq!(py * pz, p);
        assert!(
            nm % py == 0 && nm % pz == 0,
            "py={py}, pz={pz} no dividen nm={nm}"
        );
    }
}
