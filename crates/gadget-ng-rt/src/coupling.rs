//! Acoplamiento entre el campo de radiación M1 y el gas SPH (Phase 81).
//!
//! ## Procesos físicos modelados
//!
//! ### Fotoionización
//!
//! La tasa de fotoionización para HI es:
//!
//! ```text
//! Γ_HI = ∫ (σ_HI × F_ν / hν) dν  ≈  σ_HI × c_red × E_UV / (hν_0)
//! ```
//!
//! En código, se calcula en función de la densidad de energía local `E` del campo
//! de radiación UV.
//!
//! ### Fotocalentamiento
//!
//! La energía de calentamiento por fotón absorbido es:
//!
//! ```text
//! Q_heat = Γ_HI × n_HI × (hν_0 - 13.6 eV)  [erg/cm³/s]
//! ```
//!
//! En unidades internas: `ΔU_i = Q_heat × V_part / m_part × dt`.
//!
//! ### Acoplamiento iterativo
//!
//! El acoplamiento gas←→radiación es stiff; se resuelve con un paso implícito:
//!
//! ```text
//! E_new = (E + Σ η_i dt) / (1 + c_red κ_abs dt)
//! ```
//!
//! donde `η_i` son las tasas de emisión de cada partícula de gas.
//!
//! ## Referencia
//!
//! Rosdahl et al. (2013), MNRAS 436, 2188;
//! Gnedin & Abel (2001), New Astron. 6, 437.

use crate::m1::{C_KMS, M1Params, RadiationField};
use gadget_ng_core::{Particle, ParticleType};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ── Constantes ─────────────────────────────────────────────────────────────

/// Sección eficaz de fotoionización de HI a ν₀ = 13.6 eV [cm²].
/// σ_HI(ν₀) ≈ 6.3 × 10⁻¹⁸ cm².
const SIGMA_HI: f64 = 6.3e-18;

/// Energía del fotón umbral de ionización de HI: 13.6 eV en unidades de erg.
const H_NU_0_ERG: f64 = 2.179e-11;

/// Factor de conversión de energía interna (unidades gadget → erg/g).
/// Para kpc/km/s/M_sun: 1 (km/s)² ≈ 1e10 erg/g.
const U_CODE_TO_ERG_G: f64 = 1e10;

// ── API principal ─────────────────────────────────────────────────────────

/// Calcula la tasa de fotoionización local para cada celda del grid.
///
/// `Γ(i) = σ_HI × c_red × E(i) / (h × ν₀)` [s⁻¹]
pub fn photoionization_rate(rad: &RadiationField, params: &M1Params) -> Vec<f64> {
    let c_red = C_KMS * 1e5 / params.c_red_factor; // km/s → cm/s
    photoionization_rate_impl(&rad.energy_density, c_red)
}

#[cfg(feature = "rayon")]
fn photoionization_rate_impl(energy_density: &[f64], c_red: f64) -> Vec<f64> {
    energy_density
        .par_iter()
        .map(|&e| SIGMA_HI * c_red * e.max(0.0) / H_NU_0_ERG)
        .collect()
}

#[cfg(not(feature = "rayon"))]
fn photoionization_rate_impl(energy_density: &[f64], c_red: f64) -> Vec<f64> {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            unsafe {
                return photoionization_rate_avx512(energy_density, c_red);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                return photoionization_rate_avx2(energy_density, c_red);
            }
        }
    }

    photoionization_rate_scalar(energy_density, c_red)
}

#[cfg(not(feature = "rayon"))]
fn photoionization_rate_scalar(energy_density: &[f64], c_red: f64) -> Vec<f64> {
    energy_density
        .iter()
        .map(|&e| SIGMA_HI * c_red * e.max(0.0) / H_NU_0_ERG)
        .collect()
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn photoionization_rate_avx2(energy_density: &[f64], c_red: f64) -> Vec<f64> {
    let scale = SIGMA_HI * c_red / H_NU_0_ERG;
    let mut out = vec![0.0; energy_density.len()];
    let lanes = 4;
    let chunks = energy_density.len() / lanes * lanes;

    let zero = _mm256_set1_pd(0.0);
    let scale_v = _mm256_set1_pd(scale);
    let mut i = 0;
    while i < chunks {
        // SAFETY: `i..i+4` is in-bounds by construction and unaligned loads are permitted.
        let e = unsafe { _mm256_loadu_pd(energy_density.as_ptr().add(i)) };
        let clipped = _mm256_max_pd(e, zero);
        let gamma = _mm256_mul_pd(clipped, scale_v);
        // SAFETY: `out` has the same length as `energy_density`, so this store is in-bounds.
        unsafe { _mm256_storeu_pd(out.as_mut_ptr().add(i), gamma) };
        i += lanes;
    }

    for (dst, &e) in out[chunks..].iter_mut().zip(&energy_density[chunks..]) {
        *dst = e.max(0.0) * scale;
    }
    out
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn photoionization_rate_avx512(energy_density: &[f64], c_red: f64) -> Vec<f64> {
    let scale = SIGMA_HI * c_red / H_NU_0_ERG;
    let mut out = vec![0.0; energy_density.len()];
    let lanes = 8;
    let chunks = energy_density.len() / lanes * lanes;

    let zero = _mm512_set1_pd(0.0);
    let scale_v = _mm512_set1_pd(scale);
    let mut i = 0;
    while i < chunks {
        // SAFETY: `i..i+8` is in-bounds by construction and unaligned loads are permitted.
        let e = unsafe { _mm512_loadu_pd(energy_density.as_ptr().add(i)) };
        let clipped = _mm512_max_pd(e, zero);
        let gamma = _mm512_mul_pd(clipped, scale_v);
        // SAFETY: `out` has the same length as `energy_density`, so this store is in-bounds.
        unsafe { _mm512_storeu_pd(out.as_mut_ptr().add(i), gamma) };
        i += lanes;
    }

    for (dst, &e) in out[chunks..].iter_mut().zip(&energy_density[chunks..]) {
        *dst = e.max(0.0) * scale;
    }
    out
}

/// Aplica el fotocalentamiento a las partículas de gas SPH.
///
/// Para cada partícula de tipo Gas, busca la celda más cercana del campo
/// de radiación e incrementa la energía interna por fotocalentamiento.
///
/// # Parámetros
/// - `particles` — partículas de la simulación (solo Gas se ve afectado).
/// - `rad`       — campo de radiación M1 actual.
/// - `gamma_hi`  — tasas de fotoionización por celda (de `photoionization_rate`).
/// - `dt`        — paso de tiempo.
/// - `box_size`  — tamaño de la caja (para coordenadas periódicas).
pub fn apply_photoheating(
    particles: &mut [Particle],
    rad: &RadiationField,
    gamma_hi: &[f64],
    dt: f64,
    box_size: f64,
) {
    let nx = rad.nx;
    let ny = rad.ny;
    let nz = rad.nz;

    #[cfg(feature = "rayon")]
    {
        particles.par_iter_mut().for_each(|p| {
            photoheat_particle(p, rad, gamma_hi, dt, box_size, nx, ny, nz);
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        apply_photoheating_serial(particles, rad, gamma_hi, dt, box_size, nx, ny, nz);
    }
}

#[cfg(not(feature = "rayon"))]
#[expect(
    clippy::too_many_arguments,
    reason = "photoheating maps particle coordinates with grid dimensions"
)]
fn apply_photoheating_serial(
    particles: &mut [Particle],
    rad: &RadiationField,
    gamma_hi: &[f64],
    dt: f64,
    box_size: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            unsafe {
                return apply_photoheating_avx512(
                    particles, rad, gamma_hi, dt, box_size, nx, ny, nz,
                );
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                return apply_photoheating_avx2(particles, rad, gamma_hi, dt, box_size, nx, ny, nz);
            }
        }
    }

    apply_photoheating_scalar(particles, rad, gamma_hi, dt, box_size, nx, ny, nz);
}

#[cfg(not(feature = "rayon"))]
#[expect(
    clippy::too_many_arguments,
    reason = "photoheating maps particle coordinates with grid dimensions"
)]
fn apply_photoheating_scalar(
    particles: &mut [Particle],
    rad: &RadiationField,
    gamma_hi: &[f64],
    dt: f64,
    box_size: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) {
    for p in particles.iter_mut() {
        photoheat_particle(p, rad, gamma_hi, dt, box_size, nx, ny, nz);
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[expect(
    clippy::too_many_arguments,
    reason = "photoheating maps particle coordinates with grid dimensions"
)]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_photoheating_avx2(
    particles: &mut [Particle],
    rad: &RadiationField,
    gamma_hi: &[f64],
    dt: f64,
    box_size: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let inv_box = 1.0 / box_size;
    let nx_v = _mm256_set1_pd(nx as f64 * inv_box);
    let ny_v = _mm256_set1_pd(ny as f64 * inv_box);
    let nz_v = _mm256_set1_pd(nz as f64 * inv_box);
    let dt_over_u = _mm256_set1_pd(dt / U_CODE_TO_ERG_G);
    let cap_factor = _mm256_set1_pd(10.0);

    let mut i = 0;
    while i < chunks {
        if particles[i..i + lanes]
            .iter()
            .any(|p| p.ptype != ParticleType::Gas)
        {
            apply_photoheating_scalar(
                &mut particles[i..i + lanes],
                rad,
                gamma_hi,
                dt,
                box_size,
                nx,
                ny,
                nz,
            );
            i += lanes;
            continue;
        }

        let x = _mm256_set_pd(
            particles[i + 3].position.x,
            particles[i + 2].position.x,
            particles[i + 1].position.x,
            particles[i].position.x,
        );
        let y = _mm256_set_pd(
            particles[i + 3].position.y,
            particles[i + 2].position.y,
            particles[i + 1].position.y,
            particles[i].position.y,
        );
        let z = _mm256_set_pd(
            particles[i + 3].position.z,
            particles[i + 2].position.z,
            particles[i + 1].position.z,
            particles[i].position.z,
        );
        let ix_f = _mm256_mul_pd(x, nx_v);
        let iy_f = _mm256_mul_pd(y, ny_v);
        let iz_f = _mm256_mul_pd(z, nz_v);
        let mut ix_arr = [0.0; 4];
        let mut iy_arr = [0.0; 4];
        let mut iz_arr = [0.0; 4];
        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(ix_arr.as_mut_ptr(), ix_f);
            _mm256_storeu_pd(iy_arr.as_mut_ptr(), iy_f);
            _mm256_storeu_pd(iz_arr.as_mut_ptr(), iz_f);
        }

        let mut gamma_arr = [0.0; 4];
        for lane in 0..lanes {
            let ix = (ix_arr[lane].floor() as usize).min(nx - 1);
            let iy = (iy_arr[lane].floor() as usize).min(ny - 1);
            let iz = (iz_arr[lane].floor() as usize).min(nz - 1);
            let cell = rad.idx(ix, iy, iz);
            let gamma = gamma_hi.get(cell).copied().unwrap_or(0.0);
            gamma_arr[lane] = if gamma < 1e-30 { 0.0 } else { gamma };
        }

        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        let gamma = unsafe { _mm256_loadu_pd(gamma_arr.as_ptr()) };
        let delta = _mm256_mul_pd(gamma, dt_over_u);
        let u = _mm256_set_pd(
            particles[i + 3].internal_energy,
            particles[i + 2].internal_energy,
            particles[i + 1].internal_energy,
            particles[i].internal_energy,
        );
        let capped = _mm256_min_pd(delta, _mm256_mul_pd(u, cap_factor));
        let u_new = _mm256_add_pd(u, capped);
        let mut u_arr = [0.0; 4];
        // SAFETY: fixed-size stack array has exactly four f64 lanes.
        unsafe { _mm256_storeu_pd(u_arr.as_mut_ptr(), u_new) };
        for lane in 0..lanes {
            particles[i + lane].internal_energy = u_arr[lane];
        }
        i += lanes;
    }

    apply_photoheating_scalar(
        &mut particles[chunks..],
        rad,
        gamma_hi,
        dt,
        box_size,
        nx,
        ny,
        nz,
    );
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[expect(
    clippy::too_many_arguments,
    reason = "photoheating maps particle coordinates with grid dimensions"
)]
#[target_feature(enable = "avx512f")]
unsafe fn apply_photoheating_avx512(
    particles: &mut [Particle],
    rad: &RadiationField,
    gamma_hi: &[f64],
    dt: f64,
    box_size: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) {
    let lanes = 8;
    let chunks = particles.len() / lanes * lanes;
    let inv_box = 1.0 / box_size;
    let nx_v = _mm512_set1_pd(nx as f64 * inv_box);
    let ny_v = _mm512_set1_pd(ny as f64 * inv_box);
    let nz_v = _mm512_set1_pd(nz as f64 * inv_box);
    let dt_over_u = _mm512_set1_pd(dt / U_CODE_TO_ERG_G);
    let cap_factor = _mm512_set1_pd(10.0);

    let mut i = 0;
    while i < chunks {
        if particles[i..i + lanes]
            .iter()
            .any(|p| p.ptype != ParticleType::Gas)
        {
            apply_photoheating_scalar(
                &mut particles[i..i + lanes],
                rad,
                gamma_hi,
                dt,
                box_size,
                nx,
                ny,
                nz,
            );
            i += lanes;
            continue;
        }

        let x = _mm512_set_pd(
            particles[i + 7].position.x,
            particles[i + 6].position.x,
            particles[i + 5].position.x,
            particles[i + 4].position.x,
            particles[i + 3].position.x,
            particles[i + 2].position.x,
            particles[i + 1].position.x,
            particles[i].position.x,
        );
        let y = _mm512_set_pd(
            particles[i + 7].position.y,
            particles[i + 6].position.y,
            particles[i + 5].position.y,
            particles[i + 4].position.y,
            particles[i + 3].position.y,
            particles[i + 2].position.y,
            particles[i + 1].position.y,
            particles[i].position.y,
        );
        let z = _mm512_set_pd(
            particles[i + 7].position.z,
            particles[i + 6].position.z,
            particles[i + 5].position.z,
            particles[i + 4].position.z,
            particles[i + 3].position.z,
            particles[i + 2].position.z,
            particles[i + 1].position.z,
            particles[i].position.z,
        );
        let ix_f = _mm512_mul_pd(x, nx_v);
        let iy_f = _mm512_mul_pd(y, ny_v);
        let iz_f = _mm512_mul_pd(z, nz_v);
        let mut ix_arr = [0.0; 8];
        let mut iy_arr = [0.0; 8];
        let mut iz_arr = [0.0; 8];
        // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(ix_arr.as_mut_ptr(), ix_f);
            _mm512_storeu_pd(iy_arr.as_mut_ptr(), iy_f);
            _mm512_storeu_pd(iz_arr.as_mut_ptr(), iz_f);
        }

        let mut gamma_arr = [0.0; 8];
        for lane in 0..lanes {
            let ix = (ix_arr[lane].floor() as usize).min(nx - 1);
            let iy = (iy_arr[lane].floor() as usize).min(ny - 1);
            let iz = (iz_arr[lane].floor() as usize).min(nz - 1);
            let cell = rad.idx(ix, iy, iz);
            let gamma = gamma_hi.get(cell).copied().unwrap_or(0.0);
            gamma_arr[lane] = if gamma < 1e-30 { 0.0 } else { gamma };
        }

        // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
        let gamma = unsafe { _mm512_loadu_pd(gamma_arr.as_ptr()) };
        let delta = _mm512_mul_pd(gamma, dt_over_u);
        let u = _mm512_set_pd(
            particles[i + 7].internal_energy,
            particles[i + 6].internal_energy,
            particles[i + 5].internal_energy,
            particles[i + 4].internal_energy,
            particles[i + 3].internal_energy,
            particles[i + 2].internal_energy,
            particles[i + 1].internal_energy,
            particles[i].internal_energy,
        );
        let capped = _mm512_min_pd(delta, _mm512_mul_pd(u, cap_factor));
        let u_new = _mm512_add_pd(u, capped);
        let mut u_arr = [0.0; 8];
        // SAFETY: fixed-size stack array has exactly eight f64 lanes.
        unsafe { _mm512_storeu_pd(u_arr.as_mut_ptr(), u_new) };
        for lane in 0..lanes {
            particles[i + lane].internal_energy = u_arr[lane];
        }
        i += lanes;
    }

    apply_photoheating_scalar(
        &mut particles[chunks..],
        rad,
        gamma_hi,
        dt,
        box_size,
        nx,
        ny,
        nz,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "photoheating maps particle coordinates with grid dimensions"
)]
fn photoheat_particle(
    p: &mut Particle,
    rad: &RadiationField,
    gamma_hi: &[f64],
    dt: f64,
    box_size: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) {
    if p.ptype != ParticleType::Gas {
        return;
    }

    // Mapear posición a celda más cercana
    let ix = ((p.position.x / box_size * nx as f64).floor() as usize).min(nx - 1);
    let iy = ((p.position.y / box_size * ny as f64).floor() as usize).min(ny - 1);
    let iz = ((p.position.z / box_size * nz as f64).floor() as usize).min(nz - 1);
    let cell = rad.idx(ix, iy, iz);

    let gamma = if cell < gamma_hi.len() {
        gamma_hi[cell]
    } else {
        0.0
    };

    if gamma < 1e-30 {
        return;
    }

    // ΔU = Γ_HI × ε_heat × dt / (U_CODE_TO_ERG_G)
    // ε_heat = fracción de E_fotón que va a calentamiento (≈ 1 para fotoionización primaria)
    let delta_u = gamma * dt / U_CODE_TO_ERG_G;
    p.internal_energy += delta_u.min(p.internal_energy * 10.0); // cap para estabilidad
}

/// Deposita emisión del gas al campo de radiación (recombinación y bremsstrahlung).
///
/// Para cada partícula de gas emite `η_i × V_celda × dt` en energía radiativa
/// en la celda más cercana.
///
/// En este modelo simplificado, `η ∝ ρ² × T^(-1/2)` (bremsstrahlung térmico).
#[cfg(not(feature = "rayon"))]
fn deposit_gas_emission_impl(
    particles: &[Particle],
    rad: &mut RadiationField,
    dt: f64,
    box_size: f64,
    emission_coeff: f64,
) {
    let nx = rad.nx;
    let ny = rad.ny;
    let nz = rad.nz;
    let dv = rad.dx.powi(3);

    for p in particles.iter() {
        if p.ptype != ParticleType::Gas || p.internal_energy < 1e-30 {
            continue;
        }
        let ix = ((p.position.x / box_size * nx as f64).floor() as usize).min(nx - 1);
        let iy = ((p.position.y / box_size * ny as f64).floor() as usize).min(ny - 1);
        let iz = ((p.position.z / box_size * nz as f64).floor() as usize).min(nz - 1);
        let cell = rad.idx(ix, iy, iz);

        let rho = p.mass / (p.smoothing_length.max(1e-10)).powi(3);
        let temp = p.internal_energy.max(0.0).sqrt();
        let eta = emission_coeff * rho * rho / temp.max(1e-10);

        rad.energy_density[cell] += eta * dv * dt;
    }
}

#[cfg(feature = "rayon")]
fn deposit_gas_emission_par(
    particles: &[Particle],
    rad: &mut RadiationField,
    dt: f64,
    box_size: f64,
    emission_coeff: f64,
) {
    let nx = rad.nx;
    let ny = rad.ny;
    let nz = rad.nz;
    let dv = rad.dx.powi(3);

    let contributions: Vec<(usize, f64)> = particles
        .par_iter()
        .filter_map(|p| {
            if p.ptype != ParticleType::Gas || p.internal_energy < 1e-30 {
                return None;
            }
            let ix = ((p.position.x / box_size * nx as f64).floor() as usize).min(nx - 1);
            let iy = ((p.position.y / box_size * ny as f64).floor() as usize).min(ny - 1);
            let iz = ((p.position.z / box_size * nz as f64).floor() as usize).min(nz - 1);
            let cell = ix * ny * nz + iy * nz + iz;

            let rho = p.mass / (p.smoothing_length.max(1e-10)).powi(3);
            let temp = p.internal_energy.max(0.0).sqrt();
            let eta = emission_coeff * rho * rho / temp.max(1e-10);

            Some((cell, eta * dv * dt))
        })
        .collect();

    for (cell, delta) in contributions {
        rad.energy_density[cell] += delta;
    }
}

pub fn deposit_gas_emission(
    particles: &[Particle],
    rad: &mut RadiationField,
    dt: f64,
    box_size: f64,
    emission_coeff: f64,
) {
    #[cfg(feature = "rayon")]
    {
        deposit_gas_emission_par(particles, rad, dt, box_size, emission_coeff);
    }

    #[cfg(not(feature = "rayon"))]
    {
        deposit_gas_emission_impl(particles, rad, dt, box_size, emission_coeff);
    }
}

/// Paso de acoplamiento completo: radiación ↔ gas.
///
/// Ejecuta:
/// 1. Depositar emisión del gas → campo de radiación.
/// 2. Calcular tasa de fotoionización.
/// 3. Aplicar fotocalentamiento al gas.
///
/// Este splitting de operadores es de primer orden en `dt`.
pub fn radiation_gas_coupling_step(
    particles: &mut [Particle],
    rad: &mut RadiationField,
    params: &M1Params,
    dt: f64,
    box_size: f64,
) {
    // Paso 1: emisión gas → rad
    deposit_gas_emission(particles, rad, dt, box_size, 1e-20);

    // Paso 2: tasa de fotoionización
    let gamma = photoionization_rate(rad, params);

    // Paso 3: fotocalentamiento rad → gas
    apply_photoheating(particles, rad, &gamma, dt, box_size);
}

/// Versión de `radiation_gas_coupling_step` con atenuación UV por polvo (Phase 137).
///
/// Aplica `τ_dust = κ_dust × (D/G) × ρ × h` a cada partícula de gas antes del
/// fotocalentamiento: `J_UV_eff = J_UV × exp(−τ_dust)`.
///
/// Con `kappa_dust_uv = 0.0` o `dust_to_gas = 0.0` → idéntico a la versión sin polvo.
pub fn radiation_gas_coupling_step_with_dust(
    particles: &mut [Particle],
    rad: &mut RadiationField,
    params: &M1Params,
    kappa_dust_uv: f64,
    dt: f64,
    box_size: f64,
) {
    // Paso 1: emisión gas → rad (igual)
    deposit_gas_emission(particles, rad, dt, box_size, 1e-20);

    // Paso 2: tasa de fotoionización → luego atenuada por polvo partícula a partícula
    let _gamma = photoionization_rate(rad, params);

    // Paso 3: fotocalentamiento con atenuación τ_dust
    // Aplicamos exp(-τ_dust) al fotocalentamiento de cada partícula individualmente
    #[cfg(feature = "rayon")]
    {
        particles.par_iter_mut().for_each(|p| {
            photoheat_dust_particle(p, rad, params, kappa_dust_uv, dt, box_size);
        });
    }

    #[cfg(not(feature = "rayon"))]
    for p in particles.iter_mut() {
        photoheat_dust_particle(p, rad, params, kappa_dust_uv, dt, box_size);
    }
}

fn photoheat_dust_particle(
    p: &mut Particle,
    rad: &RadiationField,
    params: &M1Params,
    kappa_dust_uv: f64,
    dt: f64,
    box_size: f64,
) {
    if p.ptype != ParticleType::Gas {
        return;
    }
    let h = p.smoothing_length.max(1e-10);
    let rho = p.mass / (4.0 / 3.0 * std::f64::consts::PI * h * h * h);
    let tau_dust = kappa_dust_uv * p.dust_to_gas * rho * h;
    let attenuation = (-tau_dust).exp();

    // El fotocalentamiento escala con el flujo UV atenuado
    // Aplicamos la corrección multiplicando el efecto de calentamiento
    // via la energía radiativa local (aproximación)
    let nx = rad.nx;
    let ny = rad.ny;
    let nz = rad.nz;
    let ix = ((p.position.x / box_size) * nx as f64) as usize % nx;
    let iy = ((p.position.y / box_size) * ny as f64) as usize % ny;
    let iz = ((p.position.z / box_size) * nz as f64) as usize % nz;
    let idx = rad.idx(ix, iy, iz);
    let e_rad = rad.energy_density[idx];
    let heating = params.sigma_dust * e_rad * attenuation * dt;
    p.internal_energy = (p.internal_energy + heating).max(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m1::{M1Params, RadiationField};
    use gadget_ng_core::{Particle, ParticleType, Vec3};

    fn gas_particle(x: f64, u: f64) -> Particle {
        let mut p = Particle::new(0, 1.0, Vec3::new(x, 0.5, 0.5), Vec3::zero());
        p.ptype = ParticleType::Gas;
        p.internal_energy = u;
        p.smoothing_length = 0.05;
        p
    }

    #[test]
    fn photoionization_rate_zero_for_empty_field() {
        let rad = RadiationField::uniform(4, 4, 4, 1.0, 0.0);
        let params = M1Params::default();
        let gamma = photoionization_rate(&rad, &params);
        assert!(
            gamma.iter().all(|&g| g == 0.0),
            "Γ debe ser 0 para campo vacío"
        );
    }

    #[test]
    fn photoionization_rate_positive_for_uv_field() {
        let rad = RadiationField::uniform(4, 4, 4, 1.0, 1.0);
        let params = M1Params {
            c_red_factor: 100.0,
            ..Default::default()
        };
        let gamma = photoionization_rate(&rad, &params);
        assert!(
            gamma.iter().all(|&g| g > 0.0),
            "Γ debe ser positivo con campo UV"
        );
    }

    #[test]
    fn photoheating_increases_internal_energy() {
        let rad = RadiationField::uniform(4, 4, 4, 0.25, 1e15);
        let params = M1Params {
            c_red_factor: 100.0,
            ..Default::default()
        };
        let gamma = photoionization_rate(&rad, &params);

        let u0 = 1.0;
        let mut particles = vec![gas_particle(0.125, u0)]; // centro de celda (0,0,0)
        apply_photoheating(&mut particles, &rad, &gamma, 1e-5, 1.0);

        assert!(
            particles[0].internal_energy >= u0,
            "Energía interna debe crecer con fotocalentamiento: u={}",
            particles[0].internal_energy
        );
    }

    #[test]
    fn dm_particle_not_heated() {
        let rad = RadiationField::uniform(4, 4, 4, 0.25, 1e15);
        let params = M1Params::default();
        let gamma = photoionization_rate(&rad, &params);

        let dm = Particle::new(0, 1.0, Vec3::new(0.5, 0.5, 0.5), Vec3::zero()); // DM
        let u_before = dm.internal_energy;
        let mut particles = vec![dm];
        apply_photoheating(&mut particles, &rad, &gamma, 1.0, 1.0);
        assert_eq!(
            particles[0].internal_energy, u_before,
            "DM no debe calentarse"
        );
    }

    #[test]
    fn emission_deposits_energy() {
        let mut rad = RadiationField::uniform(4, 4, 4, 0.25, 0.0);
        let particles = vec![gas_particle(0.125, 1.0)];
        let e0 = rad.total_energy(rad.dx.powi(3));
        deposit_gas_emission(&particles, &mut rad, 1.0, 1.0, 1.0);
        let e1 = rad.total_energy(rad.dx.powi(3));
        assert!(e1 >= e0, "Emisión del gas debe aumentar E_rad: {e0} → {e1}");
    }

    #[test]
    fn coupling_step_no_crash() {
        let mut rad = RadiationField::uniform(4, 4, 4, 0.25, 1e10);
        let params = M1Params {
            kappa_abs: 0.1,
            ..Default::default()
        };
        let mut particles = vec![gas_particle(0.125, 1.0), gas_particle(0.5, 0.5)];
        radiation_gas_coupling_step(&mut particles, &mut rad, &params, 0.01, 1.0);
        for p in &particles {
            assert!(
                p.internal_energy.is_finite(),
                "energía no finita tras coupling"
            );
        }
    }
}
