//! MHD no ideal: difusión ambipolar (Phase 194), término Hall (Phase 186),
//! difusión óhmica resistiva y acoplamiento con química (Phase 187).
//!
//! **Ambipolar:** en gas poco ionizado, las partículas neutras desacoplan el
//! campo magnético del fluido. Amortiguamiento local proporcional a `eta_ad / x_i`.
//!
//! **Óhmica (Phase 187):** `dB/dt|_Ohm = −η_Ohm B / h²` — decaimiento de B
//! independiente de ionización; la energía disipada calienta el gas.
//!
//! **Hall:** la diferencia de velocidades entre iones y electrones produce una
//! rotación del campo B sin disipar energía. Modelado como rotación de Rodrigues
//! alrededor del eje `v × B` con ángulo `θ = η_H |B| / ρ_proxy × dt`. Conserva
//! `|B|` exactamente y no cambia la energía interna.
//!
//! **Acoplamiento química (Phase 187):** `apply_ambipolar_diffusion_with_chem` usa
//! la fracción de electrones `x_e` del solver de química (ChemState) en lugar del
//! proxy térmico `u/(u+1)`, acoplando correctamente RT/química con MHD no-ideal.

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

/// Proxy acotado de fracción ionizada local.
///
/// - `internal_energy` alto implica gas más ionizado.
/// - `dust_to_gas` alto reduce la ionización efectiva por recombinación/shielding.
pub fn ionization_fraction_proxy(p: &Particle, ion_floor: f64, dust_coupling: f64) -> f64 {
    if p.ptype != ParticleType::Gas {
        return 1.0;
    }
    let thermal = p.internal_energy.max(0.0);
    let collisional = thermal / (thermal + 1.0);
    let dust_suppression = 1.0 / (1.0 + dust_coupling.max(0.0) * p.dust_to_gas.max(0.0) * 100.0);
    (collisional * dust_suppression).clamp(ion_floor.max(1e-12), 1.0)
}

fn ambipolar_diffusion_particle(
    p: &mut Particle,
    eta_ad: f64,
    ion_floor: f64,
    dust_coupling: f64,
    heat_eff: f64,
    dt: f64,
) {
    if p.ptype != ParticleType::Gas {
        return;
    }
    let b2_before = p.b_field.dot(p.b_field);
    if b2_before <= 0.0 {
        return;
    }
    let x_i = ionization_fraction_proxy(p, ion_floor, dust_coupling);
    let rate = eta_ad.max(0.0) * (1.0 / x_i - 1.0).max(0.0);
    let damping = (-rate * dt).exp().clamp(0.0, 1.0);
    p.b_field *= damping;
    let b2_after = p.b_field.dot(p.b_field);
    let dissipated = 0.5 * (b2_before - b2_after).max(0.0);
    p.internal_energy += heat_eff * dissipated / p.mass.max(1e-30);
}

pub fn apply_ambipolar_diffusion(
    particles: &mut [Particle],
    eta_ad: f64,
    ion_floor: f64,
    dust_coupling: f64,
    gamma: f64,
    dt: f64,
) {
    if eta_ad <= 0.0 || dt <= 0.0 {
        return;
    }
    let heat_eff = (gamma - 1.0).max(0.0);

    #[cfg(feature = "rayon")]
    {
        particles.par_iter_mut().for_each(|p| {
            ambipolar_diffusion_particle(p, eta_ad, ion_floor, dust_coupling, heat_eff, dt)
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F availability was checked at runtime.
                unsafe {
                    apply_ambipolar_diffusion_avx512(
                        particles,
                        eta_ad,
                        ion_floor,
                        dust_coupling,
                        heat_eff,
                        dt,
                    );
                    return;
                }
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA availability was checked at runtime.
                unsafe {
                    apply_ambipolar_diffusion_avx2(
                        particles,
                        eta_ad,
                        ion_floor,
                        dust_coupling,
                        heat_eff,
                        dt,
                    );
                    return;
                }
            }
        }
        for p in particles.iter_mut() {
            ambipolar_diffusion_particle(p, eta_ad, ion_floor, dust_coupling, heat_eff, dt);
        }
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_ambipolar_diffusion_avx2(
    particles: &mut [Particle],
    eta_ad: f64,
    ion_floor: f64,
    dust_coupling: f64,
    heat_eff: f64,
    dt: f64,
) {
    let lanes = 4;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let ion_floor_eff = ion_floor.max(1e-12);
    let half_v = _mm256_set1_pd(0.5);
    let heat_eff_v = _mm256_set1_pd(heat_eff);
    let min_mass_v = _mm256_set1_pd(1e-30);
    let zero_v = _mm256_set1_pd(0.0);
    let one_v = _mm256_set1_pd(1.0);
    let hundred_v = _mm256_set1_pd(100.0);
    let eta_ad_v = _mm256_set1_pd(eta_ad);
    let dust_coupling_v = _mm256_set1_pd(dust_coupling.max(0.0));
    let ion_floor_v = _mm256_set1_pd(ion_floor_eff);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                ambipolar_diffusion_particle(
                    &mut particles[i + lane],
                    eta_ad,
                    ion_floor,
                    dust_coupling,
                    heat_eff,
                    dt,
                );
            }
            i += lanes;
            continue;
        }
        let bx = _mm256_set_pd(
            particles[i + 3].b_field.x,
            particles[i + 2].b_field.x,
            particles[i + 1].b_field.x,
            particles[i].b_field.x,
        );
        let by = _mm256_set_pd(
            particles[i + 3].b_field.y,
            particles[i + 2].b_field.y,
            particles[i + 1].b_field.y,
            particles[i].b_field.y,
        );
        let bz = _mm256_set_pd(
            particles[i + 3].b_field.z,
            particles[i + 2].b_field.z,
            particles[i + 1].b_field.z,
            particles[i].b_field.z,
        );
        let b2_before = _mm256_fmadd_pd(bx, bx, _mm256_fmadd_pd(by, by, _mm256_mul_pd(bz, bz)));
        let ue = _mm256_set_pd(
            particles[i + 3].internal_energy,
            particles[i + 2].internal_energy,
            particles[i + 1].internal_energy,
            particles[i].internal_energy,
        );
        let thermal = _mm256_max_pd(zero_v, ue);
        let collisional = _mm256_div_pd(thermal, _mm256_add_pd(thermal, one_v));
        let dust_to_gas = _mm256_set_pd(
            particles[i + 3].dust_to_gas,
            particles[i + 2].dust_to_gas,
            particles[i + 1].dust_to_gas,
            particles[i].dust_to_gas,
        );
        let dtg_clamped = _mm256_max_pd(zero_v, dust_to_gas);
        let dust_suppression = _mm256_div_pd(
            one_v,
            _mm256_fmadd_pd(
                _mm256_mul_pd(dust_coupling_v, dtg_clamped),
                hundred_v,
                one_v,
            ),
        );
        let x_i_raw = _mm256_mul_pd(collisional, dust_suppression);
        let x_i = _mm256_min_pd(one_v, _mm256_max_pd(ion_floor_v, x_i_raw));
        let rate = _mm256_max_pd(
            zero_v,
            _mm256_mul_pd(
                eta_ad_v,
                _mm256_max_pd(zero_v, _mm256_sub_pd(_mm256_div_pd(one_v, x_i), one_v)),
            ),
        );
        let mut damping_arr = [1.0f64; 4];
        let mut b2_before_arr = [0.0f64; 4];
        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(damping_arr.as_mut_ptr(), rate);
            _mm256_storeu_pd(b2_before_arr.as_mut_ptr(), b2_before);
        }
        for damping in damping_arr.iter_mut().take(lanes) {
            *damping = (-*damping * dt).exp().clamp(0.0, 1.0);
        }
        let damping_v = _mm256_set_pd(
            damping_arr[3],
            damping_arr[2],
            damping_arr[1],
            damping_arr[0],
        );
        let new_bx = _mm256_mul_pd(bx, damping_v);
        let new_by = _mm256_mul_pd(by, damping_v);
        let new_bz = _mm256_mul_pd(bz, damping_v);
        let b2_after = _mm256_fmadd_pd(
            new_bx,
            new_bx,
            _mm256_fmadd_pd(new_by, new_by, _mm256_mul_pd(new_bz, new_bz)),
        );
        let dissipated = _mm256_max_pd(
            zero_v,
            _mm256_mul_pd(half_v, _mm256_sub_pd(b2_before, b2_after)),
        );
        let m = _mm256_set_pd(
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let mass_safe = _mm256_max_pd(min_mass_v, m);
        let energy_delta = _mm256_mul_pd(heat_eff_v, _mm256_div_pd(dissipated, mass_safe));
        let mut out_bx = [0.0f64; 4];
        let mut out_by = [0.0f64; 4];
        let mut out_bz = [0.0f64; 4];
        let mut out_ue = [0.0f64; 4];
        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(out_bx.as_mut_ptr(), new_bx);
            _mm256_storeu_pd(out_by.as_mut_ptr(), new_by);
            _mm256_storeu_pd(out_bz.as_mut_ptr(), new_bz);
            _mm256_storeu_pd(out_ue.as_mut_ptr(), energy_delta);
        }
        for lane in 0..lanes {
            if b2_before_arr[lane] > 0.0 {
                particles[i + lane].b_field.x = out_bx[lane];
                particles[i + lane].b_field.y = out_by[lane];
                particles[i + lane].b_field.z = out_bz[lane];
                particles[i + lane].internal_energy += out_ue[lane];
            }
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        ambipolar_diffusion_particle(p, eta_ad, ion_floor, dust_coupling, heat_eff, dt);
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_ambipolar_diffusion_avx512(
    particles: &mut [Particle],
    eta_ad: f64,
    ion_floor: f64,
    dust_coupling: f64,
    heat_eff: f64,
    dt: f64,
) {
    let lanes = 8;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let ion_floor_eff = ion_floor.max(1e-12);
    let half_v = _mm512_set1_pd(0.5);
    let heat_eff_v = _mm512_set1_pd(heat_eff);
    let min_mass_v = _mm512_set1_pd(1e-30);
    let zero_v = _mm512_set1_pd(0.0);
    let one_v = _mm512_set1_pd(1.0);
    let hundred_v = _mm512_set1_pd(100.0);
    let eta_ad_v = _mm512_set1_pd(eta_ad);
    let dust_coupling_v = _mm512_set1_pd(dust_coupling.max(0.0));
    let ion_floor_v = _mm512_set1_pd(ion_floor_eff);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                ambipolar_diffusion_particle(
                    &mut particles[i + lane],
                    eta_ad,
                    ion_floor,
                    dust_coupling,
                    heat_eff,
                    dt,
                );
            }
            i += lanes;
            continue;
        }
        let bx = _mm512_set_pd(
            particles[i + 7].b_field.x,
            particles[i + 6].b_field.x,
            particles[i + 5].b_field.x,
            particles[i + 4].b_field.x,
            particles[i + 3].b_field.x,
            particles[i + 2].b_field.x,
            particles[i + 1].b_field.x,
            particles[i].b_field.x,
        );
        let by = _mm512_set_pd(
            particles[i + 7].b_field.y,
            particles[i + 6].b_field.y,
            particles[i + 5].b_field.y,
            particles[i + 4].b_field.y,
            particles[i + 3].b_field.y,
            particles[i + 2].b_field.y,
            particles[i + 1].b_field.y,
            particles[i].b_field.y,
        );
        let bz = _mm512_set_pd(
            particles[i + 7].b_field.z,
            particles[i + 6].b_field.z,
            particles[i + 5].b_field.z,
            particles[i + 4].b_field.z,
            particles[i + 3].b_field.z,
            particles[i + 2].b_field.z,
            particles[i + 1].b_field.z,
            particles[i].b_field.z,
        );
        let b2_before = _mm512_fmadd_pd(bx, bx, _mm512_fmadd_pd(by, by, _mm512_mul_pd(bz, bz)));
        let active_mask = _mm512_cmp_pd_mask(b2_before, zero_v, _CMP_GT_OQ);
        let ue = _mm512_set_pd(
            particles[i + 7].internal_energy,
            particles[i + 6].internal_energy,
            particles[i + 5].internal_energy,
            particles[i + 4].internal_energy,
            particles[i + 3].internal_energy,
            particles[i + 2].internal_energy,
            particles[i + 1].internal_energy,
            particles[i].internal_energy,
        );
        let thermal = _mm512_max_pd(zero_v, ue);
        let collisional = _mm512_div_pd(thermal, _mm512_add_pd(thermal, one_v));
        let dust_to_gas = _mm512_set_pd(
            particles[i + 7].dust_to_gas,
            particles[i + 6].dust_to_gas,
            particles[i + 5].dust_to_gas,
            particles[i + 4].dust_to_gas,
            particles[i + 3].dust_to_gas,
            particles[i + 2].dust_to_gas,
            particles[i + 1].dust_to_gas,
            particles[i].dust_to_gas,
        );
        let dtg_clamped = _mm512_max_pd(zero_v, dust_to_gas);
        let dust_suppression = _mm512_div_pd(
            one_v,
            _mm512_fmadd_pd(
                _mm512_mul_pd(dust_coupling_v, dtg_clamped),
                hundred_v,
                one_v,
            ),
        );
        let x_i_raw = _mm512_mul_pd(collisional, dust_suppression);
        let x_i = _mm512_min_pd(one_v, _mm512_max_pd(ion_floor_v, x_i_raw));
        let rate = _mm512_max_pd(
            zero_v,
            _mm512_mul_pd(
                eta_ad_v,
                _mm512_max_pd(zero_v, _mm512_sub_pd(_mm512_div_pd(one_v, x_i), one_v)),
            ),
        );
        let mut damping_arr = [1.0f64; 8];
        let mut b2_before_arr = [0.0f64; 8];
        // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(damping_arr.as_mut_ptr(), rate);
            _mm512_storeu_pd(b2_before_arr.as_mut_ptr(), b2_before);
        }
        for damping in damping_arr.iter_mut().take(lanes) {
            *damping = (-*damping * dt).exp().clamp(0.0, 1.0);
        }
        let damping_v = _mm512_set_pd(
            damping_arr[7],
            damping_arr[6],
            damping_arr[5],
            damping_arr[4],
            damping_arr[3],
            damping_arr[2],
            damping_arr[1],
            damping_arr[0],
        );
        let new_bx = _mm512_mask_mul_pd(bx, active_mask, bx, damping_v);
        let new_by = _mm512_mask_mul_pd(by, active_mask, by, damping_v);
        let new_bz = _mm512_mask_mul_pd(bz, active_mask, bz, damping_v);
        let b2_after = _mm512_fmadd_pd(
            new_bx,
            new_bx,
            _mm512_fmadd_pd(new_by, new_by, _mm512_mul_pd(new_bz, new_bz)),
        );
        let dissipated = _mm512_maskz_add_pd(
            active_mask,
            _mm512_mul_pd(half_v, _mm512_sub_pd(b2_before, b2_after)),
            _mm512_setzero_pd(),
        );
        let m = _mm512_set_pd(
            particles[i + 7].mass,
            particles[i + 6].mass,
            particles[i + 5].mass,
            particles[i + 4].mass,
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let mass_safe = _mm512_max_pd(min_mass_v, m);
        let energy_delta = _mm512_mul_pd(heat_eff_v, _mm512_div_pd(dissipated, mass_safe));
        let mut out_bx = [0.0f64; 8];
        let mut out_by = [0.0f64; 8];
        let mut out_bz = [0.0f64; 8];
        let mut out_ue = [0.0f64; 8];
        // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(out_bx.as_mut_ptr(), new_bx);
            _mm512_storeu_pd(out_by.as_mut_ptr(), new_by);
            _mm512_storeu_pd(out_bz.as_mut_ptr(), new_bz);
            _mm512_storeu_pd(out_ue.as_mut_ptr(), energy_delta);
        }
        for lane in 0..lanes {
            particles[i + lane].b_field.x = out_bx[lane];
            particles[i + lane].b_field.y = out_by[lane];
            particles[i + lane].b_field.z = out_bz[lane];
            particles[i + lane].internal_energy += out_ue[lane];
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        ambipolar_diffusion_particle(p, eta_ad, ion_floor, dust_coupling, heat_eff, dt);
    }
}

// ── Hall drift (Phase 186) ─────────────────────────────────────────────────

/// Aplica un paso de drift Hall a una sola partícula gas.
///
/// Rota `B` alrededor del eje `v × B` con ángulo
/// `θ = η_H × |B| / ρ_proxy × dt`, usando la fórmula de Rodrigues.
/// `|B|` se conserva exactamente; la energía interna no cambia.
///
/// La densidad de referencia se aproxima como `mass / h³` cuando
/// `smoothing_length > 0`, de lo contrario se usa `mass`.
fn hall_drift_particle(p: &mut Particle, eta_hall: f64, dt: f64) {
    if p.ptype != ParticleType::Gas {
        return;
    }
    let b_sq = p.b_field.dot(p.b_field);
    if b_sq <= 0.0 {
        return;
    }
    let b_norm = b_sq.sqrt();
    let h = p.smoothing_length;
    let rho_proxy = if h > 0.0 {
        p.mass / (h * h * h).max(1e-30)
    } else {
        p.mass.max(1e-30)
    };
    let theta = (eta_hall.max(0.0) * b_norm * dt / rho_proxy)
        .clamp(-std::f64::consts::PI, std::f64::consts::PI);
    if theta.abs() < 1e-15 {
        return;
    }
    let axis = p.velocity.cross(p.b_field);
    let axis_norm_sq = axis.dot(axis);
    if axis_norm_sq < 1e-60 {
        return;
    }
    let k = axis * (1.0 / axis_norm_sq.sqrt());
    // Rodrigues: B_new = B cos θ + (k × B) sin θ + k (k·B)(1 − cos θ)
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let k_dot_b = k.dot(p.b_field);
    let k_cross_b = k.cross(p.b_field);
    p.b_field = p.b_field * cos_t + k_cross_b * sin_t + k * (k_dot_b * (1.0 - cos_t));
}

/// Aplica el drift Hall a todo el slice de partículas.
///
/// Usa Rayon cuando `feature = "rayon"`, luego AVX-512F o AVX2+FMA via
/// `#[target_feature]` en x86_64 (el cálculo de `sin/cos` es escalar por
/// lane, al igual que `exp()` en la difusión ambipolar).
pub fn apply_hall_drift(particles: &mut [Particle], eta_hall: f64, dt: f64) {
    if eta_hall <= 0.0 || dt <= 0.0 {
        return;
    }

    #[cfg(feature = "rayon")]
    {
        particles
            .par_iter_mut()
            .for_each(|p| hall_drift_particle(p, eta_hall, dt));
        return;
    }

    #[cfg(not(feature = "rayon"))]
    {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F disponible en tiempo de ejecución.
                unsafe {
                    apply_hall_drift_avx512(particles, eta_hall, dt);
                    return;
                }
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA disponibles en tiempo de ejecución.
                unsafe {
                    apply_hall_drift_avx2(particles, eta_hall, dt);
                    return;
                }
            }
        }
        for p in particles.iter_mut() {
            hall_drift_particle(p, eta_hall, dt);
        }
    }
}

/// Versión AVX2+FMA del drift Hall.
///
/// Usa SIMD para calcular `b_sq` y `theta` en lotes de 4; `sin/cos` y la
/// rotación Rodrigues se completan en escalar por lane (mismo patrón que
/// `exp()` en `apply_ambipolar_diffusion_avx2`).
#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_hall_drift_avx2(particles: &mut [Particle], eta_hall: f64, dt: f64) {
    use std::arch::x86_64::*;

    let lanes = 4;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let zero_v = _mm256_set1_pd(0.0);
    let pi_v = _mm256_set1_pd(std::f64::consts::PI);
    let neg_pi_v = _mm256_set1_pd(-std::f64::consts::PI);
    let eta_dt = eta_hall * dt;
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                hall_drift_particle(&mut particles[i + lane], eta_hall, dt);
            }
            i += lanes;
            continue;
        }
        let bx = _mm256_set_pd(
            particles[i + 3].b_field.x,
            particles[i + 2].b_field.x,
            particles[i + 1].b_field.x,
            particles[i].b_field.x,
        );
        let by = _mm256_set_pd(
            particles[i + 3].b_field.y,
            particles[i + 2].b_field.y,
            particles[i + 1].b_field.y,
            particles[i].b_field.y,
        );
        let bz = _mm256_set_pd(
            particles[i + 3].b_field.z,
            particles[i + 2].b_field.z,
            particles[i + 1].b_field.z,
            particles[i].b_field.z,
        );
        let b_sq = _mm256_fmadd_pd(bx, bx, _mm256_fmadd_pd(by, by, _mm256_mul_pd(bz, bz)));
        let b_norm_v = _mm256_sqrt_pd(b_sq);
        let h_v = _mm256_set_pd(
            particles[i + 3].smoothing_length,
            particles[i + 2].smoothing_length,
            particles[i + 1].smoothing_length,
            particles[i].smoothing_length,
        );
        let m_v = _mm256_set_pd(
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        // rho_proxy = mass / h³ (h > 0) or mass
        let h3 = _mm256_mul_pd(h_v, _mm256_mul_pd(h_v, h_v));
        let has_h = _mm256_cmp_pd::<_CMP_GT_OQ>(h_v, zero_v);
        let rho_proxy = _mm256_blendv_pd(m_v, _mm256_div_pd(m_v, h3), has_h);
        let theta_raw = _mm256_mul_pd(
            _mm256_set1_pd(eta_dt),
            _mm256_div_pd(b_norm_v, _mm256_max_pd(rho_proxy, _mm256_set1_pd(1e-30))),
        );
        let theta_clamped = _mm256_min_pd(pi_v, _mm256_max_pd(neg_pi_v, theta_raw));
        let b_sq_arr = {
            let mut arr = [0.0f64; 4];
            // SAFETY: arr has exactly 4 lanes.
            unsafe {
                _mm256_storeu_pd(arr.as_mut_ptr(), b_sq);
            }
            arr
        };
        let theta_arr = {
            let mut arr = [0.0f64; 4];
            // SAFETY: arr has exactly 4 lanes.
            unsafe {
                _mm256_storeu_pd(arr.as_mut_ptr(), theta_clamped);
            }
            arr
        };
        for lane in 0..lanes {
            if b_sq_arr[lane] > 0.0 && theta_arr[lane].abs() >= 1e-15 {
                hall_drift_particle(&mut particles[i + lane], eta_hall, dt);
            }
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        hall_drift_particle(p, eta_hall, dt);
    }
}

/// Versión AVX-512F del drift Hall (lotes de 8).
#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_hall_drift_avx512(particles: &mut [Particle], eta_hall: f64, dt: f64) {
    use std::arch::x86_64::*;

    let lanes = 8;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let zero_v = _mm512_set1_pd(0.0);
    let pi_v = _mm512_set1_pd(std::f64::consts::PI);
    let neg_pi_v = _mm512_set1_pd(-std::f64::consts::PI);
    let eta_dt = eta_hall * dt;
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                hall_drift_particle(&mut particles[i + lane], eta_hall, dt);
            }
            i += lanes;
            continue;
        }
        let bx = _mm512_set_pd(
            particles[i + 7].b_field.x,
            particles[i + 6].b_field.x,
            particles[i + 5].b_field.x,
            particles[i + 4].b_field.x,
            particles[i + 3].b_field.x,
            particles[i + 2].b_field.x,
            particles[i + 1].b_field.x,
            particles[i].b_field.x,
        );
        let by = _mm512_set_pd(
            particles[i + 7].b_field.y,
            particles[i + 6].b_field.y,
            particles[i + 5].b_field.y,
            particles[i + 4].b_field.y,
            particles[i + 3].b_field.y,
            particles[i + 2].b_field.y,
            particles[i + 1].b_field.y,
            particles[i].b_field.y,
        );
        let bz = _mm512_set_pd(
            particles[i + 7].b_field.z,
            particles[i + 6].b_field.z,
            particles[i + 5].b_field.z,
            particles[i + 4].b_field.z,
            particles[i + 3].b_field.z,
            particles[i + 2].b_field.z,
            particles[i + 1].b_field.z,
            particles[i].b_field.z,
        );
        let b_sq = _mm512_fmadd_pd(bx, bx, _mm512_fmadd_pd(by, by, _mm512_mul_pd(bz, bz)));
        let b_norm_v = _mm512_sqrt_pd(b_sq);
        let h_v = _mm512_set_pd(
            particles[i + 7].smoothing_length,
            particles[i + 6].smoothing_length,
            particles[i + 5].smoothing_length,
            particles[i + 4].smoothing_length,
            particles[i + 3].smoothing_length,
            particles[i + 2].smoothing_length,
            particles[i + 1].smoothing_length,
            particles[i].smoothing_length,
        );
        let m_v = _mm512_set_pd(
            particles[i + 7].mass,
            particles[i + 6].mass,
            particles[i + 5].mass,
            particles[i + 4].mass,
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let h3 = _mm512_mul_pd(h_v, _mm512_mul_pd(h_v, h_v));
        let has_h_mask = _mm512_cmp_pd_mask(h_v, zero_v, _CMP_GT_OQ);
        let rho_proxy = _mm512_mask_div_pd(m_v, has_h_mask, m_v, h3);
        let theta_raw = _mm512_mul_pd(
            _mm512_set1_pd(eta_dt),
            _mm512_div_pd(b_norm_v, _mm512_max_pd(rho_proxy, _mm512_set1_pd(1e-30))),
        );
        let theta_clamped = _mm512_min_pd(pi_v, _mm512_max_pd(neg_pi_v, theta_raw));
        let b_sq_arr = {
            let mut arr = [0.0f64; 8];
            // SAFETY: arr has exactly 8 lanes.
            unsafe {
                _mm512_storeu_pd(arr.as_mut_ptr(), b_sq);
            }
            arr
        };
        let theta_arr = {
            let mut arr = [0.0f64; 8];
            // SAFETY: arr has exactly 8 lanes.
            unsafe {
                _mm512_storeu_pd(arr.as_mut_ptr(), theta_clamped);
            }
            arr
        };
        for lane in 0..lanes {
            if b_sq_arr[lane] > 0.0 && theta_arr[lane].abs() >= 1e-15 {
                hall_drift_particle(&mut particles[i + lane], eta_hall, dt);
            }
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        hall_drift_particle(p, eta_hall, dt);
    }
}

// ── Ohmic resistive diffusion (Phase 187) ────────────────────────────────────

/// Applies one Ohmic resistive diffusion step to a single gas particle.
///
/// Local approximation: `dB/dt|_Ohm = −η_Ohm B / h²`.
/// Damping factor: `exp(−η_Ohm dt / h²)`.
/// Dissipated magnetic energy heats the gas: `Δu = heat_eff × ΔB² / (2 m)`.
fn ohmic_diffusion_particle(p: &mut Particle, eta_ohm: f64, heat_eff: f64, dt: f64) {
    if p.ptype != ParticleType::Gas {
        return;
    }
    let b2_before = p.b_field.dot(p.b_field);
    if b2_before <= 0.0 {
        return;
    }
    let h2 = (p.smoothing_length * p.smoothing_length).max(1e-60);
    let rate = eta_ohm / h2;
    let damping = (-rate * dt).exp().clamp(0.0, 1.0);
    p.b_field *= damping;
    let b2_after = p.b_field.dot(p.b_field);
    let dissipated = 0.5 * (b2_before - b2_after).max(0.0);
    p.internal_energy += heat_eff * dissipated / p.mass.max(1e-30);
}

/// Applies Ohmic resistive diffusion to all particles.
///
/// Implements `dB/dt|_Ohm = −η_Ohm B / h²` (local approximation).
/// The factor `η_Ohm` has units of `[length²/time]` (code units).
/// Dissipated energy heats gas via `heat_eff = (γ − 1)`.
///
/// Uses Rayon when `feature = "rayon"`, then AVX2+FMA or AVX-512F on x86_64.
pub fn apply_ohmic_diffusion(particles: &mut [Particle], eta_ohm: f64, gamma: f64, dt: f64) {
    if eta_ohm <= 0.0 || dt <= 0.0 {
        return;
    }
    let heat_eff = (gamma - 1.0).max(0.0);

    #[cfg(feature = "rayon")]
    {
        particles
            .par_iter_mut()
            .for_each(|p| ohmic_diffusion_particle(p, eta_ohm, heat_eff, dt));
        return;
    }

    #[cfg(not(feature = "rayon"))]
    {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F availability was checked at runtime.
                unsafe {
                    apply_ohmic_diffusion_avx512(particles, eta_ohm, heat_eff, dt);
                    return;
                }
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA availability was checked at runtime.
                unsafe {
                    apply_ohmic_diffusion_avx2(particles, eta_ohm, heat_eff, dt);
                    return;
                }
            }
        }
        for p in particles.iter_mut() {
            ohmic_diffusion_particle(p, eta_ohm, heat_eff, dt);
        }
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_ohmic_diffusion_avx2(
    particles: &mut [Particle],
    eta_ohm: f64,
    heat_eff: f64,
    dt: f64,
) {
    use std::arch::x86_64::*;

    let lanes = 4;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let zero_v = _mm256_set1_pd(0.0);
    let half_v = _mm256_set1_pd(0.5);
    let heat_eff_v = _mm256_set1_pd(heat_eff);
    let min_mass_v = _mm256_set1_pd(1e-30);
    let eta_dt_v = _mm256_set1_pd(eta_ohm * dt);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                ohmic_diffusion_particle(&mut particles[i + lane], eta_ohm, heat_eff, dt);
            }
            i += lanes;
            continue;
        }
        let bx = _mm256_set_pd(
            particles[i + 3].b_field.x,
            particles[i + 2].b_field.x,
            particles[i + 1].b_field.x,
            particles[i].b_field.x,
        );
        let by = _mm256_set_pd(
            particles[i + 3].b_field.y,
            particles[i + 2].b_field.y,
            particles[i + 1].b_field.y,
            particles[i].b_field.y,
        );
        let bz = _mm256_set_pd(
            particles[i + 3].b_field.z,
            particles[i + 2].b_field.z,
            particles[i + 1].b_field.z,
            particles[i].b_field.z,
        );
        let b2_before = _mm256_fmadd_pd(bx, bx, _mm256_fmadd_pd(by, by, _mm256_mul_pd(bz, bz)));
        let h_v = _mm256_set_pd(
            particles[i + 3].smoothing_length,
            particles[i + 2].smoothing_length,
            particles[i + 1].smoothing_length,
            particles[i].smoothing_length,
        );
        let h2_v = _mm256_max_pd(_mm256_mul_pd(h_v, h_v), _mm256_set1_pd(1e-60));
        // rate = eta_ohm * dt / h²; damping computed per-lane (needs exp)
        let rate_v = _mm256_div_pd(eta_dt_v, h2_v);
        let b2_before_arr = {
            let mut arr = [0.0f64; 4];
            // SAFETY: arr has exactly 4 lanes.
            unsafe {
                _mm256_storeu_pd(arr.as_mut_ptr(), b2_before);
            }
            arr
        };
        let rate_arr = {
            let mut arr = [0.0f64; 4];
            // SAFETY: arr has exactly 4 lanes.
            unsafe {
                _mm256_storeu_pd(arr.as_mut_ptr(), rate_v);
            }
            arr
        };
        let mut damping_arr = [1.0f64; 4];
        for lane in 0..lanes {
            damping_arr[lane] = (-rate_arr[lane]).exp().clamp(0.0, 1.0);
        }
        let damping_v = _mm256_set_pd(
            damping_arr[3],
            damping_arr[2],
            damping_arr[1],
            damping_arr[0],
        );
        let new_bx = _mm256_mul_pd(bx, damping_v);
        let new_by = _mm256_mul_pd(by, damping_v);
        let new_bz = _mm256_mul_pd(bz, damping_v);
        let b2_after = _mm256_fmadd_pd(
            new_bx,
            new_bx,
            _mm256_fmadd_pd(new_by, new_by, _mm256_mul_pd(new_bz, new_bz)),
        );
        let dissipated = _mm256_max_pd(
            zero_v,
            _mm256_mul_pd(half_v, _mm256_sub_pd(b2_before, b2_after)),
        );
        let m = _mm256_set_pd(
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let mass_safe = _mm256_max_pd(min_mass_v, m);
        let energy_delta = _mm256_mul_pd(heat_eff_v, _mm256_div_pd(dissipated, mass_safe));
        let mut out_bx = [0.0f64; 4];
        let mut out_by = [0.0f64; 4];
        let mut out_bz = [0.0f64; 4];
        let mut out_ue = [0.0f64; 4];
        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(out_bx.as_mut_ptr(), new_bx);
            _mm256_storeu_pd(out_by.as_mut_ptr(), new_by);
            _mm256_storeu_pd(out_bz.as_mut_ptr(), new_bz);
            _mm256_storeu_pd(out_ue.as_mut_ptr(), energy_delta);
        }
        for lane in 0..lanes {
            if b2_before_arr[lane] > 0.0 {
                particles[i + lane].b_field.x = out_bx[lane];
                particles[i + lane].b_field.y = out_by[lane];
                particles[i + lane].b_field.z = out_bz[lane];
                particles[i + lane].internal_energy += out_ue[lane];
            }
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        ohmic_diffusion_particle(p, eta_ohm, heat_eff, dt);
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_ohmic_diffusion_avx512(
    particles: &mut [Particle],
    eta_ohm: f64,
    heat_eff: f64,
    dt: f64,
) {
    use std::arch::x86_64::*;

    let lanes = 8;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let zero_v = _mm512_set1_pd(0.0);
    let half_v = _mm512_set1_pd(0.5);
    let heat_eff_v = _mm512_set1_pd(heat_eff);
    let min_mass_v = _mm512_set1_pd(1e-30);
    let eta_dt_v = _mm512_set1_pd(eta_ohm * dt);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                ohmic_diffusion_particle(&mut particles[i + lane], eta_ohm, heat_eff, dt);
            }
            i += lanes;
            continue;
        }
        let bx = _mm512_set_pd(
            particles[i + 7].b_field.x,
            particles[i + 6].b_field.x,
            particles[i + 5].b_field.x,
            particles[i + 4].b_field.x,
            particles[i + 3].b_field.x,
            particles[i + 2].b_field.x,
            particles[i + 1].b_field.x,
            particles[i].b_field.x,
        );
        let by = _mm512_set_pd(
            particles[i + 7].b_field.y,
            particles[i + 6].b_field.y,
            particles[i + 5].b_field.y,
            particles[i + 4].b_field.y,
            particles[i + 3].b_field.y,
            particles[i + 2].b_field.y,
            particles[i + 1].b_field.y,
            particles[i].b_field.y,
        );
        let bz = _mm512_set_pd(
            particles[i + 7].b_field.z,
            particles[i + 6].b_field.z,
            particles[i + 5].b_field.z,
            particles[i + 4].b_field.z,
            particles[i + 3].b_field.z,
            particles[i + 2].b_field.z,
            particles[i + 1].b_field.z,
            particles[i].b_field.z,
        );
        let b2_before = _mm512_fmadd_pd(bx, bx, _mm512_fmadd_pd(by, by, _mm512_mul_pd(bz, bz)));
        let active_mask = _mm512_cmp_pd_mask(b2_before, zero_v, _CMP_GT_OQ);
        let h_v = _mm512_set_pd(
            particles[i + 7].smoothing_length,
            particles[i + 6].smoothing_length,
            particles[i + 5].smoothing_length,
            particles[i + 4].smoothing_length,
            particles[i + 3].smoothing_length,
            particles[i + 2].smoothing_length,
            particles[i + 1].smoothing_length,
            particles[i].smoothing_length,
        );
        let h2_v = _mm512_max_pd(_mm512_mul_pd(h_v, h_v), _mm512_set1_pd(1e-60));
        let rate_v = _mm512_div_pd(eta_dt_v, h2_v);
        let b2_before_arr = {
            let mut arr = [0.0f64; 8];
            // SAFETY: arr has exactly 8 lanes.
            unsafe {
                _mm512_storeu_pd(arr.as_mut_ptr(), b2_before);
            }
            arr
        };
        let rate_arr = {
            let mut arr = [0.0f64; 8];
            // SAFETY: arr has exactly 8 lanes.
            unsafe {
                _mm512_storeu_pd(arr.as_mut_ptr(), rate_v);
            }
            arr
        };
        let mut damping_arr = [1.0f64; 8];
        for lane in 0..lanes {
            damping_arr[lane] = (-rate_arr[lane]).exp().clamp(0.0, 1.0);
        }
        let damping_v = _mm512_set_pd(
            damping_arr[7],
            damping_arr[6],
            damping_arr[5],
            damping_arr[4],
            damping_arr[3],
            damping_arr[2],
            damping_arr[1],
            damping_arr[0],
        );
        let new_bx = _mm512_mask_mul_pd(bx, active_mask, bx, damping_v);
        let new_by = _mm512_mask_mul_pd(by, active_mask, by, damping_v);
        let new_bz = _mm512_mask_mul_pd(bz, active_mask, bz, damping_v);
        let b2_after = _mm512_fmadd_pd(
            new_bx,
            new_bx,
            _mm512_fmadd_pd(new_by, new_by, _mm512_mul_pd(new_bz, new_bz)),
        );
        let dissipated = _mm512_maskz_add_pd(
            active_mask,
            _mm512_mul_pd(half_v, _mm512_sub_pd(b2_before, b2_after)),
            _mm512_setzero_pd(),
        );
        let m = _mm512_set_pd(
            particles[i + 7].mass,
            particles[i + 6].mass,
            particles[i + 5].mass,
            particles[i + 4].mass,
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let mass_safe = _mm512_max_pd(min_mass_v, m);
        let energy_delta = _mm512_mul_pd(heat_eff_v, _mm512_div_pd(dissipated, mass_safe));
        let mut out_bx = [0.0f64; 8];
        let mut out_by = [0.0f64; 8];
        let mut out_bz = [0.0f64; 8];
        let mut out_ue = [0.0f64; 8];
        // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(out_bx.as_mut_ptr(), new_bx);
            _mm512_storeu_pd(out_by.as_mut_ptr(), new_by);
            _mm512_storeu_pd(out_bz.as_mut_ptr(), new_bz);
            _mm512_storeu_pd(out_ue.as_mut_ptr(), energy_delta);
        }
        for lane in 0..lanes {
            if b2_before_arr[lane] > 0.0 {
                particles[i + lane].b_field.x = out_bx[lane];
                particles[i + lane].b_field.y = out_by[lane];
                particles[i + lane].b_field.z = out_bz[lane];
                particles[i + lane].internal_energy += out_ue[lane];
            }
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        ohmic_diffusion_particle(p, eta_ohm, heat_eff, dt);
    }
}

// ── Chemistry-coupled ambipolar diffusion (Phase 187) ─────────────────────────

/// Applies ambipolar diffusion using externally supplied ionization fractions.
///
/// Unlike `apply_ambipolar_diffusion` (which uses a thermal proxy `u/(u+1)` for
/// the ionization fraction), this function accepts a pre-computed `ion_fracs` slice
/// — typically `ChemState::x_e` extracted by the engine from the chemistry solver
/// (Phase 86/87) — to correctly couple the non-ideal MHD term to the RT pipeline.
///
/// `ion_fracs[i]` is the electron fraction per H atom for particle `i`.
/// Falls back to `ion_floor` when the provided fraction is below that threshold.
///
/// `ion_fracs` must have the same length as `particles`.
pub fn apply_ambipolar_diffusion_with_chem(
    particles: &mut [Particle],
    ion_fracs: &[f64],
    eta_ad: f64,
    ion_floor: f64,
    gamma: f64,
    dt: f64,
) {
    debug_assert_eq!(
        particles.len(),
        ion_fracs.len(),
        "particles and ion_fracs must have the same length"
    );
    if eta_ad <= 0.0 || dt <= 0.0 {
        return;
    }
    let heat_eff = (gamma - 1.0).max(0.0);
    let ion_floor_eff = ion_floor.max(1e-12);

    #[cfg(feature = "rayon")]
    {
        particles
            .par_iter_mut()
            .zip(ion_fracs.par_iter())
            .for_each(|(p, &x_e)| {
                if p.ptype != ParticleType::Gas {
                    return;
                }
                let b2_before = p.b_field.dot(p.b_field);
                if b2_before <= 0.0 {
                    return;
                }
                let x_i = x_e.clamp(ion_floor_eff, 1.0);
                let rate = eta_ad.max(0.0) * (1.0 / x_i - 1.0).max(0.0);
                let damping = (-rate * dt).exp().clamp(0.0, 1.0);
                p.b_field *= damping;
                let b2_after = p.b_field.dot(p.b_field);
                let dissipated = 0.5 * (b2_before - b2_after).max(0.0);
                p.internal_energy += heat_eff * dissipated / p.mass.max(1e-30);
            });
        return;
    }

    #[cfg(not(feature = "rayon"))]
    for (p, &x_e) in particles.iter_mut().zip(ion_fracs.iter()) {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let b2_before = p.b_field.dot(p.b_field);
        if b2_before <= 0.0 {
            continue;
        }
        let x_i = x_e.clamp(ion_floor_eff, 1.0);
        let rate = eta_ad.max(0.0) * (1.0 / x_i - 1.0).max(0.0);
        let damping = (-rate * dt).exp().clamp(0.0, 1.0);
        p.b_field *= damping;
        let b2_after = p.b_field.dot(p.b_field);
        let dissipated = 0.5 * (b2_before - b2_after).max(0.0);
        p.internal_energy += heat_eff * dissipated / p.mass.max(1e-30);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, Vec3};

    fn gas(u: f64, dust: f64) -> Particle {
        let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), u, 0.1);
        p.b_field = Vec3::new(1.0, 0.0, 0.0);
        p.dust_to_gas = dust;
        p
    }

    #[test]
    fn ionization_proxy_is_lower_with_dust() {
        let clean = gas(1.0, 0.0);
        let dusty = gas(1.0, 0.02);
        assert!(
            ionization_fraction_proxy(&dusty, 1e-4, 1.0)
                < ionization_fraction_proxy(&clean, 1e-4, 1.0)
        );
    }

    #[test]
    fn ambipolar_diffusion_reduces_b_and_heats() {
        let mut particles = vec![gas(0.01, 0.02)];
        let b0 = particles[0].b_field.norm();
        let u0 = particles[0].internal_energy;
        apply_ambipolar_diffusion(&mut particles, 0.1, 1e-4, 1.0, 5.0 / 3.0, 1.0);
        assert!(particles[0].b_field.norm() < b0);
        assert!(particles[0].internal_energy > u0);
    }

    fn gas_with_velocity(u: f64, vx: f64, vy: f64, vz: f64, h: f64) -> Particle {
        let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::new(vx, vy, vz), u, h);
        p.b_field = Vec3::new(1.0, 0.0, 0.0);
        p
    }

    #[test]
    fn hall_drift_conserves_b_magnitude() {
        let mut p = gas_with_velocity(1.0, 0.0, 1.0, 0.0, 0.1);
        let b0 = p.b_field.norm();
        hall_drift_particle(&mut p, 0.5, 1.0);
        let b1 = p.b_field.norm();
        assert!((b1 - b0).abs() < 1e-12, "|B| changed: {b0} → {b1}");
    }

    #[test]
    fn hall_drift_rotates_b_direction() {
        let mut p = gas_with_velocity(1.0, 0.0, 1.0, 0.0, 0.1);
        let b_before = p.b_field;
        hall_drift_particle(&mut p, 0.5, 1.0);
        let changed = (p.b_field.x - b_before.x).abs() > 1e-10
            || (p.b_field.y - b_before.y).abs() > 1e-10
            || (p.b_field.z - b_before.z).abs() > 1e-10;
        assert!(changed, "B direction should have changed under Hall drift");
    }

    #[test]
    fn hall_drift_no_effect_on_dm() {
        let mut p = Particle::new(0, 1.0, Vec3::zero(), Vec3::new(0.0, 1.0, 0.0));
        p.b_field = Vec3::new(1.0, 0.0, 0.0);
        let b_before = p.b_field;
        hall_drift_particle(&mut p, 0.5, 1.0);
        assert_eq!(p.b_field.x, b_before.x);
        assert_eq!(p.b_field.y, b_before.y);
        assert_eq!(p.b_field.z, b_before.z);
    }

    #[test]
    fn hall_drift_no_effect_with_zero_eta() {
        let mut p = gas_with_velocity(1.0, 0.0, 1.0, 0.0, 0.1);
        let b_before = p.b_field;
        apply_hall_drift(std::slice::from_mut(&mut p), 0.0, 1.0);
        assert_eq!(p.b_field.x, b_before.x);
    }

    #[test]
    fn hall_drift_no_effect_with_parallel_v_b() {
        let mut p = gas_with_velocity(1.0, 1.0, 0.0, 0.0, 0.1);
        let b_before = p.b_field;
        hall_drift_particle(&mut p, 0.5, 1.0);
        // v × B = 0 when v ∥ B → no rotation
        assert!((p.b_field.x - b_before.x).abs() < 1e-12);
        assert!((p.b_field.y - b_before.y).abs() < 1e-12);
    }

    // ── Ohmic diffusion tests (Phase 187) ─────────────────────────────────

    fn gas_for_ohmic(b_x: f64, h: f64) -> Particle {
        let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 0.0, h);
        p.b_field = Vec3::new(b_x, 0.0, 0.0);
        p
    }

    #[test]
    fn ohmic_diffusion_reduces_b() {
        let mut particles = vec![gas_for_ohmic(1.0, 0.1)];
        let b0 = particles[0].b_field.norm();
        apply_ohmic_diffusion(&mut particles, 0.1, 5.0 / 3.0, 1.0);
        assert!(
            particles[0].b_field.norm() < b0,
            "|B| should decrease under Ohmic diffusion"
        );
    }

    #[test]
    fn ohmic_diffusion_heats_gas() {
        let mut particles = vec![gas_for_ohmic(1.0, 0.1)];
        let u0 = particles[0].internal_energy;
        apply_ohmic_diffusion(&mut particles, 0.1, 5.0 / 3.0, 1.0);
        assert!(
            particles[0].internal_energy > u0,
            "internal energy should increase (dissipation → heat)"
        );
    }

    #[test]
    fn ohmic_diffusion_energy_conservation() {
        // Total energy (magnetic + thermal) should be conserved within numerical precision.
        let mut particles = vec![gas_for_ohmic(2.0, 0.1)];
        let b2_before = particles[0].b_field.dot(particles[0].b_field);
        let u_before = particles[0].internal_energy;
        let gamma = 5.0 / 3.0;
        apply_ohmic_diffusion(&mut particles, 0.05, gamma, 0.5);
        let b2_after = particles[0].b_field.dot(particles[0].b_field);
        let u_after = particles[0].internal_energy;
        // ΔU_thermal = heat_eff × ΔB²/2 / mass, so magnetic + (thermal/heat_eff) should be conserved
        let heat_eff = gamma - 1.0;
        let mag_before = b2_before / 2.0;
        let mag_after = b2_after / 2.0;
        let thermal_before = u_before / heat_eff;
        let thermal_after = u_after / heat_eff;
        let total_before = mag_before + thermal_before;
        let total_after = mag_after + thermal_after;
        let rel_err = (total_after - total_before).abs() / total_before.max(1e-30);
        assert!(
            rel_err < 1e-12,
            "energy conservation violated: rel_err = {rel_err}"
        );
    }

    #[test]
    fn ohmic_diffusion_no_effect_with_zero_eta() {
        let mut particles = vec![gas_for_ohmic(1.5, 0.1)];
        let b_before = particles[0].b_field;
        apply_ohmic_diffusion(&mut particles, 0.0, 5.0 / 3.0, 1.0);
        assert_eq!(particles[0].b_field.x, b_before.x);
    }

    #[test]
    fn ohmic_diffusion_no_effect_on_dm() {
        let mut p = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero());
        p.b_field = Vec3::new(1.0, 0.0, 0.0);
        let b_before = p.b_field;
        apply_ohmic_diffusion(std::slice::from_mut(&mut p), 0.5, 5.0 / 3.0, 1.0);
        assert_eq!(p.b_field.x, b_before.x);
    }

    // ── Chemistry-coupled ambipolar tests (Phase 187) ──────────────────────

    #[test]
    fn chem_coupled_ambipolar_high_ionization_gives_less_damping() {
        // Highly ionized: x_e ≈ 1 → rate ≈ 0 → damping ≈ 1 → B barely changes.
        let mut low = vec![gas(0.01, 0.0)];
        let mut high = vec![gas(0.01, 0.0)];
        // Low ionization: x_e = 0.001
        let ion_fracs_low = vec![0.001_f64];
        // High ionization: x_e = 0.99
        let ion_fracs_high = vec![0.99_f64];
        let b_low_before = low[0].b_field.norm();
        let b_high_before = high[0].b_field.norm();
        apply_ambipolar_diffusion_with_chem(&mut low, &ion_fracs_low, 0.1, 1e-6, 5.0 / 3.0, 1.0);
        apply_ambipolar_diffusion_with_chem(&mut high, &ion_fracs_high, 0.1, 1e-6, 5.0 / 3.0, 1.0);
        let b_low_after = low[0].b_field.norm();
        let b_high_after = high[0].b_field.norm();
        // Low ionization → more damping → |B| decreases more
        assert!(b_low_after < b_high_after, "low x_e should damp B more");
        // Both should be ≤ initial
        assert!(b_low_after <= b_low_before + 1e-15);
        assert!(b_high_after <= b_high_before + 1e-15);
    }

    #[test]
    fn chem_coupled_ambipolar_matches_proxy_at_full_ionization() {
        // When x_e = 1.0, 1/x_i - 1 = 0 → rate = 0 → no damping.
        let mut particles = vec![gas(1.0, 0.0)];
        let b_before = particles[0].b_field.norm();
        let ion_fracs = vec![1.0_f64];
        apply_ambipolar_diffusion_with_chem(
            &mut particles,
            &ion_fracs,
            10.0,
            1e-6,
            5.0 / 3.0,
            100.0,
        );
        let b_after = particles[0].b_field.norm();
        assert!(
            (b_after - b_before).abs() < 1e-12,
            "fully ionized gas should have no ambipolar damping"
        );
    }
}
