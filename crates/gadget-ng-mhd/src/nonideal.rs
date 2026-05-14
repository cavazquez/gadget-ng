//! MHD no ideal: difusión ambipolar dependiente de ionización (Phase 194).
//!
//! Modelo reducido: en gas poco ionizado, las partículas neutras desacoplan el
//! campo magnético del fluido. Representamos esto como una difusión local que
//! amortigua `B` con una tasa proporcional a `eta_ad / x_i`, suavizada por un
//! proxy de ionización térmica y por el contenido de polvo.

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
}
