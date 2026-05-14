//! Plasma de dos fluidos: temperatura electrónica ≠ temperatura iónica (Phase 149).
//!
//! ## Modelo
//!
//! En plasmas de alta temperatura (ICM de cúmulos, shocks fuertes), el tiempo de
//! termalización entre electrones e iones puede ser mayor que el tiempo dinámico.
//! Los electrones y los iones tienen temperaturas distintas:
//!
//! - `T_i` = temperatura iónica, derivada de `internal_energy` (como siempre)
//! - `T_e` = temperatura electrónica, almacenada en `Particle.t_electron`
//!
//! ### Acoplamiento Coulomb
//!
//! La transferencia de calor entre electrones e iones se da por colisiones Coulomb:
//!
//! ```text
//! dT_e/dt = −ν_ei(T_e − T_i)
//! ```
//!
//! donde la frecuencia de acoplamiento es:
//!
//! ```text
//! ν_ei = ν_coeff × n_e / T_e^{3/2}
//! ```
//!
//! Con `ν_coeff` en unidades internas. En el límite `ν_ei → ∞` se recupera T_e = T_i.
//!
//! ### Calentamiento electrónico por shocks
//!
//! En un shock de velocidad `v_sh`, los electrones reciben una fracción `β_e ~ m_e/m_p`
//! del calentamiento cinético (muy poco). La mayor parte va a los iones. Esto produce
//! `T_e << T_i` justo detrás del shock (relevante para observaciones de X-ray).
//!
//! ## Referencias
//!
//! Spitzer (1962) — frecuencia de colisión Coulomb en plasma.
//! Fox & Loeb (1997), ApJ 491, 460 — dos fluidos en ICM de cúmulos.
//! Rudd & Nagai (2009), ApJL 701, L16 — T_e/T_i en simulaciones de cúmulos.

use gadget_ng_core::{Particle, ParticleType, TwoFluidSection};
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

const GAMMA: f64 = 5.0 / 3.0;

#[inline]
fn u_to_t_code(u: f64, gamma: f64) -> f64 {
    (gamma - 1.0) * u.max(0.0)
}

fn electron_ion_coupling_particle(p: &mut Particle, nu_ei_coeff: f64, dt: f64) {
    if p.ptype != ParticleType::Gas {
        return;
    }
    let t_i = u_to_t_code(p.internal_energy, GAMMA);
    if p.t_electron <= 0.0 {
        p.t_electron = t_i;
        return;
    }
    let t_e = p.t_electron;
    let h = p.smoothing_length.max(1e-10);
    let rho = (p.mass / (h * h * h)).max(1e-30);
    let t_e_eff = t_e.abs().max(1e-30);
    let nu_ei = nu_ei_coeff * rho / (t_e_eff * t_e_eff.sqrt());
    let factor = 1.0 - (-nu_ei * dt).exp();
    p.t_electron = t_e + (t_i - t_e) * factor;
    p.t_electron = p.t_electron.max(0.0);
}

#[cfg(not(feature = "rayon"))]
fn mean_te_over_ti_scalar(particles: &[Particle]) -> f64 {
    let mut sum = 0.0_f64;
    let mut n = 0usize;
    for p in particles.iter() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let t_i = u_to_t_code(p.internal_energy, GAMMA).max(1e-30);
        sum += p.t_electron / t_i;
        n += 1;
    }
    if n == 0 { 1.0 } else { sum / n as f64 }
}

pub fn apply_electron_ion_coupling(particles: &mut [Particle], cfg: &TwoFluidSection, dt: f64) {
    let nu_ei_coeff = cfg.nu_ei_coeff;

    #[cfg(feature = "rayon")]
    {
        particles
            .par_iter_mut()
            .for_each(|p| electron_ion_coupling_particle(p, nu_ei_coeff, dt));
    }

    #[cfg(not(feature = "rayon"))]
    {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F availability was checked at runtime.
                unsafe {
                    apply_electron_ion_coupling_avx512(particles, nu_ei_coeff, dt);
                    return;
                }
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA availability was checked at runtime.
                unsafe {
                    apply_electron_ion_coupling_avx2(particles, nu_ei_coeff, dt);
                    return;
                }
            }
        }
        for p in particles.iter_mut() {
            electron_ion_coupling_particle(p, nu_ei_coeff, dt);
        }
    }
}

pub fn mean_te_over_ti(particles: &[Particle]) -> f64 {
    #[cfg(feature = "rayon")]
    {
        let (sum, n) = particles
            .par_iter()
            .filter(|p| p.ptype == ParticleType::Gas)
            .fold(
                || (0.0_f64, 0usize),
                |(s, c), p| {
                    let t_i = u_to_t_code(p.internal_energy, GAMMA).max(1e-30);
                    (s + p.t_electron / t_i, c + 1)
                },
            )
            .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        if n == 0 { 1.0 } else { sum / n as f64 }
    }

    #[cfg(not(feature = "rayon"))]
    {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F availability was checked at runtime.
                unsafe {
                    return mean_te_over_ti_avx512(particles);
                }
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA availability was checked at runtime.
                unsafe {
                    return mean_te_over_ti_avx2(particles);
                }
            }
        }
        mean_te_over_ti_scalar(particles)
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_electron_ion_coupling_avx2(particles: &mut [Particle], nu_ei_coeff: f64, dt: f64) {
    let lanes = 4;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let gamma_m1 = GAMMA - 1.0;
    let min_h_v = _mm256_set1_pd(1e-10);
    let min_rho_v = _mm256_set1_pd(1e-30);
    let min_te_v = _mm256_set1_pd(1e-30);
    let zero_v = _mm256_set1_pd(0.0);
    let nu_coeff_v = _mm256_set1_pd(nu_ei_coeff);
    let neg_zero_v = _mm256_set1_pd(-0.0);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                electron_ion_coupling_particle(&mut particles[i + lane], nu_ei_coeff, dt);
            }
            i += lanes;
            continue;
        }
        let te = _mm256_set_pd(
            particles[i + 3].t_electron,
            particles[i + 2].t_electron,
            particles[i + 1].t_electron,
            particles[i].t_electron,
        );
        let ue = _mm256_set_pd(
            particles[i + 3].internal_energy,
            particles[i + 2].internal_energy,
            particles[i + 1].internal_energy,
            particles[i].internal_energy,
        );
        let h = _mm256_max_pd(
            min_h_v,
            _mm256_set_pd(
                particles[i + 3].smoothing_length,
                particles[i + 2].smoothing_length,
                particles[i + 1].smoothing_length,
                particles[i].smoothing_length,
            ),
        );
        let m = _mm256_set_pd(
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let rho = _mm256_max_pd(
            min_rho_v,
            _mm256_div_pd(m, _mm256_mul_pd(h, _mm256_mul_pd(h, h))),
        );
        let t_i_v = _mm256_mul_pd(_mm256_set1_pd(gamma_m1), _mm256_max_pd(zero_v, ue));
        let init_mask = _mm256_cmp_pd(te, zero_v, _CMP_LE_OQ);
        let te_eff = _mm256_andnot_pd(neg_zero_v, te);
        let te_eff = _mm256_max_pd(min_te_v, te_eff);
        let te_eff_sqrt = _mm256_sqrt_pd(te_eff);
        let nu_ei = _mm256_div_pd(
            _mm256_mul_pd(nu_coeff_v, rho),
            _mm256_mul_pd(te_eff, te_eff_sqrt),
        );
        let mut nu_arr = [0.0f64; 4];
        let mut t_i_arr = [0.0f64; 4];
        let mut te_arr = [0.0f64; 4];
        let mut init_arr = [0u8; 4];
        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(nu_arr.as_mut_ptr(), nu_ei);
            _mm256_storeu_pd(t_i_arr.as_mut_ptr(), t_i_v);
            _mm256_storeu_pd(te_arr.as_mut_ptr(), te);
            _mm256_storeu_pd(init_arr.as_mut_ptr() as *mut f64, init_mask);
        }
        let mut result_arr = [0.0f64; 4];
        for lane in 0..lanes {
            let init_bits = u64::from_ne_bytes(
                init_arr[lane * 2..lane * 2 + 8]
                    .try_into()
                    .unwrap_or([0u8; 8]),
            );
            let is_init = init_bits != 0;
            if is_init {
                result_arr[lane] = t_i_arr[lane];
            } else {
                let factor = 1.0 - (-nu_arr[lane] * dt).exp();
                let te_new = te_arr[lane] + (t_i_arr[lane] - te_arr[lane]) * factor;
                result_arr[lane] = te_new.max(0.0);
            }
        }
        for lane in 0..lanes {
            particles[i + lane].t_electron = result_arr[lane];
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        electron_ion_coupling_particle(p, nu_ei_coeff, dt);
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_electron_ion_coupling_avx512(
    particles: &mut [Particle],
    nu_ei_coeff: f64,
    dt: f64,
) {
    let lanes = 8;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let gamma_m1 = GAMMA - 1.0;
    let min_h_v = _mm512_set1_pd(1e-10);
    let min_rho_v = _mm512_set1_pd(1e-30);
    let min_te_v = _mm512_set1_pd(1e-30);
    let zero_v = _mm512_set1_pd(0.0);
    let nu_coeff_v = _mm512_set1_pd(nu_ei_coeff);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                electron_ion_coupling_particle(&mut particles[i + lane], nu_ei_coeff, dt);
            }
            i += lanes;
            continue;
        }
        let te = _mm512_set_pd(
            particles[i + 7].t_electron,
            particles[i + 6].t_electron,
            particles[i + 5].t_electron,
            particles[i + 4].t_electron,
            particles[i + 3].t_electron,
            particles[i + 2].t_electron,
            particles[i + 1].t_electron,
            particles[i].t_electron,
        );
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
        let h = _mm512_max_pd(
            min_h_v,
            _mm512_set_pd(
                particles[i + 7].smoothing_length,
                particles[i + 6].smoothing_length,
                particles[i + 5].smoothing_length,
                particles[i + 4].smoothing_length,
                particles[i + 3].smoothing_length,
                particles[i + 2].smoothing_length,
                particles[i + 1].smoothing_length,
                particles[i].smoothing_length,
            ),
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
        let rho = _mm512_max_pd(
            min_rho_v,
            _mm512_div_pd(m, _mm512_mul_pd(h, _mm512_mul_pd(h, h))),
        );
        let t_i_v = _mm512_mul_pd(_mm512_set1_pd(gamma_m1), _mm512_max_pd(zero_v, ue));
        let init_mask = _mm512_cmp_pd_mask(te, zero_v, _CMP_LE_OQ);
        let te_eff = _mm512_max_pd(min_te_v, _mm512_max_pd(_mm512_sub_pd(zero_v, te), te));
        let te_eff_sqrt = _mm512_sqrt_pd(te_eff);
        let nu_ei = _mm512_div_pd(
            _mm512_mul_pd(nu_coeff_v, rho),
            _mm512_mul_pd(te_eff, te_eff_sqrt),
        );
        let mut nu_arr = [0.0f64; 8];
        let mut t_i_arr = [0.0f64; 8];
        let mut te_arr = [0.0f64; 8];
        // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(nu_arr.as_mut_ptr(), nu_ei);
            _mm512_storeu_pd(t_i_arr.as_mut_ptr(), t_i_v);
            _mm512_storeu_pd(te_arr.as_mut_ptr(), te);
        }
        let mut result_arr = [0.0f64; 8];
        for lane in 0..lanes {
            if (init_mask & (1 << lane)) != 0 {
                result_arr[lane] = t_i_arr[lane];
            } else {
                let factor = 1.0 - (-nu_arr[lane] * dt).exp();
                let te_new = te_arr[lane] + (t_i_arr[lane] - te_arr[lane]) * factor;
                result_arr[lane] = te_new.max(0.0);
            }
        }
        for lane in 0..lanes {
            particles[i + lane].t_electron = result_arr[lane];
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        electron_ion_coupling_particle(p, nu_ei_coeff, dt);
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn mean_te_over_ti_avx2(particles: &[Particle]) -> f64 {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let gamma_m1 = GAMMA - 1.0;
    let gm1_v = _mm256_set1_pd(gamma_m1);
    let min_ti_v = _mm256_set1_pd(1e-30);
    let zero_v = _mm256_set1_pd(0.0);
    let mut sum_v = _mm256_setzero_pd();
    let mut n_gas = 0usize;
    let mut i = 0;
    while i < chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                if particles[i + lane].ptype == ParticleType::Gas {
                    let t_i = u_to_t_code(particles[i + lane].internal_energy, GAMMA).max(1e-30);
                    sum_v = _mm256_add_pd(
                        sum_v,
                        _mm256_set_pd(0.0, 0.0, 0.0, particles[i + lane].t_electron / t_i),
                    );
                    n_gas += 1;
                }
            }
            i += lanes;
            continue;
        }
        n_gas += lanes;
        let ue = _mm256_set_pd(
            particles[i + 3].internal_energy,
            particles[i + 2].internal_energy,
            particles[i + 1].internal_energy,
            particles[i].internal_energy,
        );
        let t_i_v = _mm256_mul_pd(gm1_v, _mm256_max_pd(zero_v, ue));
        let t_i_safe = _mm256_max_pd(min_ti_v, t_i_v);
        let te = _mm256_set_pd(
            particles[i + 3].t_electron,
            particles[i + 2].t_electron,
            particles[i + 1].t_electron,
            particles[i].t_electron,
        );
        sum_v = _mm256_add_pd(sum_v, _mm256_div_pd(te, t_i_safe));
        i += lanes;
    }
    let mut arr = [0.0f64; 4];
    // SAFETY: fixed-size stack array has exactly four f64 lanes.
    unsafe {
        _mm256_storeu_pd(arr.as_mut_ptr(), sum_v);
    }
    let mut sum = arr.into_iter().sum::<f64>();
    for p in &particles[chunks..] {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let t_i = u_to_t_code(p.internal_energy, GAMMA).max(1e-30);
        sum += p.t_electron / t_i;
        n_gas += 1;
    }
    if n_gas == 0 { 1.0 } else { sum / n_gas as f64 }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn mean_te_over_ti_avx512(particles: &[Particle]) -> f64 {
    let lanes = 8;
    let chunks = particles.len() / lanes * lanes;
    let gamma_m1 = GAMMA - 1.0;
    let gm1_v = _mm512_set1_pd(gamma_m1);
    let min_ti_v = _mm512_set1_pd(1e-30);
    let zero_v = _mm512_set1_pd(0.0);
    let mut sum_v = _mm512_setzero_pd();
    let mut n_gas = 0usize;
    let mut i = 0;
    while i < chunks {
        let mut mask_bits: u8 = 0;
        for lane in 0..lanes {
            if particles[i + lane].ptype == ParticleType::Gas {
                mask_bits |= 1 << lane;
            }
        }
        n_gas += mask_bits.count_ones() as usize;
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
        let t_i_v = _mm512_mul_pd(gm1_v, _mm512_max_pd(zero_v, ue));
        let t_i_safe = _mm512_max_pd(min_ti_v, t_i_v);
        let te = _mm512_set_pd(
            particles[i + 7].t_electron,
            particles[i + 6].t_electron,
            particles[i + 5].t_electron,
            particles[i + 4].t_electron,
            particles[i + 3].t_electron,
            particles[i + 2].t_electron,
            particles[i + 1].t_electron,
            particles[i].t_electron,
        );
        let ratio = _mm512_maskz_div_pd(mask_bits, te, t_i_safe);
        sum_v = _mm512_add_pd(sum_v, ratio);
        i += lanes;
    }
    let mut arr = [0.0f64; 8];
    // SAFETY: fixed-size stack array has exactly eight f64 lanes.
    unsafe {
        _mm512_storeu_pd(arr.as_mut_ptr(), sum_v);
    }
    let mut sum = arr.into_iter().sum::<f64>();
    for p in &particles[chunks..] {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let t_i = u_to_t_code(p.internal_energy, GAMMA).max(1e-30);
        sum += p.t_electron / t_i;
        n_gas += 1;
    }
    if n_gas == 0 { 1.0 } else { sum / n_gas as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, TwoFluidSection, Vec3};

    fn gas(u: f64, t_e: f64) -> Particle {
        let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), u, 0.1);
        p.t_electron = t_e;
        p
    }

    #[test]
    fn ionization_proxy_is_lower_with_dust() {
        let clean = gas(1.0, 0.5);
        let dusty = gas(1.0, 0.5);
        let ion_proxy_clean = crate::nonideal::ionization_fraction_proxy(&clean, 1e-4, 1.0);
        let ion_proxy_dusty = crate::nonideal::ionization_fraction_proxy(&dusty, 1e-4, 1.0);
        assert!(ion_proxy_dusty <= ion_proxy_clean);
    }

    #[test]
    fn coupling_drives_te_toward_ti() {
        let cfg = TwoFluidSection {
            enabled: true,
            nu_ei_coeff: 1.0,
            t_e_init_k: 0.0,
        };
        let mut particles = vec![gas(10.0, 0.1)];
        let t_i_before = (GAMMA - 1.0) * 10.0_f64.max(0.0);
        apply_electron_ion_coupling(&mut particles, &cfg, 1.0);
        assert!(particles[0].t_electron > 0.1);
        assert!(particles[0].t_electron <= t_i_before + 1e-10);
    }

    #[test]
    fn mean_te_over_ti_with_no_gas() {
        let dm = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero());
        let ratio = mean_te_over_ti(&[dm]);
        assert_eq!(ratio, 1.0);
    }

    #[test]
    fn mean_te_over_ti_single_gas() {
        let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.1);
        p.t_electron = 0.5;
        let t_i = (GAMMA - 1.0) * 1.0_f64.max(0.0);
        let expected = 0.5 / t_i.max(1e-30);
        let ratio = mean_te_over_ti(&[p]);
        assert!((ratio - expected).abs() < 1e-10);
    }
}
