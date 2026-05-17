//! Esquema de Dedner div-B cleaning (Phase 125).
//!
//! ## Formulación
//!
//! El esquema de Dedner et al. (2002) introduce un campo escalar `ψ` que
//! transporta y disipa el error de divergencia `∇·B`:
//!
//! ```text
//! ∂B/∂t + ∇ψ = 0
//! ∂ψ/∂t + c_h² ∇·B = −c_r ψ
//! ```
//!
//! donde:
//! - `c_h` es la velocidad de propagación de las ondas de limpieza (típicamente la
//!   velocidad de Alfvén máxima en la caja).
//! - `c_r` es la tasa de amortiguamiento (control la disipación de `ψ`).
//!
//! En la integración explícita de Euler:
//!
//! ```text
//! ψ_new = ψ × exp(−c_r × dt)  [disipación]
//! B_new  = B − ∇ψ × dt         [corrección del campo]
//! ```
//!
//! ## Referencia
//!
//! Dedner et al. (2002), J. Comput. Phys. 175, 645–673.
//! Tricco & Price (2012), J. Comput. Phys. 231, 7214.

use gadget_ng_core::{Particle, ParticleType, Vec3};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "bench-all-dedner-paths")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum DednerBenchSimdTier {
    Scalar,
    Avx2,
    Avx512,
}

#[cfg(feature = "bench-all-dedner-paths")]
fn dedner_bench_simd_tier_from_env() -> Option<DednerBenchSimdTier> {
    match std::env::var("GADGET_NG_MHD_BENCH_SIMD_TIER") {
        Ok(s) if s == "scalar" => Some(DednerBenchSimdTier::Scalar),
        Ok(s) if s == "avx2" => Some(DednerBenchSimdTier::Avx2),
        Ok(s) if s == "avx512" => Some(DednerBenchSimdTier::Avx512),
        _ => None,
    }
}

/// Gradiente SPH del campo escalar ψ para un par (i, j).
fn grad_w_scalar(r_vec: Vec3, h: f64) -> Vec3 {
    let r2 = r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z;
    let r = r2.sqrt();
    if r < 1e-10 || h <= 0.0 {
        return Vec3::zero();
    }
    let q = r / h;
    let dw_dr = if q < 1.0 {
        let norm = 8.0 / (std::f64::consts::PI * h.powi(3));
        norm * (-6.0 * q + 9.0 * q * q) / h
    } else if q < 2.0 {
        let norm = 8.0 / (std::f64::consts::PI * h.powi(3));
        norm * (-6.0 * (2.0 - q).powi(2)) / (4.0 * h)
    } else {
        0.0
    };
    Vec3 {
        x: dw_dr * r_vec.x / r,
        y: dw_dr * r_vec.y / r,
        z: dw_dr * r_vec.z / r,
    }
}

#[cfg(any(
    not(feature = "rayon"),
    feature = "bench-all-dedner-paths",
    all(feature = "rayon", feature = "simd")
))]
fn dedner_pair_increment(
    particles: &[Particle],
    rho: &[f64],
    i: usize,
    j: usize,
    b_i: Vec3,
    psi_i: f64,
) -> (f64, Vec3) {
    if j == i || particles[j].ptype != ParticleType::Gas {
        return (0.0, Vec3::zero());
    }
    let b_j = particles[j].b_field;
    let psi_j = particles[j].psi_div;
    let h_ij = 0.5 * (particles[i].smoothing_length + particles[j].smoothing_length).max(1e-10);
    let r_ij = Vec3 {
        x: particles[j].position.x - particles[i].position.x,
        y: particles[j].position.y - particles[i].position.y,
        z: particles[j].position.z - particles[i].position.z,
    };
    let grad_w = grad_w_scalar(r_ij, h_ij);
    let factor = particles[j].mass / rho[j];
    let db = Vec3 {
        x: b_j.x - b_i.x,
        y: b_j.y - b_i.y,
        z: b_j.z - b_i.z,
    };
    let div_inc = factor * (db.x * grad_w.x + db.y * grad_w.y + db.z * grad_w.z);
    let dpsi = psi_j - psi_i;
    let gp = Vec3::new(
        factor * dpsi * grad_w.x,
        factor * dpsi * grad_w.y,
        factor * dpsi * grad_w.z,
    );
    (div_inc, gp)
}

#[cfg(all(
    feature = "rayon",
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64"),
))]
fn dedner_sum_for_i_scalar_loop(particles: &[Particle], rho: &[f64], i: usize) -> (f64, Vec3) {
    if particles[i].ptype != ParticleType::Gas {
        return (0.0, Vec3::zero());
    }
    let b_i = particles[i].b_field;
    let psi_i = particles[i].psi_div;
    let mut div_acc = 0.0_f64;
    let mut gx_acc = 0.0_f64;
    let mut gy_acc = 0.0_f64;
    let mut gz_acc = 0.0_f64;
    for j in 0..particles.len() {
        let (d, g) = dedner_pair_increment(particles, rho, i, j, b_i, psi_i);
        div_acc += d;
        gx_acc += g.x;
        gy_acc += g.y;
        gz_acc += g.z;
    }
    (div_acc, Vec3::new(gx_acc, gy_acc, gz_acc))
}

/// Ruta serial por pares (densidad + acumulación i–j) usada sin `rayon`, en benches
/// `bench-all-dedner-paths`, o en tests (`cfg(test)`).
fn dedner_pairwise_accumulate_scalar(
    particles: &[Particle],
    rho: &[f64],
    div_b: &mut [f64],
    grad_psi: &mut [Vec3],
) {
    let n = particles.len();
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let b_i = particles[i].b_field;
        let psi_i = particles[i].psi_div;
        for j in 0..n {
            let (d, g) = dedner_pair_increment(particles, rho, i, j, b_i, psi_i);
            div_b[i] += d;
            grad_psi[i].x += g.x;
            grad_psi[i].y += g.y;
            grad_psi[i].z += g.z;
        }
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn dedner_pairwise_accumulate_dispatch(
    particles: &[Particle],
    rho: &[f64],
    div_b: &mut [f64],
    grad_psi: &mut [Vec3],
) {
    #[cfg(feature = "bench-all-dedner-paths")]
    if let Some(tier) = dedner_bench_simd_tier_from_env() {
        match tier {
            DednerBenchSimdTier::Scalar => {
                dedner_pairwise_accumulate_scalar(particles, rho, div_b, grad_psi);
                return;
            }
            DednerBenchSimdTier::Avx2 => {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: AVX2+FMA forced by bench tier after runtime checks.
                    unsafe {
                        dedner_pairwise_accumulate_avx2(particles, rho, div_b, grad_psi);
                    }
                } else {
                    dedner_pairwise_accumulate_scalar(particles, rho, div_b, grad_psi);
                }
                return;
            }
            DednerBenchSimdTier::Avx512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx512f") {
                        // SAFETY: AVX-512F forced by bench tier after runtime check.
                        unsafe {
                            dedner_pairwise_accumulate_avx512(particles, rho, div_b, grad_psi);
                        }
                    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        // SAFETY: AVX2+FMA fallback when AVX-512 is unavailable.
                        unsafe {
                            dedner_pairwise_accumulate_avx2(particles, rho, div_b, grad_psi);
                        }
                    } else {
                        dedner_pairwise_accumulate_scalar(particles, rho, div_b, grad_psi);
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        // SAFETY: AVX2+FMA on non-x86_64 SIMD targets.
                        unsafe {
                            dedner_pairwise_accumulate_avx2(particles, rho, div_b, grad_psi);
                        }
                    } else {
                        dedner_pairwise_accumulate_scalar(particles, rho, div_b, grad_psi);
                    }
                }
                return;
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: `avx512f` was detected immediately above.
            unsafe {
                dedner_pairwise_accumulate_avx512(particles, rho, div_b, grad_psi);
            }
            return;
        }
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: `avx2` and `fma` were detected immediately above.
        unsafe {
            dedner_pairwise_accumulate_avx2(particles, rho, div_b, grad_psi);
        }
        return;
    }
    dedner_pairwise_accumulate_scalar(particles, rho, div_b, grad_psi);
}

fn dedner_pairwise_accumulate(
    particles: &[Particle],
    rho: &[f64],
    div_b: &mut [f64],
    grad_psi: &mut [Vec3],
) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        dedner_pairwise_accumulate_dispatch(particles, rho, div_b, grad_psi);
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
    {
        dedner_pairwise_accumulate_scalar(particles, rho, div_b, grad_psi);
    }
}

#[cfg(any(
    not(feature = "rayon"),
    feature = "bench-all-dedner-paths",
    all(feature = "rayon", feature = "simd")
))]
fn dedner_apply_final_update(
    particles: &mut [Particle],
    div_b: &[f64],
    grad_psi: &[Vec3],
    c_h: f64,
    c_r: f64,
    dt: f64,
) {
    let n = particles.len();
    let decay = (-c_r * dt).exp();

    #[cfg(all(
        any(
            not(feature = "rayon"),
            feature = "bench-all-dedner-paths",
            all(feature = "rayon", feature = "simd")
        ),
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        #[cfg(feature = "bench-all-dedner-paths")]
        if let Some(tier) = dedner_bench_simd_tier_from_env() {
            match tier {
                DednerBenchSimdTier::Avx2 => {
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        // SAFETY: AVX2+FMA forced by bench tier after runtime checks.
                        unsafe {
                            dedner_cleaning_update_avx2(
                                particles,
                                div_b,
                                grad_psi,
                                c_h * c_h * dt,
                                decay,
                                dt,
                            );
                        }
                        return;
                    }
                }
                DednerBenchSimdTier::Avx512 => {
                    #[cfg(target_arch = "x86_64")]
                    if is_x86_feature_detected!("avx512f") {
                        // SAFETY: AVX-512F forced by bench tier after runtime check.
                        unsafe {
                            dedner_cleaning_update_avx512(
                                particles,
                                div_b,
                                grad_psi,
                                c_h * c_h * dt,
                                decay,
                                dt,
                            );
                        }
                        return;
                    }
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        // SAFETY: AVX2+FMA fallback when AVX-512 is unavailable.
                        unsafe {
                            dedner_cleaning_update_avx2(
                                particles,
                                div_b,
                                grad_psi,
                                c_h * c_h * dt,
                                decay,
                                dt,
                            );
                        }
                        return;
                    }
                }
                DednerBenchSimdTier::Scalar => {}
            }
        }
        #[cfg(feature = "bench-all-dedner-paths")]
        let skip_auto_simd_updates = matches!(
            dedner_bench_simd_tier_from_env(),
            Some(DednerBenchSimdTier::Scalar)
        );
        #[cfg(not(feature = "bench-all-dedner-paths"))]
        let skip_auto_simd_updates = false;

        if !skip_auto_simd_updates {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F availability was checked at runtime.
                unsafe {
                    dedner_cleaning_update_avx512(
                        particles,
                        div_b,
                        grad_psi,
                        c_h * c_h * dt,
                        decay,
                        dt,
                    );
                }
                return;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA availability was checked at runtime.
                unsafe {
                    dedner_cleaning_update_avx2(
                        particles,
                        div_b,
                        grad_psi,
                        c_h * c_h * dt,
                        decay,
                        dt,
                    );
                }
                return;
            }
        }
    }

    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].psi_div = particles[i].psi_div * decay - c_h * c_h * div_b[i] * dt;
            particles[i].b_field.x -= grad_psi[i].x * dt;
            particles[i].b_field.y -= grad_psi[i].y * dt;
            particles[i].b_field.z -= grad_psi[i].z * dt;
        }
    }
}

#[cfg(any(not(feature = "rayon"), feature = "bench-all-dedner-paths", test))]
// Con `rayon`+`simd` en x86 el paso público usa `dedner_cleaning_step_par_simd`; esta
// función solo sirve para benches `bench-all-dedner-paths` y queda referenciada en tests
// vía `cfg(test)` aunque ningún test la llame en esa combinación de features.
#[cfg_attr(
    all(
        feature = "rayon",
        feature = "simd",
        not(feature = "bench-all-dedner-paths")
    ),
    allow(dead_code)
)]
fn dedner_cleaning_step_impl(particles: &mut [Particle], c_h: f64, c_r: f64, dt: f64) {
    let n = particles.len();

    let rho = dedner_density(particles);

    let mut div_b = vec![0.0_f64; n];
    let mut grad_psi = vec![Vec3::zero(); n];

    dedner_pairwise_accumulate(particles, &rho, &mut div_b, &mut grad_psi);

    dedner_apply_final_update(particles, &div_b, &grad_psi, c_h, c_r, dt);
}

#[cfg(any(
    not(feature = "rayon"),
    feature = "bench-all-dedner-paths",
    all(feature = "rayon", feature = "simd")
))]
fn dedner_density(particles: &[Particle]) -> Vec<f64> {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(feature = "bench-all-dedner-paths")]
        if let Some(tier) = dedner_bench_simd_tier_from_env() {
            match tier {
                DednerBenchSimdTier::Scalar => return dedner_density_scalar(particles),
                DednerBenchSimdTier::Avx2 => {
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        // SAFETY: AVX2+FMA forced by bench tier after runtime checks.
                        unsafe {
                            return dedner_density_avx2(particles);
                        }
                    }
                    return dedner_density_scalar(particles);
                }
                DednerBenchSimdTier::Avx512 => {
                    #[cfg(target_arch = "x86_64")]
                    if is_x86_feature_detected!("avx512f") {
                        // SAFETY: AVX-512F forced by bench tier after runtime check.
                        unsafe {
                            return dedner_density_avx512(particles);
                        }
                    }
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        // SAFETY: AVX2+FMA fallback when AVX-512 is unavailable.
                        unsafe {
                            return dedner_density_avx2(particles);
                        }
                    }
                    return dedner_density_scalar(particles);
                }
            }
        }
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime dispatch checked `avx512f` immediately above.
            unsafe {
                return dedner_density_avx512(particles);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime dispatch checked `avx2` and `fma` immediately above.
            unsafe {
                return dedner_density_avx2(particles);
            }
        }
    }

    dedner_density_scalar(particles)
}

#[cfg(any(
    not(feature = "rayon"),
    feature = "bench-all-dedner-paths",
    all(feature = "rayon", feature = "simd")
))]
fn dedner_density_scalar(particles: &[Particle]) -> Vec<f64> {
    particles
        .iter()
        .map(|p| {
            let h = p.smoothing_length.max(1e-10);
            (p.mass / (h * h * h)).max(1e-30)
        })
        .collect()
}

#[cfg(feature = "rayon")]
fn dedner_cleaning_step_par(particles: &mut [Particle], c_h: f64, c_r: f64, dt: f64) {
    let n = particles.len();

    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let h_sml: Vec<f64> = particles
        .iter()
        .map(|p| p.smoothing_length.max(1e-10))
        .collect();
    let rho: Vec<f64> = h_sml
        .iter()
        .zip(mass.iter())
        .map(|(&h, &m)| (m / (h * h * h)).max(1e-30))
        .collect();
    let b_field: Vec<Vec3> = particles.iter().map(|p| p.b_field).collect();
    let psi_div: Vec<f64> = particles.iter().map(|p| p.psi_div).collect();
    let is_gas: Vec<bool> = particles
        .iter()
        .map(|p| p.ptype == ParticleType::Gas)
        .collect();

    let updates: Vec<Option<(f64, Vec3)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !is_gas[i] {
                return None;
            }
            let b_i = b_field[i];
            let psi_i = psi_div[i];
            let mut div_b_i = 0.0_f64;
            let mut grad_psi_i = Vec3::zero();

            for j in 0..n {
                if j == i || !is_gas[j] {
                    continue;
                }
                let b_j = b_field[j];
                let psi_j = psi_div[j];
                let h_ij = 0.5 * (h_sml[i] + h_sml[j]);
                let r_ij = Vec3 {
                    x: pos[j].x - pos[i].x,
                    y: pos[j].y - pos[i].y,
                    z: pos[j].z - pos[i].z,
                };
                let grad_w = grad_w_scalar(r_ij, h_ij);
                let factor = mass[j] / rho[j];

                let db = Vec3 {
                    x: b_j.x - b_i.x,
                    y: b_j.y - b_i.y,
                    z: b_j.z - b_i.z,
                };
                div_b_i += factor * (db.x * grad_w.x + db.y * grad_w.y + db.z * grad_w.z);

                let dpsi = psi_j - psi_i;
                grad_psi_i.x += factor * dpsi * grad_w.x;
                grad_psi_i.y += factor * dpsi * grad_w.y;
                grad_psi_i.z += factor * dpsi * grad_w.z;
            }
            Some((div_b_i, grad_psi_i))
        })
        .collect();

    let decay = (-c_r * dt).exp();
    for (p, update) in particles.iter_mut().zip(updates) {
        if let (true, Some((div_b, grad_psi))) = (p.ptype == ParticleType::Gas, update) {
            p.psi_div = p.psi_div * decay - c_h * c_h * div_b * dt;
            p.b_field.x -= grad_psi.x * dt;
            p.b_field.y -= grad_psi.y * dt;
            p.b_field.z -= grad_psi.z * dt;
        }
    }
}

#[cfg(all(
    feature = "rayon",
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
fn dedner_sum_for_i_dispatch_parallel_cpu(
    particles: &[Particle],
    rho: &[f64],
    i: usize,
) -> (f64, Vec3) {
    #[cfg(feature = "bench-all-dedner-paths")]
    if let Some(tier) = dedner_bench_simd_tier_from_env() {
        match tier {
            DednerBenchSimdTier::Scalar => return dedner_sum_for_i_scalar_loop(particles, rho, i),
            DednerBenchSimdTier::Avx2 => {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: AVX2+FMA forced by bench tier after runtime checks.
                    return unsafe { dedner_sum_for_i_avx2(particles, rho, i) };
                }
                return dedner_sum_for_i_scalar_loop(particles, rho, i);
            }
            DednerBenchSimdTier::Avx512 => {
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx512f") {
                    // SAFETY: AVX-512F forced by bench tier after runtime check.
                    return unsafe { dedner_sum_for_i_avx512(particles, rho, i) };
                }
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: AVX2+FMA fallback when AVX-512 is unavailable.
                    return unsafe { dedner_sum_for_i_avx2(particles, rho, i) };
                }
                return dedner_sum_for_i_scalar_loop(particles, rho, i);
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: `avx512f` was detected immediately above.
        return unsafe { dedner_sum_for_i_avx512(particles, rho, i) };
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: `avx2` and `fma` were detected immediately above.
        return unsafe { dedner_sum_for_i_avx2(particles, rho, i) };
    }
    dedner_sum_for_i_scalar_loop(particles, rho, i)
}

#[cfg(all(
    feature = "rayon",
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
fn dedner_cleaning_step_par_simd(particles: &mut [Particle], c_h: f64, c_r: f64, dt: f64) {
    use rayon::prelude::*;
    let rho = dedner_density(particles);
    let mut div_b = vec![0.0_f64; particles.len()];
    let mut grad_psi = vec![Vec3::zero(); particles.len()];

    let particles_rf: &[Particle] = particles;

    div_b
        .par_iter_mut()
        .zip(grad_psi.par_iter_mut())
        .enumerate()
        .for_each(|(i, (db, gp))| {
            let (d, g) = dedner_sum_for_i_dispatch_parallel_cpu(particles_rf, &rho, i);
            *db = d;
            *gp = g;
        });

    dedner_apply_final_update(particles, &div_b, &grad_psi, c_h, c_r, dt);
}

/// Aplica un paso del esquema de limpieza de Dedner para div-B (Phase 125).
///
/// # Parámetros
///
/// - `particles` — slice mutable de partículas.
/// - `c_h`       — velocidad de las ondas de limpieza (típicamente velocidad de Alfvén máx.).
/// - `c_r`       — tasa de amortiguamiento de ψ (s⁻¹).
/// - `dt`        — paso de tiempo.
///
/// # Algoritmo
///
/// 1. Calcula la divergencia SPH de B para cada partícula: `div_B_i = Σ_j (m_j/ρ_j) (B_j − B_i)·∇W_ij`.
/// 2. Actualiza ψ: `ψ_new = ψ × exp(−c_r × dt) − c_h² × div_B × dt`.
/// 3. Calcula el gradiente SPH de ψ: `∇ψ_i = Σ_j (m_j/ρ_j) (ψ_j − ψ_i) ∇W_ij`.
/// 4. Corrige B: `B_new = B − ∇ψ × dt`.
pub fn dedner_cleaning_step(particles: &mut [Particle], c_h: f64, c_r: f64, dt: f64) {
    #[cfg(feature = "rayon")]
    {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let use_simd_rayon = is_x86_feature_detected!("avx512f")
                || (is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
            if use_simd_rayon {
                dedner_cleaning_step_par_simd(particles, c_h, c_r, dt);
                return;
            }
        }
        dedner_cleaning_step_par(particles, c_h, c_r, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        dedner_cleaning_step_impl(particles, c_h, c_r, dt);
    }
}

/// Variante de backend solo para benchmarks (`feature = "bench-all-dedner-paths"`).
///
/// Permite comparar en un único binario **cinco** backends: CPU serial escalar,
/// Rayon escalar, SIMD+Rayon multihilo (x86 con AVX2+FMA o AVX-512F), y SIMD
/// forzado AVX2 o AVX-512F sin Rayon vía `GADGET_NG_MHD_BENCH_SIMD_TIER` (no usar
/// en producción).
#[cfg(feature = "bench-all-dedner-paths")]
#[derive(Clone, Copy, Debug)]
pub enum DednerCleaningBackend {
    /// `dedner_cleaning_step_impl` con `GADGET_NG_MHD_BENCH_SIMD_TIER=scalar`.
    CpuSinRayonScalar,
    /// Paralelismo exterior por partícula gas (`dedner_cleaning_step_par`).
    CpuRayon,
    /// Rayon + SIMD en x86/x86_64 cuando hay AVX2+FMA o AVX-512F (`dedner_cleaning_step_par_simd`).
    SimdConRayon,
    /// SIMD forzado AVX2+FMA en densidad + pares + actualización final.
    SimdSinRayonAvx2,
    /// SIMD forzado AVX-512F cuando está disponible (si no, cae a AVX2 o escalar).
    SimdSinRayonAvx512,
}

#[cfg(feature = "bench-all-dedner-paths")]
pub fn dedner_cleaning_step_with_backend(
    particles: &mut [Particle],
    c_h: f64,
    c_r: f64,
    dt: f64,
    backend: DednerCleaningBackend,
) {
    use std::env;
    // SAFETY: bench-only API; mutates `GADGET_NG_MHD_BENCH_SIMD_TIER` for tier dispatch.
    // Call from Criterion single-threaded harness only (see `dedner_backend_bench`).
    unsafe {
        match backend {
            DednerCleaningBackend::CpuRayon => {
                env::remove_var("GADGET_NG_MHD_BENCH_SIMD_TIER");
                dedner_cleaning_step_par(particles, c_h, c_r, dt);
            }
            DednerCleaningBackend::SimdConRayon => {
                env::remove_var("GADGET_NG_MHD_BENCH_SIMD_TIER");
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    let use_simd_rayon = is_x86_feature_detected!("avx512f")
                        || (is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
                    if use_simd_rayon {
                        dedner_cleaning_step_par_simd(particles, c_h, c_r, dt);
                    } else {
                        dedner_cleaning_step_par(particles, c_h, c_r, dt);
                    }
                }
                #[cfg(not(all(
                    feature = "simd",
                    any(target_arch = "x86", target_arch = "x86_64")
                )))]
                dedner_cleaning_step_par(particles, c_h, c_r, dt);
            }
            DednerCleaningBackend::CpuSinRayonScalar => {
                env::set_var("GADGET_NG_MHD_BENCH_SIMD_TIER", "scalar");
                dedner_cleaning_step_impl(particles, c_h, c_r, dt);
            }
            DednerCleaningBackend::SimdSinRayonAvx2 => {
                env::set_var("GADGET_NG_MHD_BENCH_SIMD_TIER", "avx2");
                dedner_cleaning_step_impl(particles, c_h, c_r, dt);
            }
            DednerCleaningBackend::SimdSinRayonAvx512 => {
                env::set_var("GADGET_NG_MHD_BENCH_SIMD_TIER", "avx512");
                dedner_cleaning_step_impl(particles, c_h, c_r, dt);
            }
        }
    }
}

#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dedner_density_avx2(particles: &[Particle]) -> Vec<f64> {
    let n = particles.len();
    let mut rho = vec![0.0_f64; n];
    let chunks = n / 4 * 4;
    let min_h = _mm256_set1_pd(1e-10);
    let min_rho = _mm256_set1_pd(1e-30);

    for i in (0..chunks).step_by(4) {
        let mass = _mm256_set_pd(
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let h_raw = _mm256_set_pd(
            particles[i + 3].smoothing_length,
            particles[i + 2].smoothing_length,
            particles[i + 1].smoothing_length,
            particles[i].smoothing_length,
        );
        let h = _mm256_max_pd(h_raw, min_h);
        let h3 = _mm256_mul_pd(_mm256_mul_pd(h, h), h);
        let rho_v = _mm256_max_pd(_mm256_div_pd(mass, h3), min_rho);
        // SAFETY: `rho[i..i+4]` has four contiguous f64 slots for the unaligned store.
        unsafe {
            _mm256_storeu_pd(rho[i..].as_mut_ptr(), rho_v);
        }
    }
    for i in chunks..n {
        let h = particles[i].smoothing_length.max(1e-10);
        rho[i] = (particles[i].mass / (h * h * h)).max(1e-30);
    }
    rho
}

#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dedner_density_avx512(particles: &[Particle]) -> Vec<f64> {
    let n = particles.len();
    let mut rho = vec![0.0_f64; n];
    let chunks = n / 8 * 8;
    let min_h = _mm512_set1_pd(1e-10);
    let min_rho = _mm512_set1_pd(1e-30);

    for i in (0..chunks).step_by(8) {
        let mass = _mm512_set_pd(
            particles[i + 7].mass,
            particles[i + 6].mass,
            particles[i + 5].mass,
            particles[i + 4].mass,
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let h_raw = _mm512_set_pd(
            particles[i + 7].smoothing_length,
            particles[i + 6].smoothing_length,
            particles[i + 5].smoothing_length,
            particles[i + 4].smoothing_length,
            particles[i + 3].smoothing_length,
            particles[i + 2].smoothing_length,
            particles[i + 1].smoothing_length,
            particles[i].smoothing_length,
        );
        let h = _mm512_max_pd(h_raw, min_h);
        let h3 = _mm512_mul_pd(_mm512_mul_pd(h, h), h);
        let rho_v = _mm512_max_pd(_mm512_div_pd(mass, h3), min_rho);
        // SAFETY: `rho[i..i+8]` has eight contiguous f64 slots for the unaligned store.
        unsafe {
            _mm512_storeu_pd(rho[i..].as_mut_ptr(), rho_v);
        }
    }
    for i in chunks..n {
        let h = particles[i].smoothing_length.max(1e-10);
        rho[i] = (particles[i].mass / (h * h * h)).max(1e-30);
    }
    rho
}

#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dedner_cleaning_update_avx2(
    particles: &mut [Particle],
    div_b: &[f64],
    grad_psi: &[Vec3],
    c_h_sq_dt: f64,
    decay: f64,
    dt: f64,
) {
    let lanes = 4;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let c_h_sq_dt_v = _mm256_set1_pd(c_h_sq_dt);
    let decay_v = _mm256_set1_pd(decay);
    let dt_v = _mm256_set1_pd(dt);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                if particles[i + lane].ptype == ParticleType::Gas {
                    particles[i + lane].psi_div =
                        particles[i + lane].psi_div * decay - c_h_sq_dt * div_b[i + lane];
                    particles[i + lane].b_field.x -= grad_psi[i + lane].x * dt;
                    particles[i + lane].b_field.y -= grad_psi[i + lane].y * dt;
                    particles[i + lane].b_field.z -= grad_psi[i + lane].z * dt;
                }
            }
            i += lanes;
            continue;
        }
        let psi = _mm256_set_pd(
            particles[i + 3].psi_div,
            particles[i + 2].psi_div,
            particles[i + 1].psi_div,
            particles[i].psi_div,
        );
        let div_b_v = _mm256_set_pd(div_b[i + 3], div_b[i + 2], div_b[i + 1], div_b[i]);
        let new_psi = _mm256_sub_pd(
            _mm256_mul_pd(psi, decay_v),
            _mm256_mul_pd(c_h_sq_dt_v, div_b_v),
        );
        let gp_x = _mm256_set_pd(
            grad_psi[i + 3].x,
            grad_psi[i + 2].x,
            grad_psi[i + 1].x,
            grad_psi[i].x,
        );
        let gp_y = _mm256_set_pd(
            grad_psi[i + 3].y,
            grad_psi[i + 2].y,
            grad_psi[i + 1].y,
            grad_psi[i].y,
        );
        let gp_z = _mm256_set_pd(
            grad_psi[i + 3].z,
            grad_psi[i + 2].z,
            grad_psi[i + 1].z,
            grad_psi[i].z,
        );
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
        let new_bx = _mm256_sub_pd(bx, _mm256_mul_pd(dt_v, gp_x));
        let new_by = _mm256_sub_pd(by, _mm256_mul_pd(dt_v, gp_y));
        let new_bz = _mm256_sub_pd(bz, _mm256_mul_pd(dt_v, gp_z));
        let mut out_psi = [0.0f64; 4];
        let mut out_bx = [0.0f64; 4];
        let mut out_by = [0.0f64; 4];
        let mut out_bz = [0.0f64; 4];
        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(out_psi.as_mut_ptr(), new_psi);
            _mm256_storeu_pd(out_bx.as_mut_ptr(), new_bx);
            _mm256_storeu_pd(out_by.as_mut_ptr(), new_by);
            _mm256_storeu_pd(out_bz.as_mut_ptr(), new_bz);
        }
        for lane in 0..lanes {
            particles[i + lane].psi_div = out_psi[lane];
            particles[i + lane].b_field.x = out_bx[lane];
            particles[i + lane].b_field.y = out_by[lane];
            particles[i + lane].b_field.z = out_bz[lane];
        }
        i += lanes;
    }
    for j in chunks..n {
        if particles[j].ptype == ParticleType::Gas {
            particles[j].psi_div = particles[j].psi_div * decay - c_h_sq_dt * div_b[j];
            particles[j].b_field.x -= grad_psi[j].x * dt;
            particles[j].b_field.y -= grad_psi[j].y * dt;
            particles[j].b_field.z -= grad_psi[j].z * dt;
        }
    }
}

#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dedner_cleaning_update_avx512(
    particles: &mut [Particle],
    div_b: &[f64],
    grad_psi: &[Vec3],
    c_h_sq_dt: f64,
    decay: f64,
    dt: f64,
) {
    let lanes = 8;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let c_h_sq_dt_v = _mm512_set1_pd(c_h_sq_dt);
    let decay_v = _mm512_set1_pd(decay);
    let dt_v = _mm512_set1_pd(dt);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                if particles[i + lane].ptype == ParticleType::Gas {
                    particles[i + lane].psi_div =
                        particles[i + lane].psi_div * decay - c_h_sq_dt * div_b[i + lane];
                    particles[i + lane].b_field.x -= grad_psi[i + lane].x * dt;
                    particles[i + lane].b_field.y -= grad_psi[i + lane].y * dt;
                    particles[i + lane].b_field.z -= grad_psi[i + lane].z * dt;
                }
            }
            i += lanes;
            continue;
        }
        let psi = _mm512_set_pd(
            particles[i + 7].psi_div,
            particles[i + 6].psi_div,
            particles[i + 5].psi_div,
            particles[i + 4].psi_div,
            particles[i + 3].psi_div,
            particles[i + 2].psi_div,
            particles[i + 1].psi_div,
            particles[i].psi_div,
        );
        let div_b_v = _mm512_set_pd(
            div_b[i + 7],
            div_b[i + 6],
            div_b[i + 5],
            div_b[i + 4],
            div_b[i + 3],
            div_b[i + 2],
            div_b[i + 1],
            div_b[i],
        );
        let new_psi = _mm512_sub_pd(
            _mm512_mul_pd(psi, decay_v),
            _mm512_mul_pd(c_h_sq_dt_v, div_b_v),
        );
        let gp_x = _mm512_set_pd(
            grad_psi[i + 7].x,
            grad_psi[i + 6].x,
            grad_psi[i + 5].x,
            grad_psi[i + 4].x,
            grad_psi[i + 3].x,
            grad_psi[i + 2].x,
            grad_psi[i + 1].x,
            grad_psi[i].x,
        );
        let gp_y = _mm512_set_pd(
            grad_psi[i + 7].y,
            grad_psi[i + 6].y,
            grad_psi[i + 5].y,
            grad_psi[i + 4].y,
            grad_psi[i + 3].y,
            grad_psi[i + 2].y,
            grad_psi[i + 1].y,
            grad_psi[i].y,
        );
        let gp_z = _mm512_set_pd(
            grad_psi[i + 7].z,
            grad_psi[i + 6].z,
            grad_psi[i + 5].z,
            grad_psi[i + 4].z,
            grad_psi[i + 3].z,
            grad_psi[i + 2].z,
            grad_psi[i + 1].z,
            grad_psi[i].z,
        );
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
        let new_bx = _mm512_sub_pd(bx, _mm512_mul_pd(dt_v, gp_x));
        let new_by = _mm512_sub_pd(by, _mm512_mul_pd(dt_v, gp_y));
        let new_bz = _mm512_sub_pd(bz, _mm512_mul_pd(dt_v, gp_z));
        let mut out_psi = [0.0f64; 8];
        let mut out_bx = [0.0f64; 8];
        let mut out_by = [0.0f64; 8];
        let mut out_bz = [0.0f64; 8];
        // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(out_psi.as_mut_ptr(), new_psi);
            _mm512_storeu_pd(out_bx.as_mut_ptr(), new_bx);
            _mm512_storeu_pd(out_by.as_mut_ptr(), new_by);
            _mm512_storeu_pd(out_bz.as_mut_ptr(), new_bz);
        }
        for lane in 0..lanes {
            particles[i + lane].psi_div = out_psi[lane];
            particles[i + lane].b_field.x = out_bx[lane];
            particles[i + lane].b_field.y = out_by[lane];
            particles[i + lane].b_field.z = out_bz[lane];
        }
        i += lanes;
    }
    for j in chunks..n {
        if particles[j].ptype == ParticleType::Gas {
            particles[j].psi_div = particles[j].psi_div * decay - c_h_sq_dt * div_b[j];
            particles[j].b_field.x -= grad_psi[j].x * dt;
            particles[j].b_field.y -= grad_psi[j].y * dt;
            particles[j].b_field.z -= grad_psi[j].z * dt;
        }
    }
}

#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
fn dedner_grad_kernel_batch_avx2(
    particles: &[Particle],
    j: usize,
    pos_i: Vec3,
    h_i: f64,
) -> (__m256d, __m256d, __m256d) {
    let px = _mm256_set_pd(
        particles[j + 3].position.x,
        particles[j + 2].position.x,
        particles[j + 1].position.x,
        particles[j].position.x,
    );
    let py = _mm256_set_pd(
        particles[j + 3].position.y,
        particles[j + 2].position.y,
        particles[j + 1].position.y,
        particles[j].position.y,
    );
    let pz = _mm256_set_pd(
        particles[j + 3].position.z,
        particles[j + 2].position.z,
        particles[j + 1].position.z,
        particles[j].position.z,
    );
    let dx = _mm256_sub_pd(px, _mm256_set1_pd(pos_i.x));
    let dy = _mm256_sub_pd(py, _mm256_set1_pd(pos_i.y));
    let dz = _mm256_sub_pd(pz, _mm256_set1_pd(pos_i.z));
    let r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
    let r = _mm256_sqrt_pd(r2);

    let h_j = _mm256_max_pd(
        _mm256_set1_pd(1e-10),
        _mm256_set_pd(
            particles[j + 3].smoothing_length,
            particles[j + 2].smoothing_length,
            particles[j + 1].smoothing_length,
            particles[j].smoothing_length,
        ),
    );
    let h_ij = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_add_pd(_mm256_set1_pd(h_i), h_j));
    let q = _mm256_div_pd(r, h_ij);
    let norm = _mm256_div_pd(
        _mm256_set1_pd(8.0 / std::f64::consts::PI),
        _mm256_mul_pd(h_ij, _mm256_mul_pd(h_ij, h_ij)),
    );
    let q2 = _mm256_mul_pd(q, q);
    let dw_inner = _mm256_mul_pd(
        norm,
        _mm256_fmadd_pd(
            _mm256_set1_pd(9.0),
            q2,
            _mm256_mul_pd(_mm256_set1_pd(-6.0), q),
        ),
    );
    let two_minus_q = _mm256_sub_pd(_mm256_set1_pd(2.0), q);
    let dw_outer = _mm256_mul_pd(
        norm,
        _mm256_mul_pd(
            _mm256_set1_pd(-1.5),
            _mm256_mul_pd(two_minus_q, two_minus_q),
        ),
    );
    let dw_dq = _mm256_blendv_pd(
        _mm256_setzero_pd(),
        _mm256_blendv_pd(
            dw_outer,
            dw_inner,
            _mm256_cmp_pd(q, _mm256_set1_pd(1.0), _CMP_LT_OQ),
        ),
        _mm256_cmp_pd(q, _mm256_set1_pd(2.0), _CMP_LT_OQ),
    );
    let scale = _mm256_div_pd(
        _mm256_div_pd(dw_dq, h_ij),
        _mm256_blendv_pd(
            r,
            _mm256_set1_pd(1.0),
            _mm256_cmp_pd(r, _mm256_set1_pd(1e-10), _CMP_LT_OQ),
        ),
    );
    let valid_r = _mm256_cmp_pd(r, _mm256_set1_pd(1e-10), _CMP_GE_OQ);
    let grad_x = _mm256_and_pd(_mm256_mul_pd(scale, dx), valid_r);
    let grad_y = _mm256_and_pd(_mm256_mul_pd(scale, dy), valid_r);
    let grad_z = _mm256_and_pd(_mm256_mul_pd(scale, dz), valid_r);
    (grad_x, grad_y, grad_z)
}

#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
fn dedner_div_gradpsi_contrib_batch_avx2(
    particles: &[Particle],
    rho: &[f64],
    j: usize,
    pos_i: Vec3,
    b_i: Vec3,
    psi_i: f64,
    h_i: f64,
) -> (f64, Vec3) {
    let (grad_x, grad_y, grad_z) = dedner_grad_kernel_batch_avx2(particles, j, pos_i, h_i);
    let rho_j = _mm256_max_pd(
        _mm256_set1_pd(1e-300),
        _mm256_set_pd(rho[j + 3], rho[j + 2], rho[j + 1], rho[j]),
    );
    let mass = _mm256_set_pd(
        particles[j + 3].mass,
        particles[j + 2].mass,
        particles[j + 1].mass,
        particles[j].mass,
    );
    let factor = _mm256_div_pd(mass, rho_j);
    let bjx = _mm256_set_pd(
        particles[j + 3].b_field.x,
        particles[j + 2].b_field.x,
        particles[j + 1].b_field.x,
        particles[j].b_field.x,
    );
    let bjy = _mm256_set_pd(
        particles[j + 3].b_field.y,
        particles[j + 2].b_field.y,
        particles[j + 1].b_field.y,
        particles[j].b_field.y,
    );
    let bjz = _mm256_set_pd(
        particles[j + 3].b_field.z,
        particles[j + 2].b_field.z,
        particles[j + 1].b_field.z,
        particles[j].b_field.z,
    );
    let dbx = _mm256_sub_pd(bjx, _mm256_set1_pd(b_i.x));
    let dby = _mm256_sub_pd(bjy, _mm256_set1_pd(b_i.y));
    let dbz = _mm256_sub_pd(bjz, _mm256_set1_pd(b_i.z));
    let div_lane = _mm256_mul_pd(
        factor,
        _mm256_fmadd_pd(
            dbx,
            grad_x,
            _mm256_fmadd_pd(dby, grad_y, _mm256_mul_pd(dbz, grad_z)),
        ),
    );
    let psij = _mm256_set_pd(
        particles[j + 3].psi_div,
        particles[j + 2].psi_div,
        particles[j + 1].psi_div,
        particles[j].psi_div,
    );
    let dpsi = _mm256_sub_pd(psij, _mm256_set1_pd(psi_i));
    let gpx = _mm256_mul_pd(_mm256_mul_pd(factor, dpsi), grad_x);
    let gpy = _mm256_mul_pd(_mm256_mul_pd(factor, dpsi), grad_y);
    let gpz = _mm256_mul_pd(_mm256_mul_pd(factor, dpsi), grad_z);

    let mut div_buf = [0.0f64; 4];
    let mut gx_buf = [0.0f64; 4];
    let mut gy_buf = [0.0f64; 4];
    let mut gz_buf = [0.0f64; 4];
    // SAFETY: fixed-size stack arrays for four f64 lanes.
    unsafe {
        _mm256_storeu_pd(div_buf.as_mut_ptr(), div_lane);
        _mm256_storeu_pd(gx_buf.as_mut_ptr(), gpx);
        _mm256_storeu_pd(gy_buf.as_mut_ptr(), gpy);
        _mm256_storeu_pd(gz_buf.as_mut_ptr(), gpz);
    }
    let div = div_buf.iter().sum();
    let gx = gx_buf.iter().sum();
    let gy = gy_buf.iter().sum();
    let gz = gz_buf.iter().sum();
    (div, Vec3::new(gx, gy, gz))
}

#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
fn dedner_sum_for_i_avx2(particles: &[Particle], rho: &[f64], i: usize) -> (f64, Vec3) {
    if particles[i].ptype != ParticleType::Gas {
        return (0.0, Vec3::zero());
    }
    let lanes = 4;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let h_i = particles[i].smoothing_length.max(1e-10);
    let pos_i = particles[i].position;
    let b_i = particles[i].b_field;
    let psi_i = particles[i].psi_div;
    let mut div_acc = 0.0_f64;
    let mut gx_acc = 0.0_f64;
    let mut gy_acc = 0.0_f64;
    let mut gz_acc = 0.0_f64;
    let mut j = 0usize;
    while j < chunks {
        let all_valid = i < j || i >= j + lanes;
        let all_gas = particles[j..j + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_valid || !all_gas {
            for lane in 0..lanes {
                let (d, g) = dedner_pair_increment(particles, rho, i, j + lane, b_i, psi_i);
                div_acc += d;
                gx_acc += g.x;
                gy_acc += g.y;
                gz_acc += g.z;
            }
            j += lanes;
            continue;
        }
        let (d, g) =
            dedner_div_gradpsi_contrib_batch_avx2(particles, rho, j, pos_i, b_i, psi_i, h_i);
        div_acc += d;
        gx_acc += g.x;
        gy_acc += g.y;
        gz_acc += g.z;
        j += lanes;
    }
    for j_tail in chunks..n {
        let (d, g) = dedner_pair_increment(particles, rho, i, j_tail, b_i, psi_i);
        div_acc += d;
        gx_acc += g.x;
        gy_acc += g.y;
        gz_acc += g.z;
    }
    (div_acc, Vec3::new(gx_acc, gy_acc, gz_acc))
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2", enable = "fma")]
fn dedner_pairwise_accumulate_avx2(
    particles: &[Particle],
    rho: &[f64],
    div_b: &mut [f64],
    grad_psi: &mut [Vec3],
) {
    for i in 0..particles.len() {
        let (d, g) = dedner_sum_for_i_avx2(particles, rho, i);
        div_b[i] = d;
        grad_psi[i] = g;
    }
}

#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
fn dedner_grad_kernel_batch_avx512(
    particles: &[Particle],
    j: usize,
    pos_i: Vec3,
    h_i: f64,
) -> (__m512d, __m512d, __m512d) {
    let px = _mm512_set_pd(
        particles[j + 7].position.x,
        particles[j + 6].position.x,
        particles[j + 5].position.x,
        particles[j + 4].position.x,
        particles[j + 3].position.x,
        particles[j + 2].position.x,
        particles[j + 1].position.x,
        particles[j].position.x,
    );
    let py = _mm512_set_pd(
        particles[j + 7].position.y,
        particles[j + 6].position.y,
        particles[j + 5].position.y,
        particles[j + 4].position.y,
        particles[j + 3].position.y,
        particles[j + 2].position.y,
        particles[j + 1].position.y,
        particles[j].position.y,
    );
    let pz = _mm512_set_pd(
        particles[j + 7].position.z,
        particles[j + 6].position.z,
        particles[j + 5].position.z,
        particles[j + 4].position.z,
        particles[j + 3].position.z,
        particles[j + 2].position.z,
        particles[j + 1].position.z,
        particles[j].position.z,
    );
    let dx = _mm512_sub_pd(px, _mm512_set1_pd(pos_i.x));
    let dy = _mm512_sub_pd(py, _mm512_set1_pd(pos_i.y));
    let dz = _mm512_sub_pd(pz, _mm512_set1_pd(pos_i.z));
    let r2 = _mm512_fmadd_pd(dx, dx, _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dz, dz)));
    let r = _mm512_sqrt_pd(r2);

    let h_j = _mm512_max_pd(
        _mm512_set1_pd(1e-10),
        _mm512_set_pd(
            particles[j + 7].smoothing_length,
            particles[j + 6].smoothing_length,
            particles[j + 5].smoothing_length,
            particles[j + 4].smoothing_length,
            particles[j + 3].smoothing_length,
            particles[j + 2].smoothing_length,
            particles[j + 1].smoothing_length,
            particles[j].smoothing_length,
        ),
    );
    let h_ij = _mm512_mul_pd(_mm512_set1_pd(0.5), _mm512_add_pd(_mm512_set1_pd(h_i), h_j));
    let q = _mm512_div_pd(r, h_ij);
    let norm = _mm512_div_pd(
        _mm512_set1_pd(8.0 / std::f64::consts::PI),
        _mm512_mul_pd(h_ij, _mm512_mul_pd(h_ij, h_ij)),
    );
    let q2 = _mm512_mul_pd(q, q);
    let dw_inner = _mm512_mul_pd(
        norm,
        _mm512_fmadd_pd(
            _mm512_set1_pd(9.0),
            q2,
            _mm512_mul_pd(_mm512_set1_pd(-6.0), q),
        ),
    );
    let two_minus_q = _mm512_sub_pd(_mm512_set1_pd(2.0), q);
    let dw_outer = _mm512_mul_pd(
        norm,
        _mm512_mul_pd(
            _mm512_set1_pd(-1.5),
            _mm512_mul_pd(two_minus_q, two_minus_q),
        ),
    );
    let inner_mask = _mm512_cmp_pd_mask(q, _mm512_set1_pd(1.0), _CMP_LT_OQ);
    let support_mask = _mm512_cmp_pd_mask(q, _mm512_set1_pd(2.0), _CMP_LT_OQ);
    let dw_dq = _mm512_mask_blend_pd(
        support_mask,
        _mm512_setzero_pd(),
        _mm512_mask_blend_pd(inner_mask, dw_outer, dw_inner),
    );
    let safe_r = _mm512_mask_blend_pd(
        _mm512_cmp_pd_mask(r, _mm512_set1_pd(1e-10), _CMP_LT_OQ),
        r,
        _mm512_set1_pd(1.0),
    );
    let scale = _mm512_div_pd(_mm512_div_pd(dw_dq, h_ij), safe_r);
    let valid_r = _mm512_cmp_pd_mask(r, _mm512_set1_pd(1e-10), _CMP_GE_OQ);
    let grad_x = _mm512_maskz_mul_pd(valid_r, scale, dx);
    let grad_y = _mm512_maskz_mul_pd(valid_r, scale, dy);
    let grad_z = _mm512_maskz_mul_pd(valid_r, scale, dz);
    (grad_x, grad_y, grad_z)
}

#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
fn dedner_div_gradpsi_contrib_batch_avx512(
    particles: &[Particle],
    rho: &[f64],
    j: usize,
    pos_i: Vec3,
    b_i: Vec3,
    psi_i: f64,
    h_i: f64,
) -> (f64, Vec3) {
    let (grad_x, grad_y, grad_z) = dedner_grad_kernel_batch_avx512(particles, j, pos_i, h_i);
    let rho_j = _mm512_max_pd(
        _mm512_set1_pd(1e-300),
        _mm512_set_pd(
            rho[j + 7],
            rho[j + 6],
            rho[j + 5],
            rho[j + 4],
            rho[j + 3],
            rho[j + 2],
            rho[j + 1],
            rho[j],
        ),
    );
    let mass = _mm512_set_pd(
        particles[j + 7].mass,
        particles[j + 6].mass,
        particles[j + 5].mass,
        particles[j + 4].mass,
        particles[j + 3].mass,
        particles[j + 2].mass,
        particles[j + 1].mass,
        particles[j].mass,
    );
    let factor = _mm512_div_pd(mass, rho_j);
    let bjx = _mm512_set_pd(
        particles[j + 7].b_field.x,
        particles[j + 6].b_field.x,
        particles[j + 5].b_field.x,
        particles[j + 4].b_field.x,
        particles[j + 3].b_field.x,
        particles[j + 2].b_field.x,
        particles[j + 1].b_field.x,
        particles[j].b_field.x,
    );
    let bjy = _mm512_set_pd(
        particles[j + 7].b_field.y,
        particles[j + 6].b_field.y,
        particles[j + 5].b_field.y,
        particles[j + 4].b_field.y,
        particles[j + 3].b_field.y,
        particles[j + 2].b_field.y,
        particles[j + 1].b_field.y,
        particles[j].b_field.y,
    );
    let bjz = _mm512_set_pd(
        particles[j + 7].b_field.z,
        particles[j + 6].b_field.z,
        particles[j + 5].b_field.z,
        particles[j + 4].b_field.z,
        particles[j + 3].b_field.z,
        particles[j + 2].b_field.z,
        particles[j + 1].b_field.z,
        particles[j].b_field.z,
    );
    let dbx = _mm512_sub_pd(bjx, _mm512_set1_pd(b_i.x));
    let dby = _mm512_sub_pd(bjy, _mm512_set1_pd(b_i.y));
    let dbz = _mm512_sub_pd(bjz, _mm512_set1_pd(b_i.z));
    let div_lane = _mm512_mul_pd(
        factor,
        _mm512_fmadd_pd(
            dbx,
            grad_x,
            _mm512_fmadd_pd(dby, grad_y, _mm512_mul_pd(dbz, grad_z)),
        ),
    );
    let psij = _mm512_set_pd(
        particles[j + 7].psi_div,
        particles[j + 6].psi_div,
        particles[j + 5].psi_div,
        particles[j + 4].psi_div,
        particles[j + 3].psi_div,
        particles[j + 2].psi_div,
        particles[j + 1].psi_div,
        particles[j].psi_div,
    );
    let dpsi = _mm512_sub_pd(psij, _mm512_set1_pd(psi_i));
    let gpx = _mm512_mul_pd(_mm512_mul_pd(factor, dpsi), grad_x);
    let gpy = _mm512_mul_pd(_mm512_mul_pd(factor, dpsi), grad_y);
    let gpz = _mm512_mul_pd(_mm512_mul_pd(factor, dpsi), grad_z);

    let mut div_buf = [0.0f64; 8];
    let mut gx_buf = [0.0f64; 8];
    let mut gy_buf = [0.0f64; 8];
    let mut gz_buf = [0.0f64; 8];
    // SAFETY: fixed-size stack arrays for eight f64 lanes.
    unsafe {
        _mm512_storeu_pd(div_buf.as_mut_ptr(), div_lane);
        _mm512_storeu_pd(gx_buf.as_mut_ptr(), gpx);
        _mm512_storeu_pd(gy_buf.as_mut_ptr(), gpy);
        _mm512_storeu_pd(gz_buf.as_mut_ptr(), gpz);
    }
    let div = div_buf.iter().sum();
    let gx = gx_buf.iter().sum();
    let gy = gy_buf.iter().sum();
    let gz = gz_buf.iter().sum();
    (div, Vec3::new(gx, gy, gz))
}

#[cfg(all(
    any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
fn dedner_sum_for_i_avx512(particles: &[Particle], rho: &[f64], i: usize) -> (f64, Vec3) {
    if particles[i].ptype != ParticleType::Gas {
        return (0.0, Vec3::zero());
    }
    let lanes = 8;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let h_i = particles[i].smoothing_length.max(1e-10);
    let pos_i = particles[i].position;
    let b_i = particles[i].b_field;
    let psi_i = particles[i].psi_div;
    let mut div_acc = 0.0_f64;
    let mut gx_acc = 0.0_f64;
    let mut gy_acc = 0.0_f64;
    let mut gz_acc = 0.0_f64;
    let mut j = 0usize;
    while j < chunks {
        let all_valid = i < j || i >= j + lanes;
        let all_gas = particles[j..j + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_valid || !all_gas {
            for lane in 0..lanes {
                let (d, g) = dedner_pair_increment(particles, rho, i, j + lane, b_i, psi_i);
                div_acc += d;
                gx_acc += g.x;
                gy_acc += g.y;
                gz_acc += g.z;
            }
            j += lanes;
            continue;
        }
        let (d, g) =
            dedner_div_gradpsi_contrib_batch_avx512(particles, rho, j, pos_i, b_i, psi_i, h_i);
        div_acc += d;
        gx_acc += g.x;
        gy_acc += g.y;
        gz_acc += g.z;
        j += lanes;
    }
    for j_tail in chunks..n {
        let (d, g) = dedner_pair_increment(particles, rho, i, j_tail, b_i, psi_i);
        div_acc += d;
        gx_acc += g.x;
        gy_acc += g.y;
        gz_acc += g.z;
    }
    (div_acc, Vec3::new(gx_acc, gy_acc, gz_acc))
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
fn dedner_pairwise_accumulate_avx512(
    particles: &[Particle],
    rho: &[f64],
    div_b: &mut [f64],
    grad_psi: &mut [Vec3],
) {
    for i in 0..particles.len() {
        let (d, g) = dedner_sum_for_i_avx512(particles, rho, i);
        div_b[i] = d;
        grad_psi[i] = g;
    }
}

/// Calcula solo la divergencia SPH de B para cada partícula gas.
///
/// Útil para el path híbrido CPU+GPU donde la divergencia se computa en CPU
/// y el paso de actualización ψ/B se delega al kernel CUDA `cuda_mhd_dedner_cleaning`.
/// El array devuelto está en `f32` para coincidir con la firma del wrapper CUDA.
pub fn compute_dedner_div_b(particles: &[Particle]) -> Vec<f32> {
    let n = particles.len();
    if n == 0 {
        return Vec::new();
    }
    let rho = dedner_density(particles);
    let mut div_b_f64 = vec![0.0_f64; n];
    let mut grad_psi = vec![gadget_ng_core::Vec3::zero(); n];
    dedner_pairwise_accumulate(particles, &rho, &mut div_b_f64, &mut grad_psi);
    div_b_f64.iter().map(|&v| v as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;

    fn cleaning_particles() -> Vec<Particle> {
        (0..13)
            .map(|i| {
                let mut particle = Particle::new_gas(
                    i,
                    1.0 + 0.07 * i as f64,
                    Vec3::new(0.09 * i as f64, 0.03 * (i % 5) as f64, 0.04 * i as f64),
                    Vec3::zero(),
                    1.0,
                    0.4 + 0.02 * (i % 6) as f64,
                );
                particle.b_field = Vec3::new(
                    0.1 + 0.01 * i as f64,
                    -0.2 + 0.015 * (i % 4) as f64,
                    0.05 * (i % 3) as f64,
                );
                particle.psi_div = -0.2 + 0.03 * i as f64;
                if i % 5 == 0 {
                    particle.ptype = ParticleType::DarkMatter;
                }
                particle
            })
            .collect()
    }

    fn reset_bench_simd_tier_env() {
        #[cfg(feature = "bench-all-dedner-paths")]
        // SAFETY: test-only reset of bench tier env between cases.
        unsafe {
            std::env::remove_var("GADGET_NG_MHD_BENCH_SIMD_TIER");
        }
    }

    #[test]
    #[cfg(any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ))]
    fn dedner_density_dispatch_matches_scalar_with_tail_and_dark_matter() {
        reset_bench_simd_tier_env();
        let particles = cleaning_particles();
        let scalar = dedner_density_scalar(&particles);
        let dispatch = dedner_density(&particles);

        for (actual, expected) in dispatch.iter().zip(scalar) {
            // Density is a local m/h^3 computation; SIMD only batches lanes, so
            // scalar and vector paths should agree to roundoff for this setup.
            assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
        }
    }

    #[test]
    #[cfg(all(
        any(
            not(feature = "rayon"),
            feature = "bench-all-dedner-paths",
            all(feature = "rayon", feature = "simd")
        ),
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    fn dedner_pairwise_dispatch_matches_scalar() {
        reset_bench_simd_tier_env();
        let particles = cleaning_particles();
        let n = particles.len();
        let rho = dedner_density_scalar(&particles);
        let mut div_s = vec![0.0_f64; n];
        let mut gp_s = vec![Vec3::zero(); n];
        dedner_pairwise_accumulate_scalar(&particles, &rho, &mut div_s, &mut gp_s);
        let mut div_d = vec![0.0_f64; n];
        let mut gp_d = vec![Vec3::zero(); n];
        dedner_pairwise_accumulate(&particles, &rho, &mut div_d, &mut gp_d);
        for i in 0..n {
            assert_abs_diff_eq!(div_d[i], div_s[i], epsilon = 1e-9);
            assert_abs_diff_eq!(gp_d[i].x, gp_s[i].x, epsilon = 1e-9);
            assert_abs_diff_eq!(gp_d[i].y, gp_s[i].y, epsilon = 1e-9);
            assert_abs_diff_eq!(gp_d[i].z, gp_s[i].z, epsilon = 1e-9);
        }
    }

    #[test]
    #[cfg(any(
        not(feature = "rayon"),
        feature = "bench-all-dedner-paths",
        all(feature = "rayon", feature = "simd")
    ))]
    fn dedner_cleaning_dispatch_leaves_dark_matter_unchanged() {
        reset_bench_simd_tier_env();
        let mut particles = cleaning_particles();
        let before = particles.clone();

        dedner_cleaning_step(&mut particles, 0.8, 0.3, 0.02);

        for (actual, expected) in particles.iter().zip(before) {
            if expected.ptype != ParticleType::Gas {
                assert_eq!(actual.psi_div, expected.psi_div);
                assert_eq!(actual.b_field, expected.b_field);
            }
        }
    }
}
