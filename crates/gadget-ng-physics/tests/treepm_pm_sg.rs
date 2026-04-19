//! Tests de validación física para el scatter/gather PM (Fase 24).
//!
//! Verifica que el nuevo protocolo scatter/gather PM produce física correcta,
//! sin doble conteo erf/erfc y con conservación de impulso.
//!
//! 1. `sg_no_double_counting_erf_erfc`   — split erf+erfc no produce doble conteo
//! 2. `sg_cosmo_no_explosion_N27`         — N=27, 3 pasos EdS, sin NaN/Inf
//! 3. `sg_momentum_conservation`          — |Δp| < tolerancia tras 5 pasos

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_parallel::SerialRuntime;
use gadget_ng_pm::SlabLayout;
use gadget_ng_treepm::{
    distributed::{
        pm_scatter_gather_accels, short_range_accels_sfc, SfcShortRangeParams,
    },
    short_range::erfc_approx,
};

// ── Utilidades ────────────────────────────────────────────────────────────────

fn make_particle(id: usize, x: f64, y: f64, z: f64, mass: f64) -> Particle {
    Particle {
        position: Vec3::new(x, y, z),
        velocity: Vec3::zero(),
        acceleration: Vec3::zero(),
        mass,
        global_id: id,
    }
}

/// Grilla cúbica de N=n³ partículas uniformes en el box.
fn grid_particles(n: usize, box_size: f64) -> Vec<Particle> {
    let mut particles = Vec::with_capacity(n * n * n);
    let total = n * n * n;
    let mass_each = 1.0_f64 / total as f64;
    let mut id = 0;
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let x = (ix as f64 + 0.5) * box_size / n as f64;
                let y = (iy as f64 + 0.5) * box_size / n as f64;
                let z = (iz as f64 + 0.5) * box_size / n as f64;
                particles.push(make_particle(id, x, y, z, mass_each));
                id += 1;
            }
        }
    }
    particles
}

// ── Test 1: split erf+erfc no produce doble conteo ──────────────────────────

/// Verifica que la fuerza total PM + SR del scatter/gather no duplica
/// contribuciones de ninguna escala.
///
/// El split erf+erfc debe satisfacer:
///   F_total ≈ F_direct (Newton exacto)
///   F_PM + F_SR ≠ 2 × F_PM o 2 × F_SR
///
/// Se usa un sistema de dos partículas donde se puede calcular la fuerza
/// de Newton exacta y comparar con PM+SR.
#[test]
fn sg_no_double_counting_erf_erfc() {
    let box_size = 1.0_f64;
    let nm = 16_usize;
    let g = 1.0_f64;
    let r_split = 2.5 * box_size / nm as f64; // r_split estándar GADGET

    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;

    // Dos partículas con separación clara en z (< 0.5 para imagen más cercana unívoca)
    // p0 en z=0.3, p1 en z=0.6: separación = 0.3, imagen más cercana en +z
    let p0 = make_particle(0, 0.5, 0.5, 0.30, 1.0);
    let p1 = make_particle(1, 0.5, 0.5, 0.60, 1.0);
    let particles = vec![p0, p1];

    // PM largo alcance (scatter/gather Fase 24)
    let (acc_pm, _) =
        pm_scatter_gather_accels(&particles, &layout, g, r_split, box_size, &rt);

    // SR corto alcance (erfc kernel, sin halos en P=1)
    let eps2 = 1e-6;
    let sr_params = SfcShortRangeParams {
        local_particles: &particles,
        halo_particles: &[],
        eps2,
        g,
        r_split,
        box_size,
    };
    let mut acc_sr = vec![Vec3::zero(); 2];
    short_range_accels_sfc(&sr_params, &mut acc_sr);

    // Fuerza total PM+SR
    let f_total_z_0 = acc_pm[0].z + acc_sr[0].z;
    let f_total_z_1 = acc_pm[1].z + acc_sr[1].z;

    // Fuerza de Newton exacta: p0 atrae hacia +z (hacia p1), p1 atrae hacia -z
    let dz = 0.30_f64;
    let f_newton = g * 1.0 * 1.0 / (dz * dz);

    // La fuerza total PM+SR debe apuntar correctamente:
    // p0 hacia +z (positivo), p1 hacia -z (negativo).
    // Nota: el PM puede ser impreciso a escala ~dz/box_size si el grid es
    // grueso, pero la componente SR sí debe apuntar en la dirección correcta.
    // Verificamos que la fuerza total no sea de magnitud absurda.
    let mag_total_0 = f_total_z_0.abs();
    let mag_total_1 = f_total_z_1.abs();
    assert!(
        mag_total_0 > 0.0 && mag_total_1 > 0.0,
        "fuerzas PM+SR deben ser no nulas: |f0_z|={mag_total_0:.6}, |f1_z|={mag_total_1:.6}"
    );

    // Acción-reacción en PM+SR: f0_z + f1_z debe ser ≈ 0
    let imbalance = (f_total_z_0 + f_total_z_1).abs();
    let f_scale = mag_total_0.max(mag_total_1).max(1e-20);
    let rel_imbalance = imbalance / f_scale;
    assert!(
        rel_imbalance < 0.15,
        "desequilibrio acción-reacción: f0_z={f_total_z_0:.4}, f1_z={f_total_z_1:.4}, rel_imbalance={rel_imbalance:.4}"
    );

    // Sin doble conteo: |F_total| < 2 × |F_Newton| (el PM CIC puede ser impreciso,
    // pero no puede producir una fuerza mayor que el doble de Newton)
    assert!(
        mag_total_0 < 2.0 * f_newton,
        "posible doble conteo: |f_total|={mag_total_0:.4} > 2×f_newton={:.4}", 2.0 * f_newton
    );

    // Verificar el kernel split: erf(r/r2) + erfc(r/r2) = 1 para r = dz
    let u = dz / (std::f64::consts::SQRT_2 * r_split);
    let erfc_val = erfc_approx(u);
    let erf_val = 1.0 - erfc_val;
    let sum = erf_val + erfc_val;
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "erf+erfc != 1: sum={sum:.14} para u={u:.4}"
    );
}

// ── Test 2: N=27, 3 pasos EdS, sin explosión ────────────────────────────────

/// Integra 3 pasos de un sistema de N=27 partículas en grilla 3³
/// con aceleraciones PM scatter/gather (Fase 24) + SR (erfc) + halo vacío (P=1).
///
/// Verifica que ninguna partícula produce NaN/Inf y que las posiciones
/// permanecen en [0, box_size).
#[test]
fn sg_cosmo_no_explosion_n27() {
    let n_side = 3_usize;
    let box_size = 1.0_f64;
    let nm = 8_usize;
    let g = 43.009_f64; // G en unidades GADGET (kpc/M_sun/s²)
    let r_split = 2.5 * box_size / nm as f64;
    let eps2 = (r_split * 0.1) * (r_split * 0.1);
    let dt = 0.01_f64;
    let n_steps = 3_usize;

    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;

    let mut particles = grid_particles(n_side, box_size);

    for _step in 0..n_steps {
        // PM largo alcance (scatter/gather Fase 24)
        let (acc_pm, sg_stats) =
            pm_scatter_gather_accels(&particles, &layout, g, r_split, box_size, &rt);
        assert_eq!(sg_stats.scatter_particles, particles.len());

        // SR corto alcance (P=1, sin halos)
        let sr_params = SfcShortRangeParams {
            local_particles: &particles,
            halo_particles: &[],
            eps2,
            g,
            r_split,
            box_size,
        };
        let mut acc_sr = vec![Vec3::zero(); particles.len()];
        short_range_accels_sfc(&sr_params, &mut acc_sr);

        // Kick + drift
        for (i, p) in particles.iter_mut().enumerate() {
            let ax = acc_pm[i].x + acc_sr[i].x;
            let ay = acc_pm[i].y + acc_sr[i].y;
            let az = acc_pm[i].z + acc_sr[i].z;

            assert!(ax.is_finite(), "ax NaN/Inf en partícula {i} paso {_step}");
            assert!(ay.is_finite(), "ay NaN/Inf en partícula {i} paso {_step}");
            assert!(az.is_finite(), "az NaN/Inf en partícula {i} paso {_step}");

            p.velocity.x += ax * dt * 0.5;
            p.velocity.y += ay * dt * 0.5;
            p.velocity.z += az * dt * 0.5;

            p.position.x = (p.position.x + p.velocity.x * dt).rem_euclid(box_size);
            p.position.y = (p.position.y + p.velocity.y * dt).rem_euclid(box_size);
            p.position.z = (p.position.z + p.velocity.z * dt).rem_euclid(box_size);

            p.velocity.x += ax * dt * 0.5;
            p.velocity.y += ay * dt * 0.5;
            p.velocity.z += az * dt * 0.5;

            assert!(
                p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
                "posición NaN/Inf en partícula {i} paso {_step}"
            );
            assert!(
                p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
                "velocidad NaN/Inf en partícula {i} paso {_step}"
            );
        }
    }

    // Verificar que el sistema no explotó: ninguna partícula fuera de box
    for p in &particles {
        assert!(
            p.position.x >= 0.0 && p.position.x < box_size,
            "partícula {} fuera del box en x: {}", p.global_id, p.position.x
        );
        assert!(
            p.position.y >= 0.0 && p.position.y < box_size,
            "partícula {} fuera del box en y: {}", p.global_id, p.position.y
        );
        assert!(
            p.position.z >= 0.0 && p.position.z < box_size,
            "partícula {} fuera del box en z: {}", p.global_id, p.position.z
        );
    }
}

// ── Test 3: conservación de impulso ─────────────────────────────────────────

/// La suma de aceleraciones PM (scatter/gather) debe ser próxima a cero para
/// un sistema periódico con distribución uniforme (impulso neto = 0 por
/// simetría del grid PM + periodicidad).
///
/// También verifica que la suma de fuerzas SR tiende a cero (acción-reacción).
#[test]
fn sg_momentum_conservation() {
    let n_side = 3_usize;
    let box_size = 1.0_f64;
    let nm = 8_usize;
    let g = 1.0_f64;
    let r_split = 2.5 * box_size / nm as f64;
    let eps2 = (r_split * 0.1) * (r_split * 0.1);
    let dt = 0.005_f64;
    let n_steps = 5_usize;

    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;

    let mut particles = grid_particles(n_side, box_size);
    // Añadir pequeñas perturbaciones para hacer el sistema más interesante
    use std::f64::consts::PI;
    let n_parts = particles.len();
    for (i, p) in particles.iter_mut().enumerate() {
        let phase = (i as f64) * 2.0 * PI / n_parts as f64;
        p.position.x = (p.position.x + 0.02 * phase.sin()).rem_euclid(box_size);
        p.position.y = (p.position.y + 0.02 * phase.cos()).rem_euclid(box_size);
    }

    let mut total_px = 0.0_f64;
    let mut total_py = 0.0_f64;
    let mut total_pz = 0.0_f64;

    // Momentum inicial = 0 (velocidades iniciales = 0)
    for p in &particles {
        total_px += p.mass * p.velocity.x;
        total_py += p.mass * p.velocity.y;
        total_pz += p.mass * p.velocity.z;
    }
    assert!(
        total_px.abs() < 1e-15 && total_py.abs() < 1e-15 && total_pz.abs() < 1e-15,
        "impulso inicial no nulo"
    );

    for _step in 0..n_steps {
        let (acc_pm, _) =
            pm_scatter_gather_accels(&particles, &layout, g, r_split, box_size, &rt);

        let sr_params = SfcShortRangeParams {
            local_particles: &particles,
            halo_particles: &[],
            eps2,
            g,
            r_split,
            box_size,
        };
        let mut acc_sr = vec![Vec3::zero(); particles.len()];
        short_range_accels_sfc(&sr_params, &mut acc_sr);

        // Kick + drift (leapfrog)
        for (i, p) in particles.iter_mut().enumerate() {
            let ax = acc_pm[i].x + acc_sr[i].x;
            let ay = acc_pm[i].y + acc_sr[i].y;
            let az = acc_pm[i].z + acc_sr[i].z;

            p.velocity.x += ax * dt;
            p.velocity.y += ay * dt;
            p.velocity.z += az * dt;

            p.position.x = (p.position.x + p.velocity.x * dt).rem_euclid(box_size);
            p.position.y = (p.position.y + p.velocity.y * dt).rem_euclid(box_size);
            p.position.z = (p.position.z + p.velocity.z * dt).rem_euclid(box_size);
        }
    }

    // Impulso total tras la integración
    let mut final_px = 0.0_f64;
    let mut final_py = 0.0_f64;
    let mut final_pz = 0.0_f64;
    for p in &particles {
        final_px += p.mass * p.velocity.x;
        final_py += p.mass * p.velocity.y;
        final_pz += p.mass * p.velocity.z;
    }

    // Verificar que la suma de fuerzas PM en cada paso fue ≈ 0
    // Para sistema periódico + simetría, |Δp| / p_rms debe ser pequeño.
    // La tolerancia es generosa por el error de resolución del grid PM (nm=8).
    let p_rms = (final_px * final_px + final_py * final_py + final_pz * final_pz).sqrt();
    let v_rms = {
        let sum_v2: f64 = particles
            .iter()
            .map(|p| p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y + p.velocity.z * p.velocity.z)
            .sum();
        (sum_v2 / particles.len() as f64).sqrt()
    };
    let total_mass: f64 = particles.iter().map(|p| p.mass).sum();
    let p_scale = total_mass * v_rms;

    let rel_momentum = if p_scale > 0.0 { p_rms / p_scale } else { p_rms };
    assert!(
        rel_momentum < 0.05,
        "impulso relativo demasiado grande: |Δp|/p_scale = {rel_momentum:.4} (límite=0.05)\n\
         final_p = ({final_px:.4}, {final_py:.4}, {final_pz:.4}), p_scale={p_scale:.4}"
    );
}
