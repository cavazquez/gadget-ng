//! Tests de integración — Phase 174-176: SZ Effect, Weak Lensing, Ly-α Forest.

use gadget_ng_analysis::{
    LensingMap, LightconeHit, LyaCosmoParams, LyaParams, SzParams, TomographyParams,
    accumulate_tomographic_lensing, analyze_lya_forest, compute_compton_y_map,
    compute_kinetic_sz_map, convergence_angular_cl, kaiser_squires_reconstruct,
};
use gadget_ng_core::{Particle, Vec3};

const GAMMA: f64 = 5.0 / 3.0;

fn gas_particle(u: f64, h: f64, mass: f64, x: f64, y: f64, z: f64) -> Particle {
    Particle::new_gas(0, mass, Vec3::new(x, y, z), Vec3::zero(), u, h)
}

// ── Phase 174: SZ Effect ──────────────────────────────────────────────────

// T1: Compton-y es positivo para gas caliente
#[test]
fn sz_compton_y_positive_for_hot_gas() {
    let particles = vec![gas_particle(1e4, 5.0, 1.0, 50.0, 50.0, 50.0)];
    let params = SzParams {
        n_pixels: 64,
        axis: 'z',
    };
    let result = compute_compton_y_map(&particles, 100.0, &params, GAMMA);
    assert!(
        result.mean_y > 0.0,
        "mean_y > 0 for hot gas, got {}",
        result.mean_y
    );
}

// T2: kSZ RMS es positivo para gas con velocidad peculiar along LOS
#[test]
fn sz_ksz_rms_positive_with_peculiar_velocity() {
    let mut p = gas_particle(1e4, 5.0, 1.0, 50.0, 50.0, 50.0);
    p.velocity = Vec3::new(0.0, 0.0, 100.0); // velocity along z (LOS)
    let particles = vec![p];
    let params = SzParams {
        n_pixels: 64,
        axis: 'z',
    };
    let result = compute_kinetic_sz_map(&particles, 100.0, &params, GAMMA);
    assert!(
        result.rms_ksz > 0.0,
        "rms_ksz > 0 with peculiar velocity, got {}",
        result.rms_ksz
    );
}

// T3: Gas frío → Compton-y muy pequeño
#[test]
fn sz_compton_y_tiny_for_cold_gas() {
    let particles = vec![gas_particle(1e-6, 5.0, 1.0, 50.0, 50.0, 50.0)];
    let params = SzParams {
        n_pixels: 64,
        axis: 'z',
    };
    let result_hot = compute_compton_y_map(&particles, 100.0, &params, GAMMA);
    let hot_particles = vec![gas_particle(1e4, 5.0, 1.0, 50.0, 50.0, 50.0)];
    let result_cold = compute_compton_y_map(&hot_particles, 100.0, &params, GAMMA);
    assert!(
        result_cold.mean_y > result_hot.mean_y * 100.0,
        "Hot gas y should dominate cold gas: cold_y={} vs hot_y={}",
        result_hot.mean_y,
        result_cold.mean_y
    );
}

// ── Phase 175: Weak Lensing ───────────────────────────────────────────────

// T4: KS reconstruction of zero shear gives near-zero convergence
#[test]
fn wl_ks_zero_shear_zero_kappa() {
    let map = LensingMap::new(64);
    let result = kaiser_squires_reconstruct(&map, 0.1);
    let max_abs_kappa = result.kappa.iter().cloned().fold(0.0f64, f64::max).abs();
    assert!(
        max_abs_kappa < 1e-8,
        "Zero shear should give ~zero convergence, max |κ| = {max_abs_kappa}"
    );
}

// T5: C_ell of constant kappa field has signal at expected scales
#[test]
fn wl_cl_constant_kappa_has_signal() {
    let mut map = LensingMap::new(32);
    // Inject a constant convergence
    for v in map.kappa.iter_mut() {
        *v = 1.0;
    }
    let cl = convergence_angular_cl(&map, 0.1, 4);
    // A constant field should have all power at low-ell
    assert!(!cl.is_empty(), "C_ell should have bins");
    assert!(cl[0].cl > 0.0, "Low-ell C_ell should be positive");
}

// T6: Tomographic lensing assigns particles to correct z-bins
#[test]
fn wl_tomography_correct_bin_assignment() {
    let observer = Vec3::zero();
    let hits = vec![
        LightconeHit {
            particle_index: 0,
            position: Vec3::new(0.5, 0.0, 0.5),
            distance: (0.25_f64 + 0.25_f64).sqrt(),
        },
        LightconeHit {
            particle_index: 1,
            position: Vec3::new(0.0, 0.5, 0.5),
            distance: (0.25_f64 + 0.25_f64).sqrt(),
        },
    ];
    let masses = vec![1.0, 2.0];
    let redshifts = vec![0.1, 1.5];
    let params = TomographyParams {
        z_edges: vec![0.0, 0.5, 1.0, 2.0],
        n_pixels: 16,
    };
    let result = accumulate_tomographic_lensing(&hits, &masses, &redshifts, observer, &params);
    assert_eq!(result.kappa_tomo.len(), 3, "Should have 3 z-bins");
    // z=0.1 → bin 0, z=1.5 → bin 2
    let sum_bin0: f64 = result.kappa_tomo[0].iter().sum();
    let sum_bin1: f64 = result.kappa_tomo[1].iter().sum();
    let sum_bin2: f64 = result.kappa_tomo[2].iter().sum();
    assert!(sum_bin0 > 0.0, "Bin 0 (z=0.1) should have signal");
    assert!(sum_bin1.abs() < 1e-10, "Bin 1 (0.5<z<1.0) should be empty");
    assert!(sum_bin2 > 0.0, "Bin 2 (z=1.5) should have signal");
}

// ── Phase 176: Ly-α Forest ───────────────────────────────────────────────

// T7: analyze_lya_forest returns valid structure with sightlines
#[test]
fn lya_analyze_returns_valid_structure() {
    let cosmo = LyaCosmoParams::default();
    let params = LyaParams {
        n_sightlines: 4,
        n_velocity_cells: 64,
        z_source: 3.0,
        dv_kms: 25.0,
        t_igm_kelvin: 1e4,
    };
    let particles = vec![gas_particle(1e4, 0.5, 1e10, 50.0, 50.0, 50.0)];
    let result = analyze_lya_forest(&particles, 100.0, &params, &cosmo, 'z', None);
    assert_eq!(result.n_sightlines, 4, "Should return 4 sightlines");
    assert!(
        result.mean_flux >= 0.0,
        "Mean flux should be non-negative, got {}",
        result.mean_flux
    );
}

// T8: Fully ionized gas is transparent (⟨F⟩ close to 1)
#[test]
fn lya_ionized_gas_transparent() {
    use gadget_ng_analysis::LyaChemState;
    let cosmo = LyaCosmoParams::default();
    let params = LyaParams {
        n_sightlines: 4,
        n_velocity_cells: 64,
        z_source: 3.0,
        dv_kms: 25.0,
        t_igm_kelvin: 1e4,
    };
    let particles = vec![gas_particle(1e4, 0.5, 1e10, 50.0, 50.0, 50.0)];
    let chem = vec![LyaChemState::fully_ionized()];
    let result = analyze_lya_forest(&particles, 100.0, &params, &cosmo, 'z', Some(&chem));
    // Fully ionized gas: x_HI ≈ 0, so τ should be very small → ⟨F⟩ ≈ 1
    assert!(
        result.mean_flux > 0.9,
        "Ionized gas should have ⟨F⟩ > 0.9, got {}",
        result.mean_flux
    );
}

// T9: compute_tau_along_sightline gives absorption for neutral gas
#[test]
fn lya_single_sightline_neutral_absorption() {
    use gadget_ng_analysis::compute_tau_along_sightline;
    let cosmo = LyaCosmoParams::default();
    let params = LyaParams {
        n_sightlines: 4,
        n_velocity_cells: 128,
        z_source: 3.0,
        dv_kms: 25.0,
        t_igm_kelvin: 1e4,
    };
    // Place a dense gas cloud at center of box, sightline through it
    let particles = vec![gas_particle(1e4, 2.0, 1e10, 50.0, 50.0, 50.0)];
    let sightline =
        compute_tau_along_sightline(&particles, 'z', 50.0, 50.0, 100.0, &params, &cosmo, None);
    // Neutral gas should produce some optical depth
    let max_tau = sightline.tau.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        max_tau > 0.0,
        "Neutral gas should produce τ > 0, got max_tau = {max_tau}"
    );
}

// T10: Higher density → more absorption (higher max τ)
#[test]
fn lya_higher_density_more_absorption() {
    use gadget_ng_analysis::compute_tau_along_sightline;
    let cosmo = LyaCosmoParams::default();
    let params = LyaParams {
        n_sightlines: 4,
        n_velocity_cells: 128,
        z_source: 3.0,
        dv_kms: 25.0,
        t_igm_kelvin: 1e4,
    };
    let p_low = vec![gas_particle(1e4, 5.0, 1e10, 50.0, 50.0, 50.0)];
    let p_high = vec![gas_particle(1e4, 1.0, 1e10, 50.0, 50.0, 50.0)];
    let sl_low = compute_tau_along_sightline(&p_low, 'z', 50.0, 50.0, 100.0, &params, &cosmo, None);
    let sl_high =
        compute_tau_along_sightline(&p_high, 'z', 50.0, 50.0, 100.0, &params, &cosmo, None);
    let max_tau_low = sl_low.tau.iter().cloned().fold(0.0f64, f64::max);
    let max_tau_high = sl_high.tau.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        max_tau_high > max_tau_low,
        "Higher density should produce larger max(τ): {} vs {}",
        max_tau_high,
        max_tau_low
    );
}
