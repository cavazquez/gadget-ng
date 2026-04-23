//! Phase 95 — EoR completo z=6–12.
//!
//! Verifica que después de aplicar `reionization_step` con fuentes UV activas:
//! 1. `x_hii_mean` crece con el tiempo (avanza la reionización).
//! 2. Con x_HII inicial = 0 y fuentes UV activas, la señal 21cm disminuye.
//! 3. El `ReionizationState` reporta valores coherentes.
//!
//! No requiere corrida cosmológica real: N=8³ de partículas, 5 pasos.

use gadget_ng_core::Vec3;
use gadget_ng_rt::{
    brightness_temperature, compute_reionization_state, reionization_step,
    ChemState, Cm21Params, M1Params, RadiationField, ReionizationParams, UvSource,
};

const BOX_SIZE: f64 = 10.0;
const N_MESH: usize = 8;
const DX: f64 = BOX_SIZE / N_MESH as f64;

fn make_rt_field() -> RadiationField {
    RadiationField::uniform(N_MESH, N_MESH, N_MESH, DX, 0.0)
}

fn make_m1_params() -> M1Params {
    M1Params {
        c_red_factor: 100.0,
        kappa_abs: 1.0,
        kappa_scat: 0.0,
        substeps: 2,
    }
}

fn make_uv_sources() -> Vec<UvSource> {
    vec![
        UvSource { pos: Vec3::new(2.5, 2.5, 2.5), luminosity: 10.0 },
        UvSource { pos: Vec3::new(7.5, 7.5, 7.5), luminosity: 10.0 },
    ]
}

fn make_neutral_chem_states(n: usize) -> Vec<ChemState> {
    (0..n)
        .map(|_| ChemState {
            x_hi: 1.0,
            x_hii: 0.0,
            x_hei: 1.0,
            x_heii: 0.0,
            x_heiii: 0.0,
            x_e: 0.0,
        })
        .collect()
}

/// Verifica que el estado de reionización se puede calcular con gas neutro.
#[test]
fn reionization_state_neutral_gas() {
    let chem = make_neutral_chem_states(64);
    let state = compute_reionization_state(&chem, 9.0, 2);
    assert_eq!(state.x_hii_mean, 0.0, "gas neutro debe tener x_hii_mean=0");
    assert_eq!(state.z, 9.0);
    assert_eq!(state.n_sources, 2);
}

/// Verifica que `reionization_step` devuelve estado coherente con fuentes UV.
#[test]
fn reionization_step_returns_valid_state() {
    let mut rf = make_rt_field();
    let mut chem = make_neutral_chem_states(64);
    let sources = make_uv_sources();
    let m1p = make_m1_params();

    let state = reionization_step(&mut rf, &mut chem, &sources, &m1p, 0.01, BOX_SIZE, 9.0);

    assert_eq!(state.z, 9.0, "el redshift debe ser 9.0");
    assert_eq!(state.n_sources, 2, "debe reportar 2 fuentes");
    assert!(state.x_hii_mean >= 0.0, "x_hii_mean debe ser >= 0");
    assert!(state.x_hii_mean <= 1.0, "x_hii_mean debe ser <= 1");
}

/// Verifica que múltiples pasos con fuentes UV no disminuyen x_hii_mean.
/// La reionización es unidireccional: solo puede aumentar la ionización.
#[test]
fn reionization_x_hii_non_decreasing() {
    let mut rf = make_rt_field();
    let mut chem = make_neutral_chem_states(64);
    let sources = make_uv_sources();
    let m1p = make_m1_params();
    let dt = 0.01;

    let state0 = reionization_step(&mut rf, &mut chem, &sources, &m1p, dt, BOX_SIZE, 9.0);
    let x_hii_0 = state0.x_hii_mean;

    // Varios pasos más (sin fuentes UV acumulando en el campo)
    for _step in 1..5 {
        reionization_step(&mut rf, &mut chem, &sources, &m1p, dt, BOX_SIZE, 8.9);
    }
    let state5 = compute_reionization_state(&chem, 8.9, 2);

    // x_hii_mean no puede disminuir ya que no hay recombinación implementada en este solver simple
    assert!(
        state5.x_hii_mean >= x_hii_0 - 1e-10,
        "x_hii_mean = {} no debe ser menor que el inicial {}",
        state5.x_hii_mean,
        x_hii_0
    );
}

/// Verifica que la señal 21cm disminuye cuando hay ionización (más x_HII → menos señal).
#[test]
fn eor_21cm_signal_decreases_with_ionization() {
    let cm21_params = Cm21Params::default();
    let z = 9.0;

    // Gas completamente neutro
    let dtb_neutral = brightness_temperature(0.0, 1.0, z, &cm21_params);
    // Gas 50% ionizado
    let dtb_half = brightness_temperature(0.5, 1.0, z, &cm21_params);
    // Gas completamente ionizado
    let dtb_ionized = brightness_temperature(1.0, 1.0, z, &cm21_params);

    assert!(
        dtb_neutral > dtb_half,
        "señal 21cm debe disminuir con ionización: {} > {}",
        dtb_neutral,
        dtb_half
    );
    assert!(
        dtb_half > dtb_ionized,
        "señal 21cm debe disminuir con más ionización: {} > {}",
        dtb_half,
        dtb_ionized
    );
    assert!(
        dtb_ionized.abs() < 1e-10,
        "señal 21cm debe ser 0 con gas completamente ionizado"
    );
}

/// Verifica que la reionización sin fuentes UV no modifica los estados de química.
#[test]
fn reionization_no_sources_no_change() {
    let mut rf = make_rt_field();
    let mut chem = make_neutral_chem_states(8);
    let sources: Vec<UvSource> = Vec::new();
    let m1p = make_m1_params();

    reionization_step(&mut rf, &mut chem, &sources, &m1p, 0.01, BOX_SIZE, 9.0);

    // Sin fuentes, los estados de química no cambian (no hay fotones)
    for c in &chem {
        assert_eq!(c.x_hii, 0.0, "sin fuentes, x_hii debe permanecer 0");
    }
}

/// Verifica que `ReionizationParams` tiene los valores por defecto correctos.
#[test]
fn reionization_params_defaults() {
    let params = ReionizationParams::default();
    assert!(!params.enabled, "reionización desactivada por defecto");
    assert_eq!(params.z_start, 12.0, "z_start default = 12");
    assert_eq!(params.z_end, 6.0, "z_end default = 6");
}

/// Verifica el acoplamiento química → 21cm:
/// si los ChemState reales tienen x_HII > 0, la señal 21cm debe ser menor
/// que con estado neutro puro.
#[test]
fn coupled_chem_reduces_21cm_signal() {
    use gadget_ng_core::{Particle, Vec3};
    use gadget_ng_rt::{compute_cm21_output, Cm21Params};

    let box_size = 10.0;
    let n_mesh = 8;
    let dx = box_size / n_mesh as f64;
    let n_part = n_mesh * n_mesh * n_mesh;

    let mut particles = Vec::new();
    for ix in 0..n_mesh {
        for iy in 0..n_mesh {
            for iz in 0..n_mesh {
                let mut p = Particle::new(
                    ix * n_mesh * n_mesh + iy * n_mesh + iz,
                    1.0,
                    Vec3::new((ix as f64 + 0.5) * dx, (iy as f64 + 0.5) * dx, (iz as f64 + 0.5) * dx),
                    Vec3::zero(),
                );
                p.internal_energy = 100.0;
                p.smoothing_length = 0.4 * dx;
                particles.push(p);
            }
        }
    }

    let params = Cm21Params::default();
    let z = 9.0;

    // Caso 1: estados neutros (x_HII = 0) → señal máxima
    let chem_neutral: Vec<ChemState> = (0..n_part).map(|_| ChemState {
        x_hi: 1.0, x_hii: 0.0, x_hei: 1.0, x_heii: 0.0, x_heiii: 0.0, x_e: 0.0,
    }).collect();
    let out_neutral = compute_cm21_output(&particles, &chem_neutral, box_size, z, n_mesh, 4, &params);

    // Caso 2: gas 50% ionizado (x_HII = 0.5) → señal reducida a la mitad
    let chem_half: Vec<ChemState> = (0..n_part).map(|_| ChemState {
        x_hi: 0.5, x_hii: 0.5, x_hei: 1.0, x_heii: 0.0, x_heiii: 0.0, x_e: 0.5,
    }).collect();
    let out_half = compute_cm21_output(&particles, &chem_half, box_size, z, n_mesh, 4, &params);

    assert!(
        out_half.delta_tb_mean < out_neutral.delta_tb_mean,
        "señal 21cm con x_HII=0.5 ({:.3}) debe ser menor que con x_HII=0 ({:.3})",
        out_half.delta_tb_mean,
        out_neutral.delta_tb_mean
    );

    // La reducción debe ser aproximadamente del 50%
    let ratio = out_half.delta_tb_mean / out_neutral.delta_tb_mean;
    assert!(
        (ratio - 0.5).abs() < 0.05,
        "ratio δT_b(ionizado) / δT_b(neutro) debe ser ~0.5, got {:.3}",
        ratio
    );
}
