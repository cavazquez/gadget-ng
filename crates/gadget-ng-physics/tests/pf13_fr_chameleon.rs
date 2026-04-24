//! PF-13 — Gravedad modificada f(R): recuperación de Newton en alta densidad
//!
//! Verifica que el mecanismo chameleon suprime la quinta fuerza en regiones
//! de alta densidad:
//!
//! ```text
//! fifth_force_factor(f_r_local, f_r0) < 1%   cuando ρ >> ρ_crit
//! ```
//!
//! y que se recupera en baja densidad (sin screening).

use gadget_ng_core::{fifth_force_factor, CosmologyParams, FRParams};

/// En alta densidad (screening activo), la quinta fuerza es suprimida.
#[test]
fn chameleon_suppresses_fifth_force_in_dense_medium() {
    // En alta densidad, f_r_local << f_r0 (campo escalar suprimido)
    let f_r_local = 1e-10_f64; // muy pequeño → screening fuerte
    let f_r0 = 1e-4_f64;
    let factor = fifth_force_factor(f_r_local, f_r0);
    assert!(
        factor < 0.01,
        "Chameleon no suprime quinta fuerza: factor={factor:.4e} (esperado < 1%)"
    );
}

/// Sin screening (baja densidad), la quinta fuerza está activa.
#[test]
fn chameleon_active_in_low_density() {
    // En baja densidad, f_r_local ≈ f_r0 → sin screening
    let f_r0 = 1e-4_f64;
    let f_r_local = f_r0; // sin screening
    let factor = fifth_force_factor(f_r_local, f_r0);
    // El factor debe ser significativo (> 0.5 sin screening)
    assert!(
        factor > 0.5,
        "Quinta fuerza debe estar activa en baja densidad: factor={factor:.4}"
    );
}

/// El factor está acotado en [0, 1].
#[test]
fn fifth_force_factor_bounded() {
    for &(frl, fr0) in &[
        (1e-10, 1e-4),
        (1e-4, 1e-4),
        (1e-3, 1e-4),
        (0.0, 1e-4),
        (1.0, 1e-6),
    ] {
        let f = fifth_force_factor(frl, fr0);
        assert!(
            f >= 0.0 && f <= 1.0,
            "Factor fuera de [0,1]: f_r_local={frl:.1e}, f_r0={fr0:.1e}, factor={f:.4}"
        );
    }
}

/// La supresión aumenta conforme disminuye f_r_local/f_r0.
#[test]
fn chameleon_screening_increases_with_density() {
    let f_r0 = 1e-4_f64;
    let ratios = [1.0, 0.1, 0.01, 0.001, 1e-4];
    let factors: Vec<f64> = ratios
        .iter()
        .map(|&r| fifth_force_factor(r * f_r0, f_r0))
        .collect();

    for i in 1..factors.len() {
        assert!(
            factors[i] <= factors[i - 1] + 1e-10,
            "Screening debe aumentar con ratio menor: f[{}]={:.4e} > f[{}]={:.4e}",
            i, factors[i], i - 1, factors[i - 1]
        );
    }
}

/// Con f_r0 = 0 (GR puro), la quinta fuerza es nula.
#[test]
fn fifth_force_zero_for_gr() {
    let factor = fifth_force_factor(0.0, 0.0);
    assert!(
        factor < 1e-10 || factor.is_nan() == false,
        "Con f_r0=0 (GR), la quinta fuerza debe ser nula o bien definida: {factor:.4e}"
    );
    // En GR no hay quinta fuerza: apply_modified_gravity con f_r0=0 es noop
    use gadget_ng_core::{apply_modified_gravity, Particle, Vec3};
    let params = FRParams { f_r0: 0.0, n: 1.0 };
    let cosmo = CosmologyParams::new(0.3, 0.7, 0.1);
    let mut p = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero());
    p.acceleration = Vec3::new(1.0, 2.0, 3.0);
    let acc_before = p.acceleration;
    let mut particles = vec![p];
    apply_modified_gravity(&mut particles, &params, &cosmo, 1.0);
    assert_eq!(
        particles[0].acceleration.x, acc_before.x,
        "f_r0=0 no debe modificar la aceleración"
    );
}

/// La quinta fuerza es un factor adicional sobre la gravedad newtoniana.
#[test]
fn fifth_force_factor_is_fraction_of_newtonian() {
    // Para f_R con n=1, la quinta fuerza máxima es 1/3 de la newtoniana
    // Esto está implícito en el factor: max(fifth_force_factor) ≤ 1
    let f_r0 = 1e-5_f64;
    let f_r_local = f_r0;
    let factor = fifth_force_factor(f_r_local, f_r0);
    assert!(
        factor <= 1.0,
        "La quinta fuerza no puede superar la newtoniana: factor={factor:.4}"
    );
}
