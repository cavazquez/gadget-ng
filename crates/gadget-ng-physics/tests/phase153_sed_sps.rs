//! Tests de integración — Phase 153: SED completa con tablas SPS BC03-lite.

use gadget_ng_analysis::luminosity::galaxy_sed;
use gadget_ng_analysis::sps_tables::{sps_luminosity, SpsGrid, Spsband};
use gadget_ng_core::{Particle, Vec3};

fn star_particle(age_gyr: f64, metallicity: f64, mass: f64) -> Particle {
    let mut p = Particle::new_star(0, mass, Vec3::zero(), Vec3::zero(), metallicity);
    p.stellar_age = age_gyr;
    p
}

// T1: L_B decrece con edad estelar
#[test]
fn lb_decreases_with_age() {
    let l_b_young = sps_luminosity(0.01, 0.02, Spsband::B);
    let l_b_old = sps_luminosity(10.0, 0.02, Spsband::B);
    assert!(l_b_young > l_b_old, "L_B debe decrecer con edad: joven={}, viejo={}", l_b_young, l_b_old);
}

// T2: B-V se enrojece con la edad
#[test]
fn bv_reddens_with_age() {
    let grid = SpsGrid::bc03_lite();
    let l_b_young = grid.interpolate(0.01, 0.02, Spsband::B);
    let l_v_young = grid.interpolate(0.01, 0.02, Spsband::V);
    let l_b_old = grid.interpolate(10.0, 0.02, Spsband::B);
    let l_v_old = grid.interpolate(10.0, 0.02, Spsband::V);
    let bv_young = -2.5 * (l_b_young / l_v_young).log10();
    let bv_old = -2.5 * (l_b_old / l_v_old).log10();
    assert!(bv_old > bv_young, "B-V debe enrojecerse con la edad: joven={:.3}, viejo={:.3}", bv_young, bv_old);
}

// T3: metalicidad más alta → más luminosa (L_B mayor para Z alta)
#[test]
fn higher_metallicity_more_luminous() {
    let l_b_low_z = sps_luminosity(5.0, 0.0004, Spsband::B);
    let l_b_high_z = sps_luminosity(5.0, 0.05, Spsband::B);
    assert!(l_b_high_z > l_b_low_z, "Metalicidad alta debe dar más luminosidad: low={}, high={}", l_b_low_z, l_b_high_z);
}

// T4: interpolación bilineal en nodos exactos
#[test]
fn bilinear_interpolation_on_node() {
    let grid = SpsGrid::bc03_lite();
    let v_exact = grid.interpolate(0.01, 0.0004, Spsband::V);
    assert_eq!(v_exact, grid.l_v[0][0], "Interpolación debe coincidir en nodo de la tabla");
}

// T5: galaxy_sed retorna SedResult con campos válidos
#[test]
fn galaxy_sed_returns_valid_result() {
    let particles = vec![
        star_particle(1.0, 0.02, 1.0),
        star_particle(5.0, 0.02, 1.0),
    ];
    let sed = galaxy_sed(&particles);
    assert!(sed.l_b > 0.0 && sed.l_v > 0.0, "L_B y L_V deben ser positivas");
    assert_eq!(sed.n_stars, 2);
    assert!(sed.mass_weighted_age > 0.0);
}

// T6: N=200 estrellas sin panics
#[test]
fn galaxy_sed_n200_no_panic() {
    let mut particles = Vec::new();
    for i in 0..200 {
        let age = 0.01 + i as f64 * 0.065;
        let z = (0.001 + i as f64 * 0.000195).min(0.05);
        particles.push(star_particle(age, z, 0.01));
    }
    let sed = galaxy_sed(&particles);
    assert_eq!(sed.n_stars, 200);
    assert!(sed.l_b > 0.0 && sed.l_r > 0.0);
}
