/// Phase 118 — Función de luminosidad y colores galácticos
///
/// Tests: L escala con masa, L decrece con edad, DM no contribuye, galaxy_luminosity
///        suma estrellas, colores BV y GR dentro de rango, sin estrellas L=0, serde.
use gadget_ng_analysis::{
    LuminosityResult, bv_color, galaxy_luminosity, gr_color, stellar_luminosity_solar,
};
use gadget_ng_core::{Particle, Vec3};

fn star(id: usize, mass: f64, age: f64, z: f64) -> Particle {
    let mut p = Particle::new_star(id, mass, Vec3::zero(), Vec3::zero(), z);
    p.stellar_age = age;
    p
}

// ── 1. Luminosidad escala con masa ────────────────────────────────────────

#[test]
fn luminosity_scales_with_mass() {
    let l1 = stellar_luminosity_solar(1.0, 1.0, 0.02);
    let l2 = stellar_luminosity_solar(2.0, 1.0, 0.02);
    assert!(
        (l2 / l1 - 2.0).abs() < 1e-10,
        "L debe escalar linealmente con masa"
    );
}

// ── 2. Luminosidad decrece con edad ───────────────────────────────────────

#[test]
fn luminosity_decreases_with_age() {
    let l_young = stellar_luminosity_solar(1.0, 0.01, 0.02);
    let l_old = stellar_luminosity_solar(1.0, 10.0, 0.02);
    assert!(
        l_young > l_old,
        "Estrella joven debe ser más luminosa que vieja"
    );
}

// ── 3. DM no contribuye a luminosidad galácticca ──────────────────────────

#[test]
fn dm_does_not_contribute_to_luminosity() {
    let particles = vec![Particle::new(0, 1e10, Vec3::zero(), Vec3::zero())];
    let result = galaxy_luminosity(&particles);
    assert_eq!(result.l_total, 0.0, "DM no debe contribuir a luminosidad");
    assert_eq!(result.n_stars, 0);
}

// ── 4. galaxy_luminosity suma correctamente ────────────────────────────────

#[test]
fn galaxy_luminosity_sums_stars() {
    let age = 1.0;
    let z = 0.02;
    let particles = vec![
        star(0, 1.0, age, z),
        star(1, 2.0, age, z),
        Particle::new(2, 1.0, Vec3::zero(), Vec3::zero()), // DM, no cuenta
    ];
    let l_expected = stellar_luminosity_solar(1.0, age, z) + stellar_luminosity_solar(2.0, age, z);
    let result = galaxy_luminosity(&particles);
    assert!(
        (result.l_total - l_expected).abs() / l_expected < 1e-10,
        "galaxy_luminosity debe sumar L de todas las estrellas"
    );
    assert_eq!(result.n_stars, 2);
}

// ── 5. B-V dentro del rango físico ────────────────────────────────────────

#[test]
fn bv_color_in_physical_range() {
    for &age in &[0.01, 0.1, 1.0, 5.0, 10.0] {
        for &z in &[0.001, 0.01, 0.02, 0.04] {
            let bv = bv_color(age, z);
            assert!(bv > -0.5, "B-V muy azul: {bv} para age={age} Z={z}");
            assert!(bv < 2.0, "B-V muy rojo: {bv} para age={age} Z={z}");
        }
    }
}

// ── 6. g-r dentro del rango físico ────────────────────────────────────────

#[test]
fn gr_color_in_physical_range() {
    for &age in &[0.01, 0.1, 1.0, 5.0, 10.0] {
        for &z in &[0.001, 0.01, 0.02, 0.04] {
            let gr = gr_color(age, z);
            assert!(gr > -0.5, "g-r muy azul: {gr}");
            assert!(gr < 1.5, "g-r muy rojo: {gr}");
        }
    }
}

// ── 7. Sin estrellas, galaxy_luminosity retorna ceros ─────────────────────

#[test]
fn no_stars_returns_zero_luminosity() {
    let particles: Vec<Particle> = vec![];
    let result = galaxy_luminosity(&particles);
    assert_eq!(
        result,
        LuminosityResult {
            l_total: 0.0,
            bv: 0.0,
            gr: 0.0,
            n_stars: 0
        }
    );
}
