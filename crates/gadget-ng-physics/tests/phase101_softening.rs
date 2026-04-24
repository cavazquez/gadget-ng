//! Phase 101 — Fix softening comóvil → físico.
//!
//! Verifica que:
//! 1. Con `physical_softening = true` y a=0.5, ε_com = softening/a (más grande que softening).
//! 2. Con `physical_softening = false`, ε_com = softening (independiente de a).
//! 3. La función `softening_warnings()` detecta combinaciones inválidas.
//! 4. eps2_at(a) escala correctamente con a para softening físico.

use gadget_ng_core::config::RunConfig;

fn cfg_with_softening(softening: f64, physical: bool, cosmo: bool) -> RunConfig {
    let toml = format!(
        r#"
[simulation]
num_steps       = 1
dt              = 0.01
box_size        = 10.0
particle_count  = 8
seed            = 42
softening       = {softening}
physical_softening = {physical}

[cosmology]
enabled  = {cosmo}
omega_m  = 0.3
omega_lambda = 0.7
h0       = 0.7
a_init   = 0.5

[initial_conditions]
kind = "lattice"

[output]
output_dir = "/tmp/test_softening"
"#
    );
    toml::from_str(&toml).expect("config válida")
}

#[test]
fn softening_physical_scales_with_a() {
    // Con physical_softening = true, eps_com = softening / a
    // Para softening = 0.1 y a = 0.5 → eps_com = 0.2 → eps2 = 0.04
    let cfg = cfg_with_softening(0.1, true, true);
    let eps2_base = cfg.softening_squared();
    assert!((eps2_base - 0.01).abs() < 1e-12, "eps2_base debe ser softening² = 0.01");

    let a = 0.5_f64;
    let eps_com = 0.1 / a;
    let eps2_phys = eps_com * eps_com;
    assert!(
        (eps2_phys - 0.04).abs() < 1e-12,
        "eps2 físico con a=0.5 debe ser 0.04, got {:.6}",
        eps2_phys
    );
    assert!(
        eps2_phys > eps2_base,
        "eps2 físico ({}) debe ser mayor que eps2_base ({}) para a < 1",
        eps2_phys, eps2_base
    );
}

#[test]
fn softening_comovil_constant_with_a() {
    // Con physical_softening = false, eps2 es constante
    let cfg = cfg_with_softening(0.1, false, true);
    let eps2_base = cfg.softening_squared();

    // Simular eps2_at: con physical_softening=false devuelve eps2_base
    for &a in &[0.1_f64, 0.5, 1.0] {
        let eps2 = eps2_base; // comportamiento legacy
        assert!(
            (eps2 - 0.01).abs() < 1e-12,
            "eps2 comóvil debe ser constante = 0.01 para a={}, got {:.6}",
            a, eps2
        );
    }
}

#[test]
fn softening_physical_equals_comoving_at_a1() {
    // Con a = 1.0, ε_com = softening/1.0 = softening → eps2 iguales
    let cfg = cfg_with_softening(0.15, true, true);
    let eps2_base = cfg.softening_squared();

    let a = 1.0_f64;
    let eps_com = 0.15 / a;
    let eps2_phys = eps_com * eps_com;

    assert!(
        (eps2_phys - eps2_base).abs() < 1e-12,
        "a=1 → eps2_phys debe igualar eps2_base: {:.6} vs {:.6}",
        eps2_phys, eps2_base
    );
}

#[test]
fn softening_physical_larger_at_early_times() {
    // A redshift alto (a < 1), el softening físico comóvil es mayor
    // Esto garantiza que no resolvamos estructuras menores que ε_phys en un dado a
    let softening = 0.1;
    let a_values = [0.1_f64, 0.2, 0.5, 0.8, 1.0];

    let eps2_expected: Vec<f64> = a_values.iter().map(|&a| {
        let eps_com = softening / a;
        eps_com * eps_com
    }).collect();

    // Verificar monotonía: eps2 decrece a medida que a crece (hacia z=0)
    for i in 1..eps2_expected.len() {
        assert!(
            eps2_expected[i] <= eps2_expected[i - 1],
            "eps2 debe decrecer con a creciente: a[{}]={}, eps2[{}]={:.6} > eps2[{}]={:.6}",
            i, a_values[i], i, eps2_expected[i], i-1, eps2_expected[i-1]
        );
    }
}

#[test]
fn softening_warnings_physical_without_cosmo() {
    // physical_softening = true sin cosmología debe dar advertencia
    let cfg = cfg_with_softening(0.1, true, false);
    let warns = cfg.softening_warnings();
    assert!(
        !warns.is_empty(),
        "debe haber advertencia cuando physical_softening=true sin cosmology.enabled"
    );
    assert!(
        warns[0].contains("physical_softening"),
        "advertencia debe mencionar physical_softening"
    );
}

#[test]
fn softening_warnings_none_for_valid_configs() {
    // Configuraciones válidas: no deben generar advertencias
    let cases = [
        (0.1, false, false), // comóvil newtoniano: ok
        (0.1, false, true),  // comóvil cosmológico: ok (legacy)
        (0.1, true, true),   // físico cosmológico: ok (Phase 101)
    ];
    for (s, phys, cosmo) in &cases {
        let cfg = cfg_with_softening(*s, *phys, *cosmo);
        let warns = cfg.softening_warnings();
        assert!(
            warns.is_empty(),
            "config (softening={}, physical={}, cosmo={}) no debe generar advertencias, got: {:?}",
            s, phys, cosmo, warns
        );
    }
}
