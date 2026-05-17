//! Phase 185 — f(R) no lineal en malla PM.

use gadget_ng_core::{FRParams, ModifiedGravitySection};
use gadget_ng_pm::{
    FrMeshParams, fr_screening_field, pm_fifth_force_boost, solve_forces_fr_screened_mesh,
    solve_forces_modified_gravity,
};

#[test]
fn screening_field_suppresses_dense_cells() {
    let nm = 4;
    let mut density = vec![1.0; nm * nm * nm];
    density[0] = 100.0;
    let params = FRParams {
        f_r0: 1.0e-5,
        n: 1.0,
    };

    let screen = fr_screening_field(&density, nm, &params, 0, 0.0);

    assert!(screen[0] < 0.01);
    assert!(screen.iter().any(|&s| s > 0.5));
}

#[test]
fn screened_mesh_is_weaker_than_homogeneous_boost_near_dense_source() {
    let nm = 8;
    let box_size = 1.0;
    let mut density = vec![0.0; nm * nm * nm];
    let idx = |ix: usize, iy: usize, iz: usize| -> usize { iz * nm * nm + iy * nm + ix };
    density[idx(4, 4, 4)] = 1.0;
    density[idx(5, 4, 4)] = 0.1;
    let params = FRParams {
        f_r0: 1.0e-6,
        n: 1.0,
    };

    let homogeneous = solve_forces_modified_gravity(&density, 1.0, nm, box_size, &params, None);
    let screened = solve_forces_fr_screened_mesh(
        &density,
        1.0,
        nm,
        box_size,
        FrMeshParams {
            fr: &params,
            iterations: 2,
            smoothing: 0.5,
            plummer_eps: None,
            screening_override: None,
        },
    );
    let cell = idx(5, 4, 4);
    let f_h = homogeneous[0][cell].abs() + homogeneous[1][cell].abs() + homogeneous[2][cell].abs();
    let f_s = screened[0][cell].abs() + screened[1][cell].abs() + screened[2][cell].abs();

    assert!(pm_fifth_force_boost(&params) > 1.0);
    assert!(f_s < f_h);
}

#[test]
fn modified_gravity_section_serde_accepts_mesh_knobs() {
    let cfg: ModifiedGravitySection = toml::from_str(
        r#"
enabled = true
nonlinear_mesh = true
mesh_iterations = 8
screening_smoothing = 0.25
"#,
    )
    .expect("modified gravity section should deserialize");

    assert!(cfg.nonlinear_mesh);
    assert_eq!(cfg.mesh_iterations, 8);
    assert_eq!(cfg.screening_smoothing, 0.25);
}
