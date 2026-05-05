use gadget_ng_core::{DirectGravity, GravitySolver, Vec3};
use gadget_ng_tree::BarnesHutGravity;

fn sample_cloud(n: usize) -> (Vec<Vec3>, Vec<f64>, Vec<usize>) {
    let mut pos = Vec::with_capacity(n);
    let mut mass = Vec::with_capacity(n);
    for i in 0..n {
        // nube concentrada + cola difusa para tensionar el MAC
        let t = i as f64 / n as f64;
        let x = if i % 5 == 0 {
            0.7 + 0.25 * (13.0 * t).sin()
        } else {
            0.5 + 0.05 * (41.0 * t).sin()
        };
        let y = 0.5 + 0.09 * (29.0 * t).cos();
        let z = 0.5 + 0.08 * (17.0 * t).sin();
        pos.push(Vec3::new(x, y, z));
        mass.push(1.0 / n as f64);
    }
    let idx: Vec<usize> = (0..n).step_by(2).collect();
    (pos, mass, idx)
}

fn rms_rel_err(got: &[Vec3], reference: &[Vec3]) -> f64 {
    let n = got.len().min(reference.len()).max(1);
    let mut s = 0.0;
    for i in 0..n {
        let dg = (got[i] - reference[i]).norm();
        let nr = reference[i].norm().max(1e-12);
        let r = dg / nr;
        s += r * r;
    }
    (s / n as f64).sqrt()
}

#[test]
fn relative_mac_error_cost_is_competitive() {
    let (pos, mass, idx) = sample_cloud(256);
    let eps2 = 1e-4;
    let g = 1.0;

    let mut a_ref = vec![Vec3::zero(); idx.len()];
    DirectGravity.accelerations_for_indices(&pos, &mass, eps2, g, &idx, &mut a_ref);

    let geom = BarnesHutGravity {
        theta: 0.5,
        multipole_order: 3,
        use_relative_criterion: false,
        use_bmax_criterion: false,
        err_tol_force_acc: 0.005,
        softened_multipoles: true,
        mac_softening: gadget_ng_core::MacSoftening::Consistent,
    };
    let mut a_geom = vec![Vec3::zero(); idx.len()];
    let mut c_geom = Vec::new();
    geom.accelerations_with_costs(&pos, &mass, eps2, g, &idx, &mut a_geom, &mut c_geom);

    let rel = BarnesHutGravity {
        use_relative_criterion: true,
        ..geom
    };
    let mut a_rel = vec![Vec3::zero(); idx.len()];
    let mut c_rel = Vec::new();
    rel.accelerations_with_costs(&pos, &mass, eps2, g, &idx, &mut a_rel, &mut c_rel);

    let e_geom = rms_rel_err(&a_geom, &a_ref);
    let e_rel = rms_rel_err(&a_rel, &a_ref);
    let cost_geom = c_geom.iter().sum::<u64>() as f64 / c_geom.len().max(1) as f64;
    let cost_rel = c_rel.iter().sum::<u64>() as f64 / c_rel.len().max(1) as f64;

    // El criterio relativo debe ser competitivo en precisión/costo.
    assert!(
        e_rel <= 1.15 * e_geom,
        "MAC relativo degrada demasiado el error: e_rel={e_rel:.3e}, e_geom={e_geom:.3e}"
    );
    assert!(
        cost_rel <= 4.5 * cost_geom,
        "MAC relativo abre demasiados nodos: c_rel={cost_rel:.1}, c_geom={cost_geom:.1}"
    );
}

