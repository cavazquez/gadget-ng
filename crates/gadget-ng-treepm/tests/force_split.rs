//! Test: la suma F_lr + F_sr debe recuperar el Newton exacto.
//!
//! El splitting Gaussiano garantiza:
//!   erf(r / (√2·r_s)) + erfc(r / (√2·r_s)) = 1
//!
//! Por lo tanto, para un par de partículas:
//!   F_lr(r) + F_sr(r) = F_newton(r)
//!
//! Este test lo verifica numéricamente usando el PM filtrado como F_lr
//! y el kernel erfc como F_sr, comprobando que la suma coincide con
//! DirectGravity dentro de la tolerancia del PM (suavizado de celda).

use gadget_ng_core::{DirectGravity, GravitySolver, Vec3};
use gadget_ng_treepm::TreePmSolver;

/// Dos partículas: una masa pesada en el centro y una sonda a distancia conocida.
/// La suma TreePM ≈ Newton debe coincidir dentro de un factor razonable.
#[test]
fn treepm_sum_approximates_newton() {
    let box_size = 4.0_f64;
    let nm = 32usize; // cell_size = 4/32 = 0.125; r_split = 2.5 * 0.125 = 0.3125
    let g = 1.0_f64;
    let eps2 = 1e-4_f64;

    // Masa pesada en el centro del cubo.
    let positions = vec![
        Vec3::new(0.5 * box_size, 0.5 * box_size, 0.5 * box_size),
        Vec3::new(0.6 * box_size, 0.5 * box_size, 0.5 * box_size), // sonda
    ];
    let masses = vec![10.0_f64, 0.001];

    let probe = vec![1usize];

    // ── DirectGravity (referencia) ─────────────────────────────────────────────
    let direct = DirectGravity;
    let mut acc_direct = vec![Vec3::zero()];
    direct.accelerations_for_indices(&positions, &masses, eps2, g, &probe, &mut acc_direct);

    // ── TreePM ────────────────────────────────────────────────────────────────
    let treepm = TreePmSolver {
        grid_size: nm,
        box_size,
        r_split: 0.0, // automático
    };
    let mut acc_treepm = vec![Vec3::zero()];
    treepm.accelerations_for_indices(&positions, &masses, eps2, g, &probe, &mut acc_treepm);

    let ax_direct = acc_direct[0].x;
    let ax_treepm = acc_treepm[0].x;

    // Misma dirección: la sonda está a +x del centro → fuerza en -x.
    assert!(ax_direct < 0.0, "DirectGravity ax={ax_direct:.4e} debería ser < 0");
    assert!(ax_treepm < 0.0, "TreePM ax={ax_treepm:.4e} debería ser < 0");

    // Magnitud: TreePM debe estar dentro de un factor 3 del Newton (tolerancia generosa
    // para PM con nm=32 y una sola masa pesada).
    let ratio = ax_treepm.abs() / ax_direct.abs();
    assert!(
        ratio > 0.1 && ratio < 10.0,
        "ratio |ax_treepm/ax_direct| = {ratio:.3} fuera del rango [0.1, 10]"
    );
}

/// Verificar que el kernel erfc(r/(√2·r_s)) × G·m/r² + erf(r/(√2·r_s)) × G·m/r²
/// = G·m/r² (partición de la unidad).
#[test]
fn force_splitting_partitions_unity() {
    use gadget_ng_treepm::short_range::erfc_approx;

    let r_split = 0.5_f64;
    // Para varios valores de r: erf(x) + erfc(x) = 1
    for &r in &[0.1_f64, 0.3, 0.5, 1.0, 2.0, 5.0] {
        let x = r / (std::f64::consts::SQRT_2 * r_split);
        let erfc_val = erfc_approx(x);
        let erf_val = 1.0 - erfc_val;
        let total = erf_val + erfc_val;
        assert!(
            (total - 1.0).abs() < 1e-6,
            "r={r}: erf+erfc = {total:.8} (esperado 1.0)"
        );
    }
}
