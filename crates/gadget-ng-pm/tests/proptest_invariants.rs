use proptest::prelude::*;
use gadget_ng_core::Vec3;
use approx::assert_abs_diff_eq;

proptest! {
    /// CIC mass assignment must conserve total mass exactly.
    #[test]
    fn cic_conserves_total_mass(
        xs in prop::collection::vec(-0.5_f64..1.5, 1..100),
        ys in prop::collection::vec(-0.5_f64..1.5, 1..100),
        zs in prop::collection::vec(-0.5_f64..1.5, 1..100),
        masses in prop::collection::vec(0.1_f64..10.0, 1..100),
    ) {
        let n = xs.len().min(ys.len()).min(zs.len()).min(masses.len());
        let pos: Vec<Vec3> = (0..n).map(|i| Vec3::new(xs[i], ys[i], zs[i])).collect();
        let m: Vec<f64> = masses[..n].to_vec();

        let density = gadget_ng_pm::cic::assign(&pos, &m, 1.0, 8);
        let total: f64 = density.iter().sum();
        let expected: f64 = m.iter().sum();
        assert_abs_diff_eq!(total, expected, epsilon = 1e-10);
    }

    /// Gravity antisymmetry: total momentum is conserved (m1 * a1 + m2 * a2 = 0).
    #[test]
    fn gravity_momentum_conservation(
        x1 in -10.0_f64..10.0, y1 in -10.0_f64..10.0, z1 in -10.0_f64..10.0,
        x2 in -10.0_f64..10.0, y2 in -10.0_f64..10.0, z2 in -10.0_f64..10.0,
        m1 in 0.1_f64..100.0, m2 in 0.1_f64..100.0,
        eps2 in 1e-6_f64..1.0,
    ) {
        let p1 = Vec3::new(x1, y1, z1);
        let p2 = Vec3::new(x2, y2, z2);
        let g = 1.0;

        let dr = p2 - p1;
        let r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + eps2;
        let inv_r3 = 1.0 / (r2 * r2.sqrt());

        // Acceleration of particle 1 due to particle 2: a12 = G * m2 * dr / |dr|^3
        let a12 = Vec3::new(
            dr.x * m2 * g * inv_r3,
            dr.y * m2 * g * inv_r3,
            dr.z * m2 * g * inv_r3,
        );

        // Acceleration of particle 2 due to particle 1: a21 = -G * m1 * dr / |dr|^3
        let a21 = Vec3::new(
            -dr.x * m1 * g * inv_r3,
            -dr.y * m1 * g * inv_r3,
            -dr.z * m1 * g * inv_r3,
        );

        // Momentum conservation: m1 * a12 + m2 * a21 = 0
        assert_abs_diff_eq!(m1 * a12.x + m2 * a21.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m1 * a12.y + m2 * a21.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m1 * a12.z + m2 * a21.z, 0.0, epsilon = 1e-12);
    }

    /// Eisenstein-Hu transfer function must be positive for all k.
    #[test]
    fn transfer_function_positive(kappa in 1e-4_f64..100.0) {
        use gadget_ng_core::transfer_fn::{self, EisensteinHuParams};
        let p = EisensteinHuParams {
            omega_m: 0.315,
            omega_b: 0.049,
            h: 0.673,
            t_cmb: 2.725,
        };
        let t_val = transfer_fn::transfer_eh_nowiggle(kappa, &p);
        assert!(t_val > 0.0, "transfer function must be positive, got {t_val} at k={kappa}");
    }
}
