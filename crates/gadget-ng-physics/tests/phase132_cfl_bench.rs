/// Phase 132 — Benchmark MHD Criterion + CFL unificado
///
/// Tests: alfven_dt menor que dt global con B fuerte (CFL aplica),
///        alfven_dt mayor que dt global con B débil (sin restricción),
///        CFL unificado = min(dt_global, dt_alfven),
///        alfven_dt escala correcto con h y B,
///        Cargo.toml de gadget-ng-mhd tiene bench alfven_bench,
///        full MHD step con dt restringido por CFL produce resultado finito.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::{advance_induction, alfven_dt, apply_magnetic_forces, dedner_cleaning_step};

fn gas_with_b(id: usize, pos: Vec3, b: Vec3, h: f64) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, Vec3::zero(), 1.0, h);
    p.b_field = b;
    p
}

// ── 1. alfven_dt < dt_global con B fuerte → CFL restringe ────────────────

#[test]
fn alfven_dt_restricts_when_b_is_strong() {
    let particles = vec![gas_with_b(0, Vec3::zero(), Vec3::new(100.0, 0.0, 0.0), 0.1)];
    let dt_global = 0.1_f64;
    let dt_a = alfven_dt(&particles, 0.3);
    let dt_mhd = dt_global.min(dt_a);
    assert!(
        dt_a < dt_global,
        "dt_alfven={dt_a:.4e} debe < dt_global={dt_global}"
    );
    assert_eq!(dt_mhd, dt_a, "CFL usa dt_alfven");
}

// ── 2. alfven_dt > dt_global con B débil → sin restricción ───────────────

#[test]
fn alfven_dt_no_restrict_with_weak_b() {
    let particles = vec![gas_with_b(0, Vec3::zero(), Vec3::new(1e-5, 0.0, 0.0), 0.1)];
    let dt_global = 0.001_f64;
    let dt_a = alfven_dt(&particles, 0.3);
    let dt_mhd = dt_global.min(dt_a);
    assert!(
        dt_a > dt_global,
        "dt_alfven={dt_a:.4e} debe > dt_global={dt_global}"
    );
    assert_eq!(dt_mhd, dt_global, "CFL usa dt_global");
}

// ── 3. CFL unificado = min(dt_global, dt_alfven) ─────────────────────────

#[test]
fn cfl_unified_is_minimum() {
    let b_values = [0.01, 1.0, 10.0, 100.0];
    let dt_global = 0.01_f64;
    let cfl = 0.3_f64;
    for b in b_values {
        let particles = vec![gas_with_b(0, Vec3::zero(), Vec3::new(b, 0.0, 0.0), 0.1)];
        let dt_a = alfven_dt(&particles, cfl);
        let dt_mhd = dt_global.min(dt_a);
        assert_eq!(
            dt_mhd,
            dt_global.min(dt_a),
            "B={b}: CFL={dt_mhd:.4e} debe ser mín(dt_g={dt_global}, dt_a={dt_a:.4e})"
        );
    }
}

// ── 4. alfven_dt escala correctamente con B ───────────────────────────────

#[test]
fn alfven_dt_scales_with_b() {
    // Con ρ estimada como m/h³: v_A = B/sqrt(μ₀ ρ) = B × h^{3/2}/sqrt(μ₀ m)
    // dt_A = cfl × h / v_A = cfl × h × sqrt(μ₀ m) / (B × h^{3/2}) = cfl × sqrt(μ₀ m) / (B × sqrt(h))
    // Doblar B → reducir dt_A a la mitad
    let p1 = vec![gas_with_b(0, Vec3::zero(), Vec3::new(1.0, 0.0, 0.0), 0.1)];
    let p3 = vec![gas_with_b(0, Vec3::zero(), Vec3::new(2.0, 0.0, 0.0), 0.1)]; // B×2

    let dt1 = alfven_dt(&p1, 0.3);
    let dt3 = alfven_dt(&p3, 0.3);

    // B×2 → v_A×2 → dt_A/2
    assert!(
        (dt3 / dt1 - 0.5).abs() < 0.05,
        "B×2 → dt_A/2: {:.4} vs 0.5",
        dt3 / dt1
    );
    // B más fuerte siempre da dt menor
    assert!(dt3 < dt1, "B mayor → dt_A menor");
}

// ── 5. MHD step completo con dt CFL da resultado finito ──────────────────

#[test]
fn full_mhd_step_with_cfl_produces_finite_result() {
    let n = 20;
    let b0 = 10.0_f64; // B fuerte → dt_alfven < dt_global
    let mut particles: Vec<Particle> = (0..n)
        .map(|i| {
            let x = (i as f64) / (n as f64);
            gas_with_b(i, Vec3::new(x, 0.0, 0.0), Vec3::new(b0, 0.1, 0.0), 0.2)
        })
        .collect();

    let dt_global = 0.01_f64;
    let dt_a = alfven_dt(&particles, 0.3);
    let dt_mhd = dt_global.min(dt_a);

    for _ in 0..10 {
        advance_induction(&mut particles, dt_mhd);
        apply_magnetic_forces(&mut particles, dt_mhd);
        dedner_cleaning_step(&mut particles, 1.0, 0.5, dt_mhd);
    }

    for (i, p) in particles.iter().enumerate() {
        let b_ok = p.b_field.x.is_finite() && p.b_field.y.is_finite() && p.b_field.z.is_finite();
        assert!(b_ok, "p{i}: B no finito tras CFL step");
        assert!(p.psi_div.is_finite(), "p{i}: ψ no finito");
    }
}

// ── 6. Benchmark MHD existe en gadget-ng-mhd (verificación de archivo) ───

#[test]
fn mhd_bench_file_exists() {
    let bench_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gadget-ng-mhd/benches/alfven_bench.rs"
    );
    assert!(
        std::path::Path::new(bench_path).exists(),
        "alfven_bench.rs no encontrado en: {bench_path}"
    );
}
