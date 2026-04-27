use gadget_ng_analysis::{
    AnalysisParams, analyse, read_halo_catalog, read_power_spectrum, write_halo_catalog,
    write_power_spectrum,
};
use gadget_ng_core::{Particle, Vec3};

/// Genera una lattice cúbica de N³ partículas de masa uniforme dentro de una caja.
fn lattice_particles(side: usize, box_size: f64) -> Vec<Particle> {
    let n = side * side * side;
    let spacing = box_size / side as f64;
    let mass = 1.0 / n as f64;
    let mut out = Vec::with_capacity(n);
    for ix in 0..side {
        for iy in 0..side {
            for iz in 0..side {
                let gid = ix * side * side + iy * side + iz;
                let pos = Vec3::new(
                    (ix as f64 + 0.5) * spacing,
                    (iy as f64 + 0.5) * spacing,
                    (iz as f64 + 0.5) * spacing,
                );
                out.push(Particle::new(gid, mass, pos, Vec3::zero()));
            }
        }
    }
    out
}

/// Genera dos cúmulos esféricos densos separados en la caja.
fn two_cluster_particles(cluster_n: usize, box_size: f64) -> Vec<Particle> {
    use std::f64::consts::PI;
    let mut out = Vec::new();
    let centers = [
        Vec3::new(box_size * 0.25, box_size * 0.5, box_size * 0.5),
        Vec3::new(box_size * 0.75, box_size * 0.5, box_size * 0.5),
    ];
    let r = box_size * 0.05;
    for (halo_idx, &center) in centers.iter().enumerate() {
        for i in 0..cluster_n {
            let phi = 2.0 * PI * (i as f64 / cluster_n as f64);
            let theta = PI * (i as f64 / cluster_n as f64 * 0.5 + 0.1);
            let rr = r * (0.3 + 0.7 * (i as f64 / cluster_n as f64));
            let pos = center
                + Vec3::new(
                    rr * theta.sin() * phi.cos(),
                    rr * theta.sin() * phi.sin(),
                    rr * theta.cos(),
                );
            out.push(Particle::new(
                halo_idx * cluster_n + i,
                1.0 / (2 * cluster_n) as f64,
                pos,
                Vec3::zero(),
            ));
        }
    }
    out
}

#[test]
fn analyse_lattice_no_halos_pk_finite() {
    let particles = lattice_particles(4, 1.0); // 64 partículas uniformes
    let params = AnalysisParams {
        box_size: 1.0,
        b: 0.2,
        min_particles: 10,
        pk_mesh: 8,
        ..Default::default()
    };
    let result = analyse(&particles, &params);
    // Lattice uniforme: separación = l̄, ningún par conectado con b=0.2
    assert_eq!(
        result.halos.len(),
        0,
        "lattice uniforme no debe producir halos"
    );
    // P(k) debe tener bins y ser finito.
    assert!(!result.power_spectrum.is_empty(), "P(k) debe tener bins");
    for b in &result.power_spectrum {
        assert!(b.pk.is_finite(), "P(k={}) no debe ser NaN/inf", b.k);
        assert!(b.pk >= 0.0, "P(k={}) debe ser >= 0", b.k);
    }
}

#[test]
fn analyse_two_clusters_finds_two_halos() {
    let particles = two_cluster_particles(50, 1.0); // 2 cúmulos de 50 partículas
    let params = AnalysisParams {
        box_size: 1.0,
        b: 0.2,
        min_particles: 20,
        pk_mesh: 8,
        ..Default::default()
    };
    let result = analyse(&particles, &params);
    assert_eq!(
        result.halos.len(),
        2,
        "deben encontrarse exactamente 2 halos"
    );
    // Halos ordenados por masa descendente; deben ser similares.
    let m0 = result.halos[0].mass;
    let m1 = result.halos[1].mass;
    assert!(
        (m0 - m1).abs() / m0 < 0.05,
        "masas de ambos halos deben ser similares"
    );
    // Propiedades físicas coherentes.
    for h in &result.halos {
        assert!(h.n_particles >= 20);
        assert!(h.mass > 0.0);
        assert!(h.velocity_dispersion.is_finite());
        assert!(h.r_vir > 0.0);
    }
}

#[test]
fn catalog_roundtrip_halos_and_pk() {
    let dir = tempfile::tempdir().unwrap();
    let particles = two_cluster_particles(30, 1.0);
    let params = AnalysisParams {
        box_size: 1.0,
        b: 0.2,
        min_particles: 10,
        pk_mesh: 8,
        ..Default::default()
    };
    let result = analyse(&particles, &params);

    write_halo_catalog(dir.path(), &result.halos).unwrap();
    write_power_spectrum(dir.path(), &result.power_spectrum).unwrap();

    let halos_rt = read_halo_catalog(dir.path()).unwrap();
    let pk_rt = read_power_spectrum(dir.path()).unwrap();

    assert_eq!(halos_rt.len(), result.halos.len());
    assert_eq!(pk_rt.len(), result.power_spectrum.len());

    for (a, b) in result.halos.iter().zip(halos_rt.iter()) {
        assert_eq!(a.halo_id, b.halo_id);
        assert_eq!(a.n_particles, b.n_particles);
        assert!((a.mass - b.mass).abs() < 1e-12);
    }
}
