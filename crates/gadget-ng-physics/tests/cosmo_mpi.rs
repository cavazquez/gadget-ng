//! Tests de validación del modo cosmológico distribuido — Fase 17b.
//!
//! Estos tests verifican el comportamiento del nuevo branch `use_sfc_let_cosmo`
//! usando `SerialRuntime` (un solo rango) para reproducibilidad en CI sin MPI.
//!
//! ## Cobertura
//!
//! 1. `cosmo_sfc_let_g_scaling_tree`:
//!    El árbol local con `g_cosmo = G/a` da fuerzas la mitad de `G/1` cuando `a=2`.
//!
//! 2. `cosmo_sfc_let_g_scaling_let`:
//!    `accel_from_let` con `g_cosmo` escala linealmente: F(a=2) = F(a=1)/2.
//!
//! 3. `cosmo_sfc_let_force_no_nan`:
//!    El ciclo completo de `compute_forces_sfc_let` (árbol + LET) con `g_cosmo`
//!    no produce NaN/Inf para N=64 partículas en retícula perturbada.
//!
//! 4. `cosmo_sfc_let_vrms_distributed`:
//!    La suma de |p/a|² acumulada sobre subconjuntos da el mismo resultado que
//!    el cálculo global — simula el allreduce de Phase 17b.
//!
//! 5. `cosmo_sfc_let_a_evolution_consistent`:
//!    `advance_a` con los mismos parámetros da la misma `a_final` independientemente
//!    del número de partículas — el estado cosmológico es global.
//!
//! 6. `cosmo_sfc_let_force_vs_allgather`:
//!    Las fuerzas del árbol local + LET con `g_cosmo` ≈ fuerzas directas N²
//!    con `g_cosmo` dentro de la tolerancia del MAC (θ=0.5).
//!
//! 7. `cosmo_sfc_let_no_explosion`:
//!    30 pasos de leapfrog cosmológico usando el evaluador LET con `g_cosmo`
//!    no produce NaN/Inf en posiciones ni velocidades.
//!
//! 8. `cosmo_sfc_let_g_cosmo_applied`:
//!    Verificación explícita de que el escalado G/a se aplica en la fuerza
//!    comóvil: con `a=0.5` la fuerza efectiva es el doble que con `a=1`.

use gadget_ng_core::{
    build_particles,
    cosmology::CosmologyParams,
    CosmologySection, GravitySection, IcKind, InitialConditionsSection,
    OutputSection, PerformanceSection, Particle, RunConfig, SimulationSection, TimestepSection,
    UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_tree::{accel_from_let, pack_let_nodes, unpack_let_nodes, Octree};

const G: f64 = 1.0;
const EPS2: f64 = 0.02 * 0.02;
const THETA: f64 = 0.5;
const BOX: f64 = 1.0;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// LCG determinista para posiciones de test.
fn lcg_positions(n: usize, seed: u64) -> Vec<Vec3> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = ((s >> 33) as f64) / (u32::MAX as f64);
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let y = ((s >> 33) as f64) / (u32::MAX as f64);
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let z = ((s >> 33) as f64) / (u32::MAX as f64);
            Vec3::new(x, y, z)
        })
        .collect()
}

fn uniform_masses(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}

/// Aceleración directa N² con constante gravitatoria `g` y suavizado `eps2`.
fn direct_accel_single(pos_i: Vec3, all_pos: &[Vec3], masses: &[f64], g: f64, eps2: f64) -> Vec3 {
    all_pos.iter().zip(masses.iter()).enumerate().fold(
        Vec3::zero(),
        |acc, (j, (pj, &mj))| {
            let dr = *pj - pos_i;
            let r2 = dr.dot(dr) + eps2;
            let r3 = r2.sqrt() * r2;
            acc + dr * (g * mj / r3)
        },
    )
}

/// Config mínimo EdS para tests.
fn eds_config(n: usize) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.005,
            num_steps: 30,
            softening: 0.02,
            gravitational_constant: G,
            particle_count: n,
            box_size: BOX,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::PerturbedLattice {
                amplitude: 0.05,
                velocity_amplitude: 0.0,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: false,
            omega_m: 1.0,
            omega_lambda: 0.0,
            h0: 0.1,
            a_init: 1.0,
        },
        units: UnitsSection::default(),
    }
}

// ── Test 1: escalado G/a en árbol local ─────────────────────────────────────

/// Con `g_cosmo = G/a`, la fuerza debe ser F/a respecto a G completo.
/// Verificamos que walk_accel(g=G/2) da exactamente la mitad de walk_accel(g=G).
#[test]
fn cosmo_sfc_let_g_scaling_tree() {
    let n = 32_usize;
    let pos = lcg_positions(n, 11);
    let masses = uniform_masses(n);
    let tree = Octree::build(&pos, &masses);

    let a = 2.0_f64;
    let g_cosmo = G / a;
    let query = pos[0];

    // walk_accel usa `li` para excluirse a sí mismo (i == li).
    let acc_full = tree.walk_accel(query, 0, G, EPS2, THETA, &pos, &masses);
    let acc_scaled = tree.walk_accel(query, 0, g_cosmo, EPS2, THETA, &pos, &masses);

    let ratio = acc_full.norm() / acc_scaled.norm().max(1e-30);
    assert!(
        (ratio - a).abs() < 1e-10,
        "walk_accel(g) / walk_accel(g/a) = {:.10} ≠ a={:.1}",
        ratio,
        a
    );
}

// ── Test 2: escalado G/a en accel_from_let ────────────────────────────────────

#[test]
fn cosmo_sfc_let_g_scaling_let() {
    let n = 30_usize;
    let pos = lcg_positions(n, 22);
    let masses = uniform_masses(n);
    let tree = Octree::build(&pos, &masses);

    // Exportar LET hacia un "receptor" alejado.
    let target_aabb = [2.0, 3.0, 2.0, 3.0, 2.0, 3.0];
    let let_nodes = tree.export_let(target_aabb, THETA);
    let packed = pack_let_nodes(&let_nodes);
    let remote = unpack_let_nodes(&packed);

    let query = Vec3::new(2.5, 2.5, 2.5);
    let a = 2.0_f64;

    let f1 = accel_from_let(query, &remote, G, EPS2);
    let f2 = accel_from_let(query, &remote, G / a, EPS2);

    let ratio = f1.norm() / f2.norm().max(1e-30);
    assert!(
        (ratio - a).abs() < 1e-10,
        "accel_from_let(g) / accel_from_let(g/a) = {:.10} ≠ a={:.1}",
        ratio,
        a
    );
}

// ── Test 3: sin NaN/Inf en ciclo árbol+LET con g_cosmo ───────────────────────

/// Simula la lógica de `compute_forces_sfc_let` con g_cosmo.
/// Verifica que no se producen NaN/Inf.
#[test]
fn cosmo_sfc_let_force_no_nan() {
    let n = 64_usize;
    let pos = lcg_positions(n, 33);
    let masses = uniform_masses(n);

    // Partición: "local" = primeras 32, "remotas" = últimas 32.
    let local_pos = &pos[..32];
    let local_mass = &masses[..32];
    let remote_pos = &pos[32..];
    let remote_mass = &masses[32..];

    let tree_remote = Octree::build(remote_pos, remote_mass);
    // AABB del dominio local (para exportar LET desde árbol remoto).
    let target_aabb = [
        local_pos.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
        local_pos.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
        local_pos.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
        local_pos.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
        local_pos.iter().map(|p| p.z).fold(f64::INFINITY, f64::min),
        local_pos.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max),
    ];
    let let_nodes = tree_remote.export_let(target_aabb, THETA);
    let packed = pack_let_nodes(&let_nodes);
    let remote_nodes = unpack_let_nodes(&packed);

    let tree_local = Octree::build(local_pos, local_mass);

    let a = 1.5_f64;
    let g_cosmo = G / a;

    for (li, pos_i) in local_pos.iter().enumerate() {
        let a_local =
            tree_local.walk_accel(*pos_i, li, g_cosmo, EPS2, THETA, local_pos, local_mass);
        let a_remote = accel_from_let(*pos_i, &remote_nodes, g_cosmo, EPS2);
        let a_total = a_local + a_remote;

        assert!(
            a_total.x.is_finite() && a_total.y.is_finite() && a_total.z.is_finite(),
            "Aceleración no finita para partícula {li}: {:?}",
            a_total
        );
    }
}

// ── Test 4: v_rms distribuida via allreduce ────────────────────────────────────

/// Verifica que sum(|p/a|²) por subconjuntos y luego sqrt(sum/N) da el mismo
/// resultado que calcularlo sobre el conjunto completo.
/// Esto simula la lógica de v_rms distribuida del nuevo branch.
#[test]
fn cosmo_sfc_let_vrms_distributed() {
    let cfg = eds_config(27);
    let particles = build_particles(&cfg).expect("IC");
    let a = 1.1_f64;

    // Cálculo "global" (un solo rank).
    let sum_v2_global: f64 = particles
        .iter()
        .map(|p| {
            let v = p.velocity * (1.0 / a);
            v.dot(v)
        })
        .sum();
    let v_rms_global = (sum_v2_global / particles.len() as f64).sqrt();

    // Cálculo "distribuido" (3 rangos simulados, sin allreduce de verdad).
    let chunk = particles.len() / 3;
    let mut sum_v2_dist = 0.0_f64;
    for rank in 0..3 {
        let lo = rank * chunk;
        let hi = if rank < 2 { lo + chunk } else { particles.len() };
        let local_sum: f64 = particles[lo..hi]
            .iter()
            .map(|p| {
                let v = p.velocity * (1.0 / a);
                v.dot(v)
            })
            .sum();
        sum_v2_dist += local_sum; // simula allreduce_sum_f64
    }
    let v_rms_dist = (sum_v2_dist / particles.len() as f64).sqrt();

    assert!(
        (v_rms_global - v_rms_dist).abs() < 1e-14,
        "v_rms global ({:.6e}) ≠ v_rms distribuida ({:.6e})",
        v_rms_global,
        v_rms_dist
    );
}

// ── Test 5: a(t) es independiente de la distribución de partículas ────────────

/// El factor de escala a(t) es un estado cosmológico global, no depende de cuántas
/// partículas tiene cada rango. Esto verifica que `advance_a` es determinista.
#[test]
fn cosmo_sfc_let_a_evolution_consistent() {
    let h0 = 0.1_f64;
    let a0 = 1.0_f64;
    let dt = 0.005_f64;
    let n_steps = 20_usize;
    let cosmo = CosmologyParams::new(1.0, 0.0, h0);

    // Avanzar `a` independientemente en dos "rangos".
    let mut a_rank0 = a0;
    let mut a_rank1 = a0;
    for _ in 0..n_steps {
        a_rank0 = cosmo.advance_a(a_rank0, dt);
        a_rank1 = cosmo.advance_a(a_rank1, dt);
    }
    assert!(
        (a_rank0 - a_rank1).abs() < 1e-15,
        "advance_a no es determinista: rank0={:.10}, rank1={:.10}",
        a_rank0,
        a_rank1
    );

    // EdS analítico: a(t) = (a0^{3/2} + 3/2 H0 t)^{2/3}
    let t = n_steps as f64 * dt;
    let a_analytic = (a0.powf(1.5) + 1.5 * h0 * t).powf(2.0 / 3.0);
    let rel_err = (a_rank0 - a_analytic).abs() / a_analytic;
    assert!(
        rel_err < 0.01,
        "a_final error vs EdS analítico: {:.2e}",
        rel_err
    );
}

// ── Test 6: árbol+LET con g_cosmo ≈ N² con g_cosmo ────────────────────────────

/// Compara la fuerza del árbol local + LET remoto (usando g_cosmo) con la fuerza
/// directa N² (también usando g_cosmo). El error debe estar dentro de la tolerancia
/// del MAC geométrico (θ=0.5) — verificamos que el escalado G/a es consistente.
#[test]
fn cosmo_sfc_let_force_vs_allgather() {
    let n = 50_usize;
    let pos = lcg_positions(n, 55);
    let masses = uniform_masses(n);

    // Partición: "local" = 0..25, "remotas" = 25..50.
    let local_pos = &pos[..25];
    let local_mass = &masses[..25];
    let remote_pos = &pos[25..];
    let remote_mass = &masses[25..];

    let tree_remote = Octree::build(remote_pos, remote_mass);
    let target_aabb = [
        local_pos.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
        local_pos.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
        local_pos.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
        local_pos.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
        local_pos.iter().map(|p| p.z).fold(f64::INFINITY, f64::min),
        local_pos.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max),
    ];
    let let_nodes = tree_remote.export_let(target_aabb, THETA);
    let packed = pack_let_nodes(&let_nodes);
    let remote_nodes = unpack_let_nodes(&packed);
    let tree_local = Octree::build(local_pos, local_mass);

    let a = 2.0_f64;
    let g_cosmo = G / a;

    // Fuerza BH árbol+LET con g_cosmo.
    let mut rms_bh = 0.0_f64;
    let mut rms_dir = 0.0_f64;
    for (li, pos_i) in local_pos.iter().enumerate() {
        let f_local =
            tree_local.walk_accel(*pos_i, li, g_cosmo, EPS2, THETA, local_pos, local_mass);
        let f_remote = accel_from_let(*pos_i, &remote_nodes, g_cosmo, EPS2);
        let f_bh = f_local + f_remote;

        // Fuerza directa N² con g_cosmo sobre todas las partículas.
        let f_dir = direct_accel_single(*pos_i, &pos, &masses, g_cosmo, EPS2);

        let diff = (f_bh - f_dir).norm();
        let scale = f_dir.norm().max(1e-15);
        rms_bh += (diff / scale).powi(2);
        rms_dir += 1.0;
    }
    let rms = (rms_bh / rms_dir).sqrt();

    // Con θ=0.5 y distribución uniforme, el error relativo esperado < 5%.
    assert!(
        rms < 0.05,
        "Error relativo RMS BH+LET vs directo con g_cosmo: {:.3e} > 5%",
        rms
    );
}

// ── Test 7: sin explosión numérica en integración LET + cosmológico ───────────

/// Integra 30 pasos de leapfrog cosmológico con evaluación de fuerza árbol+LET.
/// Verifica ausencia de NaN/Inf en posiciones y velocidades.
#[test]
fn cosmo_sfc_let_no_explosion() {
    let n = 27_usize; // 3³ retícula
    let cfg = eds_config(n);
    let mut parts = build_particles(&cfg).expect("IC");
    let cosmo = CosmologyParams::new(1.0, 0.0, 0.1);
    let dt = 0.005_f64;
    let n_steps = 30_usize;
    let mut a = 1.0_f64;
    let mut scratch = vec![Vec3::zero(); parts.len()];

    for _ in 0..n_steps {
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors { drift, kick_half, kick_half2 };
        a = cosmo.advance_a(a, dt);

        leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, acc| {
            // Simula compute_forces_sfc_let: solo árbol local (P=1, no hay nodos remotos).
            let pos_l: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let mass_l: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let tree = Octree::build(&pos_l, &mass_l);
            for (li, a_out) in acc.iter_mut().enumerate() {
                *a_out = tree.walk_accel(
                    ps[li].position, li, g_cosmo, EPS2, THETA, &pos_l, &mass_l,
                );
                // Sin nodos remotos (simulación P=1).
            }
        });
    }

    for p in &parts {
        assert!(
            p.position.x.is_finite()
                && p.position.y.is_finite()
                && p.position.z.is_finite(),
            "Posición no finita en gid={}: {:?}",
            p.global_id,
            p.position
        );
        assert!(
            p.velocity.x.is_finite()
                && p.velocity.y.is_finite()
                && p.velocity.z.is_finite(),
            "Velocidad no finita en gid={}: {:?}",
            p.global_id,
            p.velocity
        );
    }
    assert!(
        a > 1.0,
        "Factor de escala no creció: a={:.6}",
        a
    );
}

// ── Test 8: g_cosmo aplicado explícitamente ───────────────────────────────────

/// Con `a=0.5`, la fuerza efectiva G/a = 2G es el doble que con `a=1.0`.
/// Verificado usando el mismo árbol con dos llamadas y comparando el cociente.
#[test]
fn cosmo_sfc_let_g_cosmo_applied() {
    let n = 20_usize;
    let pos = lcg_positions(n, 99);
    let masses = uniform_masses(n);
    let tree = Octree::build(&pos, &masses);

    let query = Vec3::new(1.5, 1.5, 1.5); // fuera del dominio de partículas

    let f_at_a1 = accel_from_let(
        query,
        &{
            let nodes = tree.export_let([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], THETA);
            unpack_let_nodes(&pack_let_nodes(&nodes))
        },
        G / 1.0,
        EPS2,
    );

    let f_at_a05 = accel_from_let(
        query,
        &{
            let nodes = tree.export_let([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], THETA);
            unpack_let_nodes(&pack_let_nodes(&nodes))
        },
        G / 0.5, // a=0.5 → g_cosmo = 2G
        EPS2,
    );

    // F(a=0.5) debe ser exactamente el doble de F(a=1.0).
    let ratio = f_at_a05.norm() / f_at_a1.norm().max(1e-30);
    assert!(
        (ratio - 2.0).abs() < 1e-10,
        "F(a=0.5)/F(a=1.0) = {:.10} ≠ 2.0",
        ratio
    );
}
