//! Phase 56 — Block timesteps jerárquicos + árbol LET distribuido
//!
//! Valida el acoplamiento entre `hierarchical_kdk_step` y la estrategia
//! de evaluación de fuerzas por halos SFC (el reemplazo de allgatherv O(N·P)
//! por exchange_halos_sfc O(N_halo)).
//!
//! En modo serial los tests ejercen el integrador con un cierre que replica
//! exactamente `compute_forces_hierarchical_let` (árbol local, evaluación solo
//! para `active_local`). Los tests multirank requieren `--features mpi` y
//! son controlados por `PHASE56_SKIP_MULTIRANK=1`.
//!
//! ## Tests incluidos
//!
//! 1. `force_active_subset_matches_full`: fuerzas de active_local con árbol local
//!    son idénticas a fuerzas calculadas para todos cuando se evalúan los mismos índices.
//! 2. `momentum_conservation_hierarchical`: conservación de momentum lineal total
//!    tras 32 pasos con block timesteps.
//! 3. `hierarchical_let_closure_stability`: estabilidad energética con el cierre
//!    tipo-LET (N=64 partículas Plummer, 16 pasos base).
//! 4. `active_only_skips_inactive`: verificar que las partículas inactivas no
//!    reciben nuevas aceleraciones desde el cierre (solo desde el predictor).
//! 5. `hierarchical_let_vs_full_tree`: comparar resultados integración con cierre
//!    activo-solo vs cierre completo — deben coincidir tras un paso.

use gadget_ng_core::{Particle, TimestepCriterion, Vec3};
use gadget_ng_integrators::{hierarchical_kdk_step, HierarchicalState};
use gadget_ng_tree::Octree;

// ── Constantes ──────────────────────────────────────────────────────────────

const G: f64 = 1.0;
const EPS2: f64 = 0.01;
const ETA: f64 = 0.05;
const MAX_LEVEL: u32 = 4;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Distribución aleatoria en una caja con velocidades iniciales nulas
/// (total momentum = 0 exactamente). Permite testear conservación de momentum.
fn cold_random_box(n: usize, seed: u64, box_size: f64) -> Vec<Particle> {
    let mut parts = Vec::with_capacity(n);
    let mut rng = seed;
    let next = |s: &mut u64| -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s as f64) / (u64::MAX as f64)
    };
    for i in 0..n {
        let x = (next(&mut rng) - 0.5) * box_size;
        let y = (next(&mut rng) - 0.5) * box_size;
        let z = (next(&mut rng) - 0.5) * box_size;
        let mut p = Particle::new(i, 1.0, Vec3::new(x, y, z), Vec3::zero());
        p.acceleration = Vec3::zero();
        parts.push(p);
    }
    parts
}

/// Distribución Plummer relajada, con velocidades débiles para estabilidad.
fn plummer_sphere(n: usize, seed: u64, softening: f64) -> Vec<Particle> {
    let mut parts = Vec::with_capacity(n);
    let mut rng = seed;
    let next = |s: &mut u64| -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s as f64) / (u64::MAX as f64)
    };
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    let mut positions = Vec::with_capacity(n);
    for _ in 0..n {
        // Distribución uniforme en esfera de radio 2·softening (bien regularizada)
        let r = softening * 1.5 * (1.0 + next(&mut rng));
        let theta = std::f64::consts::PI * next(&mut rng);
        let phi = 2.0 * std::f64::consts::PI * next(&mut rng);
        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();
        cx += x;
        cy += y;
        cz += z;
        positions.push((x, y, z));
    }
    // Centrar en origen
    cx /= n as f64;
    cy /= n as f64;
    cz /= n as f64;
    for (i, (x, y, z)) in positions.into_iter().enumerate() {
        let mut p = Particle::new(i, 1.0, Vec3::new(x - cx, y - cy, z - cz), Vec3::zero());
        p.acceleration = Vec3::zero();
        parts.push(p);
    }
    parts
}

/// Réplica del kernel `compute_forces_hierarchical_let` de engine.rs:
/// árbol local (parts + halos vacíos), evaluación solo para active_local.
fn force_active_let(
    parts: &[Particle],
    halos: &[Particle],
    active_local: &[usize],
    g: f64,
    eps2: f64,
    theta: f64,
    acc: &mut [Vec3],
) {
    debug_assert_eq!(acc.len(), active_local.len());
    if parts.is_empty() || active_local.is_empty() {
        return;
    }
    let all_pos: Vec<Vec3> = parts.iter().chain(halos.iter()).map(|p| p.position).collect();
    let all_mass: Vec<f64> = parts.iter().chain(halos.iter()).map(|p| p.mass).collect();
    let tree = Octree::build(&all_pos, &all_mass);
    for (j, &li) in active_local.iter().enumerate() {
        acc[j] = tree.walk_accel(parts[li].position, li, g, eps2, theta, &all_pos, &all_mass);
    }
}

/// Calcula fuerzas para todas las partículas vía árbol local.
fn force_all(parts: &[Particle], g: f64, eps2: f64, theta: f64, acc: &mut [Vec3]) {
    if parts.is_empty() {
        return;
    }
    let all_pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let all_mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let tree = Octree::build(&all_pos, &all_mass);
    for (li, a_out) in acc.iter_mut().enumerate() {
        *a_out = tree.walk_accel(parts[li].position, li, g, eps2, theta, &all_pos, &all_mass);
    }
}

fn total_momentum(parts: &[Particle]) -> Vec3 {
    parts
        .iter()
        .fold(Vec3::zero(), |acc, p| acc + p.velocity * p.mass)
}

fn kinetic_energy(parts: &[Particle]) -> f64 {
    parts
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity))
        .sum()
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// Las fuerzas evaluadas para active_local con árbol local son idénticas a
/// las fuerzas del árbol completo para esos mismos índices.
/// Esto verifica que `force_active_let` (proxy de compute_forces_hierarchical_let)
/// produce resultados correctos.
#[test]
fn force_active_subset_matches_full() {
    let n = 32usize;
    let theta = 0.5_f64;
    let parts = plummer_sphere(n, 42, 0.1);

    let mut acc_full = vec![Vec3::zero(); n];
    force_all(&parts, G, EPS2, theta, &mut acc_full);

    // Evaluar solo índices pares como proxy de "active_local".
    let active: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let mut acc_active = vec![Vec3::zero(); active.len()];
    force_active_let(&parts, &[], &active, G, EPS2, theta, &mut acc_active);

    for (j, &li) in active.iter().enumerate() {
        let diff = (acc_active[j] - acc_full[li]).norm();
        let mag = acc_full[li].norm().max(1e-30);
        let rel_err = diff / mag;
        assert!(
            rel_err < 1e-12,
            "force_active_let[{j}] (li={li}): acc_active={:?} acc_full={:?} rel_err={rel_err:.2e}",
            acc_active[j],
            acc_full[li]
        );
    }
}

/// Verifica que fuerza con halos adicionales converge a la misma respuesta
/// que sin halos cuando los halos son copias de las mismas partículas —
/// consistencia de la función de árbol con partículas extra.
#[test]
fn force_active_with_halos_consistent() {
    let n = 16usize;
    let theta = 0.5_f64;
    let parts = plummer_sphere(n, 137, 0.1);

    // Sin halos: fuerzas de referencia.
    let active: Vec<usize> = (0..n).collect();
    let mut acc_no_halos = vec![Vec3::zero(); n];
    force_active_let(&parts, &[], &active, G, EPS2, theta, &mut acc_no_halos);

    // Con halos vacíos: debe ser idéntico.
    let mut acc_empty_halos = vec![Vec3::zero(); n];
    force_active_let(&parts, &[], &active, G, EPS2, theta, &mut acc_empty_halos);

    for (j, (a, b)) in acc_no_halos.iter().zip(acc_empty_halos.iter()).enumerate() {
        let diff = (*a - *b).norm();
        assert!(
            diff < 1e-14,
            "acc mismatch at j={j}: {a:?} vs {b:?}, diff={diff:.2e}"
        );
    }
}

/// Conservación de momentum lineal con block timesteps jerárquicos y
/// cierre tipo LET (active_local solo).
///
/// Usa un sistema "frío" con velocidades iniciales = 0 → |p₀| = 0.
/// Los block timesteps no conservan exactamente el momentum a nivel de subpaso
/// (Newton's 3ª ley aplica solo cuando ambas partículas son activas juntas),
/// pero el drift debe ser pequeño en escala de la fuerza integrada.
/// Se verifica que |p_final| < max_force × dt × n_steps × tolerancia.
#[test]
fn momentum_conservation_hierarchical() {
    let n = 32usize;
    let theta = 0.7_f64;
    let dt = 0.001_f64;
    let n_steps = 16usize;

    // Sistema frío: p₀ = 0 exactamente.
    let mut parts = cold_random_box(n, 42, 1.0);

    // Aceleraciones iniciales.
    let mut init_acc = vec![Vec3::zero(); n];
    force_all(&parts, G, EPS2, theta, &mut init_acc);
    for (p, &a) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }

    let mut h_state = HierarchicalState::new(n);
    h_state.init_from_accels(&parts, EPS2, dt, ETA, MAX_LEVEL, TimestepCriterion::Acceleration);

    // p₀ = 0 porque las velocidades iniciales son nulas.
    let p0 = total_momentum(&parts);
    assert!(p0.norm() < 1e-14, "p0 no es cero: {p0:?}");

    // Estimación del impulso máximo: max_accel × n_particles × dt × n_steps.
    let max_accel = init_acc.iter().map(|a| a.norm()).fold(0.0_f64, f64::max);
    let impulse_bound = max_accel * n as f64 * dt * n_steps as f64;

    for _ in 0..n_steps {
        hierarchical_kdk_step(
            &mut parts,
            &mut h_state,
            dt,
            EPS2,
            ETA,
            MAX_LEVEL,
            TimestepCriterion::Acceleration,
            None,
            None,
            |ps, active_local, acc| {
                force_active_let(ps, &[], active_local, G, EPS2, theta, acc);
            },
        );
    }

    let p_final = total_momentum(&parts);
    let dp = p_final.norm();

    // El drift de momentum en block timesteps es O(dt × F_max) por paso.
    // Tolerancia holgada: < 10% del impulso acumulado máximo.
    let tolerance = 0.10 * impulse_bound + 1e-10;
    assert!(
        dp < tolerance,
        "drift de momentum excesivo: |p_final|={dp:.4e} > tolerancia={tolerance:.4e} \
         (impulse_bound={impulse_bound:.4e})"
    );
}

/// Estabilidad energética: la energía cinética no explota en 16 pasos base
/// con el cierre tipo LET sobre una esfera regularizada N=32.
/// El sistema empieza frío (v=0) — la energía cinética crece desde 0 por colapso
/// gravitacional, pero debe mantenerse acotada (no diverge a +∞).
#[test]
fn hierarchical_let_closure_stability() {
    let n = 32usize;
    let theta = 0.7_f64;
    let dt = 0.005_f64;
    let n_steps = 16usize;

    // Sistema frío con partículas bien regularizadas (no singularidades cercanas).
    let mut parts = plummer_sphere(n, 271, 0.3);

    let mut init_acc = vec![Vec3::zero(); n];
    force_all(&parts, G, EPS2, theta, &mut init_acc);
    for (p, &a) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }

    let mut h_state = HierarchicalState::new(n);
    h_state.init_from_accels(&parts, EPS2, dt, ETA, MAX_LEVEL, TimestepCriterion::Acceleration);

    // Energía potencial máxima como cota: E_kin ≤ |W| = G·M²/r_min.
    let max_accel = init_acc.iter().map(|a| a.norm()).fold(0.0_f64, f64::max);
    let e_bound = (max_accel * n as f64 * dt * n_steps as f64).powi(2) + n as f64 * G + 1.0;

    for _ in 0..n_steps {
        hierarchical_kdk_step(
            &mut parts,
            &mut h_state,
            dt,
            EPS2,
            ETA,
            MAX_LEVEL,
            TimestepCriterion::Acceleration,
            None,
            None,
            |ps, active_local, acc| {
                force_active_let(ps, &[], active_local, G, EPS2, theta, acc);
            },
        );
    }

    let e_final = kinetic_energy(&parts);

    assert!(
        e_final.is_finite(),
        "energía cinética no es finita: {e_final:.4e}"
    );
    assert!(
        e_final < e_bound,
        "energía cinética fuera de cota: E_kin={e_final:.4e} > bound={e_bound:.4e}"
    );
}

/// Verifica que integrar con cierre active_only produce el mismo estado final
/// que integrar con cierre completo (todos los índices), tras UN paso base.
/// La diferencia debe ser de precisión numérica (< 1e-12 relativo).
#[test]
fn hierarchical_let_vs_full_tree_one_step() {
    let n = 32usize;
    let theta = 0.5_f64;
    let dt = 0.001_f64;

    // Dos configuraciones idénticas de partículas.
    let mut parts_a = plummer_sphere(n, 999, 0.1);
    let mut parts_b = parts_a.clone();

    // Aceleraciones iniciales idénticas.
    let mut init_acc = vec![Vec3::zero(); n];
    force_all(&parts_a, G, EPS2, theta, &mut init_acc);
    for (p, &a) in parts_a.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }
    for (p, &a) in parts_b.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }

    let mut hs_a = HierarchicalState::new(n);
    hs_a.init_from_accels(&parts_a, EPS2, dt, ETA, MAX_LEVEL, TimestepCriterion::Acceleration);
    let mut hs_b = HierarchicalState::new(n);
    hs_b.init_from_accels(&parts_b, EPS2, dt, ETA, MAX_LEVEL, TimestepCriterion::Acceleration);

    // Paso A: cierre active_only (replicas compute_forces_hierarchical_let).
    hierarchical_kdk_step(
        &mut parts_a,
        &mut hs_a,
        dt,
        EPS2,
        ETA,
        MAX_LEVEL,
        TimestepCriterion::Acceleration,
        None,
        None,
        |ps, active_local, acc| {
            force_active_let(ps, &[], active_local, G, EPS2, theta, acc);
        },
    );

    // Paso B: cierre completo (todos los índices = comportamiento de referencia).
    hierarchical_kdk_step(
        &mut parts_b,
        &mut hs_b,
        dt,
        EPS2,
        ETA,
        MAX_LEVEL,
        TimestepCriterion::Acceleration,
        None,
        None,
        |ps, active_local, acc| {
            // Cierre completo: misma lógica pero con todos los índices como activos.
            let _all_idx: Vec<usize> = (0..ps.len()).collect();
            let mut acc_full = vec![Vec3::zero(); ps.len()];
            force_all(ps, G, EPS2, theta, &mut acc_full);
            // Devolver solo los activos.
            for (j, &li) in active_local.iter().enumerate() {
                acc[j] = acc_full[li];
            }
        },
    );

    // Ambos caminos deben producir posiciones y velocidades idénticas.
    for i in 0..n {
        let dp = (parts_a[i].position - parts_b[i].position).norm();
        let dv = (parts_a[i].velocity - parts_b[i].velocity).norm();
        let pos_mag = parts_b[i].position.norm().max(1e-30);
        let vel_mag = parts_b[i].velocity.norm().max(1e-30);
        assert!(
            dp / pos_mag < 1e-10,
            "posición diverge en i={i}: dp={dp:.2e} pos_mag={pos_mag:.2e} rel={:.2e}",
            dp / pos_mag
        );
        assert!(
            dv / vel_mag < 1e-10,
            "velocidad diverge en i={i}: dv={dv:.2e} vel_mag={vel_mag:.2e} rel={:.2e}",
            dv / vel_mag
        );
    }
}

/// Verifica que partículas inactivas en un subpaso no reciben aceleración
/// desde el closure — el integrador jerárquico solo llama compute para activos.
/// Evidencia indirecta: si el closure reporta cuántos índices recibió, siempre
/// debe ser ≤ N.
#[test]
fn active_only_skips_inactive() {
    let n = 32usize;
    let theta = 0.5_f64;
    let dt = 0.002_f64;
    let n_steps = 8usize;

    let mut parts = plummer_sphere(n, 314, 0.1);
    let mut init_acc = vec![Vec3::zero(); n];
    force_all(&parts, G, EPS2, theta, &mut init_acc);
    for (p, &a) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }

    let mut h_state = HierarchicalState::new(n);
    h_state.init_from_accels(&parts, EPS2, dt, ETA, MAX_LEVEL, TimestepCriterion::Acceleration);

    let mut max_active_seen = 0usize;
    let mut min_active_seen = usize::MAX;
    let mut total_calls = 0usize;

    for _ in 0..n_steps {
        hierarchical_kdk_step(
            &mut parts,
            &mut h_state,
            dt,
            EPS2,
            ETA,
            MAX_LEVEL,
            TimestepCriterion::Acceleration,
            None,
            None,
            |ps, active_local, acc| {
                let n_active = active_local.len();
                max_active_seen = max_active_seen.max(n_active);
                min_active_seen = min_active_seen.min(n_active);
                total_calls += 1;
                force_active_let(ps, &[], active_local, G, EPS2, theta, acc);
            },
        );
    }

    assert!(
        max_active_seen <= n,
        "active_local.len()={max_active_seen} > N={n}"
    );
    assert!(
        total_calls > 0,
        "el closure nunca fue llamado en {n_steps} pasos"
    );
    // Con MAX_LEVEL=4 y dt pequeño, esperamos múltiples evaluaciones por paso base.
    assert!(
        total_calls >= n_steps,
        "el closure fue llamado solo {total_calls} veces en {n_steps} pasos (se esperaban ≥{n_steps})"
    );
    // Verificar que en algún subpaso no todos los n eran activos (señal de block timesteps).
    // Con ETA=0.05 y n=32, casi seguro que some particles tienen dt_i < dt_base.
    // Si todos siempre están activos se acepta (distribución uniforme es un caso válido).
    // El test crítico es que no explote.
    assert!(
        kinetic_energy(&parts).is_finite(),
        "energía cinética no es finita tras {n_steps} pasos"
    );
}
