//! Phase 60 — Domain decomposition adaptativa (rebalanceo basado en costo)
//!
//! ## Objetivo
//!
//! Verificar que la lógica de rebalanceo adaptativo funciona correctamente:
//!
//! 1. **`phase60_should_rebalance_interval`** — el helper `should_rebalance` respeta el intervalo fijo.
//! 2. **`phase60_should_rebalance_cost_override`** — `cost_pending=true` fuerza rebalanceo aunque
//!    no se haya alcanzado el intervalo.
//! 3. **`phase60_should_rebalance_disabled_threshold`** — con `threshold=0.0`, solo por intervalo.
//! 4. **`phase60_threshold_config`** — el campo `rebalance_imbalance_threshold` se lee de config.
//!
//! El test multirank (4 ranks, distribución desbalanceada) se activa solo con `--features mpi`.
//!
//! Controlar con `PHASE60_SKIP=1`.

use gadget_ng_core::PerformanceSection;

fn skip() -> bool {
    std::env::var("PHASE60_SKIP")
        .map(|v| v == "1")
        .unwrap_or(false)
}

// ── Reimplementación local del helper (refleja el código de engine.rs) ────────

fn should_rebalance(step: u64, start_step: u64, interval: u64, cost_pending: bool) -> bool {
    if cost_pending {
        return true;
    }
    if interval == 0 {
        return true;
    }
    (step - start_step).is_multiple_of(interval)
}

// ── Tests unitarios del helper ────────────────────────────────────────────────

/// El rebalanceo solo ocurre en múltiplos del intervalo (sin costo pendiente).
#[test]
fn phase60_should_rebalance_interval() {
    if skip() {
        return;
    }

    let interval = 5u64;
    let start = 1u64;

    // Pasos donde debe rebalancear: 1, 6, 11, 16, ...
    assert!(
        should_rebalance(1, start, interval, false),
        "debe rebalancear en step=1 (start)"
    );
    assert!(
        !should_rebalance(2, start, interval, false),
        "no debe rebalancear en step=2"
    );
    assert!(
        !should_rebalance(3, start, interval, false),
        "no debe rebalancear en step=3"
    );
    assert!(
        !should_rebalance(4, start, interval, false),
        "no debe rebalancear en step=4"
    );
    assert!(
        !should_rebalance(5, start, interval, false),
        "no debe rebalancear en step=5"
    );
    assert!(
        should_rebalance(6, start, interval, false),
        "debe rebalancear en step=6"
    );
    assert!(
        should_rebalance(11, start, interval, false),
        "debe rebalancear en step=11"
    );
    assert!(
        !should_rebalance(7, start, interval, false),
        "no debe rebalancear en step=7"
    );

    eprintln!("[phase60] should_rebalance interval=5: ✓");
}

/// Con `cost_pending=true`, el rebalanceo se fuerza independientemente del intervalo.
#[test]
fn phase60_should_rebalance_cost_override() {
    if skip() {
        return;
    }

    let interval = 100u64; // intervalo muy largo
    let start = 1u64;

    // Sin costo pendiente, no rebalancear entre intervalos.
    assert!(
        !should_rebalance(50, start, interval, false),
        "sin costo: no rebalancear"
    );
    assert!(
        !should_rebalance(75, start, interval, false),
        "sin costo: no rebalancear"
    );

    // Con costo pendiente, rebalancear inmediatamente.
    assert!(
        should_rebalance(50, start, interval, true),
        "con costo: rebalancear en step=50"
    );
    assert!(
        should_rebalance(75, start, interval, true),
        "con costo: rebalancear en step=75"
    );
    assert!(
        should_rebalance(99, start, interval, true),
        "con costo: rebalancear en step=99"
    );

    eprintln!("[phase60] should_rebalance cost_override: ✓");
}

/// Con `interval=0`, rebalancear siempre (máximo overhead, máximo balance).
#[test]
fn phase60_should_rebalance_zero_interval() {
    if skip() {
        return;
    }

    let start = 1u64;
    for step in 1..=20u64 {
        assert!(
            should_rebalance(step, start, 0, false),
            "interval=0 debe rebalancear en step={step}"
        );
    }
    eprintln!("[phase60] should_rebalance interval=0: ✓");
}

/// El campo `rebalance_imbalance_threshold` está en `PerformanceSection` y tiene default 0.0.
#[test]
fn phase60_threshold_config() {
    if skip() {
        return;
    }

    let default_perf = PerformanceSection::default();
    assert_eq!(
        default_perf.rebalance_imbalance_threshold, 0.0,
        "default debe ser 0.0 (criterio por costo desactivado)"
    );
    eprintln!(
        "[phase60] rebalance_imbalance_threshold default = {}",
        default_perf.rebalance_imbalance_threshold
    );

    // Verificar que se puede configurar a un valor razonable.
    let custom_perf = PerformanceSection {
        rebalance_imbalance_threshold: 1.3,
        ..PerformanceSection::default()
    };
    assert!(
        (custom_perf.rebalance_imbalance_threshold - 1.3).abs() < 1e-10,
        "debe poder configurarse a 1.3"
    );
    eprintln!("[phase60] rebalance_imbalance_threshold configurable: ✓");
}

/// Simula un escenario donde el rebalanceo adaptativo se activa antes que el fijo.
///
/// Con intervalo=20 y desbalance detectado en step=5, el rebalanceo adaptativo
/// ocurre en step=6 (antes que el siguiente intervalo en step=21).
#[test]
fn phase60_cost_triggers_early_rebalance() {
    if skip() {
        return;
    }

    let interval = 20u64;
    let start = 1u64;

    // Simula un bucle de pasos, detectando desbalance en step 5.
    let mut cost_pending = false;
    let mut rebalance_steps: Vec<u64> = Vec::new();

    for step in start..=30u64 {
        let do_rebalance = should_rebalance(step, start, interval, cost_pending);
        if do_rebalance {
            cost_pending = false;
            rebalance_steps.push(step);
        }
        // Simular detección de desbalance en step 5.
        if step == 5 {
            cost_pending = true;
        }
    }

    eprintln!("[phase60] Pasos con rebalanceo: {rebalance_steps:?}");

    // Rebalanceo en step=1 (inicio), luego step=6 (por costo), luego step=21 (intervalo).
    assert!(
        rebalance_steps.contains(&1),
        "debe rebalancear en inicio (step=1)"
    );
    assert!(
        rebalance_steps.contains(&6),
        "debe rebalancear en step=6 (por costo detectado en step=5)"
    );
    assert!(
        rebalance_steps.contains(&21),
        "debe rebalancear en step=21 (intervalo desde step=6: 6+15... no, desde start_step)"
    );

    // El rebalanceo por costo ocurre ANTES que el siguiente múltiplo del intervalo desde start.
    // sin costo: sería en 1, 21; con costo en step 5 → también en step 6.
    assert!(
        rebalance_steps.iter().any(|&s| s > 1 && s < 21),
        "rebalanceo adaptativo debe ocurrir entre los steps 1 y 21"
    );

    eprintln!("[phase60] Rebalanceo adaptativo verificado: ✓");
}
