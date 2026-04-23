# Phase 98 — EoR con química acoplada

**Fecha:** 2026-04-23  
**Crates afectados:** `gadget-ng-cli`, `gadget-ng-physics`  
**Tipo:** Corrección de bug + mejora de fidelidad física

---

## Objetivo

Hacer que el paso de reionización en `engine.rs` use los `ChemState` reales de
las partículas de gas (fracción de ionización x_HII acumulada paso a paso) en
lugar de un vector scratch descartado que se reinicializaba vacío en cada llamada.

## Problema anterior

En la Phase 95, `maybe_reionization!` declaraba un scratch local:

```rust
let mut _reion_chem: Vec<gadget_ng_rt::ChemState> = Vec::new();
let _reion_state = gadget_ng_rt::reionization_step(
    rf,
    &mut _reion_chem,   // ← vacío, descartado al salir del bloque
    ...
);
```

Consecuencia: el estado de ionización nunca se acumulaba. Cada paso partía desde
cero (gas 100% neutro), lo que hacía imposible simular la evolución temporal del
frente de ionización.

## Solución

### `crates/gadget-ng-cli/src/engine.rs`

**1. Vector de química global (`sph_chem_states`)**

Declarado junto al campo de radiación `rt_field_opt`, antes de las macros:

```rust
let mut sph_chem_states: Vec<gadget_ng_rt::ChemState> = if cfg.reionization.enabled {
    local.iter()
        .map(|_| gadget_ng_rt::ChemState::neutral())
        .collect()
} else {
    Vec::new()
};
```

Un `ChemState` por partícula, paralelo a `local[]`. Se inicializa en estado
completamente neutro (x_HII = 0) al inicio de la simulación.

**2. Macro `maybe_reionization!` actualizada**

- Sincroniza `sph_chem_states.len()` con `local.len()` antes de cada paso
  (maneja el caso en que carguen más partículas en un restart)
- Pasa `&mut sph_chem_states` a `reionization_step()` → el estado **persiste**

```rust
// Sincronización de longitud
if sph_chem_states.len() < local.len() {
    let extra = local.len() - sph_chem_states.len();
    sph_chem_states.extend(
        std::iter::repeat(gadget_ng_rt::ChemState::neutral()).take(extra)
    );
} else if sph_chem_states.len() > local.len() {
    sph_chem_states.truncate(local.len());
}

// Paso de reionización con química acoplada real
let _reion_state = gadget_ng_rt::reionization_step(
    rf,
    &mut sph_chem_states,  // ← estados reales, persisten
    &sources,
    &m1p,
    cfg.simulation.dt,
    bsz,
    _z_eor,
);
```

**3. Macro `maybe_insitu!` actualizada**

```rust
macro_rules! maybe_insitu {
    ($step:expr) => {
        crate::insitu::maybe_run_insitu(
            &local,
            &cfg.insitu_analysis,
            cfg.simulation.box_size,
            a_current,
            $step,
            out_dir,
            // Inyectar química real si está disponible
            if sph_chem_states.is_empty() { None } else { Some(&sph_chem_states) },
        );
    };
}
```

### `crates/gadget-ng-cli/src/insitu.rs`

Nueva firma de `maybe_run_insitu`:

```rust
pub fn maybe_run_insitu(
    particles: &[Particle],
    cfg: &InsituAnalysisSection,
    box_size: f64,
    a: f64,
    step: u64,
    default_out_dir: &Path,
    chem_states_opt: Option<&[gadget_ng_rt::ChemState]>,  // NUEVO
) -> bool
```

El cálculo de estadísticas 21cm usa los estados reales:

```rust
let chem_for_21cm: Vec<gadget_ng_rt::ChemState> = if let Some(chem) = chem_states_opt {
    gas_indices.iter()
        .map(|&i| {
            if i < chem.len() { chem[i].clone() }
            else { gadget_ng_rt::ChemState::neutral() }
        })
        .collect()
} else {
    vec![gadget_ng_rt::ChemState::neutral(); gas_particles.len()]
};
```

## Test nuevo: `coupled_chem_reduces_21cm_signal`

**Archivo:** `crates/gadget-ng-physics/tests/phase95_eor.rs`

Verifica que la señal 21cm refleja correctamente el estado de ionización real:

```rust
// Caso 1: gas neutro (x_HII = 0) → señal máxima
let out_neutral = compute_cm21_output(&particles, &chem_neutral, ...);

// Caso 2: gas 50% ionizado (x_HII = 0.5) → señal reducida
let out_half = compute_cm21_output(&particles, &chem_half, ...);

// Verificación: reducción ~50%
assert!(out_half.delta_tb_mean < out_neutral.delta_tb_mean);
let ratio = out_half.delta_tb_mean / out_neutral.delta_tb_mean;
assert!((ratio - 0.5).abs() < 0.05);
```

Resultado: `coupled_chem_reduces_21cm_signal ... ok` ✅

## Comportamiento físico ahora correcto

| Paso | Antes (Phase 95) | Ahora (Phase 98) |
|------|-----------------|-----------------|
| t=0 | x_HII = 0 (neutro) | x_HII = 0 (neutro) |
| t=1 | x_HII = 0 (**reiniciado**) | x_HII > 0 (acumulado) |
| t=N | x_HII = 0 siempre | x_HII → 1 (reionización avanza) |
| Señal 21cm | Siempre máxima | Decrece correctamente con z |

## Estado

✅ Implementado, testeado (7/7 tests) y commiteado en `main` (commit `8a9a512`)
