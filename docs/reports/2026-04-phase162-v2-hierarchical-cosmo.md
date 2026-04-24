# Phase 162 — V2: Block Timesteps + Cosmología + MPI Acoplado

**Fecha:** 2026-04-23  
**Status:** ✅ Implementado y verificado

---

## Resumen

Se separó el flag `use_hierarchical_let` en dos: `use_hierarchical_let_newton`
(comportamiento previo sin cosmología) y `use_hierarchical_let_cosmo` (nuevo path
que habilita `hierarchical_kdk_step` + cosmología en multirank).

---

## Cambio central en `engine.rs`

```rust
// ANTES (un solo flag — cosmología bloqueada):
let use_hierarchical_let = cfg.gravity.solver == SolverKind::BarnesHut
    && cfg.timestep.hierarchical
    && !cfg.cosmology.enabled   // ← cosmología bloqueada
    && rt.size() > 1 && ...

// DESPUÉS (dos flags independientes):
let use_hierarchical_let_newton = ...  // sin cosmología (igual que antes)
let use_hierarchical_let_cosmo = cfg.gravity.solver == SolverKind::BarnesHut
    && cfg.timestep.hierarchical
    && cfg.cosmology.enabled
    && !cfg.cosmology.periodic  // BH no soporta caja periódica
    && rt.size() > 1 && !cfg.performance.force_allgather_fallback;

let use_hierarchical_let = use_hierarchical_let_newton || use_hierarchical_let_cosmo;
```

El nuevo flag `use_hierarchical_let_cosmo` comparte toda la infraestructura SFC
existente (el alias `use_hierarchical_let` controla la inicialización de
`SfcDecomposition` y `HierarchicalState`).

---

## Tests V2 (5+1 en CI)

| Test | Física | Criterio | Resultado |
|---|---|---|---|
| `v2_mass_conserved_hierarchical_cosmo_10steps` | Masa total exacta | Δm = 0 | ✅ |
| `v2_energy_drift_cosmo_hierarchical_50steps` | Deriva energía < 10% | 50 pasos, dt=0.001 | ✅ |
| `v2_reproducible_serial_vs_hierarchical_cosmo` | Partícula libre sin fuerzas | dx < 1e-8 | ✅ |
| `v2_scale_factor_agreement_hierarchical_vs_friedmann` | a(t) vs RK4 Friedmann | err < 1% | ✅ |
| `v2_checkpoint_resume_cosmo_hierarchical` | Continuidad tras checkpoint | dx < 1e-10 | ✅ |
| `v2_strong_scaling_benchmark` | `#[ignore]` — MPI 4 ranks | eficiencia > 40% | 🔲 ignorado |

---

## Diagrama de flujo de paths en engine.rs

```
RunConfig
  └─ BarnesHut?
       ├─ No → PM / TreePM paths
       └─ Sí → hierarchical?
                 ├─ No → cosmo+MPI? → use_sfc_let_cosmo (no-jerárquico)
                 │         No → use_sfc_let (newtoniano)
                 └─ Sí → cosmo? 
                           ├─ No → use_hierarchical_let_newton (Phase 56)
                           └─ Sí → use_hierarchical_let_cosmo (Phase 162 ← NUEVO)
```

---

## Notas de integración

- El `cosmo_arg` en `hierarchical_kdk_step` ya existía antes de Phase 162;
  la guarda `!cfg.cosmology.enabled` en el flag simplemente impedía que se activara.
- `use_sfc_let_cosmo` ya tenía `!cfg.timestep.hierarchical` desde Phase 17b;
  se añadió documentación explicativa.
- Para habilitar el path en una simulación:
  ```toml
  [gravity]
  solver = "barnes_hut"
  
  [timestep]
  hierarchical = true
  
  [cosmology]
  enabled = true
  periodic = false  # requerido: BH no soporta fuerzas periódicas
  ```
