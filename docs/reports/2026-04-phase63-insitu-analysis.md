# Phase 63 — Análisis in-situ en el loop stepping

**Fecha:** abril 2026  
**Crates:** `gadget-ng-core` (config), `gadget-ng-cli` (insitu.rs + engine.rs)  
**Archivos nuevos/modificados:**  
- `crates/gadget-ng-core/src/config.rs` (`InsituAnalysisSection`)  
- `crates/gadget-ng-cli/src/insitu.rs` (nuevo)  
- `crates/gadget-ng-cli/src/engine.rs` (macro `maybe_insitu!`, 7 loops)  
- `crates/gadget-ng-physics/tests/phase63_insitu_analysis.rs`

---

## Contexto

Antes de Phase 63, el análisis post-proceso requería guardar snapshots completos
y analizarlos externamente. Para corridas largas esto es costoso en I/O.
El análisis in-situ calcula P(k), FoF y ξ(r) directamente sobre las partículas
en memoria durante el loop `stepping`, sin necesidad de snapshots intermedios.

---

## Configuración TOML

```toml
[insitu_analysis]
enabled      = true
interval     = 20          # cada 20 pasos (0 = desactivado)
pk_mesh      = 32          # grid para P(k), por lado
fof_b        = 0.2         # parámetro de enlace FoF
fof_min_part = 20          # mínimo de partículas para un halo
xi_bins      = 10          # bins para ξ(r), 0 = desactivado
output_dir   = "runs/cosmo/insitu"  # None → <out>/insitu/
```

### `InsituAnalysisSection` en `RunConfig`

```rust
pub struct RunConfig {
    // ...campos existentes...
    pub insitu_analysis: InsituAnalysisSection,  // default: enabled=false
}
```

---

## Implementación

### `maybe_run_insitu`

```rust
pub fn maybe_run_insitu(
    particles: &[Particle],
    cfg: &InsituAnalysisSection,
    box_size: f64,
    a: f64,        // factor de escala actual (1.0 en Newtoniano)
    step: u64,
    default_out_dir: &Path,
) -> bool
```

Guarda: condición `enabled && interval > 0 && step % interval == 0`.

### Macro en `engine.rs`

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
        );
    };
}
```

Insertada en los **7 loops de stepping** del motor (Leapfrog serial, Leapfrog SFC,
BarnesHut SFC+LET, jerárquico LET, cosmológico SFC, TreePM, allgather legacy).

---

## Formato de salida

**Archivo:** `<output_dir>/insitu_{step:06}.json`

```json
{
  "step": 20,
  "a": 0.35,
  "z": 1.857,
  "n_halos": 47,
  "m_total_halos": 3.2e14,
  "power_spectrum": [
    { "k": 0.628, "pk": 1450.3, "n_modes": 8 },
    { "k": 1.257, "pk": 987.1,  "n_modes": 24 }
  ],
  "xi_r": [
    { "r": 5.0, "xi": 1.23 },
    { "r": 10.0, "xi": 0.45 }
  ]
}
```

---

## Tests

| Test | Descripción |
|------|-------------|
| `phase63_default_config_disabled` | `InsituAnalysisSection::default()` → `enabled=false`, `interval=0`, `pk_mesh=32`, `fof_b=0.2` |
| `phase63_interval_logic` | Pasos {1,2,3,4,7,8} no disparan; {5,10,15,20} sí disparan con `interval=5` |
| `phase63_disabled_no_output` | Con `enabled=false`, la condición `should_run` es siempre false |
| `phase63_analysis_params_defaults` | `AnalysisParams::default()` tiene `pk_mesh > 0` y `b ∈ (0,1)` |
| `phase63_pk_finite_on_uniform` | Lattice 4³, P(k) finito y k > 0 para todos los bins |

---

## Impacto en el motor

El hook se ejecuta **después** de `maybe_checkpoint!` y `maybe_snap_frame!` al final
de cada iteración del loop. La operación es O(N log N) por el FoF, lo que añade
latencia perceptible solo cuando `interval` es pequeño (< 10 pasos).

Recomendación: usar `interval ≥ 20` para corridas de producción.
