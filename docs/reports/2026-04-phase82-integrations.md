# Phase 82 — Integraciones in-situ + CLI

**Fecha**: 2026-04-23  
**Crates**: `gadget-ng-cli`, `gadget-ng-core`  
**Tipo**: Integración / fix

---

## Resumen

Conecta los módulos ya existentes (SPH, RT, bispectrum, assembly bias, catálogo HDF5)
al motor principal de simulación y al CLI, cerrando el ciclo implementación → ejecución.

---

## 82a — Fix: invocar `maybe_sph!` en el loop de stepping

**Problema detectado**: La macro `maybe_sph!` estaba definida en `engine.rs` (~L1198)
pero **nunca se invocaba** en ninguno de los 7 loops de integración.

**Fix aplicado**:
- Rediseño de la macro: de `maybe_sph!($cf:expr)` a `maybe_sph!($sph_step:expr)`.
- La macro construye internamente los `CosmoFactors` desde `cosmo_state` si la
  cosmología está activa, o usa `CosmoFactors::flat(dt)` en modo newtoniano.
- El seed de feedback usa `$sph_step as u64` (el índice del loop) en lugar de
  `tpm_step_count` que no estaba en scope por las reglas de hygiene de `macro_rules!`.
- Corregido `rank` → `rt.rank()` en la semilla de SN kicks.
- Agregado `maybe_sph!(step)` después de cada `maybe_insitu!(step)` en los 7 loops.

## 82b — Hook `maybe_rt!` en `engine.rs`

Nueva macro `maybe_rt!()` que:
- Se activa solo si `cfg.rt.enabled = true`.
- Inicializa `rt_field_opt: Option<RadiationField>` **antes** de las definiciones de
  macros (requisito de scope para `macro_rules!` en función body).
- Llama a `gadget_ng_rt::m1_update` y `radiation_gas_coupling_step` en cada paso.
- Agregado `maybe_rt!()` en los 7 loops de integración (después de `maybe_sph!`).
- `gadget-ng-rt` añadido como dependencia directa de `gadget-ng-cli/Cargo.toml`.

## 82c — Bispectrum + Assembly Bias en `insitu.rs`

Nuevos campos en `InsituResult`:
```rust
pub bk_equilateral: Vec<BkBinOut>,
pub assembly_bias: Option<AssemblyBiasOut>,
```

Nuevos campos en `InsituAnalysisSection` (`config.rs`):
```toml
[insitu_analysis]
bispectrum_bins = 20          # 0 = desactivado
assembly_bias_enabled = true
assembly_bias_smooth_r = 5.0
```

Lógica de cálculo:
- `bispectrum_equilateral(positions, masses, box_size, pk_mesh, n_bins)` si `bispectrum_bins > 0`.
- `compute_assembly_bias(...)` si `assembly_bias_enabled` y hay ≥ 4 halos.
- Proxy de spin: dispersión de velocidad del halo (in-situ sin membership de partículas).

## 82d — CLI `--hdf5-catalog` en `analyze_cmd.rs`

Nuevo campo `hdf5_catalog: bool` en `AnalyzeParams` y nuevo flag en `main.rs`:
```bash
gadget-ng analyze --snapshot out/snap --out analysis/ --hdf5-catalog
# → analysis/halos.hdf5   (con feature hdf5)
# → analysis/halos.jsonl  (sin feature hdf5)
```

El catálogo incluye: masa, posición, velocidad, R₂₀₀, Npart para cada halo FoF.

---

## Fix: tests con RunConfig incompleto

Añadido `rt: Default::default()` a todas las inicializaciones de `RunConfig` en tests
de los crates `gadget-ng-core`, `gadget-ng-integrators`, `gadget-ng-tree`,
`gadget-ng-parallel` y `gadget-ng-physics` (45 archivos).

---

## Tests

- Build completo del workspace: ✅
- `cargo test --workspace --exclude gadget-ng-physics`: ✅ todos los tests pasan.
