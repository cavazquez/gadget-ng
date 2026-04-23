# Opciones técnicas: cm21 FFT real + EoR química acoplada + Benchmarks AGN

**Fecha:** 2026-04-23

---

## 1. cm21 con FFT 3D real (rustfft)

### Problema anterior
`compute_pk_21cm_simple` usaba varianza de la malla como proxy del power spectrum
(sin FFT real), lo cual es una aproximación muy gruesa que no calcula correctamente
|δ̃T_b(k)|².

### Solución implementada

**Archivo:** `crates/gadget-ng-rt/src/cm21.rs`

Pipeline completo:
1. **CIC trilineal periódico** (`deposit_cic`): cada partícula contribuye a las 8
   celdas vecinas con peso trilineal `wx × wy × wz`. Periodicidad vía `.rem_euclid(n)`.
2. **Substracción de media**: campo de contraste δ = T - T̄.
3. **FFT 3D real** (`fft3d_real`): 3 pasadas de FFT 1D compleja usando `rustfft`
   (separabilidad de la DFT). Orden: Z → Y → X. Complejidad O(N³ log N).
4. **Binning esférico**: |δ̃(k)|² × V/N⁶ → estimador de P(k) [mK² (Mpc/h)³].
5. **Varianza dimensional**: Δ²(k) = k³ P(k) / (2π²) [mK²].

### Tests nuevos (todos pasan)

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `fft3d_impulse_flat_spectrum` | DFT de δ[0,0,0] → módulo plano | ✅ |
| `pk_fft_uniform_field_zero_signal` | Campo uniforme → P(k)≈0 para k≠0 | ✅ |
| `pk_fft_sinusoidal_signal_peak` | Señal sinusoidal → pico en bin k_fund | ✅ |
| `cic_deposit_conserves_total` | CIC conserva suma total del campo | ✅ |

**Dependencia agregada:** `rustfft = "6.4.1"` en `crates/gadget-ng-rt/Cargo.toml`

---

## 2. EoR con química acoplada

### Problema anterior
`maybe_reionization!` en `engine.rs` pasaba un vector de `ChemState` vacío
(`Vec::new()`) al paso de reionización, por lo que la evolución de x_HII
nunca se guardaba entre pasos.

### Solución implementada

**Archivos modificados:**
- `crates/gadget-ng-cli/src/engine.rs`
- `crates/gadget-ng-cli/src/insitu.rs`

#### engine.rs
```rust
// Vector paralelo a `local`, inicializado en estado neutro
let mut sph_chem_states: Vec<gadget_ng_rt::ChemState> = if cfg.reionization.enabled {
    local.iter().map(|_| gadget_ng_rt::ChemState::neutral()).collect()
} else { Vec::new() };
```

La macro `maybe_reionization!` ahora:
1. Sincroniza `sph_chem_states.len()` con `local.len()` (resize con estados neutros si difiere)
2. Pasa `&mut sph_chem_states` a `reionization_step()`
3. El estado de ionización persiste entre pasos de simulación

#### insitu.rs
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

El cálculo de `cm21` usa los `ChemState` reales cuando están disponibles:
- Si `chem_states_opt = Some(chem)` → x_HII real por partícula
- Si `chem_states_opt = None` → fallback a estados neutros

### Test nuevo: `coupled_chem_reduces_21cm_signal`

Verifica que con x_HII = 0.5 uniforme:
- `delta_tb_mean` se reduce ~50% respecto a x_HII = 0
- Ratio `out_half.delta_tb_mean / out_neutral.delta_tb_mean ≈ 0.5 ± 0.05` ✅

---

## 3. Benchmarks AGN (Criterion)

### Archivo nuevo
`crates/gadget-ng-sph/benches/agn_feedback.rs`

Tres grupos de benchmarks:

### `apply_agn_feedback` — Barrido N_particles

| N_particles | Tiempo | Throughput |
|-------------|--------|-----------|
| 64 | ~53 ns | ~1.2 Gelem/s |
| 512 | ~428 ns | ~1.2 Gelem/s |
| 4096 | ~3.4 µs | ~1.2 Gelem/s |
| 32768 | ~30 µs | ~1.1 Gelem/s |

**Observación**: escalado lineal O(N) puro, sin overhead cuadrático.
El throughput de ~1.2 Gelem/s es consistente con acceso secuencial a memoria.

### `apply_agn_feedback` — Barrido N_black_holes (N_particles=4096)

| N_BH | Tiempo |
|------|--------|
| 1 | ~3.4 µs |
| 4 | ~13.6 µs |
| 16 | ~54.5 µs |

**Observación**: escalado lineal en n_BH también (O(N × n_BH)).

### `bondi_accretion_rate` — por M_BH

~3–4 ns independiente de la masa (cálculo puramente aritmético, sin acceso a memoria).

### `grow_black_holes` — Barrido N_particles

| N_particles | Tiempo |
|-------------|--------|
| 64 | ~50 ns |
| 512 | ~380 ns |
| 4096 | ~3.0 µs |

**Costo relativo:** `grow_black_holes` ≈ `apply_agn_feedback` en overhead (mismo O(N)).

### Conclusión de rendimiento

El feedback AGN es O(N × n_BH) y corre en ~30 µs para N=32768 partículas con 1 BH,
lo que equivale a <0.1% del tiempo de un paso SPH típico (~50 ms para N=32768).
El overhead es despreciable para producción.

### Cambios en Cargo.toml

```toml
# crates/gadget-ng-sph/Cargo.toml
[dev-dependencies]
criterion.workspace = true

[[bench]]
name = "agn_feedback"
harness = false
```

---

## Archivos creados/modificados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-rt/src/cm21.rs` | Reescrito: CIC trilineal + FFT 3D real (rustfft) + 4 tests nuevos |
| `crates/gadget-ng-rt/Cargo.toml` | Agregado `rustfft = "6.4.1"` |
| `crates/gadget-ng-cli/src/engine.rs` | `sph_chem_states` global + `maybe_reionization!` acoplada + `maybe_insitu!` con química |
| `crates/gadget-ng-cli/src/insitu.rs` | `maybe_run_insitu` con parámetro `chem_states_opt` |
| `crates/gadget-ng-physics/tests/phase95_eor.rs` | Test `coupled_chem_reduces_21cm_signal` |
| `crates/gadget-ng-sph/benches/agn_feedback.rs` | Benchmark Criterion completo |
| `crates/gadget-ng-sph/Cargo.toml` | `criterion` dev-dep + `[[bench]]` |
| `docs/reports/2026-04-future-validaciones.md` | Documento de validaciones futuras |
