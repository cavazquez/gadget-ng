# AP-17: Cierre completo CUDA — Reporte de validación

**Fecha:** 2026-05-16  
**Hardware:** NVIDIA GTX 1050 Ti (sm_61, 4 GB VRAM)

---

## Resumen ejecutivo

AP-17 cierra todos los huecos CUDA restantes después de AP-16. Las piezas
atacadas fueron:

1. **Wirings faltantes** — kernels CUDA ya implementados que no estaban
   conectados al motor de stepping o al CLI.
2. **Funcionalidad incompleta** — IGM percentiles (t_median, t_p16, t_p84)
   devueltos como 0 en la GPU.
3. **Kernel pairwise nuevo** — conducción anisótropa / CR diffusion O(N²)
   reemplazando la aproximación campo-medio de AP-16.
4. **Correcciones de documentación** — Braginskii y reconexión marcados
   incorrectamente como pendientes SIMD-without-Rayon.

---

## Cambios implementados

### Parte 1 — Wirings faltantes

| Item | Archivo | Cambio |
|------|---------|--------|
| Dedner CUDA | `context.rs` | `compute_dedner_div_b` CPU + `try_dedner_cleaning` GPU bajo `cuda_mhd` |
| Halo spin real | `analyze_cmd.rs` | `find_halos_with_membership` → `halo_spin` CPU; CUDA via `try_halo_spin` |
| Flag `--xray` | `main.rs` + `analyze_cmd.rs` | `try_xray_luminosity` CUDA + `total_xray_luminosity` CPU fallback |
| 21cm insitu | `insitu.rs` | `CudaRtSolver::try_cm21_field` → `Cm21Output` (mean/sigma) en `InsituResult.cm21` |

### Parte 2 — Funcionalidad incompleta

| Item | Archivo | Cambio |
|------|---------|--------|
| IGM percentiles | `rt_kernels.cu` | Nuevo `rt_igm_compact_kernel` + `cuda_rt_igm_temp_full` |
| IGM percentiles Rust | `rt_solver.rs` | Sort del array compacto en host → `t_median`, `t_p16`, `t_p84` reales |

### Parte 3 — Kernel anisótropo O(N²)

| Item | Archivo | Cambio |
|------|---------|--------|
| `mhd_anisotropic_pair_kernel` | `mhd_kernels.cu` | Kernel O(N²) Wendland-C6, κ_eff = κ_⊥ + (κ_∥ − κ_⊥)cos²θ |
| `cuda_mhd_anisotropic_conduction` | `mhd_kernels.cu` | Wrapper C para térmica |
| `cuda_mhd_cr_diffusion_anisotropic` | `mhd_kernels.cu` | Wrapper C para CR |
| `try_anisotropic_conduction` | `mhd_solver.rs` | Wrapper Rust térmica |
| `try_cr_diffusion_anisotropic` | `mhd_solver.rs` | Wrapper Rust CR |
| Wiring context.rs | `context.rs` | Reemplaza `try_scalar_diffusion` con pairwise O(N²) |

### Parte 4 — Docs

- `2026-05-accelerator-parity-pending.md`: tablas AP-17 añadidas; Braginskii/reconnección corregidos
- `2026-05-simd-cuda-coverage.md`: filas Braginskii/reconnección SIMD-without-Rayon marcadas **implementado**; IGM percentiles y anisotropic O(N²) actualizados

---

## Resultados hardware (sm_61)

Todos los tests `#[ignore]` pasaron en GPU real:

```
cargo test -p gadget-ng-cuda -- --ignored
```

### Nuevos tests AP-17

| Test | Estado | Notas |
|------|--------|-------|
| `cuda_mhd_dedner_match_cpu` | ✅ OK | psi_div dentro del 5% — kernel CUDA usa corrección B escalar (aprox.) |
| `cuda_mhd_anisotropic_conduction_match_cpu` | ✅ OK | L2 rel < 5% (f32 vs f64 en suma O(N²)) |
| `cuda_rt_igm_temp_percentiles_match_cpu` | ✅ OK | t_median, t_p16, t_p84 dentro del 5% |

### Tests previos (AP-03 a AP-16) — sin regresiones

Todos los tests anteriores continúan pasando.

---

## Notas de diseño

### Dedner híbrido CPU+GPU
El kernel CUDA `mhd_dedner_cleaning_kernel` sólo implementa el paso de
actualización de ψ/B (O(N) por partícula) con una corrección escalar media
de B (`B -= dt·ψ/3` por componente). La parte cara (divergencia SPH O(N²))
se calcula en CPU via `compute_dedner_div_b`. Esto limita la aceleración real
a sólo el paso de update, pero completa el circuito GPU para futura extensión.

### Kernel anisótropo O(N²)
`mhd_anisotropic_pair_kernel` comparte el mismo cuerpo para conducción
térmica y CR diffusion mediante un parámetro `gamma_m1`:
- `gamma_m1 > 0`: campo escalar es `u`; delta se aplica como `delta / (γ-1)`
- `gamma_m1 == 0`: campo escalar es `cr_energy`; delta se aplica directamente

La banda del kernel Wendland-C6 usa `h_eff = h_i + h_j` (= 2·h_avg), igual
que el path CPU en `anisotropic.rs`.

### IGM percentiles
Se usa un segundo kernel (`rt_igm_compact_kernel`) en paralelo con el kernel
de reducción para escribir las temperaturas filtradas en un array compacto
con índice atómico. El sort y cálculo de percentiles se hace en CPU host
con `sort_unstable_by` sobre el array descargado.

---

## Gaps resueltos

Tras AP-17 no quedan huecos CUDA conocidos. La tabla de parity pending queda
vacía.
