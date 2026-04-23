# Phase 99 — Benchmarks AGN (Criterion)

**Fecha:** 2026-04-23  
**Crate afectado:** `gadget-ng-sph`  
**Tipo:** Benchmarking — medición de rendimiento de feedback AGN

---

## Objetivo

Medir el overhead real de `apply_agn_feedback`, `bondi_accretion_rate` y
`grow_black_holes` con el framework Criterion, para cuantificar el costo
computacional del módulo AGN y determinar si es viable en producción.

## Archivos modificados/creados

### `crates/gadget-ng-sph/Cargo.toml`

```toml
[dev-dependencies]
criterion.workspace = true

[[bench]]
name = "agn_feedback"
harness = false
```

### `crates/gadget-ng-sph/benches/agn_feedback.rs` (nuevo)

Tres grupos de benchmarks con barridos paramétricos:

#### Grupo 1: `apply_agn_feedback` — barrido en N_particles

Mide tiempo total para 1 agujero negro y N = 64, 512, 4096, 32768 partículas.

#### Grupo 2: `apply_agn_feedback` — barrido en N_black_holes

N_particles = 4096 fijo; n_BH = 1, 4, 16.

#### Grupo 3: `bondi_accretion_rate` — barrido en M_BH

Masa del agujero negro M = 10⁶, 10⁷, 10⁸, 10⁹ M_sol/h.

#### Grupo 4: `grow_black_holes` — barrido en N_particles

N = 64, 512, 4096 partículas.

## Resultados

### `apply_agn_feedback` — escalado en N_particles (1 BH)

| N_particles | Tiempo median | Throughput |
|-------------|---------------|-----------|
| 64 | 53 ns | 1.20 Gelem/s |
| 512 | 428 ns | 1.20 Gelem/s |
| 4096 | 3.4 µs | 1.20 Gelem/s |
| 32768 | 30 µs | 1.08 Gelem/s |

**Escalado: perfectamente lineal O(N).** El throughput constante (~1.2 Gelem/s)
indica que el bottleneck es el ancho de banda de memoria, no cómputo.

### `apply_agn_feedback` — escalado en N_black_holes (N=4096)

| N_BH | Tiempo median | Escalado |
|------|---------------|---------|
| 1 | 3.4 µs | 1.0× |
| 4 | 13.6 µs | 4.0× |
| 16 | ~54 µs | ~16× |

**Escalado: lineal O(N × n_BH).** Sin costos fijos por agujero negro.

### `bondi_accretion_rate`

| M_BH | Tiempo median |
|------|---------------|
| 1×10⁶ | ~3.5 ns |
| 1×10⁷ | ~3.5 ns |
| 1×10⁸ | ~3.4 ns |
| 1×10⁹ | ~3.5 ns |

**Completamente independiente de la masa.** Calculo puramente aritmético
(4 multiplicaciones, 1 división), sin acceso a memoria adicional.

### `grow_black_holes`

| N_particles | Tiempo median |
|-------------|---------------|
| 64 | ~50 ns |
| 512 | ~380 ns |
| 4096 | ~3.0 µs |

Overhead equivalente a `apply_agn_feedback`. O(N) con throughput ~1.1 Gelem/s.

## Análisis de viabilidad

Costo de un paso AGN completo en producción:

| Escenario | N_part | n_BH | Tiempo AGN | Tiempo SPH estimado | Ratio |
|-----------|--------|------|-----------|---------------------|-------|
| Test N=8³ | 512 | 1 | ~0.8 µs | ~5 ms | **0.02%** |
| Dev N=32³ | 32768 | 4 | ~120 µs | ~200 ms | **0.06%** |
| Prod N=128³ | 2M | 16 | ~50 ms | ~20 s | **0.25%** |

**Conclusión: el módulo AGN es computacionalmente despreciable** para todas las
escalas de producción previstas. No requiere optimización adicional.

## Cómo ejecutar los benchmarks

```bash
# Benchmark completo con reportes HTML en target/criterion/
cargo bench -p gadget-ng-sph --bench agn_feedback

# Solo una función específica
cargo bench -p gadget-ng-sph --bench agn_feedback -- apply_agn_feedback

# Con tiempo de medición personalizado
cargo bench -p gadget-ng-sph --bench agn_feedback -- --warm-up-time 2 --measurement-time 5
```

Los reportes HTML se generan en `target/criterion/agn_feedback/`.

## Estado

✅ Implementado, compilado y ejecutado. Resultados reproducibles.
Commit `8a9a512` en `main`.
