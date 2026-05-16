# AP-19: Pipeline SPH CUDA Persistente

**Fecha:** 2026-05-16  
**Hardware:** NVIDIA GTX 1060 — sm_61, 6 GB GDDR5, PCIe 3.0 x16 (~16 GB/s teórico, ~12 GB/s medido)  
**Crate afectado:** `gadget-ng-cuda`

---

## Motivación

El método `try_sph_density_and_forces_core` (introducido en AP-18) ejecutaba tres kernels CUDA con **tres rondas PCIe independientes**. Tras cada kernel, el pool se reseteaba y todos los buffers de entrada se volvían a subir al device aunque los datos del host no hubieran cambiado:

| Ronda | Datos subidos H→D (por partícula, f32) | Datos bajados D→H |
|-------|----------------------------------------|-------------------|
| Density | x,y,z,mass,is_gas,u,h → **7 campos** + 4 alloc | h_out,rho,pressure,entropy = 4 campos |
| Balsara | x,y,z,vx,vy,vz,mass,is_gas,rho,pressure,h_out → **11 campos** + 1 alloc | balsara = 1 campo |
| Forces  | x,y,z,vx,vy,vz,mass,is_gas,rho,pressure,h_out → **11 campos** + 4 alloc | ax,ay,az,du_dt = 4 campos |

**Total transferido H→D:** ~(7+11+11) × N × 4 B = **116 bytes/partícula** subidos.  
**Total transferido D→H:** ~(4+1+4) × N × 4 B = **36 bytes/partícula** bajados.

Adicionalmente, el kernel de forces usaba `cuda_sph_forces` (clásico), que **ignora el factor Balsara** computado en el paso anterior. El limitador de viscosidad no se aplicaba, lo cual es un bug físico.

---

## Cambios implementados (AP-19)

### Pipeline persistente (slot map 0–20)

Un único `pool.reset()` al inicio. Los 21 slots se asignan de forma permanente:

| Slot | Campo | Rol |
|------|-------|-----|
| 0–2 | x, y, z | upload una vez → density + balsara + forces |
| 3–5 | vx, vy, vz | upload una vez → balsara + forces |
| 6 | mass | upload una vez → los tres kernels |
| 7 | is_gas (u8) | upload una vez → los tres kernels |
| 8 | u_arr | upload una vez → density |
| 9 | h_in | upload una vez → density |
| 10 | h_out | alloc; density escribe, balsara+forces leen vía device ptr |
| 11 | rho | alloc; density escribe, balsara+forces leen |
| 12 | pressure | alloc; density escribe, balsara+forces leen |
| 13 | entropy | alloc; density escribe (descartado) |
| 14 | balsara | alloc; balsara escribe, gadget2_forces lee |
| 15–17 | ax, ay, az | alloc; gadget2_forces escribe (salida) |
| 18 | da_dt | alloc; gadget2_forces escribe (descartado) |
| 19 | du_dt | alloc; gadget2_forces escribe (salida) |
| 20 | max_vsig | alloc; gadget2_forces escribe (descartado) |

**Total subido H→D ahora:** ~(10 × f32 + 1 × u8) × N ≈ **41 bytes/partícula** (una sola vez).  
**Total bajado D→H ahora:** (h_out + rho + ax + ay + az + du_dt) × 4 B = **24 bytes/partícula**.

### Fix físico: `cuda_sph_forces` → `cuda_sph_gadget2_forces`

El tercer kernel ahora usa `cuda_sph_gadget2_forces`, que acepta `d_balsara` como parámetro y aplica correctamente el limitador de viscosidad de Balsara (1990) en las fuerzas SPH. La física es equivalente al path CPU `sph_forces_gadget2`.

### Write-back de `h_sml` diferido

Antes, `smoothing_length` se escribía de vuelta al host entre density y balsara. Con el pipeline persistente se difiere al final, tras descargar `h_out` del device, sin impacto en la corrección (ningún kernel posterior en este método depende del valor host).

---

## Análisis de reducción de transferencias

### Datos PCIe comparados

| Escenario | H→D (bytes/part) | D→H (bytes/part) | Total (bytes/part) |
|-----------|-------------------|-------------------|--------------------|
| AP-18 (3 resets) | 116 | 36 | 152 |
| AP-19 (1 reset)  |  41 | 24 |  65 |
| **Reducción**    | **−65%** | **−33%** | **−57%** |

### Break-even estimado

Con PCIe 3.0 x16 (12 GB/s efectivo):

- AP-18 latencia PCIe ≈ 3 × (launch overhead ~5 µs + transfer ~N×152/12e9 s)
- AP-19 latencia PCIe ≈ 1 × (launch overhead ~5 µs + transfer ~N×65/12e9 s)

A N=64: PCIe contribuye ~1 µs; overhead de kernel (~3 µs cada uno) domina → break-even no cambia  
A N=128: AP-19 ahorra ~1.2 µs de PCIe → break-even baja de N≈300 a N≈130–150  
A N=256: AP-19 ahorra ~2.5 µs → GPU ya compite claramente con CPU O(N²)

**Break-even estimado:** N ≈ 120–150 (frente a N ≈ 300 de AP-18).

El benchmark `bench_sph_core_cuda_vs_cpu` en `crates/gadget-ng-cuda/benches/cuda_vs_simd.rs` 
está preparado para medir este punto en hardware real.

---

## Verificación

```bash
# Clippy sin warnings
cargo clippy -p gadget-ng-cuda -- -D warnings

# Test parity (requiere CUDA)
cargo test -p gadget-ng-cuda -- --ignored cuda_parity_sph_core_pipeline

# Benchmark (requiere CUDA)
cargo bench -p gadget-ng-cuda --bench cuda_vs_simd -- bench_sph_core_cuda_vs_cpu
```

---

## Resumen de cambios

- **`crates/gadget-ng-cuda/src/sph_solver.rs`**: `try_sph_density_and_forces_core` reescrito con pipeline persistente (21 slots), `cuda_sph_forces` reemplazado por `cuda_sph_gadget2_forces`.
- **Reducción PCIe:** −65% en uploads (116→41 bytes/partícula).
- **Fix físico:** limitador Balsara ahora alimenta correctamente las fuerzas SPH.
- **Break-even teórico:** N≈120–150 (frente a N≈300 anterior).
