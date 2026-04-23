# Rápidas — `gadget-ng analyze` + `gadget-ng-vis` PPM

**Fecha:** abril 2026  
**Crates:** `gadget-ng-cli` (nuevo subcomando), `gadget-ng-vis` (nuevo módulo)

---

## R1 — `gadget-ng analyze`

### Objetivo

Subcomando CLI que calcula el pipeline completo de análisis desde un snapshot:
FoF + P(k) + ξ(r) + c(M), escribiendo un único `results.json`.

### Uso

```
gadget-ng analyze \
  --snapshot out/snapshot_final \
  --output analysis/results.json \
  --fof-b 0.2 \
  --pk-mesh 64 \
  --xi-bins 20 \
  --nfw-min-part 50 \
  --box-size-mpc-h 300.0
```

### Pipeline

```
snapshot JSONL
    ↓
FoF (b × l_mean)
    ↓ halos
P(k) CIC+FFT
    ↓ bins P(k)
ξ(r) via Hankel de P(k)    ←  two_point_correlation_fft
    ↓
c(M) NFW para halos ≥ N    ←  fit_nfw_concentration
    ↓
results.json
```

### Estructura del JSON de salida

```json
{
  "n_particles": 32768,
  "box_size_mpc_h": 300.0,
  "halos": [...],
  "power_spectrum": [{"k": ..., "pk": ..., "n_modes": ...}, ...],
  "xi_r": [{"r": ..., "xi": ...}, ...],
  "concentration_mass": [
    {"halo_id": 0, "m200_msun_h": 5e13, "c_measured": 4.2,
     "c_duffy2008": 4.1, "c_ludlow2016": 3.4}, ...
  ]
}
```

### Implementación

- `crates/gadget-ng-cli/src/analyze_cmd.rs`: nueva función `run_analyze`.
- `crates/gadget-ng-cli/src/main.rs`: nuevo comando `Commands::Analyze`.

---

## R2 — `gadget-ng-vis` PPM

### Objetivo

Renderizado de partículas como imagen PPM (Portable Pixel Map) sin dependencias
externas (`png`, `image`, etc.).

### API

```rust
/// Renderiza proyección XY en buffer RGB.
pub fn render_ppm(positions: &[Vec3], box_size: f64, width: usize, height: usize) -> Vec<u8>

/// Escribe buffer RGB en formato P6 (PPM binario).
pub fn write_ppm(path: &Path, pixels: &[u8], width: usize, height: usize) -> io::Result<()>
```

### Uso desde CLI

```
gadget-ng stepping --config sim.toml --out out/ --vis-snapshot 1
# Genera: out/snapshot_final.ppm
```

### Formato PPM (P6)

```
P6
<width> <height>
255
<raw RGB bytes>
```

El formato es legible por cualquier visor de imágenes (GIMP, ImageMagick, etc.)
sin dependencias en el binario de gadget-ng.

### Tests

| Test | Descripción |
|------|-------------|
| `ppm_empty_is_black` | Buffer sin partículas = RGB(0,0,0) |
| `ppm_particle_at_origin_is_white` | Partícula en (0,0) → pixel blanco |
| `ppm_size_correct` | `len(pixels) = width × height × 3` |
| `ppm_write_and_read_back` | Escribe y verifica cabecera `P6\n` |
| `ppm_particle_out_of_bounds_ignored` | Partícula fuera de caja no aparece |
