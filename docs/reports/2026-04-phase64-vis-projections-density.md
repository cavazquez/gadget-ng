# Phase 64 — gadget-ng-vis: proyecciones adicionales, mapa de densidad y PNG

**Fecha:** abril 2026  
**Crates:** `gadget-ng-vis`, `gadget-ng-cli`  
**Archivos modificados/nuevos:**  
- `crates/gadget-ng-vis/src/ppm.rs` (3 funciones nuevas)  
- `crates/gadget-ng-vis/src/lib.rs` (re-exports)  
- `crates/gadget-ng-cli/src/main.rs` (flags `--vis-proj`, `--vis-mode`, `--vis-format`)  
- `crates/gadget-ng-vis/tests/ppm_extended.rs` (nuevo)

---

## Contexto

`render_ppm` solo proyectaba en XY con puntos blancos (Phase ~60). Phase 64 añade:

1. **Proyecciones configurables**: XY, XZ, YZ usando el enum `Projection` existente.
2. **Mapa de densidad**: escala logarítmica + colormap Viridis.
3. **Exportación PNG nativa** usando la crate `png` (ya dependencia).
4. **Flags CLI** para controlar todo desde la línea de comandos.

---

## Nuevas funciones en `ppm.rs`

### `render_ppm_projection`

```rust
pub fn render_ppm_projection(
    positions: &[Vec3],
    box_size: f64,
    width: usize,
    height: usize,
    proj: Projection,   // XY | XZ | YZ
) -> Vec<u8>  // RGB plano, 3 × width × height bytes
```

Igual que `render_ppm` pero acepta cualquier plano de proyección ortográfica.
Reutiliza `Projection::project(&pos)` del módulo `projection.rs`.

### `render_density_ppm`

```rust
pub fn render_density_ppm(
    positions: &[Vec3],
    box_size: f64,
    width: usize,
    height: usize,
    proj: Projection,
) -> Vec<u8>
```

Algoritmo:
```
1. Acumular counts[pixel] += 1 para cada partícula proyectada
2. max_log = max(log10(1 + count[i]))
3. Para cada pixel: t = log10(1 + count) / max_log  ∈ [0, 1]
4. Color = viridis(t)   (azul=vacío → amarillo=denso)
```

### `write_png`

```rust
pub fn write_png(path: &Path, pixels: &[u8], width: usize, height: usize) -> io::Result<()>
```

Usa `png::Encoder` para escribir formato PNG RGB-8. Los 8 bytes mágicos del PNG
(`\x89PNG\r\n\x1a\n`) quedan verificados en el test `write_png_header`.

---

## Flags CLI nuevos en `Commands::Stepping`

| Flag | Valores | Default |
|------|---------|---------|
| `--vis-proj` | `xy`, `xz`, `yz` | `xy` |
| `--vis-mode` | `points`, `density` | `points` |
| `--vis-format` | `ppm`, `png` | `ppm` |

Ejemplo:
```bash
gadget-ng stepping --config cosmo.toml --out runs/cosmo \
  --snapshot --vis-snapshot 10 \
  --vis-proj xz --vis-mode density --vis-format png
```

Genera `runs/cosmo/snapshot_final.png` con mapa de densidad en proyección XZ.

---

## Tests en `tests/ppm_extended.rs`

| Test | Descripción |
|------|-------------|
| `density_map_concentrated_bright` | Cluster de 100 partículas → pixel con viridis(1)=[255,255,0] más brillante que vacío [0,0,255] |
| `density_map_empty_is_dark` | Imagen vacía → todos los pixels viridis(0)=[0,0,255], uniformes |
| `projection_xz_correct` | Partícula en (25, 0, 75) → pixel `(25, height-1-75)` en proyección XZ |
| `projection_yz_correct` | Partícula en (0, 30, 60) → pixel `(30, height-1-60)` en proyección YZ |
| `write_png_header` | Archivo PNG empieza con `[0x89, 0x50, 0x4e, 0x47]` = `\x89PNG` |
| `write_png_minimal` | PNG 1×1 pixel negro es un archivo válido con longitud > 8 bytes |

---

## Colormap Viridis en `ppm.rs`

La función `viridis(t)` en `color.rs` usa tres segmentos lineales:

| Rango | Color |
|-------|-------|
| t=0.0 | `[0, 0, 255]` — azul puro |
| t=0.5 | `[0, 255, 0]` — verde puro |
| t=1.0 | `[255, 255, 0]` — amarillo |

Para un campo uniforme (sin clustering) todos los pixels del mapa de densidad
son aproximadamente del mismo color. La diferencia se hace visible cuando hay
sobredensidades de ≥10× la media.
