# Phase 154 — Mock catalogues con efectos de selección

**Fecha:** 2026-04-23  
**Crate afectado:** `gadget-ng-analysis`  
**Archivo nuevo:** `crates/gadget-ng-analysis/src/mock_catalog.rs`

## Resumen

Implementación de mock catalogues galácticos sintéticos con: asignación SMHM (Behroozi+2013 simplificado), magnitud aparente con corrección-k lineal, corte en magnitud límite y espectro de potencia angular C_l simplificado.

## Física implementada

### SMHM (Stellar-to-Halo Mass)

Relación de Behroozi et al. (2013) simplificada a z=0:
```text
log10(M_*/M_h) = log10(ε) + f(log10(M_h/M1)) - f(0)
```

### Magnitud aparente

```text
m = M + 5 × log10(d_L / 10 pc) + K(z)
```
con k-correction lineal K(z) ≈ 1.8×z.

### C_l angular

Proyección Fourier plana sobre malla 2D (plano XY) normalizada.

## API pública

| Función | Descripción |
|---------|-------------|
| `apparent_magnitude(m_abs, z, omega_m)` | Magnitud aparente |
| `selection_flux_limit(m_app, m_lim)` | Corte por flujo |
| `build_mock_catalog(particles, halos, z, omega_m, m_lim)` | Catálogo completo |
| `angular_power_spectrum_cl(catalog, l_max, box_size)` | C_l angular |

## Tests (6/6 OK)

1–6: catálogo no vacío, m crece con z, selección reduce N, SMHM correcto, C_l ≥ 0, N=500 sin panics.

## Referencias

- Behroozi, Wechsler & Conroy (2013) ApJ 770, 57
- Pen (1999) ApJS 120, 49
