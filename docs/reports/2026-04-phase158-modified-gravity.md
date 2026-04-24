# Phase 158 — Gravedad modificada f(R) con screening chameleon

**Fecha:** 2026-04-23  
**Crate afectado:** `gadget-ng-core`  
**Archivo nuevo:** `crates/gadget-ng-core/src/modified_gravity.rs`  
**Engine:** hook `maybe_fr!` en `engine.rs`

## Resumen

Implementación del modelo Hu-Sawicki f(R) con mecanismo de screening chameleon. La quinta fuerza amplifica la gravedad en regiones de baja densidad (no screened) mientras queda suprimida en regiones densas.

## Física implementada

### Campo chameleon

```text
f_R_local ≈ f_R0 × (1 + δ)^{-(n+1)}
```

### Factor de quinta fuerza

```text
fifth_force_factor = min(1, |f_R_local / f_R0|)
Δg/g = (1/3) × fifth_force_factor
```

### Aplicación

```text
a_modified = a_GR × (1 + fifth_force_factor / 3)
```

## API pública

| Función/Struct | Descripción |
|----------------|-------------|
| `FRParams { f_r0, n }` | Parámetros del modelo |
| `chameleon_field(delta_rho, f_r0, n)` | Campo escalar local |
| `fifth_force_factor(f_r_local, f_r0)` | Amplificación |
| `apply_modified_gravity(particles, params, cosmo, a)` | Aplica modificación |

## Configuración TOML

```toml
[modified_gravity]
enabled = true
model   = "hu_sawicki"
f_r0    = 1.0e-4
n       = 1
```

## Tests (6/6 OK)

1–6: f_r0=0 es GR, factor en [0,1], screening en densidad alta, a aumenta en vacío, chameleon decrece, N=100 sin panics.

## Referencias

- Hu & Sawicki (2007) PRD 76, 064004
- Khoury & Weltman (2004) PRD 69, 044026
