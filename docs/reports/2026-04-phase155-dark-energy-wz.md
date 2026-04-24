# Phase 155 — Energía oscura dinámica w(z) CPL

**Fecha:** 2026-04-23  
**Crate afectado:** `gadget-ng-core`  
**Archivos modificados:** `cosmology.rs`, `config.rs`, `lib.rs`

## Resumen

Extensión del integrador cosmológico para soportar el modelo CPL (Chevallier-Polarski-Linder) de energía oscura dinámica. Retrocompatible: w0=-1, wa=0 recupera ΛCDM exactamente.

## Física implementada

### Parámetro de ecuación de estado CPL

```text
w(a) = w0 + wa × (1 − a)
```

### Ecuación de Friedmann generalizada

```text
H²(a) = H₀² [ Ω_m/a³ + Ω_DE(a) ]
Ω_DE(a) = Ω_Λ × a^{-3(1+w0+wa)} × exp(3·wa·(a-1))
```

## Cambios en config.rs

```toml
[cosmology]
w0 = -1.0   # default: ΛCDM
wa = 0.0    # default: ΛCDM
```

## API pública nueva

| Función | Descripción |
|---------|-------------|
| `dark_energy_eos(a, w0, wa)` | w(a) CPL |
| `CosmologyParams::new_cpl(...)` | Constructor con CPL |

## Tests (6/6 OK)

1–6: w(a=1)=-1, advance_a estable 1000 pasos, w varía con a, H en límites, config round-trip.

## Referencias

- Chevallier & Polarski (2001) IJMPD 10, 213
- Linder (2003) PRL 90, 091301
