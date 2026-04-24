# Phase 156 — Neutrinos masivos Ω_ν

**Fecha:** 2026-04-23  
**Crate afectado:** `gadget-ng-core`  
**Archivos modificados:** `cosmology.rs`, `config.rs`, `ic_zeldovich.rs`

## Resumen

Implementación del efecto de neutrinos masivos en la cosmología: contribución a H(a) mediante Ω_ν y supresión del espectro de potencia P(k) en las condiciones iniciales Zel'dovich.

## Física implementada

### Fracción de neutrinos

```text
Ω_ν = Σm_ν / (93.14 eV × h²)
```

### Supresión del espectro de potencia (Lesgourgues & Pastor 2006)

```text
ΔP/P ≈ -8 × f_ν   (f_ν = Ω_ν/Ω_m)
```

Aplicada en las ICs Zel'dovich como factor `√(1-8f_ν)` sobre la amplitud del campo δ(k).

## Cambios en config.rs

```toml
[cosmology]
m_nu_ev = 0.06   # suma de masas en eV; 0.0 = sin neutrinos
```

## API pública nueva

| Función | Descripción |
|---------|-------------|
| `omega_nu_from_mass(m_nu_ev, h100)` | Ω_ν desde masa |
| `neutrino_suppression(f_nu)` | Factor (1-8f_ν) |
| `CosmologyParams::new_with_nu(...)` | Constructor con Ω_ν |

## Tests (6/6 OK)

1–6: m_nu=0→Ω_ν=0, m_nu=0.06 suprime, formula correcta, advance_a estable, supresión lineal, clamp a 0.

## Referencias

- Lesgourgues & Pastor (2006) Phys. Rept. 429, 307
- Lesgourgues et al. (2013) "Neutrino Cosmology" (Cambridge)
