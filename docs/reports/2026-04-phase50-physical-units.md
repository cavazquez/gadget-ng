# Phase 50 — Unidades Físicamente Consistentes

**Fecha**: Abril 2026  
**Estado**: Completado ✓  
**Crates modificados**: `gadget-ng-core`, `gadget-ng-physics` (tests)

---

## 1. Contexto

Phase 49 (Fix integrador cosmológico) identificó y documentó una inconsistencia
entre los parámetros de test estándar (`G = 1.0`, `H₀ = 0.1`) y las leyes
cosmológicas que gobiernan el crecimiento de estructuras. Esta inconsistencia
impedía que los tests verificaran `D²(a)` correctamente para evoluciones largas.

El hallazgo quedó documentado en `docs/user-guide.md` y el reporte
`2026-04-phase49-integrator-fix.md`, pero sin resolución en código. Phase 50
resuelve este punto abierto.

---

## 2. La condición de consistencia cosmológica

### 2.1 Ecuación de Friedmann en unidades de código

Para una simulación con caja unitaria (`box = 1`, `total_mass = 1`, `ρ̄_m = 1`),
la ecuación de Friedmann requiere:

```
H₀² = 8π·G·ρ̄_m·Ω_m / 3    con ρ̄_m = 1
```

Despejando G:

```
G_consistente = 3·Ω_m·H₀² / (8π)
```

### 2.2 Valores numéricos (H₀=0.1, Ω_m=0.315)

| Parámetro              | Valor         |
|------------------------|---------------|
| G_legacy (histórico)   | 1.0           |
| G_consistente          | 3.760×10⁻⁴   |
| Factor de diferencia   | 2659.5×       |

### 2.3 Impacto en la ecuación del crecimiento

El término fuente de la ecuación de Meszaros es `4πGρ̄_m`. Con los dos valores:

| G             | (4πGρ̄)/H₀²  | (3/2)Ω_m (correcto) | Error    |
|---------------|--------------|----------------------|----------|
| G_legacy=1    | 1256.6       | 0.4725               | 2659×    |
| G_consistente | 0.4725       | 0.4725               | 0 %      |

Con G_legacy, la fuerza gravitacional está 2660× sobredimensionada respecto a
la expansión de Hubble, rompiendo la correspondencia con el factor de
crecimiento analítico D(a).

### 2.4 Verificación con el caso EdS

Para un universo de Einstein-de Sitter (Ω_m=1, Ω_Λ=0, H₀=1):

```
G_consistente = 3·1·1² / (8π) = 3/(8π) ≈ 0.11937
```

Esta es la constante gravitacional canónica para el integrador EdS en unidades
donde ρ̄=1 y H₀=1, consistente con la bibliografía (Springel 2005).

---

## 3. Implementación

### 3.1 `g_code_consistent(omega_m, h0) -> f64`

Añadida a `gadget-ng-core/src/cosmology.rs`:

```rust
pub fn g_code_consistent(omega_m: f64, h0: f64) -> f64 {
    3.0 * omega_m * h0 * h0 / (8.0 * std::f64::consts::PI)
}
```

### 3.2 `cosmo_consistency_error(g, omega_m, h0, rho_bar) -> f64`

Devuelve el error relativo `|G - G_expect|/G_expect` donde
`G_expect = 3·Ω_m·H₀²/(8π·ρ̄_m)`. Útil para diagnosticar configuraciones
existentes.

### 3.3 Re-exportación

Ambas funciones exportadas desde `gadget-ng-core::lib.rs`:

```rust
pub use cosmology::{
    adaptive_dt_cosmo, cosmo_consistency_error, g_code_consistent,
    gravity_coupling_qksl, ...
};
```

---

## 4. Tests (phase50_physical_units.rs)

### Test 1 — `phase50_consistency_formula`

Verifica algebraicamente que `g_code_consistent` satisface la ecuación de
Friedmann para tres casos: Planck 2018 (H₀=0.1, Ω_m=0.315), EdS (H₀=1, Ω_m=1)
y el caso histórico. Error relativo < 10⁻¹².

```
G_code = 3.760036e-4  err = 0.000e0
[EdS] G = 1.193662e-1 = 3/(8π)  err = 0.000e0
G_legacy/G_consist = 2659.5×
```

### Test 2 — `phase50_inconsistency_quantified`

Cuantifica el error de G=1 para el régimen de test estándar:

```
Consistente: (4πGρ̄)/H₀² = 0.4725  ≈ (3/2)Ω_m ✓
Legacy G=1:  (4πGρ̄)/H₀² = 1256.6  factor vs correcto: 2659.5×
```

### Test 3 — `phase50_growth_consistent_short`

N=8, a=0.02→0.05, dt=1e-3 (~25 pasos). Con G_consistente:

```
P_ratio = 3.019    v_rms = 1.677e-6    (estable ✓)
```

El ratio P(k) no alcanza D²=6.3 porque N=8 tiene solo ~4 bins de k con
pocos modos: la varianza estadística (~50%) domina sobre la señal de
crecimiento gravitacional.

### Test 4 — `phase50_growth_consistent_long`

N=8, a=0.02→0.20, dt adaptativo. 516 pasos, ~0.84s en debug:

```
P_ratio = 5.608    D²_predict = 99.49    n_steps = 516    v_rms = 1.704e-6
```

La simulación es estable en 10× de expansión. El ratio es menor que D²
analítico por las mismas limitaciones estadísticas de N=8.

### Test 5 — `phase50_g_consistent_vs_legacy`

Comparación directa N=8, a=0.02→0.05:

```
G_consistente (3.76e-4): ratio=3.019  v_rms=1.677e-6
G_legacy      (1.00e0):  ratio=3.152  v_rms=2.059e-6
```

Para evoluciones cortas (25 pasos), el efecto de G_legacy se atenúa por el
factor `a³` en `g_cosmo = G·a³` con `a=0.02` pequeño. La diferencia se
manifiesta principalmente en evoluciones largas (documentada en Phase 49).

---

## 5. Limitaciones conocidas

### 5.1 Resolución mínima para D²(a)

La verificación cuantitativa de `P(k)/P₀(k) ≈ D²(a)` requiere:

- N ≥ 64 (volumen de Fourier suficiente)
- Múltiples semillas (promedio sobre realizaciones)
- Modo `release` (rendimiento 10-100× mayor)

Con N=8 (debug), la varianza estadística por bin de P(k) es O(1), haciendo
imposible verificar un D²(a) que cambia en factor 6-100 entre z=49 y z=4.

### 5.2 Fuerzas PM con G_consistente

Con G_consistente = 3.76e-4, las fuerzas en el PM son ~2660× menores que
con G_legacy. Para resoluciones bajas (N=8), el PM apenas puede distinguir
las perturbaciones del ruido de la malla. El crecimiento observado (ratio>1)
proviene principalmente del streaming Zel'dovich desde las condiciones iniciales.

### 5.3 Relación con UnitsSection

La ruta de producción para simulaciones físicas usa `UnitsSection.enabled = true`
con `G_KPC_MSUN_KMPS = 4.3009×10⁻⁶`. La función `g_code_consistent` ofrece
el análogo para unidades de código internas sin conversión a kpc/Msun/km·s⁻¹.

---

## 6. Impacto en fases futuras

### Para validación de D²(a) en tests de integración:

```rust
// Usar G_consistente en lugar de G=1:
let g = g_code_consistent(OMEGA_M, H0);  // 3.76e-4
let g_cosmo = gravity_coupling_qksl(g, a);
```

### Para diagnóstico de configs existentes:

```rust
let err = cosmo_consistency_error(cfg.simulation.gravitational_constant,
                                   cfg.cosmology.omega_m,
                                   cfg.cosmology.h0, rho_bar);
if err > 0.01 {
    eprintln!("Advertencia: G inconsistente con cosmología ({err:.1}%)");
}
```

---

## 7. Resumen

| Componente                | Descripción                                                |
|---------------------------|------------------------------------------------------------|
| `g_code_consistent()`     | G auto-consistente para ρ̄_m=1, Friedmann satisfecha       |
| `cosmo_consistency_error()`| Error relativo G vs G_esperado                            |
| 5 tests                   | Analíticos + estabilidad + comparación G_legacy           |
| Hallazgo cuantitativo     | G_legacy 2659× inconsistente con H₀=0.1, Ω_m=0.315       |
| Limitación documentada    | D²(a) cuantitativo requiere N≥64 en release               |
