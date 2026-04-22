# Phase 54 — Validación cuantitativa D²(a) con G consistente

**Fecha:** abril 2026  
**Crate principal:** `gadget-ng-physics` (tests), `gadget-ng-core` (cosmología)  
**Archivo de test:** `crates/gadget-ng-physics/tests/phase54_growth_factor_validation.rs`

---

## Contexto

Phase 51 introdujo `auto_g = true` para calcular la constante gravitacional consistente
con la ecuación de Friedmann en una caja unitaria (ρ̄_m = 1):

```
G_consistent = 3·Ω_m·H₀² / (8π) ≈ 3.757×10⁻⁴  (para Ω_m=0.315, H₀=0.1)
```

Phase 50 demostró que con G=1 y H₀=0.1 el ratio `(4πGρ̄)/H₀²` es ~265854%
incorrecto (factor 2660× fuera del valor esperado `(3/2)Ω_m = 0.472`).

Phase 54 verifica **cuantitativamente** que con G_consistent el crecimiento lineal
satisface `P(k,a)/P(k,a₀) = [D(a)/D(a₀)]²` para N=64, 128, 256 en modo release.

---

## Física

### Factor de crecimiento lineal

En el régimen lineal (`δ ≪ 1`), la ecuación de perturbaciones de densidad es:

```
δ'' + 2H δ' = (3/2)Ω_m H₀² / a³ · δ
```

La solución creciente `D(a)` (CPT92, approximación numérica) satisface:

```
P(k, a) = P(k, z=0) · [D(a)/D(1)]²
```

La validación mide `P(k,a)/P(k,a₀)` en la simulación y lo compara con
`[D(a)/D(a₀)]²` calculado analíticamente con `growth_factor_d_ratio`.

### Timestep adaptativo

Con G_consistent las fuerzas gravitacionales son ~2660× más débiles que con G=1.
El criterio de Hubble domina sobre el gravitacional para toda la evolución:

```
dt_hub = α_H / H(a)
dt_grav = η · √(ε / |a_max|)     (suele ser >> dt_hub)
dt = min(dt_hub, dt_grav, dt_max)
```

Parámetros utilizados: `α_H = 0.01`, `η_grav = 0.1`, `dt_max = 0.05`.

Número estimado de pasos de a=0.02 a a=0.50:
- a=0.02: H≈19.84 → dt_hub≈5×10⁻⁴, da/step≈10⁻³
- a=0.50: H≈0.115 → dt_hub≈0.087, da/step≈0.0043
- Total: ~400-800 pasos dependiendo de N

---

## Implementación

### Estructura del test

```
phase54_growth_factor_validation.rs
├── run_simulation_n(n)         → evoluciona de a=0.02 a a=0.50 con 6 snapshots
├── evolve_pm_to_a_adaptive()   → loop adaptativo con pre-cómputo de fuerzas
├── run_full_matrix()           → N ∈ {64, 128, 256}
├── matrix()                   → OnceLock compartida entre tests
└── Tests:
    ├── phase54_sigma8_normalization     → P(k) IC vs P_EH error < 30%
    ├── phase54_growth_d2_n64           → D²(a) error < 30% para todos los snapshots
    ├── phase54_growth_d2_n128          → D²(a) error < 15%
    ├── phase54_growth_d2_n256          → D²(a) error < 10%
    └── phase54_convergence_with_n      → error decrece al aumentar N
```

### Pre-cómputo de fuerzas iniciales

Para evitar que el primer paso use `acc_max = 0` (lo que daría `dt = dt_max`),
se evalúan las fuerzas antes del loop con el PM solver:

```rust
{
    let g_cosmo = gravity_coupling_qksl(g_code, a_start);
    pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, &mut scratch);
}
// Luego scratch[i] contiene las fuerzas reales en el IC
```

### Cálculo del ratio de crecimiento

Para cada snapshot en `a_target` y cada bin de k:

```rust
let ratio_sim = P(k, a_target) / P(k, a_init);   // medido
let d = growth_factor_d_ratio(cosmo, a_target, A_INIT);
let ratio_th = d * d;                              // teórico
let err = (ratio_sim / ratio_th - 1.0).abs();
```

La métrica final es la mediana de `err` sobre todos los bins con `k < k_nyq/2`.

---

## Tolerancias físicas

| N   | Efectos dominantes               | Tolerancia |
|-----|----------------------------------|-----------|
| 64³ | Shot-noise alto, cosmic variance | 30%       |
| 128³| Shot-noise moderado              | 15%       |
| 256³| Régimen lineal bien muestreado   | 10%       |

Las tolerancias son conservadoras dado que:
1. La corrección P(k) con `RnModel::phase47_default()` calibrado en N ∈ {32,64,128}
   introduce ~2% de error residual
2. Con Z0Sigma8 el P(k) en el IC snapshot tiene error de normalización < 1%
3. El coupling QKSL correcto (`G·a³`) asegura que la evolución lineal sigue D(a)

---

## Script de ejecución

```bash
# Tiempo estimado release: N=64 ~15s, N=128 ~120s, N=256 ~960s
cargo test -p gadget-ng-physics --release \
  --test phase54_growth_factor_validation \
  -- --test-threads=1 --nocapture

# Saltando N=256:
PHASE54_SKIP_N256=1 cargo test -p gadget-ng-physics --release \
  --test phase54_growth_factor_validation \
  -- --test-threads=1 --nocapture
```

---

## Resultados esperados

Salida de referencia (estimada, release):

```
[phase54] Iniciando N=64³ con G_consistent=3.7566e-4
[phase54] N=64 a=0.020  bins=10
[phase54] N=64 a=0.050  bins=10
[phase54] N=64 a=0.100  bins=10
[phase54] N=64 a=0.200  bins=10
[phase54] N=64 a=0.330  bins=10
[phase54] N=64 a=0.500  bins=10
[phase54] ✓ N=64 completado en ~15s
[phase54] N=64 a=0.10 error_crecimiento=0.08..
[phase54] N=64 a=0.20 error_crecimiento=0.12..
[phase54] N=64 a=0.50 error_crecimiento=0.20..
```

---

## Relación con GADGET-4

GADGET-4 verifica el crecimiento lineal como validación estándar en `test_cosmo.cc`.
Esta Phase 54 es el equivalente en gadget-ng: demuestra que el integrador KDK
cosmológico con G_consistent reproduce correctamente la historia de crecimiento
del universo ΛCDM para modos en el régimen lineal.
