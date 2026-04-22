# Phase 51 — G Auto-Consistente en el Motor de Producción

**Fecha**: Abril 2026  
**Estado**: Completado ✓  
**Crates modificados**: `gadget-ng-core` (config.rs), `gadget-ng-cli` (engine.rs), `gadget-ng-physics` (tests)

---

## 1. Contexto

Phase 50 introdujo `g_code_consistent()` como función standalone que calcula
el G correcto para que la ecuación de Friedmann se satisfaga en unidades de
código con ρ̄_m = 1. Sin embargo, esa función solo era accesible desde código
Rust; ninguna simulación lanzada desde la CLI la usaba automáticamente.

Phase 51 cierra esa brecha: lleva la corrección al motor de producción.

---

## 2. Cambios en `config.rs`

### 2.1 Campo `CosmologySection::auto_g`

```rust
pub struct CosmologySection {
    pub enabled: bool,
    pub periodic: bool,
    pub omega_m: f64,
    pub omega_lambda: f64,
    pub h0: f64,
    pub a_init: f64,
    pub auto_g: bool,   // ← NUEVO
}
```

Default: `false` (retrocompatible — `simulation.gravitational_constant` sin cambios).

### 2.2 `RunConfig::effective_g()` extendido

Jerarquía de prioridad actualizada:

```rust
pub fn effective_g(&self) -> f64 {
    if self.units.enabled {
        self.units.compute_g()                           // prioridad 1
    } else if self.cosmology.enabled && self.cosmology.auto_g {
        g_code_consistent(self.cosmology.omega_m, self.cosmology.h0)  // prioridad 2
    } else {
        self.simulation.gravitational_constant           // fallback
    }
}
```

### 2.3 `RunConfig::cosmo_g_diagnostic()`

Nueva función que devuelve `Option<(g_consistent, rel_err)>`:

```rust
pub fn cosmo_g_diagnostic(&self) -> Option<(f64, f64)> {
    if !self.cosmology.enabled { return None; }
    let g_consistent = g_code_consistent(self.cosmology.omega_m, self.cosmology.h0);
    let g_used = self.effective_g();
    let rel_err = (g_used - g_consistent).abs() / g_consistent;
    Some((g_consistent, rel_err))
}
```

---

## 3. Cambios en `engine.rs`

Inmediatamente después de calcular `let g = cfg.effective_g()`:

```rust
if let Some((g_consistent, rel_err)) = cfg.cosmo_g_diagnostic() {
    if cfg.cosmology.auto_g {
        rt.root_eprintln(&format!(
            "[gadget-ng] cosmology.auto_g=true → G auto-consistente: {g:.4e} \
             (3·Ω_m·H₀²/8π, condición de Friedmann satisfecha)"
        ));
    } else if rel_err > 0.01 {
        rt.root_eprintln(&format!(
            "[gadget-ng] ADVERTENCIA: G ({g:.4e}) inconsistente con cosmología \
             ({:.1}% fuera de G_consistente={g_consistent:.4e}). \
             Usa [cosmology] auto_g = true para corregir automáticamente.",
            rel_err * 100.0
        ));
    }
}
```

**Casos cubiertos:**

| Situación | Acción del motor |
|-----------|-----------------|
| `auto_g = true` | `info!` con G calculado |
| `auto_g = false`, G manual > 1% fuera | `warn!` con porcentaje de error |
| `auto_g = false`, G manual ≤ 1% fuera | Silencio (ya es consistente) |
| `cosmology.enabled = false` | Sin diagnóstico |

---

## 4. Tests (phase51_auto_g.rs)

### Test 1 — `phase51_auto_g_effective_g`

```
G_effective = 3.760036e-4  g_code_consistent = 3.760036e-4  err = 0.00e0
```

Con `auto_g=true` y `gravitational_constant=1.0`, `effective_g()` devuelve
el valor Friedmann-consistente, no el manual. Error < 10⁻¹².

### Test 2 — `phase51_legacy_g_diagnostic`

```
G_used = 1.0  G_consistent = 3.76e-4  err = 265854.9%
```

G legacy está 265855% fuera del valor correcto. El motor emitiría `ADVERTENCIA`.

### Test 3 — `phase51_consistent_g_no_warning`

```
G_used = 3.760036e-4  G_consistent = 3.760036e-4  err = 0.00e0
```

Con G manual puesto manualmente al valor correcto, el error es 0. Sin warning.

### Test 4 — `phase51_units_priority`

```
G_units = 4.3009e4  G_effective = 4.3009e4  G_auto_g = 3.7600e-4
```

`units.enabled = true` toma prioridad sobre `auto_g = true`. El G de
UnitsSection (GADGET clásico: 4.3009e4) prevalece sobre el Friedmann (3.76e-4).

### Test 5 — `phase51_auto_g_simulation_stable`

```
G_auto = 3.7600e-4  a_final = 0.0402  v_rms = 1.6768e-6
✓ auto_g=true: G=3.7600e-4 (Friedmann), simulación estable.
```

Simulación N=8, a=0.02→0.04 con `auto_g=true`. Estable, v_rms finito.

---

## 5. Retrocompatibilidad

- `auto_g = false` (default TOML) → comportamiento idéntico a antes de Phase 51.
- Configs existentes sin el campo `auto_g` deserializan correctamente (`#[serde(default)]`).
- El `warn!` por G inconsistente solo se emite en stderr, no afecta la ejecución.
- Todos los tests históricos actualizados con `auto_g: false` explícito en inicializadores literales.

---

## 6. Uso recomendado

Para simulaciones cosmológicas nuevas:

```toml
[cosmology]
enabled      = true
auto_g       = true   # G = 3·Ω_m·H₀²/(8π), condición de Friedmann
omega_m      = 0.315
omega_lambda = 0.685
h0           = 0.1    # H₀ en unidades internas
a_init       = 0.02   # z=49
```

Para configuraciones legacy (sin cambios):

```toml
[cosmology]
enabled = true
# auto_g no especificado → default false → usa simulation.gravitational_constant
# Se emitirá ADVERTENCIA si G ≠ G_consistente por más del 1%

[simulation]
gravitational_constant = 1.0  # ← inconsistente; recibirá warning en stderr
```

---

## 7. Resumen

| Componente | Cambio |
|-----------|--------|
| `CosmologySection::auto_g` | Nuevo campo bool (default false) |
| `RunConfig::effective_g()` | Prioridad 2: auto_g calcula G Friedmann-consistente |
| `RunConfig::cosmo_g_diagnostic()` | Nueva función de diagnóstico |
| `engine.rs` | warn si G>1% inconsistente; info si auto_g activo |
| `phase51_auto_g.rs` | 5 tests cubriendo toda la lógica |
| Tests históricos | `auto_g: false` añadido en inicializadores literales |
