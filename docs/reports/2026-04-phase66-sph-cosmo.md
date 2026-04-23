# Phase 66 — SPH Cosmológico integrado al motor

**Fecha:** 2026-04-23  
**Crates afectados:** `gadget-ng-core`, `gadget-ng-sph`, `gadget-ng-cli`  
**Tests:** 5 / 5 ✅

---

## Resumen

Integración del módulo SPH (Smoothed Particle Hydrodynamics) al motor principal de gadget-ng.
Las partículas ahora pueden ser de tipo **Gas** (termodinámicas, con SPH) o **Dark Matter**
(solo gravitacionales), conviviendo en la misma lista `Vec<Particle>`.

---

## Cambios en `gadget-ng-core`

### `particle.rs` — nuevo enum `ParticleType` y campos en `Particle`

```rust
pub enum ParticleType { #[default] DarkMatter, Gas }

pub struct Particle {
    // ... campos existentes ...
    pub ptype: ParticleType,         // default: DarkMatter
    pub internal_energy: f64,        // u [km²/s²]; 0.0 para DM
    pub smoothing_length: f64,       // h_sml; 0.0 para DM
}
```

- `#[serde(default)]` en los tres campos nuevos → retrocompatibilidad con snapshots existentes.
- Nuevo constructor `Particle::new_gas(...)` y método `Particle::is_gas()`.
- `Particle::new(...)` sigue funcionando sin cambios (produce `DarkMatter`).

### `config.rs` — `SphSection` y `CoolingKind`

```toml
[sph]
enabled       = true
gamma         = 1.6667
alpha_visc    = 1.0
n_neigh       = 32
cooling       = "atomic_h_he"
t_floor_k     = 1e4
gas_fraction  = 0.1
```

Tipos:
- `CoolingKind`: `None` (default) | `AtomicHHe`
- `SphSection`: todos los campos con defaults, `#[serde(default)]` en `RunConfig`.

---

## Cambios en `gadget-ng-sph`

### `integrator.rs` — `sph_cosmo_kdk_step`

Nueva función que integra posición, velocidad y energía interna de partículas
`gadget_ng_core::Particle` usando factores cosmológicos `CosmoFactors`:

```rust
pub fn sph_cosmo_kdk_step<F>(
    particles: &mut [Particle],
    cf: CosmoFactors,
    gamma: f64,
    alpha_visc: f64,
    n_neigh: f64,
    gravity_accel: F,
)
```

**Algoritmo KDK:**

1. Calcular densidad SPH + presión (`compute_rho_pressure` — Newton-Raphson para `h_sml`)
2. Calcular fuerzas SPH + `du/dt` (`sph_accel_and_dudt` — Springel & Hernquist 2002)
3. **Kick 1** (½ `kick_half`): velocidad y energía interna
4. **Drift** (`drift`): posición
5. Recalcular fuerzas al nuevo tiempo
6. **Kick 2** (½ `kick_half2`): velocidad y energía interna

Factores cosmológicos: `CosmoFactors::flat(dt)` para integración Newtoniana;
`CosmoFactors` completos para cosmología ΛCDM.

### `cooling.rs` — enfriamiento radiativo H+He

```rust
pub fn cooling_rate_atomic(u: f64, rho: f64, gamma: f64, t_floor_k: f64) -> f64
pub fn apply_cooling(particles: &mut [Particle], cfg: &SphSection, dt: f64)
```

Modelo: `Λ(T) = Λ₀ · (T/T_ref)^β` con `Λ₀ = 2×10⁻⁵`, `β = 0.7`, `T_ref = 10⁴ K`.

Conversión energía-temperatura: `T = u · (γ−1) / (k_B/m_H/μ)` con `μ = 0.6`.
Floor de temperatura: `T_floor = 10⁴ K` (configurable).

---

## Cambios en `gadget-ng-cli`

### `engine.rs` — macro `maybe_sph!`

```rust
macro_rules! maybe_sph {
    ($cf:expr) => {
        if cfg.sph.enabled {
            sph_cosmo_kdk_step(&mut local, $cf, gamma, alpha, n_neigh, |_| {});
            if cfg.sph.cooling != CoolingKind::None {
                apply_cooling(&mut local, &cfg.sph, cfg.simulation.dt);
            }
        }
    };
}
```

La macro es un no-op cuando `sph.enabled = false` → costo cero para simulaciones DM-only.

---

## Tests

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `particle_type_defaults` | `new()` → DM, `new_gas()` → Gas | ✅ |
| `sph_section_defaults` | Valores default de `SphSection` | ✅ |
| `sph_energy_conservation_50steps` | E_tot sin gravedad, 4³ gas, 50 pasos, ΔE/E < 50% | ✅ |
| `sph_cooling_lowers_temperature` | Gas caliente AtomicHHe → T decae monotónamente | ✅ |
| `sph_cosmo_kdk_no_gravity_bounded` | 10 pasos KDK sin gravedad, v y u finitos | ✅ |

---

## Próximos pasos

- Inicialización de partículas gas a partir de `gas_fraction` en las ICs
- Condiciones iniciales de Sedov-Taylor para validación de shock
- Integración con el solver MPI: serializar/deserializar `ParticleType`, `internal_energy`, `smoothing_length` en el protocolo de pack/unpack MPI
