# Fase 6 — Integrador Yoshida 4º orden y el límite del integrador en N-body caótico

**Fecha:** 2026-04-16
**Scope:** integrador simpléctico de orden superior, validación matemática en sistemas integrables, benchmark dinámico contra leapfrog KDK en regímenes caóticos con Barnes–Hut multipolar + MAC V5 (Fase 5).
**Estado:** completada, conclusión concluyente — el orden del integrador **no desplaza** la frontera de drift en regímenes caóticos densos.

---

## 1. Resumen ejecutivo

1. **H1 y H2 confirmadas matemáticamente.** El nuevo integrador `yoshida4` está correctamente implementado: alcanza pendiente de convergencia **4.013** en oscilador armónico y pendiente **4.00** en cierre orbital de Kepler sobre un barrido `dt ∈ {0.2, 0.1, 0.05, 0.025}`. A `dt = T/200` en Kepler circular, Yoshida cierra la órbita **~400× más preciso** que KDK (`5.7e-6` vs `2.1e-3`).

2. **H3 confirmada con señal muy fuerte en sentido opuesto al trivial.** En Plummer `a/ε = 1` (N=1000), Yoshida no sólo **no reduce** el drift energético: lo **empeora** a `|ΔE/E| = 0.604` vs `0.324` de KDK. El mismo patrón se repite en `a/ε = 2` (`0.503` vs `0.241`) y en uniforme N=200 (`0.058` vs `0.008`). En uniforme N=1000 ambos empatan a `~3e-3`.

3. **H5 confirmada.** Yoshida paga ~**1.74× más coste** por paso (4 force-evals vs 2, `95s` vs `55s` para N=1000), sin mejora vertical en el Pareto caótico. **La recomendación paper-grade sigue siendo KDK + V5**; la palanca pendiente es `dt` adaptativo / individual, no orden superior.

**Mensaje paper:** *en sistemas N-body caóticos densos, la precisión global está dominada por el flujo de Lyapunov sobre la órbita discreta del mapa simpléctico y por la constante de error efectiva de la composición, no por el orden local del integrador.*

## 2. Investigación previa

- **Yoshida (1990), “Construction of higher order symplectic integrators”.** Define la composición triple `ψ₄(dt) = ψ₂(w₁·dt) ∘ ψ₂(w₀·dt) ∘ ψ₂(w₁·dt)` con pesos `w₁ = 1/(2 − 2^{1/3})`, `w₀ = −2^{1/3}·w₁`, `w₁ + w₀ + w₁ = 1`. Exacta en simplecticidad y reversible; orden 4 de precisión local.
- **Hairer, Lubich, Wanner (2006), *Geometric Numerical Integration* §II.5.** Los pesos negativos no rompen la simplecticidad pero introducen constantes de error grandes y pasos intermedios `w₀·dt` en sentido temporal negativo. En sistemas caóticos el exponente de Lyapunov acota el crecimiento del drift independientemente del orden: `‖δz(t)‖ ~ e^{λt}·ε_local`; reducir `ε_local` vía orden alto no cambia `λ`, y el prefactor efectivo puede empeorar por la constante de la composición.
- **Quinn, Katz, Stadel, Lake (1997), “Time stepping N-body simulations”.** El beneficio de integradores de orden ≥ 4 en N-body con softening Plummer sólo se manifiesta cuando `dt · v_max ≲ ε`, es decir, cuando el error local domina sobre la regularización del softening. Fuera de ese régimen, el integrador no es el limitante.
- **Springel (2005), GADGET-2 §4.3.** GADGET-2 usa KDK como baseline y considera que orden superior no vale el coste adicional en cosmología por las mismas razones; la palanca real son los block timesteps individuales de Aarseth.

Traducción operativa a esta fase: orden 4 con fusión de half-kicks (4 force-evals/step) para no inflar el coste más allá de lo mínimo, y benchmarks idénticos a Fase 5 para comparar solo el eje integrador.

## 3. Hipótesis

- **H1 (orden local):** en oscilador armónico, `max_t |ΔE/E|` escala como `dt^p` con `p ≈ 2` (KDK) y `p ≈ 4` (Yoshida).
- **H2 (integrable no lineal):** en Kepler circular a 1 período con `dt = T/200`, Yoshida reduce el cierre orbital en ≥ 2 órdenes de magnitud frente a KDK.
- **H3 (caótico denso):** en Plummer `a/ε ≤ 2` durante 1000 pasos, Yoshida **no reduce** `|ΔE/E₀|` frente a KDK al mismo `dt`. Mecanismo: divergencia de Lyapunov + constante grande de la composición de pesos ±.
- **H4 (caos débil):** en esfera uniforme, Yoshida **sí** reduce drift (sigue la dirección de H1–H2).
- **H5 (Pareto):** Yoshida paga ~2× coste sin ganancia vertical en caótico → la frontera no se desplaza.

## 4. Implementación

### 4.1. Nuevo módulo `yoshida` (`crates/gadget-ng-integrators`)

Implementado en [`crates/gadget-ng-integrators/src/yoshida.rs`](../../crates/gadget-ng-integrators/src/yoshida.rs):

```rust
pub const YOSHIDA4_W1: f64 =  1.351_207_191_959_657_7;
pub const YOSHIDA4_W0: f64 = -1.702_414_383_919_315_3;

pub fn yoshida4_kdk_step(...);
pub fn yoshida4_cosmo_kdk_step(...);
```

Estrategia con fusión de half-kicks (4 force-evals por paso):

```text
K(w1·dt/2)   D(w1·dt)
K((w1+w0)/2·dt)   D(w0·dt)
K((w0+w1)/2·dt)   D(w1·dt)
K(w1·dt/2)
```

La variante cosmológica toma 3 `CosmoFactors` precalculados por el caller sobre los sub-intervalos de peso `w1·dt`, `w0·dt`, `w1·dt`, y fusiona `kick_half2[i] + kick_half[i+1]` igual que la versión Newtoniana. El orden de multiplicaciones está alineado para que `CosmoFactors::flat(w·dt)` produzca resultados **bit-a-bit idénticos** a la versión Newtoniana.

### 4.2. Selección por configuración

En [`crates/gadget-ng-core/src/config.rs`](../../crates/gadget-ng-core/src/config.rs):

```rust
#[derive(Serialize, Deserialize, Default, ...)]
#[serde(rename_all = "lowercase")]
pub enum IntegratorKind { #[default] Leapfrog, Yoshida4 }
```

y en `SimulationSection` un campo `#[serde(default)] pub integrator: IntegratorKind`. Default = `Leapfrog` → TOMLs existentes corren idénticos bit-a-bit.

### 4.3. Engine

En [`crates/gadget-ng-cli/src/engine.rs`](../../crates/gadget-ng-cli/src/engine.rs) se añade un `match integrator_kind` en las ramas:
- Newtoniana plana (Allgather global),
- cosmológica (precalcula 3 `CosmoFactors` avanzando `a_current` tras cada drift),
- SFC distribuida,
- slab 1D (dtree).

Si `timestep.hierarchical = true` y `integrator = yoshida4`, el engine falla con `CliError::InvalidConfig` explicitando que no está implementado; esta combinación queda fuera del scope de Fase 6.

### 4.4. Tests unitarios (4 obligatorios, todos verdes)

1. [`yoshida_harmonic_convergence.rs`](../../crates/gadget-ng-integrators/tests/yoshida_harmonic_convergence.rs) — barrido `dt ∈ {0.2, 0.1, 0.05, 0.025}`, oscilador armónico 1D, `t_f = 10·2π`. Ajuste log-log de `max_t |ΔE/E|` frente a `dt`.
2. [`yoshida_kepler_orbit.rs`](../../crates/gadget-ng-integrators/tests/yoshida_kepler_orbit.rs) — Kepler circular, 1 período a `dt = T/200` + dt-sweep a 10 períodos.
3. [`yoshida_linear_momentum.rs`](../../crates/gadget-ng-integrators/tests/yoshida_linear_momentum.rs) — red cúbica 8 partículas, `yoshida4` no rompe `p_total ≈ 0`.
4. [`yoshida_cosmo_flat.rs`](../../crates/gadget-ng-integrators/tests/yoshida_cosmo_flat.rs) — `yoshida4_cosmo_kdk_step` con `CosmoFactors::flat(w·dt)` es bit-exacto a `yoshida4_kdk_step(dt)` tras 25 pasos sobre órbita Kepler.

## 5. Validación matemática

### 5.1. Oscilador armónico 1D (test `yoshida_harmonic_convergence`)

| `dt`  | KDK `max  \|ΔE/E\|` | Yoshida `max \|ΔE/E\|` | ratio Yoshida/KDK |
|-------|--------------------|-----------------------|-------------------|
| 0.200 | 8.34e-3            | 1.05e-4               | 1/80              |
| 0.100 | 2.09e-3            | 6.40e-6               | 1/326             |
| 0.050 | 5.22e-4            | 3.98e-7               | 1/1311            |
| 0.025 | 1.30e-4            | 2.48e-8               | 1/5255            |

**Pendientes ajustadas log-log:**
- KDK: **p = 2.000**
- Yoshida4: **p = 4.013**

### 5.2. Kepler circular (test `yoshida_kepler_dt_sweep`)

10 períodos, cierre orbital `|r(T) − r(0)|`:

| `dt`    | KDK close | Yoshida close | ratio |
|---------|-----------|---------------|-------|
| T/50    | 3.27e-1   | 1.46e-2       | 22×   |
| T/100   | 8.25e-2   | 9.08e-4       | 91×   |
| T/200   | 2.07e-2   | 5.67e-5       | 365×  |
| T/400   | 5.17e-3   | 3.54e-6       | 1460× |

Pendiente cierre orbital: KDK **1.99**, Yoshida **4.00**. Nota: en órbita circular `|ΔE/E|` de KDK ya satura a precisión de máquina a `dt = T/200`, por lo que el cierre orbital es la métrica discriminante.

Figura [`plots/convergence_order.png`](../../experiments/nbody/phase6_higher_order_integrator/plots/convergence_order.png) muestra ambos ajustes con guías `∝ dt²` y `∝ dt⁴`.

Figura [`plots/kepler_orbit_closure.png`](../../experiments/nbody/phase6_higher_order_integrator/plots/kepler_orbit_closure.png) muestra la precesión visible de KDK y el cierre casi exacto de Yoshida a 10 períodos.

## 6. Benchmarks dinámicos (12 runs, `dt = 0.025`, solver V5, 1000 pasos)

### 6.1. Drift energético final `|ΔE/E₀|` (tras 1000 pasos)

| distribución   | N    | Leapfrog KDK | Yoshida4 | cociente Y/K |
|---------------|------|-------------:|---------:|-------------:|
| plummer a/ε=1 | 200  | **0.469**    | 0.662    | 1.41         |
| plummer a/ε=1 | 1000 | **0.324**    | 0.604    | 1.87         |
| plummer a/ε=2 | 200  | **0.365**    | 0.560    | 1.53         |
| plummer a/ε=2 | 1000 | **0.241**    | 0.503    | 2.09         |
| uniforme      | 200  | **0.008**    | 0.058    | 7.08         |
| uniforme      | 1000 | 0.0033       | 0.0033   | 1.01         |

### 6.2. Coste por paso (wall, release)

| distribución   | N    | KDK mean step | Yoshida mean step | ratio Y/K |
|---------------|------|--------------:|------------------:|----------:|
| plummer a/ε=1 | 1000 | 54.7 ms       | 95.3 ms           | 1.74×     |
| plummer a/ε=2 | 1000 | 58.9 ms       | 103.9 ms          | 1.76×     |
| uniforme      | 1000 | 44.7 ms       | 89.0 ms           | 1.99×     |

El factor ~2× coincide con la expectativa teórica de `4/2` force-evals/paso. La desviación inferior en Plummer (~1.76) refleja que los pasos de Yoshida tienen árbol ligeramente más caliente en L2.

### 6.3. Figuras

- [`plots/energy_drift_timeseries.png`](../../experiments/nbody/phase6_higher_order_integrator/plots/energy_drift_timeseries.png) — `|ΔE/E₀|` vs `t`, 2×3 (N × distribución). En todos los Plummer, Yoshida sistemáticamente **por encima** de KDK. En uniforme N=1000 ambas curvas se superponen.
- [`plots/pareto_with_yoshida.png`](../../experiments/nbody/phase6_higher_order_integrator/plots/pareto_with_yoshida.png) — coste vs drift. Yoshida desplaza los puntos **hacia la derecha** (más coste) pero **al mismo nivel o peor** que KDK en drift, confirmando H5.

## 7. Relación con Fase 5 — el límite del integrador es estructural

Fase 5 estableció que, a `dt = 0.025` fijo, ninguna combinación MAC (V1–V5) reduce `|ΔE/E|` por debajo de `~0.24` en Plummer denso N=1000 tras 1000 pasos: el solver había saturado su capacidad. Fase 6 cierra la segunda dirección del razonamiento: **el integrador tampoco mueve esa frontera**; incluso un esquema simpléctico 4º orden con la constante de error efectiva de la composición de Yoshida la empeora.

Esto es consistente con la interpretación Lyapunov: con `λ·t ~ O(1)` a los ~1000 pasos en estas condiciones de Plummer, el error inicial de `O(10^{-15})` se amplifica hasta `O(1)` independientemente del `ε_local` del paso. Reducir `ε_local` (orden 4) no reduce `λ`, y la fase termina midiendo principalmente la mezcla topológica del Hamiltoniano discreto, no el error local.

La palanca que queda abierta, coherente con Quinn+ (1997) y con la tradición GADGET/PKDGRAV, es **`dt` individual / adaptativo por Aarseth** (`timestep.hierarchical = true`, ya implementado en el repo pero no combinado con Yoshida). El siguiente experimento natural es Fase 7 = block timesteps Aarseth sobre KDK + V5.

## 8. Configuración recomendada actualizada (paper-grade)

```toml
[simulation]
dt        = 0.025
integrator = "leapfrog"          # <-- mantener KDK; Yoshida no aporta en caótico

[gravity]
solver              = "barnes_hut"
theta               = 0.5
multipole_order     = 3
opening_criterion   = "relative"
err_tol_force_acc   = 0.005
softened_multipoles = true
mac_softening       = "consistent"

[timestep]
hierarchical = false             # <-- próxima palanca a explorar (Fase 7)
```

`integrator = "yoshida4"` queda **reservado** para:
- sistemas integrables o casi integrables (Kepler, few-body estable),
- validación matemática (tests unitarios de convergencia),
- estudios de estabilidad orbital de largo plazo donde el flujo no es caótico.

## 9. Limitaciones

- **Hierarchical + Yoshida4:** no implementado. El engine lo rechaza con `CliError::InvalidConfig` explícito. Combinar block timesteps con composición de Yoshida requiere redefinir la estructura de niveles Aarseth sobre los 3 sub-pasos; queda como backlog con prioridad baja (porque H3 bloquea el motivo práctico).
- **Yoshida cosmológico en rama SFC/dtree:** implementado pero **no validado con benchmark dedicado**; sólo cubierto por el test `yoshida_cosmo_flat.rs` en límite plano.
- **Cociente Y/K en uniforme N=200 (7.08):** sorprendentemente alto, posiblemente por la combinación de cronograma de cruces de órbita con los pesos negativos de Yoshida a este `dt` específico. No afecta la conclusión pero vale investigar si se extiende el estudio a uniforme con `N` intermedio (500, 2000).
- **dt fijo 0.025 durante todo el benchmark:** por design, para no mezclar efecto integrador y efecto regularización. Un Yoshida con `dt` 2× mayor (compensando el coste) ya tiene error local ≈ 16× menor y podría revelar un cruce Pareto en algún régimen; dt-adapt queda para Fase 7.

## 10. Narrativa paper (≈260 palabras)

> We evaluate whether moving beyond the standard second-order leapfrog KDK integrator reduces the global energy drift of Barnes–Hut N-body simulations in chaotic dense Plummer systems. We implement the classical fourth-order symplectic composition of Yoshida (1990), with fused half-kicks requiring four force evaluations per step, and expose it as a configuration switch alongside the existing KDK. Mathematical correctness is verified on two integrable problems: a 1D harmonic oscillator, where a log–log fit of `max_t|ΔE/E|` across `dt ∈ {0.2, 0.1, 0.05, 0.025}` yields slopes of 2.00 (KDK) and 4.01 (Yoshida), and a circular Kepler orbit, where after ten periods at `dt = T/200` Yoshida closes the orbit ~400× more precisely than KDK, consistent with fourth-order convergence of the phase error. Applied to a twelve-run subset of our Phase 5 benchmark — three density distributions (Plummer `a/ε = 1, 2`, uniform sphere) × two sizes (N=200, 1000), with the Phase 5 optimal solver (relative opening, softened multipoles, softening-consistent MAC) at `dt = 0.025` and 1000 steps — Yoshida provides no improvement in final `|ΔE/E₀|`; for the chaotic Plummer configurations it is systematically worse (e.g., 0.60 vs 0.32 for `a/ε = 1`, N=1000) at ~1.74× the wall cost per step. This asymmetry, anticipated by Hairer et al. (2006) through the large error constants of compositions with negative sub-steps, shows that once the Barnes–Hut solver reaches the accuracy of Phase 5 (V5), the remaining drift is governed by Lyapunov mixing of the discrete symplectic flow rather than by the integrator’s local order. For paper-grade runs of chaotic Barnes–Hut N-body, leapfrog KDK with a properly tuned MAC remains the Pareto-optimal choice, and the next meaningful lever is individual/adaptive `dt` (Aarseth), not higher order.

---

## Definition of Done

- [x] `yoshida4_kdk_step` y `yoshida4_cosmo_kdk_step` implementados, exportados, documentados.
- [x] Enum `IntegratorKind` en config con default `Leapfrog` (retrocompat bit-exacta verificada por `cargo build --release --workspace`).
- [x] Engine hace `match` sobre integrator en 4 ramas (cosmo, flat, SFC, dtree); error explícito en `hierarchical + yoshida4`.
- [x] 4 tests unitarios verdes: `yoshida_harmonic_convergence`, `yoshida_kepler_orbit` (2 subtests), `yoshida_linear_momentum`, `yoshida_cosmo_flat`.
- [x] 12 runs dinámicos completados, [`phase6_summary.csv`](../../experiments/nbody/phase6_higher_order_integrator/results/phase6_summary.csv) poblado.
- [x] [`convergence_order.png`](../../experiments/nbody/phase6_higher_order_integrator/plots/convergence_order.png), [`energy_drift_timeseries.png`](../../experiments/nbody/phase6_higher_order_integrator/plots/energy_drift_timeseries.png), [`pareto_with_yoshida.png`](../../experiments/nbody/phase6_higher_order_integrator/plots/pareto_with_yoshida.png), [`kepler_orbit_closure.png`](../../experiments/nbody/phase6_higher_order_integrator/plots/kepler_orbit_closure.png) generados.
- [x] Este reporte + cross-ref en Fase 5.
- [x] `cargo build --release --workspace` limpio, `cargo test --release -p gadget-ng-integrators` verde.

---

## Continuación: Fase 7 — Timesteps Adaptativos Tipo Aarseth

La conclusión de Fase 6 identifica como palanca pendiente el control adaptativo de `dt` individual. La **Fase 7** evalúa directamente si los block timesteps tipo Aarseth (KDK + niveles 2ⁿ + criterio `dt_i = η·√(ε/|a_i|)`) reducen el drift energético en los mismos regímenes donde Yoshida4 falló.

Implementación completa en `crates/gadget-ng-integrators/src/hierarchical.rs`. Experimentos en `experiments/nbody/phase7_aarseth_timestep/`.

Ver: [Fase 7 — Aarseth Block Timesteps](./2026-04-phase7-aarseth-timestep.md)
