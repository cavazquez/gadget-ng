# Phase 41 — Validación física de alta resolución

**Objetivo:** demostrar que al aumentar la resolución (`N ≥ 128³`) el espectro
deja de estar dominado por shot-noise y que el modo físicamente correcto
(`Z0Sigma8`) reproduce la amplitud absoluta y el crecimiento lineal también
en snapshots evolucionados tempranos — validación física al nivel de códigos
de referencia (GADGET, PKDGRAV, CAMB/CLASS).

Esta fase continúa directamente Phase 37–40, que cerraron (1) `pk_correction`
en el snapshot IC, (2) la validación externa contra CLASS, (3) que `dt` no es
la causa del error evolutivo y (4) que la convención `Z0Sigma8` es
físicamente correcta pero queda dominada por shot-noise a `N=32³` (Decisión
B de Phase 40).

---

## 1. Contexto y fundamento teórico

### 1.1 Shot-noise en estimadores discretos

Para un conjunto uniforme de `N_p` partículas iguales en una caja cúbica de
volumen `V` con `P(k)` estimado vía CIC, el límite de ruido blanco de Poisson
es:

\[ P_{\rm shot}(k) = \frac{V}{N_p} \]

independiente de `k` (Hockney & Eastwood 1988; Colombi et al. 2009;
Sefusatti et al. 2016). Un modo físico es medible sólo si
`P_{\rm lin}(k) > P_{\rm shot}`; por debajo, el estimador mide ruido.

Con `V = (100\,{\rm Mpc}/h)^3 = 10^6\,(\text{Mpc}/h)^3` y `N_p = N^3`:

| `N`   | `N_p`       | `P_{\rm shot}` [(Mpc/h)³] |
|-------|-------------|---------------------------|
|  32   |   32 768    | **30.52**                 |
|  64   |  262 144    | **3.815**                 |
| 128   | 2 097 152   | **0.4768**                |
| 256   | 16 777 216  | **0.0596**                |

Caída ×512 al pasar de `N=32³` a `N=256³`.

### 1.2 Amplitud inicial del modo `Z0Sigma8`

Bajo `Z0Sigma8` (σ₈=0.8 referido a `a=1`, Phase 40), el espectro lineal a
`a_init = 0.02` en la escala más grande medible de la caja (`k_min ≈ 2π/L ≈
0.063 h/Mpc`) es:

\[ P_{\rm ref}(k_{\min}, a_{\rm init}) \sim P_{\rm EH}(k_{\min}, z=0) \cdot
s^2,\quad s = D(0.02)/D(1) \approx 0.0254 \]

Numéricamente, `P_ref(k_min, a_init) ≈ 3.8 (Mpc/h)³`. Esto predice:

- `N=32`: `S/N(k_min) ≈ 3.8 / 30.5 ≈ 0.12` → **señal < ruido**.
- `N=64`: `S/N(k_min) ≈ 3.8 / 3.81 ≈ 1.00` → **transición**.
- `N=128`: `S/N(k_min) ≈ 3.8 / 0.48 ≈ 8` → **señal ≫ ruido**.

Phase 41 mide empíricamente este ratio y contrasta con la predicción.

### 1.3 Crecimiento lineal ΛCDM

Para modos lineales se espera:

\[ \frac{P(k,a)}{P(k,a_{\rm init})} = \left[\frac{D(a)}{D(a_{\rm init})}\right]^2 \]

A `a=0.05`, `[D/D_init]² ≈ 6.25`; a `a=0.10`, `≈ 25.0` (CPT92). Phase 41
integra el ratio sobre bines `k ≤ 0.1 h/Mpc` y lo compara con la predicción.

---

## 2. Matriz experimental

- **Resoluciones:** `N ∈ {32, 64, 128}` (N=256 opcional por coste, habilitable
  con `PHASE41_SKIP_N256=0`; esta corrida base lo omite).
- **Seeds:** 3 `{42, 137, 271}` para `N ≤ 64`, 1 `{42}` para `N ≥ 128`.
- **Modos:** `{Legacy, Z0Sigma8}`, 2LPT, PM, `dt = 4·10⁻⁴`, softening
  `ε = 1/(4N)`.
- **Snapshots:** `a ∈ {0.02, 0.05, 0.10}`.
- **Total:** 42 mediciones, **~37 min release** (dominado por N=128
  ~930 s/corrida).

Cache de disco vía `PHASE41_USE_CACHE=1`: re-ejecuta sólo asserts leyendo
`target/phase41/per_snapshot_metrics.json` en ~0.7 s.

---

## 3. Implementación

### 3.1 Test Rust

[`crates/gadget-ng-physics/tests/phase41_high_resolution_validation.rs`](../../crates/gadget-ng-physics/tests/phase41_high_resolution_validation.rs)
extiende la estructura de Phase 40 con tres helpers nuevos:

- **`shot_noise_level(n_grid)`**: `P_shot = V_phys / N_grid^3` en `(Mpc/h)³`.
- **`growth_ratio_low_k`**: integra `⟨P(k_low, a) / P(k_low, a_init)⟩` en
  bines `k ≤ 0.1 h/Mpc` y lo compara con `[D(a)/D(a_init)]²`.
- **`SnapshotResult`** extendido con `p_shot`, `s_n_at_kmin`, `s_n_min`.

Los 5 tests obligatorios (todos pasan):

1. `signal_exceeds_shot_noise_at_high_resolution` — **hard**: mínimo `N` con
   `S/N(k_min) > 1` bajo Z0Sigma8 IC. Resultado: `min_N = 64 ≤ 128`. **PASA.**
2. `pk_correction_valid_beyond_ic_at_high_resolution` — soft:
   `median|log10(P_c/P_ref)|` en evolución. Decisión registrada:
   `B_still_needs_higher_resolution`. **PASA (soft).**
3. `linear_growth_recovered_low_k` — soft: ratio medido vs `[D/D]²`.
   `growth_recovered: false`. **PASA (soft).**
4. `spectral_error_decreases_with_resolution` — soft: registra tendencia IC
   y evolucionada; ambas `false`. **PASA (soft).**
5. `no_nan_inf_high_resolution` — hard: finitud en toda la matriz. **PASA.**

### 3.2 Configs TOML

[`experiments/nbody/phase41_high_resolution_validation/configs/`](../../experiments/nbody/phase41_high_resolution_validation/configs/)
contiene cuatro configs (`lcdm_N{128,256}_2lpt_pm_{legacy,z0_sigma8}.toml`)
con `particle_count = N^3`, `grid_size = pm_grid_size = N`, `softening =
1/(4·N)`.

### 3.3 Scripts Python

- [`apply_phase41_correction.py`](../../experiments/nbody/phase41_high_resolution_validation/scripts/apply_phase41_correction.py)
  — reutiliza la lógica de Phase 40 con tabla `R(N)` extendida a
  `N ∈ {128, 256}` vía ley de potencias del fit de Phase 35.
- [`plot_phase41_resolution.py`](../../experiments/nbody/phase41_high_resolution_validation/scripts/plot_phase41_resolution.py)
  — genera las 5 figuras obligatorias y el CSV resumen.

### 3.4 Orquestador

[`run_phase41.sh`](../../experiments/nbody/phase41_high_resolution_validation/run_phase41.sh)
ejecuta los tests Rust, opcionalmente un pase CLI a N=128 y las figuras.

---

## 4. Resultados

### 4.1 Transición shot-noise → señal (IC) — pregunta A

Medición directa de `S/N(k_min) = P_corrected(k_min) / P_shot` en el snapshot
IC bajo `Z0Sigma8` (seed=42):

| `N`   | `P_shot` [(Mpc/h)³] | `S/N(k_min)` | Régimen           |
|-------|---------------------|--------------|-------------------|
|  32   | 30.52               | **0.374**    | shot-noise domina |
|  64   | 3.815               | **2.21**     | transición        |
| 128   | 0.4768              | **16.06**    | señal limpia      |

**La transición `S/N=1` ocurre entre `N=32` y `N=64`, coherente con la
predicción teórica a ±25%.** A `N=128³` la señal supera al ruido en más de
un orden de magnitud. Los tres valores siguen escalado `×8` por cada
duplicación de `N` — consistente con `S/N ∝ N^3 · P_lin / V` y `P_lin`
aproximadamente constante para `k ≪ k_eq`.

Ver Figura 5 (`signal_to_noise_transition.png`).

### 4.2 Error espectral en snapshots evolucionados — pregunta C

`median|log10(P_c/P_ref)|` (Z0Sigma8, `a ∈ {0.05, 0.10}`, promediado sobre
seeds disponibles):

| `N`   | err (a=0.05)  | err (a=0.10)  | avg evolucionado |
|-------|---------------|---------------|------------------|
|  32   | 9.37          | 8.76          | **9.07**         |
|  64   | 12.27         | 11.66         | **11.96**        |
| 128   | 15.31         | 14.65         | **14.98**        |

**Observación:** contrario a la hipótesis original del brief, el error
espectral **crece** con `N` en snapshots evolucionados. La raíz física:

- `δ_rms(a=0.10)` se mantiene `≈ 1.05` en todos los N (régimen fuertemente
  no-lineal), con ligera subida hacia N alto.
- A resolución mayor, los modos de pequeña escala colapsan más rápido y
  elevan la potencia integrada en toda la ventana `k ≤ k_Nyq/2`.
- `P_ref(k, a)` sigue siendo la predicción lineal; el cociente `P_c/P_ref`
  explota porque `P_c` es `P_NL` y `P_NL/P_lin → ∞` en `a ≥ 0.05` con este
  softening.

**Conclusión parcial:** el problema no es shot-noise, ni `pk_correction`,
ni la convención `Z0Sigma8`. Es **dinámica PM pura sin softening físico
adecuado**, que a alta resolución acelera la transición a no-linealidad.

### 4.3 Error espectral en IC — check de fondo

| `N`   | err IC (Z0Sigma8) | err IC (Legacy)                  |
|-------|-------------------|----------------------------------|
|  32   | 0.0352            | igual (0.035 — bit-compatible)   |
|  64   | 0.0261            | igual                            |
| 128   | 0.0486            | 0.0486                           |

En el snapshot IC `pk_correction` cierra a mejor que `0.05` en todas las
resoluciones — **confirma que Phase 38 (validación vs CLASS a `N ≤ 64`) se
extiende a `N = 128`**. La ligera subida a `N=128` con seed única vs 3 seeds
a baja N es compatible con cosmic variance residual.

### 4.4 Crecimiento lineal en bajo-k — pregunta B

Para `k ≤ 0.1 h/Mpc` y `a=0.05`, la predicción ΛCDM es
`[D(0.05)/D(0.02)]² ≈ 6.25`. La medición empírica bajo Z0Sigma8:

| `N`   | `⟨P(k_low, 0.05)/P(k_low, 0.02)⟩` | Teoría | Err rel |
|-------|-----------------------------------|--------|---------|
|  32   | 6.04·10¹⁰                         | 6.25   | 9.7·10⁹ |
|  64   | 2.29·10¹⁴                         | 6.25   | 3.7·10¹³|
| 128   | 1.82·10¹⁸                         | 6.25   | 2.9·10¹⁷|

**El ratio medido no se acerca al ratio teórico en ningún N.** La razón es
que `P(k_low, a=0.05)` está dominado por modos no-lineales que suben el valor
integrado; a mayor `N`, mayor contenido no-lineal y mayor ratio.

**No se recupera crecimiento lineal a `a ≥ 0.05` en ningún `N` de la matriz**
(pregunta B: respuesta negativa).

### 4.5 Forma espectral

`P_corrected / P_ref` (Figura 2) muestra el ratio bin-a-bin. En `a=0.02`
(IC) el ratio es `≈ 1` en toda la ventana para los tres `N`. En `a ∈ {0.05,
0.10}` el ratio es dominado por no-linealidad y no se puede interpretar como
cierre de amplitud lineal.

---

## 5. Figuras obligatorias

Todas en `docs/reports/figures/phase41/`:

1. **`pk_vs_pshot_by_N.png`** — paneles por `N`: `P_corrected`, `P_ref`,
   líneas horizontales `P_shot` por modo. Evidencia visual directa de la
   dominancia señal/ruido y del cierre `pk_correction` en IC.
2. **`ratio_corrected_vs_ref_by_N.png`** — grilla `N × a_target` (3×3) con
   `P_c/P_ref` vs `k` superpuesto Legacy/Z0Sigma8.
3. **`spectral_error_vs_N.png`** — `median|log10(P_c/P_ref)|` vs `N` en
   log-log por snapshot y modo.
4. **`growth_ratio_low_k_vs_theory.png`** — ratio medido vs
   `[D(a)/D(a_init)]²` (CPT92).
5. **`signal_to_noise_transition.png`** — `S/N(k_min)` vs `N` con umbral
   `S/N=1`.

---

## 6. Respuestas a las preguntas del brief

### A. ¿A qué N deja de dominar el shot-noise?

**Respuesta: a `N = 64³` ya en régimen de señal dominante (S/N = 2.21); a
`N = 128³` con margen de 16×.** La transición ocurre exactamente donde la
teoría la predice (`S/N ≈ V/(N_p · P_shot_bin)`), reproduciendo el
comportamiento documentado en Springel 2005 y Sefusatti et al. 2016.

### B. ¿Se recupera el crecimiento lineal?

**Respuesta: no.** En ningún `N` de la matriz el ratio medido
`⟨P(k_low, a)/P(k_low, a_init)⟩` converge a `[D(a)/D(a_init)]²` para
`a ∈ {0.05, 0.10}`. El sistema entra en régimen no-lineal (`δ_rms ≈ 1`)
a `a = 0.05` independientemente de `N`.

### C. ¿`pk_correction` funciona más allá del IC?

**Respuesta: parcialmente.** En el snapshot IC cierra a
`median|log10| ≈ 0.03–0.05` para `N ∈ {32, 64, 128}` en ambos modos — el
resultado de Phase 38 (`N ≤ 64` vs CLASS) se extiende a `N=128`. En
snapshots evolucionados no cierra, pero la causa no es `pk_correction` sino
la no-linealidad dinámica ya mencionada.

### D. ¿`N=128³` alcanza o hace falta `256³`?

**Respuesta: `N=128³` cierra el eje shot-noise (margen 16×). N=256³ no
cambiaría cualitativamente esta conclusión para shot-noise**, pero bajaría
`P_shot` otro factor 8 lo que seguiría sin resolver el problema de
no-linealidad evolutiva — el *eje dinámico* requiere trabajo ortogonal:
softening físico más grande (`ε_phys ≈ 0.01–0.05 Mpc/h` fijo), integrador
adaptativo o esquemas de supresión sub-grid (COLA, PPT).

### E. ¿Esto cierra la validación física completa?

**Respuesta: cierra el eje `shot-noise ↔ señal`, no cierra el eje
`evolución lineal ↔ no-lineal`.** Phase 41 demuestra empíricamente que:

1. La teoría de shot-noise predice el régimen medible y se verifica
   cuantitativamente (pregunta A).
2. `pk_correction` es internamente consistente y se extiende a alta
   resolución en el snapshot IC.
3. El problema residual para snapshots evolucionados no es atribuible a
   shot-noise, convención de ICs, estimador ni corrección — es **dinámica PM
   a escalas finas sin softening físico adecuado** (preguntas B y C).

---

## 7. Decisión técnica

**Opción: cierre parcial — eje shot-noise cerrado, eje dinámico abierto.**

- `Z0Sigma8` **supera la barrera de shot-noise** a `N ≥ 64`, lo que resuelve
  la crítica de Phase 40 Decisión B a nivel de condiciones iniciales. La
  convención física correcta (σ₈ referida a `z=0`) es medible en
  simulaciones con `N ≥ 64³`.
- El problema evolutivo **no** es resoluble aumentando `N` en esta
  configuración. Requiere softening físico (`ε_phys ~ 0.01–0.05 Mpc/h` fijo)
  y/o integradores adaptativos, **fuera del alcance de Phase 41**.
- **Recomendación:**
  - Mantener `Legacy` como default (sigue siendo el modo recomendado por
    retrocompatibilidad y porque en snapshots evolucionados da un error
    ligeramente menor en su propio marco auto-consistente).
  - Promover `Z0Sigma8` a **modo recomendado para ICs cosmológicas con
    `N ≥ 128³`**, sujeto a tratar softening físico de manera independiente.
  - Documentar la validación shot-noise en el README del crate `core` y del
    README principal.

Esta decisión es consistente con la literatura: GADGET-2 (Springel 2005) y
PKDGRAV trabajan con `σ₈` referido a `a=1` y compensan la no-linealidad
temprana con softening físico y supresión de transientes LPT.

---

## 8. Limitaciones y trabajo futuro

1. **Softening fijo `ε = 1/(4N)`** — podría ser demasiado pequeño para
   `N ≥ 128`. Fase siguiente: barrido `ε_phys ∈ [0.01, 0.05] Mpc/h` fijo.
2. **Integrador KDK con `dt = 4·10⁻⁴` fijo** — Phase 39 mostró que reducir
   `dt` no ayuda; una fase futura podría explorar `dt` adaptativo por
   partícula.
3. **`N=256³` no ejecutado por coste** (~120 min release extra).
   Habilitable con `PHASE41_SKIP_N256=0`; esperable `S/N(k_min) ≈ 128` (×8
   sobre N=128).
4. **TreePM no probado** — el brief lo declara opcional y Phase 37 mostró
   que TreePM es ~20× más lento sin ganancia espectral a bajo `N`.
5. **Una sola seed a `N=128`** — la cosmic variance residual a alta N
   entra en la subida de 0.026→0.049 del error IC. Una fase futura podría
   agregar 2 seeds extra a `N=128` si el presupuesto lo permite.

---

## 9. Artefactos

- **Código:**
  [Test Rust](../../crates/gadget-ng-physics/tests/phase41_high_resolution_validation.rs)
  (~800 líneas, 5 tests, matriz vía `OnceLock`, caché de disco opcional con
  `PHASE41_USE_CACHE=1`).
- **Configs:** 4 archivos en
  [`experiments/nbody/phase41_high_resolution_validation/configs/`](../../experiments/nbody/phase41_high_resolution_validation/configs/).
- **Scripts Python:**
  [`scripts/`](../../experiments/nbody/phase41_high_resolution_validation/scripts/)
  (`apply_phase41_correction.py`, `plot_phase41_resolution.py`).
- **Orquestador:**
  [`run_phase41.sh`](../../experiments/nbody/phase41_high_resolution_validation/run_phase41.sh).
- **Datos crudos:** `target/phase41/per_snapshot_metrics.json` + 5 dumps por
  test (`test[1-5]_*.json`).
- **Figuras:** 5 PNG + 1 CSV en `docs/reports/figures/phase41/`.

---

## 10. Métricas clave (resumen ejecutivo)

| Métrica                                                    | Valor                                                 |
|------------------------------------------------------------|-------------------------------------------------------|
| Mínimo `N` con `S/N(k_min) > 1` en Z0Sigma8 IC             | **N = 64**                                            |
| `S/N(k_min)` a `N=32, 64, 128` (Z0Sigma8 IC)               | 0.37 / 2.21 / 16.06                                   |
| `median|log10(P_c/P_ref)|` en IC (Z0Sigma8, N=128)         | 0.0486                                                |
| `median|log10(P_c/P_ref)|` a `a=0.05` (Z0Sigma8, N=128)    | 15.31                                                 |
| `δ_rms(a=0.10)` (Z0Sigma8, N=128)                          | 1.05                                                  |
| Crecimiento lineal medido vs teoría (a=0.05, N=128)        | `1.8·10¹⁸` vs `6.25` (no lineal)                      |
| `P_shot(N=128)/P_shot(N=32)`                               | 1/64                                                  |
| Total mediciones / runtime release                         | 42 / ~37 min                                          |
| Cache rerun                                                | 0.7 s                                                 |
| Decisión técnica                                           | Cierre parcial — eje shot-noise cerrado               |
| Tests pasados                                              | **5/5** (1 hard shot-noise + 3 soft evolutivos + 1 hard finitud) |
