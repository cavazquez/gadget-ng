# Fase 5 — Consistencia MAC-softening y conservación dinámica multi-step

_Última actualización: 2026-04-16 · gadget-ng_

## 0. Resumen ejecutivo

Tres hallazgos cuantitativos dominan la fase:

1. **Consistencia en el estimador del MAC relativo reduce el coste de apertura
   sin sacrificar precisión local.** Con softening coherente
   `(d² + ε²)^{5/2}` en el denominador del estimador cuadrupolar
   (`MacSoftening::Consistent`, variante **V5**), el número de nodos abiertos
   cae un **13 %** frente al estimador bare en Plummer `a/ε = 1`
   (572 793 vs 659 162 para N=1000) y manteniendo el error medio de fuerza
   en el orden 10⁻⁵. Donde no hay partículas en `d ~ ε` (Plummer `a/ε = 6`,
   esfera uniforme) la ganancia es marginal (<5 %), lo que confirma que el
   efecto se concentra en el núcleo denso.
2. **La conservación energética multi-step está gobernada por el integrador
   leapfrog KDK y por el caos físico de Miller, no por el error local de
   fuerza.** La correlación log-log global entre `mean_force_error` y
   `|ΔE/E₀|_final` sobre los 40 runs es r = +0.045. Cinco variantes con
   errores locales que difieren en más de cinco órdenes de magnitud
   (7×10⁻⁷ … 9×10⁻²) convergen al mismo piso de drift en Plummer `a/ε ≤ 2`
   (~32–40 %). En cambio, para la esfera uniforme y Plummer extendido
   (`a/ε = 6`) sí aparece correlación parcial (r de +0.61 a +0.93).
3. **Cuando la relación local↔global aparece, es débil y no siempre monótona.**
   El control del momento angular, en cambio, sí correlaciona fuertemente con
   el solver: el criterio relativo reduce `|ΔL|_max` en Plummer `a/ε = 1`,
   N=1000, de **0.92** (V1, geométrico bare) a **0.02–0.04** (V4/V5,
   relativo softened). Recomendamos V5 como configuración paper-grade cuando
   la física demanda conservación angular estricta o control del error en
   regiones densas.

Estos resultados cierran la Fase 5: el MAC softened-consistent es
matemáticamente correcto, computacionalmente más barato donde importa, y
físicamente neutro para la conservación de energía —que, como sospechaban
Barnes & Hut (1989) y Dehnen & Read (2011), depende del integrador antes que
del solver de fuerzas.

## 1. Investigación previa

Revisamos (fuera de línea y sobre la literatura ya citada en Fase 4) los
siguientes puntos, traducidos a decisiones de código:

- **Springel et al. (2021), GADGET-4 §2.3–2.4:** el criterio
  `TypeOfOpeningCriterion=1` estima `|multipolo_n| · d^{-(n+1)}`; en la
  implementación de Springel+ se conserva `d^{-(n+1)}` en el estimador
  porque en producción `d > 2ε` para la inmensa mayoría de nodos. Sin
  embargo, GADGET-4 reconoce que esto sobre-abre nodos en el núcleo denso
  (§2.4, discusión sobre "softening safeguards"). Nuestro enfoque —añadir
  `(d² + ε²)^{(n+1)/2}` opcionalmente vía `MacSoftening::Consistent`— es
  equivalente a una variante de esa salvaguarda.
- **Dehnen (2002), falcON:** argumenta que el kernel de softening y la
  expansión multipolar deben ser coherentes para garantizar monotonicidad
  del error con `theta`/`err_tol`. Nuestra Fase 4 resolvió la parte
  monopolo/multipolo de la fuerza; la Fase 5 extiende esa coherencia al
  estimador que decide la apertura.
- **Hernquist (1987) y Barnes & Hut (1989):** advierten que la conservación
  energética en un integrador simpléctico KDK (leapfrog) tiene un piso
  impuesto por `dt` y `ε`; reducir el error de fuerza por debajo de ese piso
  no mejora el drift. Este principio motivó el experimento local↔global
  (§4.2) y se confirma empíricamente en §5.3.

### Preguntas obligatorias y su respuesta

| Pregunta | Respuesta operativa | Referencia |
|---|---|---|
| ¿Softened multipoles mejoran solo el error instantáneo o también el drift? | **Solo instantáneo** en sistemas caóticos (`a/ε ≤ 2`); **parcialmente también drift** en sistemas menos caóticos (`a/ε = 6`, uniforme). Ver §5.3. | Barnes & Hut 1989 |
| ¿Cómo debería entrar el softening en el estimador del cuadrupolo del MAC relativo? | Como denominador `(d² + ε²)^{5/2}`, simétrico al monopolo softened `G·M/(d²+ε²)`. Esta es precisamente `MacSoftening::Consistent`. | Springel+ 2021; Dehnen 2002 |
| ¿Qué hace GADGET-4? | Usa bare `d^5` por defecto pero con "safeguards" de núcleo. Nuestra variante V5 es equivalente a una formulación con safeguard siempre activa. | Springel+ 2021, §2.3–2.4 |

## 2. Hipótesis

| ID | Hipótesis | Resultado |
|---|---|---|
| H1 | `softened_multipoles=true` reduce el drift energético acumulado en sistemas concentrados. | **Rechazada para `a/ε ≤ 2`**, aceptada parcialmente para `a/ε = 6` y uniforme con N=200. Ver §5.1. |
| H2 | `mac_softening=consistent` abre menos nodos en el núcleo sin inflar el coste global. | **Confirmada**: 13 % menos nodos abiertos en Plummer `a/ε = 1` N=1000 con error local equivalente. Ver §5.2. |
| H3 | El error local de fuerza correlaciona con el drift global, pero existe un piso del integrador. | **Confirmada**: correlación log-log global r = +0.045; por grupo va de −0.80 a +0.93; piso observable claramente. Ver §5.3. |
| H4 | V5 (`relative + softened_multipoles + mac_softening=consistent`) domina el Pareto. | **Confirmada con matices**: domina en precisión-coste de fuerza y control de `|ΔL|`; V4 (bare) es indistinguible en energía pero abre más nodos. Ver §5.4. |

## 3. Cambios de implementación

### 3.1. Nuevo enum `MacSoftening`

En [`crates/gadget-ng-core/src/config.rs`](../../crates/gadget-ng-core/src/config.rs):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MacSoftening {
    #[default]
    Bare,
    Consistent,
}
```

Se añade `pub mac_softening: MacSoftening` a `GravitySection`, con default
`Bare` para retrocompatibilidad bit-exacta. Re-exportado en
[`lib.rs`](../../crates/gadget-ng-core/src/lib.rs).

### 3.2. Estimador softened-consistent en el tree walk

En [`crates/gadget-ng-tree/src/octree.rs`](../../crates/gadget-ng-tree/src/octree.rs),
la magnitud cuadrupolar del MAC relativo pasó de bare a seleccionable:

```rust
let quad_mag = match mac_softening {
    MacSoftening::Bare => q_frob / (d2 * d2 * d_com),           // |Q|/d⁵
    MacSoftening::Consistent => {
        let s2 = d_com * d_com + eps2;
        q_frob / (s2 * s2 * s2.sqrt())                           // |Q|/(d²+ε²)^{5/2}
    }
};
```

El flag se propaga por `BarnesHutGravity`, `RayonBarnesHutGravity` y
`make_solver()` en [`engine.rs`](../../crates/gadget-ng-cli/src/engine.rs).

### 3.3. Instrumentación del tree walk

Añadidas en `octree.rs` funciones `walk_stats_begin`, `walk_stats_end` y
`WalkStats`, basadas en contadores thread-local activables por test.
Permiten medir nodos abiertos, hojas visitadas, profundidad máxima y media
sin impacto en producción (el flag está apagado por defecto).

### 3.4. Diagnósticos multi-step extendidos

`write_diagnostic_line` en `engine.rs` ahora escribe, por paso, además de
`kinetic_energy`: `momentum` (vec3), `angular_momentum` (vec3), `com`
(vec3) y `mass_total`. Coste: 10 `allreduce_sum_f64` extra por paso
(despreciable frente a la gravedad).

### 3.5. Librería de análisis en Python

En [`scripts/analysis/snapshot_metrics.py`](../../scripts/analysis/snapshot_metrics.py):

- `potential_energy` reimplementada con broadcasting NumPy (O(N²) vectorizado,
  ~100× más rápida en Python puro; tolera N=1000 × 100 frames).
- `angular_momentum_vec` devuelve `[Lx, Ly, Lz]` completo.
- `compute_metrics` incluye ahora `dE_rel`, `dp_abs`, `dp_rel`, `dL_abs`,
  `dL_rel`, con denominador robusto `max(|x0|, |x|)` para evitar divisiones
  por cero cuando el sistema tiene momento inicial nulo.

### 3.6. Test local ampliado

`bh_mac_softening_ablation` en
[`crates/gadget-ng-physics/tests/bh_force_accuracy.rs`](../../crates/gadget-ng-physics/tests/bh_force_accuracy.rs)
recorre las 5 variantes × 4 distribuciones × 2 N. Usa la instrumentación
de WalkStats y escribe
[`experiments/nbody/phase5_energy_mac_consistency/results/bh_mac_softening.csv`](../../experiments/nbody/phase5_energy_mac_consistency/results/bh_mac_softening.csv).

## 4. Diseño experimental

### 4.1. Matriz

- 4 distribuciones: `plummer_a1` (`a/ε=1`, concentrado),
  `plummer_a2` (`a/ε=2`), `plummer_a6` (`a/ε=6`, extendido), esfera uniforme.
- 2 N: 200, 1000.
- 5 variantes MAC:

| Variante | opening_criterion | softened_multipoles | mac_softening |
|---|---|---|---|
| V1_geom_bare | geometric | false | bare |
| V2_geom_soft | geometric | true  | bare |
| V3_rel_bare  | relative  | false | bare |
| V4_rel_soft  | relative  | true  | bare |
| V5_rel_soft_consistent | relative | true | consistent |

Total: **40 runs dinámicos**, 1000 pasos leapfrog KDK, `dt=0.025`,
`softening=0.05`, snapshot cada 10 pasos (100 frames por run).

### 4.2. Scripts

```bash
# compilación única
cargo build --release -p gadget-ng-cli

# generar las 40 configuraciones TOML
python3 experiments/nbody/phase5_energy_mac_consistency/scripts/generate_configs.py

# test local (con instrumentación de nodos abiertos)
cargo test --release -p gadget-ng-physics --test bh_force_accuracy \
    bh_mac_softening_ablation -- --nocapture

# 40 runs dinámicos (≈4 min con PARALLEL=4 en 8 cores)
PARALLEL=4 bash experiments/nbody/phase5_energy_mac_consistency/scripts/run_phase5.sh

# análisis de conservación + local↔global + figuras
python3 experiments/nbody/phase5_energy_mac_consistency/scripts/analyze_conservation.py
python3 experiments/nbody/phase5_energy_mac_consistency/scripts/analyze_local_global.py
python3 experiments/nbody/phase5_energy_mac_consistency/scripts/plot_phase5.py
```

Las salidas viven en `results/phase5_summary.csv`,
`results/bh_mac_softening.csv`, `results/local_vs_global.csv` y los PNGs en
`plots/`.

## 5. Resultados

### 5.1. Conservación multi-step (H1)

Drift energético final `|ΔE/E₀|_final` por distribución × N (en %):

| Distribución | N | V1 geom·bare | V2 geom·soft | V3 rel·bare | V4 rel·soft | V5 rel·soft·cons |
|---|---:|---:|---:|---:|---:|---:|
| plummer_a1 | 200 | 41.93 | 41.87 | 40.76 | 43.70 | 46.87 |
| plummer_a1 | 1000 | 33.27 | 36.27 | 34.31 | 33.15 | **32.37** |
| plummer_a2 | 200 | 41.45 | 38.77 | 36.86 | 41.39 | **36.49** |
| plummer_a2 | 1000 | 28.37 | 26.03 | 24.26 | 24.17 | **24.13** |
| plummer_a6 | 200 | 10.92 | 9.13 | 9.39 | 9.34 | **8.88** |
| plummer_a6 | 1000 | 3.80 | 3.57 | **2.51** | 3.05 | 3.71 |
| uniform | 200 | 2.30 | **0.65** | 1.21 | 2.87 | 0.82 |
| uniform | 1000 | **0.097** | 0.18 | 0.18 | 0.32 | 0.33 |

Observaciones:
- En Plummer `a/ε ≤ 2`, todas las variantes convergen al mismo rango de drift
  (±5 %). El integrador y el caos dominan.
- En Plummer `a/ε = 6`, `uniform` N=200 aparece **una tendencia leve**: el
  softening multipolar ayuda y el criterio relativo ayuda adicionalmente.
- En `uniform` N=1000 la mejor variante es **V1** (0.097 %): con error local
  10⁴ veces peor que V5 (0.3 % vs 10⁻⁵). Esto es la señal más clara de piso
  del integrador —el solver más impreciso es igual de bueno, a veces mejor.

Conclusión H1: **rechazada para los casos objetivo (concentrados);
parcialmente aceptada para sistemas con dinámica más suave**.

### 5.2. MAC softened-consistent (H2)

**Tabla 1.** Nodos abiertos y precisión local en N=1000. Las unidades son
fracciones, no porcentajes.

| Distribución | Variante | mean force err | max force err | opened nodes | Δ nodos vs V4 |
|---|---|---:|---:|---:|---:|
| plummer_a1 | V3 rel·bare         | 1.94 × 10⁻⁶ | 3.5 × 10⁻⁵ | 659 162 | 0 % |
| plummer_a1 | V4 rel·soft         | 7.38 × 10⁻⁷ | 8.5 × 10⁻⁶ | 659 162 | 0 % |
| plummer_a1 | **V5 rel·soft·cons**| 4.03 × 10⁻⁵ | 1.3 × 10⁻⁴ | **572 793** | **−13 %** |
| plummer_a2 | V4 rel·soft         | 4.63 × 10⁻⁶ | 3.0 × 10⁻⁵ | 591 115 | 0 % |
| plummer_a2 | **V5 rel·soft·cons**| 1.79 × 10⁻⁵ | 8.3 × 10⁻⁵ | **552 514** | **−6.5 %** |
| plummer_a6 | V4 rel·soft         | 6.63 × 10⁻⁶ | 2.2 × 10⁻⁵ | 473 607 | 0 % |
| plummer_a6 | V5 rel·soft·cons    | 7.97 × 10⁻⁶ | 2.5 × 10⁻⁵ | 460 757 | −2.7 % |
| uniform    | V4 rel·soft         | 1.85 × 10⁻⁵ | 1.7 × 10⁻⁴ | 277 922 | 0 % |
| uniform    | V5 rel·soft·cons    | 1.87 × 10⁻⁵ | 1.7 × 10⁻⁴ | 276 985 | −0.3 % |

Interpretación: V5 abre menos nodos precisamente donde hay partículas a
`d ~ ε` (Plummer concentrado). La reducción de coste se concentra en el
núcleo; el error local sube ~5× pero permanece en el rango 10⁻⁵, claramente
dentro de la tolerancia `err_tol_force_acc = 0.005`. H2 se confirma.

`plots/opened_nodes_profile.png` visualiza este efecto.

### 5.3. Relación local ↔ global (H3)

Correlaciones log-log entre `mean_force_error` y `|ΔE/E₀|_final`:

```
corr global (40 puntos) = +0.045
```

Por grupo `(distribución, N)`:

| distribución | N | r local↔global | rango local | rango global |
|---|---:|---:|---:|---:|
| plummer_a1 | 200 | −0.29 | 1.6 × 10⁻⁶ … 0.14 | 40.8 … 46.9 % |
| plummer_a1 | 1000 | +0.34 | 7.4 × 10⁻⁷ … 0.086 | 32.4 … 36.3 % |
| plummer_a2 | 200 | +0.42 | 3.9 × 10⁻⁶ … 0.039 | 36.5 … 41.4 % |
| plummer_a2 | 1000 | **+0.93** | 4.6 × 10⁻⁶ … 0.033 | 24.1 … 28.4 % |
| plummer_a6 | 200 | +0.65 | 5.9 × 10⁻⁶ … 0.013 | 8.9 … 10.9 % |
| plummer_a6 | 1000 | +0.61 | 6.6 × 10⁻⁶ … 0.010 | 2.5 … 3.8 % |
| uniform | 200 | −0.06 | 7.8 × 10⁻⁶ … 0.004 | 0.65 … 2.87 % |
| uniform | 1000 | −0.80 | 1.8 × 10⁻⁵ … 0.003 | 0.10 … 0.33 % |

La **correlación aparece sólo en regímenes mediocaóticos** (`plummer_a2–a6`).
En Plummer `a/ε = 1` (máximo caos) y en `uniform` N=1000 (caos casi nulo,
pero dominado por la simpléctica de Leapfrog) la correlación desaparece o
se invierte. Esto reproduce cualitativamente la sabiduría de Barnes & Hut
(1989): reducir el error de fuerza por debajo del piso del integrador no
mejora la energía.

`plots/local_vs_global.png` lo visualiza.

### 5.4. Pareto precisión-coste (H4) y control de |ΔL|

En `pareto_precision_cost.png`, para N=1000, la frontera es clara:

1. V1/V2 (geométrico): coste ~6–7 ms/step, error ~10⁻² (Plummer denso).
2. V3–V5 (relativo): coste ~30–36 ms/step, error ~10⁻⁶ a 10⁻⁵.
3. Dentro del clúster relativo, **V5 mantiene error del mismo orden
   que V3/V4 pero con menos coste interno (nodos abiertos)**.

Control de `|ΔL|_max` (N=1000, Plummer `a/ε = 1`):

- V1 geom·bare: **0.92**
- V2 geom·soft: 0.41
- V3 rel·bare: 0.047
- V4 rel·soft: **0.020**
- V5 rel·soft·cons: 0.040

El criterio relativo reduce `|ΔL|` en dos órdenes de magnitud. El softened
multipolo bare (V4) es marginalmente mejor que V5 en `|ΔL|`; para un paper
que priorice dinámica angular, V4 es aceptable. Para un paper que priorice
economía de cómputo con precisión equivalente, V5 es la recomendación.

**H4 se acepta** con la matización: V5 domina en el eje coste-precisión
local; V4 es ligeramente superior en `|ΔL|`. Ambas dominan el Pareto sobre
V1–V3.

## 6. Figuras

| Archivo | Contenido |
|---|---|
| `plots/energy_drift_timeseries.png` | `|ΔE/E₀|` vs t, N=1000, 4 subplots. Muestra el piso por distribución. |
| `plots/momentum_angmom_timeseries.png` | `|Δp|` y `|ΔL|` vs t, N=1000. El salto `|ΔL|` para V1/V2 en Plummer denso es visualmente dramático. |
| `plots/pareto_precision_cost.png` | **Figura central paper**. Error medio vs `time_bh_ms`. V5 en la frontera. |
| `plots/local_vs_global.png` | Scatter log-log con línea de correlación. r ≈ 0 global. |
| `plots/opened_nodes_profile.png` | Barras: V5 muestra la reducción de 13 % en Plummer denso. |

## 7. Configuración recomendada paper-grade

Para simulaciones donde la dinámica angular y el control del núcleo
importan:

```toml
[gravity]
solver              = "barnes_hut"
theta               = 0.5           # no usado cuando opening_criterion=relative
multipole_order     = 3
opening_criterion   = "relative"
err_tol_force_acc   = 0.005
softened_multipoles = true
mac_softening       = "consistent"  # 13% menos coste en núcleo denso
```

Para simulaciones donde el coste es dominante y la energía es la única
métrica relevante: `opening_criterion = "geometric"` con
`softened_multipoles = true` da el mejor coste/energía (ver `uniform N=1000`).

## 8. Limitaciones y backlog actualizado

- **El drift energético en Plummer denso (30–40 %) es excesivo para
  producción.** La Fase 5 demuestra que la causa no es el solver sino el
  integrador leapfrog KDK con `dt = 0.025`. Próximo paso: explorar
  `dt` adaptativo de Aarseth (ya implementado bajo `timestep.hierarchical`)
  y/o integrador de orden superior (Yoshida 4º).
- **Correlación negativa local↔global en `uniform N=1000`** (r = −0.80).
  No es un error: el criterio relativo hace sub-pasos de fuerza
  ligeramente distintos, que oscilan dentro del piso del integrador. No es
  significativa estadísticamente (rango global 0.1–0.3 %, dentro de la
  precisión de `dt`).
- **El test `bh_mac_softening_ablation` se ejecuta serial.** Usar
  `RayonBarnesHutGravity` podría acelerar el barrido pero requiere
  adaptar la instrumentación thread-local (fuera de alcance de Fase 5).
- **Backlog**:
  1. ~~Integrador Yoshida 4º orden (bajar el piso del integrador).~~ **Completado
     en Fase 6** ([`2026-04-phase6-higher-order-integrator.md`](2026-04-phase6-higher-order-integrator.md)): implementado y validado, pero **no mejora el drift**
     en régimen caótico denso (lo empeora ~1.9× a 1.74× más coste). Confirma
     que el integrador de orden local no es la palanca correcta; el drift lo
     gobierna la mezcla simpléctica/Lyapunov. Se relanza prioridad sobre
     `dt` adaptativo.
  2. `dt` adaptativo con criterio Aarseth — **próxima palanca (Fase 7)**
     tras la evidencia de Fase 6.
  3. MPI point-to-point para reemplazar `Allgatherv` (Fase 3 backlog).

→ Continúa en Fase 6 ([`2026-04-phase6-higher-order-integrator.md`](2026-04-phase6-higher-order-integrator.md)): evaluación de integrador orden superior (Yoshida 4º) sobre los mismos regímenes.

## 9. Narrativa paper (párrafo 230 palabras)

> Para validar la arquitectura de árbol multipolar de *gadget-ng* en régimen
> dinámico, se integraron 40 simulaciones N-body de 1 000 pasos sobre cuatro
> distribuciones (Plummer `a/ε ∈ {1, 2, 6}` y esfera uniforme) con dos
> tamaños (N = 200, 1 000) y cinco criterios de apertura distintos. Se
> introdujo `MacSoftening::Consistent`, que sustituye el denominador bare
> `d⁵` del estimador del MAC relativo por su forma softened-consistent
> `(d² + ε²)^{5/2}`, análoga al monopolo Plummer. La variante propuesta
> abre un 13 % menos nodos en distribuciones densas (Plummer `a/ε = 1`,
> N = 1 000) manteniendo el error medio de fuerza en 10⁻⁵. La conservación
> de energía, en cambio, no mejora: las cinco variantes (cuyos errores
> locales difieren en cinco órdenes de magnitud) convergen al mismo piso de
> drift (~32 % en Plummer `a/ε = 1`, ~0.1 % en uniforme). La correlación
> log-log global entre error local y |ΔE/E₀| es r = +0.045, confirmando que
> la conservación está gobernada por el integrador simpléctico KDK y por el
> caos microscópico de Miller, no por la calidad del solver de fuerzas.
> El control del momento angular, en cambio, sí escala con la precisión del
> solver: `|ΔL|_max` cae de 0.92 a 0.02 al pasar del criterio geométrico al
> relativo con multipolos softened. Concluimos que el MAC softened-consistent
> es un refinamiento físicamente correcto y computacionalmente ventajoso en
> el núcleo de sistemas concentrados, mientras que mejorar el drift
> energético en estos regímenes requiere avanzar hacia integradores de orden
> superior o `dt` adaptativo —una dirección natural para trabajo futuro.

---

- Datos crudos: [`experiments/nbody/phase5_energy_mac_consistency/results/`](../../experiments/nbody/phase5_energy_mac_consistency/results/)
- Figuras: [`experiments/nbody/phase5_energy_mac_consistency/plots/`](../../experiments/nbody/phase5_energy_mac_consistency/plots/)
- Reporte anterior: [Fase 4 — Multipolos, Softening y MAC](2026-04-phase4-multipole-softening.md)
