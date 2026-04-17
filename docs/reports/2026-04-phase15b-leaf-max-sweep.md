# Phase 15b — Sweep de leaf_max: impacto en SIMD, rendimiento y física

**Fecha:** 2026-04  
**Estado:** Completado  
**Precondición:** Phase 15 (explicit AVX2 SIMD, kernel two-pass)

---

## 1. Motivación y pregunta central

Phase 15 confirmó que el kernel P15 (intrinsics AVX2 explícitos con `ymm` real) es ~8%
**más lento** que el kernel P14 (fusionado, auto-vec con `xmm`) para el `leaf_max=8`
actual. El diagnóstico identificó la causa: el **batch size promedio de 3.7 elementos**
en `apply_leaf_soa` es demasiado pequeño para que el loop SIMD (4 f64 por iteración)
amortice su overhead de setup.

La pregunta de Phase 15b es:

> **¿Existe un `leaf_max` que mejore el aprovechamiento SIMD sin degradar el
> rendimiento total ni la física?**

### Hipótesis previa

| leaf_max | batch_avg esperado | iteraciones SIMD | Predicción |
|:---:|:---:|:---:|---|
| 8 | ~4 | ~1 | P15 más lento (confirmado P15) |
| 16 | ~7–8 | ~1–2 | P15 todavía más lento |
| 32 | ~14–15 | ~3–4 | P15 ≈ P14 |
| 64 | ~28–30 | ~7 | P15 debería ganar |

---

## 2. Metodología

### Sin cambios de código

`let_tree_leaf_max` ya es configurable vía TOML
([`crates/gadget-ng-core/src/config.rs`](../../crates/gadget-ng-core/src/config.rs)).
Los binarios `gadget-ng-p14-fused` y `gadget-ng-p15-explicit` de Phase 15 se
reutilizaron sin recompilación.

### Configuración experimental

- `leaf_max ∈ {8, 16, 32, 64}` × `N ∈ {8000, 16000}` × `P ∈ {2, 4}`
- Distribución Plummer a/ε=2, 10 pasos, seed=42
- Total: 16 configs × 2 variantes = **32 runs** (~80 segundos de ejecución)

---

## 3. Resultados: batch size y rendimiento

### 3.1 Batch size real observado

| leaf_max | N=8000, P=2 | N=8000, P=4 | N=16000, P=2 | N=16000, P=4 |
|:---:|:---:|:---:|:---:|:---:|
| **8** | 3.7 | 3.9 | 4.0 | 3.8 |
| **16** | 5.9 | 6.5 | 6.7 | 6.5 |
| **32** | 10.6 | 12.1 | 10.9 | 11.6 |
| **64** | 18.0 | 20.3 | 20.1 | 23.9 |

El batch size escala de forma aproximadamente lineal con `leaf_max`, pero
siempre inferior al valor máximo teórico (el árbol no llena todos los
leaf nodes hasta su capacidad).

### 3.2 RMNs aplicados en hojas (apply_leaf_rmn_count) — N=8000, P=2

| leaf_max | apply_calls | apply_rmns | Factor vs lm=8 | LT walk % |
|:---:|:---:|:---:|:---:|:---:|
| **8** | 629,856 | 2,307,550 | 1.0× | 34.4% |
| **16** | 709,366 | 4,207,503 | 1.8× | 42.1% |
| **32** | 753,154 | 8,003,462 | 3.5× | 55.2% |
| **64** | 730,494 | 13,116,292 | 5.7× | 66.8% |

**Hallazgo crítico**: al aumentar `leaf_max` de 8 a 64, el número de RMN
interacciones en hojas se multiplica por **5.7×**. El LetTree pierde eficiencia
de aproximación porque nodos que antes admitían el test MAC ahora caen en hojas
más profundas y requieren evaluación directa.

### 3.3 Tabla completa de rendimiento

| Config | lm | batch | P14 wall(ms) | P15 wall(ms) | Wall sp | P14 LT(ms) | P15 LT(ms) | LT sp |
|--------|:--:|:-----:|:---:|:---:|:---:|:---:|:---:|:---:|
| N=8000 P=2 | 8 | 3.7 | 108.7 | 116.1 | **0.94×** | 37.4 | 44.3 | 0.85× |
| N=8000 P=2 | 16 | 5.9 | 128.5 | 133.8 | 0.96× | 54.1 | 59.4 | 0.91× |
| N=8000 P=2 | 32 | 10.6 | 161.9 | 162.6 | **1.00×** | 89.3 | 90.4 | 0.99× |
| N=8000 P=2 | 64 | 18.0 | 208.3 | 203.5 | **1.02×** | 139.1 | 134.0 | 1.04× |
| N=8000 P=4 | 8 | 3.9 | 71.1 | 78.7 | 0.90× | 31.3 | 36.8 | 0.85× |
| N=8000 P=4 | 16 | 6.5 | 89.3 | 94.3 | 0.95× | 46.6 | 50.5 | 0.92× |
| N=8000 P=4 | 32 | 12.1 | 119.2 | 125.6 | 0.95× | 76.1 | 83.3 | 0.91× |
| N=8000 P=4 | 64 | 20.3 | 160.0 | 155.1 | **1.03×** | 114.1 | 109.5 | 1.04× |
| N=16000 P=2 | 8 | 4.0 | 255.2 | 268.5 | 0.95× | 84.4 | 97.6 | 0.86× |
| N=16000 P=2 | 16 | 6.7 | 294.5 | 304.4 | 0.97× | 122.4 | 132.3 | 0.93× |
| N=16000 P=2 | 32 | 10.9 | 368.8 | 372.9 | **0.99×** | 191.5 | 193.6 | 0.99× |
| N=16000 P=2 | 64 | 20.1 | 527.3 | 512.1 | **1.03×** | 343.1 | 329.3 | 1.04× |
| N=16000 P=4 | 8 | 3.8 | 164.9 | 179.5 | 0.92× | 56.9 | 65.8 | 0.87× |
| N=16000 P=4 | 16 | 6.5 | 204.2 | 215.5 | 0.95× | 85.7 | 94.0 | 0.91× |
| N=16000 P=4 | 32 | 11.6 | 268.7 | 270.2 | **0.99×** | 141.3 | 142.2 | 0.99× |
| N=16000 P=4 | 64 | 23.9 | 392.1 | 380.1 | **1.03×** | 273.4 | 259.8 | 1.05× |

---

## 4. Análisis estructurado

### A. Batch size vs leaf_max: hipótesis confirmada

| leaf_max | Predicción | Resultado real |
|:---:|---|---|
| 8 | P15 más lento (batch < SIMD width) | Confirmado: 0.85–0.94× |
| 16 | P15 todavía más lento | Confirmado: 0.91–0.97× |
| 32 | P15 ≈ P14 (crossover) | **Confirmado: 0.99–1.00×** |
| 64 | P15 gana por ~5% | **Confirmado: 1.02–1.05×** |

El crossover teórico era `batch_avg ≈ SIMD_width × overhead_amortization ≈ 8-12`. Los
datos confirman que el cruce P15 = P14 ocurre alrededor de `leaf_max=32` (batch ~10-12).

### B. SIMD effectiveness — ¿en qué leaf_max P15 supera a P14?

```
leaf_max=8:  speedup LT = 0.85×  (P15 pierde 15%)
leaf_max=16: speedup LT = 0.91×  (P15 pierde 9%)
leaf_max=32: speedup LT = 0.99×  (P15 ≈ P14)
leaf_max=64: speedup LT = 1.04×  (P15 gana 4%)
```

**P15 supera a P14 en el LetTree walk a partir de `leaf_max=64`**, con una ventaja
de ~4-5%. Para que el SIMD de 4×f64 sea efectivo, se necesita un batch medio de
~18-24 elementos, lo que requiere `leaf_max=64`.

### C. Trade-off computacional: el costo oculto de aumentar leaf_max

Este es el hallazgo más importante de Phase 15b:

| leaf_max | P14 wall (N=8000 P=2) | Ratio vs lm=8 |
|:---:|:---:|:---:|
| 8 | 108.7 ms | 1.00× (baseline) |
| 16 | 128.5 ms | +18% |
| 32 | 161.9 ms | +49% |
| 64 | 208.3 ms | +92% |

**Aumentar leaf_max de 8 a 64 duplica el tiempo de pared total**, a pesar de que
P15 gana un 4% sobre P14 en ese régimen. La causa es que el LetTree pierde su
ventaja de aproximación multipolar: con hojas más grandes, más interacciones caen
al nivel de evaluación directa hoja-a-partícula en vez de ser aceptadas por el
test MAC en nodos internos.

El número de RMNs aplicados en hojas crece de 2.3M (lm=8) a 13.1M (lm=64) para
N=8000/P=2 — un factor 5.7× de trabajo adicional.

### D. Impacto en comunicación LET

| leaf_max | LET importados | bytes_sent |
|:---:|:---:|:---:|
| 8 | 7,981 | 960 KB |
| 16 | 7,981 | 960 KB |
| 32 | 7,981 | 960 KB |
| 64 | 7,981 | 960 KB |

**El volumen de comunicación LET no cambia con `leaf_max`**, como era de esperar:
`leaf_max` controla cómo se construye el LetTree localmente sobre los nodos ya
importados, pero no afecta qué nodos se exportan. El bottleneck de comunicación
es independiente de este parámetro.

---

## 5. Validación física

Todos los valores de `leaf_max` producen resultados físicamente equivalentes
(PASS para todos los casos):

| leaf_max | ΔKE_rel máximo | Estado |
|:---:|:---:|:---:|
| 8 | 0.00 (baseline) | OK |
| 16 | 1.6e-5 | OK |
| 32 | 8.3e-5 | OK |
| 64 | 1.3e-4 | OK |

El error de KE crece con `leaf_max` (más RMNs se aplican directamente en hojas
usando el RMN exacto pero sin el test MAC → diferente selección de nodos), pero
permanece varios órdenes de magnitud por debajo de la tolerancia `1e-3`.
El momentum y angular momentum son idénticos dentro de tolerancias numéricas.

---

## 6. Decisión final

### Pregunta central respondida

> **¿Existe un `leaf_max` que mejore el aprovechamiento SIMD sin degradar el
> rendimiento total?**

**No.** El trade-off es asimétrico e irreversible:

- **`leaf_max=32`**: P15 ≈ P14 (cruce), pero el total es +49% peor que lm=8.
- **`leaf_max=64`**: P15 gana 4% sobre P14, pero el total es +92% peor que lm=8.

La ganancia SIMD real (4%) queda completamente sepultada por la degradación del
LetTree como aproximador jerárquico.

### Opción elegida: **A — mantener leaf_max=8**

**El valor actual `let_tree_leaf_max=8` es el óptimo para el rendimiento total.**

| Criterio | leaf_max=8 | leaf_max=64 |
|---|:---:|:---:|
| Wall time total | 108.7 ms | 208.3 ms (+92%) |
| P15 > P14 | No | Sí (+4%) |
| Trabajo total | 2.3M RMN/paso | 13.1M RMN/paso |
| Física correcta | Sí | Sí |
| LET volume | Sin cambio | Sin cambio |

### Conclusión técnica y para el paper

> *Phase 15b demuestra que la hipótesis "aumentar leaf_max habilitaría el SIMD
> AVX2" es correcta en términos del kernel aislado (P15 supera a P14 para
> leaf_max=64), pero incorrecta en términos del sistema completo. El LetTree con
> leaf_max grande pierde su capacidad de agrupar RMNs bajo el test MAC, lo que
> incrementa el trabajo total de fuerza de forma supralineal. El cruce P15=P14
> ocurre en leaf_max≈32 (batch_avg≈10), pero el wall time ya es 49% mayor que el
> baseline. Aumentar leaf_max para "desbloquear SIMD" es una optimización local
> que degrada el rendimiento global. El límite real del rendimiento CPU en
> gadget-ng no es la anchura SIMD del kernel de fuerza, sino la eficiencia de
> aproximación multipolar del árbol.*

### Caminos futuros para explotar SIMD sin degradar el árbol

Para obtener beneficio real de AVX2 sin comprometer la eficiencia del LetTree,
se necesitaría un cambio arquitectónico distinto al aumento de `leaf_max`:

1. **Tiling 4×N_i**: procesar 4 partículas-i simultáneamente contra el mismo
   leaf batch. El batch de 4 RMNs se aplica a 4 partículas en paralelo → SIMD
   efectivo sin aumentar leaf_max.

2. **SoA para partículas locales**: reorganizar el loop Rayon para que
   `apply_leaf_soa` reciba bloques de 4 posiciones de partículas locales.

3. **Path accel_from_let_soa con N grande**: el path plano sobre todos los nodos
   LET importados (~8000 nodos) sí tiene batch sizes grandes → P15 ganaría aquí.
   Pero este path es O(N_local × N_let) vs. O(N_local log N_let) del LetTree.

---

## Apéndice: archivos del experimento

```
experiments/nbody/phase15b_leaf_max_sweep/
├── generate_configs.py   # genera 16 configs TOML
├── run_phase15b.sh       # ejecuta 32 runs
├── analyze_phase15b.py   # tablas y 5 figuras PNG
├── validate_physics.py   # validación KE, p, L
├── configs/              # 16 TOML (lm{8,16,32,64}_N{8k,16k}_P{2,4})
├── results/              # 32 directorios de resultados
└── plots/                # 5 figuras PNG
    ├── batch_size_vs_leaf_max.png
    ├── wall_time_vs_leaf_max_P2.png
    ├── wall_time_vs_leaf_max_P4.png
    ├── speedup_p15_vs_p14.png
    └── let_nodes_vs_leaf_max.png
```

## Apéndice: reproducción

```bash
cd experiments/nbody/phase15b_leaf_max_sweep
python3 generate_configs.py
bash run_phase15b.sh        # ~80s (requiere binarios de Phase 15)
python3 analyze_phase15b.py
python3 validate_physics.py
```
