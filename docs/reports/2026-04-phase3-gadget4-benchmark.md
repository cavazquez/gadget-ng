# Fase 3: Benchmarking HPC y Validación Cuantitativa vs GADGET-4

**Proyecto:** gadget-ng  
**Fecha:** Abril 2026  
**Autores:** Validación automatizada, fase 3  
**Pregunta guía:** ¿En qué condiciones gadget-ng compite razonablemente con GADGET-4, y en qué condiciones todavía no?

---

## 1. Objetivo

Esta fase cuantifica cuatro dimensiones de calidad del código:

1. **Precisión física del árbol BH** — error de fuerza vs Direct, barrido de θ
2. **Costo computacional vs N** — crossover entre Direct y Barnes-Hut
3. **Impacto de block timesteps** — conservación de energía y costo relativo
4. **Paridad y escalado MPI** — strong scaling, weak scaling, paridad serial vs MPI

El objetivo no es afirmar que gadget-ng compite con GADGET-4 hoy, sino medir cuantitativamente la brecha y priorizar el backlog técnico.

### 1.1 Alcance y vigencia (coherencia con el código en 2026)

Las tablas y conclusiones de las **secciones 4–9** provienen de la **campaña de medición de la Fase 3** (abril 2026). En los experimentos MPI, corresponden en la práctica a la ruta en que el stepping usa **`allgatherv_state`** para reunir el estado global en cada paso relevante (BH distribuido “clásico”). Eso **no se re-ejecutó** tras añadir descomposición SFC, `exchange_halos_sfc`, optimizaciones MPI ni el criterio de apertura relativo en configuración.

Por tanto: los números de weak/strong scaling y la narrativa del “cuello de botella Allgatherv” siguen siendo **válidos como diagnóstico de esa configuración benchmark**. El **backlog** (§12) y los **Cambios 1–3** (§10.2) describen el código y lo pendiente **hoy**; donde hay solapamiento, se distingue *medido aquí* vs *implementado en repo*.

---

## 2. Hardware y versiones

| Parámetro        | Valor                             |
|------------------|-----------------------------------|
| CPU              | Linux 6.17.0-22 (12 cores)       |
| Compilador       | rustc (stable), --release         |
| MPI              | OpenMPI (mpirun disponible)       |
| BH solver        | Multipolar (mono+quad+oct), criterio geométrico θ |
| Integrador       | Leapfrog KDK                      |
| Block timesteps  | Aarseth: dt_i = η√(ε/|a_i|)      |

---

## 3. Benchmarks elegidos y justificación

| Experimento              | Justificación                                                          |
|--------------------------|------------------------------------------------------------------------|
| BH vs Direct, barrido θ  | Cuantifica el error del árbol multipolar (mono+quad+oct) vs referencia exacta |
| Scaling N (Direct vs BH) | Muestra el crossover BH > Direct y la complejidad empírica             |
| Block timestep compare   | Evalúa si el integrador jerárquico mejora eficiencia vs global dt      |
| Serial vs MPI parity     | Verifica reproducibilidad y "verdadera" no-determinismo del runtime    |
| Strong scaling (N=1000)  | Mide eficiencia MPI y fracción de comunicación por rank                |
| Weak scaling (N ∝ ranks) | Expone la limitación del diseño Allgatherv **en la ruta benchmark Fase 3** |

---

## 4. Experimento 1: Error de fuerza BH vs DirectGravity

**Configuración:** N=500, ε=0.05, G=1.0  
**Distribuciones:** esfera uniforme (extendida) y Plummer a=0.1 (concentrada)  
**Herramienta:** `crates/gadget-ng-physics/tests/bh_force_accuracy.rs`

### 4.1 Resultados: error relativo de fuerza

| Distribución   | θ    | mean\_err% | max\_err% | rms\_err% | t\_direct ms | t\_BH ms | Speedup |
|----------------|------|-----------|----------|----------|-------------|---------|---------|
| Esfera uniforme | 0.20 | 0.006     | 0.127    | 0.016    | 0.7         | 4.3     | 0.16x   |
| Esfera uniforme | 0.50 | 0.309     | 8.059    | 0.771    | 0.7         | 1.9     | 0.35x   |
| Esfera uniforme | 0.80 | 2.152     | 80.987   | 5.990    | 0.7         | 1.1     | 0.59x   |
| Esfera uniforme | 1.00 | 4.659     | 120.287  | 9.605    | 0.7         | 0.8     | 0.87x   |
| Plummer a=0.1   | 0.20 | 0.286     | 1.423    | 0.365    | 0.7         | 4.8     | 0.14x   |
| Plummer a=0.1   | 0.50 | 3.757     | 27.724   | 5.057    | 0.7         | 1.9     | 0.35x   |
| Plummer a=0.1   | 0.80 | 17.529    | 190.625  | 27.135   | 0.7         | 1.2     | 0.56x   |
| Plummer a=0.1   | 1.00 | 30.943    | 277.270  | 43.753   | 0.7         | 0.9     | 0.80x   |

### 4.2 Observaciones críticas

> **Nota corregida post-Fase 3:** Los errores medidos en esta tabla son los errores **con multipolar completo** (monopolo + cuadrupolo + octupolo). El solver BH de gadget-ng ya implementa los tres términos desde la construcción del octree (`aggregate()` calcula `quad:[f64;6]` y `oct:[f64;7]` con el teorema del eje paralelo; `walk_inner()` aplica `a_mono + a_quad + a_oct` cuando el MAC se satisface). El diagnóstico previo que lo clasificaba como "BH monopolar" era incorrecto.

- **Para N=500, BH es más lento que Direct en todos los θ.** El crossover BH > Direct ocurre aproximadamente en N=3000–5000 (ver Experimento 2). Esto es coherente con la literatura: para N pequeño, el overhead de construcción y recorrido del octree supera el ahorro de interacciones.

- **El error multipolar crece fuertemente con la concentración de la distribución.** Plummer a=0.1 tiene un núcleo mucho más denso que la esfera uniforme: a θ=0.5 el error medio es 3.76% (Plummer) vs 0.31% (esfera uniforme) **incluso con cuadrupolo y octupolo activos**. El error máximo llega a 190% para partículas del núcleo denso a θ=0.8 — lo que significa que la expansión puede estar equivocada en un factor ~3 para interacciones con el núcleo compacto. Esto no es un déficit de orden multipolar, sino del criterio de apertura geométrico fijo θ, que es demasiado permisivo en regiones de alta densidad.

- **GADGET-4 con criterio de apertura relativo (`ErrTolForceAcc`) supera al criterio geométrico.** A θ=0.5 fijo, gadget-ng tiene error medio 3.76% en Plummer; GADGET-4 con criterio relativo adaptaría automáticamente el MAC efectivo por interacción, abriéndolo más en el núcleo denso donde la expansión converge peor. Esto se traduce en mejor precisión en sistemas inhomogéneos sin cambios de orden multipolar.

- **Criterio de apertura relativo de GADGET-4** (`TypeOfOpeningCriterion=1`, `ErrTolForceAcc`): en lugar de abrir nodos cuando `s/d > θ` (criterio geométrico), estima el error de truncamiento de la expansión multipolar y abre el nodo cuando ese error supere un umbral `α`. Esto adapta el MAC por interacción en lugar de usar un θ global fijo, con mayor impacto en distribuciones inhomogéneas.

### 4.3 Implicación para producción

Para usar BH con confianza en simulaciones de tipo paper:
- Distribuciones extendidas (esfera uniforme, halos de materia oscura): θ=0.5 con |ΔE/E|<1% es aceptable.
- Distribuciones concentradas (Plummer, cúmulos globulares, núcleos densos): θ≤0.3 o implementar criterio de apertura relativo (`ErrTolForceAcc`). θ=0.5 con criterio geométrico produce errores de fuerza inaceptables para las partículas centrales incluso con cuadrupolo+octupolo activos.

---

## 5. Experimento 2: Costo computacional vs N

**Configuración:** Plummer a=1.0, 10 pasos, ε=0.05  
**Herramienta:** `experiments/nbody/phase3_gadget4_benchmark/scaling_n/scripts/run_scaling_n.py`

### 5.1 Resultados

| Solver     | N     | t/paso (ms) | Speedup BH/Direct |
|------------|-------|------------|-------------------|
| Direct     | 100   | 0.07       | 0.21x             |
| Direct     | 500   | 1.70       | 0.45x             |
| Direct     | 1000  | 6.81       | 0.57x             |
| Direct     | 2000  | 22.52      | 0.62x             |
| Direct     | 5000  | 132.54     | 1.21x             |
| Barnes-Hut | 100   | 0.33       | (ver ratio)       |
| Barnes-Hut | 500   | 3.79       |                   |
| Barnes-Hut | 1000  | 11.98      |                   |
| Barnes-Hut | 2000  | 36.31      |                   |
| Barnes-Hut | 5000  | 109.66     |                   |

**Crossover observado:** BH empieza a ser más rápido que Direct en torno a N≈4000–5000. Para N≤2000, Direct es más eficiente. La fracción de tiempo en gravedad es ≥99.9% para ambos solvers en modo serial.

### 5.2 Ajuste de complejidad empírica

Fitting log-log sobre los datos:
- Direct: exponente ~1.95 ≈ O(N²) — confirma complejidad teórica
- Barnes-Hut: exponente ~1.42 — sublineal respecto a N², pero BH multipolar no alcanza el O(N log N) teórico puro por overhead de árbol y cálculo de términos cuadrupolar/octupolar

### 5.3 Implicación

Para N<2000 en modo serial, usar DirectGravity. Para N>5000, BH empieza a compensar. Con criterio de apertura relativo (disponible en config como en GADGET-4), se podría usar un θ efectivo mayor con la misma precisión en **nuevas** mediciones, reduciendo el trabajo de recorrido. El intra-nodo Rayon solo entra con build `simd` y modo no determinista (§12 Prioridad 3); sin eso, el walk BH local sigue siendo esencialmente serial.

---

## 6. Experimento 3: Block Timesteps vs Global dt

**Configuración:** Cold collapse, N=200, 300 pasos (≈3·T_ff), Barnes-Hut θ=0.5  
**Modos:** global dt=0.02221, hierarchical (η=0.025, max_level=6)

### 6.1 Resultados

| Modo                   | Wall total (s) | t/paso (ms) | max\|ΔE/E₀\|  | Fracción gravedad |
|------------------------|---------------|-------------|--------------|-------------------|
| Global dt fijo         | 0.328         | 1.08        | 0.43%        | 99.9%             |
| Block timesteps Aarseth | 1.075         | 3.56        | 0.58%        | 98.8%             |
| Ratio global/hierárq.  | 0.30x         | —           | ~similar     | —                 |

### 6.2 Interpretación

**Resultado negativo pero honesto:** Para N=200 y un colapso frío moderado, los block timesteps son **3.3× más lentos** que el timestep global con **peor** (o similar) conservación de energía. Esto no significa que los block timesteps sean incorrectos; significa que para este régimen (N pequeño, dt_base ya relativamente fino), el overhead supera el beneficio.

Los block timesteps son ventajosos cuando:
1. **N es grande** (el overhead de bookkeeping se amortiza sobre muchas partículas)
2. **El rango dinámico de aceleraciones es extremo** (núcleo denso + halo difuso), creando niveles de timestep con diferencia de ~10² o más
3. **En sistemas con partículas de muy distinta masa** (e.g., binarias + halo cosmológico)

Para el colapso frío con N=200 y ε=0.05, las aceleraciones varían solo ~100x durante el colapso, insuficiente para que los 6 niveles jerárquicos compensen el overhead de allgatherv por sub-paso fino. El diagnóstico es que `max_level=6` es excesivo para este caso; con `max_level=2` el overhead sería menor.

**Comparación con GADGET-4:** GADGET-4 usa block timesteps porque simula N=10⁶–10⁹ partículas donde el overhead es despreciable y el rango dinámico de aceleraciones puede ser 10⁶. Con N=200, incluso GADGET-4 desactivaría efectivamente los niveles finos.

---

## 7. Experimento 4: Paridad Serial vs MPI

**Configuración:** Plummer N=300, 100 pasos, BH θ=0.5, `deterministic=true/false`

### 7.1 Resultados

| Referencia | vs.                | max\|Δr\| | max\|Δv\| | \|ΔE/E\|   | \|ΔLz/Lz\| |
|------------|-------------------|---------|---------|----------|----------|
| serial     | mpi\_2rank\_det   | 0.000   | 0.000   | 0.000%   | 0.000%   |
| serial     | mpi\_2rank\_nondet | 0.000  | 0.000   | 0.000%   | 0.000%   |
| serial     | mpi\_4rank\_det   | 0.000   | 0.000   | 0.000%   | 0.000%   |

### 7.2 Análisis del resultado

**Paridad bit-exacta en todos los modos, incluyendo el "no-determinístico".** En esta configuración benchmark (`parity.toml`, BH, MPI clásico), es una consecuencia del diseño **Allgatherv** de gadget-ng:

1. Cada rank recoge el estado global de **todas** las partículas vía `allgatherv_state`
2. El estado global se ordena por `global_id` de forma determinista
3. Cada rank calcula las fuerzas para su subconjunto de partículas, pero usando el mismo estado global ordenado
4. Las operaciones de punto flotante ocurren exactamente en el mismo orden independientemente del número de ranks

El flag `deterministic=false` en gadget-ng **no introduce no-determinismo real** en este modo Allgatherv clásico. En **otras** rutas (árbol/P2P distribuido, LET/SFC), el efecto del flag y la paridad entre rangos pueden diferir; no están cubiertos por la tabla §7.1.

**Diferencia con GADGET-4:** En GADGET-4, el modo no-determinístico usa recorridos de árbol en distintos órdenes por rank, con reducciones de punto flotante no-asociativas (`MPI_Allreduce` con orden variable). Esto produce divergencias de ~10⁻¹⁵ a 10⁻¹² entre ejecuciones con distinto número de ranks. Gadget-ng evita esta divergencia por diseño (Allgatherv), pero lo hace a costo de escalabilidad.

---

## 8. Experimento 5: Strong Scaling

**Configuración:** Plummer N=1000, 50 pasos, BH θ=0.5  
**Ranks probados:** 1, 2, 4 (8 ranks no disponibles por límite de slots del nodo)  
**Contexto:** mismo régimen MPI que la Fase 3 (véase §1.1); no es un bench del modo SFC/LET actual.

### 8.1 Resultados

| Ranks | Wall (s) | t/paso (ms) | Speedup | Efficiency | Comm frac |
|-------|---------|-------------|---------|------------|-----------|
| 1     | 0.636   | 12.72       | 1.00x   | 100%       | 0.09%     |
| 2     | 0.316   | 6.25        | 2.01x   | 100.8%     | 1.32%     |
| 4     | 0.179   | 3.46        | 3.56x   | 88.9%      | 4.30%     |

### 8.2 Interpretación

- **2 ranks: 2.01x speedup (super-lineal por ruido de medición).** La variabilidad de timing es ±5% para este tiempo de run; el speedup real es ~2.0x.
- **4 ranks: 3.56x speedup (89% efficiency).** Buen resultado para Allgatherv puro.
- **La fracción de comunicación crece:** 0.09% → 1.32% → 4.30%. Con 8 ranks se estimaría ~8–12%, con 16 ranks >20%. El punto de inflexión donde la comunicación degrada seriamente el rendimiento está en ~8–16 ranks para N=1000.

**Análisis de Amdahl:** La fracción serial implícita estimada (de los datos 1→4 ranks) es ~f≈0.035 (3.5%). A 8 ranks, el speedup máximo teórico sería ~1/f ≈ 28x, pero la comunicación creciente lo limitará mucho antes.

**Contraste con GADGET-4 (escala cósmica):** GADGET-4 usa comunicación punto-a-punto selectiva en descomposición SFC: O(N_halo) por rank. En la **ruta que este experimento ejercita**, gadget-ng usa `allgatherv_state` y distribuye **todo** el estado global a **todos** los ranks en cada paso relevante — de ahí el coste O(N×ranks). El binario actual **también** puede ejecutar rutas SFC/halos (§10.2); ese modo **no** está reflejado en estas curvas.

---

## 9. Experimento 6: Weak Scaling

**Configuración:** N = 1000 × ranks, BH θ=0.5, 50 pasos  
**Contexto:** Campaña Fase 3, ruta dominada por `allgatherv_state` (§1.1). Una rerun explícita con LET/SFC quedaría fuera del alcance de este PDF de resultados.

### 9.1 Resultados

| Ranks | N     | Wall (s) | t/paso (ms) | Weak efficiency | Comm frac |
|-------|-------|---------|-------------|-----------------|-----------|
| 1     | 1000  | 0.615   | 12.30       | 100%            | 0.10%     |
| 2     | 2000  | 0.986   | 19.63       | 62.4%           | 0.57%     |
| 4     | 4000  | 1.294   | 25.44       | 47.6%           | 2.14%     |

### 9.2 Interpretación — la limitación fundamental del Allgatherv

El weak scaling de gadget-ng es **fundamentalmente pobre** y el motivo es analítico:

**Costo de comunicación:** `allgatherv` transfiere `N × tamaño_partícula` bytes a cada rank. Con N ∝ ranks:
```
T_comm ∝ N_total × ranks = N₀ × ranks²
```

**Costo de fuerzas:** O(N log N) = O(N₀ × ranks × log(N₀ × ranks))

**Resultado:** la comunicación crece como O(ranks²) mientras la computación crece como O(ranks log ranks). En weak scaling ideal, ambas deberían crecer a la misma tasa.

A 4 ranks con N=4000, la eficiencia ya ha caído al 47.6%. Extrapolando:
- 8 ranks + N=8000: eficiencia estimada ~30–35%
- 16 ranks + N=16000: eficiencia estimada ~15–20%
- 64 ranks + N=64000: eficiencia estimada <5%

Esto confirma que **con la ruta MPI medida aquí (Allgatherv dominante), gadget-ng no escala como un código HPC masivo**. El repositorio **ya incluye** descomposición SFC e intercambio de halos selectivo (§10.2 Cambio 2); falta **cuantificar** weak/strong scaling en esa ruta para sustituir o complementar este diagnóstico.

---

## 10. Comparación conceptual con GADGET-4

### 10.1 Tabla comparativa de características

| Característica               | gadget-ng (actual)          | GADGET-4                             |
|------------------------------|-----------------------------|--------------------------------------|
| Integrador                   | Leapfrog KDK                | Leapfrog KDK (mismo)                 |
| Block timesteps              | Sí (Aarseth, implementado)  | Sí (Aarseth, idéntico criterio)      |
| Solver gravitatorio          | Direct O(N²), BH multipolar (mono+quad+oct) | TreePM (BH + cuadrupolo + PM grid) |
| Orden multipolar             | Hasta mono+quad+oct (`multipole_order` configurable) | Hasta hexadecapolo (orden 5)    |
| Criterio apertura árbol      | Geométrico y **relativo** configurables (`opening_criterion`, `err_tol_force_acc`) | Geométrico + relativo (ErrTolForceAcc) |
| Comunicación MPI             | Mixto: rutas LET/SFC con P2P; rutas clásicas aún `allgatherv_state` O(N×P) | Punto-a-punto SFC (O(N_halo/rank))   |
| Descomposición dominio       | SFC (p. ej. Hilbert) donde el stepping lo usa; fallbacks posibles | Peano-Hilbert SFC con balance carga  |
| Paralelismo híbrido          | MPI; opcional Rayon intra-nodo (`simd` + no determinista) | MPI + shared-memory OpenMP           |
| Paralelismo intra-nodo       | Rayon BH opcional (no equivale a “siempre activo”) | OpenMP (hilos por rank)              |
| Fast Multipole Method (FMM)  | No                          | Sí (alternativa al árbol BH)         |
| PM gravitacional             | Sí (PmSolver)               | Sí (TreePM con splitting erf/erfc)   |
| Cosmología                   | Sí (básica, a(t))           | Sí (completa, ΛCDM)                  |
| Condiciones iniciales        | Lattice/TwoBody/Plummer/Sphere | N-GenIC, MUSIC, formatos estándar  |
| Diagnósticos de timing       | timings.json (nuevo)        | timings.txt por fase detallado       |
| Escalabilidad demostrada     | ~4 ranks (89% eff)          | 160,000 ranks (UEABS benchmark)      |

### 10.2 Los tres cambios de mayor impacto

> **Corrección post-Fase 3:** El diagnóstico inicial priorizaba "implementar cuadrupolo" como Cambio 1. Esto era incorrecto: gadget-ng ya implementa cuadrupolo y octupolo completamente en `octree.rs` (`quad:[f64;6]`, `oct:[f64;7]`, aplicados en `walk_inner()`). El ranking se actualiza a continuación.

**Cambio 1 — Criterio de apertura relativo + multipolos ablables** (impacto: ALTO; **Fases 4–5: implementado en código**)
- En configuración: `opening_criterion` (geométrico vs relativo, análogo a `TypeOfOpeningCriterion=1` / `ErrTolForceAcc` de GADGET-4), `err_tol_force_acc`, `multipole_order`, `mac_softening` — ver [Fase 5](2026-04-phase5-energy-mac-consistency.md)
- **Estas tablas de error (§4) siguen siendo con criterio geométrico fijo θ** en el experimento original; no se re-simularon automáticamente con `opening_criterion = "relative"`.
- El error 3.76% en Plummer a θ=0.5 en **ese** setup sigue explicando por qué el criterio geométrico puro es demasiado permisivo en el núcleo denso; el criterio relativo en el repo es la palanca para mejorar presión en sistemas inhomogéneos en **nuevas** corridas

**Cambio 2 — Comunicación con SFC y halos selectivos** (impacto: CRÍTICO para HPC; esfuerzo histórico ALTO — **parcialmente cerrado en 2026**)
- **Implementado:** descomposición SFC real (`gadget-ng-parallel`: `exchange_domain_sfc`, `exchange_halos_sfc`), halos por AABB con poda geométrica conservadora, fase de datos por P2P (`Isend`/`Irecv`) y, cuando `P>8` y hay pares sin solape en ningún sentido, intercambio de **conteos** disperso en lugar de `MPI_Alltoall` de enteros sobre todos los rangos
- **Objetivo seguido:** sustituir el coste dominante de `allgatherv_state` O(N×ranks) en rutas LET/SFC por trabajo proporcional a halos/migración en lugar de reunir todo el estado global cada vez
- **Sigue abierto:** rutas que aún llaman `allgatherv_state` (p. ej. BH multi-rank sin LET/SFC o fallback); weak scaling formal a >>8 ranks por medir/documentar con esta base

**Cambio 3 — Paralelismo intra-nodo (Rayon)** (impacto: MEDIO, esfuerzo: MEDIO)
- `rayon_bh.rs` + wiring en CLI; activación condicionada (véase Prioridad 3 abajo)
- Objetivo: reducir wall time serial del walk BH para N>5000 cuando SIMD no-determinista está activo
- Especialmente útil para el recorrido del árbol con términos cuadrupolar/octupolar

---

## 11. Limitaciones actuales cuantificadas

| Limitación                  | Impacto medido                                      | Umbral de aceptabilidad          |
|-----------------------------|-----------------------------------------------------|----------------------------------|
| Criterio apertura geométrico fijo θ | mean\_err=3.76% Plummer θ=0.5 (con quad+oct activos) | <0.5% para publicaciones  |
| Max\_err = 190% (Plummer, θ=0.8) | Partículas del núcleo con MAC demasiado permisivo incluso con multipolos | Inaceptable para núcleos densos |
| Allgatherv weak scaling (Fase 3, ruta medida) | 47.6% efficiency a 4 ranks            | >80% para HPC serio; medir de nuevo con SFC/LET |
| Block timesteps para N pequeño | 3.3x más lento sin mejora de precisión           | Solo útil para N>10⁴ con alto rango dinámico |
| Intra-nodo (sin `simd` / Rayon) | 1 hilo útil para BH local salvo build `simd` + no determinista | Multi-core ayuda para >N≈5000 |
| Crossover BH > Direct       | Ocurre en N≈4000–5000, no en N≈1000                | Con criterio relativo podría bajar a N≈2000–3000 |

---

## 12. Backlog técnico priorizado

> **Corrección post-Fase 3:** "Cuadrupolo en octree" ya está completamente implementado (`quad:[f64;6]`, `oct:[f64;7]` calculados en `aggregate()`, aplicados en `walk_inner()`). El backlog se actualiza eliminando esa tarea y reordenando prioridades.
>
> **Actualización post-Fase 5:** Prioridad 1 (criterio relativo + `multipole_order` + softening consistente entre monopolo y multipolos) está **completada** por las Fases 4 y 5. La Fase 5 añadió además `mac_softening=consistent`, que hace el estimador del MAC relativo coherente con el kernel Plummer, validó multi-step y confirmó que el drift energético está dominado por el integrador, no por el solver. Ver [Fase 5 — Consistencia MAC-softening](2026-04-phase5-energy-mac-consistency.md).

### Prioridad 1 — Criterio de apertura relativo + `multipole_order` + MAC-softening (retorno: ALTO) — **completada (Fases 4–5)**
- Implementado en `RunConfig` / gravedad: `opening_criterion`, `err_tol_force_acc`, `multipole_order`, `mac_softening`, etc.; uso en `octree.rs` / `barnes_hut.rs`
- Documentación y validación energética: [Fase 5 — Consistencia MAC-softening](2026-04-phase5-energy-mac-consistency.md)
- **Trabajo residual (no “implementar el criterio”):** campañas de benchmark que repitan §4 con `opening_criterion = "relative"` y ablación de `multipole_order` para tablas nuevas

### Prioridad 2 — Comunicación punto-a-punto / SFC (retorno: CRÍTICO para HPC)
- Ficheros: `crates/gadget-ng-parallel/src/` (`exchange_domain_sfc`, `exchange_halos_sfc`, `mpi_rt`, `halo3d`), uso en `gadget-ng-cli/src/engine/stepping.rs`
- Impacto: objetivo sigue siendo subir weak scaling (referencia histórica ~47% @ 4 ranks) y habilitar rangos altos; la base SFC + halos + poda + MPI disperso ya está **implementada**
- **Restante:** eliminar o reducir llamadas a `allgatherv_state` en rutas que aún las usan; benchmarks weak scaling con MPI real (≫8 ranks) y afinado por caso de uso
- Estimación: ya no 3–4 semanas “desde cero”; orden semanas según alcance de rutas y mediciones

### Prioridad 3 — Paralelismo intra-nodo (Rayon; análogo práctico a OpenMP) (retorno: MEDIO)
- Ficheros: `crates/gadget-ng-tree/src/rayon_bh.rs`; activación vía `gadget-ng-cli` (`local_bh_use_rayon`, `compute_forces_*`)
- Impacto: paralelismo intra-nodo sobre el walk BH cuando está habilitado; reduce wall time serial para N grandes
- **Estado 2026:** habilitado solo con **feature `simd`**, solver BH y `performance.deterministic = false`; ejemplo `examples/nbody_bh_dtree_rayon_smoke.toml`. Sin `simd`, intra-nodo BH sigue serial
- Estimación: ampliar cobertura/defaults/documentación, ~días a 1 semana (no “implementar Rayon desde cero”)

### Prioridad 4 — Diagnósticos de timing por fase (completado en Fase 3)
- Implementado: `timings.json` con desglose comm/gravity/integration
- Suficiente para benchmarking; GADGET-4 tiene diagnósticos más detallados (stack/fetch time)

### Prioridad 5 — TreePM real (retorno: BAJO para N<10⁴)
- PM grid ya implementado (`PmSolver`, `TreePmSolver`)
- El splitting erf/erfc para TreePM ya existe en `gadget-ng-treepm`
- Validación formal con benchmarks cosmológicos pendiente
- Estimación: 2 semanas (validación + integración con criterio relativo)

---

## 13. Conclusiones

**¿En qué condiciones gadget-ng ya es físicamente confiable?**
- Sistemas con N≤500 y distribuciones extendidas: |ΔE/E|<0.5% con BH θ=0.5
- Validaciones de conservación (Kepler, Plummer virial, colapso frío): cumplidas en Fase 2
- Modo serial para investigación de N-body pequeño a mediano: funcional

**¿Dónde el Barnes-Hut actual todavía falla?**
- Distribuciones concentradas (Plummer a<0.3): error máximo >100% para θ≥0.8 **incluso con cuadrupolo y octupolo activos**
- Para θ=0.5, el error medio ya es 3.76% en Plummer → no aceptable para papers
- La causa raíz es el **criterio de apertura geométrico fijo θ**, que es demasiado permisivo en el núcleo denso; el cuadrupolo ya está implementado pero no puede compensar un MAC incorrecto
- Requiere criterio de apertura relativo (Prioridad 1) para sistemas tipo cúmulo estelar

**¿Cuánto escala MPI realmente?**
- En la **campaña Fase 3** (§8–9): strong ~89% eff. a 4 ranks; weak 47.6% — diagnóstico válido para la ruta **Allgatherv medida**
- La limitación **cuantificada** ahí es esa ruta (coste global O(N×ranks)); el código actual **también** ofrece SFC/halos (§10.2), pero **sin nuevas mediciones** no sustituye las tablas de este informe
- La “paridad no-determinística” ilusoria de §7 aplica al experimento Allgatherv descrito; no generalizar a todos los modos MPI

**¿Qué falta para acercarse técnicamente a GADGET-4 (estado repo vs benchmark histórico)?**
1. **Precisión / MAC:** criterio relativo y `multipole_order` **ya están en el código**; falta **actualizar benchmarks públicos** (repitiendo §4 con `opening_criterion = "relative"`) y comparar con GADGET-4 en igualdad de condiciones
2. **Escalabilidad MPI:** base SFC + halos **implementada**; falta **weak/strong scaling medido** con esa ruta y reducir usos residuales de `allgatherv_state` donde aún aplique
3. **Intra-nodo masivo:** GADGET-4 usa OpenMP; gadget-ng tiene **Rayon** condicionado (`simd`); cerrar brecha de modelo de hilos si se persigue paridad operativa

**Próximo paso con mayor retorno (después de Fase 5):**
Campanas cuantitativas nuevas: (a) errores de fuerza y energía con MAC relativo vs tabla §4; (b) MPI scaling con configuración LET/SFC explícita frente a §8–9. Fragmentos TOML de partida: **Apéndice B**. El backlog detallado está en §12.

---

## Apéndice: Comandos de reproducibilidad

```bash
# Clonar y compilar
git clone <repo> gadget-ng && cd gadget-ng
cargo build --release

# Experimento 1: BH force accuracy
cargo test -p gadget-ng-physics --test bh_force_accuracy --release -- --nocapture bh_force_accuracy_full_sweep
python3 experiments/nbody/phase3_gadget4_benchmark/bh_force_error/scripts/plot_bh_accuracy.py

# Experimento 2: Scaling N
python3 experiments/nbody/phase3_gadget4_benchmark/scaling_n/scripts/run_scaling_n.py
python3 experiments/nbody/phase3_gadget4_benchmark/scaling_n/scripts/plot_scaling_n.py

# Experimento 3: Block timesteps
bash experiments/nbody/phase3_gadget4_benchmark/block_timestep_compare/scripts/run_block_compare.sh
python3 experiments/nbody/phase3_gadget4_benchmark/block_timestep_compare/scripts/analyze_block_compare.py
python3 experiments/nbody/phase3_gadget4_benchmark/block_timestep_compare/scripts/plot_block_compare.py

# Experimento 4: Paridad serial vs MPI
bash experiments/nbody/phase3_gadget4_benchmark/serial_vs_mpi_parity/scripts/run_parity.sh
python3 experiments/nbody/phase3_gadget4_benchmark/serial_vs_mpi_parity/scripts/analyze_parity.py

# Experimento 5: Strong scaling
bash experiments/nbody/phase3_gadget4_benchmark/mpi_strong_scaling/scripts/run_strong_scaling.sh
python3 experiments/nbody/phase3_gadget4_benchmark/mpi_strong_scaling/scripts/analyze_strong_scaling.py

# Experimento 6: Weak scaling
bash experiments/nbody/phase3_gadget4_benchmark/mpi_weak_scaling/scripts/run_weak_scaling.sh
python3 experiments/nbody/phase3_gadget4_benchmark/mpi_weak_scaling/scripts/analyze_weak_scaling.py
```

Todos los resultados numéricos son reproducibles desde el mismo commit con los comandos anteriores.

---

## Apéndice B: Fragmentos TOML de referencia (campañas nuevas vs §4 y §8–9)

Los siguientes bloques **no** sustituyen automáticamente las tablas históricas de este informe: sirven como punto de partida para reruns con MAC relativo y para mediciones MPI en la ruta **SFC + LET** descrita en §10.2 / §12. Validar siempre contra `[RunConfig]` actual (`crates/gadget-ng-core/src/config.rs`).

### B.1 Barnes–Hut con criterio de apertura relativo (contraste cuantitativo con §4)

Alineado con `ErrTolForceAcc` típico de GADGET-4 y con la Fase 5 (MAC + softening consistente). El test `cargo test -p gadget-ng-physics --test bh_force_accuracy` ya genera CSV separados para MAC geométrico (`bh_accuracy.csv`) y **relativo** (`bh_accuracy_relative.csv`) con `err_tol_force_acc = 0.0025`.

Para integrar el mismo criterio en un `.toml` de **stepping**:

```toml
[gravity]
solver              = "barnes_hut"
theta               = 0.5
opening_criterion   = "relative"
err_tol_force_acc   = 0.0025
multipole_order     = 3
softened_multipoles = true
mac_softening       = "consistent"
```

Valores de `[simulation]` / `[initial_conditions]` pueden coincidir con el §4 (p. ej. N=500, ε=0.05, esfera uniforme o Plummer) para comparación directa con la tabla 4.1.

### B.2 MPI con descomposición SFC y LET (contraste de escalado con §8–9)

Compilar con MPI: `cargo build --release -p gadget-ng-cli --features mpi`. En multirank con solver Barnes–Hut y sin forzar el fallback Allgather, el stepping puede usar migración/halo vía SFC + `exchange_halos_sfc` (véase documentación en `gadget-ng-parallel`).

```toml
[performance]
deterministic              = true   # false si se usa build --features simd y se desea Rayon intra-nodo
use_distributed_tree       = true
use_sfc                    = true
sfc_kind                   = "hilbert"   # o "morton"
force_allgather_fallback   = false       # true reproduce el baseline O(N·P) deliberadamente
halo_factor                = 0.5
# opcionales con defaults razonables: let_nonblocking, use_let_tree, sfc_rebalance_interval
```

Ejemplo de lanzamiento (ajustar rutas y número de rangos):

```bash
cargo build --release -p gadget-ng-cli --features mpi
mpirun -n 4 ./target/release/gadget-ng stepping --config mi_escala_sfc.toml --out runs/sfc_weak --snapshot
```

**Nota:** Los scripts bajo `experiments/nbody/phase3_gadget4_benchmark/mpi_strong_scaling/` y `mpi_weak_scaling/` reproducen la **campaña Fase 3**; para curvas comparables con §8–9 pero en la ruta SFC hay que **sustituir o derivar** el TOML del experimento a partir del bloque B.2 (u homologar `particle_count` / pasos con esos scripts).
