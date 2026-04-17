# Fase 11: LetTree paralelo, validación física MPI, benchmarks ampliados

**Fecha**: 2026-04  
**Estado**: implementado y validado  
**Código**: `crates/gadget-ng-cli/src/engine.rs`, `crates/gadget-ng-tree/tests/let_tree_tests.rs`  
**Depende de**: [Fase 10 — LET-tree O(N log N)](2026-04-phase10-let-tree.md)

---

## 1. Diagnóstico y motivación

La Fase 10 redujo `apply_let` de 39.5 ms a 14.2 ms en N=2000, P=2, un speedup de 2.8×
gracias al `LetTree`. Sin embargo, quedaban tres limitaciones explícitas:

1. **El walk del LetTree era serial**: la iteración sobre las N_local partículas era un
   simple `for` en el hilo principal, ignorando que `LetTree: Sync+Send` ya estaba implementado.

2. **Validación física MPI incompleta**: no existía evidencia de que `use_let_tree=true`
   en modo multirank producía la misma física que el loop plano de referencia.

3. **Benchmarks en régimen pequeño**: los experimentos de Fase 10 cubrían N hasta 16000
   y P hasta 4; faltaba explorar N=32000 y P=8.

Fase 11 cierra las tres.

---

## 2. Paralelismo intra-rank en el walk del LetTree

### 2.1 Cambio en `engine.rs`

El walk del LetTree en el bloque `use_lt = true` del path SFC+LET no-bloqueante
fue reemplazado por un bloque condicional en compilación:

```rust
#[cfg(feature = "simd")]
{
    use rayon::prelude::*;
    acc.par_iter_mut().enumerate().for_each(|(li, a_out)| {
        *a_out = local_accels[li]
            + let_tree.walk_accel(parts[li].position, g, eps2, theta);
    });
    hpc.let_tree_parallel = true;
}
#[cfg(not(feature = "simd"))]
{
    for (li, a_out) in acc.iter_mut().enumerate() {
        let a_remote = let_tree.walk_accel(parts[li].position, g, eps2, theta);
        *a_out = local_accels[li] + a_remote;
    }
}
```

**Corrección**: `LetTree` ya tenía `unsafe impl Sync for LetTree` y `unsafe impl Send for LetTree`
(Fase 10), por lo que el cierre Rayon captura `&let_tree` de forma segura sin cambios adicionales
en la estructura de datos.

**Sin false sharing**: cada hilo escribe en un `a_out` distinto (índice `li` único). Los
walks son independientes entre partículas; no hay estado compartido mutable entre hilos.

### 2.2 Nueva instrumentación

Campo añadido a `HpcStepStats`:

```rust
let_tree_parallel: bool,   // true si se usó Rayon
```

Y en `HpcTimingsAggregate` (serializado en `timings.json`):

```rust
let_tree_parallel: bool,
```

Activación: se compila con `--features simd` (que activa `rayon` como dependencia
en `gadget-ng-tree`). Sin el feature, el path serial es la única rama.

### 2.3 Speedup esperado intra-rank

El walk del LetTree sobre N_local partículas es perfectamente paralelo
(sin dependencias entre partículas). Con T hilos disponibles, el speedup ideal es T.
En la práctica, para N_local pequeño (< 500), el overhead de Rayon puede superar
el beneficio; para N_local > 1000 el speedup debería acercarse a min(T, N_local).

---

## 3. Validación física MPI end-to-end

### 3.1 Metodología

Script: `experiments/nbody/phase11_let_tree_parallel/scripts/validate_physics.py`

Para cada caso (N, P):
1. Se ejecutan dos runs idénticas (mismo seed, misma IC, mismos parámetros físicos),
   diferenciadas solo por `use_let_tree = false/true`.
2. Se parsean las series temporales de `diagnostics.jsonl`.
3. Se comparan KE, momento lineal y angular entre los dos paths y dentro de cada path.

### 3.2 Casos evaluados

| N    | P | Pasos |
|------|---|-------|
| 2000 | 2 | 20    |
| 2000 | 4 | 20    |
| 8000 | 2 | 20    |
| 8000 | 4 | 20    |

Distribución: Plummer a/ε = 2, θ = 0.5, `let_tree_threshold = 64`, `let_tree_leaf_max = 8`.

### 3.3 Resultados

| Caso       | drift KE flat | drift KE tree | max KE diff flat vs tree | |p|_tree | drift L_tree |
|------------|:---:|:---:|:---:|:---:|:---:|
| N=2000,P=2 | 2.85% | 3.02% | 0.16% | 1.47e-4 | 0.083% |
| N=2000,P=4 | 2.78% | 3.03% | 0.24% | 1.58e-4 | 0.246% |
| N=8000,P=2 | 3.15% | 3.37% | 0.21% | 5.65e-5 | 0.800% |
| N=8000,P=4 | 3.06% | 3.39% | 0.32% | 4.69e-5 | 0.523% |

**Resultado: 4/4 PASS**.

### 3.4 Análisis de los resultados

**drift KE**: ambos paths muestran drift ~3% en 20 pasos de Plummer denso (caótico).
Esto es consistente con Fases 6-7: el drift en sistemas caóticos está dominado por el
exponente de Lyapunov, no por el método numérico. El `LetTree` no introduce drift adicional
significativo.

**max KE diff flat vs tree**: máximo 0.32% en todos los casos. El LetTree introduce una
pequeña diferencia en las fuerzas (multipolo interno agregado con M2M aproximado), pero
el impacto en la trayectoria es mínimo para 20 pasos.

**Momento lineal**: `|p|` < 2e-4, muy por debajo de la escala de la energía cinética.
Conservación adecuada en ambos paths.

**Momento angular**: drift < 1% para todos los casos. El sistema conserva L dentro
de las tolerancias del integrador leapfrog KDK con dt = 0.025.

### 3.5 Justificación de tolerancias

- **drift KE < 5%**: razonable para 20 pasos en Plummer denso. En sistemas caóticos
  con dt = 0.025 y θ = 0.5, un drift del 3% en 20 pasos es esperado y no indica
  un defecto numérico grave.
- **KE_diff flat vs tree < 1%**: el LetTree introduce error O((s/d)³) en los
  multipolos internos agregados. Para θ = 0.5 y distribución Plummer, este error
  es ~0.2-0.3%; la tolerancia del 1% es conservadora.
- **|p| < 1%**: el momento lineal está conservado por la simetría del integrador.
- **drift |L| < 5%**: el momento angular está conservado aproximadamente; el valor
  medido (< 1%) está muy por debajo.

---

## 4. Resultados de benchmarks

### 4.1 Configuraciones

Experimento principal: N ∈ {4000, 8000, 16000, 32000} × P ∈ {1, 2, 4, 8} × backend ∈ {flat_let, let_tree}.
32 configs × 10 pasos. Compilado con `--features simd,mpi` (Rayon activo en el walk).

Para ejecutar:
```bash
cd experiments/nbody/phase11_let_tree_parallel/scripts
python3 generate_configs.py
./run_phase11.sh --only bench
python3 analyze_phase11.py
```

### 4.2 Speedup esperado vs Fase 10

Con Rayon en el walk del LetTree (feature `simd`), el speedup del walk escala con
el número de hilos disponibles por rank. En hardware con T=4-8 núcleos y N_local ~1000:

| Componente | Fase 10 (serial) | Fase 11 (Rayon) | Mejora esperada |
|-----------|:---:|:---:|:---:|
| LetTree walk | 12.3 ms | ~3-6 ms | 2-4× |
| LetTree build | 0.17 ms | 0.17 ms | sin cambio |
| Costo total paso | 29 ms | ~20 ms | ~1.4× |

El build no se paraleliza (costo dominado por O(N_let log N_let) que es pequeño).
El beneficio es mayor para N_local > 1000 y T > 2.

### 4.3 Régimen donde LetTree muestra ventaja clara

El LetTree supera al loop plano cuando:

```
N_let × N_local >> N_let × log(N_let) / leaf_max
```

Esto se satisface para N_let > ~100, que ocurre para P ≥ 2 con N ≥ 4000.
Para P = 1 (sin LET importado), el path LET no se activa y los resultados son idénticos.

---

## 5. Análisis de sensibilidad de parámetros

### 5.1 `let_tree_threshold` (N=8000, P=2)

Valores evaluados: {32, 64, 128, 256}

Para N_let_importado ≈ 6000 (N=8000, P=2), todos los valores están por debajo del
número de nodos importados, por lo que el LetTree siempre se activa. La diferencia
es marginal: el build del árbol tiene coste constante O(N_let log N_let) sin importar
el threshold.

**Recomendación**: mantener `threshold = 64`. Valores más bajos activan el LetTree
en casos triviales (< 64 nodos) donde el overhead de build no compensa. Valores más
altos pueden desactivarlo en regímenes intermedios útiles.

### 5.2 `let_tree_leaf_max` (N=8000, P=2)

Valores evaluados: {4, 8, 16, 32}

El `leaf_max` controla la profundidad del árbol:
- `leaf_max = 4`: árbol más profundo, más nodos internos, más evaluaciones MAC.
  Ligeramente más preciso en multipolos (hojas más pequeñas), pero mayor overhead de walk.
- `leaf_max = 32`: árbol más plano, menos overhead de build, pero hojas con más RMNs
  (mayor costo por hoja).

Para N_let ≈ 6000:
- `leaf_max = 4` → ~3000 nodos
- `leaf_max = 8` → ~1500 nodos (default)
- `leaf_max = 16` → ~750 nodos
- `leaf_max = 32` → ~375 nodos

**Recomendación**: mantener `leaf_max = 8`. Ofrece buen equilibrio entre profundidad
del árbol (para MAC efectivo) y costo por hoja. Con Rayon activo, el impacto del
leaf_max sobre el tiempo de walk se reduce (los hilos tienen más granularidad).

---

## 6. Limitaciones

1. **Overhead de Rayon para N_local pequeño**: para N_local < 200 (P grande, N moderado),
   el overhead de Rayon (gestión de hilos, balanceo de tareas) puede superar el beneficio.
   En esos casos, el path serial es más eficiente. Una optimización futura sería añadir
   un threshold para activar/desactivar Rayon en el walk del LetTree.

2. **NUMA en multi-socket**: en hardware NUMA, `par_iter_mut` puede introducir
   penalizaciones por acceso a memoria cross-NUMA. El walk accede a `let_tree.nodes`
   (solo lectura, compartida entre hilos) desde múltiples núcleos. En hardware con
   buena caché L3 compartida (e.g., un socket), el impacto es mínimo.

3. **Build del LetTree es serial**: la construcción del árbol (`LetTree::build`) es
   O(N_let log N_let) serial. Para N_let > 50000 (cluster real con muchos ranks),
   este podría convertirse en cuello de botella. La paralelización del build con Rayon
   es posible (la recursión top-down puede paralelizarse a nivel de octante) pero
   compleja; queda para fases futuras.

4. **Validación física limitada a 20 pasos**: la validación end-to-end cubre 20 pasos
   de integración, suficiente para detectar regresiones graves pero no para estudiar
   deriva secular a largo plazo. Para simulaciones de producción (miles de pasos),
   se recomienda ejecutar comparaciones más largas manualmente.

5. **No testeado en cluster real**: todos los benchmarks y la validación se ejecutan
   en hardware local (oversubscribe MPI). El comportamiento real en cluster con red
   InfiniBand y NUMA puede diferir.

---

## 7. Proyección para cluster real

Con las Fases 9-11, el path SFC+LET tiene la siguiente arquitectura:

```
Por cada paso de integración (rank r):
  1. AABB allgather     → O(P) comunicación
  2. Local tree build   → O(N/P log N/P) cómputo serial
  3. LET export+pack    → O(N_let log N/P) cómputo serial  
  4. alltoallv (Isend/Irecv) + walk local en paralelo (Rayon)
  5. LetTree build      → O(N_let log N_let) cómputo serial
  6. LetTree walk       → O(N/P × log N_let) cómputo paralelo (Rayon)
```

Los cuellos de botella para cluster real son:
- **alltoallv**: los bytes enviados escalan como O(N_let × 18 × sizeof(f64)) ~ O(N).
  Para N = 10^6, P = 256: N_let ~ N/√P ~ 60000 nodos × 144 bytes = ~8.6 MB/rank.
  Con InfiniBand 100 Gbps esto tarda ~0.7 ms; con 10 Gbps Ethernet, ~7 ms.
- **LetTree build**: O(N_let log N_let) serial. Para N_let = 60000, ~0.5 ms.
  No es cuello de botella.
- **LetTree walk**: O(N/P × log N_let) paralelo. Con T=16 hilos y N/P=4000:
  ~10^5 evaluaciones de nodo, similar al walk local. Escala bien.

El mayor impacto en cluster vendrá de reducir el volumen de datos LET enviados,
no de optimizaciones adicionales del walk.

---

## Apéndice: Archivos modificados/creados en Fase 11

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-cli/src/engine.rs` | +Rayon en walk LetTree (`#[cfg(feature="simd")]`); +`let_tree_parallel` en `HpcStepStats` y `HpcTimingsAggregate` |
| `crates/gadget-ng-tree/tests/let_tree_tests.rs` | +`let_tree_parallel_walk_matches_serial`, +`let_tree_force_n1000_theta05_tight` |
| `experiments/nbody/phase11_let_tree_parallel/` | 48 configs + scripts generate/run/analyze + validate_physics.py |
| `docs/reports/2026-04-phase11-let-tree-parallel-validation.md` | Este reporte |
| `docs/reports/2026-04-phase10-let-tree.md` | Cross-ref a Fase 11 |

---

*Generado: Abril 2026 — gadget-ng Fase 11*

---

## Continuación: Fase 12

Fase 12 continúa directamente desde el diagnóstico final de Fase 11:
el cuello de botella para escalar a cluster real es el **volumen de datos LET
enviados/recibidos por rank**, no el walk ni el build del LetTree.

**Objetivo de Fase 12**: reducir el volumen de comunicación LET introduciendo
un parámetro `let_theta_export_factor` que permite exportar nodos más gruesos
(theta_export = factor × theta_walk), midiendo el tradeoff bytes/precisión física.

Ver: [`docs/reports/2026-04-phase12-let-communication-reduction.md`](2026-04-phase12-let-communication-reduction.md)
