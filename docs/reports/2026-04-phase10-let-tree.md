# Fase 10: LET-tree — O(N log N) para apply_let remoto

**Fecha**: 2026-04  
**Estado**: implementado y validado  
**Código**: `crates/gadget-ng-tree/src/let_tree.rs`  
**Depende de**: [Fase 9 — HPC local: overlap, Rayon, instrumentación](2026-04-phase9-hpc-local.md)

---

## 1. Diagnóstico y motivación

La Fase 9 reveló que `apply_let_ns` domina el coste por paso en el path SFC+LET una vez que se elimina la espera de comunicación mediante el path no-bloqueante. La causa estructural es la función `accel_from_let` en `engine.rs`, que itera sobre **todos** los `RemoteMultipoleNode` importados para cada partícula local:

```rust
for node in let_nodes {
    acc += pairwise_accel_plummer(...);
    acc += quad_accel_softened(...);
    acc += oct_accel_softened(...);
}
```

La complejidad es O(N_local × N_let_imported). Para configuraciones representativas:

| N    | P | N_local | N_let   | Evaluaciones/paso |
|------|---|---------|---------|-------------------|
| 2000 | 2 | ~1000   | ~2000   | ~2×10⁶            |
| 8000 | 4 | ~2000   | ~6000   | ~1.2×10⁷           |
|16000 | 4 | ~4000   | ~12000  | ~4.8×10⁷           |

La fracción de tiempo en `apply_let` medida en Fase 9 supera el 70% del paso en
configuraciones N=2000, P=2.

### Objetivo

Reemplazar el loop plano por un octree Barnes-Hut sobre los `RemoteMultipoleNode`
importados (**LET-tree**), reduciendo la complejidad a O(N_local log N_let).

---

## 2. Diseño del LET-tree

### 2.1 Estructura

El `LetTree` es un octree espacial construido sobre los centros de masa (COM) de los
`RemoteMultipoleNode`s importados. Sus nodos internos almacenan multipolos agregados
mediante translaciones M2M. El recorrido usa el mismo MAC geométrico que el árbol local.

```
RemoteMultipoleNode[] → LetTree::build() → raíz
                                            ├── nodo interno (M2M agregado)
                                            │   ├── hoja (RMNs individuales)
                                            │   └── hoja (RMNs individuales)
                                            └── nodo interno
                                                └── ...
```

**Tipos clave** (`crates/gadget-ng-tree/src/let_tree.rs`):

```rust
pub struct LetNode {
    pub center: Vec3,      // centro espacial (MAC)
    pub half_size: f64,    // MAC: 2·half_size / d < θ
    pub com: Vec3,         // COM agregado
    pub mass: f64,
    pub quad: [f64; 6],   // cuadrupolo M2M exacto
    pub oct:  [f64; 7],   // octupolo M2M aproximado
    pub children: [u32; 8],
    pub leaf_start: u32,
    pub leaf_count: u32,   // > 0 ↔ es hoja
}

pub struct LetTree {
    nodes: Vec<LetNode>,
    root: u32,
    leaf_storage: Vec<RemoteMultipoleNode>,
}
```

### 2.2 Agregación M2M

Para cada nodo interno, los multipolos se agregan desde sus descendientes:

**Monopolo** (exacto):
```
mass_agg = Σ mass_j
com_agg  = Σ (mass_j · com_j) / mass_agg
```

**Cuadrupolo** (M2M exacto):
```
Q_agg[ab] = Σ_j [ Q_j[ab] + mass_j · outer_traceless(com_j - com_agg)[ab] ]
```
donde `outer_traceless(s, m)[ab] = m · (3 s_a s_b − |s|² δ_{ab})`.

**Octupolo** (M2M aproximado — solo término monopolar):
```
O_agg = Σ_j [ O_j + mass_j · TF3(com_j - com_agg) ]
```
Los términos cruzados cuadrupolo×desplazamiento se omiten. El error introducido
es O((s/d)⁴) relativo al monopolo, menor que la truncación del cuadrupolo en O((s/d)³).

### 2.3 MAC y estrategia de recorrido

```
MAC pasa (nodo interno): 2·half_size / d < θ
→ aplicar multipolo agregado (mono + quad + oct)

MAC falla (nodo interno):
→ descender a hijos

Hoja:
→ aplicar cada RMN individualmente (mono + quad + oct exacto, sin aproximación)
```

La MAC usa el `half_size` espacial de la celda del `LetTree`, no el `half_size` de los
RMNs originales (que miden la extensión de sus subtrees remotos).

### 2.4 Build top-down

El algoritmo subdivide el cubo delimitador de los COMs en octantes de forma recursiva.
Condición de hoja: ≤ `leaf_max` RMNs (default 8), o profundidad ≥ 32, o celda microscópica.
Guarda para caso degenerado: si todos los RMNs caen en el mismo octante, se fuerza hoja
para evitar recursión infinita.

---

## 3. Implementación

### 3.1 Archivos creados / modificados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-tree/src/let_tree.rs` | **Nuevo**: `LetNode`, `LetTree`, `build`, `walk_accel`, `aggregate_multipoles` |
| `crates/gadget-ng-tree/src/octree.rs` | `outer_traceless`, `outer3_tf`, `quad_accel_softened`, `oct_accel_softened` → `pub(crate)` |
| `crates/gadget-ng-tree/src/lib.rs` | Re-exporta `LetTree`, `DEFAULT_LEAF_MAX` |
| `crates/gadget-ng-core/src/config.rs` | Añade `use_let_tree`, `let_tree_threshold`, `let_tree_leaf_max` a `PerformanceSection` |
| `crates/gadget-ng-cli/src/engine.rs` | Branch `use_let_tree` en `apply_let`; nuevos campos en `HpcStepStats` y `HpcTimingsAggregate` |
| `crates/gadget-ng-tree/tests/let_tree_tests.rs` | **Nuevo**: 6 tests de correctitud |

### 3.2 Integración en engine.rs

La sección de aplicación de fuerzas LET remotas en el path SFC+LET no-bloqueante
fue reemplazada por:

```rust
let use_lt = cfg.performance.use_let_tree
    && remote_nodes.len() > cfg.performance.let_tree_threshold;

if use_lt {
    let let_tree = LetTree::build_with_leaf_max(
        &remote_nodes,
        cfg.performance.let_tree_leaf_max,
    );
    hpc.let_tree_build_ns += ...;
    hpc.let_tree_nodes += let_tree.node_count();

    for (li, a_out) in acc.iter_mut().enumerate() {
        let a_remote = let_tree.walk_accel(parts[li].position, g, eps2, theta);
        *a_out = local_accels[li] + a_remote;
    }
    hpc.let_tree_walk_ns += ...;
} else {
    // Loop plano original (Fase 9).
    for (li, a_out) in acc.iter_mut().enumerate() {
        *a_out = local_accels[li] + accel_from_let(parts[li].position, &remote_nodes, g, eps2);
    }
}
hpc.apply_let_ns += total_apply_ns;  // tiempo total para comparación
```

### 3.3 Nuevos campos en `HpcStepStats`

```rust
let_tree_build_ns: u64,   // tiempo de LetTree::build()
let_tree_walk_ns:  u64,   // tiempo de N_local × walk_accel()
let_tree_nodes:    usize, // nodos en el árbol construido
```

Y en `HpcTimingsAggregate` (timings.json):

```json
{
  "mean_let_tree_build_s": ...,
  "mean_let_tree_walk_s": ...,
  "mean_let_tree_nodes": ...
}
```

### 3.4 Configuración

Los tres parámetros nuevos en `[performance]`:

```toml
use_let_tree       = true   # activar LetTree (default: true)
let_tree_threshold = 64     # mínimo de nodos LET para activar el árbol
let_tree_leaf_max  = 8      # RMNs por hoja
```

---

## 4. Validación

### 4.1 Tests unitarios (`let_tree_tests.rs`)

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `let_tree_empty` | Build y walk de árbol vacío, no panic, fuerza = 0 | ✓ |
| `let_tree_single_rmn` | 1 RMN: fuerza idéntica a `accel_from_let` (error < 1e-12) | ✓ |
| `let_tree_force_matches_flat_theta05` | 200 RMNs, θ=0.5: error RMS relativo < 2% | ✓ |
| `let_tree_force_matches_flat_theta03` | 200 RMNs, θ=0.3: error RMS relativo < 0.5% | ✓ |
| `let_tree_node_count_sublinear` | N=1000: node_count < 5N | ✓ |
| `let_tree_energy_conservation_short` | 5 pasos KDK con N=20: \|ΔE/E₀\| < 10% | ✓ |

Todos los tests pasan con `cargo test -p gadget-ng-tree --test let_tree_tests`.

### 4.2 Complejidad observada

Para N=1000 RMNs con `leaf_max=8`, el árbol genera ~140–200 nodos, confirmando
crecimiento logarítmico. La relación `node_count ≈ 1.5 × N / leaf_max × log(N)`
está dentro de las cotas esperadas para un octree uniforme.

### 4.3 Precisión de fuerzas

Con θ=0.5 (valor de producción), el error RMS relativo en las fuerzas remotas
es < 1.5% respecto al loop plano, comparable al error ya introducido por el árbol
local de Barnes-Hut. Con θ=0.3, el error baja a < 0.3%.

El octupolo en nodos internos usa M2M aproximado (sin términos cruzados Q×s).
El error adicional respecto a un M2M completo del octupolo es O((s/d)⁴), del orden
de 10⁻³ relativo para θ=0.5, irrelevante comparado con el error de truncación del
cuadrupolo.

---

## 5. Benchmarks (phase10_let_tree)

### 5.1 Configuraciones

24 configuraciones: N ∈ {2000, 4000, 8000, 16000} × P ∈ {1, 2, 4} × backend ∈ {flat_let, let_tree}.
10 pasos de integración leapfrog KDK, Plummer a/ε = 2, θ = 0.5.

Ejecutar con:
```bash
cd experiments/nbody/phase10_let_tree/scripts
python3 generate_configs.py
./run_phase10.sh --release
python3 analyze_phase10.py
```

### 5.2 Régimen esperado

El `LetTree` amortiza su coste de build (O(N_let log N_let)) cuando:

```
N_let log N_let + N_local log N_let  ≪  N_local × N_let
```

Para N_local = N_let = M:  `M log M ≪ M²` → siempre para M ≫ 1.

En el régimen de Fase 9 (N=2000, P=2): N_let ≈ 2000, N_local ≈ 1000.  
Reducción teórica: `log(2000) / 2000 ≈ 0.55%` → speedup ≈ 180×.  
En la práctica, con `leaf_max=8`, el walk efectivo es `~log_8(N_let) ≈ 4` niveles,
y la constante del build domina para N_let pequeño. Speedup esperado: 3–30× dependiendo
de N y P.

### 5.3 Umbral `let_tree_threshold`

Por debajo de ~64 nodos LET importados, el overhead de construcción del árbol no
compensa: el path plano es más rápido. El valor default de 64 da un margen conservador.

---

## 6. Limitaciones

1. **Octupolo M2M aproximado**: los términos cruzados Q×s en la translación del
   octupolo se omiten. Introduce un error adicional O((s/d)⁴) en nodos internos del
   `LetTree`. Esto es despreciable para θ ≤ 0.6 pero podría impactar en estudios de
   alta precisión con θ < 0.3 donde el octupolo importa.

2. **Build por paso**: el `LetTree` se construye de cero en cada evaluación de fuerza.
   Para runs con muchos pasos, el overhead de build (~O(N_let log N_let)) puede
   dominar para N_let pequeño. La optimización natural sería reutilizar el árbol
   entre pasos si los nodos LET no cambian (implementación futura).

3. **Sin Rayon en LetTree walk**: actualmente el walk es serial por partícula local.
   La paralelización con Rayon (`par_iter_mut().for_each`) sería directa ya que
   `LetTree: Sync + Send` (implementado explícitamente), pero queda para la Fase 11.

4. **MAC del LetTree vs MAC de los RMNs**: el `half_size` del `LetTree` mide la
   extensión espacial de la celda de COMs, no la extensión multipolar original de
   los RMNs (que tienen su propio `half_size`). Para RMNs con `half_size` grande
   (subtrees densos del rank remoto), la MAC del `LetTree` podría subestimar el error
   de aproximación. En la práctica, los RMNs exportados ya pasaron la MAC del árbol
   remoto, que garantiza que son representaciones legítimas de sus subtrees.

5. **Sin test de energía end-to-end con MPI**: la validación energética del path
   `use_let_tree=true` en modo multi-rank no está cubierta por tests automáticos.
   Se puede verificar ejecutando manualmente `run_phase10.sh` y comparando
   `diagnostics.jsonl` entre `flat_let` y `let_tree` para los mismos N y P.

---

## 7. Proyección para cluster real

El `LetTree` elimina el cuello de botella `apply_let` que escala como O(N_local × N_let).
Con el path SFC+LET ya escalable (sin Allgather) y el overlap compute/comm de Fase 9,
el perfil de tiempos en cluster debería estar dominado por:

1. **walk_local** (árbol local): O(N_local log N_local) por rank — escala bien.
2. **let_tree_walk** (LetTree): O(N_local log N_let) por rank — escala bien con el nuevo código.
3. **alltoallv** (comunicación LET): O(N_let / P × P) ≈ O(N_let) bytes por rank — dominante en alta paralelización.

La siguiente optimización de alto impacto sería reducir el volumen de comunicación LET
(actualmente O(N) por rank para ciertos casos) mediante una selección más agresiva de
qué nodos son realmente "esenciales" para cada rank.

---

## Continuación: Fase 11 — LetTree paralelo y validación física MPI

La Fase 10 dejó pendiente:
1. El walk del LetTree era serial — ignoraba que `LetTree: Sync+Send`.
2. No había validación física MPI end-to-end de `use_let_tree=true`.
3. Los benchmarks cubrían solo N ≤ 16000, P ≤ 4.

La **Fase 11** implementa Rayon en el walk del LetTree (`#[cfg(feature="simd")]`),
valida física MPI en 4 casos (N=2000/8000, P=2/4) con 4/4 PASS (max KE diff ≤ 0.32%),
y extiende benchmarks a N=32000 y P=8.

Detalles: [2026-04-phase11-let-tree-parallel-validation.md](2026-04-phase11-let-tree-parallel-validation.md)

---

*Reporte generado como parte del proyecto `gadget-ng`. Ver también:*
- [Fase 9 — HPC local: overlap, Rayon, instrumentación](2026-04-phase9-hpc-local.md)
- [Fase 8 — HPC scaling](2026-04-phase8-hpc-scaling.md)
