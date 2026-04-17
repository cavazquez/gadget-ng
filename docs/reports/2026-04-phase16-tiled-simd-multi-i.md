# Phase 16 — Tiling 4×N_i: SIMD sobre partículas locales vs. RMNs

**Fecha**: 2026-04-16  
**Fase**: 16  
**Objetivo**: Implementar y evaluar un kernel tileado 4×N_i que procese 4 partículas locales simultáneamente contra los mismos RMNs, buscando SIMD efectivo sin sacrificar la eficiencia jerárquica del `LetTree`.

---

## 1. Motivación

Las fases anteriores establecieron el siguiente diagnóstico:

| Fase | Estrategia SIMD | Resultado |
|------|-----------------|-----------|
| P14 | SoA + kernel fusionado (auto-vec, `xmm`) | +1.8x speedup global vs P13 |
| P15 | Intrinsics explícitos 1 partícula × 4 RMNs (`ymm`) | −8% (regresión) con `leaf_max=8` |
| P15b | Sweep `leaf_max` ∈ {8,16,32,64} | Aumentar `leaf_max` mejora P15 pero degrada el árbol global |

La raíz del fracaso de P15 era el tamaño medio de batch por hoja (3.7 RMNs con `leaf_max=8`):
con 4 RMNs por iteración SIMD, P15 producía ~0.9 iteraciones SIMD útiles por llamada de hoja.

**Hipótesis P16**: Invertir el rol SIMD. En lugar de 4 RMNs × 1 partícula, usar
**1 RMN × 4 partículas**. El registro `ymm` tiene [xi₀,xi₁,xi₂,xi₃] (las 4 partículas del tile).
Con SFC ordering, `parts[4k..4k+4]` son espacialmente próximas → las 4 comparten MAC decisions → MAC conservativo tiene overhead mínimo.

---

## 2. Implementación

### 2.1 Kernel `mono_pass_avx2_4xi` (rmn_soa.rs)

Loop sobre **N RMNs** (no sobre partículas):

```
Por cada RMN j:
  cxj = vbroadcastsd(cx[j])           → ymm = [cx[j], cx[j], cx[j], cx[j]]
  xi4 = vmovupd([xi[0],xi[1],xi[2],xi[3]])  → ymm (4 partículas)
  rx4 = xi4 - cxj                          (vsub ymm)
  r2_4 = vfmadd(rz,rz, vfmadd(ry,ry, vfmadd(rx,rx, eps2)))
  r_inv4 = 1 / vsqrtpd(r2_4)              ← vsqrtpd ymm garantizado
  store r_inv_out[j] = r_inv4             → buffer para pass 2
  factor4 = neg_g * mj * r_inv4³
  ax4 = vfmadd(factor4, rx4, ax4)        ← sin reducción horizontal
```

**Ventaja clave**: no se necesita reducción horizontal (`hadd`). Cada lane k del acumulador
`ax4` acumula independientemente para la partícula k. Al finalizar el loop, `ax4[k]` =
aceleración monopolar total de la partícula k. Esta es la diferencia esencial respecto a P15.

### 2.2 Pass 2 escalar `quad_oct_pass_scalar_4xi` (rmn_soa.rs)

Misma estructura que P15 pero para `tile_size` partículas:
para cada RMN j y cada partícula k: usa `r_inv_buf[j][k]` pre-computado → sin `sqrt` adicional.

### 2.3 Walk tileado `walk_accel_4xi` / `walk_inner_4xi` (let_tree.rs)

**MAC conservativo**: abre un nodo si CUALQUIERA de las `tile_size` partículas válidas falla el criterio.

```rust
let all_pass = pos[..tile_size].iter().all(|p| {
    let d = (*p - node.com).norm();
    d > 1e-300 && 2.0 * node.half_size / d < theta
});
```

Si `all_pass`: aplica multipolo a las `tile_size` partículas (escalar).  
Si no: desciende a todos los hijos.  
Si hoja: `apply_leaf_soa_4xi` → `accel_range_4xi`.

### 2.4 Loop Rayon tileado (engine.rs)

```rust
acc.par_chunks_mut(4)
   .zip(local_accels.par_chunks(4))
   .zip(parts.par_chunks(4))
   .for_each(|((acc_tile, local_tile), parts_tile)| {
       let tile_size = parts_tile.len();
       let mut pos = [Vec3::zero(); 4];
       for k in 0..tile_size { pos[k] = parts_tile[k].position; }
       let result = let_tree.walk_accel_4xi(pos, tile_size, g, eps2, theta);
       for k in 0..tile_size { acc_tile[k] = local_tile[k] + result[k]; }
   });
```

El manejo de tail (`N mod 4 ≠ 0`) es automático: el último chunk tiene `tile_size` < 4.

---

## 3. Verificación ASM

`objdump` sobre el binario de tests en release (`RUSTFLAGS="-C target-cpu=native"`):

```
3f985:  vbroadcastsd 0xd0(%rsp),%ymm17
3fd5b:  vfmadd213pd %ymm17,%ymm2,%ymm3
3fd66:  vbroadcastsd %xmm4,%ymm4
3fd6b:  vfmadd231pd %ymm0,%ymm0,%ymm3
3fd70:  vsqrtpd %ymm3,%ymm3             ← sqrt de 4 doubles simultáneos
3fd7a:  vmulpd %ymm3,%ymm3,%ymm5
3fd8f:  vmulpd %ymm3,%ymm4,%ymm3
3fd93:  vfmadd231pd %ymm2,%ymm3,%ymm21  ← acumulación ax4
3fd99:  vfmadd231pd %ymm1,%ymm3,%ymm22  ← acumulación ay4
3fd9f:  vfmadd231pd %ymm0,%ymm3,%ymm20  ← acumulación az4
```

Instrucciones garantizadas: `vbroadcastsd ymm`, `vsqrtpd ymm`, `vfmadd231pd ymm`, `vmulpd ymm`.
**El kernel explota 256-bit AVX2/FMA tal como se diseñó.**

---

## 4. Benchmarks

### 4.1 Wall time total (s, 10 pasos, `leaf_max=8`, `θ=0.5`)

| Config | P14 (fused) | P15 (1xi) | P16 (4xi) | sp P15/P14 | sp P16/P14 |
|--------|-------------|-----------|-----------|-----------|-----------|
| N=8000, P=2  | 1.089 | 1.162 | **1.407** | 0.937x | **0.774x** |
| N=8000, P=4  | 0.712 | 0.787 | **0.900** | 0.905x | **0.791x** |
| N=16000, P=2 | 2.592 | 2.685 | **3.346** | 0.965x | **0.775x** |
| N=16000, P=4 | 1.724 | 1.787 | **2.180** | 0.964x | **0.791x** |

### 4.2 LetTree walk time (s/paso, media de 10 pasos)

| Config | P14 (fused) | P15 (1xi) | P16 (4xi) | sp P16/P14 |
|--------|-------------|-----------|-----------|-----------|
| N=8000, P=2  | 0.0374 | 0.0443 | **0.0675** | **0.554x** |
| N=8000, P=4  | 0.0313 | 0.0368 | **0.0487** | **0.642x** |
| N=16000, P=2 | 0.0844 | 0.0976 | **0.1596** | **0.529x** |
| N=16000, P=4 | 0.0654 | 0.0657 | **0.1045** | **0.626x** |

### 4.3 Métricas de tiles P16

| Config | tile_calls/paso | tile_i/paso | util_ratio | leaf_calls_P14 | factor_calls |
|--------|----------------|-------------|-----------|----------------|-------------|
| N=8000, P=2  | 354,851 | 1,418,480 | **0.9993** | 629,856 | 0.563× |
| N=8000, P=4  | 251,485 | 1,005,274 | **0.9993** | 514,506 | 0.489× |
| N=16000, P=2 | 720,482 | 2,881,593 | **0.9999** | 1,233,517 | 0.584× |
| N=16000, P=4 | 471,988 | 1,887,262 | **0.9996** | 894,297 | 0.528× |

> `util_ratio` = tile_i / (tile_calls × 4) ≈ 1.0 → tiles casi completamente llenos.  
> `factor_calls` = tile_calls / leaf_calls_P14 → P16 hace ~0.5× las llamadas de P14.

---

## 5. Diagnóstico: por qué P16 es más lento

### 5.1 Causa primaria: overhead del MAC conservativo

El `tile_utilization_ratio ≈ 0.9999` confirma que el kernel SIMD funciona perfectamente.
Sin embargo, P16 es ~60-88% más lento en `let_tree_walk_ns` que P14.

Comparación de trabajo efectivo (N=8000, P=2):

| Métrica | P14 | P16 |
|---------|-----|-----|
| leaf_calls/paso | 629,856 | — |
| tile_calls/paso | — | 354,851 |
| tile_i/paso     | — | 1,418,480 |
| RMNs evaluados (aprox.) | 2,307,550 | ≈ tile_i × 3.66* = 5,191,638 |

*batch_avg de P14 aplicado a tile_calls.

> P16 evalúa ~2.25× más pares (partícula, RMN) que P14, aunque tiene ~0.56× las llamadas.

**Por qué**: el MAC conservativo ("abrir si CUALQUIERA de las 4 falla") hace que cada tile
visite tantos nodos como la partícula más "desfavorable" del grupo. Las 4 partículas cubren
un volumen espacial mayor que una sola; nodos que pasarían MAC para la partícula más cercana
se abren para el tile completo. Con θ=0.5 y `leaf_max=8`, el árbol tiene ~3 niveles útiles
y el error de MAC conservativo se acumula en cada nivel.

### 5.2 Cuantificación del overhead

Factor de overhead del MAC = (RMNs evaluados P16) / (RMNs evaluados P14)
≈ 5.19e6 / 2.31e6 ≈ **2.25×**

El kernel SIMD 4xi ofrece una aceleración teórica máxima de ~4× sobre el kernel escalar.
Pero el overhead de MAC (2.25×) más el overhead de recursión tileada (~1.3× por flops
adicionales en nodos internos) consumien completamente el beneficio SIMD.

### 5.3 Invarianza del kernel — el SIMD funciona

Los tests unitarios confirman RMS < 1e-12 para:
- `p16_4xi_vs_scalar_various_n`: N ∈ {1,4,8,16,17,64,65,128}
- `p16_4xi_tail_handling`: tile_size ∈ {1,2,3}
- `p16_4xi_vs_p15_rms`: 200 RMNs con quad+oct completo

El problema no está en el kernel SIMD, sino en la interacción del tiling con la estructura del árbol.

---

## 6. Validación física

Run de validación: N=2000, P=2, 5 pasos, θ=0.5, `leaf_max=8`.

| Métrica | Valor | Tolerancia | Estado |
|---------|-------|------------|--------|
| \|ΔKE_rel\|(P16 vs P14) / KE₀ | 1.14×10⁻⁵ | 1×10⁻³ | **PASS** |
| \|Δpx\| | 6.35×10⁻⁷ | 1×10⁻⁴ | **PASS** |
| \|Δpy\| | 3.04×10⁻⁷ | 1×10⁻⁴ | **PASS** |
| \|Δpz\| | 1.24×10⁻⁶ | 1×10⁻⁴ | **PASS** |
| \|ΔL\| (P16 vs P14) | 2.76×10⁻⁶ | 1×10⁻³ | **PASS** |

P16 produce resultados físicamente idénticos a P14 dentro de la precisión del MAC (θ=0.5).
De hecho, P16 podría ser ligeramente más preciso porque el MAC conservativo abre más nodos
(más evaluaciones directas), reduciendo el error de aproximación multipolar.

---

## 7. Conclusión

### 7.1 Logros confirmados

- **Kernel correcto**: `mono_pass_avx2_4xi` emite `vbroadcastsd ymm`, `vsqrtpd ymm`,
  `vfmadd231pd ymm` — el kernel AVX2 4xi es correcto y explota 256-bit real.
- **Utilización perfecta**: `tile_utilization_ratio ≈ 0.9999` — tiles llenos de 4 partículas.
- **Física idéntica**: todos los deltas < tolerancias por 1-2 órdenes de magnitud.
- **Sin reducción horizontal**: al acumular sobre RMNs (no sobre partículas), no se necesita
  `hadd` — esta es la ventaja arquitectural del enfoque 4×N_i.

### 7.2 Limitación fundamental

El MAC conservativo anula el beneficio SIMD. La raíz es geométrica: 4 partículas SFC-adyacentes
cubren un volumen finito (aunque pequeño); un nodo del árbol que pasaría MAC para la partícula
más cercana puede fallar para la más lejana. Con θ=0.5 esto ocurre frecuentemente, ampliando
el árbol efectivamente recorrido en ~2.25×.

### 7.3 Lecciones para fases futuras

| Estrategia | Descripción | Esperada |
|-----------|-------------|---------|
| **Tiled MAC con bbox** | Comprobar MAC contra la bounding box del tile, no por partícula individual | Reduce overhead MAC; más complejo |
| **Leaf-level batching** | Mantener walk individual; batchear solo en hojas cuando múltiples partículas convergen | Evita MAC overhead; requiere scheduler |
| **Software prefetch** | Para N grandes, la latencia de memoria en quad_oct_pass domina; prefetch puede ayudar | Beneficio marginal en N moderados |
| **N mayor** | Con N=64k+, los árboles son más profundos y las hojas tienen más RMNs; ratio SIMD/overhead mejora | Requiere más memoria/nodos |

### 7.4 Estado del proyecto

La búsqueda de SIMD efectivo en el `LetTree` ha identificado una limitación arquitectural:
el balance entre granularidad de árbol (`leaf_max`) y SIMD width (4 doubles por ymm) requiere
estrategias que no comprometan el MAC o la estructura jerárquica. P16 cierra la exploración
de estrategias de tiling directo con MAC conservativo.

---

## Apéndice A: Archivos modificados

| Archivo | Cambios |
|---------|---------|
| `crates/gadget-ng-tree/src/rmn_soa.rs` | +`RINV_CHUNK_4XI`, +`mono_pass_avx2_4xi`, +`quad_oct_pass_scalar_4xi`, +`accel_p16_avx2_range_4xi`, +`RmnSoa::accel_range_4xi`; 3 tests nuevos |
| `crates/gadget-ng-tree/src/let_tree.rs` | +`LT_TILE_CALLS`, +`LT_TILE_I_COUNT`, +`let_tree_tile_prof_read`, +`walk_accel_4xi`, +`walk_inner_4xi`, +`apply_leaf_soa_4xi` |
| `crates/gadget-ng-tree/src/lib.rs` | +`let_tree_tile_prof_read` en re-exports |
| `crates/gadget-ng-cli/src/engine.rs` | Loop Rayon → `par_chunks_mut(4)` + `walk_accel_4xi`; +`apply_leaf_tile_calls`, +`apply_leaf_tile_i_count`, +`tile_utilization_ratio` en HPC metrics |
| `experiments/nbody/phase16_tiled_simd/` | Scripts y configs de benchmark |

## Apéndice B: Instrucciones ASM verificadas

```
vbroadcastsd ... %ymm   — broadcast de 1 double a 4 lanes (256-bit)
vsqrtpd %ymm, %ymm      — sqrt simultánea de 4 doubles
vfmadd231pd %ymm, %ymm, %ymm — FMA de 4 doubles
vmulpd %ymm, %ymm, %ymm — multiplicación de 4 doubles
vmovupd %ymm, ...       — store de 4 doubles
```

Binario analizado: `target/release/deps/gadget_ng_tree-4a768692baf750eb`  
Símbolo: `_ZN14gadget_ng_tree7rmn_soa24accel_p16_avx2_range_4xi17hbe97f7c1758ec277E`  
Dirección relevante: `0x3fd70` (`vsqrtpd %ymm3,%ymm3`)
