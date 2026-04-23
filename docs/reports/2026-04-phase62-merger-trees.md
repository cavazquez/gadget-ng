# Phase 62 — Merger Trees single-pass (historia de ensamble de masa)

**Fecha:** abril 2026  
**Crates:** `gadget-ng-analysis`, `gadget-ng-cli`  
**Archivos nuevos:**  
- `crates/gadget-ng-analysis/src/merger_tree.rs`  
- `crates/gadget-ng-cli/src/merge_tree_cmd.rs`  
- `crates/gadget-ng-physics/tests/phase62_merger_trees.rs`

---

## Contexto

Los merger trees conectan catálogos FoF de snapshots consecutivos para reconstruir
la historia de ensamble de masa de cada halo. Son la base del modelo semi-analítico
de formación galáctica y del análisis de clustering a diferentes épocas.

Phase 62 implementa un algoritmo single-pass que recorre los snapshots del más
antiguo al más reciente, siguiendo partículas por ID.

---

## Algoritmo

```
Para cada par (S_i, S_{i+1}) con S_i más antiguo:
  1. Construir mapa: particle_id → halo_id del snapshot siguiente S_{i+1}
  2. Para cada halo H en S_i:
     a. Contar votos: cuántas partículas de H aparecen en cada halo de S_{i+1}
     b. Progenitor principal → halo que recibe más votos (fracción ≥ min_shared)
     c. Mergers secundarios → halos que reciben ≥ min_shared de partículas de H
  3. Anotar en el nodo de H (snapshot i) su descendente principal
```

### Estructuras de datos

```rust
pub struct MergerTreeNode {
    pub snapshot: usize,          // índice del snapshot (0 = más antiguo)
    pub halo_id: u64,
    pub mass_msun_h: f64,
    pub n_particles: usize,
    pub x_com: [f64; 3],
    pub prog_main_id: Option<u64>, // descendente principal en snapshot+1
    pub merger_ids: Vec<u64>,      // descendentes secundarios (mergers)
    pub merger_mass_ratio: Vec<f64>,
}

pub struct MergerForest {
    pub nodes: Vec<MergerTreeNode>,
    pub roots: Vec<u64>,  // halos del snapshot más reciente (z=0)
}
```

### Tipo auxiliar para seguimiento de partículas

```rust
pub struct ParticleSnapshot {
    pub id: u64,           // ID único de la partícula (invariante entre snapshots)
    pub halo_idx: Option<usize>, // índice en el catálogo FoF (None = campo)
}
```

---

## API pública

```rust
pub fn build_merger_forest(
    catalogs: &[(Vec<FofHalo>, Vec<ParticleSnapshot>)],
    min_shared_fraction: f64,
) -> MergerForest
```

Los catálogos se ordenan del más antiguo (índice 0) al más reciente (último).
`min_shared_fraction` controla el umbral para registrar progenitores y mergers
(típico 0.1–0.3).

---

## Subcomando CLI `merge-tree`

```bash
gadget-ng merge-tree \
  --snapshots "runs/cosmo/snap_000,runs/cosmo/snap_001,runs/cosmo/snap_002" \
  --catalogs  "runs/cosmo/halos_000.jsonl,runs/cosmo/halos_001.jsonl,runs/cosmo/halos_002.jsonl" \
  --out       runs/cosmo/merger_tree.json \
  --min-shared 0.1
```

Escribe `merger_tree.json` con todos los nodos y la lista de raíces.

---

## Formato de salida (JSON)

```json
{
  "nodes": [
    {
      "snapshot": 0,
      "halo_id": 0,
      "mass_msun_h": 1.2e14,
      "n_particles": 256,
      "x_com": [50.1, 49.8, 50.3],
      "prog_main_id": 0,
      "merger_ids": [],
      "merger_mass_ratio": []
    }
  ],
  "roots": [0, 1, 2]
}
```

---

## Tests

| Test | Descripción |
|------|-------------|
| `phase62_trivial_no_mergers` | 1 halo con las mismas partículas en 2 snapshots → progenitor único, sin mergers |
| `phase62_binary_merger` | 2 halos en snap_0 → 1 en snap_1 → ambos apuntan al mismo descendente |
| `phase62_roundtrip_json` | Serializar/deserializar `MergerForest` preserva snapshot, halo_id y masa |
| `phase62_single_snapshot_no_progenitors` | Un solo snapshot → raíces definidas, `prog_main_id = None` para todos |

---

## Notas de diseño

- La relación `prog_main_id` apunta al **descendente** (snapshot+1), no al progenitor.
  El nombre refleja que es el principal entre los progenitores del descendente.
- El algoritmo es O(N_partículas × N_snapshots), adecuado para corridas medianas.
- Para N > 10⁷ partículas se requiere FoF paralelo (Phase 61) para construir los catálogos.
