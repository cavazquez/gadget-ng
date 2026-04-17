#!/usr/bin/env python3
"""
Phase 14 — SoA + SIMD para kernels calientes
Genera configuraciones TOML para comparar baseline (AoS) vs SoA+SIMD:

Grupos:
  - profiling_p1:  N ∈ {4000, 8000, 16000}, P=1, 10 pasos  → tabla top hotspots
  - scaling_mpi:   N ∈ {8000, 16000},       P ∈ {2, 4},    10 pasos
  - validation:    N ∈ {2000, 8000},         P ∈ {1, 2, 4}, 20 pasos

Cada configuración se corre con dos binarios:
  - baseline (sin feature simd)
  - soa_simd (con feature simd = RmnSoa + kernel fusionado + Rayon)

Total configs: 7 + 8 + 18 = 33 (×2 binarios = 66 runs)
"""

import os
import itertools

OUT_DIR = "configs"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_TEMPLATE = """
[simulation]
num_steps      = {num_steps}
dt             = 0.025
softening      = 0.5
particle_count = {n_particles}
box_size       = 20.0
seed           = 42

[initial_conditions]
kind = {{ plummer = {{ a = 2.0 }} }}

[gravity]
solver              = "barnes_hut"
theta               = 0.5
mac_softening       = "consistent"
opening_criterion   = "relative"
softened_multipoles = true
multipole_order     = 3

[performance]
use_distributed_tree     = false
force_allgather_fallback = false
use_sfc                  = true
sfc_rebalance_interval   = 5
let_nonblocking          = true
use_let_tree             = {use_let_tree}
let_tree_threshold       = 64
let_tree_leaf_max        = 8
let_theta_export_factor  = 0.0
sfc_kind                 = "morton"

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"""

configs = []

# ── Profiling P=1: identificar hotspots por N ─────────────────────────────────
for n in [4000, 8000, 16000]:
    # Path let_tree (default)
    configs.append({
        "group": "profiling_p1",
        "n": n, "p": 1,
        "num_steps": 10,
        "use_let_tree": "true",
    })

# ── Scaling MPI ───────────────────────────────────────────────────────────────
for n, p in itertools.product([8000, 16000], [2, 4]):
    configs.append({
        "group": "scaling_mpi",
        "n": n, "p": p,
        "num_steps": 10,
        "use_let_tree": "true",
    })

# ── Validación física ─────────────────────────────────────────────────────────
for n, p in itertools.product([2000, 8000], [1, 2, 4]):
    configs.append({
        "group": "validation",
        "n": n, "p": p,
        "num_steps": 20,
        "use_let_tree": "true",
    })

print(f"Total configs: {len(configs)}")

for cfg in configs:
    name = f"{cfg['group']}_N{cfg['n']}_P{cfg['p']}"
    toml = BASE_TEMPLATE.format(
        num_steps    = cfg["num_steps"],
        n_particles  = cfg["n"],
        use_let_tree = cfg["use_let_tree"],
    )
    path = os.path.join(OUT_DIR, f"{name}.toml")
    with open(path, "w") as f:
        f.write(toml.strip() + "\n")

print(f"Generated {len(configs)} configs in {OUT_DIR}/")
