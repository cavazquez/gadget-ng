#!/usr/bin/env python3
"""
Phase 15 — Explicit AVX2 SIMD para el kernel monopolar
Genera configuraciones TOML para comparar:
  - p14_fused:    kernel fusionado Fase 14 (auto-vectorización)
  - p15_explicit: kernel two-pass Fase 15 (intrinsics AVX2 explícitos)

Grupos:
  - scaling_mpi:  N ∈ {8000, 16000},       P ∈ {2, 4}, 10 pasos
  - large_mpi:    N = 32000,               P ∈ {4, 8},  10 pasos
  - validation:   N ∈ {2000, 8000},        P ∈ {2, 4},  20 pasos

Ambos binarios usan --features mpi,simd (RmnSoa + LetTree + Rayon).
La diferencia es que p15_explicit tiene el dispatch actualizado a accel_p15_avx2_range.
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
use_let_tree             = true
let_tree_threshold       = 64
let_tree_leaf_max        = 8
let_theta_export_factor  = 0.0
sfc_kind                 = "morton"

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"""

configs = []

# ── Scaling MPI: N=8k/16k, P=2/4 (benchmark principal) ───────────────────────
for n, p in itertools.product([8000, 16000], [2, 4]):
    configs.append({
        "group": "scaling_mpi",
        "n": n, "p": p,
        "num_steps": 10,
    })

# ── Large MPI: N=32k, P=4/8 (régimen de cluster) ─────────────────────────────
for n, p in itertools.product([32000], [4, 8]):
    configs.append({
        "group": "large_mpi",
        "n": n, "p": p,
        "num_steps": 10,
    })

# ── Validación física ─────────────────────────────────────────────────────────
for n, p in itertools.product([2000, 8000], [2, 4]):
    configs.append({
        "group": "validation",
        "n": n, "p": p,
        "num_steps": 20,
    })

print(f"Total configs: {len(configs)} (x2 variantes = {2*len(configs)} runs)")

for cfg in configs:
    name = f"{cfg['group']}_N{cfg['n']}_P{cfg['p']}"
    toml = BASE_TEMPLATE.format(
        num_steps   = cfg["num_steps"],
        n_particles = cfg["n"],
    )
    path = os.path.join(OUT_DIR, f"{name}.toml")
    with open(path, "w") as f:
        f.write(toml.strip() + "\n")

print(f"Generated {len(configs)} configs in {OUT_DIR}/")
