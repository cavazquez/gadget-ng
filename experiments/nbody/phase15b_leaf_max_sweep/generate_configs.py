#!/usr/bin/env python3
"""
Phase 15b — Sweep de leaf_max para aprovechamiento SIMD.

Genera configs TOML para barrer let_tree_leaf_max ∈ {8, 16, 32, 64}
cruzado con N ∈ {8000, 16000} y P ∈ {2, 4}.

Cada config se corre con dos variantes:
  - p14_fused:    kernel fusionado (accel_soa_scalar, auto-vec, xmm)
  - p15_explicit: intrinsics AVX2 explícitos (accel_p15_avx2_range, ymm)

Total: 4 × 2 × 2 = 16 configs × 2 variantes = 32 runs.
"""

import os
import itertools

OUT_DIR = "configs"
os.makedirs(OUT_DIR, exist_ok=True)

TEMPLATE = """
[simulation]
num_steps      = 10
dt             = 0.025
softening      = 0.5
particle_count = {n}
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
let_tree_leaf_max        = {leaf_max}
let_theta_export_factor  = 0.0
sfc_kind                 = "morton"

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"""

LEAF_MAX_VALUES = [8, 16, 32, 64]
N_VALUES = [8000, 16000]
P_VALUES = [2, 4]

configs = []
for leaf_max, n, p in itertools.product(LEAF_MAX_VALUES, N_VALUES, P_VALUES):
    configs.append({"leaf_max": leaf_max, "n": n, "p": p})

print(f"Total configs: {len(configs)} (x2 variantes = {2*len(configs)} runs)")

for cfg in configs:
    name = f"lm{cfg['leaf_max']}_N{cfg['n']}_P{cfg['p']}"
    toml = TEMPLATE.format(n=cfg["n"], leaf_max=cfg["leaf_max"])
    path = os.path.join(OUT_DIR, f"{name}.toml")
    with open(path, "w") as f:
        f.write(toml.strip() + "\n")

print(f"Generated {len(configs)} configs in {OUT_DIR}/")
