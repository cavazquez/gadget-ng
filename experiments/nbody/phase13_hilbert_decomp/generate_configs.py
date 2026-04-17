#!/usr/bin/env python3
"""
Phase 13 — Hilbert 3D Domain Decomposition
Genera 34 configuraciones TOML para comparar Morton vs Hilbert:
  - Scaling:     N×P×sfc_kind (3N × 3P × 2curvas = 18 configs, 10 pasos)
  - Sensitivity: N=16000 × P×sfc_kind (4P × 2curvas = 8 configs, 10 pasos)
  - Validación:  N×P×sfc_kind (2N × 2P × 2curvas = 8 configs, 20 pasos)
Total: 34 configs
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
use_distributed_tree    = false
force_allgather_fallback = false
use_sfc                 = true
sfc_rebalance_interval  = 5
let_nonblocking         = true
use_let_tree            = true
let_tree_threshold      = 64
let_tree_leaf_max       = 8
let_theta_export_factor = 0.0
sfc_kind                = "{sfc_kind}"

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"""

configs = []

# ── Scaling: N ∈ {8000,16000,32000} × P ∈ {2,4,8} × sfc_kind ∈ {morton,hilbert}
for n, p, kind in itertools.product(
    [8000, 16000, 32000],
    [2, 4, 8],
    ["morton", "hilbert"],
):
    configs.append({
        "group":       "scaling",
        "n":           n,
        "p":           p,
        "sfc_kind":    kind,
        "num_steps":   10,
    })

# ── Sensitivity: N=16000 × P ∈ {1,2,4,8} × sfc_kind
for p, kind in itertools.product([1, 2, 4, 8], ["morton", "hilbert"]):
    configs.append({
        "group":       "sensitivity_p",
        "n":           16000,
        "p":           p,
        "sfc_kind":    kind,
        "num_steps":   10,
    })

# ── Validación: N ∈ {2000,8000} × P ∈ {2,4} × sfc_kind
for n, p, kind in itertools.product([2000, 8000], [2, 4], ["morton", "hilbert"]):
    configs.append({
        "group":       "valid",
        "n":           n,
        "p":           p,
        "sfc_kind":    kind,
        "num_steps":   20,
    })

print(f"Total configs: {len(configs)}")

for cfg in configs:
    name = f"{cfg['group']}_N{cfg['n']}_P{cfg['p']}_{cfg['sfc_kind']}"
    out_dir = f"results/{name}"
    toml = BASE_TEMPLATE.format(
        num_steps   = cfg["num_steps"],
        n_particles = cfg["n"],
        out_dir     = out_dir,
        sfc_kind    = cfg["sfc_kind"],
    )
    path = os.path.join(OUT_DIR, f"{name}.toml")
    with open(path, "w") as f:
        f.write(toml.strip() + "\n")

print(f"Generated {len(configs)} configs in {OUT_DIR}/")
