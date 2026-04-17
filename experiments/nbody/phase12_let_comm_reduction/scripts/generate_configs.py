#!/usr/bin/env python3
"""
Genera configuraciones TOML para Phase 12 — Reducción de comunicación LET.

Grupos:
  A. Scaling principal   : N ∈ {4000,8000,16000} × P ∈ {2,4,8} × factor ∈ {0.0,1.2,1.4,1.6}  → 36 configs
  B. Sensitivity factor  : N=8000, P=4, factor ∈ {0.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0}      →  9 configs
  C. Validación física   : N ∈ {2000,8000} × P ∈ {2,4} × factor ∈ {0.0,1.4}                  →  8 configs

Total: 53 configs.
"""

import os
import pathlib

OUT = pathlib.Path(__file__).parent.parent / "configs"
OUT.mkdir(exist_ok=True)

TOML_TEMPLATE = """\
[simulation]
num_steps       = {num_steps}
dt              = 0.025
softening       = 0.5
particle_count  = {N}
box_size        = 20.0
seed            = 42

[initial_conditions]
kind = {{ plummer = {{ a = 2.0 }} }}

[gravity]
solver             = "barnes_hut"
theta              = 0.5
mac_softening      = "consistent"
opening_criterion  = "relative"
softened_multipoles = true
multipole_order    = 3

[performance]
use_distributed_tree      = false
force_allgather_fallback  = false
use_sfc                   = true
sfc_rebalance_interval    = 10
let_nonblocking           = true
use_let_tree              = true
let_tree_threshold        = 64
let_tree_leaf_max         = 8
let_theta_export_factor   = {factor}

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"""

generated = []


def write_config(name: str, N: int, factor: float, num_steps: int) -> str:
    path = OUT / f"{name}.toml"
    content = TOML_TEMPLATE.format(N=N, factor=factor, num_steps=num_steps)
    path.write_text(content)
    generated.append(str(path))
    return str(path)


# ── A. Scaling principal ──────────────────────────────────────────────────────
for N in [4000, 8000, 16000]:
    for P in [2, 4, 8]:
        for factor in [0.0, 1.2, 1.4, 1.6]:
            factor_str = f"{factor:.1f}".replace(".", "p")
            name = f"scale_n{N}_p{P}_f{factor_str}"
            write_config(name, N, factor, num_steps=10)

# ── B. Sensitivity factor ─────────────────────────────────────────────────────
for factor in [0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
    factor_str = f"{factor:.1f}".replace(".", "p")
    name = f"sens_n8000_p4_f{factor_str}"
    write_config(name, 8000, factor, num_steps=10)

# ── C. Validación física ──────────────────────────────────────────────────────
for N in [2000, 8000]:
    for P in [2, 4]:
        for factor in [0.0, 1.4]:
            factor_str = f"{factor:.1f}".replace(".", "p")
            name = f"valid_n{N}_p{P}_f{factor_str}"
            write_config(name, N, factor, num_steps=20)

print(f"Generados {len(generated)} configs en {OUT}")
for p in sorted(generated):
    print(f"  {os.path.basename(p)}")
