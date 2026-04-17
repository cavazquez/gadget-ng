#!/usr/bin/env python3
"""
Genera configuraciones TOML para benchmarks de Fase 10 (LET-tree).

Experimento: comparar `flat_let` vs `let_tree` en:
  - N = 2000, 4000, 8000, 16000
  - P = 1, 2, 4
  - backend = flat_let (use_let_tree=false) | let_tree (use_let_tree=true)

Totales: 4 N × 3 P × 2 backends = 24 configs.
"""
import os
import itertools

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
os.makedirs(OUT_DIR, exist_ok=True)

Ns = [2000, 4000, 8000, 16000]
Ps = [1, 2, 4]
BACKENDS = ["flat_let", "let_tree"]

# Número de pasos: suficiente para obtener timings estables sin ejecutar demasiado.
NUM_STEPS = 10

TEMPLATE = """\
[simulation]
num_steps       = {num_steps}
dt              = 0.025
softening       = 0.5
particle_count  = {n}
box_size        = 20.0
seed            = 42

[initial_conditions]
kind = {{ plummer = {{ a = 1.0 }} }}

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
use_let_tree              = {use_let_tree}
let_tree_threshold        = 32
let_tree_leaf_max         = 8

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"""


def make_config(n, p, backend):
    use_let_tree = "true" if backend == "let_tree" else "false"
    run_name = f"n{n}_p{p}_{backend}"
    content = TEMPLATE.format(
        num_steps=NUM_STEPS,
        n=n,
        use_let_tree=use_let_tree,
        run_name=run_name,
    )
    fname = os.path.join(OUT_DIR, f"{run_name}.toml")
    with open(fname, "w") as f:
        f.write(content)
    return run_name


configs = []
for n, p, backend in itertools.product(Ns, Ps, BACKENDS):
    name = make_config(n, p, backend)
    configs.append((name, n, p, backend))

print(f"Generated {len(configs)} configs in {os.path.abspath(OUT_DIR)}")
for name, n, p, backend in sorted(configs):
    print(f"  {name}")
