#!/usr/bin/env python3
"""
Genera configuraciones TOML para benchmarks de Fase 11 (LetTree paralelo).

Experimentos:
  A. Scaling principal: flat_let vs let_tree
     N ∈ {4000, 8000, 16000, 32000} × P ∈ {1, 2, 4, 8} × backend ∈ {flat_let, let_tree}
     = 32 configs

  B. Sensitivity: let_tree_threshold y let_tree_leaf_max (N=8000, P=2)
     threshold ∈ {32, 64, 128, 256}    → 4 configs
     leaf_max  ∈ {4, 8, 16, 32}         → 4 configs
     = 8 configs extra

  C. Validación física (num_steps=20 para comparar drift): N=2000,8000 × P=2,4
     = 8 configs (flat_let y let_tree para cada combo)

Total: ~48 configs.
"""
import os
import itertools

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
os.makedirs(OUT_DIR, exist_ok=True)

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
let_tree_threshold        = {threshold}
let_tree_leaf_max         = {leaf_max}

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"""

configs = []


def write_config(name, n, num_steps, use_let_tree, threshold=64, leaf_max=8):
    use_lt_str = "true" if use_let_tree else "false"
    content = TEMPLATE.format(
        num_steps=num_steps,
        n=n,
        use_let_tree=use_lt_str,
        threshold=threshold,
        leaf_max=leaf_max,
    )
    fname = os.path.join(OUT_DIR, f"{name}.toml")
    with open(fname, "w") as f:
        f.write(content)


# ── A. Scaling principal ──────────────────────────────────────────────────────
Ns = [4000, 8000, 16000, 32000]
Ps = [1, 2, 4, 8]
BACKENDS = ["flat_let", "let_tree"]
NUM_STEPS_BENCH = 10

for n, p, backend in itertools.product(Ns, Ps, BACKENDS):
    use_lt = backend == "let_tree"
    name = f"bench_n{n}_p{p}_{backend}"
    write_config(name, n, NUM_STEPS_BENCH, use_lt)
    configs.append((name, p, "bench"))

# ── B. Sensitivity (N=8000, P=2) ─────────────────────────────────────────────
THRESHOLDS = [32, 64, 128, 256]
LEAF_MAXES = [4, 8, 16, 32]

for thr in THRESHOLDS:
    name = f"sens_threshold_{thr}"
    write_config(name, 8000, NUM_STEPS_BENCH, use_let_tree=True, threshold=thr, leaf_max=8)
    configs.append((name, 2, "sensitivity"))

for lm in LEAF_MAXES:
    name = f"sens_leafmax_{lm}"
    write_config(name, 8000, NUM_STEPS_BENCH, use_let_tree=True, threshold=64, leaf_max=lm)
    configs.append((name, 2, "sensitivity"))

# ── C. Validación física (num_steps=20) ──────────────────────────────────────
VALID_Ns = [2000, 8000]
VALID_Ps = [2, 4]

for n, p, backend in itertools.product(VALID_Ns, VALID_Ps, BACKENDS):
    use_lt = backend == "let_tree"
    name = f"valid_n{n}_p{p}_{backend}"
    write_config(name, n, 20, use_lt)
    configs.append((name, p, "valid"))

print(f"Generados {len(configs)} configs en {os.path.abspath(OUT_DIR)}")
by_type = {}
for name, p, t in configs:
    by_type.setdefault(t, []).append(name)
for t, names in sorted(by_type.items()):
    print(f"\n  [{t}] ({len(names)} configs):")
    for n in sorted(names)[:6]:
        print(f"    {n}")
    if len(names) > 6:
        print(f"    ... (+{len(names)-6} más)")
