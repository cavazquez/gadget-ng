#!/usr/bin/env python3
"""
Genera configs TOML para benchmarks de Fase 8 (strong/weak scaling).

Configura:
  - Plummer a/ε=2 (medio, buen balance), distribución uniforme
  - N fijo N=2000 para strong scaling (1, 2, 4, 8 rangos)
  - N escalable [500, 1000, 2000, 4000] para weak scaling (N/P=500 por rango)
  - Backends: allgather (fallback), sfc_let (default)

Uso:
  python3 generate_configs.py [--out-dir <dir>]
"""
import os, sys, argparse

TEMPLATE_ALLGATHER = """\
[simulation]
dt            = 0.025
num_steps     = 100
softening     = 0.5
particle_count= {n}
box_size      = 20.0
seed          = 42

[initial_conditions]
kind = {{ plummer = {{ a = 1.0 }} }}

[gravity]
solver              = "barnes_hut"
theta               = 0.7
multipole_order     = 3
opening_criterion   = "relative"
err_tol_force_acc   = 0.005
softened_multipoles = true
mac_softening       = "consistent"

[performance]
use_distributed_tree      = false
force_allgather_fallback  = true

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"""

TEMPLATE_SFC_LET = """\
[simulation]
dt            = 0.025
num_steps     = 100
softening     = 0.5
particle_count= {n}
box_size      = 20.0
seed          = 42

[initial_conditions]
kind = {{ plummer = {{ a = 1.0 }} }}

[gravity]
solver              = "barnes_hut"
theta               = 0.7
multipole_order     = 3
opening_criterion   = "relative"
err_tol_force_acc   = 0.005
softened_multipoles = true
mac_softening       = "consistent"

[performance]
use_distributed_tree      = false
force_allgather_fallback  = false

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "config"))
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Strong scaling: N=2000 fijo, 1/2/4/8 rangos.
    n_strong = 2000
    for ranks in [1, 2, 4, 8]:
        for backend, tmpl in [("allgather", TEMPLATE_ALLGATHER), ("sfc_let", TEMPLATE_SFC_LET)]:
            fname = f"strong_N{n_strong}_{backend}_R{ranks}.toml"
            content = tmpl.format(n=n_strong)
            with open(os.path.join(out_dir, fname), "w") as f:
                f.write(content)
            print(f"  {fname}")

    # Weak scaling: N/P=500 → N = 500*P, para P=1,2,4,8.
    n_per_rank = 500
    for ranks in [1, 2, 4, 8]:
        n = n_per_rank * ranks
        for backend, tmpl in [("allgather", TEMPLATE_ALLGATHER), ("sfc_let", TEMPLATE_SFC_LET)]:
            fname = f"weak_N{n}_{backend}_R{ranks}.toml"
            content = tmpl.format(n=n)
            with open(os.path.join(out_dir, fname), "w") as f:
                f.write(content)
            print(f"  {fname}")

    print(f"\n{len(os.listdir(out_dir))} configs generados en {out_dir}")


if __name__ == "__main__":
    main()
