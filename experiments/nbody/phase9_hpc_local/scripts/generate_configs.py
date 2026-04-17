#!/usr/bin/env python3
"""
Genera configuraciones TOML para los benchmarks de Fase 9.

Experimentos:
  Strong scaling: N=2000, 4000, 8000 × P=1,2,4,8 × backends {allgather, blocking, overlap}
  Weak scaling:   N/P=500 y N/P=1000 × P=1,2,4,8 × backends {blocking, overlap}

Total aprox: 3*4*3 + 2*4*2 = 36 + 16 = 52 configs
"""
import os
import itertools

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
os.makedirs(OUT_DIR, exist_ok=True)

# Parámetros físicos comunes
EPS = 0.1
G = 1.0
DT = 0.025
NUM_STEPS = 5      # pocos pasos para benchmark de timing
THETA = 0.5
SEED = 42

TEMPLATE = """\
[simulation]
particle_count = {n}
num_steps = {num_steps}
dt = {dt}
seed = {seed}
box_size = 20.0
softening = 0.5
integrator = "leapfrog"

[initial_conditions]
kind = {{ plummer = {{ a = 1.0 }} }}

[gravity]
solver = "barnes_hut"
theta = {theta}
multipole_order = 3
opening_criterion = "relative"
err_tol_force_acc = 0.005
softened_multipoles = true
mac_softening = "consistent"

[performance]
deterministic = false
use_distributed_tree = true
use_sfc = false
sfc_rebalance_interval = 1
halo_factor = 0.5
force_allgather_fallback = {allgather}
let_nonblocking = {nonblocking}

[output]
snapshot_format = "jsonl"
checkpoint_interval = 0
snapshot_interval = 0
"""

def gen_config(n, backend, num_steps=NUM_STEPS, label=""):
    """Genera y guarda un archivo de config TOML."""
    allgather = "true" if backend == "allgather" else "false"
    nonblocking = "true" if backend == "overlap" else "false"
    tag = f"N{n}_{backend}{label}"
    content = TEMPLATE.format(
        n=n,
        num_steps=num_steps,
        dt=DT,
        seed=SEED,
        theta=THETA,
        eps=EPS,
        allgather=allgather,
        nonblocking=nonblocking,
    )
    path = os.path.join(OUT_DIR, f"{tag}.toml")
    with open(path, "w") as f:
        f.write(content)
    return tag, path

generated = []

# ── Strong scaling: N fijo, P=1..8 ──────────────────────────────────────────
# (el mismo config se usa con distintos P al lanzar con mpirun -n P)
for n in [2000, 4000, 8000]:
    for backend in ["allgather", "blocking", "overlap"]:
        tag, path = gen_config(n, backend)
        generated.append((tag, path, "strong"))

# ── Weak scaling: N/P constante ──────────────────────────────────────────────
# Para N/P=500: N=500(P=1), 1000(P=2), 2000(P=4), 4000(P=8)
# Para N/P=1000: N=1000(P=1), 2000(P=2), 4000(P=4), 8000(P=8)
for np_ratio in [500, 1000]:
    for np in [1, 2, 4, 8]:
        n = np_ratio * np
        for backend in ["blocking", "overlap"]:
            tag, path = gen_config(n, backend, label=f"_weak{np_ratio}_P{np}")
            generated.append((tag, path, f"weak{np_ratio}_P{np}"))

print(f"Generados {len(generated)} archivos de configuración en {OUT_DIR}/")
for tag, path, kind in generated:
    print(f"  [{kind}] {tag}")
