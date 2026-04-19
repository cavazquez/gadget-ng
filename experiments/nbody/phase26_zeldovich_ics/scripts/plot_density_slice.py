#!/usr/bin/env python3
"""
plot_density_slice.py — Proyección 2D del campo de densidad inicial.

Lee un snapshot JSONL de gadget-ng y genera una proyección del campo de
densidad (suma a lo largo del eje Z) como imagen 2D.

Uso:
    python3 plot_density_slice.py --snapshot results/eds_N32_ns-2_pm/snapshot_init.jsonl
    python3 plot_density_slice.py --results-dir results/
"""

import argparse
import json
import pathlib
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib no disponible; se omitirán las figuras")


def load_snapshot(path: pathlib.Path):
    """Carga snapshot JSONL → arrays (x, y, z, mass)."""
    xs, ys, zs, ms = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Formato: {"gid": ..., "mass": ..., "x": ..., "y": ..., "z": ...}
            # o anidado como {"position": [x, y, z], "mass": ...}
            m = obj.get("mass", 1.0)
            if "x" in obj:
                x, y, z = obj["x"], obj["y"], obj["z"]
            elif "position" in obj:
                pos = obj["position"]
                if isinstance(pos, list):
                    x, y, z = pos[0], pos[1], pos[2]
                else:
                    x = pos.get("x", 0)
                    y = pos.get("y", 0)
                    z = pos.get("z", 0)
            else:
                continue
            xs.append(x)
            ys.append(y)
            zs.append(z)
            ms.append(m)
    return np.array(xs), np.array(ys), np.array(zs), np.array(ms)


def density_projection(x, y, z, mass, box_size, n_grid):
    """
    Proyección 2D del campo de densidad (suma sobre el eje Z).
    Usa asignación NGP (Nearest Grid Point) para simplicidad.
    """
    dx = box_size / n_grid
    ix = np.clip((x / dx).astype(int), 0, n_grid - 1)
    iy = np.clip((y / dx).astype(int), 0, n_grid - 1)

    proj = np.zeros((n_grid, n_grid))
    for i in range(len(x)):
        proj[iy[i], ix[i]] += mass[i]
    return proj


def plot_snapshot(snap_file: pathlib.Path, out_dir: pathlib.Path,
                  box_size: float = 1.0, n_grid: int = 32, label: str = ""):
    print(f"\n  Cargando snapshot: {snap_file}")
    x, y, z, mass = load_snapshot(snap_file)

    if len(x) == 0:
        print(f"  [WARN] Snapshot vacío: {snap_file}")
        return

    n_part = len(x)
    print(f"  N = {n_part} partículas")
    print(f"  x ∈ [{x.min():.4f}, {x.max():.4f}]")
    print(f"  y ∈ [{y.min():.4f}, {y.max():.4f}]")
    print(f"  z ∈ [{z.min():.4f}, {z.max():.4f}]")

    proj = density_projection(x, y, z, mass, box_size, n_grid)
    mean_dens = proj.mean()
    delta_proj = (proj - mean_dens) / (mean_dens + 1e-30)

    print(f"  δ_rms (proyección 2D) = {delta_proj.std():.4f}")
    print(f"  δ máx = {delta_proj.max():.4f}, δ mín = {delta_proj.min():.4f}")

    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(f"Campo de densidad inicial — {label or snap_file.stem}", fontsize=12)

        # Proyección de densidad.
        ax = axes[0]
        im = ax.imshow(proj, origin="lower", cmap="inferno",
                       extent=[0, box_size, 0, box_size], aspect="equal")
        plt.colorbar(im, ax=ax, label="Masa proyectada")
        ax.set_title("Densidad proyectada (suma eje Z)")
        ax.set_xlabel("x  [L]")
        ax.set_ylabel("y  [L]")

        # Contraste de densidad.
        ax = axes[1]
        vmax = max(abs(delta_proj.max()), abs(delta_proj.min()), 1e-6)
        im2 = ax.imshow(delta_proj, origin="lower", cmap="RdBu_r",
                        extent=[0, box_size, 0, box_size], aspect="equal",
                        vmin=-vmax, vmax=vmax)
        plt.colorbar(im2, ax=ax, label="δ = (ρ - ρ̄)/ρ̄")
        ax.set_title("Contraste de densidad δ")
        ax.set_xlabel("x  [L]")
        ax.set_ylabel("y  [L]")

        fig.tight_layout()
        out_fig = out_dir / f"density_slice_{label or snap_file.stem}.png"
        fig.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f"  Figura guardada: {out_fig}")

        # También scatter de partículas (muestra aleatoria).
        n_show = min(5000, n_part)
        idx = np.random.choice(n_part, n_show, replace=False)
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        ax2.scatter(x[idx], y[idx], s=0.5, alpha=0.5, c=z[idx],
                    cmap="viridis", vmin=0, vmax=box_size)
        ax2.set_xlim(0, box_size)
        ax2.set_ylim(0, box_size)
        ax2.set_aspect("equal")
        ax2.set_title(f"Partículas (N_show={n_show}/{n_part}) — color = z")
        ax2.set_xlabel("x  [L]")
        ax2.set_ylabel("y  [L]")
        fig2.tight_layout()
        out_fig2 = out_dir / f"particles_{label or snap_file.stem}.png"
        fig2.savefig(out_fig2, dpi=150)
        plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Proyección 2D del campo de densidad")
    parser.add_argument("--snapshot", type=pathlib.Path, default=None)
    parser.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--box-size", type=float, default=1.0)
    parser.add_argument("--grid", type=int, default=32)
    args = parser.parse_args()

    if args.snapshot:
        plot_snapshot(args.snapshot, args.snapshot.parent,
                      args.box_size, args.grid, label=args.snapshot.parent.name)
    else:
        for subdir in sorted(args.results_dir.iterdir()):
            snap = subdir / "snapshot_init.jsonl"
            if not snap.exists():
                continue
            plot_snapshot(snap, subdir, args.box_size, args.grid, label=subdir.name)


if __name__ == "__main__":
    main()
