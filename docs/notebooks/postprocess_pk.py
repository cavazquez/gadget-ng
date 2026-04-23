#!/usr/bin/env python3
"""
postprocess_pk.py — Post-proceso de P(k) por snapshot.

Lee los archivos de análisis in-situ (insitu_NNNNNN.json) y genera:
  - pk_evolution.json  : P(k,z) para todos los snapshots
  - pk_z0.png          : comparación P(k)(z=0) vs CAMB lineal
  - pk_evolution.png   : P(k) a múltiples redshifts

Uso:
  python3 docs/notebooks/postprocess_pk.py \\
    --snapshots runs/production_256/frames \\
    --insitu    runs/production_256/insitu \\
    --out       runs/production_256/analysis/pk_evolution.json
"""

import argparse
import json
import sys
from pathlib import Path

# ── Dependencias opcionales ────────────────────────────────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARN: numpy no encontrado — análisis limitado", file=sys.stderr)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARN: matplotlib no encontrado — sin figuras", file=sys.stderr)


def load_insitu_files(insitu_dir: Path) -> list[dict]:
    """Carga todos los archivos insitu_NNNNNN.json en orden."""
    files = sorted(insitu_dir.glob("insitu_*.json"))
    data = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
            d["_file"] = str(f)
            data.append(d)
    return data


def extract_pk(record: dict) -> tuple[list, list] | None:
    """Extrae k, P(k) de un registro in-situ."""
    if "pk" not in record:
        return None
    pk = record["pk"]
    if isinstance(pk, dict) and "k" in pk and "power" in pk:
        return pk["k"], pk["power"]
    return None


def summarize_pk_evolution(insitu_data: list[dict]) -> dict:
    """Genera resumen de P(k,z) para todos los snapshots."""
    result = {
        "snapshots": [],
        "metadata": {
            "n_snapshots": len(insitu_data),
            "description": "P(k) medido desde análisis in-situ de gadget-ng"
        }
    }

    for rec in insitu_data:
        entry = {
            "step": rec.get("step", -1),
            "a": rec.get("a", None),
            "z": (1.0 / rec["a"] - 1.0) if rec.get("a") else None,
        }
        pk_data = extract_pk(rec)
        if pk_data is not None:
            k, pk = pk_data
            entry["k_h_mpc"] = k
            entry["pk_mpc3_h3"] = pk
        result["snapshots"].append(entry)

    return result


def plot_pk_evolution(pk_summary: dict, out_dir: Path) -> None:
    """Genera figuras de P(k) si matplotlib está disponible."""
    if not HAS_MPL or not HAS_NUMPY:
        return

    snapshots = [s for s in pk_summary["snapshots"] if "k_h_mpc" in s]
    if not snapshots:
        print("WARN: no hay datos de P(k) para graficar", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 0.85, len(snapshots)))

    for i, snap in enumerate(snapshots):
        k = np.array(snap["k_h_mpc"])
        pk = np.array(snap["pk_mpc3_h3"])
        z = snap.get("z", i)
        label = f"z={z:.1f}" if z is not None else f"snap {i}"
        ax.loglog(k, pk, color=colors[i], alpha=0.8, label=label, lw=1.5)

    ax.set_xlabel(r"$k$ [$h$/Mpc]", fontsize=12)
    ax.set_ylabel(r"$P(k)$ [Mpc/$h$]³", fontsize=12)
    ax.set_title("Espectro de potencia P(k,z) — gadget-ng N=256³", fontsize=13)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "pk_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {out_dir}/pk_evolution.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-proceso P(k) gadget-ng")
    parser.add_argument("--snapshots", type=Path, default=Path("runs/production_256/frames"),
                        help="Directorio con frames/snapshots")
    parser.add_argument("--insitu", type=Path, default=Path("runs/production_256/insitu"),
                        help="Directorio con archivos insitu_*.json")
    parser.add_argument("--out", type=Path, default=Path("runs/production_256/analysis/pk_evolution.json"),
                        help="Archivo JSON de salida")
    args = parser.parse_args()

    if not args.insitu.exists():
        print(f"ERROR: directorio insitu no encontrado: {args.insitu}", file=sys.stderr)
        return 1

    print(f"Cargando archivos in-situ desde {args.insitu}...")
    insitu_data = load_insitu_files(args.insitu)
    if not insitu_data:
        print("WARN: no se encontraron archivos insitu_*.json", file=sys.stderr)
        return 0

    print(f"  {len(insitu_data)} snapshots cargados")

    pk_summary = summarize_pk_evolution(insitu_data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(pk_summary, fp, indent=2)
    print(f"  Guardado: {args.out}")

    out_dir = args.out.parent
    plot_pk_evolution(pk_summary, out_dir)

    print("postprocess_pk.py completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
