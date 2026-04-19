#!/usr/bin/env python3
"""
compare_pm_treepm.py — Comparación PM vs TreePM en el régimen lineal.

Superpone los espectros de potencia final (o diagnósticos delta_rms)
de los runs PM y TreePM con las mismas ICs.

Uso:
    python3 compare_pm_treepm.py --results-dir results/
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


def load_pk(path: pathlib.Path):
    ks, pks, nms = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ks.append(obj["k"])
            pks.append(obj["pk"])
            nms.append(obj["n_modes"])
    return np.array(ks), np.array(pks), np.array(nms)


def load_diagnostics_cosmo(path: pathlib.Path):
    a_arr, delta_arr = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cosmo = obj.get("cosmo")
            if cosmo is None:
                continue
            a = cosmo.get("a")
            d = cosmo.get("delta_rms")
            if a is not None and d is not None:
                a_arr.append(a)
                delta_arr.append(d)
    return np.array(a_arr), np.array(delta_arr)


def main():
    parser = argparse.ArgumentParser(description="Comparar PM vs TreePM")
    parser.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--ns", type=float, default=-2.0,
                        help="Índice espectral del caso a comparar")
    args = parser.parse_args()

    results_dir = args.results_dir

    # Buscar directorios PM y TreePM con el mismo n_s.
    ns_str = f"ns{int(args.ns)}" if args.ns == int(args.ns) else f"ns{args.ns}"
    ns_str = ns_str.replace("-", "-")

    pm_dirs = [d for d in results_dir.iterdir() if "pm" in d.name and "treepm" not in d.name and "ns" in d.name]
    treepm_dirs = [d for d in results_dir.iterdir() if "treepm" in d.name and "ns" in d.name]

    if not pm_dirs:
        print(f"[WARN] No se encontraron directorios PM en {results_dir}")
    if not treepm_dirs:
        print(f"[WARN] No se encontraron directorios TreePM en {results_dir}")

    print("\n=== Comparación PM vs TreePM — crecimiento δ_rms(a) ===")

    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("PM vs TreePM en régimen lineal (Zel'dovich ICs)", fontsize=13)

        colors_pm = ["steelblue", "navy"]
        colors_treepm = ["tomato", "darkred"]

        ax_growth = axes[0]
        ax_pk = axes[1]

        for i, d in enumerate(sorted(pm_dirs)):
            diag = d / "diagnostics.jsonl"
            if not diag.exists():
                continue
            a, delta = load_diagnostics_cosmo(diag)
            if len(a) < 2:
                continue
            color = colors_pm[i % len(colors_pm)]
            label = d.name
            ax_growth.plot(a, delta / delta[0], "o-", color=color, ms=4,
                           label=f"PM: {label}", linewidth=1.5)
            # EdS: D(a)/D(a0) = a/a0.
            a0 = a[0]
            ax_growth.plot(a, a / a0, "--", color=color, alpha=0.4, linewidth=1)

        for i, d in enumerate(sorted(treepm_dirs)):
            diag = d / "diagnostics.jsonl"
            if not diag.exists():
                continue
            a, delta = load_diagnostics_cosmo(diag)
            if len(a) < 2:
                continue
            color = colors_treepm[i % len(colors_treepm)]
            label = d.name
            ax_growth.plot(a, delta / delta[0], "s-", color=color, ms=4,
                           label=f"TreePM: {label}", linewidth=1.5)

        ax_growth.set_xlabel("Factor de escala  a")
        ax_growth.set_ylabel("δ_rms(a) / δ_rms(a₀)")
        ax_growth.set_title("Crecimiento de estructura (línea punteada = EdS)")
        ax_growth.legend(fontsize=8)
        ax_growth.grid(True, alpha=0.3)

        # P(k) final.
        for i, d in enumerate(sorted(pm_dirs)):
            pk_file = d / "pk_initial.jsonl"
            if not pk_file.exists():
                pk_file = d / "power_spectrum.jsonl"
            if not pk_file.exists():
                continue
            k, pk, _ = load_pk(pk_file)
            mask = pk > 0
            color = colors_pm[i % len(colors_pm)]
            ax_pk.loglog(k[mask], pk[mask], "o-", color=color, ms=4,
                         label=f"PM: {d.name}", linewidth=1.5)

        for i, d in enumerate(sorted(treepm_dirs)):
            pk_file = d / "pk_initial.jsonl"
            if not pk_file.exists():
                pk_file = d / "power_spectrum.jsonl"
            if not pk_file.exists():
                continue
            k, pk, _ = load_pk(pk_file)
            mask = pk > 0
            color = colors_treepm[i % len(colors_treepm)]
            ax_pk.loglog(k[mask], pk[mask], "s-", color=color, ms=4,
                         label=f"TreePM: {d.name}", linewidth=1.5)

        ax_pk.set_xlabel("k  [2π/L]")
        ax_pk.set_ylabel("P(k)  [L³]")
        ax_pk.set_title("Espectro de potencia inicial")
        ax_pk.legend(fontsize=8)
        ax_pk.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        out_fig = results_dir / "pm_vs_treepm_pk.png"
        fig.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f"  Figura guardada: {out_fig}")

    # Resumen numérico: comparar delta_rms final de PM vs TreePM.
    pm_data, treepm_data = {}, {}

    for d in sorted(pm_dirs):
        diag = d / "diagnostics.jsonl"
        if diag.exists():
            a, delta = load_diagnostics_cosmo(diag)
            if len(a) >= 2:
                pm_data[d.name] = {"a": a, "delta": delta}

    for d in sorted(treepm_dirs):
        diag = d / "diagnostics.jsonl"
        if diag.exists():
            a, delta = load_diagnostics_cosmo(diag)
            if len(a) >= 2:
                treepm_data[d.name] = {"a": a, "delta": delta}

    print("\n  {'Solver':20s}  {'δ_rms inicial':>15}  {'δ_rms final':>13}  {'Crecimiento':>12}")
    print(f"  {'-'*65}")
    for name, data in {**pm_data, **treepm_data}.items():
        d0 = data["delta"][0]
        df = data["delta"][-1]
        ratio = df / d0 if d0 > 0 else float("nan")
        print(f"  {name:40s}  {d0:15.4e}  {df:13.4e}  {ratio:12.4f}")

    print("\nNota: PM y TreePM deben producir crecimiento similar en el régimen lineal.")
    print("Discrepancias > 20% indican diferencia significativa entre solvers.")


if __name__ == "__main__":
    main()
