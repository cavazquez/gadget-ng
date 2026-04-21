#!/usr/bin/env python3
"""Phase 36 — genera las 5 figuras obligatorias de validación.

Entradas:
- `<target>/phase36/per_snapshot_metrics.json` (matriz in-process)
- `<out>/cli_evidence.json` (opcional, pase CLI)

Salidas en `<figures_dir>/`:
1. `pk_measured_corrected_theory.png`
2. `ratio_raw_vs_corr.png`
3. `log_error_before_after.png`
4. `snapshot_evolution.png`
5. `resolution_comparison.png`
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_matrix(path: Path):
    with path.open() as f:
        return json.load(f)


def pick(snapshots, n=None, seed=None, ic_kind=None, a=None):
    out = []
    for s in snapshots:
        if n is not None and s["n"] != n:
            continue
        if seed is not None and s["seed"] != seed:
            continue
        if ic_kind is not None and s["ic_kind"] != ic_kind:
            continue
        if a is not None and abs(s["a_target"] - a) > 1e-9:
            continue
        out.append(s)
    return out


def safe_median_abs_log_ratio(xs, ys):
    vals = [abs(math.log10(x / y)) for x, y in zip(xs, ys)
            if x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)]
    if not vals:
        return float("nan")
    return float(np.median(vals))


def fig1_pk(data, out_path: Path):
    """P_m · (box_mpc_h)³, P_corr y P_ref vs k para cada N en a=0.02."""
    ns = sorted({s["n"] for s in data["snapshots"]})
    fig, axes = plt.subplots(1, len(ns), figsize=(6 * len(ns), 5), sharey=True)
    if len(ns) == 1:
        axes = [axes]

    box_mpc_h = 100.0
    unit = box_mpc_h ** 3

    for ax, n in zip(axes, ns):
        for a, style in [(0.02, "o-"), (0.10, "s--")]:
            snaps = pick(data["snapshots"], n=n, ic_kind="2lpt", a=a)
            if not snaps:
                continue
            s = snaps[0]
            ks = np.array(s["ks_hmpc"])
            pm_internal = np.array(s["pk_measured_internal"])
            pm_physical = pm_internal * unit
            pc = np.array(s["pk_corrected_mpc_h3"])
            pr = np.array(s["pk_reference_mpc_h3"])
            ax.plot(ks, pm_physical, style, color="tab:blue", alpha=0.7,
                    label=f"$P_m$·$L^3$ (a={a:.2f})")
            ax.plot(ks, pc, style, color="tab:orange",
                    label=f"$P_c$ (a={a:.2f})")
            ax.plot(ks, pr, style, color="tab:green",
                    label=f"$P_r$ (a={a:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$k$ [h/Mpc]")
        ax.set_title(f"$N={n}^3$, 2LPT, seed=42")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("$P(k)$ [(Mpc/h)$^3$]")
    fig.suptitle("Phase 36 — $P_m$, $P_c$ y $P_\\mathrm{ref}$ (seed=42)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig2_ratio(data, out_path: Path):
    """P_m/P_ref y P_c/P_ref vs k en dos subplots (a=0.02, N=32 y N=64)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    ax1, ax2 = axes
    box_mpc_h = 100.0
    unit = box_mpc_h ** 3
    for n, color in [(32, "tab:blue"), (64, "tab:red")]:
        snaps = pick(data["snapshots"], n=n, ic_kind="2lpt", a=0.02)
        for s in snaps:
            ks = np.array(s["ks_hmpc"])
            pm = np.array(s["pk_measured_internal"]) * unit
            pc = np.array(s["pk_corrected_mpc_h3"])
            pr = np.array(s["pk_reference_mpc_h3"])
            ax1.plot(ks, pm / pr, "o-", color=color, alpha=0.5,
                     label=f"N={n}, seed={s['seed']}")
            ax2.plot(ks, pc / pr, "o-", color=color, alpha=0.5,
                     label=f"N={n}, seed={s['seed']}")
    for ax, ttl in zip(axes, ["$P_m \\cdot L^3 / P_\\mathrm{ref}$ (crudo)",
                              "$P_c / P_\\mathrm{ref}$ (corregido)"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$k$ [h/Mpc]")
        ax.set_title(ttl)
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Ratio")
    fig.suptitle("Phase 36 — ratios crudo vs corregido (a = 0.02, 2LPT)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig3_log_error_bars(data, out_path: Path):
    """Barras med|log10 ratio| antes/después por (N, a, ic_kind)."""
    configs = [
        ("N=32, 2LPT", 32, "2lpt"),
        ("N=32, 1LPT", 32, "1lpt"),
        ("N=64, 2LPT", 64, "2lpt"),
    ]
    a_vals = sorted({s["a_target"] for s in data["snapshots"]})
    width = 0.35
    fig, axes = plt.subplots(1, len(configs), figsize=(5.5 * len(configs), 5),
                             sharey=True)
    if len(configs) == 1:
        axes = [axes]
    for ax, (label, n, ic) in zip(axes, configs):
        raws = []
        corrs = []
        for a in a_vals:
            snaps = pick(data["snapshots"], n=n, ic_kind=ic, a=a)
            raw = np.mean([s["median_abs_log10_err_raw"] for s in snaps])
            corr = np.mean([s["median_abs_log10_err_corrected"] for s in snaps])
            raws.append(raw)
            corrs.append(corr)
        x = np.arange(len(a_vals))
        ax.bar(x - width / 2, raws, width, label="crudo", color="tab:blue")
        ax.bar(x + width / 2, corrs, width, label="corregido", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels([f"a={a:.2f}" for a in a_vals])
        ax.set_title(label)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("median $|\\log_{10}(P/P_\\mathrm{ref})|$")
    fig.suptitle("Phase 36 — error antes/después de `pk_correction`")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig4_snapshot_evolution(data, out_path: Path):
    """P_c/P_ref vs k para a=0.02, 0.05, 0.10 (drift no-lineal)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    a_vals = [0.02, 0.05, 0.10]
    colors = {0.02: "tab:green", 0.05: "tab:orange", 0.10: "tab:red"}
    for ax, n in zip(axes, [32, 64]):
        for a in a_vals:
            snaps = pick(data["snapshots"], n=n, ic_kind="2lpt", a=a)
            for s in snaps:
                ks = np.array(s["ks_hmpc"])
                pc = np.array(s["pk_corrected_mpc_h3"])
                pr = np.array(s["pk_reference_mpc_h3"])
                ax.plot(ks, pc / pr, "-", color=colors[a], alpha=0.4)
            # Leyenda: una sola línea por a
            ax.plot([], [], "-", color=colors[a], label=f"a={a:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.7)
        ax.set_xlabel("$k$ [h/Mpc]")
        ax.set_title(f"$N={n}^3$, 2LPT, 3 seeds")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("$P_c / P_\\mathrm{ref}$")
    fig.suptitle(
        "Phase 36 — deriva de $P_c/P_\\mathrm{ref}$ entre snapshots\n"
        "(a ≫ a_init entra en régimen no-lineal por convención de IC de gadget-ng)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig5_resolution(data, out_path: Path):
    """N=32 vs N=64 en a=0.02 (P_c/P_ref) + curva de mejora relativa."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    colors = {32: "tab:blue", 64: "tab:red"}
    for n in [32, 64]:
        snaps = pick(data["snapshots"], n=n, ic_kind="2lpt", a=0.02)
        for s in snaps:
            ks = np.array(s["ks_hmpc"])
            pc = np.array(s["pk_corrected_mpc_h3"])
            pr = np.array(s["pk_reference_mpc_h3"])
            ax1.plot(ks, pc / pr, "o-", color=colors[n], alpha=0.5,
                     label=f"N={n}, seed={s['seed']}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.axhline(1.0, color="k", linestyle=":", alpha=0.5)
    ax1.set_xlabel("$k$ [h/Mpc]")
    ax1.set_ylabel("$P_c / P_\\mathrm{ref}$")
    ax1.set_title("Comparación de resoluciones (a=0.02)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    # Subplot 2: med |log10| por N (barras).
    ns = [32, 64]
    med_raw, med_corr = [], []
    for n in ns:
        snaps = pick(data["snapshots"], n=n, ic_kind="2lpt", a=0.02)
        med_raw.append(np.mean([s["median_abs_log10_err_raw"] for s in snaps]))
        med_corr.append(np.mean([s["median_abs_log10_err_corrected"] for s in snaps]))
    x = np.arange(len(ns))
    width = 0.35
    ax2.bar(x - width / 2, med_raw, width, label="crudo", color="tab:blue")
    ax2.bar(x + width / 2, med_corr, width, label="corregido", color="tab:orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N={n}" for n in ns])
    ax2.set_yscale("log")
    ax2.set_ylabel("median $|\\log_{10}|$")
    ax2.set_title("Reducción del error por N (a=0.02)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    fig.suptitle("Phase 36 — comparación entre resoluciones")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Genera las 5 figuras de Phase 36.")
    ap.add_argument("--matrix-json", required=True,
                    help="Ruta a per_snapshot_metrics.json (tests Rust).")
    ap.add_argument("--cli-json", default=None,
                    help="Opcional: cli_evidence.json para fig 1-2.")
    ap.add_argument("--out-dir", required=True,
                    help="Carpeta donde guardar los PNG.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_matrix(Path(args.matrix_json))

    fig1_pk(data, out_dir / "pk_measured_corrected_theory.png")
    fig2_ratio(data, out_dir / "ratio_raw_vs_corr.png")
    fig3_log_error_bars(data, out_dir / "log_error_before_after.png")
    fig4_snapshot_evolution(data, out_dir / "snapshot_evolution.png")
    fig5_resolution(data, out_dir / "resolution_comparison.png")

    if args.cli_json:
        cli = json.loads(Path(args.cli_json).read_text())
        fig, ax = plt.subplots(figsize=(7, 5))
        ks = [b["k_hmpc"] for b in cli["bins"]]
        pm = [b["pk_measured_internal"] * 100 ** 3 for b in cli["bins"]]
        pc = [b["pk_corrected_mpc_h3"] for b in cli["bins"]]
        pr = [b["pk_reference_mpc_h3"] for b in cli["bins"]]
        ax.plot(ks, pm, "o-", label="$P_m \\cdot L^3$ (CLI)")
        ax.plot(ks, pc, "s-", label="$P_c$ (CLI)")
        ax.plot(ks, pr, "--", color="k", label="$P_\\mathrm{ref}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$k$ [h/Mpc]")
        ax.set_ylabel("$P(k)$ [(Mpc/h)$^3$]")
        ax.set_title(f"Phase 36 — pase CLI (N={cli['meta']['n']}, "
                     f"a={cli['meta']['a_snapshot']:.3f})")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "cli_evidence.png", dpi=140)
        plt.close(fig)

    print(f"[phase36] Figuras generadas en {out_dir}")


if __name__ == "__main__":
    main()
