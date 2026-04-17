#!/usr/bin/env python3
"""plot_phase5.py — Figuras paper-grade de la Fase 5.

Produce:
    plots/energy_drift_timeseries.png     — |ΔE/E| vs t, 4 subplots (dist), N=1000, 5 variantes
    plots/momentum_angmom_timeseries.png  — |Δp| y |ΔL| vs t, 4 subplots × 2 métricas
    plots/pareto_precision_cost.png       — error medio de fuerza vs coste/step (figura central)
    plots/local_vs_global.png             — scatter error local vs drift global
    plots/opened_nodes_profile.png        — barras de nodos abiertos por variante

Lee:
    results/bh_mac_softening.csv
    results/phase5_summary.csv
    results/timeseries/<tag>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
PLOTS_DIR = EXP_DIR / "plots"
TIMESERIES_DIR = RESULTS_DIR / "timeseries"

VARIANTS_ORDER = [
    "V1_geom_bare",
    "V2_geom_soft",
    "V3_rel_bare",
    "V4_rel_soft",
    "V5_rel_soft_consistent",
]

VARIANT_LABELS = {
    "V1_geom_bare":           "V1: geom + bare",
    "V2_geom_soft":           "V2: geom + soft",
    "V3_rel_bare":            "V3: rel + bare",
    "V4_rel_soft":            "V4: rel + soft",
    "V5_rel_soft_consistent": "V5: rel + soft + MAC-cons",
}

DISTRIBUTIONS = ["plummer_a1", "plummer_a2", "plummer_a6", "uniform"]
DIST_LABELS = {
    "plummer_a1": r"Plummer $a/\varepsilon=1$",
    "plummer_a2": r"Plummer $a/\varepsilon=2$",
    "plummer_a6": r"Plummer $a/\varepsilon=6$",
    "uniform":    "Esfera uniforme",
}

COLORS = {
    "V1_geom_bare":           "tab:blue",
    "V2_geom_soft":           "tab:orange",
    "V3_rel_bare":            "tab:green",
    "V4_rel_soft":            "tab:red",
    "V5_rel_soft_consistent": "tab:purple",
}


def normalize_dist(d: str) -> str:
    return "uniform" if d == "uniform_sphere" else d


def plot_energy_drift_timeseries(n_target: int = 1000) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, dist in zip(axes.flat, DISTRIBUTIONS):
        for variant in VARIANTS_ORDER:
            tag = f"{dist}_N{n_target}_{variant}"
            path = TIMESERIES_DIR / f"{tag}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path).dropna(subset=["t", "dE_rel"])
            ax.semilogy(df["t"], np.clip(df["dE_rel"], 1e-12, None),
                        label=VARIANT_LABELS[variant],
                        color=COLORS[variant], lw=1.3)
        ax.set_title(f"{DIST_LABELS[dist]} (N={n_target})", fontsize=11)
        ax.set_ylabel(r"$|\Delta E / E_0|$")
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("tiempo [u.i.]")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Drift energético multi-step — Fase 5 (1000 pasos, N={n_target})",
                 fontsize=12)
    fig.tight_layout()
    out = PLOTS_DIR / "energy_drift_timeseries.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}")


def plot_momentum_angmom_timeseries(n_target: int = 1000) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for j, dist in enumerate(DISTRIBUTIONS):
        for variant in VARIANTS_ORDER:
            tag = f"{dist}_N{n_target}_{variant}"
            path = TIMESERIES_DIR / f"{tag}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path).dropna(subset=["t"])
            # |Δp| (abs) — no relativo porque p0 ~ 0 para IC virializadas
            axes[0, j].semilogy(
                df["t"], np.clip(df["dp_abs"], 1e-18, None),
                label=VARIANT_LABELS[variant], color=COLORS[variant], lw=1.1,
            )
            axes[1, j].semilogy(
                df["t"], np.clip(df["dL_abs"], 1e-18, None),
                label=VARIANT_LABELS[variant], color=COLORS[variant], lw=1.1,
            )
        axes[0, j].set_title(DIST_LABELS[dist], fontsize=10)
        axes[0, j].grid(True, which="both", alpha=0.3)
        axes[1, j].grid(True, which="both", alpha=0.3)
        axes[1, j].set_xlabel("tiempo [u.i.]")
    axes[0, 0].set_ylabel(r"$|\Delta \mathbf{p}|$")
    axes[1, 0].set_ylabel(r"$|\Delta \mathbf{L}|$")
    axes[0, 0].legend(loc="lower right", fontsize=7)
    fig.suptitle(f"Conservación de momento y momento angular — N={n_target}",
                 fontsize=12)
    fig.tight_layout()
    out = PLOTS_DIR / "momentum_angmom_timeseries.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}")


def plot_pareto(lv: pd.DataFrame) -> None:
    """Frontera de Pareto error local vs coste por step."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    for ax, n_val in zip(axes, [200, 1000]):
        sub = lv[lv["N"] == n_val]
        for variant in VARIANTS_ORDER:
            vv = sub[sub["variant"] == variant]
            if vv.empty:
                continue
            ax.scatter(
                vv["time_bh_ms"], np.clip(vv["mean_err"], 1e-12, None),
                s=60, c=COLORS[variant], label=VARIANT_LABELS[variant],
                edgecolors="black", linewidths=0.5,
            )
        # Frontera Pareto: para cada tiempo creciente, error mínimo acumulado
        pts = sub[["time_bh_ms", "mean_err"]].sort_values("time_bh_ms").values
        if len(pts) > 0:
            pareto = []
            best = np.inf
            # Orden reverso: comenzamos por el tiempo más alto, quedamos con min error
            for t, e in pts[::-1]:
                if e < best:
                    best = e
                    pareto.append((t, e))
            pareto = np.array(pareto[::-1])
            ax.plot(pareto[:, 0], np.clip(pareto[:, 1], 1e-12, None),
                    "k--", lw=1.3, alpha=0.6, label="frontera Pareto")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("tiempo BH por paso [ms]")
        ax.set_ylabel("mean force error (log)")
        ax.set_title(f"N = {n_val}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Pareto: precisión local vs coste — Fase 5", fontsize=12)
    fig.tight_layout()
    out = PLOTS_DIR / "pareto_precision_cost.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}")


def plot_local_vs_global(lv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for variant in VARIANTS_ORDER:
        sub = lv[lv["variant"] == variant]
        ax.scatter(
            np.clip(sub["mean_err"], 1e-12, None),
            np.clip(sub["dE_rel_final"], 1e-12, None),
            s=70, c=COLORS[variant], label=VARIANT_LABELS[variant],
            edgecolors="black", linewidths=0.5,
        )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean force error (local, log)")
    ax.set_ylabel(r"$|\Delta E / E_0|_{final}$ (global, log)")
    # coef correlación log-log global
    x = np.log10(np.clip(lv["mean_err"].values, 1e-12, None))
    y = np.log10(np.clip(lv["dE_rel_final"].values, 1e-12, None))
    r = float(np.corrcoef(x, y)[0, 1])
    ax.set_title(f"Local vs global — r(log-log) = {r:+.3f}\n"
                 f"(dominado por el integrador cuando r ≈ 0)",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = PLOTS_DIR / "local_vs_global.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}")


def plot_opened_nodes(lv: pd.DataFrame) -> None:
    """Barras: nodos abiertos por variante, promedio sobre distribuciones."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, n_val in zip(axes, [200, 1000]):
        sub = lv[lv["N"] == n_val]
        means = (
            sub.groupby("variant")["opened_nodes"]
               .mean()
               .reindex(VARIANTS_ORDER)
        )
        bars = ax.bar(
            range(len(means)), means.values,
            color=[COLORS[v] for v in VARIANTS_ORDER],
            edgecolor="black", linewidth=0.5,
        )
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS_ORDER],
                           rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("nodos abiertos (media sobre 4 dist)")
        ax.set_title(f"N = {n_val}")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val,
                    f"{int(val):,}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Nodos abiertos por variante (proxy del coste de tree-walk)",
                 fontsize=12)
    fig.tight_layout()
    out = PLOTS_DIR / "opened_nodes_profile.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}")


def main() -> int:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generando figuras Fase 5...")
    plot_energy_drift_timeseries(n_target=1000)
    plot_momentum_angmom_timeseries(n_target=1000)

    lv_path = RESULTS_DIR / "local_vs_global.csv"
    if lv_path.exists():
        lv = pd.read_csv(lv_path)
        # distribution is already normalized (from analyze_local_global.py)
        plot_pareto(lv)
        plot_local_vs_global(lv)
        plot_opened_nodes(lv)
    else:
        print(f"WARN: falta {lv_path}. Ejecuta primero analyze_local_global.py.")

    print("Listo. PNGs en:", PLOTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
