#!/usr/bin/env python3
"""
Fase 4: Plots de ablación multipolar, softening y criterio de apertura.

Genera:
  1. softened_ablation.png    — error de fuerza bare vs softened × a/ε
  2. radial_error.png         — error de fuerza vs r/ε (perfil radial)
  3. criterion_pareto.png     — curva Pareto error vs costo para cada criterio
  4. softening_regime.png     — régimen de mejora de softened (a/ε vs ratio)
"""
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
PLOTS_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── 1. Ablación softening: bare vs softened × a/ε ──────────────────────────────

ablation_csv = os.path.join(RESULTS_DIR, "softened_ablation.csv")
if os.path.exists(ablation_csv):
    df = pd.read_csv(ablation_csv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Ablación multipolar: bare vs softened — θ=0.5, N=500", fontsize=13)

    # Panel 1: mean_err vs orden para cada distribución/softening
    ax = axes[0]
    for dist in df["distribution"].unique():
        sub = df[df["distribution"] == dist]
        for softened in [0, 1]:
            s = sub[sub["softened"] == softened]
            label = f"{dist} ({'soft' if softened else 'bare'})"
            ax.plot(s["order"], s["mean_err"] * 100, marker="o", label=label)
    ax.set_xlabel("Orden multipolar")
    ax.set_ylabel("Error de fuerza medio (%)")
    ax.set_title("Error medio vs orden")
    ax.legend(fontsize=6)
    ax.set_yscale("log")

    # Panel 2: ratio (bare/softened) vs a/ε para orden=2
    ax = axes[1]
    order2 = df[df["order"] == 2]
    bare2 = order2[order2["softened"] == 0].set_index("distribution")
    soft2 = order2[order2["softened"] == 1].set_index("distribution")
    # Solo distribuciones Plummer (tienen a_over_eps > 0)
    plummer = order2[order2["a_plummer"] > 0].copy()
    plummer_bare = plummer[plummer["softened"] == 0]
    plummer_soft = plummer[plummer["softened"] == 1]
    if len(plummer_bare) > 0 and len(plummer_soft) > 0:
        merged = plummer_bare[["a_over_eps", "mean_err"]].merge(
            plummer_soft[["a_over_eps", "mean_err"]], on="a_over_eps",
            suffixes=("_bare", "_soft")
        )
        merged = merged.sort_values("a_over_eps")
        ratio = merged["mean_err_bare"] / merged["mean_err_soft"].clip(lower=1e-15)
        ax.plot(merged["a_over_eps"], ratio, "o-", color="tab:orange")
        ax.axhline(1.0, color="gray", ls="--", label="ratio=1 (sin mejora)")
        ax.axvline(5.0, color="tab:red", ls=":", label="a/ε = 5 (frontera de régimen)")
        ax.set_xlabel("a / ε (concentración relativa al softening)")
        ax.set_ylabel("Ratio error bare / softened (orden 2)")
        ax.set_title("Mejora del softening vs concentración")
        ax.legend(fontsize=8)
        ax.set_xscale("log")

    # Panel 3: max_err y p95_err comparación
    ax = axes[2]
    for dist in df[df["a_plummer"] == 0.1]["distribution"].unique():
        sub = df[(df["distribution"] == dist) & (df["order"] == 3)]
        labels = ["bare", "softened"]
        bars = [sub[sub["softened"] == 0]["max_err"].values[0] * 100,
                sub[sub["softened"] == 1]["max_err"].values[0] * 100]
        ax.bar(labels, bars, label=dist)
    ax.set_ylabel("Error máximo (%)")
    ax.set_title("Max error: bare vs softened (Plummer a=0.1, orden 3)")
    ax.set_yscale("log")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "softened_ablation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out}")
    plt.close()


# ── 2. Perfil radial de error ───────────────────────────────────────────────────

radial_csv = os.path.join(RESULTS_DIR, "radial_error_analysis.csv")
if os.path.exists(radial_csv):
    df = pd.read_csv(radial_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Error de fuerza radial — Plummer a=0.1, θ=0.5, N=500", fontsize=13)

    colors = {
        "mono_bare": "tab:blue",
        "quad_bare": "tab:orange",
        "quad_soft": "tab:green",
        "oct_bare": "tab:red",
        "oct_soft": "tab:purple",
    }
    styles = {
        "mono_bare": "-",
        "quad_bare": "--",
        "quad_soft": "-",
        "oct_bare": "--",
        "oct_soft": "-",
    }

    for ax, metric in zip(axes, ["mean_err", "max_err"]):
        for config in df["config"].unique():
            sub = df[(df["config"] == config) & (df["n_particles"] > 0)]
            if sub.empty:
                continue
            ax.plot(
                sub["r_over_eps_center"],
                sub[metric] * 100,
                marker="o", markersize=4,
                color=colors.get(config, "gray"),
                ls=styles.get(config, "-"),
                label=config,
            )
        ax.axvline(1.0, color="gray", ls=":", alpha=0.7, label="r=ε")
        ax.set_xlabel("r / ε (distancia normalizada al softening)")
        ax.set_ylabel(f"{metric.replace('_', ' ')} (%)")
        ax.set_title(f"{'Error medio' if 'mean' in metric else 'Error máximo'} vs r/ε")
        ax.legend(fontsize=8)
        ax.set_yscale("log")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "radial_error.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out}")
    plt.close()


# ── 3. Curva Pareto error vs costo ─────────────────────────────────────────────

sweep_csv = os.path.join(RESULTS_DIR, "bh_criterion_sweep.csv")
if os.path.exists(sweep_csv):
    df = pd.read_csv(sweep_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Pareto error vs costo: criterio de apertura — N=500", fontsize=13)

    distributions = df["distribution"].unique()
    markers = {"geometric_bare": "o", "geometric_soft": "s",
               "relative_bare": "^", "relative_soft": "D"}
    colors_map = {"geometric_bare": "tab:blue", "geometric_soft": "tab:cyan",
                  "relative_bare": "tab:orange", "relative_soft": "tab:red"}

    for ax, dist in zip(axes, distributions):
        sub = df[df["distribution"] == dist]
        for criterion in sub["criterion"].unique():
            s = sub[sub["criterion"] == criterion].sort_values("time_bh_ms")
            if s.empty:
                continue
            ax.scatter(
                s["time_bh_ms"], s["mean_err"] * 100,
                marker=markers.get(criterion, "o"),
                color=colors_map.get(criterion, "gray"),
                label=criterion, s=60, zorder=3,
            )
            ax.plot(s["time_bh_ms"], s["mean_err"] * 100,
                    color=colors_map.get(criterion, "gray"), alpha=0.4)
        ax.set_xlabel("Tiempo BH (ms)")
        ax.set_ylabel("Error de fuerza medio (%)")
        ax.set_title(f"Pareto: {dist}")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "criterion_pareto.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out}")
    plt.close()


print(f"\nTodos los plots guardados en {PLOTS_DIR}")
