#!/usr/bin/env python3
"""plot_phase7.py — Figuras paper-grade para la Fase 7 (Aarseth adaptive timesteps).

Figuras producidas (en plots/):
    1. energy_timeseries.pdf  — |ΔE/E₀| vs tiempo para cada distribución (fixed vs adaptive)
    2. pareto_cost_vs_drift.pdf — Pareto cost × drift (añade Aarseth al diagrama de Fase 6)
    3. dt_histogram.pdf       — Distribución de dt_i efectivos por nivel (última iteración)
    4. drift_vs_eta.pdf       — |ΔE/E₀| final vs η para acc y jerk criteria
    5. adaptivity_vs_dt_control.pdf — Aarseth vs controles dt fijo (aísla adaptatividad)

Uso:
    python3 plot_phase7.py
    python3 plot_phase7.py --dist plummer_a1   # solo una distribución
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
TIMESERIES_DIR = RESULTS_DIR / "timeseries"
PLOTS_DIR = EXP_DIR / "plots"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "text.usetex": False,
})

# ── Paleta y estilos ──────────────────────────────────────────────────────────

STYLE_FIXED = {
    "fixed_dt025":   {"color": "#1f77b4", "ls": "-",  "lw": 1.8, "label": "Fixed dt=0.025 (baseline)"},
    "fixed_dt0125":  {"color": "#aec7e8", "ls": "--", "lw": 1.4, "label": "Fixed dt/2=0.0125"},
    "fixed_dt00625": {"color": "#c6dbef", "ls": ":",  "lw": 1.4, "label": "Fixed dt/4=0.00625"},
}

ETA_COLORS_ACC  = {0.01: "#d62728", 0.02: "#ff7f0e", 0.05: "#e377c2"}
ETA_COLORS_JERK = {0.01: "#2ca02c", 0.02: "#17becf", 0.05: "#8c564b"}


def variant_style(variant: str, row: pd.Series) -> dict:
    if not row.get("adaptive", False):
        return STYLE_FIXED.get(variant, {"color": "gray", "ls": "-", "lw": 1.0})
    crit = str(row.get("criterion", ""))
    eta = float(row.get("eta", 0))
    if crit == "acceleration":
        color = ETA_COLORS_ACC.get(eta, "red")
        marker = "o"
    else:
        color = ETA_COLORS_JERK.get(eta, "green")
        marker = "s"
    ls = {0.01: "-", 0.02: "--", 0.05: ":"}.get(eta, "-")
    label = f"Hier {crit[:4]} η={eta:.2f}"
    return {"color": color, "ls": ls, "lw": 1.2, "marker": marker,
            "ms": 3, "markevery": 50, "label": label}


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_summary() -> pd.DataFrame | None:
    path = RESULTS_DIR / "phase7_summary.csv"
    if not path.exists():
        print(f"[warn] No se encontró {path}. Ejecuta analyze_conservation.py primero.",
              file=sys.stderr)
        return None
    return pd.read_csv(path)


def load_timeseries(tag: str) -> pd.DataFrame | None:
    path = TIMESERIES_DIR / f"{tag}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── Figura 1: energy_timeseries ───────────────────────────────────────────────

def fig_energy_timeseries(summary: pd.DataFrame, dist_filter: str | None) -> None:
    dists = summary["distribution"].unique() if dist_filter is None else [dist_filter]
    n_vals = sorted(summary["N"].unique())

    for dist in dists:
        fig, axes = plt.subplots(1, len(n_vals), figsize=(5 * len(n_vals), 4), sharey=True)
        if len(n_vals) == 1:
            axes = [axes]

        for ax, n in zip(axes, n_vals):
            sub = summary[(summary["distribution"] == dist) & (summary["N"] == n)]
            for _, row in sub.iterrows():
                ts = load_timeseries(row["tag"])
                if ts is None or "dE_rel" not in ts.columns:
                    continue
                ts = ts.dropna(subset=["dE_rel", "t"])
                style = variant_style(row["variant"], row)
                ax.semilogy(ts["t"], ts["dE_rel"],
                            color=style["color"], ls=style["ls"], lw=style["lw"],
                            label=style.get("label", row["variant"]))

            ax.set_xlabel("Tiempo")
            ax.set_title(f"{dist}, N={n}")
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

        axes[0].set_ylabel("|ΔE/E₀|")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0),
                   borderaxespad=0.1, framealpha=0.9)
        fig.suptitle(f"Fase 7 — Energía vs Tiempo: {dist}", y=1.02)
        fig.tight_layout()
        out = PLOTS_DIR / f"energy_timeseries_{dist}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig1] {out.name}")


# ── Figura 2: Pareto cost vs drift ────────────────────────────────────────────

def fig_pareto_cost_vs_drift(summary: pd.DataFrame, dist_filter: str | None) -> None:
    dists = summary["distribution"].unique() if dist_filter is None else [dist_filter]
    n_vals = sorted(summary["N"].unique())

    for dist in dists:
        fig, axes = plt.subplots(1, len(n_vals), figsize=(5 * len(n_vals), 4))
        if len(n_vals) == 1:
            axes = [axes]

        for ax, n in zip(axes, n_vals):
            sub = summary[(summary["distribution"] == dist) & (summary["N"] == n)]
            for _, row in sub.iterrows():
                x = row.get("total_wall_s", float("nan"))
                y = row.get("dE_rel_final", float("nan"))
                if np.isnan(x) or np.isnan(y):
                    continue
                style = variant_style(row["variant"], row)
                ax.scatter(x, y, color=style["color"],
                           marker=style.get("marker", "o"),
                           s=60, label=style.get("label", row["variant"]),
                           zorder=3)

            ax.set_xlabel("Coste total (s)")
            ax.set_ylabel("|ΔE/E₀| final")
            ax.set_yscale("log")
            ax.set_title(f"{dist}, N={n}")
            ax.grid(True, alpha=0.3)

        # Leyenda unificada
        handles_labels = [ax.get_legend_handles_labels() for ax in axes]
        all_h, all_l = [], []
        seen = set()
        for hs, ls in handles_labels:
            for h, l in zip(hs, ls):
                if l not in seen:
                    all_h.append(h)
                    all_l.append(l)
                    seen.add(l)
        fig.legend(all_h, all_l, loc="upper right", bbox_to_anchor=(1.0, 1.0),
                   borderaxespad=0.1, framealpha=0.9)
        fig.suptitle(f"Fase 7 — Pareto Costo vs Drift: {dist}", y=1.02)
        fig.tight_layout()
        out = PLOTS_DIR / f"pareto_cost_vs_drift_{dist}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig2] {out.name}")


# ── Figura 3: dt_histogram ────────────────────────────────────────────────────

def fig_dt_histogram(summary: pd.DataFrame, dist_filter: str | None) -> None:
    """Distribución de partículas por nivel (del histograma del último paso)."""
    level_cols = [c for c in summary.columns if c.startswith("level_") and c.endswith("_count")]
    if not level_cols:
        print("  [fig3] Sin datos de histograma de niveles (runs jerárquicos no completados)")
        return

    adaptive = summary[summary["adaptive"] == True].copy()  # noqa: E712
    if adaptive.empty:
        print("  [fig3] Sin runs adaptativos para histograma")
        return

    dists = adaptive["distribution"].unique() if dist_filter is None else [dist_filter]
    n_vals = sorted(adaptive["N"].unique())

    for dist in dists:
        fig, axes = plt.subplots(1, len(n_vals), figsize=(5 * len(n_vals), 4), sharey=False)
        if len(n_vals) == 1:
            axes = [axes]

        for ax, n in zip(axes, n_vals):
            sub = adaptive[(adaptive["distribution"] == dist) & (adaptive["N"] == n)]
            for _, row in sub.iterrows():
                levels = []
                counts = []
                for col in sorted(level_cols):
                    lvl = int(col.split("_")[1])
                    cnt = row.get(col, 0)
                    if not np.isnan(cnt):
                        levels.append(lvl)
                        counts.append(int(cnt))
                if not counts or sum(counts) == 0:
                    continue

                label = f"{row['criterion'][:4]} η={row['eta']:.2f}"
                color = (ETA_COLORS_ACC if row["criterion"] == "acceleration"
                         else ETA_COLORS_JERK).get(row["eta"], "gray")
                fracs = [c / sum(counts) for c in counts]
                ax.plot(levels, fracs, marker="o", ms=5, color=color, label=label)

            ax.set_xlabel("Nivel jerárquico k")
            ax.set_ylabel("Fracción de partículas")
            ax.set_title(f"{dist}, N={n}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Fase 7 — Histograma de niveles: {dist}", y=1.02)
        fig.tight_layout()
        out = PLOTS_DIR / f"dt_histogram_{dist}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig3] {out.name}")


# ── Figura 4: drift_vs_eta ────────────────────────────────────────────────────

def fig_drift_vs_eta(summary: pd.DataFrame, dist_filter: str | None) -> None:
    adaptive = summary[summary["adaptive"] == True].copy()  # noqa: E712
    if adaptive.empty:
        print("  [fig4] Sin runs adaptativos para drift vs η")
        return

    dists = adaptive["distribution"].unique() if dist_filter is None else [dist_filter]
    n_vals = sorted(adaptive["N"].unique())

    for dist in dists:
        fig, axes = plt.subplots(1, len(n_vals), figsize=(5 * len(n_vals), 4), sharey=True)
        if len(n_vals) == 1:
            axes = [axes]

        for ax, n in zip(axes, n_vals):
            sub = adaptive[(adaptive["distribution"] == dist) & (adaptive["N"] == n)]

            for crit, marker in [("acceleration", "o"), ("jerk", "s")]:
                csub = sub[sub["criterion"] == crit].sort_values("eta")
                if csub.empty:
                    continue
                colors = [ETA_COLORS_ACC if crit == "acceleration"
                          else ETA_COLORS_JERK][0]
                color_list = [colors.get(e, "gray") for e in csub["eta"]]
                ax.plot(csub["eta"], csub["dE_rel_final"],
                        marker=marker, ms=8, color="gray", ls="--", lw=1,
                        label=f"Hier {crit[:4]}")
                for e, de, c in zip(csub["eta"], csub["dE_rel_final"], color_list):
                    ax.scatter([e], [de], color=c, marker=marker, s=80, zorder=4)

            # Línea de referencia: baseline
            fixed = summary[(summary["distribution"] == dist)
                            & (summary["N"] == n)
                            & (summary["variant"] == "fixed_dt025")]
            if not fixed.empty:
                base_de = float(fixed["dE_rel_final"].iloc[0])
                ax.axhline(base_de, color="#1f77b4", ls="-", lw=1.5,
                           label="Fixed dt=0.025")

            ax.set_xlabel("η")
            ax.set_ylabel("|ΔE/E₀| final")
            ax.set_yscale("log")
            ax.set_title(f"{dist}, N={n}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Fase 7 — Drift vs η: {dist}", y=1.02)
        fig.tight_layout()
        out = PLOTS_DIR / f"drift_vs_eta_{dist}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig4] {out.name}")


# ── Figura 5: adaptivity_vs_dt_control ───────────────────────────────────────

def fig_adaptivity_vs_dt_control(summary: pd.DataFrame, dist_filter: str | None) -> None:
    """Compara los runs adaptativos contra los controles de dt fijo.

    Objetivo: aislar si la mejora (o ausencia de mejora) viene de adaptatividad
    real o simplemente de un dt promedio más fino.
    """
    dists = summary["distribution"].unique() if dist_filter is None else [dist_filter]
    n_vals = sorted(summary["N"].unique())

    for dist in dists:
        fig, axes = plt.subplots(1, len(n_vals), figsize=(5 * len(n_vals), 4))
        if len(n_vals) == 1:
            axes = [axes]

        for ax, n in zip(axes, n_vals):
            sub = summary[(summary["distribution"] == dist) & (summary["N"] == n)]

            # Controles de dt fijo
            for variant, style in STYLE_FIXED.items():
                row = sub[sub["variant"] == variant]
                if row.empty:
                    continue
                x = float(row["total_wall_s"].iloc[0])
                y = float(row["dE_rel_final"].iloc[0])
                ax.scatter([x], [y], color=style["color"],
                           marker="D", s=80, label=style["label"], zorder=3)

            # Runs adaptativos
            adaptive = sub[sub["adaptive"] == True]  # noqa: E712
            for _, row in adaptive.iterrows():
                x = row.get("total_wall_s", float("nan"))
                y = row.get("dE_rel_final", float("nan"))
                if np.isnan(x) or np.isnan(y):
                    continue
                style = variant_style(row["variant"], row)
                ax.scatter([x], [y], color=style["color"],
                           marker=style.get("marker", "o"),
                           s=60, label=style.get("label", row["variant"]),
                           zorder=4)

            ax.set_xlabel("Coste total (s)")
            ax.set_ylabel("|ΔE/E₀| final")
            ax.set_yscale("log")
            ax.set_title(f"{dist}, N={n}")
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

        # Leyenda
        handles_labels = [ax.get_legend_handles_labels() for ax in axes]
        all_h, all_l = [], []
        seen = set()
        for hs, ls in handles_labels:
            for h, l in zip(hs, ls):
                if l not in seen:
                    all_h.append(h)
                    all_l.append(l)
                    seen.add(l)
        fig.legend(all_h, all_l, loc="upper right", bbox_to_anchor=(1.0, 1.0),
                   borderaxespad=0.1, framealpha=0.9, fontsize=7)
        fig.suptitle(
            f"Fase 7 — Adaptatividad vs controles dt fijo: {dist}\n"
            "(aislando efecto real del block timestep)",
            y=1.04
        )
        fig.tight_layout()
        out = PLOTS_DIR / f"adaptivity_vs_dt_control_{dist}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig5] {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Genera figuras de Fase 7")
    parser.add_argument("--dist", default=None, help="Filtra por distribución")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_summary()
    if summary is None:
        return 1

    print(f"Cargados {len(summary)} runs. Generando figuras...")
    fig_energy_timeseries(summary, args.dist)
    fig_pareto_cost_vs_drift(summary, args.dist)
    fig_dt_histogram(summary, args.dist)
    fig_drift_vs_eta(summary, args.dist)
    fig_adaptivity_vs_dt_control(summary, args.dist)

    print(f"\nFiguras en: {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
