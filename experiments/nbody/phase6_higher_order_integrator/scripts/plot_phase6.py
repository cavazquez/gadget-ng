#!/usr/bin/env python3
"""plot_phase6.py — Figuras paper-grade de la Fase 6.

Produce en `plots/`:
    convergence_order.png         — log-log dt vs err en armónico y Kepler
    energy_drift_timeseries.png   — |ΔE/E| vs t, 2×3 subplots (dist × N)
    pareto_with_yoshida.png       — coste total vs drift energético, KDK vs Yoshida
    kepler_orbit_closure.png      — trayectoria 10 períodos, KDK vs Yoshida

Lee:
    results/phase6_summary.csv
    results/yoshida_convergence.csv
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

DISTRIBUTIONS = ["plummer_a1", "plummer_a2", "uniform"]
DIST_LABELS = {
    "plummer_a1": r"Plummer $a/\varepsilon=1$",
    "plummer_a2": r"Plummer $a/\varepsilon=2$",
    "uniform":    "Esfera uniforme",
}
NS = [200, 1000]
INTEGRATORS = ["leapfrog", "yoshida4"]
INT_LABELS = {"leapfrog": "Leapfrog KDK (O2)", "yoshida4": "Yoshida4 (O4)"}
INT_COLORS = {"leapfrog": "tab:blue", "yoshida4": "tab:red"}


def plot_convergence_order() -> None:
    path = RESULTS_DIR / "yoshida_convergence.csv"
    if not path.exists():
        print(f"(skip) {path} no existe", file=sys.stderr)
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (system, metric) in zip(
        axes,
        [("harmonic", "dE_rel_max"), ("kepler", "closure")],
    ):
        sub = df[(df["system"] == system) & (df["err_metric"] == metric)]
        if sub.empty:
            continue
        for integ in INTEGRATORS:
            s = sub[sub["integrator"] == integ].sort_values("dt")
            slope = float(s["fitted_order"].iloc[0]) if not s.empty else float("nan")
            ax.loglog(
                s["dt"], s["err_value"],
                marker="o", lw=1.5,
                color=INT_COLORS[integ],
                label=f"{INT_LABELS[integ]}  (pendiente {slope:.2f})",
            )
        if not sub.empty:
            dt_ref = sub["dt"].values
            dt0, dtN = dt_ref.min(), dt_ref.max()
            err0 = 1e-4
            ax.loglog(
                [dt0, dtN], [err0 * (dt0 / dtN) ** 2, err0], "k:", alpha=0.4,
                label=r"$\propto dt^2$ (guía)",
            )
            ax.loglog(
                [dt0, dtN], [err0 * (dt0 / dtN) ** 4, err0], "k--", alpha=0.4,
                label=r"$\propto dt^4$ (guía)",
            )
        ax.set_xlabel(r"$dt$")
        if metric == "closure":
            ax.set_ylabel(r"Cierre orbital $|r(T)-r(0)|$")
        else:
            ax.set_ylabel(r"$\max_t\,|\Delta E/E_0|$")
        ax.set_title(f"{system} — métrica: {metric}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Convergencia temporal: Leapfrog KDK (orden 2) vs Yoshida 4º orden",
        fontsize=12,
    )
    fig.tight_layout()
    out = PLOTS_DIR / "convergence_order.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def plot_energy_drift_timeseries() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for row, n in enumerate(NS):
        for col, dist in enumerate(DISTRIBUTIONS):
            ax = axes[row, col]
            for integ in INTEGRATORS:
                tag = f"{dist}_N{n}_{integ}"
                path = TIMESERIES_DIR / f"{tag}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path).dropna(subset=["t", "dE_rel"])
                ax.semilogy(
                    df["t"], np.clip(df["dE_rel"], 1e-14, None),
                    color=INT_COLORS[integ], lw=1.3,
                    label=INT_LABELS[integ],
                )
            ax.set_title(f"{DIST_LABELS[dist]}, N={n}", fontsize=10)
            ax.grid(True, which="both", alpha=0.3)
            if col == 0:
                ax.set_ylabel(r"$|\Delta E / E_0|$")
            if row == 1:
                ax.set_xlabel("t")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Deriva energética en sistemas caóticos: KDK vs Yoshida4 ($dt=0.025$ fijo, solver V5)",
        fontsize=11,
    )
    fig.tight_layout()
    out = PLOTS_DIR / "energy_drift_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def plot_pareto_with_yoshida() -> None:
    path = RESULTS_DIR / "phase6_summary.csv"
    if not path.exists():
        print(f"(skip) {path} no existe", file=sys.stderr)
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, n in zip(axes, NS):
        sub = df[df["N"] == n]
        if sub.empty:
            continue
        for integ in INTEGRATORS:
            s = sub[sub["integrator"] == integ]
            ax.scatter(
                s["total_wall_s"], s["dE_rel_max"],
                color=INT_COLORS[integ],
                label=INT_LABELS[integ], s=60, edgecolor="k", linewidth=0.5,
            )
            for _, r in s.iterrows():
                ax.annotate(
                    r["distribution"].replace("plummer_", "p"),
                    (r["total_wall_s"], r["dE_rel_max"]),
                    fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Coste total wall (s)")
        ax.set_ylabel(r"$\max_t\,|\Delta E/E_0|$")
        ax.set_title(f"N = {n}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Frontera precisión vs coste: Yoshida añade coste 2× sin mejorar drift en regímenes caóticos",
        fontsize=11,
    )
    fig.tight_layout()
    out = PLOTS_DIR / "pareto_with_yoshida.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def plot_kepler_orbit_closure() -> None:
    """Integra 10 períodos de Kepler circular con dt=T/200 y plotea la
    trayectoria XY para KDK y Yoshida. El integrador se reejecuta in-situ vía
    Python (cálculo liviano) para no depender de salidas externas.
    """
    G = M = 1.0
    r0 = np.array([1.0, 0.0])
    v0 = np.array([0.0, np.sqrt(G * M / 1.0)])
    T = 2 * np.pi
    dt = T / 200.0
    n_steps = int(round(10 * T / dt))

    # Yoshida weights
    w1 = 1.351207191959657634
    w0 = -1.7024143839193153

    def accel(r):
        rn = np.linalg.norm(r)
        return -G * M * r / rn ** 3

    def leapfrog_kdk(r, v, dt):
        a = accel(r)
        v = v + a * dt / 2
        r = r + v * dt
        a = accel(r)
        v = v + a * dt / 2
        return r, v

    def yoshida4(r, v, dt):
        k_outer = w1 * dt * 0.5
        k_mix = k_outer + w0 * dt * 0.5
        a = accel(r); v = v + a * k_outer
        r = r + v * (w1 * dt)
        a = accel(r); v = v + a * k_mix
        r = r + v * (w0 * dt)
        a = accel(r); v = v + a * k_mix
        r = r + v * (w1 * dt)
        a = accel(r); v = v + a * k_outer
        return r, v

    traj_kdk = [r0.copy()]
    traj_yos = [r0.copy()]
    r, v = r0.copy(), v0.copy()
    for _ in range(n_steps):
        r, v = leapfrog_kdk(r, v, dt)
        traj_kdk.append(r.copy())
    r, v = r0.copy(), v0.copy()
    for _ in range(n_steps):
        r, v = yoshida4(r, v, dt)
        traj_yos.append(r.copy())
    traj_kdk = np.array(traj_kdk)
    traj_yos = np.array(traj_yos)

    fig, ax = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k:", lw=1, alpha=0.5, label="órbita exacta")
    ax.plot(
        traj_kdk[:, 0], traj_kdk[:, 1],
        color=INT_COLORS["leapfrog"], lw=0.7, alpha=0.7,
        label=INT_LABELS["leapfrog"],
    )
    ax.plot(
        traj_yos[:, 0], traj_yos[:, 1],
        color=INT_COLORS["yoshida4"], lw=0.7, alpha=0.9,
        label=INT_LABELS["yoshida4"],
    )
    ax.plot([r0[0]], [r0[1]], "ko", markersize=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Cierre orbital Kepler tras 10 períodos, $dt=T/200$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    out = PLOTS_DIR / "kepler_orbit_closure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def main() -> int:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_convergence_order()
    plot_energy_drift_timeseries()
    plot_pareto_with_yoshida()
    plot_kepler_orbit_closure()
    return 0


if __name__ == "__main__":
    sys.exit(main())
