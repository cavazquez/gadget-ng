#!/usr/bin/env python3
"""Genera figuras del efecto del `dt` en el barrido Phase 43.

Entradas:
    --matrix : JSON producido por `phase43_dt_treepm_parallel.rs`
                (`target/phase43/per_snapshot_metrics.json`).

Salidas (en `--outdir`):
    error_vs_dt.png           — error espectral vs dt.
    growth_vs_theory.png      — P(k,a)/P(k,a_i) vs [D(a)/D(a_i)]² para cada dt.
    delta_rms_vs_a.png        — δ_rms(a) vs a por variante.
    runtime_vs_dt.png         — wall-clock vs dt.
    adaptive_dt_trace.png     — dt(a) del modo adaptativo (si presente).
    phase43_dt_sweep.csv      — tabla agregada.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def median_abs(xs):
    if not xs:
        return math.nan
    return float(np.median(np.abs(xs)))


def load_matrix(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def aggregate(snapshots):
    rows = []
    for s in snapshots:
        pc = np.asarray(s.get("pk_corrected_mpc_h3") or [])
        pr = np.asarray(s.get("pk_reference_mpc_h3") or [])
        if pc.size and pr.size and (pc > 0).all() and (pr > 0).all():
            log_err = np.log10(pc / pr)
            med = float(np.median(np.abs(log_err)))
            mean_r = float(np.mean(pc / pr))
            std_r = float(np.std(pc / pr))
        else:
            med = float("nan")
            mean_r = float("nan")
            std_r = float("nan")
        rows.append(
            dict(
                variant=s["variant"],
                dt_nominal=s.get("dt_nominal"),
                is_adaptive=bool(s.get("is_adaptive", False)),
                a_target=s["a_target"],
                a_actual=s["a_actual"],
                n_steps=s.get("n_steps"),
                wall_s=s.get("wall_time_s"),
                median_abs_log_err=med,
                mean_ratio=mean_r,
                std_ratio=std_r,
                cv_ratio=std_r / abs(mean_r) if mean_r else float("nan"),
                delta_rms=s["delta_rms"],
                v_rms=s["v_rms"],
            )
        )
    return rows


def plot_error_vs_dt(rows, outdir: Path):
    dts = sorted({r["dt_nominal"] for r in rows if not r["is_adaptive"]})
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in (0.05, 0.10):
        ys = []
        for dt in dts:
            r = [
                x
                for x in rows
                if not x["is_adaptive"] and x["dt_nominal"] == dt and x["a_target"] == a
            ]
            if r:
                ys.append(r[0]["median_abs_log_err"])
            else:
                ys.append(float("nan"))
        ax.plot(dts, ys, marker="o", label=f"a={a:.2f}")

    # Marker for adaptive (proyectado sobre su número de pasos, usado como eje)
    adapts = [r for r in rows if r["is_adaptive"] and r["a_target"] == 0.10]
    if adapts:
        a_steps = adapts[0]["n_steps"] or 1
        dt_eff = (0.10 - 0.02) / a_steps
        ax.axvline(dt_eff, color="tab:red", ls="--", lw=1, alpha=0.6,
                   label=f"⟨dt_adapt⟩ ≈ {dt_eff:.1e}")

    ax.set_xscale("log")
    ax.set_xlabel("dt [código]")
    ax.set_ylabel(r"median $|\log_{10}(P_c/P_{\rm ref})|$")
    ax.set_title("Phase 43 — error espectral vs dt (TreePM + ε_phys=0.01)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "error_vs_dt.png", dpi=150)
    plt.close(fig)


def plot_growth_vs_theory(rows, outdir: Path):
    # Para cada variante fijo, dibujar P(k,a=0.10)/P(k,a=0.02) como función de k,
    # comparado con [D(0.10)/D(0.02)]². Este valor teórico se asume tomado del
    # reporte de matriz (se lee del primer snapshot disponible si existe).
    fig, ax = plt.subplots(figsize=(6, 4))
    variants = sorted({r["variant"] for r in rows})
    theory = None
    for v in variants:
        r010 = [r for r in rows if r["variant"] == v and r["a_target"] == 0.10]
        if not r010:
            continue
        r010 = r010[0]
        # Recuperar theory si acompañó
        theory = r010.get("growth_theory_d_sq", theory)
    # Usamos el ratio ⟨P_c(a=0.10)/P_c(a=0.02)⟩ calculado externamente a partir
    # de los snapshots del JSON (no están directamente aquí; lo ploteamos
    # contra el número de pasos como proxy de dt).
    xs, ys, lbls = [], [], []
    for r in rows:
        if r["a_target"] != 0.10 or r["is_adaptive"]:
            continue
        xs.append(r["dt_nominal"])
        ys.append(r["mean_ratio"])
        lbls.append(r["variant"])
    if xs:
        ax.plot(xs, ys, "o-", label="⟨P_c/P_ref⟩(a=0.10)")
    adapts = [r for r in rows if r["is_adaptive"] and r["a_target"] == 0.10]
    if adapts:
        a_steps = adapts[0]["n_steps"] or 1
        dt_eff = (0.10 - 0.02) / a_steps
        ax.plot(
            [dt_eff], [adapts[0]["mean_ratio"]], "s", color="tab:red",
            label="adaptive", markersize=8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dt [código]")
    ax.set_ylabel(r"$\langle P_c(k\leq 0.1)/P_{\rm ref}\rangle (a=0.10)$")
    ax.set_title("Phase 43 — ratio corregido en bajo-k vs dt")
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.5, label="ideal = 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "growth_vs_theory.png", dpi=150)
    plt.close(fig)


def plot_delta_rms_vs_a(rows, outdir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    variants = sorted({r["variant"] for r in rows})
    for v in variants:
        rv = sorted([r for r in rows if r["variant"] == v], key=lambda x: x["a_target"])
        xs = [r["a_target"] for r in rv]
        ys = [r["delta_rms"] for r in rv]
        ax.plot(xs, ys, marker="o", label=v)
    ax.set_xlabel("a")
    ax.set_ylabel(r"$\delta_{\rm rms}(a)$")
    ax.set_title("Phase 43 — δ_rms(a) por variante de dt")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "delta_rms_vs_a.png", dpi=150)
    plt.close(fig)


def plot_runtime_vs_dt(rows, outdir: Path):
    # Total wall (suma sobre snapshots) por variante.
    fig, ax = plt.subplots(figsize=(6, 4))
    variants = sorted({r["variant"] for r in rows})
    for v in variants:
        rv = [r for r in rows if r["variant"] == v]
        total = sum((r["wall_s"] or 0.0) for r in rv)
        is_ad = any(r["is_adaptive"] for r in rv)
        marker = "s" if is_ad else "o"
        color = "tab:red" if is_ad else "tab:blue"
        dt_nom = (0.10 - 0.02) / (max((r["n_steps"] or 1) for r in rv))
        ax.scatter([dt_nom], [total], marker=marker, color=color, s=60, label=v)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("⟨dt⟩ efectivo [código]")
    ax.set_ylabel("wall-clock total [s]")
    ax.set_title("Phase 43 — runtime vs dt")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "runtime_vs_dt.png", dpi=150)
    plt.close(fig)


def plot_adaptive_trace(snapshots, outdir: Path):
    # Busca la variante adaptive_cosmo (o cualquier is_adaptive) y plotea dt(a).
    target = None
    for s in snapshots:
        if s.get("is_adaptive") and s.get("a_target") == 0.10:
            target = s
            break
    if target is None:
        return
    trace = target.get("dt_trace") or []
    if not trace:
        return
    xs = [row[0] for row in trace]
    ys = [row[1] for row in trace]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("a")
    ax.set_ylabel("dt adaptativo [código]")
    ax.set_title(f"Phase 43 — {target['variant']}: dt(a)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "adaptive_dt_trace.png", dpi=150)
    plt.close(fig)


def write_csv(rows, outdir: Path):
    cols = [
        "variant", "dt_nominal", "is_adaptive", "a_target", "a_actual",
        "n_steps", "wall_s", "median_abs_log_err", "mean_ratio", "std_ratio",
        "cv_ratio", "delta_rms", "v_rms",
    ]
    with (outdir / "phase43_dt_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = load_matrix(args.matrix)
    snapshots = data["snapshots"]
    rows = aggregate(snapshots)

    plot_error_vs_dt(rows, args.outdir)
    plot_growth_vs_theory(rows, args.outdir)
    plot_delta_rms_vs_a(rows, args.outdir)
    plot_runtime_vs_dt(rows, args.outdir)
    plot_adaptive_trace(snapshots, args.outdir)
    write_csv(rows, args.outdir)
    print(f"[phase43] figuras + CSV → {args.outdir}")


if __name__ == "__main__":
    main()
