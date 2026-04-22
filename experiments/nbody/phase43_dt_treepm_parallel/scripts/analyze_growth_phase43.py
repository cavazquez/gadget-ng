#!/usr/bin/env python3
"""Análisis cuantitativo del crecimiento de bajo-k para Phase 43.

Produce:
  * `growth_phase43.png` — error relativo de crecimiento vs dt.
  * `phase43_growth.csv` — tabla consolidada.
  * Resumen a stdout con la decisión efectiva del test 3.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def growth_low_k(snap_ic, snap_ev, k_max=0.1):
    if not snap_ic or not snap_ev:
        return None
    ks_ic = np.asarray(snap_ic["ks_hmpc"])
    ks_ev = np.asarray(snap_ev["ks_hmpc"])
    pc_ic = np.asarray(snap_ic["pk_corrected_mpc_h3"])
    pc_ev = np.asarray(snap_ev["pk_corrected_mpc_h3"])
    ratios = []
    for i, k in enumerate(ks_ev):
        if k > k_max:
            break
        j = np.where(np.abs(ks_ic - k) < 1e-9)[0]
        if len(j) == 0:
            continue
        if pc_ic[j[0]] > 0 and pc_ev[i] > 0:
            ratios.append(pc_ev[i] / pc_ic[j[0]])
    if not ratios:
        return None
    return float(np.mean(ratios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, type=Path)
    ap.add_argument("--test3", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--k-max", type=float, default=0.1)
    ap.add_argument("--a-init", type=float, default=0.02)
    ap.add_argument("--a-target", type=float, default=0.10)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with args.matrix.open() as f:
        data = json.load(f)
    snapshots = data["snapshots"]

    with args.test3.open() as f:
        test3 = json.load(f)

    variants = sorted({s["variant"] for s in snapshots})
    ics = {s["variant"]: s for s in snapshots if s["a_target"] == args.a_init}
    evs = {s["variant"]: s for s in snapshots if s["a_target"] == args.a_target}

    rows = []
    for v in variants:
        if v not in ics or v not in evs:
            continue
        g = growth_low_k(ics[v], evs[v], args.k_max)
        dt_nominal = evs[v].get("dt_nominal")
        is_ad = evs[v].get("is_adaptive", False)
        n_steps = evs[v].get("n_steps")
        rows.append(dict(variant=v, dt_nominal=dt_nominal, is_adaptive=is_ad,
                         growth=g, n_steps=n_steps))

    # CSV
    with (args.outdir / "phase43_growth.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","dt_nominal","is_adaptive",
                                          "growth","n_steps"])
        w.writeheader()
        w.writerows(rows)

    # Figura
    fig, ax = plt.subplots(figsize=(6, 4))
    fixed = [r for r in rows if not r["is_adaptive"] and r["growth"] is not None]
    fixed.sort(key=lambda r: r["dt_nominal"])
    if fixed:
        xs = [r["dt_nominal"] for r in fixed]
        ys = [r["growth"] for r in fixed]
        ax.plot(xs, ys, "o-", label="fijo")
    ads = [r for r in rows if r["is_adaptive"] and r["growth"] is not None]
    for r in ads:
        a_steps = r["n_steps"] or 1
        dt_eff = (args.a_target - args.a_init) / a_steps
        ax.plot([dt_eff], [r["growth"]], "s", color="tab:red", markersize=8,
                label=r["variant"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dt [código]")
    ax.set_ylabel(f"⟨P_c(k≤{args.k_max}) (a={args.a_target}) / P_c (a={args.a_init})⟩")
    ax.set_title("Phase 43 — crecimiento en bajo-k vs dt")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "growth_phase43.png", dpi=150)
    plt.close(fig)

    # Resumen
    print(f"[phase43] análisis crecimiento → {args.outdir}")
    print(f"  test3 decision: {test3.get('decision')}")
    print(f"  best fixed err: {test3.get('best_fixed_err')}")
    print(f"  adaptive err:   {test3.get('adaptive_err')}")
    for r in rows:
        print(f"  {r['variant']:<18} dt={r['dt_nominal']} adapt={r['is_adaptive']} "
              f"steps={r['n_steps']} growth={r['growth']}")


if __name__ == "__main__":
    main()
