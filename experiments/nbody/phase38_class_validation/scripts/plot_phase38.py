#!/usr/bin/env python3
"""Phase 38 — 4+1 mandatory figures from the Rust test output.

Reads `target/phase38/per_measurement.json` (written by the Rust integration
tests in `crates/gadget-ng-physics/tests/phase38_class_validation.rs`) and
produces:

1. `pk_class_vs_gadget.png`        — P_m, P_c, P_CLASS in log–log.
2. `ratio_pm_pc_vs_class.png`      — P_m/P_CLASS and P_c/P_CLASS with BAO band shaded.
3. `abs_error_before_after.png`    — |log10(P_m/P_CLASS)| vs |log10(P_c/P_CLASS)|.
4. `n32_vs_n64.png`                — cross-resolution comparison.
5. `legacy_vs_rescaled.png` (opt.) — both modes overlaid at fixed N.

All figures are produced in matplotlib with sensible defaults; no
interactivity.
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


BAO_K_MIN = 0.05
BAO_K_MAX = 0.30


def _find(ms, n, seed, mode):
    for m in ms:
        if m["n"] == n and m["seed"] == seed and m["mode"] == mode:
            return m
    raise KeyError(f"no measurement: N={n} seed={seed} mode={mode}")


def _arr(m, key):
    return np.array(m[key], dtype=float)


def plot_pk_class_vs_gadget(ms, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, n in zip(axes, (32, 64)):
        m = _find(ms, n, 42, "legacy")
        ks = _arr(m, "ks_hmpc")
        pm = _arr(m, "pk_measured_internal")
        pc = _arr(m, "pk_corrected_mpc_h3")
        pr = _arr(m, "pk_class_mpc_h3")
        ax.loglog(ks, pm, "o", label="$P_\\mathrm{m}$ (internal units)", alpha=0.5)
        ax.loglog(ks, pc, "s", label="$P_\\mathrm{c}$ (corrected)", ms=8)
        ax.loglog(ks, pr, "-", label="$P_\\mathrm{CLASS}(z=0)$", lw=2, color="black")
        ax.axvspan(BAO_K_MIN, BAO_K_MAX, color="gold", alpha=0.12, label="BAO band")
        ax.set_title(f"$N = {n}^3$, seed 42, legacy, IC")
        ax.set_xlabel("$k\\;\\mathrm{[h/Mpc]}$")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("$P(k)$")
    fig.suptitle("Phase 38 — gadget-ng corrected P(k) vs CLASS at IC")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_ratio_pm_pc_vs_class(ms, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, n in zip(axes, (32, 64)):
        for seed in (42, 137, 271):
            m = _find(ms, n, seed, "legacy")
            ks = _arr(m, "ks_hmpc")
            pm = _arr(m, "pk_measured_internal")
            pc = _arr(m, "pk_corrected_mpc_h3")
            pr = _arr(m, "pk_class_mpc_h3")
            ax.loglog(ks, pm / pr, ":", color="C0", alpha=0.6,
                      label="$P_\\mathrm{m}/P_\\mathrm{CLASS}$" if seed == 42 else None)
            ax.semilogx(ks, pc / pr, "-o", color="C3", alpha=0.7,
                        label=f"$P_\\mathrm{{c}}/P_\\mathrm{{CLASS}}$ (seed {seed})")
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.axvspan(BAO_K_MIN, BAO_K_MAX, color="gold", alpha=0.12, label="BAO band")
        ax.set_title(f"$N = {n}^3$, legacy")
        ax.set_xlabel("$k\\;\\mathrm{[h/Mpc]}$")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("ratio")
    fig.suptitle("Phase 38 — ratios vs CLASS (raw shown scaled down by A_grid×R)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_abs_error_before_after(ms, out):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    offset = 0
    for n in (32, 64):
        for mode in ("legacy", "rescaled"):
            raws = []
            corrs = []
            for seed in (42, 137, 271):
                m = _find(ms, n, seed, mode)
                raws.append(m["metrics_all"]["median_abs_log10_err_raw"])
                corrs.append(m["metrics_all"]["median_abs_log10_err_corr"])
            label = f"N={n}³ {mode}"
            x = offset
            ax.bar(x - 0.2, np.mean(raws), width=0.35, color="C0",
                   label="raw" if offset == 0 else None)
            ax.bar(x + 0.2, np.mean(corrs), width=0.35, color="C3",
                   label="corrected" if offset == 0 else None)
            ax.text(x, np.mean(raws) * 1.05, f"×{np.mean(raws)/max(np.mean(corrs), 1e-12):.0e}",
                    ha="center", fontsize=8)
            ax.text(x + 0.2, np.mean(corrs) * 2.0, f"{np.mean(corrs):.3f}",
                    ha="center", fontsize=8)
            offset += 1
    ax.set_yscale("log")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["N=32 legacy", "N=32 rescaled",
                        "N=64 legacy", "N=64 rescaled"], fontsize=9)
    ax.set_ylabel("median $|\\log_{10}(P/P_\\mathrm{CLASS})|$")
    ax.set_title("Phase 38 — absolute amplitude error: raw vs corrected (avg over seeds)")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_n32_vs_n64(ms, out):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for i, mode in enumerate(("legacy", "rescaled")):
        for seed in (42, 137, 271):
            m32 = _find(ms, 32, seed, mode)
            m64 = _find(ms, 64, seed, mode)
            ks32 = _arr(m32, "ks_hmpc")
            ks64 = _arr(m64, "ks_hmpc")
            r32 = _arr(m32, "pk_corrected_mpc_h3") / _arr(m32, "pk_class_mpc_h3")
            r64 = _arr(m64, "pk_corrected_mpc_h3") / _arr(m64, "pk_class_mpc_h3")
            ax.semilogx(ks32, r32, "--o", color=f"C{i}", alpha=0.4,
                        label=f"N=32 {mode} (seed {seed})" if seed == 42 else None)
            ax.semilogx(ks64, r64, "-s", color=f"C{i}", alpha=0.7,
                        label=f"N=64 {mode} (seed {seed})" if seed == 42 else None)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.axvspan(BAO_K_MIN, BAO_K_MAX, color="gold", alpha=0.12, label="BAO band")
    ax.set_xlabel("$k\\;\\mathrm{[h/Mpc]}$")
    ax.set_ylabel("$P_\\mathrm{c}/P_\\mathrm{CLASS}$")
    ax.set_title("Phase 38 — cross-resolution consistency (N=32³ vs N=64³)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_legacy_vs_rescaled(ms, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, n in zip(axes, (32, 64)):
        for mode, color in zip(("legacy", "rescaled"), ("C0", "C3")):
            m = _find(ms, n, 42, mode)
            ks = _arr(m, "ks_hmpc")
            r = _arr(m, "pk_corrected_mpc_h3") / _arr(m, "pk_class_mpc_h3")
            ax.semilogx(ks, r, "-o", color=color,
                        label=f"{mode} vs CLASS({'z=0' if mode=='legacy' else 'z=49'})")
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.axvspan(BAO_K_MIN, BAO_K_MAX, color="gold", alpha=0.12, label="BAO band")
        ax.set_title(f"$N = {n}^3$, seed 42, IC")
        ax.set_xlabel("$k\\;\\mathrm{[h/Mpc]}$")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=9, loc="best")
    axes[0].set_ylabel("$P_\\mathrm{c}/P_\\mathrm{CLASS}$")
    fig.suptitle("Phase 38 — legacy vs rescaled convention equivalence")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True,
                    help="Path to target/phase38/per_measurement.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.metrics).read_text())
    ms = data["measurements"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("pk_class_vs_gadget.png",     plot_pk_class_vs_gadget),
        ("ratio_pm_pc_vs_class.png",   plot_ratio_pm_pc_vs_class),
        ("abs_error_before_after.png", plot_abs_error_before_after),
        ("n32_vs_n64.png",             plot_n32_vs_n64),
        ("legacy_vs_rescaled.png",     plot_legacy_vs_rescaled),
    ]
    for name, fn in targets:
        path = out_dir / name
        fn(ms, path)
        print(f"[plot_phase38] → {path}")


if __name__ == "__main__":
    main()
