#!/usr/bin/env python3
"""Phase 44 — comparación A/B entre `Psi2Variant::Fixed` y
`Psi2Variant::LegacyBuggy` a partir de `target/phase44/per_snapshot_metrics.json`.

Genera:
  - `figures/phase44_metrics_vs_a.png`  (4 paneles: δ_rms, v_rms, |log Pc/Pref|, growth_lowk)
  - `figures/phase44_summary.csv`
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
JSON_PATH = REPO_ROOT / "target" / "phase44" / "per_snapshot_metrics.json"
FIG_DIR = REPO_ROOT / "experiments" / "nbody" / "phase44_2lpt_audit" / "figures"
CSV_PATH = FIG_DIR / "phase44_summary.csv"


def main() -> int:
    if not JSON_PATH.exists():
        print(f"[plot_ab_comparison] no encuentro {JSON_PATH}", file=sys.stderr)
        return 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = json.loads(JSON_PATH.read_text())
    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    for v in by_variant.values():
        v.sort(key=lambda r: r["a"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    metrics = [
        ("delta_rms", r"$\delta_{\rm rms}$", True),
        ("v_rms", r"$v_{\rm rms}$ (code units)", True),
        ("median_log_pcref", r"median $|\log_{10}(P_c/P_{\rm ref})|$", False),
        ("growth_ratio_lowk", r"$\langle P(k,a)/P(k,a_{\rm ini})\rangle / [D(a)/D_{\rm ini}]^2$ @ $k<0.1$", True),
    ]
    colors = {"fixed": "#1f77b4", "legacy_buggy": "#d62728"}
    markers = {"fixed": "o", "legacy_buggy": "s"}
    labels = {"fixed": "Ψ² Fixed (Phase 44)", "legacy_buggy": "Ψ² Legacy buggy"}

    for ax, (key, title, log_y) in zip(axes.flat, metrics):
        for variant, series in by_variant.items():
            xs = [r["a"] for r in series]
            ys = [r[key] for r in series]
            ax.plot(
                xs, ys,
                marker=markers.get(variant, "o"),
                color=colors.get(variant, "k"),
                label=labels.get(variant, variant),
                linewidth=2,
                markersize=8,
            )
        ax.set_xlabel(r"scale factor $a$")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=9)

    fig.suptitle("Phase 44 — Fix 2LPT vs Legacy buggy (N=32, seed=42, TreePM+ε_phys=0.01)",
                 fontsize=12)
    out = FIG_DIR / "phase44_metrics_vs_a.png"
    fig.savefig(out, dpi=140)
    print(f"[plot_ab_comparison] escrito {out}")

    # CSV resumen
    with CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "variant", "a", "delta_rms", "v_rms", "median_log_pcref",
            "mean_pcref", "cv_pcref", "growth_ratio_lowk",
        ])
        for variant, series in by_variant.items():
            for r in series:
                w.writerow([
                    variant, f"{r['a']:.6f}",
                    f"{r['delta_rms']:.6e}", f"{r['v_rms']:.6e}",
                    f"{r['median_log_pcref']:.6e}", f"{r['mean_pcref']:.6e}",
                    f"{r['cv_pcref']:.6e}", f"{r['growth_ratio_lowk']:.6e}",
                ])
    print(f"[plot_ab_comparison] escrito {CSV_PATH}")

    # Tabla resumen a stdout
    print("\n=== Phase 44 summary ===")
    for variant, series in by_variant.items():
        for r in series:
            print(
                f"  {variant:13s}  a={r['a']:.3f}  "
                f"δ_rms={r['delta_rms']:.3e}  v_rms={r['v_rms']:.3e}  "
                f"err={r['median_log_pcref']:.3e}  growth_lowk={r['growth_ratio_lowk']:.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
