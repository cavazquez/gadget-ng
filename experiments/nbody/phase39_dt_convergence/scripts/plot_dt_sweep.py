#!/usr/bin/env python3
"""Phase 39 — genera las 4 figuras del barrido de `dt` + CSV `dt_vs_error`.

Lee `target/phase39/per_cfg.json` producido por la matriz Rust y emite:

1. `error_vs_dt.png` — `median|log10(P_c/P_ref)|` vs `dt`, log-log, por `a`.
2. `ratio_per_dt.png` — `P_c/P_ref` vs `k` overlay para seed 42.
3. `delta_rms_vs_a.png` — `δ_rms(a)` vs `D(a)/D(a_init)` para seed 42.
4. `cost_vs_precision.png` — runtime vs error mediano en `a=0.05`.

Uso:
    ./plot_dt_sweep.py --per-cfg target/phase39/per_cfg.json \
        --out-dir docs/reports/figures/phase39/
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OMEGA_M = 0.315
OMEGA_L = 0.685
A_INIT = 0.02
BAO_K_MIN = 0.05
BAO_K_MAX = 0.30


def cpt92_g(a, omega_m, omega_l):
    a3 = a ** 3
    denom = omega_m + omega_l * a3
    om_a = omega_m / denom
    ol_a = omega_l * a3 / denom
    return 2.5 * om_a / (
        om_a ** (4.0 / 7.0) - ol_a + (1.0 + om_a / 2.0) * (1.0 + ol_a / 70.0)
    )


def d_of_a(a, omega_m=OMEGA_M, omega_l=OMEGA_L):
    return a * cpt92_g(a, omega_m, omega_l)


def median_abs(xs):
    xs = [abs(x) for x in xs if math.isfinite(x)]
    if not xs:
        return float('nan')
    return statistics.median(xs)


def collect(per_cfg):
    """Indexa entries por (dt, seed)."""
    out = {}
    for e in per_cfg["entries"]:
        out[(float(e["dt"]), int(e["seed"]))] = e
    return out


def snap_for(entry, a_target):
    for s in entry["snapshots"]:
        if abs(s["a_target"] - a_target) < 1e-12:
            return s
    raise RuntimeError(f"snapshot a={a_target} missing")


def figure_error_vs_dt(per_cfg, index, out_path):
    dts = sorted(set(float(d) for d in per_cfg["dts"]), reverse=True)
    seeds = sorted(set(int(s) for s in per_cfg["seeds"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for ax, a_t in zip(axes, [0.05, 0.10]):
        for seed in seeds:
            errs = []
            for dt in dts:
                s = snap_for(index[(dt, seed)], a_t)
                errs.append(s["median_abs_log10_err_corrected"])
            ax.plot(dts, errs, marker="o", alpha=0.4,
                    label=f"seed={seed}", linewidth=1)
        # Media sobre seeds.
        mean_err = []
        for dt in dts:
            vals = [snap_for(index[(dt, s)], a_t)["median_abs_log10_err_corrected"]
                    for s in seeds]
            mean_err.append(statistics.mean(vals))
        ax.plot(dts, mean_err, marker="s", color="black",
                label="mean(seeds)", linewidth=2.0)
        # Referencia O(dt^2) anclada al punto más pequeño.
        dt_ref = min(dts)
        err_ref = mean_err[dts.index(dt_ref)]
        if err_ref > 0:
            ref_line = [err_ref * (dt / dt_ref) ** 2 for dt in dts]
            ax.plot(dts, ref_line, linestyle="--", color="red",
                    label=r"$\propto dt^2$ (ancla $dt_{\min}$)", linewidth=1.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$dt$")
        ax.set_title(f"$a = {a_t:.2f}$")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$\mathrm{median}\,|\log_{10}(P_c/P_{\mathrm{ref}})|$")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Phase 39 — Error espectral corregido vs $dt$ (N=32³, 2LPT, PM, legacy)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def figure_ratio_per_dt(per_cfg, index, out_path, seed=42):
    dts = sorted(set(float(d) for d in per_cfg["dts"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for ax, a_t in zip(axes, [0.05, 0.10]):
        ax.axhspan(0, 0, color="none")  # placeholder
        ax.axvspan(BAO_K_MIN, BAO_K_MAX, color="orange", alpha=0.12,
                   label="BAO band")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        for dt in dts:
            s = snap_for(index[(dt, seed)], a_t)
            ks = s["ks_hmpc"]
            pc = s["pk_corrected_mpc_h3"]
            pr = s["pk_reference_mpc_h3"]
            rcorr = [c / r for c, r in zip(pc, pr) if r > 0]
            ks_plot = [k for k, r in zip(ks, pr) if r > 0]
            ax.plot(ks_plot, rcorr, marker="o", markersize=3,
                    label=f"dt={dt:.1e}", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k$ [h/Mpc]")
        ax.set_title(f"$a = {a_t:.2f}$, seed={seed}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$P_c / P_{\mathrm{ref}}$")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Phase 39 — Ratio corregido vs referencia, overlay 4 dts")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def figure_delta_rms_vs_a(per_cfg, index, out_path, seed=42):
    dts = sorted(set(float(d) for d in per_cfg["dts"]))
    a_vals = [float(a) for a in per_cfg["a_snapshots"]]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    # Predicción lineal (anclada al IC del primer dt disponible).
    ref_entry = index[(dts[0], seed)]
    s_ic = snap_for(ref_entry, A_INIT)
    delta_ic = s_ic["delta_rms"]
    d_ic = d_of_a(A_INIT)
    a_theory = a_vals
    delta_theory = [delta_ic * d_of_a(a) / d_ic for a in a_theory]
    ax.plot(a_theory, delta_theory, color="red", linestyle="--",
            linewidth=1.5, label=r"$\delta_{\rm rms}(a_{\rm init})\cdot D(a)/D(a_{\rm init})$")
    for dt in dts:
        ys = [snap_for(index[(dt, seed)], a)["delta_rms"] for a in a_vals]
        ax.plot(a_vals, ys, marker="o", label=f"dt={dt:.1e}")
    ax.set_xlabel("a")
    ax.set_ylabel(r"$\delta_{\rm rms}(a)$")
    ax.set_yscale("log")
    ax.set_title(f"Phase 39 — $\\delta_{{\\rm rms}}$ vs $a$ (seed={seed}, N=32³)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def figure_cost_vs_precision(per_cfg, index, out_path):
    dts = sorted(set(float(d) for d in per_cfg["dts"]))
    seeds = sorted(set(int(s) for s in per_cfg["seeds"]))
    a_t = 0.05
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = plt.get_cmap("viridis", len(dts))
    for i, dt in enumerate(dts):
        runtimes = []
        errs = []
        for seed in seeds:
            e = index[(dt, seed)]
            runtimes.append(e["runtime_s"])
            errs.append(snap_for(e, a_t)["median_abs_log10_err_corrected"])
        ax.scatter(runtimes, errs, s=38, color=colors(i),
                   label=f"dt={dt:.1e}", alpha=0.85)
        ax.scatter(
            [statistics.mean(runtimes)],
            [statistics.mean(errs)],
            marker="X", s=110, color=colors(i), edgecolor="black", linewidth=0.6,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("runtime por corrida [s]")
    ax.set_ylabel(r"$\mathrm{median}\,|\log_{10}(P_c/P_{\mathrm{ref}})|$ en $a=0.05$")
    ax.set_title("Phase 39 — Costo vs precisión (3 seeds por dt; X = media)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def export_csv(per_cfg, index, out_path):
    dts = sorted(set(float(d) for d in per_cfg["dts"]))
    seeds = sorted(set(int(s) for s in per_cfg["seeds"]))
    a_vals = [float(a) for a in per_cfg["a_snapshots"]]
    rows = ["dt,seed,a_target,median_abs_log10_err_raw,"
            "median_abs_log10_err_corr,mean_r_corr,stdev_r_corr,"
            "delta_rms,v_rms,runtime_s,steps_this_leg"]
    for dt in dts:
        for seed in seeds:
            e = index[(dt, seed)]
            rt = e["runtime_s"]
            for a_t in a_vals:
                s = snap_for(e, a_t)
                rows.append(",".join(str(x) for x in [
                    f"{dt:.6e}", seed, a_t,
                    f"{s['median_abs_log10_err_raw']:.6e}",
                    f"{s['median_abs_log10_err_corrected']:.6e}",
                    f"{s['mean_r_corr']:.6e}",
                    f"{s['stdev_r_corr']:.6e}",
                    f"{s['delta_rms']:.6e}",
                    f"{s['v_rms']:.6e}",
                    f"{rt:.3f}",
                    s["steps_this_leg"],
                ]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cfg", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--csv", default=None,
                    help="Ruta al CSV resumen; default: <out-dir>/../output/dt_vs_error.csv")
    args = ap.parse_args()

    per_cfg = json.loads(Path(args.per_cfg).read_text())
    index = collect(per_cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_error_vs_dt(per_cfg, index, out_dir / "error_vs_dt.png")
    figure_ratio_per_dt(per_cfg, index, out_dir / "ratio_per_dt.png")
    figure_delta_rms_vs_a(per_cfg, index, out_dir / "delta_rms_vs_a.png")
    figure_cost_vs_precision(per_cfg, index, out_dir / "cost_vs_precision.png")

    csv_path = Path(args.csv) if args.csv else out_dir.parent / "output" / "dt_vs_error.csv"
    export_csv(per_cfg, index, csv_path)

    print(f"[phase39] 4 figuras → {out_dir}")
    print(f"[phase39] CSV        → {csv_path}")


if __name__ == "__main__":
    main()
