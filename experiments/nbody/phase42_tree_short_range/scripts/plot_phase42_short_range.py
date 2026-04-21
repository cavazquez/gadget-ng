#!/usr/bin/env python3
"""Genera las 5 figuras + 1 CSV de Phase 42 desde `target/phase42/`.

Lee `per_snapshot_metrics.json` producido por
`phase42_tree_short_range.rs` y produce:

    1. delta_rms_vs_a_by_variant.png       — dinámica: δ_rms(a) por variante.
    2. v_rms_vs_a_by_variant.png           — dispersión de velocidades.
    3. ratio_corrected_vs_ref_by_variant.png — 3 paneles a ∈ {0.02, 0.05, 0.10},
                                              P_c/P_ref por variante.
    4. growth_vs_theory.png                — ⟨P(k_low,a)/P(k_low,a_init)⟩ vs
                                              [D(a)/D(a_init)]² por variante.
    5. nonlinearity_onset.png              — δ_rms contra umbral 0.3 por variante.
    6. phase42_summary.csv                 — una fila por (variante, a) con
                                              δ_rms, v_rms, median|log10(P_c/P_ref)|
                                              y growth_rel_err.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OMEGA_M = 0.315
OMEGA_L = 0.685
A_INIT = 0.02
K_MAX_LOW = 0.1


def growth_factor_cpt92(a: float, omega_m: float = OMEGA_M, omega_l: float = OMEGA_L) -> float:
    omega_m_a = omega_m * a ** -3 / (omega_m * a ** -3 + omega_l)
    omega_l_a = omega_l / (omega_m * a ** -3 + omega_l)
    return (
        2.5
        * a
        * omega_m_a
        / (
            omega_m_a ** (4.0 / 7.0)
            - omega_l_a
            + (1.0 + omega_m_a / 2.0) * (1.0 + omega_l_a / 70.0)
        )
    )


def d_ratio_sq(a: float, a_ref: float = A_INIT) -> float:
    return (growth_factor_cpt92(a) / growth_factor_cpt92(a_ref)) ** 2


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text())


def find_snap(snaps, variant, a):
    for s in snaps:
        if s["variant"] == variant and abs(s["a_target"] - a) < 1e-9:
            return s
    return None


def median_abs_log_err(snap) -> float:
    pc = np.asarray(snap["pk_corrected_mpc_h3"])
    pr = np.asarray(snap["pk_reference_mpc_h3"])
    mask = (pc > 0) & (pr > 0)
    if not mask.any():
        return float("nan")
    return float(np.median(np.abs(np.log10(pc[mask] / pr[mask]))))


def mean_growth_low_k(snap_ic, snap_a, k_max: float = K_MAX_LOW):
    if snap_ic is None or snap_a is None:
        return None
    ks_ic = np.asarray(snap_ic["ks_hmpc"])
    ks_a = np.asarray(snap_a["ks_hmpc"])
    pc_ic = np.asarray(snap_ic["pk_corrected_mpc_h3"])
    pc_a = np.asarray(snap_a["pk_corrected_mpc_h3"])
    # Intersección aproximada de bins
    ratios = []
    for k, p_a in zip(ks_a, pc_a):
        if k > k_max:
            break
        j = int(np.argmin(np.abs(ks_ic - k)))
        if abs(ks_ic[j] - k) < 1e-9 and pc_ic[j] > 0 and p_a > 0:
            ratios.append(p_a / pc_ic[j])
    if not ratios:
        return None
    return float(np.mean(ratios))


# ── Figuras ──────────────────────────────────────────────────────────────────

VARIANT_STYLE = {
    "pm_eps0": dict(color="#555555", marker="o", linestyle="-", label=r"PM (ε=0, baseline)"),
    "treepm_eps001": dict(color="#d62728", marker="s", linestyle="--", label=r"TreePM, ε=0.01 Mpc/h"),
    "treepm_eps002": dict(color="#1f77b4", marker="^", linestyle="-.", label=r"TreePM, ε=0.02 Mpc/h"),
    "treepm_eps005": dict(color="#2ca02c", marker="D", linestyle=":", label=r"TreePM, ε=0.05 Mpc/h"),
}


def style_for(variant: str) -> dict:
    if variant in VARIANT_STYLE:
        return VARIANT_STYLE[variant]
    return dict(color="black", marker="o", linestyle="-", label=variant)


def fig1_delta_rms(matrix, outdir: Path):
    snaps = matrix["snapshots"]
    variants = matrix["variants"]
    a_list = sorted({s["a_target"] for s in snaps})

    fig, ax = plt.subplots(figsize=(7, 5))
    for v in variants:
        ys = []
        for a in a_list:
            s = find_snap(snaps, v, a)
            ys.append(s["delta_rms"] if s else np.nan)
        st = style_for(v)
        ax.plot(a_list, ys, **st, linewidth=2, markersize=8)
    ax.axhline(1.0, color="red", linestyle=":", alpha=0.6, label="δ_rms = 1 (colapso)")
    ax.axhline(0.3, color="gray", linestyle=":", alpha=0.4, label="umbral no-lineal (0.3)")
    ax.set_xlabel("a (factor de escala)")
    ax.set_ylabel(r"$\delta_{\mathrm{rms}}(a)$")
    ax.set_title(f"Phase 42 — Dinámica δ_rms vs a, N={matrix['n']}³")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "delta_rms_vs_a_by_variant.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig2_v_rms(matrix, outdir: Path):
    snaps = matrix["snapshots"]
    variants = matrix["variants"]
    a_list = sorted({s["a_target"] for s in snaps})

    fig, ax = plt.subplots(figsize=(7, 5))
    for v in variants:
        ys = []
        for a in a_list:
            s = find_snap(snaps, v, a)
            ys.append(s["v_rms"] if s else np.nan)
        st = style_for(v)
        ax.plot(a_list, ys, **st, linewidth=2, markersize=8)
    ax.set_xlabel("a (factor de escala)")
    ax.set_ylabel(r"$v_{\mathrm{rms}}$ (unidades internas)")
    ax.set_title(f"Phase 42 — Dispersión de velocidades v_rms(a), N={matrix['n']}³")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    path = outdir / "v_rms_vs_a_by_variant.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig3_ratio(matrix, outdir: Path):
    snaps = matrix["snapshots"]
    variants = matrix["variants"]
    a_list = sorted({s["a_target"] for s in snaps})

    fig, axes = plt.subplots(1, len(a_list), figsize=(5 * len(a_list), 4.5), sharey=True)
    if len(a_list) == 1:
        axes = [axes]
    for ax, a in zip(axes, a_list):
        for v in variants:
            s = find_snap(snaps, v, a)
            if s is None:
                continue
            ks = np.asarray(s["ks_hmpc"])
            pc = np.asarray(s["pk_corrected_mpc_h3"])
            pr = np.asarray(s["pk_reference_mpc_h3"])
            mask = (pc > 0) & (pr > 0)
            if not mask.any():
                continue
            st = style_for(v)
            ax.loglog(ks[mask], pc[mask] / pr[mask], **st, linewidth=1.8, markersize=5, alpha=0.8)
        ax.axhline(1.0, color="red", linestyle=":", alpha=0.5)
        ax.set_xlabel("k [h/Mpc]")
        ax.set_title(f"a = {a:.2f}")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$P_c(k) / P_{\mathrm{ref}}(k)$")
    axes[-1].legend(loc="best", fontsize=7)
    fig.suptitle(f"Phase 42 — P_c / P_ref por variante, N={matrix['n']}³", y=1.02)
    fig.tight_layout()
    path = outdir / "ratio_corrected_vs_ref_by_variant.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def fig4_growth(matrix, outdir: Path):
    snaps = matrix["snapshots"]
    variants = matrix["variants"]
    a_evolved = sorted({s["a_target"] for s in snaps if s["a_target"] > A_INIT})

    fig, ax = plt.subplots(figsize=(7, 5))
    theory_x = np.linspace(min(a_evolved + [A_INIT]) * 0.95, max(a_evolved) * 1.05, 50)
    theory_y = [d_ratio_sq(a) for a in theory_x]
    ax.plot(theory_x, theory_y, color="black", linestyle="-", linewidth=2, label=r"teoría $[D(a)/D(a_i)]^2$")

    for v in variants:
        ys = []
        xs = []
        ic = find_snap(snaps, v, A_INIT)
        for a in a_evolved:
            s = find_snap(snaps, v, a)
            g = mean_growth_low_k(ic, s)
            if g is not None:
                xs.append(a)
                ys.append(g)
        st = style_for(v)
        ax.plot(xs, ys, **st, linewidth=2, markersize=8)
    ax.set_xlabel("a")
    ax.set_ylabel(r"$\langle P(k_{\mathrm{low}},a)/P(k_{\mathrm{low}},a_i)\rangle$")
    ax.set_title(
        f"Phase 42 — crecimiento en bajo-k (k ≤ {K_MAX_LOW} h/Mpc) vs teoría lineal, N={matrix['n']}³"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "growth_vs_theory.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig5_nonlinearity_onset(matrix, outdir: Path):
    """Barra: δ_rms(a=0.10) por variante, con el umbral crítico 1.0."""
    snaps = matrix["snapshots"]
    variants = matrix["variants"]

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = []
    ys = []
    colors = []
    for v in variants:
        s = find_snap(snaps, v, 0.10)
        if s is None:
            continue
        xs.append(v)
        ys.append(s["delta_rms"])
        colors.append(style_for(v)["color"])
    x_idx = np.arange(len(xs))
    ax.bar(x_idx, ys, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(1.0, color="red", linestyle="--", label="δ_rms = 1 (colapso)")
    ax.axhline(0.3, color="gray", linestyle=":", label="umbral no-lineal")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(xs, rotation=20)
    ax.set_ylabel(r"$\delta_{\mathrm{rms}}(a=0.10)$")
    ax.set_title(f"Phase 42 — onset de no-linealidad a a=0.10, N={matrix['n']}³")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = outdir / "nonlinearity_onset.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_csv(matrix, outdir: Path) -> Path:
    snaps = matrix["snapshots"]
    variants = matrix["variants"]
    a_list = sorted({s["a_target"] for s in snaps})

    path = outdir / "phase42_summary.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "variant",
            "eps_physical_mpc_h",
            "a_target",
            "a_actual",
            "delta_rms",
            "v_rms",
            "median_abs_log10_err_corrected",
            "growth_ratio_low_k",
            "growth_theory_d_sq",
            "growth_rel_err",
        ])
        for v in variants:
            ic = find_snap(snaps, v, A_INIT)
            for a in a_list:
                s = find_snap(snaps, v, a)
                if s is None:
                    continue
                err = median_abs_log_err(s)
                g_meas = mean_growth_low_k(ic, s) if a > A_INIT else 1.0
                g_th = d_ratio_sq(a) if a > A_INIT else 1.0
                rel = abs(g_meas - g_th) / g_th if g_meas is not None else float("nan")
                w.writerow([
                    v,
                    f"{s['eps_physical_mpc_h']:.4f}",
                    f"{a:.4f}",
                    f"{s['a_actual']:.6f}",
                    f"{s['delta_rms']:.6e}",
                    f"{s['v_rms']:.6e}",
                    f"{err:.6e}",
                    f"{g_meas:.6e}" if g_meas is not None else "nan",
                    f"{g_th:.6e}",
                    f"{rel:.6e}" if g_meas is not None else "nan",
                ])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("target/phase42/per_snapshot_metrics.json"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/nbody/phase42_tree_short_range/figures"),
    )
    args = parser.parse_args()

    if not args.matrix.exists():
        print(f"ERROR: {args.matrix} no existe", file=sys.stderr)
        sys.exit(1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    matrix = load_matrix(args.matrix)

    produced = [
        fig1_delta_rms(matrix, args.outdir),
        fig2_v_rms(matrix, args.outdir),
        fig3_ratio(matrix, args.outdir),
        fig4_growth(matrix, args.outdir),
        fig5_nonlinearity_onset(matrix, args.outdir),
        write_csv(matrix, args.outdir),
    ]
    print("[phase42] artefactos generados:")
    for p in produced:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
