#!/usr/bin/env python3
"""Genera las 6 figuras obligatorias de Phase 40 desde target/phase40/.

Lee `target/phase40/per_snapshot_metrics.json` (producido por los tests Rust
de `phase40_physical_ics_normalization`) y genera las figuras comparando
modo `legacy` vs `z0_sigma8` en 3 snapshots (a=0.02, 0.05, 0.10).

Figuras:
  1. pk_ic_legacy_vs_z0.png           — IC snapshot
  2. pk_a005_legacy_vs_z0.png         — a≈0.05
  3. pk_a010_legacy_vs_z0.png         — a≈0.10
  4. ratio_corrected_vs_ref_per_mode.png
  5. delta_rms_vs_a_legacy_vs_z0_vs_linear.png
  6. sigma8_measured_vs_expected.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text())


def group_by(snaps, seed, mode, a_target):
    """Find single snapshot entry."""
    for s in snaps:
        if (
            s["seed"] == seed
            and s["mode"] == mode
            and abs(s["a_target"] - a_target) < 1e-9
        ):
            return s
    return None


def avg_spectrum(snaps, mode, a_target, key):
    """Average `key` (list-valued) across all seeds for (mode, a_target)."""
    chosen = [
        s
        for s in snaps
        if s["mode"] == mode and abs(s["a_target"] - a_target) < 1e-9
    ]
    if not chosen:
        return np.array([]), np.array([])
    ks = np.asarray(chosen[0]["ks_hmpc"])
    ys = np.mean(
        [np.asarray(s[key]) for s in chosen],
        axis=0,
    )
    return ks, ys


def plot_pk_snapshot(snaps, a_target, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for mode, color, ls_meas, ls_corr in [
        ("legacy", "tab:blue", "-", "--"),
        ("z0_sigma8", "tab:red", "-", "--"),
    ]:
        k, pm = avg_spectrum(snaps, mode, a_target, "pk_measured_internal")
        _, pc = avg_spectrum(snaps, mode, a_target, "pk_corrected_mpc_h3")
        _, pr = avg_spectrum(snaps, mode, a_target, "pk_reference_mpc_h3")
        if len(k) == 0:
            continue
        ax.loglog(k, pc, color=color, lw=1.6, ls=ls_corr, label=f"P_corrected ({mode})")
        ax.loglog(k, pr, color=color, lw=1.1, ls=":", alpha=0.7, label=f"P_ref ({mode})")
    ax.set_xlabel(r"k  [h/Mpc]")
    ax.set_ylabel(r"P(k)  [(Mpc/h)$^3$]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[phase40] {out_path.name}")


def plot_ratio_corrected_vs_ref(snaps, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    a_snaps = [0.02, 0.05, 0.10]
    for ax, a_t in zip(axes, a_snaps):
        for mode, color in [("legacy", "tab:blue"), ("z0_sigma8", "tab:red")]:
            k, pc = avg_spectrum(snaps, mode, a_t, "pk_corrected_mpc_h3")
            _, pr = avg_spectrum(snaps, mode, a_t, "pk_reference_mpc_h3")
            if len(k) == 0:
                continue
            ratio = pc / pr
            ax.semilogx(k, ratio, color=color, lw=1.6, label=mode)
        ax.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel(r"k  [h/Mpc]")
        ax.set_title(f"a = {a_t:.2f}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$P_\mathrm{corrected}/P_\mathrm{ref}$")
    axes[0].legend(fontsize=9, loc="best")
    axes[0].set_yscale("log")
    fig.suptitle("Phase 40 — ratio P_corrected / P_ref (legacy vs z0_sigma8)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[phase40] {out_path.name}")


def plot_delta_rms_vs_a(snaps, out_path: Path, cosmo_omega_m=0.315):
    """δ_rms vs a para legacy y z0_sigma8, con referencia lineal."""
    a_snaps = sorted({s["a_target"] for s in snaps})
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for mode, color in [("legacy", "tab:blue"), ("z0_sigma8", "tab:red")]:
        means = []
        for a in a_snaps:
            vals = [
                s["delta_rms"]
                for s in snaps
                if s["mode"] == mode and abs(s["a_target"] - a) < 1e-9
            ]
            if vals:
                means.append(np.mean(vals))
            else:
                means.append(np.nan)
        ax.loglog(a_snaps, means, "o-", color=color, lw=1.8, label=f"δ_rms ({mode})")

    # Ancla el crecimiento lineal al δ_rms del primer snapshot de cada modo.
    for mode, color in [("legacy", "tab:blue"), ("z0_sigma8", "tab:red")]:
        a_init = a_snaps[0]
        vals_init = [
            s["delta_rms"]
            for s in snaps
            if s["mode"] == mode and abs(s["a_target"] - a_init) < 1e-9
        ]
        if not vals_init:
            continue
        d0 = np.mean(vals_init)
        # D(a) ∝ a en EdS; para ΛCDM usamos aproximación a como primer orden.
        dlin = d0 * np.asarray(a_snaps) / a_init
        ax.loglog(
            a_snaps,
            dlin,
            color=color,
            lw=0.9,
            ls=":",
            alpha=0.6,
            label=f"lineal ∝ a ({mode})",
        )
    ax.set_xlabel("a")
    ax.set_ylabel(r"$\delta_\mathrm{rms}(a)$")
    ax.set_title(r"Phase 40 — $\delta_\mathrm{rms}$ vs a  (legacy vs z0_sigma8 vs lineal)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[phase40] {out_path.name}")


def plot_sigma8_measured_vs_expected(snaps, out_path: Path):
    """σ₈(a_init) medido vs esperado por modo."""
    a_init = min({s["a_target"] for s in snaps})
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    modes = ["legacy", "z0_sigma8"]
    colors = {"legacy": "tab:blue", "z0_sigma8": "tab:red"}
    for mode in modes:
        ic_snaps = [
            s
            for s in snaps
            if s["mode"] == mode and abs(s["a_target"] - a_init) < 1e-9
        ]
        if not ic_snaps:
            continue
        meas = [s["sigma8_measured"] for s in ic_snaps]
        ref = [s["sigma8_from_ref"] for s in ic_snaps]
        ax.scatter(
            [f"{mode}\nP_ref" for _ in ref],
            ref,
            color=colors[mode],
            marker="s",
            s=70,
            alpha=0.6,
            label=f"σ₈ de P_ref ({mode})",
        )
        ax.scatter(
            [f"{mode}\nP_corrected" for _ in meas],
            meas,
            color=colors[mode],
            marker="o",
            s=70,
            label=f"σ₈ medido ({mode})",
        )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma_8(R=8\,\mathrm{Mpc/h})$")
    ax.set_title(r"Phase 40 — $\sigma_8(a_\mathrm{init})$ medido vs referencia lineal")
    ax.grid(True, which="both", alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[phase40] {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("target/phase40/per_snapshot_metrics.json"),
        help="Ruta al JSON con la matriz (default: target/phase40/per_snapshot_metrics.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/reports/figures/phase40"),
        help="Directorio de salida para las figuras",
    )
    args = parser.parse_args()

    if not args.matrix.exists():
        print(
            f"ERROR: matriz no encontrada en {args.matrix}. "
            f"Ejecutá los tests Rust primero: "
            f"`cargo test --release --test phase40_physical_ics_normalization`",
            file=sys.stderr,
        )
        sys.exit(1)

    data = load_matrix(args.matrix)
    snaps = data["snapshots"]
    args.out.mkdir(parents=True, exist_ok=True)

    # Figuras 1-3: P(k) en cada snapshot (legacy vs z0_sigma8).
    plot_pk_snapshot(
        snaps,
        0.02,
        args.out / "pk_ic_legacy_vs_z0.png",
        "Phase 40 — IC snapshot  (a=0.02,  legacy vs z0_sigma8)",
    )
    plot_pk_snapshot(
        snaps,
        0.05,
        args.out / "pk_a005_legacy_vs_z0.png",
        "Phase 40 — a ≈ 0.05  (legacy vs z0_sigma8)",
    )
    plot_pk_snapshot(
        snaps,
        0.10,
        args.out / "pk_a010_legacy_vs_z0.png",
        "Phase 40 — a ≈ 0.10  (legacy vs z0_sigma8)",
    )

    # Figura 4: ratios P_corrected/P_ref por modo.
    plot_ratio_corrected_vs_ref(
        snaps, args.out / "ratio_corrected_vs_ref_per_mode.png"
    )

    # Figura 5: δ_rms(a) vs crecimiento lineal.
    plot_delta_rms_vs_a(snaps, args.out / "delta_rms_vs_a_legacy_vs_z0_vs_linear.png")

    # Figura 6: σ₈(a_init) medido vs esperado.
    plot_sigma8_measured_vs_expected(
        snaps, args.out / "sigma8_measured_vs_expected.png"
    )

    # Export CSV resumen.
    csv_path = args.out / "phase40_summary.csv"
    with csv_path.open("w") as f:
        f.write(
            "mode,seed,a_target,median_abs_log10_err_corrected,mean_r_corr,"
            "delta_rms,v_rms,sigma8_measured,sigma8_from_ref,sigma8_expected\n"
        )
        for s in snaps:
            f.write(
                f"{s['mode']},{s['seed']},{s['a_target']:.4f},"
                f"{s['median_abs_log10_err_corrected']:.6e},"
                f"{s['mean_r_corr']:.6e},"
                f"{s['delta_rms']:.6e},{s['v_rms']:.6e},"
                f"{s['sigma8_measured']:.6e},{s['sigma8_from_ref']:.6e},"
                f"{s['sigma8_expected']:.6e}\n"
            )
    print(f"[phase40] CSV: {csv_path}")
    print("[phase40] todas las figuras generadas en:", args.out)


if __name__ == "__main__":
    main()
