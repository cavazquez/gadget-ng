#!/usr/bin/env python3
"""Genera las 5 figuras obligatorias de Phase 41 desde `target/phase41/`.

Lee `per_snapshot_metrics.json` producido por los tests Rust de Phase 41 y
genera figuras tipo paper (Springel 2005, Crocce et al. 2006) comparando el
régimen de shot-noise vs señal para `N ∈ {32, 64, 128, 256}`.

Figuras:
  1. pk_vs_pshot_by_N.png              — P_corrected(k, a_init) vs P_shot por N
  2. ratio_corrected_vs_ref_by_N.png   — P_c/P_ref en grilla N × snapshot
  3. spectral_error_vs_N.png           — median|log10(P_c/P_ref)| vs N
  4. growth_ratio_low_k_vs_theory.png  — ⟨P(k_low,a)/P(k_low,a_init)⟩ vs [D/D]²
  5. signal_to_noise_transition.png    — S/N(k_min) vs N (umbral S/N=1)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── CPT92 growth factor (para curva teórica en figura 4) ────────────────────
def growth_factor_cpt92(a: float, omega_m: float = 0.315, omega_l: float = 0.685) -> float:
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


def d_ratio_sq(a: float, a_ref: float) -> float:
    return (growth_factor_cpt92(a) / growth_factor_cpt92(a_ref)) ** 2


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text())


def filter_snaps(snaps, n=None, seed=None, mode=None, a=None):
    out = []
    for s in snaps:
        if n is not None and s["n"] != n:
            continue
        if seed is not None and s["seed"] != seed:
            continue
        if mode is not None and s["mode"] != mode:
            continue
        if a is not None and abs(s["a_target"] - a) > 1e-9:
            continue
        out.append(s)
    return out


def avg_spectrum(snaps, key):
    if not snaps:
        return np.array([]), np.array([])
    ks = np.asarray(snaps[0]["ks_hmpc"])
    arrs = []
    for s in snaps:
        arr = np.asarray(s[key])
        if arr.shape == ks.shape:
            arrs.append(arr)
    if not arrs:
        return ks, np.array([])
    return ks, np.mean(arrs, axis=0)


# ── Figura 1: P(k) vs P_shot, estilo Springel ───────────────────────────────
def fig1_pk_vs_pshot(snaps, out: Path, n_values):
    n_plots = len(n_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.0 * n_plots, 4.2), sharey=True)
    if n_plots == 1:
        axes = [axes]
    for ax, n in zip(axes, n_values):
        for mode, color in (("legacy", "tab:blue"), ("z0_sigma8", "tab:red")):
            sel = filter_snaps(snaps, n=n, mode=mode, a=0.02)
            k, pc = avg_spectrum(sel, "pk_corrected_mpc_h3")
            _, pr = avg_spectrum(sel, "pk_reference_mpc_h3")
            if len(k) == 0:
                continue
            ax.loglog(k, pc, color=color, lw=1.5, label=f"P_corrected ({mode})")
            ax.loglog(k, pr, color=color, lw=1.0, ls=":", alpha=0.6, label=f"P_ref ({mode})")
            if sel:
                p_shot = sel[0]["p_shot_mpc_h3"]
                ax.axhline(p_shot, color=color, ls="--", lw=0.9, alpha=0.5)
        ax.set_xlabel(r"k  [h/Mpc]")
        ax.set_title(f"N = {n}³  (IC, a=0.02)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel(r"P(k)  [(Mpc/h)$^3$]")
    fig.suptitle("Figura 1 — P(k) corregido vs shot-noise P_shot = V/N (IC)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Figura 2: P_corrected/P_ref en grilla N × snapshot ──────────────────────
def fig2_ratio_grid(snaps, out: Path, n_values, a_values):
    fig, axes = plt.subplots(
        len(n_values), len(a_values),
        figsize=(3.4 * len(a_values), 2.6 * len(n_values)),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)
    for i, n in enumerate(n_values):
        for j, a in enumerate(a_values):
            ax = axes[i, j]
            for mode, color in (("legacy", "tab:blue"), ("z0_sigma8", "tab:red")):
                sel = filter_snaps(snaps, n=n, mode=mode, a=a)
                k, pc = avg_spectrum(sel, "pk_corrected_mpc_h3")
                _, pr = avg_spectrum(sel, "pk_reference_mpc_h3")
                if len(k) == 0 or len(pr) == 0:
                    continue
                ax.semilogx(k, pc / pr, color=color, lw=1.3, label=mode)
            ax.axhline(1.0, color="k", lw=0.6, alpha=0.5)
            ax.set_ylim(0.0, 3.0)
            if i == 0:
                ax.set_title(f"a = {a:.2f}")
            if j == 0:
                ax.set_ylabel(f"N = {n}³\nP_c / P_ref")
            if i == len(n_values) - 1:
                ax.set_xlabel(r"k  [h/Mpc]")
            ax.grid(True, which="both", alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Figura 2 — P_corrected / P_ref por resolución y snapshot", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Figura 3: error espectral vs N ──────────────────────────────────────────
def fig3_error_vs_N(snaps, out: Path, n_values, a_values):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for mode, marker in (("legacy", "o"), ("z0_sigma8", "s")):
        for a, color in zip(a_values, ["tab:green", "tab:orange", "tab:purple"]):
            errs = []
            xs = []
            for n in n_values:
                sel = filter_snaps(snaps, n=n, mode=mode, a=a)
                vals = [s["median_abs_log10_err_corrected"] for s in sel if np.isfinite(s["median_abs_log10_err_corrected"])]
                if vals:
                    errs.append(np.mean(vals))
                    xs.append(n)
            if errs:
                ax.loglog(xs, errs, marker=marker, color=color,
                          label=f"{mode} a={a:.2f}", lw=1.2, ms=6,
                          ls="-" if mode == "z0_sigma8" else "--")
    ax.set_xlabel("N (grid / cbrt(N_particles))")
    ax.set_ylabel(r"median $|\log_{10}(P_c / P_{\rm ref})|$")
    ax.set_title("Figura 3 — Error espectral vs resolución")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Figura 4: ratio de crecimiento vs teoría ΛCDM ───────────────────────────
def fig4_growth(snaps, out: Path, n_values, a_values, k_low_max=0.1):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    # Curvas teóricas [D(a)/D(a_init)]² continuo
    a_grid = np.linspace(0.02, 0.12, 60)
    d2 = np.asarray([d_ratio_sq(a, 0.02) for a in a_grid])
    ax.plot(a_grid, d2, "k-", lw=1.1, alpha=0.7, label=r"$[D(a)/D(a_{\rm init})]^2$  (ΛCDM)")

    markers = {32: "o", 64: "s", 128: "D", 256: "^"}
    for n in n_values:
        xs = [0.02]
        ys = [1.0]
        for a in [aa for aa in a_values if aa > 0.02]:
            sel_ev = filter_snaps(snaps, n=n, mode="z0_sigma8", a=a)
            sel_ic = filter_snaps(snaps, n=n, mode="z0_sigma8", a=0.02)
            if not sel_ev or not sel_ic:
                continue
            # Promedio sobre seeds.
            rs = []
            for sev in sel_ev:
                sic = next((x for x in sel_ic if x["seed"] == sev["seed"]), None)
                if sic is None:
                    continue
                k = np.asarray(sev["ks_hmpc"])
                mask = (k > 0) & (k <= k_low_max)
                if mask.sum() == 0:
                    continue
                p_ev = np.asarray(sev["pk_corrected_mpc_h3"])[mask]
                p_ic = np.asarray(sic["pk_corrected_mpc_h3"])[mask]
                ok = (p_ic > 0) & (p_ev > 0)
                if not ok.any():
                    continue
                rs.append(float(np.mean(p_ev[ok] / p_ic[ok])))
            if rs:
                xs.append(a)
                ys.append(float(np.mean(rs)))
        ax.plot(xs, ys, markers.get(n, "x") + "-", ms=7,
                label=f"N={n}³ (Z0Sigma8)")

    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$\langle P(k_{\rm low},a)/P(k_{\rm low},a_{\rm init})\rangle$")
    ax.set_title(f"Figura 4 — Crecimiento lineal (k ≤ {k_low_max} h/Mpc) vs teoría")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Figura 5: transición de shot-noise → señal ──────────────────────────────
def fig5_sn_transition(snaps, out: Path, n_values):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for mode, marker, color in (
        ("legacy", "o", "tab:blue"),
        ("z0_sigma8", "s", "tab:red"),
    ):
        xs = []
        ys = []
        for n in n_values:
            sel = filter_snaps(snaps, n=n, mode=mode, a=0.02, seed=42)
            if not sel:
                continue
            xs.append(n)
            ys.append(sel[0]["s_n_at_kmin"])
        if xs:
            ax.loglog(xs, ys, marker=marker, color=color, lw=1.4, ms=8,
                      label=f"S/N(k_min), {mode}")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="S/N = 1 (umbral)")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$P_{\rm corrected}(k_{\min})\,/\,P_{\rm shot}$")
    ax.set_title("Figura 5 — Transición shot-noise → señal en IC")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_csv(snaps, out: Path):
    import csv

    cols = [
        "n", "seed", "mode", "a_target", "a_actual",
        "p_shot_mpc_h3", "s_n_at_kmin", "s_n_min",
        "median_abs_log10_err_corrected", "mean_r_corr",
        "std_r_corr", "cv_r_corr", "delta_rms",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in snaps:
            w.writerow({c: s.get(c, "") for c in cols})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path,
                        default=Path("target/phase41/per_snapshot_metrics.json"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("docs/reports/figures/phase41"))
    parser.add_argument("--csv", type=Path,
                        default=Path("docs/reports/figures/phase41/phase41_summary.csv"))
    args = parser.parse_args()

    if not args.matrix.exists():
        print(f"ERROR: {args.matrix} no existe (correr antes los tests Rust)",
              file=sys.stderr)
        sys.exit(1)

    data = load_matrix(args.matrix)
    snaps = data["snapshots"]
    n_values = sorted({s["n"] for s in snaps})
    a_values = sorted({s["a_target"] for s in snaps})

    args.outdir.mkdir(parents=True, exist_ok=True)

    fig1_pk_vs_pshot(snaps, args.outdir / "pk_vs_pshot_by_N.png", n_values)
    fig2_ratio_grid(snaps, args.outdir / "ratio_corrected_vs_ref_by_N.png",
                    n_values, a_values)
    fig3_error_vs_N(snaps, args.outdir / "spectral_error_vs_N.png",
                    n_values, a_values)
    fig4_growth(snaps, args.outdir / "growth_ratio_low_k_vs_theory.png",
                n_values, a_values)
    fig5_sn_transition(snaps, args.outdir / "signal_to_noise_transition.png",
                       n_values)
    write_csv(snaps, args.csv)

    print(f"[phase41] figuras en {args.outdir}  |  csv en {args.csv}")


if __name__ == "__main__":
    main()
