#!/usr/bin/env python3
"""Phase 37 — Genera las 6 figuras obligatorias a partir del JSON dumpeado
por el test Rust `phase37_growth_rescaled_ics.rs`
(`target/phase37/per_snapshot_metrics.json`).

Figuras:
1. `pk_ic_legacy_vs_rescaled.png` — P_m, P_c, P_ref en snapshot IC
2. `pk_a005_legacy_vs_rescaled.png` — idem a a=0.05
3. `pk_a010_legacy_vs_rescaled.png` — idem a a=0.10
4. `ratio_pc_pref_evolution.png` — P_c/P_ref vs k, 3 épocas, ambos modos
5. `delta_rms_vs_a.png` — δ_rms(a) legacy vs rescaled vs teoría lineal
6. `psi_rms_ic.png` — rms(Ψ) de ICs legacy vs rescaled (opcional/complementaria)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def cpt92_g(a, om, ol):
    a3 = a ** 3
    denom = om + ol * a3
    om_a = om / denom
    ol_a = ol * a3 / denom
    return 2.5 * om_a / (om_a ** (4.0/7.0) - ol_a + (1 + om_a/2) * (1 + ol_a/70))


def d_of_a(a, om, ol):
    return a * cpt92_g(a, om, ol)


def load_matrix(path: Path):
    with path.open() as f:
        return json.load(f)


def filter_snaps(snaps, **kwargs):
    out = []
    for s in snaps:
        ok = True
        for k, v in kwargs.items():
            if s.get(k) != v:
                ok = False
                break
        if ok:
            out.append(s)
    return out


def one_or_none(xs):
    return xs[0] if xs else None


def plot_pk_comparison(matrix, a_target, outfile, n=32, seed=42,
                       ic="2lpt", solver="pm", title=None):
    legacy = one_or_none(filter_snaps(
        matrix, n=n, seed=seed, ic_kind=ic, solver=solver, mode="legacy",
        a_target=a_target,
    ))
    rescaled = one_or_none(filter_snaps(
        matrix, n=n, seed=seed, ic_kind=ic, solver=solver, mode="rescaled",
        a_target=a_target,
    ))
    if not legacy or not rescaled:
        print(f"[plot_phase37] Saltando {outfile.name}: snapshot ausente.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, data, label in zip(axes, [legacy, rescaled], ["legacy", "rescaled"]):
        ks = data["ks_hmpc"]
        pm = data["pk_measured_internal"]
        pc = data["pk_corrected_mpc_h3"]
        pr = data["pk_reference_mpc_h3"]
        ax.loglog(ks, pm, "o--", color="tab:gray",   label="$P_{m}$ (internal)", alpha=0.7)
        ax.loglog(ks, pc, "s-",  color="tab:blue",   label="$P_{c}$ corrected",   lw=1.5)
        ax.loglog(ks, pr, "-",   color="tab:red",    label="$P_{ref}$ EH·$D^2$", lw=1.8)
        med = data["median_abs_log10_err_corrected"]
        ax.set_title(f"{label} — $a_t$={a_target}, median|Δlog|={med:.2f}")
        ax.set_xlabel("k [h/Mpc]")
        ax.set_ylabel("P(k) [(Mpc/h)³]")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(loc="lower left", fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)
    print(f"[plot_phase37] → {outfile}")


def plot_ratio_evolution(matrix, outfile, n=32, seed=42, ic="2lpt", solver="pm"):
    fig, ax = plt.subplots(figsize=(8, 5))
    a_colors = {0.02: "tab:blue", 0.05: "tab:orange", 0.10: "tab:red"}
    for a_t, color in a_colors.items():
        for mode, ls, marker in [("legacy", "--", "o"), ("rescaled", "-", "s")]:
            s = one_or_none(filter_snaps(
                matrix, n=n, seed=seed, ic_kind=ic, solver=solver,
                mode=mode, a_target=a_t,
            ))
            if not s:
                continue
            ks = s["ks_hmpc"]
            ratios = [c / r for c, r in zip(
                s["pk_corrected_mpc_h3"], s["pk_reference_mpc_h3"]
            ) if r > 0]
            if not ratios:
                continue
            ax.loglog(ks[:len(ratios)], ratios, ls=ls, marker=marker,
                      color=color, alpha=0.8,
                      label=f"a={a_t} {mode}")
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("$P_c(k) / P_{ref}(k)$")
    ax.set_title(
        f"Evolución de $P_c/P_{{ref}}$ — N={n}, ic={ic}, solver={solver}, seed={seed}"
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)
    print(f"[plot_phase37] → {outfile}")


def plot_delta_rms(matrix, outfile, om=0.315, ol=0.685, a_init=0.02,
                   n=32, ic="2lpt", solver="pm"):
    fig, ax = plt.subplots(figsize=(8, 5))

    snaps = [s for s in matrix
             if s["n"] == n and s["ic_kind"] == ic and s["solver"] == solver]

    as_sorted = sorted({s["a_target"] for s in snaps})

    for mode, ls, marker, color in [
        ("legacy",   "--", "o", "tab:orange"),
        ("rescaled", "-",  "s", "tab:blue"),
    ]:
        mean_drms = []
        for a_t in as_sorted:
            vs = [s["delta_rms"] for s in snaps
                  if s["mode"] == mode and abs(s["a_target"] - a_t) < 1e-9]
            if not vs:
                mean_drms.append(float("nan"))
            else:
                mean_drms.append(sum(vs) / len(vs))
        ax.plot(as_sorted, mean_drms, ls=ls, marker=marker, color=color,
                label=f"{mode} (seed-mean)")

    # Curva lineal teórica: δ_rms(a)/δ_rms(a_ref) = D(a)/D(a_ref)
    if len(as_sorted) >= 2:
        # Tomamos como referencia el primer snapshot del modo *rescaled* para
        # anclar la curva teórica (criterio físico: en rescaled δ es lineal).
        first = [s for s in snaps
                 if s["mode"] == "rescaled" and abs(s["a_target"] - as_sorted[0]) < 1e-9]
        if first:
            d_ai = d_of_a(as_sorted[0], om, ol)
            anchor = sum(s["delta_rms"] for s in first) / len(first)
            lin = [anchor * d_of_a(a_t, om, ol) / d_ai for a_t in as_sorted]
            ax.plot(as_sorted, lin, ":", color="black",
                    label="Crecimiento lineal $D(a)/D(a_{ref})$ (ancla rescaled IC)")

    ax.set_xlabel("factor de escala $a$")
    ax.set_ylabel(r"$\delta_{\rm rms}$")
    ax.set_yscale("log")
    ax.set_title(f"$\\delta_{{rms}}(a)$ — N={n}, ic={ic}, solver={solver}")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)
    print(f"[plot_phase37] → {outfile}")


def plot_psi_rms_ic(matrix, outfile, om=0.315, ol=0.685, a_init=0.02):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels, legacy_vals, rescaled_vals = [], [], []
    for n in sorted({s["n"] for s in matrix}):
        for ic in ["2lpt", "1lpt"]:
            legs = [s["psi_rms_ic"] for s in matrix
                    if s["n"] == n and s["ic_kind"] == ic and s["mode"] == "legacy"
                    and abs(s["a_target"] - a_init) < 1e-9]
            ress = [s["psi_rms_ic"] for s in matrix
                    if s["n"] == n and s["ic_kind"] == ic and s["mode"] == "rescaled"
                    and abs(s["a_target"] - a_init) < 1e-9]
            if not legs or not ress:
                continue
            labels.append(f"N={n} {ic}")
            legacy_vals.append(sum(legs)/len(legs))
            rescaled_vals.append(sum(ress)/len(ress))

    x = list(range(len(labels)))
    ax.bar([xi - 0.2 for xi in x], legacy_vals, 0.4, color="tab:orange", label="legacy")
    ax.bar([xi + 0.2 for xi in x], rescaled_vals, 0.4, color="tab:blue",   label="rescaled")
    s_theo = d_of_a(a_init, om, ol) / d_of_a(1.0, om, ol)
    for xi, leg in zip(x, legacy_vals):
        ax.plot([xi + 0.2], [leg * s_theo], "k_", markersize=14,
                label="$s \\cdot$ legacy" if xi == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("rms(Ψ) en IC [box]")
    ax.set_yscale("log")
    ax.set_title(f"Desplazamientos iniciales (legacy vs rescaled, s={s_theo:.3e})")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)
    print(f"[plot_phase37] → {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics",
                    default="target/phase37/per_snapshot_metrics.json")
    ap.add_argument("--out-dir", default="docs/reports/figures/phase37")
    ap.add_argument("--omega-m", type=float, default=0.315)
    ap.add_argument("--omega-lambda", type=float, default=0.685)
    ap.add_argument("--a-init", type=float, default=0.02)
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    data = load_matrix(metrics_path)
    snaps = data["snapshots"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_pk_comparison(snaps, 0.02, out_dir / "pk_ic_legacy_vs_rescaled.png",
                       title="P(k) en snapshot IC — N=32, 2LPT, PM, seed=42")
    plot_pk_comparison(snaps, 0.05, out_dir / "pk_a005_legacy_vs_rescaled.png",
                       title="P(k) en a=0.05 — N=32, 2LPT, PM, seed=42")
    plot_pk_comparison(snaps, 0.10, out_dir / "pk_a010_legacy_vs_rescaled.png",
                       title="P(k) en a=0.10 — N=32, 2LPT, PM, seed=42")
    plot_ratio_evolution(snaps, out_dir / "ratio_pc_pref_evolution.png")
    plot_delta_rms(snaps, out_dir / "delta_rms_vs_a.png",
                   om=args.omega_m, ol=args.omega_lambda, a_init=args.a_init)
    plot_psi_rms_ic(snaps, out_dir / "psi_rms_ic.png",
                    om=args.omega_m, ol=args.omega_lambda, a_init=args.a_init)


if __name__ == "__main__":
    main()
