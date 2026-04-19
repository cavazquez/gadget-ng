#!/usr/bin/env python3
"""
plot_growth.py — Crecimiento lineal de estructura vs factor de crecimiento D(a).

Carga diagnostics.jsonl y compara el crecimiento de delta_rms(a) con la
predicción del crecimiento lineal D(a)/D(a_init).

Para EdS (Ω_m=1, Ω_Λ=0): D(a) ∝ a → ratio = a/a_init.

Uso:
    python3 plot_growth.py --results-dir results/
    python3 plot_growth.py --diag-file results/eds_N32_ns-2_pm/diagnostics.jsonl
"""

import argparse
import json
import pathlib
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib no disponible; se omitirán las figuras")


def load_diagnostics(path: pathlib.Path):
    """Carga diagnostics.jsonl → lista de dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)
    return records


def extract_cosmo_diag(records):
    """Extrae a, delta_rms, v_rms de los registros de diagnóstico."""
    a_arr, delta_arr, vrms_arr = [], [], []
    for r in records:
        cosmo = r.get("cosmo")
        if cosmo is None:
            continue
        a = cosmo.get("a")
        delta = cosmo.get("delta_rms")
        vrms = cosmo.get("v_rms")
        if a is not None and delta is not None:
            a_arr.append(a)
            delta_arr.append(delta)
            vrms_arr.append(vrms if vrms is not None else 0.0)
    return np.array(a_arr), np.array(delta_arr), np.array(vrms_arr)


def plot_growth(diag_file: pathlib.Path, out_dir: pathlib.Path,
                omega_m: float = 1.0, omega_lambda: float = 0.0, label: str = ""):
    records = load_diagnostics(diag_file)
    a, delta, vrms = extract_cosmo_diag(records)

    if len(a) < 2:
        print(f"  [WARN] {diag_file}: menos de 2 puntos cosmológicos")
        return {}

    a0 = a[0]
    delta0 = delta[0] if delta[0] > 0 else delta[delta > 0][0] if (delta > 0).any() else 1.0

    # Predicción lineal EdS: D(a)/D(a0) = a/a0.
    # Para ΛCDM: aproximación D(a) ≈ a · g(a) donde g es la función de supresión,
    # pero para esta fase usamos EdS (a/a0) que es correcto para Ω_Λ=0.
    D_ratio = a / a0  # válido para EdS

    delta_ratio = delta / delta0

    # Tasa de crecimiento observada.
    measured_growth = delta_ratio[-1] if delta_ratio[-1] > 0 else float("nan")
    expected_growth = D_ratio[-1]

    print(f"\n  === Crecimiento lineal: {label or diag_file.stem} ===")
    print(f"  a_init = {a0:.4f},  a_final = {a[-1]:.4f}")
    print(f"  D(a)/D(a_init) esperado (EdS): {expected_growth:.4f}")
    print(f"  delta_rms(a_final)/delta_rms(a_init) medido: {measured_growth:.4f}")
    print(f"  Ratio medido/esperado: {measured_growth/expected_growth:.4f}")
    print(f"\n  {'a':>8}  {'delta_rms':>12}  {'delta/delta0':>13}  {'D(a)/D(a0)':>12}")
    print(f"  {'-'*52}")
    for i in range(len(a)):
        print(f"  {a[i]:8.4f}  {delta[i]:12.4e}  {delta_ratio[i]:13.4f}  {D_ratio[i]:12.4f}")

    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Crecimiento lineal — {label or diag_file.stem}", fontsize=13)

        ax = axes[0]
        ax.plot(a, delta_ratio, "o-", color="steelblue", ms=5, label="Medido δ_rms(a)/δ_rms(a₀)")
        ax.plot(a, D_ratio, "--", color="tomato", label="Teoría D(a)/D(a₀) = a/a₀  (EdS)")
        ax.set_xlabel("Factor de escala  a")
        ax.set_ylabel("Amplitud relativa")
        ax.set_title("Crecimiento de estructura")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ratio_measured_theory = delta_ratio / D_ratio
        ax.plot(a, ratio_measured_theory, "s-", color="darkorange", ms=5)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
        ax.axhline(0.9, color="green", linestyle=":", alpha=0.5, label="±10%")
        ax.axhline(1.1, color="green", linestyle=":", alpha=0.5)
        ax.set_xlabel("Factor de escala  a")
        ax.set_ylabel("(δ_rms medido) / (D(a) teórico)")
        ax.set_title("Ratio medido / esperado")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 2)

        fig.tight_layout()
        out_fig = out_dir / f"growth_linear_{label or 'result'}.png"
        fig.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f"  Figura guardada: {out_fig}")

        # También plotear v_rms.
        if vrms.sum() > 0:
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.plot(a, vrms, "v-", color="purple", ms=5, label="v_rms peculiar")
            ax2.set_xlabel("Factor de escala  a")
            ax2.set_ylabel("v_rms  [unidades internas]")
            ax2.set_title(f"Velocidad peculiar RMS — {label}")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            out_v = out_dir / f"vrms_{label or 'result'}.png"
            fig2.savefig(out_v, dpi=150)
            plt.close(fig2)

    return {
        "label": label,
        "a_init": float(a0),
        "a_final": float(a[-1]),
        "expected_growth_EdS": float(expected_growth),
        "measured_growth": float(measured_growth),
        "ratio": float(measured_growth / expected_growth) if expected_growth > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Plotear crecimiento lineal")
    parser.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--diag-file", type=pathlib.Path, default=None)
    parser.add_argument("--omega-m", type=float, default=1.0)
    parser.add_argument("--omega-lambda", type=float, default=0.0)
    args = parser.parse_args()

    summaries = []

    if args.diag_file:
        s = plot_growth(args.diag_file, args.diag_file.parent,
                        args.omega_m, args.omega_lambda,
                        label=args.diag_file.parent.name)
        summaries.append(s)
    else:
        for subdir in sorted(args.results_dir.iterdir()):
            diag_file = subdir / "diagnostics.jsonl"
            if not diag_file.exists():
                continue
            s = plot_growth(diag_file, subdir,
                            args.omega_m, args.omega_lambda, label=subdir.name)
            if s:
                summaries.append(s)

    print("\n=== RESUMEN CRECIMIENTO ===")
    for s in summaries:
        print(f"  {s.get('label','?'):40s}  "
              f"ratio={s.get('ratio', float('nan')):.3f}  "
              f"a: {s.get('a_init', 0):.3f}→{s.get('a_final', 0):.3f}")


if __name__ == "__main__":
    main()
