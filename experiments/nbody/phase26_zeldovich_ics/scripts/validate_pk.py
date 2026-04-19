#!/usr/bin/env python3
"""
validate_pk.py — Validación del espectro de potencia inicial P(k).

Carga el archivo power_spectrum.jsonl generado por gadget-ng analyse,
lo compara con el P(k) ∝ k^n_s objetivo y genera la figura.

Uso:
    python3 validate_pk.py --results-dir results/
    python3 validate_pk.py --pk-file results/eds_N32_ns-2_pm/pk_initial.jsonl \
                           --spectral-index -2.0 --amplitude 1e-4 \
                           --box-size 1.0 --grid-size 32
"""

import argparse
import json
import pathlib
import sys
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib no disponible; se omitirán las figuras")


def load_pk(path: pathlib.Path):
    """Carga power_spectrum.jsonl → arrays (k, pk, n_modes)."""
    ks, pks, nms = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ks.append(obj["k"])
            pks.append(obj["pk"])
            nms.append(obj["n_modes"])
    return np.array(ks), np.array(pks), np.array(nms)


def theoretical_pk(k: np.ndarray, spectral_index: float, amplitude: float,
                   box_size: float, grid_size: int) -> np.ndarray:
    """
    P(k) teórico en las mismas unidades que el estimador CIC.

    El generador usa P(n) = amplitude² · |n|^n_s donde n = k·L/(2π).
    El estimador de gadget-ng devuelve P(k) en unidades de L³.

    Para comparar: P_theory(k) = amplitude² · (k·L/(2π))^n_s
    """
    n_modes = k * box_size / (2.0 * np.pi)   # número de modo (flotante)
    n2 = n_modes ** 2
    pk_theory = amplitude ** 2 * np.where(n2 > 0, n2 ** (spectral_index / 2.0), 0.0)
    return pk_theory


def validate_and_plot(pk_file: pathlib.Path,
                      spectral_index: float,
                      amplitude: float,
                      box_size: float,
                      grid_size: int,
                      out_dir: pathlib.Path,
                      label: str = ""):
    k, pk, n_modes = load_pk(pk_file)
    mask = (pk > 0) & (n_modes > 0)
    k_m, pk_m = k[mask], pk[mask]

    if len(k_m) == 0:
        print(f"  [WARN] {pk_file}: no hay bins con señal")
        return {}

    pk_theory = theoretical_pk(k_m, spectral_index, amplitude, box_size, grid_size)

    # Normalización relativa: ajustar la teoría a la amplitud medida (modos bajos).
    # Esto elimina el factor de normalización absoluta y enfoca en la pendiente.
    good = pk_theory > 0
    if good.sum() > 0:
        scale = np.median(pk_m[good] / pk_theory[good])
    else:
        scale = 1.0
    pk_theory_scaled = pk_theory * scale

    # Error relativo por bin.
    rel_err = np.abs(pk_m - pk_theory_scaled) / np.clip(pk_theory_scaled, 1e-100, None)

    print(f"\n  === P(k) Validación: {label or pk_file.name} ===")
    print(f"  {'k':>10}  {'P(k)_meas':>12}  {'P(k)_theory':>12}  {'error_rel':>10}  {'n_modes':>8}")
    print(f"  {'-'*60}")
    for i in range(len(k_m)):
        flag = " ← ALTO" if rel_err[i] > 0.5 else ""
        print(f"  {k_m[i]:10.4f}  {pk_m[i]:12.4e}  {pk_theory_scaled[i]:12.4e}  {rel_err[i]:10.3f}{flag}")

    median_err = float(np.median(rel_err))
    max_err = float(np.max(rel_err))
    print(f"\n  Error relativo mediano: {median_err:.3f}")
    print(f"  Error relativo máximo:  {max_err:.3f}")

    # Pendiente medida en log-log (primeros 4 bins).
    if len(k_m) >= 2:
        n_fit = min(4, len(k_m))
        slope = np.polyfit(np.log(k_m[:n_fit]), np.log(pk_m[:n_fit]), 1)[0]
        print(f"  Pendiente medida (primeros {n_fit} bins): {slope:.2f} vs n_s={spectral_index:.1f}")
    else:
        slope = float("nan")

    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Validación P(k) — {label or pk_file.stem}", fontsize=13)

        ax = axes[0]
        ax.loglog(k_m, pk_m, "o-", label=f"Medido (gadget-ng)", color="steelblue", ms=5)
        ax.loglog(k_m, pk_theory_scaled, "--", label=f"Teoría P(k)∝k^{spectral_index:.0f} (escalada)", color="tomato")
        ax.set_xlabel("k  [2π/L]")
        ax.set_ylabel("P(k)  [L³]")
        ax.set_title("Espectro de potencia")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        ax = axes[1]
        ax.semilogx(k_m, rel_err, "s-", color="orange", ms=5)
        ax.axhline(0.3, color="red", linestyle="--", alpha=0.7, label="30% error")
        ax.axhline(0.1, color="green", linestyle="--", alpha=0.7, label="10% error")
        ax.set_xlabel("k  [2π/L]")
        ax.set_ylabel("Error relativo |P_meas/P_theory − 1|")
        ax.set_title("Error relativo por bin")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(1.5, max_err * 1.2))

        fig.tight_layout()
        out_fig = out_dir / f"pk_initial_vs_target_{label or 'result'}.png"
        fig.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f"  Figura guardada: {out_fig}")

    return {
        "label": label,
        "median_rel_error": median_err,
        "max_rel_error": max_err,
        "measured_slope": float(slope) if not np.isnan(slope) else None,
        "spectral_index": spectral_index,
        "n_bins": int(len(k_m)),
    }


def main():
    parser = argparse.ArgumentParser(description="Validar P(k) de ICs Zel'dovich")
    parser.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results"),
                        help="Directorio raíz con subdirectorios de resultados")
    parser.add_argument("--pk-file", type=pathlib.Path, default=None,
                        help="Archivo pk_initial.jsonl específico (override)")
    parser.add_argument("--spectral-index", type=float, default=-2.0)
    parser.add_argument("--amplitude", type=float, default=1.0e-4)
    parser.add_argument("--box-size", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=32)
    args = parser.parse_args()

    summaries = []

    if args.pk_file:
        s = validate_and_plot(
            args.pk_file, args.spectral_index, args.amplitude,
            args.box_size, args.grid_size,
            args.pk_file.parent, label=args.pk_file.parent.name,
        )
        summaries.append(s)
    else:
        # Buscar todos los subdirectorios con pk_initial.jsonl.
        results_dir = args.results_dir
        for subdir in sorted(results_dir.iterdir()):
            pk_file = subdir / "pk_initial.jsonl"
            if not pk_file.exists():
                # Intentar power_spectrum.jsonl del analyse.
                pk_file = subdir / "power_spectrum.jsonl"
            if not pk_file.exists():
                continue

            # Inferir n_s del nombre del directorio.
            ns = args.spectral_index
            if "ns-2" in subdir.name:
                ns = -2.0
            elif "ns-1" in subdir.name:
                ns = -1.0
            elif "ns0" in subdir.name:
                ns = 0.0

            s = validate_and_plot(
                pk_file, ns, args.amplitude,
                args.box_size, args.grid_size,
                subdir, label=subdir.name,
            )
            if s:
                summaries.append(s)

    print("\n=== RESUMEN ===")
    for s in summaries:
        print(f"  {s.get('label','?'):40s}  "
              f"err_med={s.get('median_rel_error', float('nan')):.3f}  "
              f"err_max={s.get('max_rel_error', float('nan')):.3f}  "
              f"slope={s.get('measured_slope', float('nan')):.2f}")


if __name__ == "__main__":
    main()
