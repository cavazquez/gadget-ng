#!/usr/bin/env python3
"""
measure_and_compare.py — Fase 33.

Lee uno o más archivos de estadísticas de ensemble Phase 31/32 (generados por
`compute_ensemble_stats.py`) y compara el factor A_obs = ⟨P_measured/P_theory⟩
contra la predicción analítica `A_pred = V² / N⁹`.

Reporta:
  - A_obs por archivo (ya está guardado como `r_mean` por el script upstream)
  - CV(R(k)) (ya guardado como `r_cv`)
  - A_pred de la fórmula mínima, Hermitian y con conversión Mpc/h³
  - log₁₀(A_obs / A_pred) para cada variante
  - scaling A(N_a)/A(N_b) vs (N_b/N_a)⁹

Uso:
    python measure_and_compare.py \
        --stats-files stats_N16.json stats_N32.json \
        --output ../output/A_obs_vs_pred.json
"""

import argparse
import json
import math
from pathlib import Path


def a_pred_minimal_internal(n, box_internal=1.0):
    """A_pred en unidades internas (coincide con Rust test)."""
    return (box_internal ** 6) / (n ** 9)


def a_pred_hermitian_internal(n, box_internal=1.0):
    return 2.0 * a_pred_minimal_internal(n, box_internal)


def a_pred_hmpc(n, box_internal=1.0, box_mpc_h=100.0):
    """
    A_pred para r_mean del script `compute_ensemble_stats.py`, que multiplica
    P_internal por BOX_MPC_H³ antes de dividir por P_EH.

        r_mean_python = A_obs_internal × BOX_MPC_H³
        A_pred_hmpc  = V² × BOX_MPC_H³ / N⁹
    """
    return (box_internal ** 6) * (box_mpc_h ** 3) / (n ** 9)


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def infer_grid_size(stats):
    """
    Intenta inferir N a partir de `label` (tokens N16 / N32 / N64) o del número
    de bins. Los bins del estimador son `n_nyq = N/2` por construcción.
    """
    label = stats.get("label", "")
    for tok in label.split("_"):
        if tok.startswith("N") and tok[1:].isdigit():
            return int(tok[1:])
    n_bins = len(stats.get("bins", []))
    # n_bins ≤ n_nyq = N/2, así que N ≥ 2·n_bins
    if n_bins > 0:
        candidate = 2 * n_bins
        # redondear a potencia de 2 si es cercano
        for p in (8, 16, 32, 64, 128, 256):
            if abs(candidate - p) <= 1:
                return p
        return candidate
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-files", nargs="+", required=True,
                        help="Archivos JSON de compute_ensemble_stats.py")
    parser.add_argument("--box-internal", type=float, default=1.0,
                        help="V = box_internal³, normalmente 1")
    parser.add_argument("--box-mpc-h", type=float, default=100.0,
                        help="BOX_MPC_H usado al generar stats (convierte a (Mpc/h)³)")
    parser.add_argument("--output", default=None,
                        help="Archivo JSON de salida con la comparación")
    args = parser.parse_args()

    print("=" * 90)
    print(" Fase 33 — Comparación A_obs vs A_pred")
    print("=" * 90)
    print()

    rows = []
    for path in args.stats_files:
        stats = load_stats(path)
        n = infer_grid_size(stats)
        if n == 0:
            print(f"  SKIP {path}: no se pudo inferir N")
            continue
        # `r_mean` del stats script está en unidades (Mpc/h)³ (multiplica P_m
        # internal por BOX_MPC_H³). Ese es el A_obs relevante para comparar con
        # A_pred_hmpc = V² · BOX_MPC_H³ / N⁹.
        a_obs_hmpc = stats.get("r_mean", float("nan"))
        cv_r = stats.get("r_cv", float("nan"))
        a_pred_h = a_pred_hmpc(n, args.box_internal, args.box_mpc_h)

        # Reconstrucción del A_obs en unidades internas (para consistencia con
        # el Rust test).
        if math.isfinite(a_obs_hmpc):
            a_obs_int = a_obs_hmpc / (args.box_mpc_h ** 3)
        else:
            a_obs_int = float("nan")
        a_pred_int = a_pred_minimal_internal(n, args.box_internal)

        log_hmpc = math.log10(a_obs_hmpc / a_pred_h) if (
            math.isfinite(a_obs_hmpc) and a_obs_hmpc > 0 and a_pred_h > 0
        ) else float("nan")
        log_int = math.log10(a_obs_int / a_pred_int) if (
            math.isfinite(a_obs_int) and a_obs_int > 0 and a_pred_int > 0
        ) else float("nan")

        rows.append({
            "path": str(path),
            "label": stats.get("label", ""),
            "n": n,
            "n_seeds": stats.get("n_seeds", 0),
            "a_obs_hmpc": a_obs_hmpc,
            "a_obs_internal": a_obs_int,
            "cv_r": cv_r,
            "a_pred_hmpc": a_pred_h,
            "a_pred_internal": a_pred_int,
            "log10_obs_over_pred_hmpc": log_hmpc,
            "log10_obs_over_pred_internal": log_int,
        })

        print(f"  {path}")
        print(f"    label           = {stats.get('label', '')}")
        print(f"    N               = {n}")
        print(f"    n_seeds         = {stats.get('n_seeds', 0)}")
        print(f"    A_obs (Mpc/h)³  = {a_obs_hmpc:.4e}   (r_mean)")
        print(f"    A_obs internal  = {a_obs_int:.4e}   (÷ BOX_MPC_H³)")
        print(f"    A_pred (Mpc/h)³ = {a_pred_h:.4e}   (V²·B³/N⁹)")
        print(f"    A_pred internal = {a_pred_int:.4e}   (V²/N⁹)")
        print(f"    log10(obs/pred) (Mpc/h)³   = {log_hmpc:.4f}")
        print(f"    log10(obs/pred) internal  = {log_int:.4f}")
        print(f"    CV(R(k))        = {cv_r:.4f}")
        print()

    # Scaling entre pares de N
    if len(rows) >= 2:
        print("  Scaling A(N_i) / A(N_j) vs predicción (N_j/N_i)⁹:")
        print(
            f"    {'i':>3} {'j':>3}  {'A_i/A_j':>12}  {'(N_j/N_i)⁹':>14}  "
            f"{'log₁₀(ratio)':>14}"
        )
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a_i = rows[i]["a_obs_hmpc"]
                a_j = rows[j]["a_obs_hmpc"]
                n_i = rows[i]["n"]
                n_j = rows[j]["n"]
                if a_j == 0 or not math.isfinite(a_j):
                    continue
                ratio_obs = a_i / a_j
                ratio_pred = (n_j / n_i) ** 9
                log_ratio = math.log10(ratio_obs / ratio_pred) if (
                    ratio_obs > 0 and ratio_pred > 0
                ) else float("nan")
                print(
                    f"    {n_i:>3} {n_j:>3}  {ratio_obs:>12.3e}  "
                    f"{ratio_pred:>14.3e}  {log_ratio:>14.4f}"
                )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"rows": rows}, f, indent=2)
        print(f"\nComparación guardada en: {out_path}")


if __name__ == "__main__":
    main()
