#!/usr/bin/env python3
"""
derive_normalization.py — Fase 33.

Reproduce paso a paso la derivación analítica del factor de normalización A
que relaciona `P_measured` (estimador CIC+FFT) con `P_theory` (EH continuo) en
`gadget-ng`, tal como se documenta en
`docs/reports/2026-04-phase33-pk-normalization-analysis.md`.

Imprime una tabla con:
  - A_pred(N, V, h, BOX_MPC_H) para N ∈ {8, 16, 32, 64}
  - cada factor de la cadena: estimador, generador, Hermitian, CIC
  - un resumen en formato markdown
  - opcionalmente un JSON con los valores numéricos

Uso:
    python derive_normalization.py \
        --box-internal 1.0 \
        --box-mpc-h 100.0 \
        --h-dimless 0.674 \
        --output ../output/derivation_table.json
"""

import argparse
import json
import math
from pathlib import Path


def a_pred_minimal(n, box_internal=1.0):
    """
    Fórmula mínima sin factor 2 ni BOX_MPC_H³.

        A_pred = V² / N⁹

    Esta es la versión implementada por el helper
    `analytical_normalization_factor` en phase33_pk_normalization.rs.
    """
    return (box_internal ** 6) / (n ** 9)


def a_pred_with_hermitian(n, box_internal=1.0):
    """
    Incluye el factor 2 de modos complejos:

        A_pred = 2 · V² / N⁹

    (⟨|σ_k|²⟩ = 2·P_cont(k)/N³ con Gaussian complejo 2-DOF)
    """
    return 2.0 * (box_internal ** 6) / (n ** 9)


def a_pred_with_unit_conversion(n, box_internal=1.0, box_mpc_h=100.0):
    """
    Si se quisiera interpretar P_measured en (Mpc/h)³ se multiplicaría por
    BOX_MPC_H³. Esta versión NO es la que usa el código actual, pero se
    muestra para comparación.
    """
    return 2.0 * (box_internal ** 6) * (box_mpc_h ** 3) / (n ** 9)


def format_scientific(x):
    if not math.isfinite(x):
        return "nan"
    if x == 0.0:
        return "0"
    return f"{x:.4e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-internal", type=float, default=1.0,
                        help="Tamaño de caja interno (normalmente 1)")
    parser.add_argument("--box-mpc-h", type=float, default=100.0,
                        help="Tamaño de caja físico en Mpc/h")
    parser.add_argument("--h-dimless", type=float, default=0.674,
                        help="Parámetro de Hubble adimensional h = H0/100")
    parser.add_argument("--grids", type=int, nargs="+",
                        default=[8, 16, 32, 64],
                        help="Resoluciones N a evaluar")
    parser.add_argument("--output", default=None,
                        help="Archivo JSON opcional de salida")
    args = parser.parse_args()

    print("=" * 78)
    print(" Fase 33 — Derivación del factor de normalización A")
    print("=" * 78)
    print()
    print(" Convenciones:")
    print(f"   V_internal = {args.box_internal}³ = {args.box_internal**3:g}")
    print(f"   BOX_MPC_H  = {args.box_mpc_h} Mpc/h")
    print(f"   h (adim.)  = {args.h_dimless}")
    print(f"   k_int ↔ k_hmpc: k_hmpc = k_int · h / BOX_MPC_H")
    print()
    print(" Cadena de factores (del estimador y del generador):")
    print("   [generador] σ(k)² = P_cont(k_phys) / N³         (amplitud)")
    print("   [generador] δ̂_k = σ · (g_r + i·g_i) → ⟨|δ̂|²⟩ = 2σ² = 2·P_cont/N³")
    print("   [generador] IFFT · (1/N³) → δ(x) real")
    print("   [CIC+FFT]   δ̂_DFT ≈ δ̂_generador (si partículas reproducen campo)")
    print("   [estimador] P_m = ⟨|δ̂|²⟩ · (V_int/N³)² / W²(k)")
    print("   ⇒ P_m ≈ 2·V² · P_cont(k_phys) / N⁹")
    print()
    print(" Predicción mínima (sin factor 2):   A_pred = V² / N⁹")
    print(" Con Hermitian (factor 2):           A_pred = 2·V²/N⁹")
    print(" Con conversión a (Mpc/h)³:          A_pred = 2·V² · BOX_MPC_H³ / N⁹")
    print()

    header = (
        f"{'N':>4}  {'A_min = V²/N⁹':>16}  {'A_herm = 2V²/N⁹':>18}  "
        f"{'A_mpc = 2V²·B³/N⁹':>20}  {'N⁹':>14}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for n in args.grids:
        a_min = a_pred_minimal(n, args.box_internal)
        a_herm = a_pred_with_hermitian(n, args.box_internal)
        a_mpc = a_pred_with_unit_conversion(
            n, args.box_internal, args.box_mpc_h
        )
        n9 = n ** 9
        print(
            f"{n:>4}  {format_scientific(a_min):>16}  "
            f"{format_scientific(a_herm):>18}  "
            f"{format_scientific(a_mpc):>20}  "
            f"{n9:>14.3e}"
        )
        rows.append({
            "n": n,
            "n9": n9,
            "a_pred_minimal": a_min,
            "a_pred_hermitian": a_herm,
            "a_pred_mpc_h_cubed": a_mpc,
        })

    print()
    print(" Contexto de los tests (reference Phase 32/33):")
    print("   A_obs(N=16³, 4 seeds) ≈ 3.3e-12")
    print("   A_obs(N=32³, 6 seeds) ≈ 1.66e-15")
    print("   log₁₀(A_obs/A_min) a N=32 ≈ -1.23")
    print("   ratio A(16)/A(32) observado ≈ 2000 vs predicho (32/16)⁹ = 512")
    print()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "box_internal": args.box_internal,
            "box_mpc_h": args.box_mpc_h,
            "h_dimless": args.h_dimless,
            "formulas": {
                "minimal": "V² / N⁹",
                "hermitian": "2 · V² / N⁹",
                "mpc_h_cubed": "2 · V² · BOX_MPC_H³ / N⁹",
            },
            "rows": rows,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Tabla numérica guardada en: {out_path}")


if __name__ == "__main__":
    main()
