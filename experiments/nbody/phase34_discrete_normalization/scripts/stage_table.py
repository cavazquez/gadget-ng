#!/usr/bin/env python3
"""Agrega los JSON emitidos por los tests de Phase 34 en una tabla por etapa.

Lee los 8 JSON de `target/phase34/*.json` y produce:
  * `stage_table.json`: estructura agregada consumible por otros scripts.
  * `stage_table.md`: tabla Markdown lista para pegar en el reporte.

Las columnas:

| Etapa | Observable | Factor esperado | Factor medido | Comentario |

son las exigidas por el plan de Phase 34 §6.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def cv(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    if abs(m) < 1e-300:
        return float("nan")
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) / abs(m)


def build_table(input_dir: Path) -> dict:
    r = {name: load(input_dir / f"{name}.json") for name in [
        "b1_roundtrip",
        "b1_single_mode",
        "b1_white_noise",
        "b2_cic_effect",
        "b2_global_offset",
        "b2_particle_vs_grid",
        "b3_resolutions",
        "b3_seeds",
    ]}

    # Etapa 1: δ̂(k) discreto correctamente construido.
    #   Observable: ⟨|δ̂|²⟩ / (σ²·N³) debería ser 1.
    wn = r["b1_white_noise"]
    e1 = {
        "stage": "continuo → δ̂(k) discreto",
        "observable": "⟨|δ̂|²⟩ / (σ²·N³)",
        "expected": 1.0,
        "measured": wn["ratio"],
        "cv_shape": None,
        "comment": (
            f"Test ruido blanco, N={wn['n']}, σ={wn['sigma']}. "
            f"Confirma `forward unnormalized ⇒ Var(δ̂) = σ²·N³`."
        ),
    }

    # Etapa 2: roundtrip δ̂→δ(x)→δ̂ con IFFT ×1/N³.
    rt = r["b1_roundtrip"]
    sm = r["b1_single_mode"]
    e2 = {
        "stage": "δ̂(k) → δ(x) → FFT",
        "observable": "max |δ_out - δ_in|",
        "expected": 0.0,
        "measured": rt["max_abs_error"],
        "cv_shape": None,
        "comment": (
            f"Roundtrip a precisión máquina (N={rt['n']}); "
            f"modo único recuperado con error {sm['max_amplitude_in_other_modes']:.2e}."
        ),
    }

    # Etapa 3: grilla pura vs P_cont (offset absoluto del generador+FFT).
    pvg = r["b2_particle_vs_grid"]
    a_grid = pvg["a_grid_mean"]
    # CV entre seeds de A_grid:
    a_grid_seeds = [row["a_grid"] for row in pvg["per_seed"]]
    e3 = {
        "stage": "grilla pura: P_grid / P_cont",
        "observable": "A_grid = ⟨P_grid / P_cont⟩",
        "expected": "2·V²/N⁹ (convención del código)",
        "measured": a_grid,
        "cv_shape": cv(a_grid_seeds),
        "comment": (
            f"N={pvg['n']}, 6 seeds. El factor 2 viene de σ²→⟨|δ̂|²⟩=2σ². "
            f"CV entre seeds: {cv(a_grid_seeds):.3f}."
        ),
    }

    # Etapa 4: grilla → partículas (ZA sin evolución).
    ratios_part_grid = [row["ratio_part_over_grid"] for row in pvg["per_seed"]]
    e4 = {
        "stage": "grilla → partículas ZA + CIC + deconv",
        "observable": "A_part / A_grid",
        "expected": "≈ 1 si partículas reproducen δ̂ fielmente",
        "measured": pvg["ratio_part_over_grid_mean"],
        "cv_shape": pvg["ratio_part_over_grid_cv"],
        "comment": (
            f"Factor multiplicativo exclusivo del paso a partículas: "
            f"{pvg['ratio_part_over_grid_mean']:.4f} (CV {pvg['ratio_part_over_grid_cv']:.4f}). "
            f"Extremadamente determinista: {len(ratios_part_grid)} seeds."
        ),
    }

    # Etapa 5: efecto CIC raw vs deconv.
    cic = r["b2_cic_effect"]
    e5 = {
        "stage": "deconvolución CIC",
        "observable": "pendiente de log R(k) vs log k",
        "expected": 0.0,
        "measured": cic["slope_deconv"],
        "cv_shape": cic["slope_raw"],
        "comment": (
            f"slope_raw={cic['slope_raw']:.4f} → slope_deconv={cic['slope_deconv']:.4f} "
            f"(reducción {100*(1 - abs(cic['slope_deconv'])/abs(cic['slope_raw'])):.1f} %). "
            f"El CIC introduce la única dependencia en k residual."
        ),
    }

    # Etapa 6: offset global aislado (sin solver).
    go = r["b2_global_offset"]
    e6 = {
        "stage": "offset global aislado (sin solver)",
        "observable": "CV(P_m/P_cont) en k ≤ k_Nyq/2",
        "expected": "< 0.15",
        "measured": go["CV_ratio"],
        "cv_shape": None,
        "comment": (
            f"A_mean={go['A_mean']:.3e}, {go['n_bins']} bins. "
            f"Confirma que la distorsión de forma es << 1 respecto al offset global."
        ),
    }

    # Etapa 7: escalado con N.
    res = r["b3_resolutions"]
    e7 = {
        "stage": "escalado con resolución N",
        "observable": "log₁₀(A₁₆/A₃₂)",
        "expected": res["expected_log10_ratio"],
        "measured": res["log10_ratio_A16_over_A32"],
        "cv_shape": None,
        "comment": (
            f"N∈{{16,32}}, 6 seeds. Observado {res['log10_ratio_A16_over_A32']:.3f} "
            f"vs predicción {res['expected_log10_ratio']:.3f}; exceso ≈ "
            f"{res['log10_ratio_A16_over_A32'] - res['expected_log10_ratio']:.3f} décadas."
        ),
    }

    # Etapa 8: determinismo entre seeds.
    seeds = r["b3_seeds"]
    e8 = {
        "stage": "determinismo entre seeds",
        "observable": "CV(A) sobre 6 seeds",
        "expected": "< 0.10",
        "measured": seeds["CV_A"],
        "cv_shape": None,
        "comment": f"CV={seeds['CV_A']:.4f} → A es determinista, no estadístico.",
    }

    table = [e1, e2, e3, e4, e5, e6, e7, e8]

    # Resumen: ¿el offset se decompone en factores limpios?
    summary = {
        "A_grid_mean_N32": a_grid,
        "ratio_part_over_grid_mean": pvg["ratio_part_over_grid_mean"],
        "ratio_part_over_grid_cv": pvg["ratio_part_over_grid_cv"],
        "CV_A_seeds_N32": seeds["CV_A"],
        "slope_deconv": cic["slope_deconv"],
        "slope_raw": cic["slope_raw"],
        "log10_A16_over_A32_observed": res["log10_ratio_A16_over_A32"],
        "log10_A16_over_A32_expected": res["expected_log10_ratio"],
        "roundtrip_max_error": rt["max_abs_error"],
        "white_noise_ratio": wn["ratio"],
    }

    return {"stages": table, "summary": summary}


def to_markdown(table: dict) -> str:
    def fmt(x):
        if x is None:
            return "—"
        if isinstance(x, str):
            return x
        if isinstance(x, float):
            if not math.isfinite(x):
                return "nan"
            if abs(x) > 1e-3 and abs(x) < 1e5:
                return f"{x:.4f}"
            return f"{x:.3e}"
        return str(x)

    lines = [
        "| # | Etapa | Observable | Esperado | Medido | Extra | Comentario |",
        "|---|-------|------------|----------|--------|-------|------------|",
    ]
    for i, e in enumerate(table["stages"], 1):
        extra = "—" if e["cv_shape"] is None else f"CV/raw: {fmt(e['cv_shape'])}"
        lines.append(
            f"| {i} | {e['stage']} | {e['observable']} | {fmt(e['expected'])} | "
            f"{fmt(e['measured'])} | {extra} | {e['comment']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directorio con los JSON de los tests")
    ap.add_argument("--output", required=True, help="Salida JSON agregada")
    ap.add_argument("--markdown", required=True, help="Salida Markdown")
    args = ap.parse_args()

    table = build_table(Path(args.input))
    Path(args.output).write_text(json.dumps(table, indent=2))
    Path(args.markdown).write_text(to_markdown(table))
    print(f"Escrito {args.output} y {args.markdown}")


if __name__ == "__main__":
    main()
