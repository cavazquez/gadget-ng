#!/usr/bin/env python3
"""Ajusta el modelo R(N) de Phase 35 a partir de los JSONs del test suite.

Lee `target/phase35/rn_by_seed.json` (y, si existe, `rn_fit_model_a.json`),
ajusta:

* **Modelo A**: R(N) = C · N^(-α)                (OLS log-log)
* **Modelo B**: R(N) = C · N^(-α) + R_∞          (scipy.curve_fit, fallback OLS si no hay scipy)

Emite `output/rn_model.json` con los dos modelos, R², residuo RMS, AIC y
la selección del ganador. Emite también `output/fit_summary.md` con una
tabla legible.

Uso:
    python fit_r_n.py --input ../../../target/phase35/rn_by_seed.json \
                      --output ../output/rn_model.json \
                      --summary ../output/fit_summary.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def load_table(path: Path) -> List[Tuple[int, float, float, List[float]]]:
    """Devuelve lista de (N, R_mean, CV, r_list) desde rn_by_seed.json."""
    data = json.loads(path.read_text())
    rows: List[Tuple[int, float, float, List[float]]] = []
    for entry in data["per_n"]:
        n = int(entry["n"])
        r_mean = float(entry["r_mean"])
        cv = float(entry["cv"])
        r_list = [float(x) for x in entry["r_list"]]
        rows.append((n, r_mean, cv, r_list))
    rows.sort(key=lambda row: row[0])
    return rows


def ols_loglog(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """OLS sobre log N vs log R → devuelve (C, alpha, R²)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = (sxy ** 2) / (sxx * syy) if syy > 0 else float("nan")
    c = math.exp(intercept)
    alpha = -slope
    return c, alpha, r2


def residuals_rms(ns: List[int], rs: List[float], predict) -> float:
    err = [math.log10(predict(n) / r) for n, r in zip(ns, rs)]
    return math.sqrt(sum(e ** 2 for e in err) / len(err))


def akaike(rss: float, n: int, k: int) -> float:
    """AIC para regresión: n·ln(RSS/n) + 2k."""
    if rss <= 0 or n <= 0:
        return float("nan")
    return n * math.log(rss / n) + 2 * k


def fit_model_b(
    ns: List[int], rs: List[float], c0: float, alpha0: float
) -> Tuple[float, float, float, str]:
    """Ajusta R(N) = C·N^(-α) + R_∞.

    Preferentemente con scipy; si no, devuelve (C₀, α₀, 0, "fallback_a") marcando
    que no se pudo explorar el espacio de offset.
    """
    try:
        import numpy as np
        from scipy.optimize import curve_fit  # type: ignore

        xs = np.asarray(ns, dtype=float)
        ys = np.asarray(rs, dtype=float)

        def model(x, c, alpha, r_inf):
            return c * x ** (-alpha) + r_inf

        # Inicializamos con el fit A y R_∞ chico positivo.
        p0 = [c0, alpha0, max(1e-4, rs[-1] * 0.5)]
        try:
            popt, _ = curve_fit(model, xs, ys, p0=p0, maxfev=20000)
            return float(popt[0]), float(popt[1]), float(popt[2]), "scipy_curve_fit"
        except Exception as exc:  # pragma: no cover — scipy raro que falle aquí
            return c0, alpha0, 0.0, f"fallback_a (scipy: {exc})"
    except ImportError:
        return c0, alpha0, 0.0, "fallback_a (no-scipy)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit Model A / B para R(N) en Phase 35.")
    ap.add_argument("--input", required=True, help="Ruta a rn_by_seed.json")
    ap.add_argument("--output", required=True, help="Ruta JSON para el modelo")
    ap.add_argument("--summary", default=None, help="Ruta MD para el resumen legible")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows = load_table(inp)
    ns = [r[0] for r in rows]
    r_means = [r[1] for r in rows]
    xs = [math.log(n) for n in ns]
    ys = [math.log(r) for r in r_means]

    # Model A
    c_a, alpha_a, r2_a = ols_loglog(xs, ys)
    pred_a = lambda n, c=c_a, a=alpha_a: c * n ** (-a)  # noqa: E731
    rms_a = residuals_rms(ns, r_means, pred_a)
    rss_a = sum((math.log10(pred_a(n) / r)) ** 2 for n, r in zip(ns, r_means))
    aic_a = akaike(rss_a, len(ns), 2)

    # Model B
    c_b, alpha_b, r_inf_b, source_b = fit_model_b(ns, r_means, c_a, alpha_a)
    pred_b = lambda n, c=c_b, a=alpha_b, r=r_inf_b: c * n ** (-a) + r  # noqa: E731
    rms_b = residuals_rms(ns, r_means, pred_b)
    rss_b = sum((math.log10(pred_b(n) / r)) ** 2 for n, r in zip(ns, r_means))
    aic_b = akaike(rss_b, len(ns), 3)

    # Winner: AIC menor (con penalización por parámetros extras en B).
    winner = "A"
    if math.isfinite(aic_b) and aic_b + 2.0 < aic_a:
        # 2-unit ΔAIC threshold to justify extra parameter.
        winner = "B"

    out: Dict[str, object] = {
        "input_file": str(inp),
        "table": [{"n": n, "r_mean": r, "cv": cv, "r_list": rl}
                  for (n, r, cv, rl) in rows],
        "model_a": {
            "formula": "R(N) = C * N^(-alpha)",
            "c": c_a,
            "alpha": alpha_a,
            "r_squared": r2_a,
            "rms_log10_residual": rms_a,
            "aic": aic_a,
        },
        "model_b": {
            "formula": "R(N) = C * N^(-alpha) + R_inf",
            "c": c_b,
            "alpha": alpha_b,
            "r_inf": r_inf_b,
            "rms_log10_residual": rms_b,
            "aic": aic_b,
            "source": source_b,
        },
        "selection": {
            "winner": winner,
            "reason": (
                "ΔAIC > 2 a favor de B" if winner == "B"
                else "B no mejora AIC lo suficiente; gana Modelo A (Occam)"
            ),
        },
    }
    outp.write_text(json.dumps(out, indent=2))
    print(f"[fit_r_n] escrito {outp}")

    if args.summary:
        sum_path = Path(args.summary)
        sum_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Phase 35 — Fit summary",
            "",
            "| N | R_mean | CV(seeds) |",
            "|---|-------:|---------:|",
        ]
        for (n, r, cv, _) in rows:
            lines.append(f"| {n} | {r:.6g} | {cv:.3f} |")
        lines += [
            "",
            "## Modelo A: `R(N) = C · N^(-α)`",
            "",
            f"- C = `{c_a:.6g}`",
            f"- α = `{alpha_a:.6g}`",
            f"- R² = `{r2_a:.6f}`",
            f"- RMS(log₁₀ residuos) = `{rms_a:.4f}`",
            f"- AIC = `{aic_a:.4f}`",
            "",
            "## Modelo B: `R(N) = C · N^(-α) + R_∞`",
            "",
            f"- fuente del fit: `{source_b}`",
            f"- C = `{c_b:.6g}`",
            f"- α = `{alpha_b:.6g}`",
            f"- R_∞ = `{r_inf_b:.6g}`",
            f"- RMS(log₁₀ residuos) = `{rms_b:.4f}`",
            f"- AIC = `{aic_b:.4f}`",
            "",
            f"**Ganador:** Modelo {winner} — {out['selection']['reason']}",  # type: ignore[index]
            "",
        ]
        sum_path.write_text("\n".join(lines))
        print(f"[fit_r_n] escrito {sum_path}")


if __name__ == "__main__":
    main()
