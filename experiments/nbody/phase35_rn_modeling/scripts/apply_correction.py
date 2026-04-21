#!/usr/bin/env python3
"""Demo de postproceso: carga un P(k) crudo (lista de PkBin en JSON) y
aplica la corrección `P_phys = P_m / (A_grid · R_model(N))` usando el modelo
ajustado por `fit_r_n.py`.

El formato de entrada admite dos layouts:

1. Array plano (como el que escribe `gadget_ng_analysis::catalog::write_power_spectrum`):

```json
[{"k": 0.1, "pk": 1.2e3, "n_modes": 120}, ...]
```

2. Objeto con metadata:

```json
{
  "n": 64,
  "box_size": 1.0,
  "box_mpc_h": 100.0,
  "bins": [{"k": ..., "pk": ..., "n_modes": ...}, ...]
}
```

En el layout plano los parámetros `--n`, `--box-size` y `--box-mpc-h` son obligatorios.

Uso:
    python apply_correction.py --input pk_raw.json --model output/rn_model.json \
           --n 64 --box-size 1.0 --box-mpc-h 100.0 --output pk_corrected.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def a_grid(box_size: float, n: int) -> float:
    v = box_size ** 3
    return 2.0 * v * v / (n ** 9)


def evaluate_r_model(model: dict, n: int) -> Tuple[float, str]:
    """Prioriza tabla exacta, luego usa el modelo ganador."""
    for entry in model["table"]:
        if int(entry["n"]) == n:
            return float(entry["r_mean"]), "tabla exacta"
    winner = model["selection"]["winner"]
    if winner == "B":
        mB = model["model_b"]
        return mB["c"] * n ** (-mB["alpha"]) + mB["r_inf"], "Modelo B"
    mA = model["model_a"]
    return mA["c"] * n ** (-mA["alpha"]), "Modelo A"


def load_pk(path: Path, n_cli: int | None, box_cli: float | None,
            box_mpc_cli: float | None) -> Tuple[List[Dict], int, float, float | None]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        if n_cli is None or box_cli is None:
            raise SystemExit(
                "Entrada plana: usar --n y --box-size (CLI). Re-ejecutar con esos flags."
            )
        return data, n_cli, box_cli, box_mpc_cli
    # Objeto con metadata.
    bins = data.get("bins") or data.get("pk") or []
    n = int(data.get("n", n_cli if n_cli is not None else 0))
    box = float(data.get("box_size", box_cli if box_cli is not None else 1.0))
    box_mpc = data.get("box_mpc_h", box_mpc_cli)
    if box_mpc is not None:
        box_mpc = float(box_mpc)
    return bins, n, box, box_mpc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON con P(k) crudo")
    ap.add_argument("--model", required=True, help="rn_model.json (fit_r_n)")
    ap.add_argument("--output", required=True, help="JSON de salida con P_corrected")
    ap.add_argument("--n", type=int, default=None, help="N (si no viene en el JSON)")
    ap.add_argument("--box-size", type=float, default=None, dest="box_size")
    ap.add_argument("--box-mpc-h", type=float, default=None, dest="box_mpc_h")
    args = ap.parse_args()

    model = json.loads(Path(args.model).read_text())
    bins, n, box, box_mpc = load_pk(Path(args.input), args.n, args.box_size, args.box_mpc_h)

    if n <= 0 or box <= 0:
        raise SystemExit("Se requieren N y box_size positivos (via CLI o JSON)")

    a = a_grid(box, n)
    r, source = evaluate_r_model(model, n)
    denom = a * r
    unit_factor = (box_mpc / box) ** 3 if box_mpc else 1.0

    out_bins = []
    for b in bins:
        out_bins.append({
            "k": float(b["k"]),
            "pk": float(b["pk"]) / denom * unit_factor,
            "n_modes": int(b.get("n_modes", 0)),
        })
    out = {
        "phase35_metadata": {
            "a_grid": a,
            "r_model": r,
            "r_source": source,
            "unit_factor_mpc_h3_over_internal": unit_factor,
            "n": n,
            "box_size_internal": box,
            "box_mpc_h": box_mpc,
            "denominator": denom,
        },
        "bins": out_bins,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(
        f"[apply_correction] N={n}  A_grid={a:.3e}  R={r:.3e} ({source})  "
        f"denom={denom:.3e}  unit_factor={unit_factor:.3e}"
    )
    if bins:
        first = bins[0]
        print(
            f"[apply_correction] ejemplo: k={first['k']:.4g}  "
            f"P_m={first['pk']:.4g}  →  P_phys={out_bins[0]['pk']:.4g}"
        )
    print(f"[apply_correction] escrito {args.output}")
    # sanity
    assert math.isfinite(denom) and denom > 0


if __name__ == "__main__":
    main()
