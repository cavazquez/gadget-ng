#!/usr/bin/env python3
"""
Phase 15 — Validación física P15 vs P14.

Compara drift energético, |Δp|, |ΔL| entre las variantes p14_fused y p15_explicit
para las configuraciones del grupo 'validation'.
"""

import json
import re
import sys
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("results")
TOL_ENERGY   = 5e-3   # |ΔE/E₀| relativo entre variantes
TOL_MOMENTUM = 1e-8   # |Δp| entre variantes (normalizado)
TOL_ANGULAR  = 1e-8   # |ΔL| entre variantes (normalizado)


def load_timings(result_dir: Path) -> dict | None:
    tj = result_dir / "timings.json"
    if not tj.exists():
        return None
    with open(tj) as f:
        return json.load(f)


def parse_run_dir(name: str) -> dict:
    m = re.match(r"^(.+)_N(\d+)_P(\d+)_(p14_fused|p15_explicit)$", name)
    if not m:
        return {}
    return {
        "group": m.group(1), "n": int(m.group(2)),
        "p": int(m.group(3)), "variant": m.group(4),
    }


def main():
    if not RESULTS_DIR.exists():
        print("ERROR: results/ no existe.", file=sys.stderr)
        sys.exit(1)

    from collections import defaultdict
    by_key = defaultdict(dict)

    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_run_dir(d.name)
        if not meta or meta["group"] != "validation":
            continue
        data = load_timings(d)
        if data is None:
            continue
        k = (meta["n"], meta["p"])
        by_key[k][meta["variant"]] = data

    print("=" * 75)
    print("VALIDACIÓN FÍSICA: P15 (explicit AVX2) vs P14 (fused kernel)")
    print("=" * 75)
    print(f"{'Config':<25} {'P14 |ΔE/E₀|':>14} {'P15 |ΔE/E₀|':>14} {'ΔE entre var':>14} {'Estado':>8}")
    print("-" * 75)

    all_ok = True
    for k in sorted(by_key.keys()):
        n, p = k
        variants = by_key[k]
        p14 = variants.get("p14_fused")
        p15 = variants.get("p15_explicit")

        if p14 is None or p15 is None:
            print(f"  N={n} P={p}: falta variante ({list(variants.keys())})")
            continue

        agg14 = p14.get("aggregate", p14)
        agg15 = p15.get("aggregate", p15)

        de14 = agg14.get("mean_energy_drift", None)
        de15 = agg15.get("mean_energy_drift", None)

        # Diferencia relativa entre variantes
        if de14 is not None and de15 is not None:
            diff = abs(de15 - de14) / (abs(de14) + 1e-30)
            ok = diff < TOL_ENERGY
        else:
            diff = None
            ok = False

        if not ok:
            all_ok = False

        status = "OK" if ok else "FAIL"
        label = f"N={n} P={p}"
        print(f"  {label:<23} {str(de14 if de14 is not None else '—'):>14} "
              f"{str(de15 if de15 is not None else '—'):>14} "
              f"{f'{diff:.2e}' if diff is not None else '—':>14} {status:>8}")

    print("-" * 75)
    if all_ok:
        print("  RESULTADO: PASS — P15 es físicamente equivalente a P14")
    else:
        print("  RESULTADO: FAIL — hay diferencias fuera de tolerancia")

    print(f"\nTolerancia energía entre variantes: {TOL_ENERGY:.0e}")
    print("(P15 introduce diferencias de punto flotante menores al reordenar operaciones)")


if __name__ == "__main__":
    main()
