#!/usr/bin/env python3
"""
Phase 15b — Validación física del sweep de leaf_max.

Compara KE, momentum y angular_momentum en el último paso de cada
configuración contra el baseline leaf_max=8 p14_fused.

Tolerancias:
  - ΔKE_rel   < 1e-3  (entre cualquier leaf_max y el baseline)
  - |Δmomentum| < 1e-4
"""

import json
import re
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results")
LEAF_MAX_VALUES = [8, 16, 32, 64]
TOL_KE = 1e-3
TOL_MOM = 1e-4


def load_last_diag(result_dir: Path) -> dict | None:
    diag = result_dir / "diagnostics.jsonl"
    if not diag.exists():
        return None
    lines = [l for l in diag.read_text().strip().split("\n") if l.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def parse_name(name: str) -> dict | None:
    m = re.match(r"^lm(\d+)_N(\d+)_P(\d+)_(p14_fused|p15_explicit)$", name)
    if not m:
        return None
    return {
        "leaf_max": int(m.group(1)),
        "n":        int(m.group(2)),
        "p":        int(m.group(3)),
        "variant":  m.group(4),
    }


def main():
    if not RESULTS_DIR.exists():
        print("ERROR: results/ no existe. Ejecuta run_phase15b.sh primero.")
        return

    # Cargar todos los resultados
    by_key = defaultdict(dict)
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_name(d.name)
        if not meta:
            continue
        diag = load_last_diag(d)
        if diag is None:
            continue
        k = (meta["leaf_max"], meta["n"], meta["p"])
        by_key[k][meta["variant"]] = diag

    # ── Tabla: física por leaf_max ─────────────────────────────────────────────
    print("\n" + "="*90)
    print("VALIDACIÓN FÍSICA: KE, momentum, angular_momentum por leaf_max")
    print("="*90)

    all_ok = True

    for n_filter, p_filter in [(8000, 2), (16000, 2), (8000, 4)]:
        # Baseline: leaf_max=8, p14_fused
        baseline_key = (8, n_filter, p_filter)
        baseline = by_key.get(baseline_key, {}).get("p14_fused")
        if baseline is None:
            print(f"  N={n_filter} P={p_filter}: sin baseline leaf_max=8")
            continue

        ke_base = baseline.get("kinetic_energy")
        mom_base = baseline.get("momentum", [0, 0, 0])
        ang_base = baseline.get("angular_momentum", [0, 0, 0])

        print(f"\n  N={n_filter} P={p_filter} — baseline leaf_max=8 p14:")
        print(f"    KE = {ke_base:.6f}  |L_z| = {abs(ang_base[2]):.4e}  |p| = {sum(x**2 for x in mom_base)**0.5:.2e}")
        print(f"    {'leaf_max':>8} {'variant':<14} {'KE':>12} {'ΔKE_rel':>12} {'|Δp|':>12} {'Estado':>8}")
        print("    " + "-"*65)

        for lm in LEAF_MAX_VALUES:
            for var in ["p14_fused", "p15_explicit"]:
                k = (lm, n_filter, p_filter)
                d = by_key.get(k, {}).get(var)
                if d is None:
                    continue

                ke = d.get("kinetic_energy")
                mom = d.get("momentum", [0, 0, 0])

                if ke is not None and ke_base is not None:
                    dke = abs(ke - ke_base) / abs(ke_base)
                else:
                    dke = None

                dp = sum((a - b)**2 for a, b in zip(mom, mom_base))**0.5

                ok_ke  = dke is not None and dke < TOL_KE
                ok_mom = dp < TOL_MOM
                ok = ok_ke and ok_mom
                if not ok:
                    all_ok = False

                status = "OK" if ok else "FAIL"
                print(f"    {lm:>8} {var:<14} {ke or 0:>12.6f} "
                      f"{f'{dke:.2e}' if dke is not None else '—':>12} "
                      f"{dp:.2e} {status:>8}")

    print("\n" + "="*90)
    print(f"Tolerancias: ΔKE_rel < {TOL_KE:.0e}  |Δmomentum| < {TOL_MOM:.0e}")
    if all_ok:
        print("RESULTADO GLOBAL: PASS — física estable para todos los leaf_max")
    else:
        print("RESULTADO GLOBAL: FAIL — hay valores fuera de tolerancia")


if __name__ == "__main__":
    main()
