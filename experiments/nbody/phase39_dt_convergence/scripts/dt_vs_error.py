#!/usr/bin/env python3
"""Phase 39 — extrae `dt_vs_error.csv` desde `target/phase39/per_cfg.json`.

Útil como paso independiente para CI / scripts de análisis que sólo
necesiten la tabla resumen sin instalar `matplotlib`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def snap_for(entry, a_target):
    for s in entry["snapshots"]:
        if abs(s["a_target"] - a_target) < 1e-12:
            return s
    raise RuntimeError(f"snapshot a={a_target} missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cfg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.per_cfg).read_text())
    dts = sorted(set(float(d) for d in data["dts"]))
    seeds = sorted(set(int(s) for s in data["seeds"]))
    a_vals = [float(a) for a in data["a_snapshots"]]

    index = {(float(e["dt"]), int(e["seed"])): e for e in data["entries"]}

    rows = ["dt,seed,a_target,median_abs_log10_err_raw,"
            "median_abs_log10_err_corr,mean_r_corr,stdev_r_corr,"
            "delta_rms,v_rms,runtime_s,steps_this_leg"]
    for dt in dts:
        for seed in seeds:
            e = index[(dt, seed)]
            for a_t in a_vals:
                s = snap_for(e, a_t)
                rows.append(",".join(str(x) for x in [
                    f"{dt:.6e}", seed, a_t,
                    f"{s['median_abs_log10_err_raw']:.6e}",
                    f"{s['median_abs_log10_err_corrected']:.6e}",
                    f"{s['mean_r_corr']:.6e}",
                    f"{s['stdev_r_corr']:.6e}",
                    f"{s['delta_rms']:.6e}",
                    f"{s['v_rms']:.6e}",
                    f"{e['runtime_s']:.3f}",
                    s["steps_this_leg"],
                ]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    print(f"[phase39] CSV → {out}  ({len(rows) - 1} filas)")


if __name__ == "__main__":
    main()
