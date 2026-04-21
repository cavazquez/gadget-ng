#!/usr/bin/env python3
"""Phase 38 — apply pk_correction and compare against CLASS.

Mirror of `crates/gadget-ng-analysis/src/pk_correction.rs` that loads an
external CLASS `.dat` table (`k [h/Mpc]`, `P(k) [(Mpc/h)^3]`) and computes
the raw/corrected amplitude closure against it.

Usage:

    ./apply_phase38_correction.py \\
        --pk-jsonl <path to power_spectrum JSONL from gadget-ng analyse> \\
        --class-ref <reference/pk_class_z0.dat or z49.dat> \\
        --mode {legacy|rescaled} \\
        --n 32 \\
        --box-mpc-h 100.0 \\
        --apply-unit-factor \\
        --out <output JSON>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


# ── R(N) table and A_grid (mirror of pk_correction.rs) ──────────────────────

R_TABLE_PHASE35 = {
    8:  0.415_381_774_072_765_2,
    16: 0.139_628_643_665_938_7,
    32: 0.033_752_377_475_223_0,
    64: 0.008_834_200_231_037_1,
}
R_MODEL_C = 22.108_191_932_947_1
R_MODEL_ALPHA = 1.871_411_761_656_48


def r_of_n(n: int) -> float:
    if n in R_TABLE_PHASE35:
        return R_TABLE_PHASE35[n]
    return R_MODEL_C * (n ** (-R_MODEL_ALPHA))


def a_grid(box_size: float, n: int) -> float:
    v = box_size ** 3
    return 2.0 * v * v / (n ** 9)


def correct_pk(bins, box_size_internal: float, n: int, box_mpc_h):
    a = a_grid(box_size_internal, n)
    r = r_of_n(n)
    denom = a * r
    unit_factor = 1.0
    if box_mpc_h is not None:
        unit_factor = (box_mpc_h / box_size_internal) ** 3
    return [
        {"k": b["k"], "pk": b["pk"] / denom * unit_factor, "n_modes": b["n_modes"]}
        for b in bins
    ]


# ── CLASS .dat loader + log-log interpolator ────────────────────────────────

def load_class_dat(path: Path):
    ks, pks = [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            k = float(parts[0])
            p = float(parts[1])
            if k > 0 and p > 0:
                ks.append(k)
                pks.append(p)
    if len(ks) < 16:
        raise RuntimeError(f"CLASS .dat demasiado corto: {path} ({len(ks)} filas)")
    return ks, pks


def make_interpolator(ks, pks):
    import bisect
    ln_ks = [math.log(k) for k in ks]
    ln_pks = [math.log(p) for p in pks]

    def at(k: float) -> float:
        if k <= 0:
            return float('nan')
        lk = math.log(k)
        if lk <= ln_ks[0]:
            return pks[0]
        if lk >= ln_ks[-1]:
            return pks[-1]
        i = bisect.bisect_left(ln_ks, lk)
        # ln_ks[i-1] <= lk < ln_ks[i]
        t = (lk - ln_ks[i - 1]) / (ln_ks[i] - ln_ks[i - 1])
        return math.exp(ln_pks[i - 1] * (1 - t) + ln_pks[i] * t)

    return at


# ── Metrics ─────────────────────────────────────────────────────────────────

def median_abs_log_ratio(measured, reference):
    vals = [abs(math.log10(m / r)) for m, r in zip(measured, reference)
            if m > 0 and r > 0 and math.isfinite(m) and math.isfinite(r)]
    if not vals:
        return float('nan')
    return statistics.median(vals)


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float('nan')


def stdev(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return statistics.stdev(xs) if len(xs) >= 2 else float('nan')


# ── Main ────────────────────────────────────────────────────────────────────

BAO_K_MIN = 0.05
BAO_K_MAX = 0.30


def main():
    ap = argparse.ArgumentParser(
        description="Phase 38: apply pk_correction and compare vs a CLASS "
                    ".dat table in legacy or rescaled convention.")
    ap.add_argument("--pk-jsonl", required=True)
    ap.add_argument("--class-ref", required=True,
                    help="Path to pk_class_z0.dat or pk_class_z49.dat.")
    ap.add_argument("--mode", choices=["legacy", "rescaled"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--box-internal", type=float, default=1.0)
    ap.add_argument("--box-mpc-h", type=float, default=100.0)
    ap.add_argument("--apply-unit-factor", action="store_true")
    ap.add_argument("--h", dest="h_dimless", type=float, default=0.674)
    ap.add_argument("--k-nyq-fraction", type=float, default=0.5)
    ap.add_argument("--min-modes", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pk_path = Path(args.pk_jsonl)
    bins = []
    with pk_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bins.append(json.loads(line))

    k_nyq_internal = (args.n / 2.0) * (2.0 * math.pi / args.box_internal)
    k_cut = args.k_nyq_fraction * k_nyq_internal
    filt = [b for b in bins if b["k"] > 0 and b["pk"] > 0
            and b["n_modes"] >= args.min_modes and b["k"] <= k_cut]

    box_mpc_h_arg = args.box_mpc_h if args.apply_unit_factor else None
    corrected = correct_pk(filt, args.box_internal, args.n, box_mpc_h_arg)

    class_path = Path(args.class_ref)
    ks_cl, pks_cl = load_class_dat(class_path)
    p_class_at = make_interpolator(ks_cl, pks_cl)

    records = []
    for m, c in zip(filt, corrected):
        k_hmpc = m["k"] * args.h_dimless / args.box_mpc_h
        p_ref = p_class_at(k_hmpc)
        in_bao = BAO_K_MIN <= k_hmpc <= BAO_K_MAX
        records.append({
            "k_internal":          m["k"],
            "k_hmpc":              k_hmpc,
            "n_modes":             m["n_modes"],
            "pk_measured_internal": m["pk"],
            "pk_corrected_mpc_h3": c["pk"],
            "pk_class_mpc_h3":     p_ref,
            "in_bao_band":         in_bao,
        })

    def metrics(recs):
        measured = [r["pk_measured_internal"] for r in recs]
        corr = [r["pk_corrected_mpc_h3"] for r in recs]
        ref = [r["pk_class_mpc_h3"] for r in recs]
        r_corr = [c / rr for c, rr in zip(corr, ref) if rr > 0]
        mu = mean(r_corr)
        sd = stdev(r_corr)
        return {
            "n_bins":                     len(recs),
            "median_abs_log10_err_raw":   median_abs_log_ratio(measured, ref),
            "median_abs_log10_err_corr":  median_abs_log_ratio(corr, ref),
            "mean_r_corr":                mu,
            "stdev_r_corr":               sd,
            "cv_r_corr":                  (sd / abs(mu)) if mu else float('nan'),
        }

    metrics_all = metrics(records)
    metrics_out = metrics([r for r in records if not r["in_bao_band"]])

    out = {
        "meta": {
            "pk_jsonl":         str(pk_path.resolve()),
            "class_ref":        str(class_path.resolve()),
            "mode":             args.mode,
            "n":                args.n,
            "box_internal":     args.box_internal,
            "box_mpc_h":        args.box_mpc_h,
            "apply_unit_factor": args.apply_unit_factor,
            "h":                args.h_dimless,
            "k_nyq_fraction":   args.k_nyq_fraction,
            "bao_k_min_hmpc":   BAO_K_MIN,
            "bao_k_max_hmpc":   BAO_K_MAX,
        },
        "metrics_all":          metrics_all,
        "metrics_outside_bao":  metrics_out,
        "bins":                 records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"[phase38][{args.mode}] CLI evidence → {out_path}")
    print(f"    class_ref                 : {class_path.name}")
    print(f"    bins (linear window, all) : {metrics_all['n_bins']}")
    print(f"    bins (outside BAO)        : {metrics_out['n_bins']}")
    print(f"    med |log10(P_m /P_CLASS)| : {metrics_all['median_abs_log10_err_raw']:.3f}")
    print(f"    med |log10(P_c /P_CLASS)| : {metrics_all['median_abs_log10_err_corr']:.3f}")
    print(f"    mean(P_c/P_CLASS) [all]   : {metrics_all['mean_r_corr']:.3f}")
    print(f"    mean(P_c/P_CLASS) [out]   : {metrics_out['mean_r_corr']:.3f}")


if __name__ == "__main__":
    main()
