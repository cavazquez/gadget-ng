#!/usr/bin/env python3
"""Phase 36 — aplicar `pk_correction` sobre el JSONL de `gadget-ng analyse`.

Mirror en Python de `crates/gadget-ng-analysis/src/pk_correction.rs`:

    P_phys(k) = P_measured(k) / (A_grid(V, N) · R(N))

con `A_grid(V, N) = 2 · V² / N⁹` y `R(N)` dada por la tabla congelada de
Phase 35 (`RnModel::phase35_default`).

Además calcula la referencia EH no-wiggle escalada por el crecimiento lineal
CPT92 `[D(a)/D(a_init)]²` (ver `lcdm_N32_a005_2lpt_pm.toml` Phase 30:
"σ₈ normaliza la amplitud independiente de `a_init`"), y reporta métricas
absolutas `P_measured/P_ref` y `P_corrected/P_ref`.

Salida: `<out_dir>/cli_evidence.json` con los bins crudos, corregidos y de
referencia + métricas agregadas.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


# ── Tabla R(N) y A_grid (mirror de `pk_correction.rs`) ──────────────────────

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


def correct_pk(bins, box_size_internal: float, n: int, box_mpc_h: float | None):
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


# ── Transferencia EH no-wiggle y normalización σ₈ ──────────────────────────

def transfer_eh_nowiggle(k_hmpc: float, omega_m: float, omega_b: float,
                         h: float, t_cmb: float = 2.7255) -> float:
    """EH no-wiggle (Eisenstein & Hu 1998, Eq. 26–31)."""
    if k_hmpc <= 0.0:
        return 0.0
    omega_mh2 = omega_m * h * h
    omega_bh2 = omega_b * h * h
    f_b = omega_b / omega_m
    theta = t_cmb / 2.7
    s = 44.5 * math.log(9.83 / omega_mh2) / math.sqrt(1.0 + 10.0 * omega_bh2 ** 0.75)
    alpha_gamma = 1.0 - 0.328 * math.log(431.0 * omega_mh2) * f_b \
        + 0.38 * math.log(22.3 * omega_mh2) * f_b * f_b
    k_mpc = k_hmpc * h
    gamma_eff = omega_mh2 * (alpha_gamma + (1.0 - alpha_gamma) /
                             (1.0 + (0.43 * k_mpc * s) ** 4))
    q = k_hmpc * theta * theta / gamma_eff
    l0 = math.log(2.0 * math.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return l0 / (l0 + c0 * q * q)


def tophat_window(x: float) -> float:
    if abs(x) < 1e-6:
        return 1.0
    return 3.0 * (math.sin(x) - x * math.cos(x)) / (x ** 3)


def sigma_sq_unit(r_mpc_h, n_s, omega_m, omega_b, h, t_cmb=2.7255,
                  k_min=1e-4, k_max=50.0, n_k=2048):
    ln_kmin = math.log(k_min)
    ln_kmax = math.log(k_max)
    total = 0.0
    for i in range(n_k):
        ln_k = ln_kmin + (ln_kmax - ln_kmin) * i / (n_k - 1)
        k = math.exp(ln_k)
        dlnk = (ln_kmax - ln_kmin) / (n_k - 1)
        tk = transfer_eh_nowiggle(k, omega_m, omega_b, h, t_cmb)
        w = tophat_window(k * r_mpc_h)
        integrand = k ** 3 * k ** n_s * tk * tk * w * w / (2.0 * math.pi ** 2)
        total += integrand * dlnk
    return total


def amplitude_for_sigma8(sigma8_target, n_s, omega_m, omega_b, h, t_cmb=2.7255):
    s2 = sigma_sq_unit(8.0, n_s, omega_m, omega_b, h, t_cmb)
    if s2 <= 0.0:
        return sigma8_target
    return sigma8_target / math.sqrt(s2)


def eh_pk_z0(k_hmpc, amp, n_s, omega_m, omega_b, h, t_cmb=2.7255):
    tk = transfer_eh_nowiggle(k_hmpc, omega_m, omega_b, h, t_cmb)
    return amp * amp * (k_hmpc ** n_s) * tk * tk


# ── Crecimiento CPT92 ─────────────────────────────────────────────────────

def cpt92_g(a, omega_m, omega_l):
    a3 = a ** 3
    denom = omega_m + omega_l * a3
    om_a = omega_m / denom
    ol_a = omega_l * a3 / denom
    return 2.5 * om_a / (
        om_a ** (4.0 / 7.0) - ol_a + (1.0 + om_a / 2.0) * (1.0 + ol_a / 70.0)
    )


def d_of_a(a, omega_m, omega_l):
    return a * cpt92_g(a, omega_m, omega_l)


# ── Métricas ──────────────────────────────────────────────────────────────

def median_abs_log_ratio(measured, reference):
    vals = [abs(math.log10(m / r)) for m, r in zip(measured, reference)
            if m > 0 and r > 0 and math.isfinite(m) and math.isfinite(r)]
    if not vals:
        return float('nan')
    return statistics.median(vals)


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return float('nan')
    return sum(xs) / len(xs)


def stdev(xs):
    xs = [x for x in xs if math.isfinite(x)]
    if len(xs) < 2:
        return float('nan')
    return statistics.stdev(xs)


# ── Pipeline principal ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Aplica pk_correction (Phase 35) a un power_spectrum.jsonl "
                    "de gadget-ng y compara contra EH·D(a)²."
    )
    ap.add_argument("--pk-jsonl", required=True,
                    help="Ruta a <analyse_out>/power_spectrum.jsonl")
    ap.add_argument("--n", type=int, required=True,
                    help="Resolución de la malla (N=N_part=N_mesh)")
    ap.add_argument("--box-internal", type=float, default=1.0,
                    help="box_size usado en la corrida (internal units, default 1.0)")
    ap.add_argument("--box-mpc-h", type=float, default=100.0,
                    help="box_size_mpc_h del IC (para convertir k a h/Mpc)")
    ap.add_argument("--apply-unit-factor", action="store_true",
                    help="Multiplicar P_corr por (box_mpc_h/box_internal)³. "
                         "Dejar DESACTIVADO cuando R(N) fue calibrado con "
                         "P_cont ya en (Mpc/h)³ (caso Phase 35, default).")
    ap.add_argument("--a-snapshot", type=float, required=True,
                    help="Factor de escala a del snapshot analizado")
    ap.add_argument("--a-init", type=float, default=0.02,
                    help="a_init de la simulación (para D(a)/D(a_init))")
    ap.add_argument("--omega-m", type=float, default=0.315)
    ap.add_argument("--omega-lambda", type=float, default=0.685)
    ap.add_argument("--omega-b", type=float, default=0.049)
    ap.add_argument("--h", dest="h_dimless", type=float, default=0.674)
    ap.add_argument("--n-s", type=float, default=0.965)
    ap.add_argument("--sigma8", type=float, default=0.8)
    ap.add_argument("--k-nyq-fraction", type=float, default=0.5,
                    help="Ventana lineal: k ≤ fraction · k_Nyq")
    ap.add_argument("--min-modes", type=int, default=8)
    ap.add_argument("--out", required=True,
                    help="Ruta del JSON de salida (cli_evidence.json)")
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

    amp = amplitude_for_sigma8(args.sigma8, args.n_s, args.omega_m,
                               args.omega_b, args.h_dimless)
    d_a = d_of_a(args.a_snapshot, args.omega_m, args.omega_lambda)
    d_ai = d_of_a(args.a_init, args.omega_m, args.omega_lambda)
    d_ratio = d_a / d_ai

    records = []
    for m, c in zip(filt, corrected):
        k_hmpc = m["k"] * args.h_dimless / args.box_mpc_h
        p_ref = eh_pk_z0(k_hmpc, amp, args.n_s, args.omega_m,
                         args.omega_b, args.h_dimless) * d_ratio * d_ratio
        records.append({
            "k_internal": m["k"],
            "k_hmpc": k_hmpc,
            "n_modes": m["n_modes"],
            "pk_measured_internal": m["pk"],
            "pk_corrected_mpc_h3": c["pk"],
            "pk_reference_mpc_h3": p_ref,
        })

    measured = [r["pk_measured_internal"] for r in records]
    corr = [r["pk_corrected_mpc_h3"] for r in records]
    ref = [r["pk_reference_mpc_h3"] for r in records]

    r_corr = [c / r for c, r in zip(corr, ref) if r > 0]

    metrics = {
        "n_bins_linear_window": len(records),
        "median_abs_log10_err_raw": median_abs_log_ratio(measured, ref),
        "median_abs_log10_err_corrected": median_abs_log_ratio(corr, ref),
        "mean_r_corr": mean(r_corr),
        "stdev_r_corr": stdev(r_corr),
        "cv_r_corr": (stdev(r_corr) / abs(mean(r_corr))) if mean(r_corr) else float('nan'),
        "a_grid": a_grid(args.box_internal, args.n),
        "r_of_n": r_of_n(args.n),
        "d_ratio_a_over_a_init": d_ratio,
    }

    out = {
        "meta": {
            "pk_jsonl": str(pk_path.resolve()),
            "n": args.n,
            "box_internal": args.box_internal,
            "box_mpc_h": args.box_mpc_h,
            "apply_unit_factor": args.apply_unit_factor,
            "a_snapshot": args.a_snapshot,
            "a_init": args.a_init,
            "omega_m": args.omega_m,
            "omega_lambda": args.omega_lambda,
            "omega_b": args.omega_b,
            "h": args.h_dimless,
            "n_s": args.n_s,
            "sigma8": args.sigma8,
            "k_nyq_fraction": args.k_nyq_fraction,
            "amplitude": amp,
        },
        "metrics": metrics,
        "bins": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"[phase36] CLI evidence → {out_path}")
    print(f"    bins (linear window)   : {metrics['n_bins_linear_window']}")
    print(f"    med |log10(P_m /P_ref)|: {metrics['median_abs_log10_err_raw']:.3f}")
    print(f"    med |log10(P_c /P_ref)|: {metrics['median_abs_log10_err_corrected']:.3f}")
    print(f"    mean(P_c/P_ref)        : {metrics['mean_r_corr']:.3f}")
    print(f"    CV  (P_c/P_ref)        : {metrics['cv_r_corr']:.3f}")


if __name__ == "__main__":
    main()
