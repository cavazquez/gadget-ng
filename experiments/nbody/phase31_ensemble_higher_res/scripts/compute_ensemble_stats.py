#!/usr/bin/env python3
"""
compute_ensemble_stats.py — Fase 31: Estadísticas del ensemble de P(k).

Dado un conjunto de archivos JSON de P(k) (uno por seed), calcula por bin:
  - P_mean(k), std(k), stderr(k)
  - CV(k) = std/mean
  - R(k) = P_mean / P_EH  (con la referencia EH opcional)

Salida: JSON con las estadísticas del ensemble y metadata.

Uso:
  python compute_ensemble_stats.py \\
      --pk-files pk_s001.json pk_s002.json pk_s003.json pk_s004.json \\
      --label "N32_a002_2lpt_pm" \\
      --box-mpc-h 100.0 \\
      [--ref-json reference_pk.json] \\
      --output stats_N32_a002_2lpt_pm.json
"""

import argparse
import json
import math
import sys
from pathlib import Path


# ── EH No-Wiggle (independiente de CAMB) ─────────────────────────────────────

def _eh_T(k, omega_m=0.315, omega_b=0.049, h=0.674, T_cmb=2.7255):
    """Transfer function Eisenstein–Hu no-wiggle (Eq. 29–31, EH 1998)."""
    Tcmb27 = T_cmb / 2.7
    ombh2 = omega_b * h**2
    omch2 = (omega_m - omega_b) * h**2
    omh2 = omega_m * h**2
    # Eq. 31
    z_eq = 2.50e4 * omh2 * Tcmb27**(-4)
    k_eq = 7.46e-2 * omh2 * Tcmb27**(-2)
    # Eq. 31 continued
    b1 = 0.313 * omh2**(-0.419) * (1.0 + 0.607 * omh2**0.674)
    b2 = 0.238 * omh2**0.223
    z_d = 1291.0 * omh2**0.251 / (1.0 + 0.659 * omh2**0.828) * (1.0 + b1 * ombh2**b2)
    R_eq = 31.5e3 * ombh2 * Tcmb27**(-4) * (1000.0 / z_eq)
    R_d = 31.5e3 * ombh2 * Tcmb27**(-4) * (1000.0 / z_d)
    s = (2.0 / (3.0 * k_eq)) * math.sqrt(6.0 / R_eq) * math.log(
        (math.sqrt(1.0 + R_d) + math.sqrt(R_d + R_eq)) / (1.0 + math.sqrt(R_eq))
    )
    k_silk = 1.6 * (ombh2**0.52) * (omh2**0.038) * (1.0 + (6.16 * ombh2)**(-0.46))
    alpha_gamma = 1.0 - 0.328 * math.log(431.0 * omh2) * (omega_b / omega_m) \
                  + 0.38 * math.log(22.3 * omh2) * (omega_b / omega_m)**2
    gamma_eff = omega_m * h * (
        alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k * s)**4)
    )
    q = k * Tcmb27**2 / gamma_eff
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T0 = math.log(math.e + 1.8 * q) / (math.log(math.e + 1.8 * q) + C0 * q**2)
    return T0


def eh_pk(k_hmpc_vals, sigma8=0.8, n_s=0.965,
          omega_m=0.315, omega_b=0.049, h=0.674, T_cmb=2.7255):
    """
    Calcula P_EH(k) en (Mpc/h)³ normalizado a sigma8 dado.
    Usa la integral σ₈² = ∫ P(k) W²(k×8) k² dk / (2π²) con ventana tophat.
    """
    def _tophat(x):
        if abs(x) < 1e-6:
            return 1.0
        return 3.0 * (math.sin(x) - x * math.cos(x)) / x**3

    # Calcular σ_sq_unit para normalización
    n_int = 2048
    k_min, k_max = 1e-4, 50.0
    sigma_sq_unit = 0.0
    for i in range(n_int):
        t = i / (n_int - 1)
        k = k_min * (k_max / k_min) ** t
        T = _eh_T(k, omega_m, omega_b, h, T_cmb)
        W = _tophat(k * 8.0)
        integrand = k**n_s * T**2 * W**2 * k**2 / (2.0 * math.pi**2)
        # Trapecio en log-k
        if i > 0:
            dk = k - k_prev
            sigma_sq_unit += 0.5 * (integrand + integrand_prev) * dk
        k_prev = k
        integrand_prev = integrand

    A = sigma8 / math.sqrt(sigma_sq_unit)

    result = []
    for k in k_hmpc_vals:
        T = _eh_T(k, omega_m, omega_b, h, T_cmb)
        pk = A**2 * k**n_s * T**2
        result.append(pk)
    return result


# ── Lectura de archivos P(k) ──────────────────────────────────────────────────

def load_pk_json(path):
    """
    Carga P(k) desde un JSON de gadget-ng.
    Formatos soportados:
    - {"bins": [{"k": ..., "pk": ..., "n_modes": ...}, ...]}
    - [{"k": ..., "pk": ..., "n_modes": ...}, ...]
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "bins" in data:
        return data["bins"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Formato de P(k) no reconocido en {path}")


# ── Estadísticas del ensemble ─────────────────────────────────────────────────

def ensemble_stats(pk_lists):
    """
    pk_lists: lista de N listas de bins [{k, pk, n_modes}, ...]

    Devuelve: lista de dicts con estadísticas por bin.
    Requiere que todos los P(k) tengan el mismo número de bins.
    """
    if not pk_lists:
        return []

    n_seeds = len(pk_lists)
    n_bins = len(pk_lists[0])

    # Verificar consistencia
    for i, pl in enumerate(pk_lists):
        if len(pl) != n_bins:
            raise ValueError(
                f"Seed {i} tiene {len(pl)} bins, pero seed 0 tiene {n_bins}"
            )

    stats = []
    for j in range(n_bins):
        vals = [pl[j]["pk"] for pl in pk_lists if pl[j]["pk"] > 0.0]
        k_j = pk_lists[0][j]["k"]
        n_modes_j = pk_lists[0][j].get("n_modes", 0)

        if len(vals) < 2:
            stats.append({
                "k": k_j,
                "n_modes": n_modes_j,
                "n_seeds": len(vals),
                "p_mean": vals[0] if vals else 0.0,
                "p_std": 0.0,
                "p_stderr": 0.0,
                "cv": float("nan"),
            })
            continue

        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        stderr = std / math.sqrt(len(vals))
        cv = std / mean if mean > 0.0 else float("nan")

        stats.append({
            "k": k_j,
            "n_modes": n_modes_j,
            "n_seeds": len(vals),
            "p_mean": mean,
            "p_std": std,
            "p_stderr": stderr,
            "cv": cv,
        })

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pk-files", nargs="+", required=True,
        help="Archivos JSON de P(k), uno por seed"
    )
    parser.add_argument(
        "--label", default="ensemble",
        help="Etiqueta para identificar la variante (ej. N32_a002_2lpt_pm)"
    )
    parser.add_argument(
        "--box-mpc-h", type=float, default=100.0,
        help="Tamaño de caja en Mpc/h (para conversión de k)"
    )
    parser.add_argument(
        "--h-dimless", type=float, default=0.674,
        help="Parámetro de Hubble h = H0/100"
    )
    parser.add_argument(
        "--sigma8", type=float, default=0.8,
        help="σ₈ para generar la referencia EH"
    )
    parser.add_argument(
        "--n-s", type=float, default=0.965,
        help="Índice espectral n_s"
    )
    parser.add_argument(
        "--ref-json", default=None,
        help="JSON de referencia externa (opcional, overrides EH interno)"
    )
    parser.add_argument(
        "--output", default="ensemble_stats.json",
        help="Archivo JSON de salida con estadísticas del ensemble"
    )
    args = parser.parse_args()

    # Cargar todos los P(k)
    pk_lists = []
    for path in args.pk_files:
        try:
            bins = load_pk_json(path)
            pk_lists.append(bins)
            print(f"  Cargado: {path} ({len(bins)} bins)")
        except Exception as e:
            print(f"  ERROR cargando {path}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nSeeds: {len(pk_lists)}  Bins: {len(pk_lists[0])}")

    # Calcular estadísticas del ensemble
    stats = ensemble_stats(pk_lists)

    # Convertir k a h/Mpc y calcular P_EH de referencia
    for s in stats:
        k_int = s["k"]  # k en unidades internas (2π/L_int)
        k_hmpc = k_int * args.h_dimless / args.box_mpc_h
        s["k_hmpc"] = k_hmpc
        # P medido en unidades internas → (Mpc/h)³
        s["p_mean_hmpc"] = s["p_mean"] * args.box_mpc_h**3
        s["p_std_hmpc"] = s["p_std"] * args.box_mpc_h**3
        s["p_stderr_hmpc"] = s["p_stderr"] * args.box_mpc_h**3

    # Calcular P_EH en los mismos k
    k_hmpc_vals = [s["k_hmpc"] for s in stats]
    pk_eh_vals = eh_pk(
        k_hmpc_vals,
        sigma8=args.sigma8,
        n_s=args.n_s,
    )

    # Calcular R(k) = P_mean / P_EH y su variación
    r_vals = []
    for s, pk_eh in zip(stats, pk_eh_vals):
        if pk_eh > 0.0 and s["p_mean_hmpc"] > 0.0:
            r = s["p_mean_hmpc"] / pk_eh
        else:
            r = float("nan")
        s["pk_eh"] = pk_eh
        s["r_k"] = r
        r_vals.append(r)

    valid_r = [r for r in r_vals if math.isfinite(r) and r > 0.0]
    if valid_r:
        r_mean = sum(valid_r) / len(valid_r)
        r_var = sum((r - r_mean)**2 for r in valid_r) / len(valid_r)
        r_cv = math.sqrt(r_var) / r_mean if r_mean > 0 else float("nan")
    else:
        r_mean, r_cv = float("nan"), float("nan")

    # Imprimir resumen
    cv_vals = [s["cv"] for s in stats if math.isfinite(s["cv"])]
    mean_cv = sum(cv_vals) / len(cv_vals) if cv_vals else float("nan")

    print(f"\n{'k [h/Mpc]':>12}  {'P_mean [(Mpc/h)³]':>18}  {'stderr':>10}  {'CV':>8}  {'R(k)':>10}  n_modes")
    print("-" * 80)
    for s in stats:
        print(
            f"  {s['k_hmpc']:10.4f}  {s['p_mean_hmpc']:18.4e}  "
            f"{s['p_stderr_hmpc']:10.4e}  {s['cv']:8.4f}  "
            f"{s['r_k']:10.4e}  {s['n_modes']}"
        )
    print("-" * 80)
    print(f"  mean CV(P(k)) = {mean_cv:.4f}")
    print(f"  R(k) = P_mean/P_EH: mean = {r_mean:.4e}, CV = {r_cv:.4f}")
    print(f"  N_seeds = {len(pk_lists)}")

    # Guardar JSON de salida
    output = {
        "label": args.label,
        "n_seeds": len(pk_lists),
        "box_mpc_h": args.box_mpc_h,
        "h_dimless": args.h_dimless,
        "sigma8_target": args.sigma8,
        "n_s": args.n_s,
        "mean_cv": mean_cv,
        "r_mean": r_mean,
        "r_cv": r_cv,
        "bins": stats,
        "source_files": [str(p) for p in args.pk_files],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nEstadísticas guardadas en: {out_path}")


if __name__ == "__main__":
    main()
