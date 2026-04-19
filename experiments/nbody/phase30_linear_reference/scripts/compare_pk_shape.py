#!/usr/bin/env python3
"""
compare_pk_shape.py — Fase 30: Comparación de FORMA espectral
==============================================================

Compara el espectro medido de gadget-ng con la referencia EH/CAMB,
centrándose en la FORMA (no en la amplitud absoluta).

## Observables validados

1. **Ratios entre bins**: P(k_i)/P(k_j) medido vs referencia
2. **R(k) = P_measured/P_ref**: ¿es aproximadamente constante?
3. **Pendiente efectiva**: n_eff(k) = d ln P / d ln k

## Nota sobre normalización absoluta

P_measured(k) ≠ P_EH(k) en amplitud porque el generador de ICs define las
amplitudes en unidades físicas que no son recuperadas directamente por
power_spectrum() + conversión de unidades sin un factor de normalización
adicional. Ver reporte phase30 para la discusión completa.

Si R(k) es aproximadamente constante, la FORMA está bien reproducida.

## Uso

    python compare_pk_shape.py \\
        --pk-gadget pk_init_1lpt.json --pk-ref reference_pk.json \\
        --box 100.0 --h 0.674 \\
        [--pk-gadget-2lpt pk_init_2lpt.json] \\
        [--out-prefix phase30]
"""

import argparse
import json
import math
import sys

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[AVISO] matplotlib/numpy no disponibles — solo diagnósticos de texto")


# ── Funciones auxiliares ──────────────────────────────────────────────────────

def load_pk_gadget(path):
    """Carga P(k) JSON exportado desde gadget-ng (lista de {k, pk, n_modes})."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data  # formato: [{k, pk, n_modes}, ...]
    if "bins" in data:
        return data["bins"]
    raise ValueError(f"Formato no reconocido en {path}")


def load_pk_ref(path):
    """Carga P(k) de referencia JSON (generado por generate_reference_pk.py)."""
    with open(path) as f:
        data = json.load(f)
    return data["bins"], data.get("source", "?"), data.get("cosmology", {})


def interp_pk_ref(k_target, bins_ref):
    """Interpola P_ref en log-log al valor k_target."""
    ks  = [b["k"]  for b in bins_ref]
    pks = [b["pk"] for b in bins_ref]
    if k_target <= ks[0]:
        return pks[0]
    if k_target >= ks[-1]:
        return pks[-1]
    for i in range(len(ks) - 1):
        if ks[i] <= k_target <= ks[i + 1]:
            t = math.log(k_target / ks[i]) / math.log(ks[i + 1] / ks[i])
            lp = math.log(pks[i]) + t * (math.log(pks[i + 1]) - math.log(pks[i]))
            return math.exp(lp)
    return pks[-1]


def compute_ratio_r(bins_gadget, bins_ref, box_mpc_h, h_dimless):
    """
    Calcula R(k) = P_measured / P_ref para cada bin del gadget.

    Convierte k interno → h/Mpc y P interno → (Mpc/h)³ antes de comparar.

    Retorna list of (k_hmpc, R_k).
    """
    result = []
    for b in bins_gadget:
        k_int = b["k"]
        pk_int = b["pk"]
        if pk_int <= 0.0:
            continue
        k_hmpc  = k_int * h_dimless / box_mpc_h
        pk_hmpc = pk_int * box_mpc_h**3
        p_ref   = interp_pk_ref(k_hmpc, bins_ref)
        if p_ref <= 0.0:
            continue
        result.append((k_hmpc, pk_hmpc / p_ref))
    return result


def compute_neff(bins_k_pk):
    """
    Pendiente efectiva n_eff(k_i) = d ln P / d ln k entre bins adyacentes.

    Retorna list of (k_mid, n_eff).
    """
    result = []
    ks  = [b[0] for b in bins_k_pk]
    pks = [b[1] for b in bins_k_pk]
    for i in range(len(ks) - 1):
        if pks[i] > 0 and pks[i + 1] > 0:
            k_mid  = math.sqrt(ks[i] * ks[i + 1])
            n_eff  = math.log(pks[i + 1] / pks[i]) / math.log(ks[i + 1] / ks[i])
            result.append((k_mid, n_eff))
    return result


def cv_of_r(r_vals):
    """Coeficiente de variación de R(k)."""
    if len(r_vals) < 2:
        return 0.0
    mean = sum(r_vals) / len(r_vals)
    if mean == 0.0:
        return float("inf")
    var  = sum((r - mean)**2 for r in r_vals) / len(r_vals)
    return math.sqrt(var) / mean


# ── Diagnósticos de texto ─────────────────────────────────────────────────────

def print_diagnostics(label, bins_gadget, bins_ref, box_mpc_h, h_dimless):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    r_vals = compute_ratio_r(bins_gadget, bins_ref, box_mpc_h, h_dimless)
    if not r_vals:
        print("  ERROR: No hay bins válidos")
        return

    r_only = [r for (_, r) in r_vals]
    mean_r = sum(r_only) / len(r_only)
    cv     = cv_of_r(r_only)

    print(f"  Bins gadget (k_int, pk_int) convertidos a h/Mpc:")
    for k_hmpc, r in r_vals:
        print(f"    k = {k_hmpc:.4f} h/Mpc  R(k) = {r:.4e}")

    print(f"\n  mean R(k) = {mean_r:.4e}  (offset global de normalización)")
    print(f"  CV R(k)   = {cv:.3f}      (< 0.5 → forma bien reproducida)")

    if cv < 0.20:
        print("  → EXCELENTE: R(k) muy constante, forma espectral muy bien reproducida")
    elif cv < 0.50:
        print("  → ACEPTABLE: R(k) aproximadamente constante, forma espectral OK")
    else:
        print("  → PROBLEMA: R(k) varía mucho, la forma espectral está distorsionada")

    # Ratios entre bins adyacentes
    bins_k_pk_gadget = [(b["k"] * h_dimless / box_mpc_h, b["pk"] * box_mpc_h**3)
                        for b in bins_gadget if b["pk"] > 0.0]
    bins_k_pk_ref = [(k_hmpc, interp_pk_ref(k_hmpc, bins_ref))
                     for (k_hmpc, _) in [(b * h_dimless / box_mpc_h, 0) for b in [b["k"] for b in bins_gadget if b["pk"] > 0.0]]]

    if len(bins_k_pk_gadget) >= 2:
        print("\n  Ratios entre bins adyacentes (validación de forma):")
        for i in range(len(bins_k_pk_gadget) - 1):
            k_i, pk_i = bins_k_pk_gadget[i]
            k_j, pk_j = bins_k_pk_gadget[i + 1]
            ratio_meas = pk_i / pk_j if pk_j > 0 else float("nan")
            ref_i = interp_pk_ref(k_i, bins_ref)
            ref_j = interp_pk_ref(k_j, bins_ref)
            ratio_ref  = ref_i / ref_j if ref_j > 0 else float("nan")
            if ratio_ref > 0:
                rel_err = abs(ratio_meas / ratio_ref - 1.0) * 100.0
                print(f"    P({k_i:.3f})/P({k_j:.3f}): medido={ratio_meas:.3f}  ref={ratio_ref:.3f}  "
                      f"err={rel_err:.1f}%")


# ── Figuras ───────────────────────────────────────────────────────────────────

def make_figures(bins_1lpt, bins_2lpt, bins_ref, ref_source,
                 box_mpc_h, h_dimless, out_prefix):
    if not HAS_MPL:
        print("[figuras] Saltando — matplotlib no disponible")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Fase 30: Validación forma espectral vs {ref_source}", fontsize=12)

    # Datos de referencia
    k_ref  = [b["k"]  for b in bins_ref]
    pk_ref = [b["pk"] for b in bins_ref]

    def to_hmpc(bins_g):
        ks  = [b["k"]  * h_dimless / box_mpc_h for b in bins_g]
        pks = [b["pk"] * box_mpc_h**3           for b in bins_g]
        return ks, pks

    # ── Panel 1: P(k) log-log ─────────────────────────────────────────────
    ax = axes[0]
    ax.loglog(k_ref, pk_ref, "k-",  lw=2,   label=f"Referencia ({ref_source})")
    if bins_1lpt:
        ks1, ps1 = to_hmpc(bins_1lpt)
        ax.loglog(ks1, ps1, "bs--", lw=1.5, label="gadget-ng 1LPT",  markersize=5)
    if bins_2lpt:
        ks2, ps2 = to_hmpc(bins_2lpt)
        ax.loglog(ks2, ps2, "ro-.",  lw=1.5, label="gadget-ng 2LPT", markersize=5)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("P(k) [(Mpc/h)³]")
    ax.set_title("P(k) inicial")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    note = ("NOTA: P_measured ≠ P_ref en amplitud por convención de unidades\n"
            "Ver sección 'Limitación de normalización' en el reporte")
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom", color="gray")

    # ── Panel 2: R(k) = P_measured / P_ref ────────────────────────────────
    ax = axes[1]
    ax.set_title("R(k) = P_measured / P_ref")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("R(k)")
    ax.axhline(1.0, color="k", lw=1, linestyle="--", alpha=0.5, label="Ideal")

    for label, bins_g, color in [("1LPT", bins_1lpt, "b"), ("2LPT", bins_2lpt, "r")]:
        if not bins_g:
            continue
        r_data = compute_ratio_r(bins_g, bins_ref, box_mpc_h, h_dimless)
        if r_data:
            ks_r = [x[0] for x in r_data]
            rs   = [x[1] for x in r_data]
            mean_r = sum(rs) / len(rs)
            ax.semilogx(ks_r, rs, f"{color}o-", label=f"{label} (mean={mean_r:.2e})")
            ax.axhline(mean_r, color=color, lw=1, linestyle=":", alpha=0.7)

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98,
            "Si R(k)≈constante → forma bien reproducida\n(amplitud difiere por conv. unidades)",
            transform=ax.transAxes, fontsize=7, verticalalignment="top", color="gray")

    # ── Panel 3: Pendiente efectiva n_eff(k) ─────────────────────────────
    ax = axes[2]
    ax.set_title("Pendiente efectiva n_eff(k)")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("n_eff = d ln P / d ln k")
    ax.axhline(0.965, color="gray", lw=1, linestyle=":", label="n_s=0.965 (primordial)")

    # n_eff de la referencia
    ref_pairs = list(zip(k_ref, pk_ref))
    neff_ref = compute_neff(ref_pairs)
    if neff_ref:
        k_ne = [x[0] for x in neff_ref]
        n_ne = [x[1] for x in neff_ref]
        ax.semilogx(k_ne, n_ne, "k-", lw=2, label=f"Ref ({ref_source})")

    for label, bins_g, color in [("1LPT", bins_1lpt, "b"), ("2LPT", bins_2lpt, "r")]:
        if not bins_g:
            continue
        bkp = [(b["k"] * h_dimless / box_mpc_h, b["pk"] * box_mpc_h**3)
               for b in bins_g if b["pk"] > 0.0]
        neff = compute_neff(bkp)
        if neff:
            ax.semilogx([x[0] for x in neff], [x[1] for x in neff],
                        f"{color}o--", label=f"gadget-ng {label}", markersize=5)

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{out_prefix}_shape_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Figura guardada: {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compara forma espectral gadget-ng vs EH/CAMB"
    )
    parser.add_argument("--pk-gadget",      type=str, required=True,
                        help="P(k) gadget-ng 1LPT JSON")
    parser.add_argument("--pk-gadget-2lpt", type=str, default=None,
                        help="P(k) gadget-ng 2LPT JSON (opcional)")
    parser.add_argument("--pk-ref",         type=str, required=True,
                        help="P(k) de referencia JSON (generate_reference_pk.py)")
    parser.add_argument("--box",   type=float, default=100.0,
                        help="Tamaño de caja en Mpc/h")
    parser.add_argument("--h",     type=float, default=0.674,
                        help="Parámetro de Hubble adimensional")
    parser.add_argument("--out-prefix", type=str, default="phase30",
                        help="Prefijo para archivos de salida")
    args = parser.parse_args()

    bins_1lpt           = load_pk_gadget(args.pk_gadget)
    bins_ref, ref_src, cosmo_ref = load_pk_ref(args.pk_ref)
    bins_2lpt = None
    if args.pk_gadget_2lpt:
        bins_2lpt = load_pk_gadget(args.pk_gadget_2lpt)

    print(f"\n[compare_pk_shape] box={args.box} Mpc/h, h={args.h}")
    print(f"  Referencia: {ref_src}")
    if cosmo_ref:
        print(f"  Cosmología ref: {cosmo_ref}")

    print_diagnostics("1LPT vs referencia", bins_1lpt, bins_ref,
                      args.box, args.h)
    if bins_2lpt:
        print_diagnostics("2LPT vs referencia", bins_2lpt, bins_ref,
                          args.box, args.h)

        # Comparación directa 1LPT vs 2LPT (sin referencia externa)
        print(f"\n{'='*60}")
        print("  Comparación directa 1LPT vs 2LPT (elimina offset global)")
        print(f"{'='*60}")
        for b1, b2 in zip(bins_1lpt, bins_2lpt):
            if b1["pk"] > 0 and b2["pk"] > 0:
                ratio = b2["pk"] / b1["pk"]
                k_hmpc = b1["k"] * args.h / args.box
                print(f"  k={k_hmpc:.4f} h/Mpc: P_2lpt/P_1lpt = {ratio:.5f}  "
                      f"(diff={abs(ratio-1)*100:.3f}%)")

    make_figures(bins_1lpt, bins_2lpt, bins_ref, ref_src,
                 args.box, args.h, args.out_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
