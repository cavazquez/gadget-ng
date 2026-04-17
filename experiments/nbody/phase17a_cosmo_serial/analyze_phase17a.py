#!/usr/bin/env python3
"""
analyze_phase17a.py — Análisis del modo cosmológico serial (Fase 17a)

Produce 4 figuras:
  1. a(t) numérico vs analítico EdS
  2. v_rms(t) para EdS y ΛCDM
  3. delta_rms(t) para EdS y ΛCDM
  4. H(a) numérico vs analítico para ambos modelos

Uso:
    python3 analyze_phase17a.py --eds <dir_eds> --lcdm <dir_lcdm> [--out <fig_dir>]

Dependencias:
    numpy, matplotlib
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Funciones analíticas ──────────────────────────────────────────────────────


def eds_a(a0: float, h0: float, t: float) -> float:
    """a(t) analítico para EdS: (a0^{3/2} + 3/2 h0 t)^{2/3}."""
    return (a0**1.5 + 1.5 * h0 * t) ** (2.0 / 3.0)


def hubble_analytic(omega_m: float, omega_lam: float, h0: float, a: float) -> float:
    """H(a) = H0 * sqrt(Omega_m/a^3 + Omega_Lambda)."""
    return h0 * math.sqrt(omega_m / a**3 + omega_lam)


# ── Lectura de diagnostics.jsonl ─────────────────────────────────────────────


def load_diag(path: str) -> list[dict]:
    diag_file = os.path.join(path, "diagnostics.jsonl")
    if not os.path.exists(diag_file):
        sys.exit(f"ERROR: no se encuentra {diag_file}")
    records = []
    with open(diag_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_series(records: list[dict], dt: float) -> dict:
    """Extrae series temporales de los registros cosmológicos."""
    steps = [r["step"] for r in records]
    times = [s * dt for s in steps]
    a_vals = [r.get("a", float("nan")) for r in records]
    z_vals = [r.get("z", float("nan")) for r in records]
    v_rms = [r.get("v_rms", float("nan")) for r in records]
    delta_rms = [r.get("delta_rms", float("nan")) for r in records]
    hubble = [r.get("hubble", float("nan")) for r in records]
    ke = [r.get("kinetic_energy", float("nan")) for r in records]
    return {
        "step": np.array(steps),
        "t": np.array(times),
        "a": np.array(a_vals),
        "z": np.array(z_vals),
        "v_rms": np.array(v_rms),
        "delta_rms": np.array(delta_rms),
        "hubble": np.array(hubble),
        "ke": np.array(ke),
    }


# ── Análisis ─────────────────────────────────────────────────────────────────


def analyze(eds_dir: str, lcdm_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ── Parámetros de config (hard-coded para reproducibilidad) ──────────────
    EDS_H0 = 0.1
    EDS_A0 = 1.0
    EDS_DT = 0.005
    LCDM_H0 = 0.1
    LCDM_OM = 0.3
    LCDM_OL = 0.7
    LCDM_DT = 0.005

    # ── Carga ────────────────────────────────────────────────────────────────
    eds_records = load_diag(eds_dir)
    lcdm_records = load_diag(lcdm_dir)

    eds = extract_series(eds_records, EDS_DT)
    lcdm = extract_series(lcdm_records, LCDM_DT)

    # Filtrar filas sin campo cosmológico (paso 0 inicial Newtoniano).
    eds_cosmo_mask = ~np.isnan(eds["a"])
    lcdm_cosmo_mask = ~np.isnan(lcdm["a"])

    # ── Figura 1: a(t) EdS numérico vs analítico ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Fase 17a — Modo Cosmológico Serial: Validación", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    t_eds = eds["t"][eds_cosmo_mask]
    a_num = eds["a"][eds_cosmo_mask]
    a_ana = np.array([eds_a(EDS_A0, EDS_H0, t) for t in t_eds])
    rel_err = np.abs(a_num - a_ana) / a_ana

    ax.plot(t_eds, a_num, "b-", lw=1.5, label="Numérico (RK4)")
    ax.plot(t_eds, a_ana, "r--", lw=1.5, label="Analítico EdS")
    ax.set_xlabel("t [u.i.]")
    ax.set_ylabel("a(t)")
    ax.set_title("Factor de Escala — EdS (Ω_m=1, Ω_Λ=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.semilogy(t_eds, rel_err, "g:", lw=1, alpha=0.7, label="|Δa/a|")
    ax2.set_ylabel("|Δa/a|", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2.legend(loc="upper left")

    max_err = np.nanmax(rel_err)
    print(f"EdS a(t) — error relativo máximo: {max_err:.3e}")
    verdict = "✓ PASS" if max_err < 0.01 else "✗ FAIL"
    ax.set_title(f"Factor de Escala — EdS  [{verdict}, max|Δa/a|={max_err:.1e}]")

    # ── Figura 2: v_rms(t) ───────────────────────────────────────────────────
    ax = axes[0, 1]
    t_eds_c = eds["t"][eds_cosmo_mask]
    t_lcdm_c = lcdm["t"][lcdm_cosmo_mask]
    ax.plot(t_eds_c, eds["v_rms"][eds_cosmo_mask], "b-", lw=1.5, label="EdS")
    ax.plot(t_lcdm_c, lcdm["v_rms"][lcdm_cosmo_mask], "r-", lw=1.5, label="ΛCDM")
    ax.set_xlabel("t [u.i.]")
    ax.set_ylabel("v_rms [u.i.]")
    ax.set_title("Velocidad Peculiar RMS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Figura 3: delta_rms(t) ───────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t_eds_c, eds["delta_rms"][eds_cosmo_mask], "b-", lw=1.5, label="EdS")
    ax.plot(t_lcdm_c, lcdm["delta_rms"][lcdm_cosmo_mask], "r-", lw=1.5, label="ΛCDM")
    ax.set_xlabel("t [u.i.]")
    ax.set_ylabel("δρ_rms / ρ̄")
    ax.set_title("Contraste de Densidad RMS (malla 16³)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Figura 4: H(a) numérico vs analítico ─────────────────────────────────
    ax = axes[1, 1]
    a_eds = eds["a"][eds_cosmo_mask]
    a_lcdm = lcdm["a"][lcdm_cosmo_mask]
    h_eds_num = eds["hubble"][eds_cosmo_mask]
    h_lcdm_num = lcdm["hubble"][lcdm_cosmo_mask]

    # Analítico EdS: H(a) = H0 / a^{3/2}
    h_eds_ana = np.array([hubble_analytic(1.0, 0.0, EDS_H0, a) for a in a_eds])
    h_lcdm_ana = np.array(
        [hubble_analytic(LCDM_OM, LCDM_OL, LCDM_H0, a) for a in a_lcdm]
    )

    ax.plot(a_eds, h_eds_num, "b-", lw=1.5, label="EdS numérico")
    ax.plot(a_eds, h_eds_ana, "b--", lw=1, alpha=0.7, label="EdS analítico")
    ax.plot(a_lcdm, h_lcdm_num, "r-", lw=1.5, label="ΛCDM numérico")
    ax.plot(a_lcdm, h_lcdm_ana, "r--", lw=1, alpha=0.7, label="ΛCDM analítico")
    ax.set_xlabel("a")
    ax.set_ylabel("H(a) [u.i.]")
    ax.set_title("Parámetro de Hubble H(a)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "phase17a_cosmo_validation.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {fig_path}")

    # ── Tabla resumen en terminal ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMEN Fase 17a — Cosmología Serial")
    print("=" * 60)

    # EdS
    a_eds_final = a_num[-1] if len(a_num) > 0 else float("nan")
    a_eds_ana_final = eds_a(EDS_A0, EDS_H0, t_eds[-1]) if len(t_eds) > 0 else float("nan")
    print(f"\n[EdS N=512, 100 pasos, dt={EDS_DT}]")
    print(f"  a_final numérico :  {a_eds_final:.6f}")
    print(f"  a_final analítico:  {a_eds_ana_final:.6f}")
    print(f"  Error relativo   :  {max_err:.2e}  {'✓' if max_err < 0.01 else '✗'} (tol 1%)")
    vrms_eds_f = eds["v_rms"][eds_cosmo_mask][-1] if eds_cosmo_mask.any() else float("nan")
    drms_eds_f = eds["delta_rms"][eds_cosmo_mask][-1] if eds_cosmo_mask.any() else float("nan")
    print(f"  v_rms (final)    :  {vrms_eds_f:.4e}")
    print(f"  delta_rms (final):  {drms_eds_f:.4f}")

    # ΛCDM
    a_lcdm_final = a_lcdm[-1] if len(a_lcdm) > 0 else float("nan")
    print(f"\n[ΛCDM N=1000, 50 pasos, dt={LCDM_DT}]")
    print(f"  a_final          :  {a_lcdm_final:.6f}")
    vrms_lcdm_f = lcdm["v_rms"][lcdm_cosmo_mask][-1] if lcdm_cosmo_mask.any() else float("nan")
    drms_lcdm_f = lcdm["delta_rms"][lcdm_cosmo_mask][-1] if lcdm_cosmo_mask.any() else float("nan")
    print(f"  v_rms (final)    :  {vrms_lcdm_f:.4e}")
    print(f"  delta_rms (final):  {drms_lcdm_f:.4f}")
    h_lcdm_final = lcdm["hubble"][lcdm_cosmo_mask][-1] if lcdm_cosmo_mask.any() else float("nan")
    h_lcdm_ana_final = hubble_analytic(LCDM_OM, LCDM_OL, LCDM_H0, a_lcdm_final)
    print(f"  H(a_final) num   :  {h_lcdm_final:.6f}")
    print(f"  H(a_final) ana   :  {h_lcdm_ana_final:.6f}")

    # Estabilidad
    eds_stable = all(
        math.isfinite(v) for v in eds["v_rms"][eds_cosmo_mask]
    )
    lcdm_stable = all(
        math.isfinite(v) for v in lcdm["v_rms"][lcdm_cosmo_mask]
    )
    print(f"\n  Estabilidad EdS  : {'✓ STABLE' if eds_stable else '✗ UNSTABLE'}")
    print(f"  Estabilidad ΛCDM : {'✓ STABLE' if lcdm_stable else '✗ UNSTABLE'}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis Fase 17a — Cosmología Serial")
    parser.add_argument("--eds", required=True, help="Directorio de resultados EdS")
    parser.add_argument("--lcdm", required=True, help="Directorio de resultados ΛCDM")
    parser.add_argument(
        "--out", default=None, help="Directorio de figuras (default: mismo que --eds/../figures/)"
    )
    args = parser.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(args.eds), "figures")
    analyze(args.eds, args.lcdm, out_dir)
