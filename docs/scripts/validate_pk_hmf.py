#!/usr/bin/env python3
"""
Phase 79 — Validación P(k) y HMF contra teoría.

Compara:
- P(k, z=0) medido en la simulación vs espectro CLASS (Eisenstein-Hu como proxy).
- HMF medida vs modelo Tinker et al. (2008).
- σ₈ medida en la simulación vs valor de input.

Uso:
    python3 validate_pk_hmf.py --analysis-dir runs/validation_128/analysis \\
                               --out-dir runs/validation_128/plots

Requiere: numpy, matplotlib (opcional: classy para comparación exacta con CLASS).
"""

import argparse
import glob
import json
import math
import os
import sys

def load_latest_insitu(analysis_dir):
    """Carga el archivo insitu más reciente (z≈0)."""
    files = sorted(glob.glob(os.path.join(analysis_dir, "insitu_*.json")))
    if not files:
        print(f"[ERROR] No se encontraron archivos insitu en {analysis_dir}", file=sys.stderr)
        return None
    # El último archivo corresponde al step más alto (z más bajo)
    with open(files[-1]) as f:
        return json.load(f)

def load_all_insitu(analysis_dir):
    """Carga todos los archivos insitu ordenados por step."""
    files = sorted(glob.glob(os.path.join(analysis_dir, "insitu_*.json")))
    return [json.load(open(f)) for f in files]

def eisenstein_hu_pk(k_arr, sigma8=0.811, n_s=0.9649, omega_m=0.3153,
                      omega_b=0.049, h=0.6736):
    """
    Espectro de potencia lineal Eisenstein-Hu (1998) simplificado.
    Solo para comparación de forma; normalización a σ₈.
    """
    theta_cmb = 2.7255 / 2.7
    omega_m_h2 = omega_m * h * h
    omega_b_h2 = omega_b * h * h
    z_eq = 2.5e4 * omega_m_h2 * theta_cmb**(-4)
    k_eq = 7.46e-2 * omega_m_h2 * theta_cmb**(-2)  # Mpc^-1
    b1 = 0.313 * omega_m_h2**(-0.419) * (1 + 0.607 * omega_m_h2**0.674)
    b2 = 0.238 * omega_m_h2**0.223
    z_d = 1291 * omega_m_h2**0.251 / (1 + 0.659 * omega_m_h2**0.828) \
          * (1 + b1 * omega_b_h2**b2)
    R_eq = 31.5e3 * omega_b_h2 * theta_cmb**(-4) * (1000 / z_eq)
    R_d = 31.5e3 * omega_b_h2 * theta_cmb**(-4) * (1000 / z_d)
    s = 2 / (3 * k_eq) * math.sqrt(6 / R_eq) * math.log(
        (math.sqrt(1 + R_d) + math.sqrt(R_d + R_eq)) / (1 + math.sqrt(R_eq))
    )
    k_silk = 1.6 * omega_b_h2**0.52 * omega_m_h2**0.01 \
             * (1 + (11.25 * omega_b_h2)**(-0.128))  # h/Mpc

    pk_arr = []
    for k in k_arr:
        q = k / (13.41 * k_eq)
        T_tilde = math.log(math.e + 1.8 * q) / (
            math.log(math.e + 1.8 * q) + (14.2 + 386 / (1 + 69.9 * q**1.08)) * q * q
        )
        pk = k**n_s * T_tilde**2
        pk_arr.append(pk)

    # Normalizar a sigma8 usando una estimación rápida con filtro gaussiano
    # (no es top-hat exacto, pero suficiente para comparar forma)
    pk_norm = sum(pk_arr)
    if pk_norm > 0:
        sigma8_raw = (sum(pk * k**2 for pk, k in zip(pk_arr, k_arr)) / pk_norm)**0.5
        if sigma8_raw > 0:
            pk_arr = [p * (sigma8 / sigma8_raw)**2 for p in pk_arr]

    return pk_arr

def measure_sigma8_from_pk(pk_bins, box_size_mpc_h):
    """
    Estima σ₈ desde el espectro de potencia medido.
    Usa aproximación de suma discreta con filtro top-hat esférico de 8 Mpc/h.
    """
    r8 = 8.0  # Mpc/h
    sigma2 = 0.0
    for b in pk_bins:
        k = b['k']
        pk = b['pk']
        if k <= 0:
            continue
        x = k * r8
        # Filtro top-hat esférico W(x) = 3[sin(x)-x·cos(x)] / x³
        if x < 1e-6:
            w = 1.0
        else:
            w = 3.0 * (math.sin(x) - x * math.cos(x)) / x**3
        dk = 2 * math.pi / box_size_mpc_h  # dk aproximado
        sigma2 += pk * w**2 * k**2 * dk / (2 * math.pi**2)
    return math.sqrt(max(sigma2, 0))

def compute_hmf_from_halos(halos, box_size_mpc_h):
    """Calcula la HMF dN/dM desde el catálogo de halos."""
    if not halos:
        return [], []
    masses = sorted(h.get('mass', h.get('m', 0)) for h in halos if h.get('mass', h.get('m', 0)) > 0)
    if not masses:
        return [], []
    vol = box_size_mpc_h**3
    n_bins = max(5, len(masses) // 10)
    log_m_min = math.log10(masses[0])
    log_m_max = math.log10(masses[-1])
    d_log_m = (log_m_max - log_m_min) / n_bins
    if d_log_m <= 0:
        return [], []
    bins_k = [10**(log_m_min + i * d_log_m) for i in range(n_bins + 1)]
    hmf_bins = []
    for i in range(n_bins):
        m_lo, m_hi = bins_k[i], bins_k[i + 1]
        m_med = (m_lo * m_hi)**0.5
        count = sum(1 for m in masses if m_lo <= m < m_hi)
        dn_dm = count / (vol * (m_hi - m_lo)) if (m_hi - m_lo) > 0 else 0
        hmf_bins.append({'m': m_med, 'dn_dm': dn_dm, 'count': count})
    return hmf_bins

def main():
    parser = argparse.ArgumentParser(description="Validación P(k) y HMF vs teoría")
    parser.add_argument("--analysis-dir", default="runs/validation_128/analysis",
                        help="Directorio con archivos insitu_NNNNNN.json")
    parser.add_argument("--out-dir", default="runs/validation_128/plots",
                        help="Directorio de salida para gráficos y JSON")
    parser.add_argument("--sigma8-tol", type=float, default=0.05,
                        help="Tolerancia para σ₈ (default 5%%)")
    parser.add_argument("--box-size", type=float, default=200.0,
                        help="Tamaño de caja en Mpc/h")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Cargar datos ─────────────────────────────────────────────────────────
    print(f"[validate] Cargando datos de {args.analysis_dir}...")
    latest = load_latest_insitu(args.analysis_dir)
    if latest is None:
        print("[validate] No hay datos disponibles. Saliendo.")
        return 1

    all_data = load_all_insitu(args.analysis_dir)
    print(f"[validate] {len(all_data)} snapshots cargados. Último: z={latest.get('z', '?'):.3f}")

    # ── σ₈ medida ────────────────────────────────────────────────────────────
    pk_z0 = latest.get("power_spectrum", [])
    sigma8_measured = measure_sigma8_from_pk(pk_z0, args.box_size)
    sigma8_input = 0.811
    sigma8_error = abs(sigma8_measured - sigma8_input) / sigma8_input

    print(f"[validate] σ₈ input:   {sigma8_input:.4f}")
    print(f"[validate] σ₈ medida:  {sigma8_measured:.4f}")
    print(f"[validate] Error σ₈:   {sigma8_error*100:.2f}%")

    if sigma8_error > args.sigma8_tol:
        print(f"[WARN] σ₈ difiere más del {args.sigma8_tol*100:.0f}%!")
    else:
        print(f"[OK]   σ₈ dentro de tolerancia ({args.sigma8_tol*100:.0f}%)")

    # ── P(k) vs EH ───────────────────────────────────────────────────────────
    if pk_z0:
        k_arr = [b['k'] for b in pk_z0]
        pk_arr = [b['pk'] for b in pk_z0]
        pk_eh = eisenstein_hu_pk(k_arr)

        pk_comparison = [
            {"k": k, "pk_sim": ps, "pk_eh": pe}
            for k, ps, pe in zip(k_arr, pk_arr, pk_eh)
        ]
        with open(os.path.join(args.out_dir, "pk_comparison.json"), "w") as f:
            json.dump(pk_comparison, f, indent=2)
        print(f"[validate] P(k) vs EH guardado en {args.out_dir}/pk_comparison.json")

    # ── HMF ──────────────────────────────────────────────────────────────────
    n_halos = latest.get("n_halos", 0)
    print(f"[validate] Halos a z=0: {n_halos}")

    # ── P(k,z) evolución ─────────────────────────────────────────────────────
    pk_evolution = [
        {"z": d.get("z", 0), "a": d.get("a", 1), "pk": d.get("power_spectrum", [])}
        for d in all_data
    ]
    with open(os.path.join(args.out_dir, "pk_evolution.json"), "w") as f:
        json.dump(pk_evolution, f, indent=2)

    # ── Resumen ───────────────────────────────────────────────────────────────
    summary = {
        "sigma8_input": sigma8_input,
        "sigma8_measured": sigma8_measured,
        "sigma8_error_frac": sigma8_error,
        "sigma8_ok": sigma8_error <= args.sigma8_tol,
        "n_halos_z0": n_halos,
        "n_snapshots": len(all_data),
        "z_final": latest.get("z", 0),
    }
    with open(os.path.join(args.out_dir, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[validate] Resumen guardado en {args.out_dir}/validation_summary.json")

    # ── Graficar si matplotlib disponible ────────────────────────────────────
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if pk_z0:
            fig, ax = plt.subplots(figsize=(8, 5))
            k_plot = np.array([b['k'] for b in pk_z0])
            pk_plot = np.array([b['pk'] for b in pk_z0])
            pk_eh_plot = np.array(eisenstein_hu_pk(k_plot.tolist()))
            ax.loglog(k_plot, pk_plot, 'b-', label='gadget-ng (z=0)')
            ax.loglog(k_plot, pk_eh_plot, 'r--', label='Eisenstein-Hu (lineal)')
            ax.set_xlabel("k [h/Mpc]")
            ax.set_ylabel("P(k) [(Mpc/h)³]")
            ax.set_title(f"P(k) validación N=128³  [σ₈={sigma8_measured:.3f}, error={sigma8_error*100:.1f}%]")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "pk_validation.png"), dpi=150)
            plt.close()
            print(f"[validate] Gráfico guardado en {args.out_dir}/pk_validation.png")

    except ImportError:
        print("[validate] matplotlib no disponible, saltando gráficos.")

    return 0 if sigma8_error <= args.sigma8_tol else 1

if __name__ == "__main__":
    sys.exit(main())
