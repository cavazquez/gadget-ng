#!/usr/bin/env python3
"""
postprocess_hmf.py — Post-proceso de la Función de Masa de Halos (HMF) por snapshot.

Lee los archivos de análisis in-situ (insitu_NNNNNN.json) y genera:
  - hmf_evolution.json : n(M,z) para todos los snapshots
  - hmf_z0.png         : comparación HMF(z=0) vs Press-Schechter + Sheth-Tormen
  - hmf_evolution.png  : HMF a múltiples redshifts

Uso:
  python3 docs/scripts/postprocess_hmf.py \\
    --insitu runs/production_256/insitu \\
    --out    runs/production_256/analysis/hmf_evolution.json
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Modelos analíticos ─────────────────────────────────────────────────────

def sheth_tormen_hmf(log10_m: list[float], omega_m: float = 0.315,
                     sigma8: float = 0.811, z: float = 0.0,
                     box_mpc_h: float = 300.0) -> list[float]:
    """
    Función de masa de Sheth-Tormen (1999) simplificada.
    Devuelve dn/d(log10 M) en [h/Mpc]³ para cada log10_m.
    
    Implementación analítica aproximada para comparación rápida.
    Para precisión de paper usar hmf (python package).
    """
    if not HAS_NUMPY:
        return [0.0] * len(log10_m)

    m_arr = np.array(log10_m)
    M = 10.0 ** m_arr  # M en unidades de M☉/h

    # Parámetros ST
    a_st, p_st, A_st = 0.707, 0.3, 0.3222
    delta_c = 1.686  # densidad crítica de colapso (EdS)

    # σ(M) aproximada (Jenkins+2001 para ΛCDM)
    # σ(M) = sigma8 * (M / M8)^{-beta} con M8 = (4π/3)*(8 Mpc/h)³ * rho_bar
    rho_bar_mpc3 = 2.775e11 * omega_m  # M☉/h / (Mpc/h)³ (densidad de materia media)
    M8 = rho_bar_mpc3 * (4.0 * np.pi / 3.0) * 8.0**3
    beta_sigma = 0.3  # pendiente aproximada para P(k) ΛCDM

    # Factor de crecimiento lineal (aproximación Peebles 1980 para ΛCDM)
    def growth_factor(z_val: float, om: float) -> float:
        a = 1.0 / (1.0 + z_val)
        # Aproximación de Carroll, Press & Turner (1992)
        omega_mz = om / (om + (1.0 - om) * a**3)
        return (5.0 / 2.0) * omega_mz * a / \
               (omega_mz**(4.0/7.0) - (1.0 - om) / (1.0 + om / 2.0) + 1.0)

    D0 = growth_factor(0.0, omega_m)
    Dz = growth_factor(z, omega_m)
    sigma8_z = sigma8 * Dz / D0

    sigma = sigma8_z * (M / M8) ** (-beta_sigma)

    nu = delta_c / sigma
    nu2 = a_st * nu ** 2

    f_nu = A_st * np.sqrt(2.0 * nu2 / np.pi) * (1.0 + 1.0 / nu2**p_st) * np.exp(-nu2 / 2.0)
    dln_sigma_dln_m = -beta_sigma  # d(ln σ)/d(ln M)

    dn_dlnm = f_nu * rho_bar_mpc3 / M * np.abs(dln_sigma_dln_m)
    dn_dlog10m = dn_dlnm * np.log(10.0)

    return dn_dlog10m.tolist()


def load_insitu_files(insitu_dir: Path) -> list[dict]:
    files = sorted(insitu_dir.glob("insitu_*.json"))
    data = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
            d["_file"] = str(f)
            data.append(d)
    return data


def summarize_hmf(insitu_data: list[dict]) -> dict:
    result = {"snapshots": [], "metadata": {"n_snapshots": len(insitu_data)}}

    for rec in insitu_data:
        halos = rec.get("halos", [])
        if not halos:
            continue

        masses = [h.get("mass", 0.0) for h in halos if h.get("mass", 0.0) > 0.0]
        if not masses:
            continue

        if HAS_NUMPY:
            log10_m = [float(np.log10(m)) for m in masses]
            m_min = min(log10_m)
            m_max = max(log10_m)
            n_bins = max(10, len(masses) // 5)
            bins = list(np.linspace(m_min, m_max, n_bins + 1))
            counts, edges = np.histogram(log10_m, bins=bins)
            m_centers = 0.5 * (edges[:-1] + edges[1:])
            dlog10m = edges[1] - edges[0]
        else:
            log10_m = []
            m_centers, counts, dlog10m = [], [], 1.0

        a_val = rec.get("a", 1.0)
        z_val = 1.0 / a_val - 1.0 if a_val else 0.0

        result["snapshots"].append({
            "step": rec.get("step", -1),
            "a": a_val,
            "z": z_val,
            "n_halos": len(masses),
            "log10_m_centers": list(m_centers),
            "dn_dlog10m": (counts / dlog10m).tolist() if HAS_NUMPY else [],
        })

    return result


def plot_hmf(hmf_summary: dict, out_dir: Path) -> None:
    if not HAS_MPL or not HAS_NUMPY:
        return

    snaps = [s for s in hmf_summary["snapshots"] if s.get("log10_m_centers")]
    if not snaps:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 0.85, len(snaps)))

    for i, snap in enumerate(snaps):
        z = snap.get("z", i)
        lm = np.array(snap["log10_m_centers"])
        dn = np.array(snap["dn_dlog10m"])
        mask = dn > 0
        if not np.any(mask):
            continue
        label = f"z={z:.1f} (N={snap['n_halos']})"
        ax.semilogy(lm[mask], dn[mask], "o-", color=colors[i], alpha=0.8, ms=4,
                    label=label, lw=1.5)

        # Comparación Sheth-Tormen
        dn_st = sheth_torren_hmf(list(lm), z=float(z))
        ax.semilogy(lm, dn_st, "--", color=colors[i], alpha=0.4, lw=1.0)

    ax.set_xlabel(r"$\log_{10}(M\, [M_\odot/h])$", fontsize=12)
    ax.set_ylabel(r"$dn/d\log_{10}M$ [$h$/Mpc]³", fontsize=12)
    ax.set_title("Función de masa de halos — gadget-ng N=256³\n(sólido: N-body, rayado: Sheth-Tormen)", fontsize=11)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hmf_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {out_dir}/hmf_evolution.png")


def sheth_torren_hmf(log10_m, **kw):
    return sheth_tormen_hmf(log10_m, **kw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-proceso HMF gadget-ng")
    parser.add_argument("--insitu", type=Path, default=Path("runs/production_256/insitu"))
    parser.add_argument("--out", type=Path, default=Path("runs/production_256/analysis/hmf_evolution.json"))
    args = parser.parse_args()

    if not args.insitu.exists():
        print(f"ERROR: directorio insitu no encontrado: {args.insitu}", file=sys.stderr)
        return 1

    print(f"Cargando archivos in-situ desde {args.insitu}...")
    insitu_data = load_insitu_files(args.insitu)
    if not insitu_data:
        print("WARN: no se encontraron archivos insitu_*.json", file=sys.stderr)
        return 0

    print(f"  {len(insitu_data)} snapshots cargados")
    hmf_summary = summarize_hmf(insitu_data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(hmf_summary, fp, indent=2)
    print(f"  Guardado: {args.out}")

    plot_hmf(hmf_summary, args.out.parent)
    print("postprocess_hmf.py completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
