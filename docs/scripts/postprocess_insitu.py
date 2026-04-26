#!/usr/bin/env python3
"""
postprocess_insitu.py — Procesamiento automático de análisis in-situ (Phase 83)

Carga todos los archivos `insitu_*.json` de un directorio de corrida, genera
gráficos de evolución temporal y escribe un `summary.json` con series temporales.

Estadísticas disponibles:
  - P(k) en espacio real y de redshift
  - Multipoles P₀/P₂/P₄ (Hamilton 1992)
  - σ₈(z) estimado via integración de P(k)
  - Histograma de halos n_halos(z)
  - Bispectrum equilateral B_eq(k) a z=0
  - Assembly bias (Spearman correlación spin vs entorno)

Uso:
  python postprocess_insitu.py --dir runs/cosmo/insitu --out analysis/

Dependencias: numpy, matplotlib, scipy (todas opcionales — el script detecta cuáles están disponibles)
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


# ── Configuración de matplotlib (sin display) ────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[postprocess] matplotlib no disponible — se omitirán los gráficos", file=sys.stderr)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[postprocess] numpy no disponible — algunas funciones estarán limitadas", file=sys.stderr)

try:
    from scipy import integrate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Carga de archivos in-situ ─────────────────────────────────────────────────

def load_insitu_dir(insitu_dir: str) -> list[dict]:
    """Carga todos los insitu_*.json de un directorio, ordenados por step."""
    pattern = os.path.join(insitu_dir, "insitu_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[postprocess] No se encontraron archivos en {insitu_dir}", file=sys.stderr)
        return []
    records = []
    for f in files:
        with open(f) as fh:
            try:
                records.append(json.load(fh))
            except json.JSONDecodeError as e:
                print(f"[postprocess] Error leyendo {f}: {e}", file=sys.stderr)
    print(f"[postprocess] Cargados {len(records)} archivos in-situ de {insitu_dir}")
    return records


# ── σ₈ estimado desde P(k) ───────────────────────────────────────────────────

def sigma8_from_pk(pk_bins: list[dict], box_size: float = 1.0) -> float:
    """
    Estima σ₈ integrando P(k) con una función de ventana top-hat de R=8 Mpc/h.

    σ²(R) = 1/(2π²) ∫ k² P(k) W²(kR) dk
    W(x) = 3 [sin(x) - x cos(x)] / x³

    Args:
        pk_bins: lista de {'k': ..., 'pk': ..., 'n_modes': ...}
        box_size: longitud de la caja en Mpc/h (default 1.0 si ya en Mpc/h)

    Returns:
        σ₈ estimado (0.0 si no hay datos suficientes)
    """
    if not pk_bins or not HAS_NUMPY:
        return 0.0

    k_arr = np.array([b["k"] for b in pk_bins])
    pk_arr = np.array([b["pk"] for b in pk_bins])

    if len(k_arr) < 2:
        return 0.0

    R = 8.0  # Mpc/h

    def w_tophat(x):
        x = np.atleast_1d(x)
        w = np.where(
            x < 1e-4,
            1.0,
            3.0 * (np.sin(x) - x * np.cos(x)) / x**3,
        )
        return w

    integrand = k_arr**2 * pk_arr * w_tophat(k_arr * R) ** 2

    if HAS_SCIPY:
        sigma2, _ = integrate.simpson(integrand, x=k_arr), None
        sigma2 = integrate.simpson(integrand, x=k_arr)
    else:
        # Regla trapezoidal manual
        sigma2 = float(np.trapz(integrand, k_arr))

    sigma2 /= 2.0 * np.pi**2
    return float(np.sqrt(max(sigma2, 0.0)))


# ── Gráficos ──────────────────────────────────────────────────────────────────

def plot_pk_evolution(records: list[dict], out_dir: str):
    """Gráfico de P(k) para varios redshifts."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("plasma")

    for i, rec in enumerate(records[:: max(1, len(records) // 8)]):
        pk = rec.get("power_spectrum", [])
        if not pk:
            continue
        k = [b["k"] for b in pk]
        p = [b["pk"] for b in pk]
        z = rec.get("z", 0.0)
        color = cmap(i / max(1, len(records) // 8 - 1))
        ax.loglog(k, p, label=f"z={z:.2f}", color=color, lw=1.2)

    ax.set_xlabel(r"$k$ [h/Mpc]")
    ax.set_ylabel(r"$P(k)$ [(Mpc/h)³]")
    ax.set_title("Espectro de potencia P(k) — evolución temporal")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "pk_evolution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[postprocess] → {path}")


def plot_pk_multipoles(records: list[dict], out_dir: str):
    """Gráfico de multipoles P₀/P₂/P₄ al último snapshot."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return

    last = next((r for r in reversed(records) if r.get("pk_multipoles")), None)
    if last is None:
        return

    mults = last["pk_multipoles"]
    k = [m["k"] for m in mults]
    p0 = [m["p0"] for m in mults]
    p2 = [m["p2"] for m in mults]
    p4 = [m["p4"] for m in mults]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(k, p0, label=r"$P_0(k)$", color="C0")
    ax.semilogx(k, p2, label=r"$P_2(k)$", color="C1", ls="--")
    ax.semilogx(k, p4, label=r"$P_4(k)$", color="C2", ls=":")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$k$ [h/Mpc]")
    ax.set_ylabel(r"$P_\ell(k)$")
    ax.set_title(f"Multipoles RSD (z={last.get('z', 0.0):.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "pk_multipoles.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[postprocess] → {path}")


def plot_sigma8_evolution(records: list[dict], out_dir: str, box_size: float = 1.0):
    """Gráfico de σ₈(z)."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return

    zs, s8s = [], []
    for rec in records:
        pk = rec.get("power_spectrum", [])
        if not pk:
            continue
        zs.append(rec.get("z", 0.0))
        s8s.append(sigma8_from_pk(pk, box_size))

    if not zs:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(zs, s8s, "o-", color="C3", ms=4)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"$\sigma_8$")
    ax.set_title(r"Evolución de $\sigma_8(z)$")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "sigma8_evolution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[postprocess] → {path}")


def plot_nhalos_evolution(records: list[dict], out_dir: str):
    """Gráfico de n_halos(z)."""
    if not HAS_MATPLOTLIB:
        return

    zs = [r.get("z", 0.0) for r in records]
    nh = [r.get("n_halos", 0) for r in records]
    mt = [r.get("m_total_halos", 0.0) for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(zs, nh, "s-", color="C4", ms=4)
    axes[0].set_xlabel("Redshift z")
    axes[0].set_ylabel("N halos FoF")
    axes[0].set_title("Número de halos FoF")
    axes[0].invert_xaxis()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(zs, mt, "^-", color="C5", ms=4)
    axes[1].set_xlabel("Redshift z")
    axes[1].set_ylabel(r"$M_{\rm halos}$ total")
    axes[1].set_title("Masa total en halos")
    axes[1].invert_xaxis()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "halos_evolution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[postprocess] → {path}")


def plot_bispectrum(records: list[dict], out_dir: str):
    """Gráfico de B_eq(k) al z=0 (o el más cercano)."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return

    last = next((r for r in reversed(records) if r.get("bk_equilateral")), None)
    if last is None:
        return

    bk = last["bk_equilateral"]
    k = [b["k"] for b in bk]
    bk_vals = [b["bk"] for b in bk]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(k, [abs(b) for b in bk_vals], "o-", color="C6", ms=4)
    ax.set_xlabel(r"$k$ [h/Mpc]")
    ax.set_ylabel(r"$B_{\rm eq}(k)$ [(Mpc/h)⁶]")
    ax.set_title(f"Bispectrum equilateral (z={last.get('z', 0.0):.2f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "bispectrum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[postprocess] → {path}")


# ── Summary JSON ──────────────────────────────────────────────────────────────

def build_summary(records: list[dict], box_size: float = 1.0) -> dict:
    """Construye un summary.json con series temporales."""
    summary = {
        "n_snapshots": len(records),
        "box_size": box_size,
        "timeline": [],
    }

    for rec in records:
        pk = rec.get("power_spectrum", [])
        s8 = sigma8_from_pk(pk, box_size) if pk else 0.0

        entry = {
            "step": rec.get("step", 0),
            "a": rec.get("a", 1.0),
            "z": rec.get("z", 0.0),
            "n_halos": rec.get("n_halos", 0),
            "m_total_halos": rec.get("m_total_halos", 0.0),
            "sigma8": s8,
            "n_pk_bins": len(pk),
            "has_pk_rsd": len(rec.get("pk_rsd", [])) > 0,
            "has_bispectrum": len(rec.get("bk_equilateral", [])) > 0,
        }
        if rec.get("assembly_bias"):
            ab = rec["assembly_bias"]
            entry["assembly_bias_spearman_lambda"] = ab.get("spearman_lambda", 0.0)
            entry["assembly_bias_spearman_concentration"] = ab.get("spearman_concentration", 0.0)

        summary["timeline"].append(entry)

    # P(k) final (z=0 o el último con datos)
    for rec in reversed(records):
        if rec.get("power_spectrum"):
            summary["pk_final"] = rec["power_spectrum"]
            summary["z_final"] = rec.get("z", 0.0)
            break

    # Multipoles finales
    for rec in reversed(records):
        if rec.get("pk_multipoles"):
            summary["pk_multipoles_final"] = rec["pk_multipoles"]
            break

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Procesamiento automático de análisis in-situ (Phase 83)"
    )
    parser.add_argument(
        "--dir", required=True, help="Directorio con los archivos insitu_*.json"
    )
    parser.add_argument(
        "--out", default=".", help="Directorio de salida para gráficos y summary.json"
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=1.0,
        help="Tamaño de la caja en Mpc/h (para σ₈). Default: 1.0",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    records = load_insitu_dir(args.dir)
    if not records:
        sys.exit(1)

    # Gráficos
    plot_pk_evolution(records, args.out)
    plot_pk_multipoles(records, args.out)
    plot_sigma8_evolution(records, args.out, args.box_size)
    plot_nhalos_evolution(records, args.out)
    plot_bispectrum(records, args.out)

    # Summary JSON
    summary = build_summary(records, args.box_size)
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[postprocess] → {summary_path}")

    # Resumen de σ₈
    timeline = summary.get("timeline", [])
    if timeline:
        s8_z0 = next(
            (t["sigma8"] for t in reversed(timeline) if t["z"] < 0.1), None
        )
        if s8_z0:
            print(f"[postprocess] σ₈(z≈0) ≈ {s8_z0:.4f}")

    print(f"[postprocess] Completado. {len(records)} snapshots procesados.")


if __name__ == "__main__":
    main()
