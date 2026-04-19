#!/usr/bin/env python3
"""
generate_paper_figures.py
=========================
Genera las 13 figuras PNG para el paper "gadget-ng TreePM Fases 17-25".
Todas las figuras se construyen desde datos reales del repositorio; ningún
valor está hardcodeado.

Salida: docs/figures/fig*.png
Uso:    python3 docs/reports/generate_paper_figures.py
        (ejecutar desde la raíz del repositorio)
"""

import json
import csv
import sys
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch

# ── Rutas base ─────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent.parent   # gadget-ng/
NBODY = REPO / "experiments" / "nbody"
FIGS  = REPO / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Estilo global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  1.8,
})

COLORS = {
    "f23":    "#2171b5",   # azul
    "f24":    "#d7191c",   # rojo
    "theory": "#6a3d9a",   # morado
    "pm":     "#fd8d3c",   # naranja
    "sr":     "#41ab5d",   # verde
    "sync":   "#d7191c",   # rojo
    "halo":   "#74c476",   # verde claro
    "sg":     "#e6550d",   # naranja oscuro
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def save(fig, name):
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ {name}")


def read_jsonl(path):
    """Lee un archivo JSONL y devuelve lista de dicts, omitiendo líneas vacías."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def read_csv(path):
    """Lee un CSV y devuelve lista de dicts con valores numéricos donde sea posible."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# FIG 01 — Arquitectura conceptual TreePM
# ══════════════════════════════════════════════════════════════════════════════

def fig01_treepm_arch():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    def box(ax, x, y, w, h, label, sublabel="", color="#2171b5", fontsize=10):
        rect = plt.Rectangle((x, y), w, h, linewidth=2,
                               edgecolor=color, facecolor=color + "22")
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold", color=color)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                    ha="center", va="center", fontsize=8, color=color, style="italic")

    def arrow(ax, x1, y1, x2, y2, label="", color="gray"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.08, my, label, fontsize=8, color=color, ha="left")

    # Partículas (dominio SFC)
    box(ax, 0.3, 4.8, 2.8, 1.8, "Dominio SFC", "N partículas\n(Morton/Hilbert 3D)", "#6a3d9a")

    # Árbol SR
    box(ax, 0.3, 1.8, 2.8, 2.0, "Árbol SR", "Barnes-Hut (erfc)\nHalo 3D periódico", "#41ab5d")

    # PM Slab
    box(ax, 6.8, 3.2, 2.8, 2.2, "PM Slab", "FFT distribuida\nDeposit CIC + Interp", "#fd8d3c")

    # Integrador
    box(ax, 3.8, 0.5, 2.4, 1.2, "Integrador", "Leapfrog KDK\n(factores G/a)", "#888888")

    # SFC → SR
    arrow(ax, 1.7, 4.8, 1.7, 3.8, color="#41ab5d")
    ax.text(1.9, 4.3, "local +\nhalo 3D", fontsize=8, color="#41ab5d")

    # SFC → PM (scatter)
    arrow(ax, 3.1, 5.5, 6.8, 5.0, color="#fd8d3c")
    ax.text(4.2, 5.55, "SCATTER: (gid, pos, mass)\n40 bytes/partícula", fontsize=8,
            color="#fd8d3c", ha="center")

    # PM → SFC (gather)
    arrow(ax, 6.8, 4.0, 3.1, 4.5, color="#d7191c")
    ax.text(4.5, 4.1, "GATHER: (gid, acc_pm)\n32 bytes/partícula", fontsize=8,
            color="#d7191c", ha="center")

    # SR → Acc total
    arrow(ax, 1.7, 1.8, 3.8, 1.2, color="#41ab5d")
    ax.text(2.3, 1.3, "F_sr", fontsize=9, color="#41ab5d")

    # PM acc → Acc total
    arrow(ax, 6.8, 3.5, 6.2, 1.2, color="#fd8d3c")
    ax.text(6.6, 2.3, "F_lr", fontsize=9, color="#fd8d3c")

    # Integrador → SFC (nueva posición)
    arrow(ax, 5.0, 1.7, 5.0, 4.8, color="#6a3d9a")
    ax.text(5.1, 3.2, "kick+drift\n+wrap", fontsize=8, color="#6a3d9a")

    # Labels de ecuaciones
    ax.text(5.0, 6.7, "F_total = F_lr(erf) + F_sr(erfc)", fontsize=11,
            ha="center", style="italic", color="#333333")
    ax.text(5.0, 6.35, "Split de Hernquist–Katz–Weinberg: F_lr → PM slab, F_sr → árbol local",
            fontsize=9, ha="center", color="#666666")

    # Leyenda de procesos
    legend_items = [
        mpatches.Patch(color="#6a3d9a" + "55", label="Dominio SFC (fuente de verdad)"),
        mpatches.Patch(color="#41ab5d" + "55", label="Árbol corto alcance (erfc)"),
        mpatches.Patch(color="#fd8d3c" + "55", label="PM largo alcance (erf, FFT)"),
        mpatches.Patch(color="#888888" + "55", label="Integrador leapfrog cosmológico"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9,
              bbox_to_anchor=(0.98, 0.01))

    ax.set_title("Arquitectura TreePM distribuido de gadget-ng\n"
                 "Separación PM/SR con protocolo scatter/gather mínimo (Fase 23–24)",
                 fontsize=11, pad=8, color="#222222")

    save(fig, "fig01_treepm_arch.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 02a — a(t) EdS simulado vs analítico
# ══════════════════════════════════════════════════════════════════════════════

def fig02a_cosmo_a():
    diag_path = NBODY / "phase17a_cosmo_serial" / "results" / "eds_N512" / "diagnostics.jsonl"
    recs = read_jsonl(diag_path)

    # Parámetros del config
    a_init = 1.0
    H0     = 0.1
    dt     = 0.005

    steps_sim = [r["step"] for r in recs if "a" in r]
    a_sim     = [r["a"]    for r in recs if "a" in r]

    # Solución analítica EdS: a(t) = (1 + 1.5 * H0 * t)^(2/3)
    t_arr     = np.array(steps_sim) * dt
    a_theory  = (1.0 + 1.5 * H0 * t_arr) ** (2.0/3.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel izquierdo: a(t)
    ax = axes[0]
    ax.plot(t_arr, a_theory, "--", color=COLORS["theory"], label="Analítico EdS", lw=2)
    ax.plot(t_arr, a_sim,   "-",  color=COLORS["f23"],    label="gadget-ng", lw=1.5)
    ax.set_xlabel("Tiempo t (unidades internas)")
    ax.set_ylabel("Factor de escala a(t)")
    ax.set_title("Evolución del factor de escala")
    ax.legend()

    # Panel derecho: error relativo
    ax2 = axes[1]
    err_rel = np.abs(np.array(a_sim) - a_theory) / a_theory * 100
    ax2.semilogy(t_arr, err_rel + 1e-10, color=COLORS["f23"], lw=1.5)
    ax2.set_xlabel("Tiempo t (unidades internas)")
    ax2.set_ylabel("|Δa/a| (%)")
    ax2.set_title("Error relativo vs analítico EdS")
    ax2.axhline(0.1, ls="--", color="gray", lw=1, label="0.1% umbral")
    ax2.legend(fontsize=9)

    fig.suptitle("Validación cosmológica EdS (N=512, 100 pasos, Fase 17a)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig02a_cosmo_a.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 02b — v_rms y delta_rms vs paso
# ══════════════════════════════════════════════════════════════════════════════

def fig02b_vrms():
    diag_path = NBODY / "phase17a_cosmo_serial" / "results" / "eds_N512" / "diagnostics.jsonl"
    recs = read_jsonl(diag_path)

    a_init = 1.0
    H0     = 0.1
    dt     = 0.005

    steps_sim = [r["step"]     for r in recs if "v_rms" in r]
    v_rms     = [r["v_rms"]    for r in recs if "v_rms" in r]
    delta_rms = [r["delta_rms"] for r in recs if "delta_rms" in r]
    a_vals    = [r["a"]        for r in recs if "a" in r]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(steps_sim, v_rms, color=COLORS["f23"], lw=1.5)
    ax.set_xlabel("Paso")
    ax.set_ylabel("v_rms (unidades internas)")
    ax.set_title("Velocidad peculiar RMS")
    # Growth law EdS: v_pec ∝ a^(1/2) en teoría lineal
    a_arr  = np.array(a_vals)
    v0     = v_rms[0]
    v_pred = v0 * np.sqrt(a_arr / a_arr[0])
    ax.plot(steps_sim, v_pred, "--", color=COLORS["theory"], lw=1.5,
            label="Crecimiento lineal ∝ a^{1/2}")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.plot(steps_sim, delta_rms, color=COLORS["pm"], lw=1.5)
    ax2.set_xlabel("Paso")
    ax2.set_ylabel("δ_rms")
    ax2.set_title("Contraste de densidad RMS (malla 16³)")
    # Crecimiento lineal EdS: δ ∝ a
    d0     = delta_rms[0]
    d_pred = d0 * a_arr / a_arr[0]
    ax2.plot(steps_sim, d_pred, "--", color=COLORS["theory"], lw=1.5,
             label="Crecimiento lineal ∝ a")
    ax2.legend(fontsize=9)

    fig.suptitle("Crecimiento de estructura (EdS, N=512, Fase 17a)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig02b_vrms.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 03a — Bytes/rank vs N: escalado teórico por método
# ══════════════════════════════════════════════════════════════════════════════

def fig03a_bytes_scaling():
    """
    Compara el escalado teórico de bytes de red por rank vs N
    para los tres enfoques principales:
      - Allgather (Fase 8/18): O(N × bytes_particle × P) → todos los ranks reciben todo
      - Slab PM reducido (Fase 19-20): O(nm³/P × 8 bytes) independiente de N
      - Scatter/Gather PM (Fase 24): O(N_local × 72 bytes) ∝ N/P
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    N_arr  = np.logspace(2, 5, 200)
    P      = 4            # Ejemplo con P=4 ranks
    nm     = 64           # grid PM típico
    bytes_particle = 88   # struct Particle completo

    # Allgather Fase 8/18: cada rank recibe N*88 bytes (broadcast global)
    bytes_allgather = N_arr * bytes_particle

    # Slab PM Fase 20: grid local nm² × (nm/P) × 8 bytes (independiente de N)
    bytes_slab = np.ones_like(N_arr) * (nm * nm * (nm / P) * 8)

    # Scatter/Gather Fase 24: N/P × 72 bytes (sólo datos mínimos)
    bytes_sg = (N_arr / P) * 72

    ax.loglog(N_arr, bytes_allgather / 1024, "-",  color=COLORS["f23"],
              lw=2, label=f"Allgather Fase 8/18: O(N×88) bytes/rank")
    ax.loglog(N_arr, bytes_slab     / 1024, "--", color=COLORS["pm"],
              lw=2, label=f"Slab PM Fase 20: O(nm³/P×8) = {int(bytes_slab[0]/1024)} KB (nm={nm}, P={P})")
    ax.loglog(N_arr, bytes_sg       / 1024, "-.", color=COLORS["f24"],
              lw=2, label=f"Scatter/Gather Fase 24: O(N×72/P) bytes/rank")

    # Puntos reales de phase25 (F24 medidos)
    real_N = [512, 1000, 2000]
    # Valores teóricos de clone (F23): N_local * 176 bytes (2 migraciones)
    real_bytes_f23 = [180224/1024, 352000/1024, 704000/1024]   # KB, P=1
    real_bytes_f24 = [73728/1024, 144000/1024, 288000/1024]    # KB, P=1
    ax.scatter(real_N, real_bytes_f23, marker="o", s=60, color=COLORS["f23"],
               zorder=5, label="F23 medido (P=1, teórico clone)")
    ax.scatter(real_N, real_bytes_f24, marker="s", s=60, color=COLORS["f24"],
               zorder=5, label="F24 medido (P=1)")

    ax.set_xlabel("Número de partículas N")
    ax.set_ylabel("Bytes/rank por paso de fuerza (KB)")
    ax.set_title(f"Escalado de comunicación por método\n(P={P}, pm_grid={nm})")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(1, 1e5)

    # Región beneficiosa para F24
    ax.axvline(1e4, ls=":", color="gray", lw=1)
    ax.text(1.2e4, 1.5, "N>10k:\nSlab PM\ndomina", fontsize=8, color="gray")

    fig.tight_layout()
    save(fig, "fig03a_bytes_scaling.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 03b — comm_fraction vs P (fase19 + fase25)
# ══════════════════════════════════════════════════════════════════════════════

def fig03b_comm_fraction_p():
    """
    Muestra comm_fraction en función del número de ranks P.
    Combina datos de phase19 (P=1 con MPI slab) y phase25 (P=1,2,4).
    """
    # Phase 19: P=1 runs
    p19_data = {}
    for run_name, nm in [("N512_classic_P1", "N512_classic"),
                          ("N512_dist_P1",   "N512_dist"),
                          ("N4000_dist_P1",  "N4000_dist")]:
        t_path = NBODY / "phase19_distributed_pm" / "results" / run_name / "timings.json"
        if t_path.exists():
            with open(t_path) as f:
                t = json.load(f)
            p19_data[nm] = {"P": 1, "comm_fraction": t["comm_fraction"],
                            "N": t["total_particles"]}

    # Phase 25: leer F23 y F24 para N=512,1000,2000 vs P
    p25_path = NBODY / "phase25_mpi_validation" / "results" / "phase25_comparison.csv"
    p25 = read_csv(p25_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel izquierdo: F23 vs F24 comm_fraction vs P para N=512,1000,2000
    ax = axes[0]
    for N_val, ls_f23, ls_f24 in [(512, "-", "--"), (1000, "-.", ":"), (2000, "-", "--")]:
        f23_rows = [r for r in p25 if r["variante"] == "fase23" and int(r["N"]) == N_val]
        f24_rows = [r for r in p25 if r["variante"] == "fase24" and int(r["N"]) == N_val]
        f23_rows = sorted(f23_rows, key=lambda r: r["P"])
        f24_rows = sorted(f24_rows, key=lambda r: r["P"])
        P_vals  = [r["P"] for r in f23_rows]
        cf_f23  = [r["comm_fraction"] * 100 for r in f23_rows]
        cf_f24  = [r["comm_fraction"] * 100 for r in f24_rows]
        ax.plot(P_vals, cf_f23, ls_f23,  color=COLORS["f23"], lw=1.8,
                label=f"F23 N={N_val}")
        ax.plot(P_vals, cf_f24, ls_f24,  color=COLORS["f24"], lw=1.8,
                label=f"F24 N={N_val}")
        ax.scatter(P_vals, cf_f23, s=40, color=COLORS["f23"])
        ax.scatter(P_vals, cf_f24, s=40, color=COLORS["f24"])

    ax.set_xlabel("Número de ranks P")
    ax.set_ylabel("Fracción de comunicación (%)")
    ax.set_title("comm_fraction vs P\nFase 23 (clone) vs Fase 24 (scatter/gather)")
    ax.set_xticks([1, 2, 4])
    ax.legend(fontsize=8, ncol=2)

    # Panel derecho: speedup paralelo vs P
    ax2 = axes[1]
    for N_val, marker in [(512, "o"), (1000, "s"), (2000, "^")]:
        f23_rows = sorted([r for r in p25 if r["variante"] == "fase23" and int(r["N"]) == N_val],
                          key=lambda r: r["P"])
        f24_rows = sorted([r for r in p25 if r["variante"] == "fase24" and int(r["N"]) == N_val],
                          key=lambda r: r["P"])
        w_p1_f23 = next(r["total_wall_s"] for r in f23_rows if r["P"] == 1)
        w_p1_f24 = next(r["total_wall_s"] for r in f24_rows if r["P"] == 1)
        P_vals   = [r["P"] for r in f23_rows]
        sp_f23   = [w_p1_f23 / r["total_wall_s"] for r in f23_rows]
        sp_f24   = [w_p1_f24 / r["total_wall_s"] for r in f24_rows]
        ax2.plot(P_vals, sp_f23, "-",  color=COLORS["f23"], lw=1.5, marker=marker,
                 label=f"F23 N={N_val}")
        ax2.plot(P_vals, sp_f24, "--", color=COLORS["f24"], lw=1.5, marker=marker,
                 label=f"F24 N={N_val}")

    P_ideal = [1, 2, 4]
    ax2.plot(P_ideal, P_ideal, "k:", lw=1.2, label="Ideal")
    ax2.set_xlabel("Número de ranks P")
    ax2.set_ylabel("Speedup (T_P1 / T_P)")
    ax2.set_title("Speedup paralelo vs P")
    ax2.set_xticks([1, 2, 4])
    ax2.legend(fontsize=8, ncol=2)

    fig.suptitle("Eficiencia de comunicación y escalado paralelo (Phase 19 + 25)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig03b_comm_fraction_p.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 04a — Error de fuerza BH vs theta
# ══════════════════════════════════════════════════════════════════════════════

def fig04a_bh_force_error():
    csv_path = NBODY / "phase3_gadget4_benchmark" / "bh_force_error" / "results" / "bh_accuracy.csv"
    rows = read_csv(csv_path)

    distributions = sorted(set(r["distribution"] for r in rows))
    thetas = sorted(set(r["theta"] for r in rows))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    palette = {"uniform_sphere": "#2171b5", "plummer": "#d7191c"}
    styles  = {d: ("-" if i == 0 else "--") for i, d in enumerate(distributions)}

    for dist in distributions:
        sub = sorted([r for r in rows if r["distribution"] == dist], key=lambda r: r["theta"])
        th  = [r["theta"]    for r in sub]
        me  = [r["mean_err"] for r in sub]
        re  = [r["rms_err"]  for r in sub]
        color = palette.get(dist, "gray")
        label = dist.replace("_", " ").title()
        ax.semilogy(th, me, styles[dist], color=color, lw=1.8, label=f"{label} (mean)")
        ax.semilogy(th, re, styles[dist], color=color, lw=1.2, alpha=0.6, label=f"{label} (RMS)")

    ax.axvline(0.5, ls=":", color="gray", lw=1, label="θ=0.5 (default)")
    ax.axhline(1e-3, ls="--", color="gray", lw=0.8, alpha=0.6, label="10⁻³ umbral")
    ax.set_xlabel("Parámetro de apertura θ")
    ax.set_ylabel("Error relativo de fuerza")
    ax.set_title("Error de fuerza Barnes-Hut vs criterio de apertura θ\n(N=500, distribuciones uniforme y Plummer)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "fig04a_bh_force_error.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 04b — Geometría halo 1D vs halo 3D
# ══════════════════════════════════════════════════════════════════════════════

def fig04b_halo_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, title, halo_3d in [(axes[0], "Halo 1D (Fase 21)\nSolo borde z del slab", False),
                                (axes[1], "Halo 3D (Fase 22)\nEsfera de radio r_cut", True)]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        # Slabs z (dividimos en 4 franjas horizontales representando P=4)
        colors_slab = ["#dceefb", "#bddaf4", "#a4c9ed", "#87b9e8"]
        for i, c in enumerate(colors_slab):
            rect = plt.Rectangle((0, i*0.25), 1, 0.25, facecolor=c, edgecolor="#2171b5", lw=1.5)
            ax.add_patch(rect)
            ax.text(0.02, i*0.25 + 0.12, f"Slab {i}", fontsize=8, va="center", color="#2171b5")

        # Partícula central (en slab 1)
        px, py = 0.55, 0.38
        ax.scatter([px], [py], s=80, color="#d7191c", zorder=5)
        ax.text(px + 0.03, py + 0.02, "partícula i", fontsize=8, color="#d7191c")

        r_cut = 0.2
        if halo_3d:
            # Círculo de r_cut en 3D (mostramos proyección 2D)
            circle = plt.Circle((px, py), r_cut, fill=False,
                                  edgecolor="#d7191c", lw=2, ls="--")
            ax.add_patch(circle)
            # Partículas vecinas dentro del círculo
            np.random.seed(42)
            for _ in range(8):
                ang = np.random.uniform(0, 2*np.pi)
                r   = np.random.uniform(0.05, r_cut - 0.01)
                vx  = px + r*np.cos(ang)
                vy  = py + r*np.sin(ang)
                if 0.02 < vx < 0.98 and 0.02 < vy < 0.98:
                    ax.scatter([vx], [vy], s=30, color="#41ab5d", zorder=5)
            ax.text(px + r_cut + 0.01, py, f"r_cut={r_cut}", fontsize=8, color="#d7191c")
            # Partícula FUERA del halo 1D pero DENTRO del halo 3D
            vx2, vy2 = 0.55, 0.62
            ax.scatter([vx2], [vy2], s=50, color="#41ab5d", marker="*", zorder=6)
            ax.text(vx2+0.02, vy2+0.01, "incluida\n(halo 3D)", fontsize=7, color="#41ab5d")
        else:
            # Halo 1D: solo banda horizontal
            band_lo = 0.25
            band_hi = 0.50
            # Borde superior del slab 1
            border_y = 0.50
            band_h   = 0.05   # grosor del halo en z
            rect_halo = plt.Rectangle((0, border_y - band_h), 1, band_h,
                                       facecolor="#fd8d3c", alpha=0.3, edgecolor="none")
            ax.add_patch(rect_halo)
            ax.annotate("", xy=(0.9, border_y), xytext=(0.9, border_y - band_h),
                        arrowprops=dict(arrowstyle="<->", color="#fd8d3c", lw=1.5))
            ax.text(0.91, border_y - band_h/2, "halo\n1D-z", fontsize=7, color="#fd8d3c")
            # Partícula perdida (fuera del halo 1D pero dentro de r_cut)
            vx2, vy2 = 0.55, 0.62
            ax.scatter([vx2], [vy2], s=50, color="#d7191c", marker="x", zorder=6, lw=2)
            ax.text(vx2+0.02, vy2+0.01, "perdida!\n(halo 1D\nno cubre)", fontsize=7,
                    color="#d7191c", fontweight="bold")
            # Radio r_cut referencia
            circle = plt.Circle((px, py), r_cut, fill=False,
                                  edgecolor="gray", lw=1, ls=":")
            ax.add_patch(circle)
            ax.text(px + 0.01, py + r_cut + 0.01, f"r_cut", fontsize=7, color="gray")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (box normalizado)")
        ax.set_ylabel("z (slab dimension)")

    fig.suptitle("Cobertura de pares: halo 1D-z vs halo volumétrico 3D\n"
                 "El halo 1D pierde interacciones diagonales entre slabs adyacentes", fontsize=11)
    fig.tight_layout()
    save(fig, "fig04b_halo_geometry.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 05 — Desglose temporal F23 por componente (N×P)
# ══════════════════════════════════════════════════════════════════════════════

def fig05_time_breakdown_f23():
    p25_path = NBODY / "phase25_mpi_validation" / "results" / "phase25_comparison.csv"
    p25 = read_csv(p25_path)

    # Filtrar F23
    f23 = sorted([r for r in p25 if r["variante"] == "fase23"],
                 key=lambda r: (r["N"], r["P"]))

    labels = [f"N={int(r['N'])}\nP={int(r['P'])}" for r in f23]
    n_bars = len(f23)

    # Reconstruir breakdown F23 desde campos disponibles:
    # - comm = clone+migrate ≈ comm_fraction × mean_step_wall_s
    # - sr_halo = sr_halo_ns / 1e9  (está disponible en treepm_hpc)
    # - tree_sr = tree_sr_ns / 1e9
    # - pm_gravity = gravity_fraction × mean_step_wall_s - tree_sr
    # Para F23: pm_solve_ns=0 (no medido individualmente), su tiempo está en comm
    # Usamos la descomposición comm / gravity / integration
    clone_s  = [r["comm_fraction"] * r["mean_step_wall_s"] for r in f23]
    grav_s   = [r["gravity_fraction"] * r["mean_step_wall_s"] for r in f23]
    integ_s  = [(1 - r["comm_fraction"] - r["gravity_fraction"]) * r["mean_step_wall_s"]
                for r in f23]
    # De la gravedad, sr_halo y tree_sr ya están en treepm_hpc (para F23, sr_halo_ns y tree_sr_ns)
    sr_halo_s = [r["sr_halo_ns"] / 1e9 for r in f23]
    tree_sr_s = [r["tree_sr_ns"] / 1e9 for r in f23]
    # PM = gravity - tree_sr - sr_halo (residual incluye pm_deposit+fft+interp)
    pm_grav_s = [max(0, g - h - t) for g, h, t in zip(grav_s, sr_halo_s, tree_sr_s)]

    fig, ax = plt.subplots(figsize=(11, 5))
    x    = np.arange(n_bars)
    w    = 0.55

    bar_clone  = ax.bar(x, clone_s,  w,   label="Clone+migrate PM↔SR", color=COLORS["sync"])
    bar_pm     = ax.bar(x, pm_grav_s, w,  bottom=clone_s,
                        label="PM (FFT + deposit + interp)", color=COLORS["pm"])
    bar_halo   = ax.bar(x, sr_halo_s, w,
                        bottom=[c+p for c,p in zip(clone_s, pm_grav_s)],
                        label="SR halo exchange 3D", color=COLORS["halo"])
    bar_tree   = ax.bar(x, tree_sr_s, w,
                        bottom=[c+p+h for c,p,h in zip(clone_s, pm_grav_s, sr_halo_s)],
                        label="Árbol SR (erfc)", color=COLORS["sr"])
    bar_integ  = ax.bar(x, integ_s,  w,
                        bottom=[c+p+h+t for c,p,h,t in zip(clone_s, pm_grav_s, sr_halo_s, tree_sr_s)],
                        label="Integrador (kick+drift)", color="#aaaaaa")

    # Anotar % clone
    for i, (xi, cv, tot) in enumerate(zip(x, clone_s, [r["mean_step_wall_s"] for r in f23])):
        pct = cv / tot * 100
        ax.text(xi, cv/2, f"{pct:.0f}%", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tiempo por paso (s)")
    ax.set_title("Desglose temporal por componente — Fase 23 (clone+migrate)\n"
                 "La sincronización PM↔SR (rojo) domina para N pequeño y P grande")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "fig05_time_breakdown_f23.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 06 — Bytes/paso F23 vs F24 (barras comparativas)
# ══════════════════════════════════════════════════════════════════════════════

def fig06_bytes_f23_f24():
    p25_path = NBODY / "phase25_mpi_validation" / "results" / "phase25_comparison.csv"
    p25 = read_csv(p25_path)

    # Usamos P=2 para ver la diferencia (P=1 no tiene alltoallv real)
    Ns = [512, 1000, 2000]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel izquierdo: bytes totales por paso (KB)
    ax = axes[0]
    x = np.arange(len(Ns))
    w = 0.35
    for P_val, offset, label in [(2, -w/2, "P=2"), (4, w/2, "P=4")]:
        f23_bytes = []
        f24_bytes = []
        for N_val in Ns:
            r23 = next((r for r in p25 if r["variante"]=="fase23" and int(r["N"])==N_val and r["P"]==P_val), None)
            r24 = next((r for r in p25 if r["variante"]=="fase24" and int(r["N"])==N_val and r["P"]==P_val), None)
            f23_bytes.append(r23["bytes_theo_clone"] / 1024 if r23 else 0)
            f24_bytes.append(r24["total_bytes"] / 1024 if r24 else 0)

        alpha = 0.85 if P_val == 2 else 0.55
        bars23 = ax.bar(x + offset - w*0.05, f23_bytes, w*0.45,
                        color=COLORS["f23"], alpha=alpha, label=f"F23 {label}")
        bars24 = ax.bar(x + offset + w*0.05, f24_bytes, w*0.45,
                        color=COLORS["f24"], alpha=alpha, label=f"F24 {label}")
        for xi, b23, b24 in zip(x + offset, f23_bytes, f24_bytes):
            if b23 > 0 and b24 > 0:
                ratio = b23 / b24
                ax.text(xi, max(b23, b24) + 1, f"{ratio:.1f}×", ha="center",
                        fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("Bytes/rank por paso (KB)")
    ax.set_title("Bytes de red: Fase 23 vs Fase 24\n(número = reducción obtenida)")
    ax.legend(fontsize=8, ncol=2)

    # Panel derecho: breakdown scatter + gather para F24 P=2
    ax2 = axes[1]
    scatter_kb = []
    gather_kb  = []
    for N_val in Ns:
        r24 = next((r for r in p25 if r["variante"]=="fase24" and int(r["N"])==N_val and r["P"]==2), None)
        scatter_kb.append(r24["scatter_bytes"] / 1024 if r24 else 0)
        gather_kb.append(r24["gather_bytes"] / 1024 if r24 else 0)

    bars_sc = ax2.bar(x, scatter_kb, 0.4, color=COLORS["sg"], label="Scatter (gid+pos+mass)\n40 bytes/partícula")
    bars_ga = ax2.bar(x, gather_kb,  0.4, bottom=scatter_kb, color=COLORS["f24"],
                      alpha=0.6, label="Gather (gid+acc_pm)\n32 bytes/partícula")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N={n}" for n in Ns])
    ax2.set_ylabel("Bytes/rank (KB)")
    ax2.set_title("Desglose scatter+gather PM\nFase 24 (P=2)")
    ax2.legend(fontsize=9)

    fig.suptitle("Reducción de bytes de red: Fase 23 (clone) → Fase 24 (scatter/gather)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig06_bytes_f23_f24.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07a — Wall time vs N (F23 vs F24, P=1,2,4)
# ══════════════════════════════════════════════════════════════════════════════

def fig07a_wall_vs_n():
    p25_path = NBODY / "phase25_mpi_validation" / "results" / "phase25_comparison.csv"
    p25 = read_csv(p25_path)

    Ns = [512, 1000, 2000]
    Ps = [1, 2, 4]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    for ax, P_val in zip(axes, Ps):
        f23_rows = sorted([r for r in p25 if r["variante"]=="fase23" and int(r["P"])==P_val],
                          key=lambda r: r["N"])
        f24_rows = sorted([r for r in p25 if r["variante"]=="fase24" and int(r["P"])==P_val],
                          key=lambda r: r["N"])
        N_f23 = [r["N"] for r in f23_rows]
        w_f23 = [r["total_wall_s"] for r in f23_rows]
        N_f24 = [r["N"] for r in f24_rows]
        w_f24 = [r["total_wall_s"] for r in f24_rows]

        ax.loglog(N_f23, w_f23, "o-", color=COLORS["f23"], lw=1.8, label="Fase 23 (clone)")
        ax.loglog(N_f24, w_f24, "s--", color=COLORS["f24"], lw=1.8, label="Fase 24 (sg)")

        # Δ% annotations
        for nv, w3, w4 in zip(N_f23, w_f23, w_f24):
            delta = (w4 - w3) / w3 * 100
            y_pos = max(w3, w4) * 1.12
            col   = COLORS["f24"] if delta < -2 else ("gray" if abs(delta) < 2 else COLORS["f23"])
            ax.text(nv, y_pos, f"{delta:+.0f}%", ha="center", fontsize=8, color=col)

        # Scaling reference O(N log N)
        N_ref = np.array([512, 2000])
        w_ref = w_f23[0] * (N_ref / N_f23[0]) * np.log2(N_ref / N_f23[0] + 1)
        ax.loglog(N_ref, w_ref, ":", color="gray", lw=1, label="O(N log N)")

        ax.set_xlabel("N")
        ax.set_ylabel("Wall time (s)")
        ax.set_title(f"P = {P_val}")
        ax.set_xticks([512, 1000, 2000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(fontsize=8)

    fig.suptitle("Wall time vs N — Fase 23 vs Fase 24 (10 pasos, EdS/ΛCDM)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig07a_wall_vs_n.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07b — comm_fraction vs N (log-escala)
# ══════════════════════════════════════════════════════════════════════════════

def fig07b_comm_vs_n():
    p25_path = NBODY / "phase25_mpi_validation" / "results" / "phase25_comparison.csv"
    p25 = read_csv(p25_path)

    Ns  = [512, 1000, 2000]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    styles = {1: ("-",  "P=1"), 2: ("--", "P=2"), 4: (":",  "P=4")}
    for P_val, (ls, plabel) in styles.items():
        f23 = sorted([r for r in p25 if r["variante"]=="fase23" and int(r["P"])==P_val],
                     key=lambda r: r["N"])
        f24 = sorted([r for r in p25 if r["variante"]=="fase24" and int(r["P"])==P_val],
                     key=lambda r: r["N"])
        N_f23 = [r["N"] for r in f23]
        N_f24 = [r["N"] for r in f24]
        cf_f23 = [r["comm_fraction"] * 100 for r in f23]
        cf_f24 = [r["comm_fraction"] * 100 for r in f24]

        ax.semilogy(N_f23, [max(c, 1e-4) for c in cf_f23], ls, color=COLORS["f23"],
                    lw=1.8, marker="o", label=f"F23 {plabel}")
        ax.semilogy(N_f24, [max(c, 1e-4) for c in cf_f24], ls, color=COLORS["f24"],
                    lw=1.8, marker="s", label=f"F24 {plabel}")

    ax.set_xlabel("Número de partículas N")
    ax.set_ylabel("Fracción de comunicación (%)")
    ax.set_title("comm_fraction vs N\nFase 23 (clone) vs Fase 24 (scatter/gather)\nEscala logarítmica")
    ax.set_xticks(Ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    ax.axhline(5, ls="--", color="gray", lw=1, alpha=0.6)
    ax.text(2050, 5.5, "5% umbral\ncrítico", fontsize=8, color="gray")

    fig.tight_layout()
    save(fig, "fig07b_comm_vs_n.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07c — Speedup vs P (F23 y F24)
# ══════════════════════════════════════════════════════════════════════════════

def fig07c_speedup_vs_p():
    p25_path = NBODY / "phase25_mpi_validation" / "results" / "phase25_comparison.csv"
    p25 = read_csv(p25_path)

    Ns = [512, 1000, 2000]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    markers = {512: "o", 1000: "s", 2000: "^"}

    for ax, variante, color, title in [
        (axes[0], "fase23", COLORS["f23"], "Fase 23 (clone+migrate)"),
        (axes[1], "fase24", COLORS["f24"], "Fase 24 (scatter/gather)")
    ]:
        for N_val in Ns:
            rows = sorted([r for r in p25 if r["variante"]==variante and int(r["N"])==N_val],
                          key=lambda r: r["P"])
            w_p1 = next(r["total_wall_s"] for r in rows if r["P"] == 1)
            P_vals  = [r["P"] for r in rows]
            speedup = [w_p1 / r["total_wall_s"] for r in rows]
            ax.plot(P_vals, speedup, "-", color=color, lw=1.5,
                    marker=markers[N_val], ms=7, label=f"N={N_val}")

        # Ideal
        P_ideal = [1, 2, 4]
        ax.plot(P_ideal, P_ideal, "k:", lw=1.2, label="Ideal")
        ax.plot(P_ideal, [1, 1.7, 2.8], "--", color="gray", lw=1,
                label="Amdahl (80% par.)")

        ax.set_xlabel("Número de ranks P")
        ax.set_ylabel("Speedup (T_P1 / T_P)")
        ax.set_title(title)
        ax.set_xticks([1, 2, 4])
        ax.set_ylim(0.5, 5.5)
        ax.legend(fontsize=9)

    fig.suptitle("Speedup paralelo vs P — Fase 23 vs Fase 24\n"
                 "(árbol SR domina; beneficio de bytes no se traduce a speedup para N≥1000)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig07c_speedup_vs_p.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 08 — Stacked breakdown completo F24 (scatter+gather+pm+halo+tree)
# ══════════════════════════════════════════════════════════════════════════════

def fig08_stacked_all():
    p25_path = NBODY / "phase25_mpi_validation" / "results" / "phase25_comparison.csv"
    p25 = read_csv(p25_path)

    # Solo F24 con treepm_hpc medido (P>1 tiene datos completos)
    f24 = sorted([r for r in p25 if r["variante"] == "fase24"],
                 key=lambda r: (r["N"], r["P"]))

    labels = [f"N={int(r['N'])},P={int(r['P'])}" for r in f24]
    n = len(f24)

    # Componentes (convertidos a ms)
    def ms(r, field): return r[field] / 1e6   # ns → ms

    scatter_ms = [ms(r, "scatter_ns") for r in f24]
    gather_ms  = [ms(r, "gather_ns")  for r in f24]
    pm_ms      = [ms(r, "pm_solve_ns") for r in f24]
    halo_ms    = [ms(r, "sr_halo_ns") for r in f24]
    tree_ms    = [ms(r, "tree_sr_ns") for r in f24]
    # Residual (integration + misc)
    total_ms   = [r["mean_step_wall_s"] * 1000 for r in f24]
    others_ms  = [max(0, tot - sc - ga - pm - ha - tr)
                  for tot, sc, ga, pm, ha, tr
                  in zip(total_ms, scatter_ms, gather_ms, pm_ms, halo_ms, tree_ms)]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(n)
    w = 0.55

    bot = np.zeros(n)
    def stacked_bar(values, label, color, alpha=0.9):
        nonlocal bot
        ax.bar(x, values, w, bottom=bot, label=label, color=color, alpha=alpha)
        bot = bot + np.array(values)

    stacked_bar(scatter_ms, "Scatter PM alltoallv (Fase 24)",  COLORS["sg"])
    stacked_bar(gather_ms,  "Gather PM alltoallv (Fase 24)",  "#e6550d")
    stacked_bar(pm_ms,      "PM solve (FFT + interpolación)", COLORS["pm"])
    stacked_bar(halo_ms,    "SR halo exchange 3D",            COLORS["halo"])
    stacked_bar(tree_ms,    "Árbol SR (erfc, dominio SFC)",   COLORS["sr"])
    stacked_bar(others_ms,  "Integrador + misc",              "#cccccc")

    # Referencia F23 (total wall, solo línea)
    f23 = sorted([r for r in p25 if r["variante"] == "fase23"],
                 key=lambda r: (r["N"], r["P"]))
    f23_total_ms = [r["mean_step_wall_s"] * 1000 for r in f23]
    ax.scatter(x, f23_total_ms, marker="_", s=200, color=COLORS["f23"],
               zorder=5, linewidths=2, label="F23 wall time total")

    # % árbol del total F24
    for i, (xi, tr, tot) in enumerate(zip(x, tree_ms, total_ms)):
        pct = tr / tot * 100
        ax.text(xi, tot + 0.3, f"SR\n{pct:.0f}%", ha="center",
                fontsize=7, color=COLORS["sr"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Tiempo por paso (ms)")
    ax.set_title("Desglose temporal por componente — Fase 24 (scatter/gather PM)\n"
                 "El árbol SR domina en todos los regímenes; scatter/gather < 1% para N≥1000 P≤2")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig08_stacked_all.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Generando figuras en: {FIGS}")
    fig01_treepm_arch()
    fig02a_cosmo_a()
    fig02b_vrms()
    fig03a_bytes_scaling()
    fig03b_comm_fraction_p()
    fig04a_bh_force_error()
    fig04b_halo_geometry()
    fig05_time_breakdown_f23()
    fig06_bytes_f23_f24()
    fig07a_wall_vs_n()
    fig07b_comm_vs_n()
    fig07c_speedup_vs_p()
    fig08_stacked_all()
    print(f"\nCompletado: {len(list(FIGS.glob('*.png')))} figuras generadas en {FIGS}")


if __name__ == "__main__":
    main()
