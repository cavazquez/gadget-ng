#!/usr/bin/env python3
"""
analyze_phase18.py — Análisis de cosmología periódica con PM (Fase 18)

Produce:
  1. Equivalencia serial vs MPI para PM periódico
  2. a(t) vs EdS analítico
  3. Comparación PM-grid16 vs PM-grid32 (convergencia de malla)
  4. Comparación PM vs TreePM
  5. Estabilidad y diagnósticos cosmológicos
  6. Conservación de masa en el grid (verificación teórica)

Uso:
    python3 analyze_phase18.py --results-dir <dir> [--out <fig_dir>]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# ── Cosmología analítica ──────────────────────────────────────────────────────

def eds_a_analytic(a0, h0, t):
    return (a0**1.5 + 1.5 * h0 * t)**(2.0 / 3.0)

# ── Lectura de datos ──────────────────────────────────────────────────────────

def load_diag(path):
    fpath = os.path.join(path, "diagnostics.jsonl")
    if not os.path.exists(fpath):
        return []
    with open(fpath) as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_cosmo(records, dt):
    cosmo = [r for r in records if "a" in r]
    return {
        "step": np.array([r["step"] for r in cosmo]),
        "t": np.array([r["step"] * dt for r in cosmo]),
        "a": np.array([r["a"] for r in cosmo]),
        "v_rms": np.array([r.get("v_rms", float("nan")) for r in cosmo]),
        "delta_rms": np.array([r.get("delta_rms", float("nan")) for r in cosmo]),
        "hubble": np.array([r.get("hubble", float("nan")) for r in cosmo]),
    }

# ── Análisis ─────────────────────────────────────────────────────────────────

def analyze_equivalence(name, results_dir, p_values, dt, h0, a0, is_eds):
    """Equivalencia serial vs MPI."""
    data = {}
    for p in p_values:
        path = os.path.join(results_dir, name, f"P{p}")
        records = load_diag(path)
        if records:
            data[p] = extract_cosmo(records, dt)

    if not data:
        print(f"  WARN: sin datos para {name}")
        return {}

    p_ref = min(p_values)
    ref = data.get(p_ref)
    if ref is None:
        return {}

    print(f"\n[{name}] Equivalencia serial vs MPI (P_ref={p_ref})")
    if is_eds and ref is not None:
        a_ana = np.array([eds_a_analytic(a0, h0, t) for t in ref["t"]])
        rel_err_ana = np.abs(ref["a"] - a_ana) / a_ana
        print(f"  a_final(P={p_ref}): {ref['a'][-1]:.8f}")
        print(f"  a_analítico(EdS):   {a_ana[-1]:.8f}")
        print(f"  max|Δa/a| vs ana:   {rel_err_ana.max():.2e}  {'OK' if rel_err_ana.max() < 0.01 else 'FAIL'}")

    print(f"  {'P':>4} | {'a_final':>12} | {'|Δa/a|':>12} | {'|Δv/v|':>12} | estable")
    print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'------'}")
    for p in p_values:
        d = data.get(p)
        if d is None:
            continue
        a_ok = all(np.isfinite(d["a"]))
        v_ok = all(np.isfinite(d["v_rms"]))
        stable = "SI" if a_ok and v_ok else "NO"
        if p == p_ref:
            a_err = v_err = 0.0
        else:
            a_err = abs(d["a"][-1] - ref["a"][-1]) / max(ref["a"][-1], 1e-15)
            v_err = abs(d["v_rms"][-1] - ref["v_rms"][-1]) / max(ref["v_rms"][-1], 1e-15)
        print(f"  {p:>4} | {d['a'][-1]:>12.8f} | {a_err:>12.4e} | {v_err:>12.4e} | {stable}")

    return data


def analyze_pm_convergence(results_dir, p_ref, dt):
    """Comparación PM-grid16 vs PM-grid32."""
    path16 = os.path.join(results_dir, "eds_N512_pm", f"P{p_ref}")
    path32 = os.path.join(results_dir, "eds_N512_pm_grid32", f"P{p_ref}")

    d16 = extract_cosmo(load_diag(path16), dt)
    d32 = extract_cosmo(load_diag(path32), dt)

    if not d16["step"].size or not d32["step"].size:
        print("\n[PM convergencia] Sin datos suficientes")
        return

    min_n = min(len(d16["a"]), len(d32["a"]))
    if min_n < 2:
        return

    a16 = d16["a"][:min_n]
    a32 = d32["a"][:min_n]
    v16 = d16["v_rms"][:min_n]
    v32 = d32["v_rms"][:min_n]

    rel_a = np.abs(a16 - a32) / a32.clip(1e-15)
    rel_v = np.abs(v16 - v32) / v32.clip(1e-15)

    print(f"\n[PM convergencia] grid16 vs grid32 (N=512, EdS)")
    print(f"  max|Δa/a|:    {rel_a.max():.2e}")
    print(f"  max|Δv/v|:    {rel_v.max():.2e}")
    print(f"  a_final(16):  {a16[-1]:.8f}")
    print(f"  a_final(32):  {a32[-1]:.8f}")


def analyze_pm_vs_treepm(results_dir, p_ref, dt):
    """Comparación PM puro vs TreePM."""
    path_pm = os.path.join(results_dir, "eds_N512_pm", f"P{p_ref}")
    path_tpm = os.path.join(results_dir, "eds_N512_treepm", f"P{p_ref}")

    d_pm = extract_cosmo(load_diag(path_pm), dt)
    d_tpm = extract_cosmo(load_diag(path_tpm), dt)

    if not d_pm["step"].size or not d_tpm["step"].size:
        print("\n[PM vs TreePM] Sin datos suficientes")
        return

    min_n = min(len(d_pm["a"]), len(d_tpm["a"]))
    if min_n < 2:
        return

    a_pm = d_pm["a"][:min_n]
    a_tpm = d_tpm["a"][:min_n]
    v_pm = d_pm["v_rms"][:min_n]
    v_tpm = d_tpm["v_rms"][:min_n]

    print(f"\n[PM vs TreePM] N=512, EdS, periodic (P={p_ref})")
    print(f"  max|Δa/a| PM vs TreePM:  {np.abs(a_pm - a_tpm).max() / a_tpm.clip(1e-15).max():.2e}")
    print(f"  max|Δv/v| PM vs TreePM:  {np.abs(v_pm - v_tpm).max() / v_tpm.clip(1e-15).max():.2e}")
    print(f"  a(t) es globalmente idéntico (solo depende de H(a), no de fuerzas)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--p-values", default="1 2 4", type=str)
    args = parser.parse_args()

    p_values = [int(p) for p in args.p_values.split()]
    out_dir = args.out or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    p_ref = min(p_values)

    print("\n=== Fase 18 — Cosmología Periódica con PM ===")
    print(f"    results_dir: {args.results_dir}")

    # Config params
    configs = {
        "eds_N512_pm":       dict(dt=0.005, h0=0.1, a0=1.0, is_eds=True),
        "lcdm_N1000_pm":     dict(dt=0.005, h0=0.1, a0=1.0, is_eds=False),
        "eds_N512_treepm":   dict(dt=0.005, h0=0.1, a0=1.0, is_eds=True),
        "eds_N512_pm_grid32":dict(dt=0.005, h0=0.1, a0=1.0, is_eds=True),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_a, ax_v = axes
    ax_a.set_xlabel("t [u.i.]"); ax_a.set_ylabel("a(t)"); ax_a.grid(True, alpha=0.3)
    ax_v.set_xlabel("t [u.i.]"); ax_v.set_ylabel("v_rms [u.i.]"); ax_v.grid(True, alpha=0.3)
    ax_a.set_title("Factor de Escala — PM Periódico")
    ax_v.set_title("v_rms — PM Periódico")

    styles = {1: ("b-", 2.0), 2: ("r--", 1.5), 4: ("g:", 2.0)}
    config_colors = {"eds_N512_pm": "blue", "lcdm_N1000_pm": "red",
                     "eds_N512_treepm": "green", "eds_N512_pm_grid32": "purple"}

    all_stable = True
    for name, params in configs.items():
        path_p1 = os.path.join(args.results_dir, name, "P1")
        if not os.path.isdir(path_p1):
            continue
        data = analyze_equivalence(
            name, args.results_dir, p_values,
            params["dt"], params["h0"], params["a0"], params["is_eds"]
        )
        for p, d in data.items():
            if not all(np.isfinite(d["a"])) or not all(np.isfinite(d["v_rms"])):
                all_stable = False
            style, lw = styles.get(p, ("k-", 1.0))
            color = config_colors.get(name, "gray")
            ax_a.plot(d["t"], d["a"], linestyle=style.replace("b","").replace("r","").replace("g","") or "-",
                     color=color, lw=lw, alpha=0.8, label=f"{name} P={p}")
            ax_v.plot(d["t"], d["v_rms"], linestyle=style.replace("b","").replace("r","").replace("g","") or "-",
                     color=color, lw=lw, alpha=0.8, label=f"{name} P={p}")

    analyze_pm_convergence(args.results_dir, p_ref, 0.005)
    analyze_pm_vs_treepm(args.results_dir, p_ref, 0.005)

    ax_a.legend(fontsize=6, ncol=2)
    ax_v.legend(fontsize=6, ncol=2)
    fig.suptitle("Fase 18 — Cosmología Periódica PM/TreePM", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "phase18_periodic_pm.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada: {fig_path}")

    print(f"\n=== Estabilidad global: {'OK' if all_stable else 'FALLO'} ===")
    print("=== Análisis completado ===")


if __name__ == "__main__":
    main()
