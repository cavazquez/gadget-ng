#!/usr/bin/env python3
"""
analyze_phase17b.py — Análisis de equivalencia serial vs MPI cosmológico (Fase 17b)

Produce:
  1. Tabla de equivalencia: |v_rms(P=N) - v_rms(P=1)| por paso
  2. Tabla de a(t) vs analítico EdS
  3. Tabla de wall time por P
  4. Figura de evolución de a(t) y v_rms para P=1,2,4

Uso:
    python3 analyze_phase17b.py --results-dir <dir> [--out <fig_dir>]
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

def hubble_analytic(omega_m, omega_lam, h0, a):
    return h0 * math.sqrt(omega_m / a**3 + omega_lam)

# ── Lectura de datos ──────────────────────────────────────────────────────────

def load_diag(path):
    fpath = os.path.join(path, "diagnostics.jsonl")
    if not os.path.exists(fpath):
        return []
    records = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_timings(path):
    fpath = os.path.join(path, "timings.json")
    if not os.path.exists(fpath):
        return {}
    with open(fpath) as f:
        return json.load(f)

def extract_cosmo(records, dt):
    cosmo = [r for r in records if "a" in r]
    steps = [r["step"] for r in cosmo]
    times = [s * dt for s in steps]
    return {
        "step": np.array(steps),
        "t": np.array(times),
        "a": np.array([r["a"] for r in cosmo]),
        "z": np.array([r.get("z", float("nan")) for r in cosmo]),
        "v_rms": np.array([r.get("v_rms", float("nan")) for r in cosmo]),
        "delta_rms": np.array([r.get("delta_rms", float("nan")) for r in cosmo]),
        "hubble": np.array([r.get("hubble", float("nan")) for r in cosmo]),
        "ke": np.array([r.get("kinetic_energy", float("nan")) for r in cosmo]),
    }

# ── Análisis ─────────────────────────────────────────────────────────────────

def analyze_config(name, results_dir, p_values, ax_a, ax_v, out_dir):
    """Analiza una config (eds_N512, lcdm_N1000, etc.) para todos los P."""
    # Parámetros hard-coded según configs TOML.
    params = {
        "eds_N512":   dict(dt=0.005, h0=0.1, a0=1.0, omega_m=1.0, omega_lam=0.0),
        "lcdm_N1000": dict(dt=0.005, h0=0.1, a0=1.0, omega_m=0.3, omega_lam=0.7),
        "eds_N2000":  dict(dt=0.005, h0=0.1, a0=1.0, omega_m=1.0, omega_lam=0.0),
    }.get(name, dict(dt=0.005, h0=0.1, a0=1.0, omega_m=1.0, omega_lam=0.0))
    dt = params["dt"]

    data = {}
    for p in p_values:
        path = os.path.join(results_dir, name, f"P{p}")
        records = load_diag(path)
        if records:
            data[p] = extract_cosmo(records, dt)
        else:
            print(f"  ADVERTENCIA: no hay datos para {name}/P{p}")

    if not data:
        return {}

    # ── Tabla de equivalencia a(t) ───────────────────────────────────────────
    p_ref = min(p_values)
    ref = data.get(p_ref)
    if ref is None:
        return {}

    is_eds = params["omega_m"] == 1.0 and params["omega_lam"] == 0.0

    print(f"\n[{name}] a(t) — P_ref={p_ref}")
    if is_eds:
        a_ana = np.array([eds_a_analytic(params["a0"], params["h0"], t) for t in ref["t"]])
        rel_err_ana = np.abs(ref["a"] - a_ana) / a_ana
        print(f"  a_final(P={p_ref}): {ref['a'][-1]:.8f}")
        print(f"  a_final analítico : {a_ana[-1]:.8f}")
        print(f"  max|Δa/a| vs ana  : {rel_err_ana.max():.2e}  {'✓' if rel_err_ana.max() < 0.01 else '✗'}")

    print(f"  {'P':>4} | {'a_final':>12} | {'|Δa/a| vs P='+str(p_ref):>18} | {'|Δv_rms/v_rms|':>16}")
    print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*18}-+-{'-'*16}")
    for p in p_values:
        d = data.get(p)
        if d is None:
            continue
        a_err = abs(d["a"][-1] - ref["a"][-1]) / max(ref["a"][-1], 1e-15) if p != p_ref else 0.0
        v_err = abs(d["v_rms"][-1] - ref["v_rms"][-1]) / max(ref["v_rms"][-1], 1e-15) if p != p_ref else 0.0
        print(f"  {p:>4} | {d['a'][-1]:>12.8f} | {a_err:>18.4e} | {v_err:>16.4e}")

    # ── Figura ───────────────────────────────────────────────────────────────
    styles = {1: ("b-", 1.5), 2: ("r--", 1.5), 4: ("g:", 2.0)}
    for p, d in data.items():
        style, lw = styles.get(p, ("k-", 1.0))
        ax_a.plot(d["t"], d["a"], style, lw=lw, label=f"{name} P={p}")
        ax_v.plot(d["t"], d["v_rms"], style, lw=lw, label=f"{name} P={p}")

    # a analítico EdS
    if is_eds and ref is not None:
        a_ana = np.array([eds_a_analytic(params["a0"], params["h0"], t) for t in ref["t"]])
        ax_a.plot(ref["t"], a_ana, "k--", lw=1, alpha=0.5, label=f"{name} analítico")

    return {p: data[p] for p in p_values if p in data}


def print_stability_summary(all_data):
    print("\n=== Resumen de estabilidad ===")
    for name, p_data in all_data.items():
        for p, d in p_data.items():
            v_ok = all(np.isfinite(d["v_rms"]))
            a_ok = all(np.isfinite(d["a"]))
            status = "✓ STABLE" if v_ok and a_ok else "✗ UNSTABLE"
            print(f"  {name:20s} P={p}: {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--p-values", default="1 2 4", type=str)
    args = parser.parse_args()

    p_values = [int(p) for p in args.p_values.split()]
    out_dir = args.out or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    configs = ["eds_N512", "lcdm_N1000", "eds_N2000"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_a, ax_v = axes
    ax_a.set_xlabel("t [u.i.]")
    ax_a.set_ylabel("a(t)")
    ax_a.set_title("Factor de Escala — Equivalencia serial vs MPI")
    ax_a.grid(True, alpha=0.3)

    ax_v.set_xlabel("t [u.i.]")
    ax_v.set_ylabel("v_rms [u.i.]")
    ax_v.set_title("v_rms — Equivalencia serial vs MPI")
    ax_v.grid(True, alpha=0.3)

    all_data = {}
    print(f"\n=== Fase 17b — Análisis de equivalencia serial vs MPI ===")
    print(f"    results_dir : {args.results_dir}")
    print(f"    P_values    : {p_values}")

    for name in configs:
        path_p1 = os.path.join(args.results_dir, name, "P1")
        if not os.path.isdir(path_p1):
            continue
        d = analyze_config(name, args.results_dir, p_values, ax_a, ax_v, out_dir)
        if d:
            all_data[name] = d

    ax_a.legend(fontsize=7, ncol=2)
    ax_v.legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Fase 17b — Cosmología SFC+LET Distribuida: Equivalencia Serial↔MPI",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "phase17b_equivalence.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada: {fig_path}")

    print_stability_summary(all_data)
    print("\n=== Análisis completado ===")


if __name__ == "__main__":
    main()
