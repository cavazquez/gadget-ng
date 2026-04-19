#!/usr/bin/env python3
"""
compare_results.py — Tabla comparativa de métricas para Fase 23.

Uso:
    python3 compare_results.py <dir_fase21> <dir_fase22> <dir_fase23>

Lee timings.json y diagnostics.jsonl de cada directorio de resultados
y produce una tabla comparativa de:
  - wall time (s/paso)
  - halo particles y bytes
  - sr_sync_ns (overhead PM↔SR, solo Fase 23)
  - v_rms, delta_rms (física cosmológica)
  - path_active (confirmación del dominio SR)
"""

import json
import sys
import os

def load_timings(outdir):
    path = os.path.join(outdir, "timings.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def load_last_diag(outdir):
    path = os.path.join(outdir, "diagnostics.jsonl")
    if not os.path.exists(path):
        return {}
    lines = open(path).readlines()
    if not lines:
        return {}
    return json.loads(lines[-1])

def main():
    if len(sys.argv) < 3:
        print("Uso: compare_results.py <dir_fase21> <dir_fase22> [<dir_fase23>]")
        sys.exit(1)

    dirs   = sys.argv[1:]
    labels = ["Fase21 slab-1d", "Fase22 slab-3d", "Fase23 sr-sfc"][:len(dirs)]

    rows = []
    for label, d in zip(labels, dirs):
        t = load_timings(d)
        diag = load_last_diag(d)
        hpc = t.get("hpc", {})

        rows.append({
            "label":          label,
            "wall_s":         t.get("mean_step_wall_s", float("nan")),
            "path_active":    hpc.get("path_active", "?"),
            "halo_parts":     hpc.get("mean_short_range_halo_particles", 0.0),
            "halo_bytes":     hpc.get("mean_short_range_halo_bytes", 0.0),
            "tree_short_s":   hpc.get("mean_tree_short_s", 0.0),
            "pm_long_s":      hpc.get("mean_pm_long_s", 0.0),
            "sr_sync_s":      hpc.get("mean_sr_sync_s", 0.0),
            "sr_domain_n":    hpc.get("mean_sr_domain_particle_count", 0.0),
            "v_rms":          diag.get("v_rms", float("nan")),
            "delta_rms":      diag.get("delta_rms", float("nan")),
            "a":              diag.get("a", float("nan")),
        })

    # ── Tabla principal ───────────────────────────────────────────────────────
    col_w = 20
    sep   = "─" * (col_w * 7 + 6)
    hdr   = f"{'Path':<{col_w}} {'wall s/paso':>12} {'halo N':>10} {'halo KB':>10} "
    hdr  += f"{'tree_sr (s)':>12} {'pm_lr (s)':>10} {'sync (s)':>10}"
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['label']:<{col_w}} {r['wall_s']:>12.4f} "
            f"{r['halo_parts']:>10.1f} {r['halo_bytes']/1024:>10.1f} "
            f"{r['tree_short_s']:>12.4f} {r['pm_long_s']:>10.4f} "
            f"{r['sr_sync_s']:>10.4f}"
        )
    print(sep)
    print()

    # ── Tabla física ──────────────────────────────────────────────────────────
    print("Física (último paso):")
    hdr2 = f"{'Path':<{col_w}} {'a':>8} {'v_rms':>12} {'delta_rms':>12} {'path_active'}"
    print("─" * 80)
    print(hdr2)
    print("─" * 80)
    for r in rows:
        print(
            f"{r['label']:<{col_w}} {r['a']:>8.4f} "
            f"{r['v_rms']:>12.4e} {r['delta_rms']:>12.4e} "
            f"{r['path_active']}"
        )
    print("─" * 80)
    print()

    # ── Overhead relativo Fase 23 vs Fase 22 ─────────────────────────────────
    if len(rows) >= 3:
        r22 = rows[1]
        r23 = rows[2]
        if r22["wall_s"] > 0:
            overhead = (r23["wall_s"] - r22["wall_s"]) / r22["wall_s"] * 100
            print(f"Overhead Fase 23 vs Fase 22: {overhead:+.1f}% wall time/paso")
        if r23["sr_sync_s"] > 0 and r23["wall_s"] > 0:
            sync_frac = r23["sr_sync_s"] / r23["wall_s"] * 100
            print(f"Fracción sincronización PM↔SR: {sync_frac:.1f}% del wall time")
        # Diferencia física v_rms
        if r22["v_rms"] > 0 and r23["v_rms"] > 0:
            dv = abs(r23["v_rms"] - r22["v_rms"]) / r22["v_rms"]
            print(f"Diferencia relativa v_rms (23 vs 22): {dv:.2e}")

    print()
    print("Nota: sr_sync_s es el overhead de la sincronización PM↔SR (clone + 2 migraciones + hashmap).")
    print("      Este costo es cero en paths Fase 21/22 (el dominio PM y SR son el mismo).")
    print("      Confirma que path_active='treepm_sr_sfc_3d' para Fase 23.")

if __name__ == "__main__":
    main()
