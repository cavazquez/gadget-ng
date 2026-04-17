"""snapshot_metrics.py — Librería de métricas para snapshots gadget-ng.

Parsea directorios de snapshots JSONL y calcula métricas físicas:
    KE, PE, E_total, δE_rel, p_total, L_z, Q_virial, r_hm

Uso básico:
    from snapshot_metrics import load_snapshot_dir, compute_metrics, load_timeseries

    # Un solo snapshot:
    particles, t = load_snapshot_dir("runs/snap_000010")
    row = compute_metrics(particles, t, softening=0.05)

    # Serie temporal completa (todos los frames en orden):
    df = load_timeseries("runs/", softening=0.05)
    print(df[["t", "E", "dE_rel", "Q", "r_hm"]])

Dependencias: numpy (obligatorio), pandas (obligatorio).
No requiere dependencias de astrofísica.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd


# ── Estructuras de datos ──────────────────────────────────────────────────────

class Particle:
    """Partícula N-body con posición, velocidad y masa."""

    __slots__ = ("gid", "mass", "x", "y", "z", "vx", "vy", "vz")

    def __init__(self, gid: int, mass: float,
                 x: float, y: float, z: float,
                 vx: float, vy: float, vz: float):
        self.gid = gid
        self.mass = mass
        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz


# ── Carga de snapshots ────────────────────────────────────────────────────────

def load_snapshot_dir(snap_dir: str | Path) -> Tuple[List[Particle], float]:
    """Carga un directorio de snapshot JSONL (meta.json + particles.jsonl).

    Returns
    -------
    particles : List[Particle]
    t : float — tiempo de la simulación
    """
    snap_dir = Path(snap_dir)
    meta_path = snap_dir / "meta.json"
    parts_path = snap_dir / "particles.jsonl"

    if not meta_path.exists():
        raise FileNotFoundError(f"No se encontró meta.json en {snap_dir}")
    if not parts_path.exists():
        raise FileNotFoundError(f"No se encontró particles.jsonl en {snap_dir}")

    with open(meta_path) as f:
        meta = json.load(f)
    t = float(meta.get("time", 0.0))

    particles: List[Particle] = []
    with open(parts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Soporta dos formatos:
            #   - gadget-ng JSONL nativo: campos planos px/py/pz y vx/vy/vz
            #   - formato alternativo con arrays "position" y "velocity"
            if "px" in rec:
                x, y, z = float(rec["px"]), float(rec["py"]), float(rec["pz"])
                vx, vy, vz = float(rec["vx"]), float(rec["vy"]), float(rec["vz"])
            else:
                pos = rec["position"]
                vel = rec["velocity"]
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
            particles.append(Particle(
                gid=rec["global_id"],
                mass=float(rec["mass"]),
                x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz,
            ))

    return particles, t


def iter_snapshot_dirs(runs_dir: str | Path) -> Iterator[Path]:
    """Itera sobre los subdirectorios de frames en orden temporal.

    Busca directorios con el patrón ``snap_NNNNNN`` dentro de ``runs_dir/frames/``
    o directamente en ``runs_dir``. Devuelve los directorios ordenados por nombre.
    """
    runs_dir = Path(runs_dir)

    # Gadget-ng guarda frames en <out_dir>/frames/snap_NNNNNN/
    frames_dir = runs_dir / "frames"
    if frames_dir.exists() and frames_dir.is_dir():
        candidates = sorted(frames_dir.iterdir())
    else:
        candidates = sorted(runs_dir.iterdir())

    for d in candidates:
        if d.is_dir() and (d / "particles.jsonl").exists():
            yield d


# ── Métricas físicas ──────────────────────────────────────────────────────────

def kinetic_energy(particles: List[Particle]) -> float:
    """Energía cinética total: Σ 0.5·mᵢ·|vᵢ|²"""
    ke = 0.0
    for p in particles:
        ke += 0.5 * p.mass * (p.vx**2 + p.vy**2 + p.vz**2)
    return ke


def potential_energy(particles: List[Particle], softening: float = 0.0,
                     G: float = 1.0) -> float:
    """Energía potencial gravitatoria: Σ_{i<j} -G·mᵢ·mⱼ / √(r²+ε²)

    Implementación O(N²) vectorizada con numpy broadcasting (eficiente hasta
    N ~ 5000 con <100 MiB de RAM). Para N más grandes, preferir una suma
    por filas (bloques) o un árbol.
    """
    n = len(particles)
    if n < 2:
        return 0.0
    eps2 = softening * softening
    pos = np.empty((n, 3), dtype=np.float64)
    mass = np.empty(n, dtype=np.float64)
    for i, p in enumerate(particles):
        pos[i, 0] = p.x
        pos[i, 1] = p.y
        pos[i, 2] = p.z
        mass[i] = p.mass
    # diferencias (N,N,3) → r²+ε² (N,N)
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", diff, diff) + eps2
    # enmascara i==j (auto-términos) con +∞ para que 1/r → 0
    np.fill_diagonal(r2, np.inf)
    inv_r = 1.0 / np.sqrt(r2)
    mij = mass[:, None] * mass[None, :]
    # PE total; /2 porque sumamos ambos órdenes
    pe = -0.5 * G * float(np.sum(mij * inv_r))
    return pe


# Alias retrocompatible; muchos scripts previos importan este nombre.
potential_energy_vectorized = potential_energy


def total_momentum(particles: List[Particle]) -> np.ndarray:
    """Momento lineal total: Σ mᵢ·vᵢ  (vector 3D)"""
    px = sum(p.mass * p.vx for p in particles)
    py = sum(p.mass * p.vy for p in particles)
    pz = sum(p.mass * p.vz for p in particles)
    return np.array([px, py, pz])


def angular_momentum_z(particles: List[Particle]) -> float:
    """Componente z del momento angular total: Σ mᵢ·(xᵢ·vyᵢ − yᵢ·vxᵢ)"""
    lz = 0.0
    for p in particles:
        lz += p.mass * (p.x * p.vy - p.y * p.vx)
    return lz


def angular_momentum_vec(particles: List[Particle]) -> np.ndarray:
    """Momento angular total 3D: Σ mᵢ (rᵢ × vᵢ)."""
    lx = ly = lz = 0.0
    for p in particles:
        lx += p.mass * (p.y * p.vz - p.z * p.vy)
        ly += p.mass * (p.z * p.vx - p.x * p.vz)
        lz += p.mass * (p.x * p.vy - p.y * p.vx)
    return np.array([lx, ly, lz])


def center_of_mass(particles: List[Particle]) -> np.ndarray:
    """Centro de masa (vector 3D)."""
    m_tot = sum(p.mass for p in particles)
    cx = sum(p.mass * p.x for p in particles) / m_tot
    cy = sum(p.mass * p.y for p in particles) / m_tot
    cz = sum(p.mass * p.z for p in particles) / m_tot
    return np.array([cx, cy, cz])


def half_mass_radius(particles: List[Particle]) -> float:
    """Radio de media masa: radio que encierra el 50% de la masa total.

    Se calcula respecto al centro de masa actual.
    """
    com = center_of_mass(particles)
    dists = sorted(
        math.sqrt((p.x - com[0])**2 + (p.y - com[1])**2 + (p.z - com[2])**2)
        for p in particles
    )
    m_each = particles[0].mass  # masas iguales asumidas
    m_tot = sum(p.mass for p in particles)
    half = 0.5 * m_tot
    cumulative = 0.0
    for d in dists:
        cumulative += m_each
        if cumulative >= half:
            return d
    return dists[-1]


def virial_ratio(ke: float, pe: float) -> float:
    """Ratio virial Q = -KE/PE.  En equilibrio: Q = 0.5."""
    if abs(pe) < 1e-300:
        return 0.0
    return -ke / pe


# ── Métrica compuesta ─────────────────────────────────────────────────────────

def compute_metrics(particles: List[Particle], t: float,
                    softening: float = 0.0,
                    G: float = 1.0,
                    e0: float | None = None,
                    p0: np.ndarray | None = None,
                    l0: np.ndarray | None = None) -> dict:
    """Calcula todas las métricas para un snapshot dado.

    Parameters
    ----------
    particles : lista de partículas del snapshot
    t : tiempo de la simulación
    softening : longitud de suavizado ε (mismas unidades que posiciones)
    G : constante gravitacional
    e0 : energía total inicial (para calcular dE_rel); si None, se usa E(t)
    p0 : momento lineal inicial (3D); para calcular |Δp| / max(|p0|, |p|)
    l0 : momento angular inicial (3D); para calcular |ΔL| / max(|L0|, |L|)

    Returns
    -------
    dict con claves: t, N, KE, PE, E, dE_rel, px, py, pz, p_norm, dp_abs, dp_rel,
                     Lx, Ly, Lz, L_norm, dL_abs, dL_rel, Q, r_hm,
                     com_x, com_y, com_z
    """
    ke = kinetic_energy(particles)
    pe = potential_energy(particles, softening=softening, G=G)
    e = ke + pe
    p = total_momentum(particles)
    l_vec = angular_momentum_vec(particles)
    com = center_of_mass(particles)
    r_hm = half_mass_radius(particles)
    q = virial_ratio(ke, pe)

    if e0 is None:
        e0 = e
    de_rel = abs(e - e0) / abs(e0) if abs(e0) > 1e-300 else 0.0

    if p0 is None:
        p0 = p
    dp_vec = p - p0
    dp_abs = float(np.linalg.norm(dp_vec))
    # Denominador robusto: usa max(|p0|, |p|) para evitar dividir por ~0 en sistemas
    # con momento total inicial nulo. Si ambos son ~0 reportamos dp_rel = dp_abs.
    dp_denom = max(float(np.linalg.norm(p0)), float(np.linalg.norm(p)))
    dp_rel = dp_abs / dp_denom if dp_denom > 1e-300 else dp_abs

    if l0 is None:
        l0 = l_vec
    dl_vec = l_vec - l0
    dl_abs = float(np.linalg.norm(dl_vec))
    dl_denom = max(float(np.linalg.norm(l0)), float(np.linalg.norm(l_vec)))
    dl_rel = dl_abs / dl_denom if dl_denom > 1e-300 else dl_abs

    return {
        "t": t,
        "N": len(particles),
        "KE": ke,
        "PE": pe,
        "E": e,
        "dE_rel": de_rel,
        "px": p[0],
        "py": p[1],
        "pz": p[2],
        "p_norm": float(np.linalg.norm(p)),
        "dp_abs": dp_abs,
        "dp_rel": dp_rel,
        "Lx": l_vec[0],
        "Ly": l_vec[1],
        "Lz": l_vec[2],
        "L_norm": float(np.linalg.norm(l_vec)),
        "dL_abs": dl_abs,
        "dL_rel": dl_rel,
        "Q": q,
        "r_hm": r_hm,
        "com_x": com[0],
        "com_y": com[1],
        "com_z": com[2],
    }


# ── Serie temporal completa ───────────────────────────────────────────────────

def load_timeseries(runs_dir: str | Path,
                    softening: float = 0.0,
                    G: float = 1.0,
                    verbose: bool = False) -> pd.DataFrame:
    """Carga todos los snapshots de un experimento y calcula la serie temporal.

    Parámetros
    ----------
    runs_dir : directorio raíz del experimento (contiene ``frames/`` o ``snap_*/``)
    softening : suavizado gravitatorio ε
    G : constante gravitacional
    verbose : imprimir progreso

    Returns
    -------
    pd.DataFrame con columnas: t, N, KE, PE, E, dE_rel, px, py, pz, p_norm,
                                Lz, Q, r_hm, com_x, com_y, com_z
    """
    rows = []
    e0 = None
    p0 = None
    l0 = None

    snap_dirs = list(iter_snapshot_dirs(runs_dir))
    if not snap_dirs:
        raise FileNotFoundError(
            f"No se encontraron snapshots en {runs_dir}.\n"
            "Asegúrate de que la simulación generó frames con "
            "[output] snapshot_interval > 0"
        )

    for i, snap_dir in enumerate(snap_dirs):
        if verbose:
            print(f"  cargando snap {i+1}/{len(snap_dirs)}: {snap_dir.name}", end="\r")
        particles, t = load_snapshot_dir(snap_dir)
        row = compute_metrics(
            particles, t,
            softening=softening, G=G,
            e0=e0, p0=p0, l0=l0,
        )
        if e0 is None:
            e0 = row["E"]
            p0 = np.array([row["px"], row["py"], row["pz"]])
            l0 = np.array([row["Lx"], row["Ly"], row["Lz"]])
            row["dE_rel"] = 0.0
            row["dp_abs"] = 0.0
            row["dp_rel"] = 0.0
            row["dL_abs"] = 0.0
            row["dL_rel"] = 0.0
        rows.append(row)

    if verbose:
        print(f"\n  {len(rows)} snapshots cargados.")

    return pd.DataFrame(rows)


# ── Comparación serial vs MPI ─────────────────────────────────────────────────

def compare_serial_mpi(dir_serial: str | Path,
                       dir_mpi: str | Path) -> pd.DataFrame:
    """Compara el estado final de una ejecución serial vs MPI.

    Lee el último snapshot de cada directorio y compara posiciones y velocidades
    partícula a partícula (por global_id).

    Returns
    -------
    pd.DataFrame con columnas: global_id, dx, dy, dz, dvx, dvy, dvz, dr, dv
    donde dr = norma del error posicional y dv = norma del error de velocidad.
    """
    # Obtener el último snapshot de cada ejecución.
    def last_snapshot(runs_dir: str | Path) -> List[Particle]:
        snaps = list(iter_snapshot_dirs(runs_dir))
        if not snaps:
            raise FileNotFoundError(f"Sin snapshots en {runs_dir}")
        particles, _ = load_snapshot_dir(snaps[-1])
        return sorted(particles, key=lambda p: p.gid)

    ps_s = last_snapshot(dir_serial)
    ps_m = last_snapshot(dir_mpi)

    if len(ps_s) != len(ps_m):
        raise ValueError(
            f"Número de partículas diferente: serial={len(ps_s)}, mpi={len(ps_m)}"
        )

    rows = []
    for s, m in zip(ps_s, ps_m):
        if s.gid != m.gid:
            raise ValueError(f"IDs no coinciden: serial gid={s.gid}, mpi gid={m.gid}")
        dx = s.x - m.x
        dy = s.y - m.y
        dz = s.z - m.z
        dvx = s.vx - m.vx
        dvy = s.vy - m.vy
        dvz = s.vz - m.vz
        rows.append({
            "global_id": s.gid,
            "dx": dx, "dy": dy, "dz": dz,
            "dvx": dvx, "dvy": dvy, "dvz": dvz,
            "dr": math.sqrt(dx*dx + dy*dy + dz*dz),
            "dv": math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz),
        })

    return pd.DataFrame(rows)


# ── CLI de conveniencia ───────────────────────────────────────────────────────

def _cli():
    """Uso: python snapshot_metrics.py <runs_dir> [--softening 0.05] [--G 1.0]"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calcula métricas físicas para una serie de snapshots gadget-ng."
    )
    parser.add_argument("runs_dir", help="Directorio con snapshots (frames/snap_*/)")
    parser.add_argument("--softening", type=float, default=0.0,
                        help="Longitud de suavizado ε (default: 0)")
    parser.add_argument("--G", type=float, default=1.0,
                        help="Constante gravitacional (default: 1.0)")
    parser.add_argument("--out", default=None,
                        help="CSV de salida (default: stdout)")
    args = parser.parse_args()

    df = load_timeseries(args.runs_dir, softening=args.softening, G=args.G, verbose=True)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Métricas guardadas en {args.out}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    _cli()
