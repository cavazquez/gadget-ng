import subprocess
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob

def generate_pancake_ic(path, n_part=128, box_size=100.0, a_init=0.02, a_collapse=0.2):
    """
    Genera un snapshot inicial (JSONL) para el test de Zel'dovich Pancake.
    Posiciones: x = q - (a/a_c) * (1/k) * sin(k*q)
    Velocidades: v = a_dot * dx/da * da = ...
    En gadget-ng las velocidades son momentum canonico p = a^2 * x_dot.
    """
    q = np.linspace(0, box_size, n_part, endpoint=False)
    k = 2 * np.pi / box_size
    
    # Amplitud de desplazamiento en a_init
    amplitude = (a_init / a_collapse) * (1.0 / k)
    
    x = q - amplitude * np.sin(k * q)
    # Wrap periodic
    x = np.mod(x, box_size)
    
    # Velocidad (peculiar comovil x_dot):
    # x = q - (a/a_c) * (1/k) * sin(kq)
    # x_dot = - (a_dot / a_c) * (1/k) * sin(kq)
    # p = a^2 * x_dot = - (a^2 * a_dot / a_c) * (1/k) * sin(kq)
    # H = a_dot / a. Entonces a_dot = a * H.
    # p = - (a^3 * H / a_c) * (1/k) * sin(kq)
    
    # Simplificamos asumiendo EdS para el IC o simplemente una amplitud razonable.
    # En gadget-ng comovil, p tiene unidades de [L/T].
    p_amp = (a_init**2) * (-1.0 / a_collapse) * (1.0 / k) # Aproximacion
    p_x = p_amp * np.sin(k * q)
    
    particles = []
    for i in range(n_part):
        particles.append({
            "global_id": i,
            "mass": 1.0,
            "px": x[i],
            "py": box_size * 0.5,
            "pz": box_size * 0.5,
            "vx": p_x[i],
            "vy": 0.0,
            "vz": 0.0,
            "internal_energy": 0.0,
            "smoothing_length": 0.0,
            "ptype": "DarkMatter"
        })
    
    # Snapshot Meta (gadget-ng format)
    meta = {
        "schema_version": 1,
        "provenance": {
            "crate_version": "0.1.0",
            "git_commit": None,
            "build_profile": "release",
            "enabled_features": [],
            "command_line": ["manual-ic-gen"],
            "config_hash": "none"
        },
        "particle_count": n_part,
        "time": a_init,
        "redshift": 1.0/a_init - 1.0,
        "box_size": box_size
    }
    
    # Escribir JSONL
    with open(os.path.join(path, "particles.jsonl"), 'w') as f:
        for p in particles:
            f.write(json.dumps(p) + "\n")
            
    with open(os.path.join(path, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--simd', action='store_true')
    args = parser.parse_args()

    mode = "cpu"
    features = []
    suffix = ""
    if args.cuda:
        mode = "cuda"
        features = ["cuda"]
        suffix = "_cuda"
    elif args.simd:
        mode = "simd"
        features = ["simd"]
        suffix = "_simd"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.abspath(os.path.join(base_dir, '../../..'))
    out_dir = os.path.join(base_dir, f'out{suffix}')
    ic_dir = os.path.join(base_dir, 'ic')
    
    if not os.path.exists(ic_dir):
        os.makedirs(ic_dir)
        
    print("=== 1. Generando ICs de Pancake (1D) ===")
    box_size = 100.0
    a_init = 0.02
    a_collapse = 0.2 # El colapso (caustica) debe ocurrir en a = 0.2
    generate_pancake_ic(ic_dir, n_part=512, box_size=box_size, a_init=a_init, a_collapse=a_collapse)

    print(f"\n=== 2. Ejecutando Simulacion (Modo {mode.upper()}, a_init=0.02 -> a_final=0.5) ===")
    # Use specific config if exists, else fallback to config.toml.
    # Compat: si no existe config.toml, usar el nombre legacy config_pancake.toml.
    config_name = f'config{suffix}.toml'
    config_path = os.path.join(base_dir, config_name)
    if not os.path.exists(config_path):
        config_path = os.path.join(base_dir, 'config.toml')
    if not os.path.exists(config_path):
        config_path = os.path.join(base_dir, 'config_pancake.toml')

    subprocess.run(['rm', '-rf', out_dir], check=False)
    
    cargo_cmd = ['cargo', 'run', '--release']
    if features:
        cargo_cmd += ['--features', ','.join(features)]
    cargo_cmd += ['--', 'stepping', '--config', config_path, '--out', out_dir]
    
    subprocess.run(cargo_cmd, cwd=ws_root, check=True)

    print("\n=== 3. Graficando Espacio de Fases (x, vx) ===")
    snaps = sorted(glob.glob(os.path.join(out_dir, 'frames', 'snap_*')))
    
    # Elegimos 3 estados: inicial, colapso (a~0.2), post-colapso (a~0.4)
    indices = [0, len(snaps)//2, -1]
    
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices):
        snap_path = snaps[idx]
        part_file = os.path.join(snap_path, 'particles.jsonl')
        meta_file = os.path.join(snap_path, 'meta.json')
        
        with open(meta_file, 'r') as f:
            h = json.load(f)
            a = h['time']
            
        x = []
        vx = []
        with open(part_file, 'r') as f:
            for line in f:
                p = json.loads(line)
                x.append(p['px'])
                vx.append(p['vx'])
        
        plt.subplot(1, 3, i+1)
        plt.scatter(x, vx, s=1, color='blue', alpha=0.5)
        plt.title(f'$a = {a:.2f}$ ({mode.upper()})')
        plt.xlabel('x')
        if i == 0: plt.ylabel('p_x')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(base_dir, f'zeldovich_pancake{suffix}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Listo! Grafico guardado en: {plot_path}")

if __name__ == "__main__":
    main()
