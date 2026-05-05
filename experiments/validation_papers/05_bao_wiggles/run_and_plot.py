import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_mock_class_tk(path):
    """
    Genera un archivo tabular T(k) de CLASS falso pero físicamente aproximado,
    basado en EH98 con un oscilador BAO artificial para que funcione de "mock".
    En una validación real, simplemente corres CLASS y pones el output acá.
    """
    k = np.logspace(-4, 1, 1000)
    
    # Parámetros cosmológicos de juguete
    om = 0.315
    ob = 0.049
    h = 0.674
    Gamma = om * h
    
    q = k / (Gamma * h) # aprox k_eq
    
    # L0 aprox (EH98 no-wiggle)
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q)
    T_nw = L0 / (L0 + C0 * q**2)
    
    # Wiggles (Aproximación empírica r_s ~ 105 Mpc/h)
    r_s = 105.0 
    damping = np.exp(-(k * 10.0)**2) # Damping no lineal
    wiggles = 1.0 + 0.05 * np.sin(k * r_s) * damping
    
    T_wiggle = T_nw * wiggles
    
    with open(path, 'w') as f:
        f.write("# k [h/Mpc]    T(k)\n")
        for k_val, t_val in zip(k, T_wiggle):
            f.write(f"{k_val:.6e}    {t_val:.6e}\n")

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
    
    # Use specific config if exists, else fallback to config.toml
    config_name = f'config{suffix}.toml'
    config_path = os.path.join(base_dir, config_name)
    if not os.path.exists(config_path):
        config_path = os.path.join(base_dir, 'config.toml')
        
    out_dir = os.path.join(base_dir, f'out{suffix}')
    tk_mock = os.path.join(base_dir, 'class_tk_mock.dat')
    
    print("=== 0. Generando T(k) Mock con Wiggles ===")
    generate_mock_class_tk(tk_mock)
    
    print(f"=== 1. Ejecutando Simulacion (Modo {mode.upper()}, PM N=128^3, L=1000 Mpc/h) ===")
    subprocess.run(['rm', '-rf', out_dir], check=False)
    
    cargo_cmd = ['cargo', 'run', '--release']
    if features:
        cargo_cmd += ['--features', ','.join(features)]
    cargo_cmd += ['--', 'stepping', '--config', config_path, '--out', out_dir]
    
    subprocess.run(cargo_cmd, cwd=ws_root, check=True)

    print("\n=== 2. Calculando Función de Correlación xi(r) ===")
    snap_final_dir = os.path.join(out_dir, 'snapshot_final')
    if not os.path.exists(snap_final_dir):
        import glob
        snaps = sorted(glob.glob(os.path.join(out_dir, 'frames', 'snap_*')))
        snap_final_dir = snaps[-1]
        
    results_json = os.path.join(out_dir, 'results.json')
    subprocess.run([
        'cargo', 'run', '--release', '--', 
        'analyze', '--snapshot', snap_final_dir, '--xi-bins', '50', '--xi-rmax', '160.0', '--out', results_json
    ], cwd=ws_root, check=True)
    
    print("\n=== 3. Graficando Pico Acústico (BAO) ===")
    with open(results_json, 'r') as f:
        data = json.load(f)
        
    xi_data = data.get("xi_r", [])
    if not xi_data:
        print("No se encontró la función de correlación en results.json.")
        return
        
    r = np.array([b["r"] for b in xi_data])
    xi = np.array([b["xi"] for b in xi_data])
    
    mask = r > 20.0
    r_plot = r[mask]
    xi_plot = xi[mask]
    
    plt.figure(figsize=(8, 6))
    
    # Multiplicamos por r^2 para exagerar y revelar el pico BAO
    plt.plot(r_plot, (r_plot**2) * xi_plot, 'o-', color='crimson', markersize=6, linewidth=2, label=f'Gadget-NG {mode.upper()} (Mock BAO ICs)')
    
    # Línea vertical para guiar el ojo
    plt.axvline(x=105.0, color='gray', linestyle='--', alpha=0.7, label=r'Escala Acústica Teórica $r_s \approx 105 \ h^{-1}\mathrm{Mpc}$')

    plt.xlabel(r'Separación $r \ [h^{-1}\mathrm{Mpc}]$', fontsize=14)
    plt.ylabel(r'$r^2 \xi(r)$', fontsize=14)
    plt.title(f'Detección del Pico BAO (Eisenstein et al. 2005, {mode.upper()})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(base_dir, f'bao_wiggles{suffix}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Listo! Grafico guardado en: {plot_path}")

if __name__ == "__main__":
    main()
