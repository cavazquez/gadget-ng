import subprocess
import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.abspath(os.path.join(base_dir, '../../..'))
    config_path = os.path.join(base_dir, 'config.toml')
    out_dir = os.path.join(base_dir, 'out')

    print("=== 1. Ejecutando Simulacion ===")
    # Remove old out dir if exists to prevent stale data
    subprocess.run(['rm', '-rf', out_dir], check=False)
    subprocess.run(['cargo', 'run', '--release', '--', 'stepping', '--config', config_path, '--out', out_dir], cwd=ws_root, check=True)

    snaps = sorted(glob.glob(os.path.join(out_dir, 'frames', 'snap_*')))
    data = []

    print("\n=== 2. Analizando P(k) ===")
    for snap in snaps:
        if not os.path.isdir(snap):
            continue
        print(f" -> {os.path.basename(snap)}")
        
        # Guardaremos los resultados temporales dentro de un results.json general en out_dir
        results_json = os.path.join(out_dir, 'results.json')
        subprocess.run([
            'cargo', 'run', '--release', '--', 
            'analyze', '--snapshot', snap, '--pk-mesh', '128', '--output', results_json
        ], cwd=ws_root, check=True, stdout=subprocess.DEVNULL)
        
        import json
        with open(results_json, 'r') as f:
            res = json.load(f)
            z = res['cosmology']['z']
            a = 1.0 / (1.0 + z)
            
            # Reconstruir array numpy a partir de JSON
            pk_list = res.get('power_spectrum', [])
            k_vals = [item['k'] for item in pk_list]
            pk_vals = [item['pk'] for item in pk_list]
            modes_vals = [item['n_modes'] for item in pk_list]
            
            pk_data = np.core.records.fromarrays(
                [k_vals, pk_vals, modes_vals],
                names='k, pk, n_modes'
            )
            data.append((a, pk_data))

    print("\n=== 3. Generando Grafico ===")
    plt.figure(figsize=(8, 6))
    
    # Ploteamos Delta^2(k) = k^3 P(k) / (2 pi^2)
    for a, pk in data:
        k = pk['k']
        p = pk['pk']
        mask = (p > 0) & (k > 0) & (pk['n_modes'] >= 4)
        k_plot = k[mask]
        p_plot = p[mask]
        delta2 = (k_plot**3 * p_plot) / (2 * np.pi**2)
        
        plt.loglog(k_plot, delta2, marker='o', markersize=4, label=f'$a = {a:.2f}$')

    # Añadir linea teórica de proporcionalidad k^4 en P(k) (k^7 en Delta^2) para low-k
    plt.xlabel(r'$k \ [h/\mathrm{Mpc}]$', fontsize=12)
    plt.ylabel(r'$\Delta^2(k) = \frac{k^3 P(k)}{2\pi^2}$', fontsize=12)
    plt.title('Evolución del Espectro de Potencia (Efstathiou et al. 1985)', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(base_dir, 'growth_of_structure.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Listo! Grafico guardado en: {plot_path}")

if __name__ == "__main__":
    main()
