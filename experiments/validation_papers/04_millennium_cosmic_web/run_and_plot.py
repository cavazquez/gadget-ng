import subprocess
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

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
    
    print(f"=== 1. Ejecutando Simulacion (Modo {mode.upper()}, N=128^3, L=200 Mpc/h) ===")
    subprocess.run(['rm', '-rf', out_dir], check=False)
    
    cargo_cmd = ['cargo', 'run', '--release']
    if features:
        cargo_cmd += ['--features', ','.join(features)]
    cargo_cmd += ['--', 'stepping', '--config', config_path, '--out', out_dir]
    
    subprocess.run(cargo_cmd, cwd=ws_root, check=True)

    print("\n=== 2. Procesando Slice de Densidad (z=0) ===")
    snap_final_dir = os.path.join(out_dir, 'snapshot_final')
    if not os.path.exists(snap_final_dir):
        import glob
        snaps = sorted(glob.glob(os.path.join(out_dir, 'frames', 'snap_*')))
        snap_final_dir = snaps[-1]
        
    snap_file = os.path.join(snap_final_dir, 'snapshot.hdf5')
    with h5py.File(snap_file, 'r') as f:
        pos = f['PartType1']['Coordinates'][:]
        box_size = f['Header'].attrs['BoxSize'][0]
        
    # Seleccionamos un "slice" en el eje Z (grosor de 15 Mpc/h)
    z_center = box_size / 2.0
    slice_thickness = 15.0
    
    mask = (pos[:, 2] > (z_center - slice_thickness/2.0)) & (pos[:, 2] < (z_center + slice_thickness/2.0))
    pos_slice = pos[mask]
    
    print(f"Particulas en el slice: {len(pos_slice)}")
    
    # Construimos un grid 2D (histograma de densidad)
    n_pixels = 1024
    density_grid, xedges, yedges = np.histogram2d(
        pos_slice[:, 0], pos_slice[:, 1], 
        bins=n_pixels, 
        range=[[0, box_size], [0, box_size]]
    )
    
    print("\n=== 3. Renderizando Imagen Estilo Millennium ===")
    plt.figure(figsize=(10, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Aplicamos una escala Logaritmica al mapa de densidad para resaltar los filamentos tenues
    # y usamos el colormap 'inferno' (muy usado en papers para materia oscura)
    # Transponemos el grid porque histogram2d invierte los ejes de la imagen
    im = ax.imshow(
        density_grid.T, 
        origin='lower', 
        extent=[0, box_size, 0, box_size], 
        cmap='inferno', 
        norm=colors.LogNorm(vmin=0.5, vmax=np.max(density_grid)*0.2)
    )
    
    # Detalles estéticos (como en un paper: sin bordes molestos, ticks blancos)
    ax.set_xlabel(r'$X \ [h^{-1}\mathrm{Mpc}]$', fontsize=16, color='white')
    ax.set_ylabel(r'$Y \ [h^{-1}\mathrm{Mpc}]$', fontsize=16, color='white')
    ax.tick_params(colors='white', labelsize=12)
    for spine in ax.spines.values():
        spine.set_color('white')
        
    plt.title(f'Cosmic Web - DM Density Projection ({mode.upper()})', fontsize=20, color='white', pad=20)
    plt.tight_layout()
    
    plot_path = os.path.join(base_dir, f'cosmic_web_millennium{suffix}.png')
    plt.savefig(plot_path, dpi=300, facecolor='black', edgecolor='none')
    print(f"Listo! Grafico guardado en: {plot_path}")

if __name__ == "__main__":
    main()
