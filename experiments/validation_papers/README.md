# Gadget-NG Scientific Validation Suite

Esta suite de experimentos replica empíricamente 5 papers fundamentales de cosmología computacional. Cada directorio contiene todo lo necesario (`config.toml`, scripts de generación de ICs y scripts de ploteo) para ejecutar la simulación desde cero y reproducir los gráficos finales.

## Requisitos
- Compilador de Rust (`cargo`)
- Entorno Python 3 con: `numpy`, `matplotlib`, `h5py`, `scipy`
- Instalación de dependencias (Ubuntu/Debian): `sudo apt install libhdf5-dev`

## Experimentos

1. **`01_zeldovich_pancake/`**: Colapso 1D y formación de cáusticas. Valida el integrador y conservación de fase.
2. **`02_growth_of_structure/`**: Crecimiento lineal y no lineal del Espectro de Potencia $P(k)$. Valida el solver PM/TreePM.
3. **`03_halo_mass_function/`**: Distribución de masa de halos (Press-Schechter/Sheth-Tormen). Valida FoF y clustering no lineal.
4. **`04_millennium_cosmic_web/`**: Visualización de la red cósmica. Valida contrastes de alta densidad.
5. **`05_bao_wiggles/`**: Oscilaciones Acústicas de Bariones. Valida la importación de `T(k)` tabular y la retención de la señal acústica.

## Modo de Uso
Entra a cualquier directorio y ejecuta su script de automatización:
```bash
cd 02_growth_of_structure
python3 run_and_plot.py
```
El script compilará gadget-ng (si no lo está), correrá la simulación, lanzará las herramientas de análisis y abrirá una ventana/guardará un `.png` con el gráfico final.

### Modos de ejecución

La convención en todos los experimentos es:
- `run_and_plot.py` (único script por experimento)
- `config.toml` (CPU), `config_simd.toml` y `config_cuda.toml`

Ejemplos:
```bash
python3 run_and_plot.py
python3 run_and_plot.py --simd
python3 run_and_plot.py --cuda
```
