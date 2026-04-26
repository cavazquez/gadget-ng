# Notebooks interactivos de gadget-ng

Guías paso a paso para usar gadget-ng desde Jupyter.

## Requisitos

```bash
pip install -r requirements.txt
jupyter lab  # o jupyter notebook
```

## Notebooks disponibles

| Notebook | Contenido | Tiempo estimado |
|----------|-----------|-----------------|
| [`01_primera_simulacion.ipynb`](01_primera_simulacion.ipynb) | Compilar el binario, esfera de Plummer, órbita kepleriana, visualización de snapshots | ~15 min |
| [`02_simulacion_cosmologica.ipynb`](02_simulacion_cosmologica.ipynb) | Simulación ΛCDM, condiciones iniciales 1LPT, espectro de potencia P(k), estructura de gran escala | ~20 min |
| [`03_herramientas_analisis.ipynb`](03_herramientas_analisis.ipynb) | FoF, P(k), ξ(r), c(M), diagrama de fase, dashboard de análisis completo | ~15 min |

## Orden recomendado

Sigue los notebooks en orden numérico. Cada uno construye sobre los anteriores.

## Ejecutar desde la raíz del proyecto

```bash
cd gadget-ng
jupyter lab notebooks/
```
