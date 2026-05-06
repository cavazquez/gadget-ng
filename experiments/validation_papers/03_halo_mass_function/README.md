# Experimento 03 — Halo mass function (FoF)

Validar la cadena **simulación → snapshot → FoF → histograma de masas** frente a expectativas cualitativas (forma tipo Press–Schechter / Sheth–Tormen). Este experimento **no** pretende reproducir tablas publicadas sin un estudio de **convergencia en resolución** explícito (véase literatura abajo).

## Parámetros por defecto (`config_simd.toml` / `config_cuda.toml`)

| Cantidad | Valor típico | Notas |
|----------|--------------|--------|
| Caja | \(L = 100\,\mathrm{Mpc}/h\) | Lado de la caja periódica en metadatos del snapshot |
| Partículas | \(N = 128^3 = 2\,097\,152\) | DM único |
| Separación media | \(\bar{l} = (L^3/N)^{1/3} \approx 0{,}781\,\mathrm{Mpc}/h\) | Coincide con el paso de malla \(L/128\) del IC en rejilla |
| FoF (estándar) | `b = 0.2` | Longitud de enlace \( \ell = b\,\bar{l} \approx 0{,}156\,\mathrm{Mpc}/h\) |
| Umbral catálogo | `--min-particles 20` | Post-FoF; el ajuste NFW usa `--nfw-min-part` aparte |

El script `run_and_plot.py` llama a:

```text
gadget-ng analyze --fof-b 0.2 --min-particles 20 --nfw-min-part 50 ...
```

## Por qué \(b=0.2\) puede dar **cero halos** en un snapshot

FoF enlaza pares con separación **estrictamente menor** que \(\ell = b\,\bar{l}\). En una distribución **casi de rejilla**, los primeros vecinos están a distancia \(\sim \bar{l}\), mientras que \(\ell \approx 0{,}2\,\bar{l}\) es **mucho más corta**: no hay aristas en el grafo FoF y cada partícula queda aislada → **ningún grupo con \(N \geq 8\)–\(20\)**.

Eso **no** arregla subiendo \(b\) “como dial” para comparar con una HMF de referencia a \(b=0{,}2\): la comunidad mantiene \(b\) y mejora **resolución** (más partículas, caja más pequeña o volúmenes anidados) hasta que el interior de los halos tiene **muchas** separaciones \(\ll 0{,}2\,\bar{l}\).

**Diagnóstico permitido:** ejecutar `analyze` con \(b \sim 1\) solo para comprobar que el finder y el I/O del snapshot funcionan (aparecen miles de grupos si la malla encadena). Eso **no** sustituye una HMF “tipo paper” a \(b=0{,}2\).

## Qué hacen habitualmente otros estudios (referencias)

- **FoF clásico:** longitud de enlace como fracción de la separación media; en cosmología suele citarse **\(b = 0{,}2\)** (tradición tipo Davis et al.; implementaciones modernas, p. ej. [nbodykit FoF](https://nbodykit.readthedocs.io/en/latest/results/algorithms/fof.html)).
- **HMF de precisión:** simulaciones grandes, muchas realizaciones y control de sesgos; ejemplo paradigmático: [Warren et al. (2006), ApJ 646, 881](https://ui.adsabs.harvard.edu/abs/2006ApJ...646..881W).
- **Grandes suites:** p. ej. [Millennium-II — documentación MPA](https://wwwmpa.mpa-garching.mpg.de/galform/millennium-II/documentation.html): FoF con \(b=0{,}2\), umbrales de partículas por halo y **resolución en masa** muy superior a un único cubo \(128^3\) en \(100\,\mathrm{Mpc}/h\).
- **Dependencia de FoF con la resolución** (masas, umbrales efectivos): p. ej. [Muldrew et al. (2011), ApJS 195, 4](https://ui.adsabs.harvard.edu/abs/2011ApJS..195....4M/abstract).

## Cómo ejecutar

```bash
cd experiments/validation_papers/03_halo_mass_function
python3 run_and_plot.py --simd   # o --cuda
```

Salidas generadas (`out_simd/`, `out_cuda/`, plots) están en `.gitignore` del repositorio; conservar solo configs y scripts en git salvo que se force add de figuras.

## Lecturas recomendadas antes de afirmar “validación” frente a literatura

1. Fijar **\(b\)** y **\(N_\mathrm{min}\)** en el texto del resultado.
2. Acotar el rango de masa donde la HMF está **convergida** al cambiar \(N\) o \(L\) (tests de convergencia).
3. Si el objetivo es masas virial comparables a observaciones/SO, valorar finders **SO** (\(M_{200}\), etc.) además de FoF, según el paper de referencia.
