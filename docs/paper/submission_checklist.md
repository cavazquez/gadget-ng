# JOSS Submission Checklist — gadget-ng

Referencia: [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)

## Requisitos de Software

- [x] **Licencia OSI-approved**: GPL-3.0 (`LICENSE`)
- [x] **Código abierto en repositorio público**: GitHub
- [x] **≥ 2 000 líneas de código de investigación** (excl. dependencias, tests, blancos):
      Estimado: ~18 000 líneas de Rust + ~2 000 Python/scripts
- [x] **API documentada**: `cargo doc --workspace` genera documentación completa
- [x] **Tests automáticos**: `cargo test --workspace` (~180 tests) + `scripts/check_release.sh`
- [x] **Al menos 1 GitHub Action activo**: `.github/workflows/ci.yml`

## Paper (`paper.md`)

- [x] **Summary ≤ 250 palabras** con descripción del problema y audiencia objetivo
- [x] **Statement of Need**: diferencias respecto a GADGET-4, Arepo, OpenGadget3
- [x] **Referencias clave** (≥ 5): `paper.bib` contiene 15 entradas BibTeX
- [x] **Figuras de validación**:
  - [ ] Fig 1 — P(k): `docs/paper/figures/pk_validation.png` (generada por `generate_paper_figures.py`)
  - [ ] Fig 2 — HMF: `docs/paper/figures/hmf_comparison.png`
  - [ ] Fig 3 — Strömgren: `docs/paper/figures/stromgren.png`
- [x] **Afiliaciones de autores** (completar antes de submission)
- [x] **ORCID de autores** (completar antes de submission)

## Antes de Submission

1. **Generar figuras**:
   ```bash
   pip install matplotlib numpy
   python docs/scripts/generate_paper_figures.py
   ```

2. **Compilar paper PDF** (requiere pandoc + LaTeX):
   ```bash
   docker run --rm \
     -v $PWD/docs/paper:/data openjournals/inara \
     -o paper.pdf paper.md
   ```

3. **Zenodo DOI** (placeholder):
   - Crear release en GitHub con tag `v0.1.0`
   - Activar integración Zenodo en el repositorio
   - DOI esperado: `10.5281/zenodo.XXXXXXX`
   - Agregar badge al README: `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)`

4. **Pre-submission inquiry** a JOSS:
   - Editor scope: "Astrophysics" o "Computational Science"
   - URL: https://joss.theoj.org/papers/new

5. **Completar metadatos en `paper.md`**:
   - `authors[0].orcid: "0000-XXXX-XXXX-XXXX"`
   - `authors[0].affiliation: "1"`
   - `affiliations[0].name: "Universidad / Institución"`

## Checklist final antes de enviar

- [ ] Todas las figuras generadas y referenciadas en `paper.md`
- [ ] DOI Zenodo obtenido y agregado a README + paper
- [ ] ORCIDs y afiliaciones completados
- [ ] `paper.md` compila a PDF sin errores
- [ ] CI pasa en rama `main`
- [ ] CHANGELOG actualizado con versión `v0.1.0`
