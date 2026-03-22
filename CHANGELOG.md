# Changelog

## v0.3.2 (2026-03-22) [unreleased]
- Add Databricks benchmark: cure model vs Cox PH vs KM for lapse prediction
- fix: use plain string license field for universal setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)
- Add benchmark: WeibullMixtureCure vs Kaplan-Meier for long-run retention

## v0.3.2 (2026-03-21)
- docs: replace pip install with uv add in README
- Make matplotlib an optional dependency (v0.3.2)
- Add community CTA to README
- Add MIT license
- fix: ExposureTransformer passes through all user columns, not just hardcoded list
- docs: add convergence note to Performance section from Databricks benchmark run
- fix: QA audit batch 5 — standardise on ncd_years, fix docs (v0.3.1)
- Add PyPI classifiers for financial/insurance audience
- Add actual benchmark results to Performance section of README
- Note Phase-98 P0 fixes in Performance section; flag missing benchmark script
- Fix P0/P1 bugs: CIF CI transform, CLV stub, Gray weights, exposure, risk set
- Add standalone benchmark: cure model vs KM/Cox PH for lapse prediction
- fix: add numpy<2.0 compat shim for np.trapezoid in metrics.py
- docs: add Databricks notebook link
- Add Related Libraries section to README
- fix(docs): correct ExposureTransformer schema and add data to competing risks block

