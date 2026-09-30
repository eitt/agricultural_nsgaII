# Agricultural production planning: optimization core

This annex contains the mathematical model, optimization algorithms, price forecasting, frozen benchmark instances, and execution instructions for the accompanying manuscript, without figure, table, manuscript, or live data-download generators.

## Scope

The four methods are SCIP epsilon-constraint optimization, nondominated sorting genetic algorithm II (NSGA-II), NSGA-II with full sequential least squares programming (SLSQP) refinement, and surrogate-guided hybrid NSGA-II (H-NSGA-II). The surrogate is an Extra Trees regressor that ranks refinement gains; it does not replace objective evaluation. Optimization and forecasting algorithm files are copied byte-for-byte from the source project; the command-line interface, frozen-instance loader, and forecast output locations are adapted for this annex.

Eight synthetic, data-informed instances are included, but only I01–I04 were complete enough for the reported comparisons. I05–I08 are additional designed benchmarks, not evidence of completed experimental results. Consult `instances/catalog.csv`, `MATHEMATICAL_MODEL.md`, and `provenance/release_manifest.json` for dimensions, formulation, and file-level provenance.

## Requirements and installation

Use Python 3.12.3 to match the interpreter recorded for the experiments. The package declares Python >=3.10, but other Python versions have not been tested for this annex. No graphics processor, browser, LaTeX installation, or network data service is required to run the methods on bundled inputs.

On Windows PowerShell, from the extracted package root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m agriopt --config configs/smoke.yaml --run-id smoke
```

On macOS or Linux, replace the first two interpreter forms with `python3.12` and `.venv/bin/python`, respectively; these platforms were not tested for this release. Installation requires access to a package registry unless dependencies are already available locally. If a compatible PySCIPOpt installation cannot be obtained, install `requirements.txt` and `pytest`, and run the stochastic smoke configuration; exact optimization is then unavailable.

`requirements-tested.txt` pins the environment used to validate this annex, not an unrecorded historical dependency lock from the original experimental runs. The original run manifests record Python and input hashes but do not record every dependency version, so bitwise reproduction of published fronts is not guaranteed. Solver versions, stopping conditions, parallelism, and numerical libraries can change results and runtime.

## Run configurations

```powershell
# Fast functional check: I01, three evolutionary methods, one seed.
.\.venv\Scripts\python.exe -m agriopt --config configs/smoke.yaml --run-id smoke
# Published controls: four complete instances, ten seeds, four methods.
.\.venv\Scripts\python.exe -m agriopt --config configs/publication_core.yaml --run-id replication
# One exact benchmark under its published solver limits.
.\.venv\Scripts\python.exe -m agriopt --config configs/publication_core.yaml --run-id exact_i01 --instances I01_micro --methods exact
```

The publication configuration uses population 80, 150 generations, seeds 2026–2035, a 25% selective-refinement budget, five warm-up generations, and at least 30 training examples. Exact optimization uses eleven revenue thresholds, four configured threads, relative gap 1e-4, and per-solve limits of 300, 600, 1200, and 1800 seconds for I01–I04. A strong-Pareto tie break can invoke additional solves, and time-limited incumbents must not be described as certified Pareto-optimal solutions. Replication can require many hours; the smoke configuration is not a substitute for the published protocol.

Outputs are numerical CSV fronts, decoded event lists, refinement logs, configuration snapshots, and run metadata under `results/<run-id>/`. An existing completed run is resumed rather than overwritten; choose a new run identifier when changing settings. The exact method is executed only on the configuration's `exact_instances`; this restriction is deliberate.

## Frozen instances and inputs

Each `instances/<instance>/` directory contains `arrays.npz` and `metadata.json`, representing the full `ProblemData` object without Python pickle. Arrays include plot area, farm assignment, yields, sowing calendars, eligibility, expected prices, and covariance. Metadata provides crop names, municipalities, units, and rest parameters.

```python
from agriopt.frozen_instances import load_frozen_instance
from agriopt.nsga2 import run_nsga2
data = load_frozen_instance('instances/I01_micro')
result = run_nsga2(data, population_size=10, generations=2, seed=2026)
```

The runner reconstructs the same objects from bundled inputs using the unchanged instance builder; tests compare this reconstruction with every frozen instance. Prices and yields retain their original input units, while objectives are computed in million Colombian pesos and squared million Colombian pesos. Publication conversion to US dollars is a reporting operation and is intentionally excluded.

The market-capacity extension is inactive (`demand: null`), matching the original runs' lack of a demand file. Optimization can use the original frozen forecasts or recalculated forecasts; live data acquisition is excluded, while input-source attribution and historical provenance are preserved under `provenance/`.

## Recalculate price forecasts

```powershell
.\.venv\Scripts\python.exe -m agriopt --forecast --config configs/forecast.yaml
# Then optimize using the recalculated price forecasts:
.\.venv\Scripts\python.exe -m agriopt --config configs/recomputed_core.yaml --run-id recomputed
```

The workflow compares autoregressive integrated moving average (ARIMA), Gaussian process (GP), and ARIMA with a GP residual correction (ARIMA_GP) on a terminal 52-week holdout. Product-level selection minimizes symmetric mean absolute percentage error (sMAPE), and the selected model is refitted to all historical observations before generating the 104-week forecast.

ARIMA searches p and q in {0,1} and d in {0,1}, excluding the all-zero model, and selects its order by Akaike information criterion (AIC). The direct GP uses log-price lags 1, 2, 4, and 8, annual sine/cosine terms, and a time trend; the residual GP uses lags 1, 2, and 4. Its covariance combines a scaled Matérn kernel with white noise, and kernel optimization is disabled in the preserved implementation. Insufficient residual history falls back to no residual correction during holdout evaluation.

The forecast pipeline fills missing values by interpolation with limit four followed by forward and backward filling; this original rule is preserved and should not be described as a strict four-week-only imputation policy. Evaluation also reports mean absolute error (MAE), root mean square error (RMSE), and mean absolute scaled error (MASE), without changing the selection criterion.

Forecast CSVs, validation metrics, model choices, and provenance are written to `outputs/forecast/`, leaving frozen inputs and serialized instances unchanged. The predictive-scale output is expressed on the fitted log/residual scale and is not a calibrated price prediction interval. Recalculation may differ from archived forecasts because historical dependency versions were not fully recorded; frozen configurations remain the reference for the published experiment.

## Validation and limits

Run `python -m pytest` from the package root. Checks cover frozen-instance identity, published controls, shared initialization, evolutionary feasibility, conditional refinement, quality metrics, and exact-solver objective consistency. `verify_package.py` checks all packaged file hashes against `SHA256SUMS.json`. Validation does not rerun the entire ten-seed experiment or establish that larger time-limited fronts are globally optimal.

See `VALIDATION.md` for the checks actually executed and their outcomes. The release is a snapshot of the current source tree, including uncommitted source changes; its manifest distinguishes the Git reference from exact file hashes instead of treating the commit identifier as a complete description.

## Attribution and deposit

The original MIT license is retained for the software; it does not establish ownership or unrestricted licensing of third-party datasets. Preserve input-source attribution when redistributing or citing these data. `CITATION.cff` identifies the software package without inventing a DOI, publication status, or repository URL. Attach the ZIP to Zotero, or upload it as a software deposit to Zenodo after deciding the repository metadata and data redistribution terms; this package does not perform either upload.
