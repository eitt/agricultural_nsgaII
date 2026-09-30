# Annex validation

Validation used Python 3.12.3 on Windows with the dependency versions pinned in `requirements-tested.txt`; original optimization and forecasting algorithm files were verified byte-for-byte against the source project.

The complete annex test suite finished successfully: **22 passed**, with no skipped or failed tests, in 64.54 seconds on the validation machine.

The three stochastic methods completed the bundled smoke configuration successfully, using the archived optimization inputs and writing only numerical outputs and execution metadata. Tests compare every field of all eight frozen instances against reconstruction from the bundled productivity, price-history, and forecast files.

The exact-solver test constructs I01, solves a maximum-revenue model with a ten-second validation limit, checks event feasibility, and independently recomputes revenue and risk from production. This is a functional check, not a replication of the full eleven-threshold exact experiment or a certification of every archived Pareto point.

The complete forecasting command was executed for all 21 products with a 52-week holdout and a 104-week forecast horizon, producing 63 candidate evaluations and 21 model selections. It selected ARIMA for 12 products and GP for nine products; no product selected ARIMA_GP in this run, although the hybrid candidate was evaluated.

All 104 × 21 forecast values were finite and positive, and the recomputed price matrix matched the bundled frozen forecast within relative and absolute tolerances of 1e-10. This observed numerical match does not guarantee bitwise reproduction under different Python, solver, or library versions.

The package hash manifest verifies bundled file integrity, and the archive integrity check verifies that the ZIP contains the same clean file inventory. Generated smoke and forecast outputs are deliberately stored outside the release folder; no figure, table, manuscript, local virtual environment, cache, or full experimental result directory is included.

The test suite covers frozen-instance identity, self-contained imports, publication configuration controls, matched evolutionary initialization, feasible decoding and refinement, quality metrics, exact-objective consistency, all three forecast candidates, error metrics, and rejection of nonpositive price inputs. The full publication experiment was not rerun during packaging.
