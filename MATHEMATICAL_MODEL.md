# Mathematical model and implementation correspondence

## Sets, indices, parameters, and variables

| Symbol | Unit | Meaning / implementation field |
|---|---|---|
| L, l | index | Cultivation plots; `ProblemData.L` |
| K, k | index | Crops; `crop_names` |
| T, t | week index | Planning periods numbered 0 through T−1 |
| E, e | event index | Eligible plot–crop–sowing combinations with positive yield and harvest inside the horizon |
| A_l | m² | Plot area; `areas[l]` |
| n_k | weeks | Maturation duration; `maturity[k]` |
| s_e, h_e | week index | Sowing and harvest, with h_e = s_e + n_k |
| r | weeks | Common preparation interval; `setup_weeks` |
| r_f | weeks | Additional same-family rest; `same_family_extra_rest` |
| Y_e | kg/m² | Sowing-week yield; `event_yield(l,k,s)` |
| p_kt | COP/kg | Expected harvest-week price; `prices[k,t]` |
| Sigma | dimensionless | Shrinkage covariance matrix of weekly log-price returns |
| M | COP per million COP | Monetary scale, 10⁶ |
| y_e | binary | Event selected; SCIP variable `y` |
| z_e | m² | Cultivated area; SCIP variable `z` |
| x_kt | kg | Harvested production for crop k in week t |
| v_kt | million COP | Price exposure, p_kt x_kt / M |
| R | million COP | Expected gross revenue |
| Q | (million COP)² | Contemporaneous covariance-based monetary price risk |
| rho | million COP | Revenue threshold for epsilon-constraint optimization |

`E` excludes empirically ineligible crops, forbidden sowing weeks, zero-yield events, and harvests outside the horizon; eligibility and calendar exclusions are therefore encoded before constructing decision variables.

## Bi-objective formulation

Maximize expected gross revenue and minimize monetary price risk:

$$R=\sum_{t\in T}\sum_{k\in K}v_{kt},\qquad Q=\sum_{t\in T}v_t^\top\Sigma v_t.$$

Link event selection to cultivated area and define production and exposure:

$$0\le z_e\le A_{l(e)}y_e,\quad y_e\in\{0,1\},\qquad x_{kt}=\sum_{e:k(e)=k,\,h_e=t}Y_e z_e,\qquad v_{kt}=p_{kt}x_{kt}/M.$$

For every plot and week, let C_lw contain events occupying that plot during the half-open interval [s_e,h_e+r), truncated at the horizon; the common occupancy constraint is

$$\sum_{e\in C_{lw}}y_e\le1.$$

For every plot, botanical family, and week, let F_lfw contain same-family events in [s_e,h_e+r+r_f); additional recovery is enforced by

$$\sum_{e\in F_{lfw}}y_e\le1.$$

These constraints match the occupancy and same-family cliques in `exact_scip.build_scip_model`; the evolutionary decoder instead applies the equivalent rest conditions sequentially while constructing feasible schedules.

The implementation supports group-level market capacity, $\sum_{k:g(k)=g}x_{kt}\le D_{gt}$, but the frozen publication configuration leaves this extension inactive because no independent demand-capacity input was supplied.

## Exact optimization

`exact_scip.py` minimizes risk subject to a revenue floor R ≥ rho using a nonnegative epigraph variable q and the quadratic constraint q ≥ Q. It also solves maximum-revenue endpoints and optional tie-break problems, recording solver status, gap, and runtime rather than treating all incumbents as optimal.

## Conditional continuous refinement

Fixing crops and dates gives area vector a, revenue coefficient vector c, covariance-derived positive-semidefinite matrix H, and plot-area bounds u. Refinement solves

$$\min_a a^\top H a\quad\text{subject to}\quad c^\top a\ge c^\top a^{(0)},\quad 0\le a\le u.$$

Active market-capacity constraints can add Ba ≤ d, but they are absent from the bundled publication configuration. The SLSQP implementation uses the analytic gradient 2Ha, and unsuccessful refinements preserve the baseline areas rather than claiming an improved solution.

## Evolutionary methods and assumptions

The shared chromosome represents crop identities, waiting times, and cultivated-area fractions for successive events on each plot. Initialization is shared across the three evolutionary methods, and survival always uses actual revenue and risk through nondominated sorting and crowding distance.

Full refinement applies a conditional solve to every offspring, while H-NSGA-II allocates its fixed refinement budget using predicted observed gain and random exploration. Empty slots, delayed sowing, and partial-area events remain admissible; the objective maximizes gross revenue rather than net profit, and covariance couples crops within a week rather than modeling cross-week covariance.

The price covariance is estimated with Ledoit–Wolf shrinkage from log returns, and `ProblemData` symmetrizes it and corrects materially negative minimum eigenvalues with a diagonal shift. Complete optimization input matrices are frozen in this package; no uncertainty claims beyond the implemented contemporaneous risk model are implied.
