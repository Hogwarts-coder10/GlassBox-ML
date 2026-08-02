# case_studies/

Scripts that reproduce the empirical results in the paper
(`GlassBoxML_paper.md` / `.tex`, §5.1–5.6). Distinct from:

- `tests/` — unit tests (correctness, run in CI)
- `examples/` — general usage demos, not tied to specific reported numbers

Each script here is self-contained, uses a fixed seed, and prints (or
plots) exactly what's cited in the paper. Run any of them from the repo
root, e.g.:

    python case_studies/random_forest_ensemble_size.py

## Currently included (§5.4–5.6)

| Script | Paper section | Produces |
|---|---|---|
| `random_forest_ensemble_size.py` | §5.4 | Table 4 |
| `svm_feature_scaling.py` | §5.5 | accuracy/weight numbers, `svm_loss_curves.png` (Figure 4) |
| `knn_dimensionality.py` | §5.6 | `knn_dimensionality.png` (Figure 5) |

## Not yet included (§5.1–5.3)

The linear regression, logistic regression, and decision tree case
studies (§5.1–5.3, including `loss_curves.png` / Figure 1) predate this
directory and weren't recovered as standalone scripts — the Methods
section states scripts are available for every case study, so these
three should be reconstructed and added here before submission if that
claim needs to hold up under review. The exact hyperparameters and
seeds used are documented in the paper's Methods section (§Methods) if
you're rebuilding them from scratch:

- **§5.1 Linear Regression**: `LinearRegression` + `Momentum` optimizer,
  lr 0.15 (degrees 1–2) / 0.02 (degree 18), 400 / 6000 epochs,
  `numpy.random.seed(42)`, 65/35 split, `PolynomialFeatures` +
  `StandardScaler`.
- **§5.2 Logistic Regression**: `LogisticRegression` + `GradientDescent`
  (lr 0.1). BCE-vs-MSE: 300 samples, `numpy.random.seed(7)`, 300 epochs,
  70/30 split. Class imbalance: 400 samples (380/20 split), 500 epochs,
  70/30 split.
- **§5.3 Decision Tree**: `DecisionTreeClassifier`, `criterion="gini"`,
  `min_samples_split=2`, `max_depth` in {3, 6, 50}, 200 samples,
  `numpy.random.seed(11)`, 8% label noise, 65/35 split.
