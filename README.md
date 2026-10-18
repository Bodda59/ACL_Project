# Formula 1 Podium Prediction

A machine-learning project that predicts whether an F1 driver will finish on the podium using information available **after qualifying and before the Grand Prix starts**. The project combines relational data auditing, cleaning, exploratory analysis, feature engineering, statistical models, a TensorFlow neural network, and model explanations in one Jupyter notebook.

The main implementation is [`acl-ms1.ipynb`](acl-ms1.ipynb). The notebook includes the reasoning, intermediate tables, plots, validation assertions, experiments, and inference demonstrations needed to understand the pipeline.

## Prediction task

| Item | Definition |
|---|---|
| Problem | Binary classification: podium versus no podium |
| Observation | One driver in one race |
| Unique modeling key | `(raceId, driverId)` |
| Positive label | `positionOrder` is 1, 2, or 3 |
| Negative label | Any other recorded finishing order |
| Prediction cutoff | After qualifying, before the main race starts |
| Historical scope | Dataset covering 1950–2024 |
| Output | Estimated podium probability and a threshold-based classification |

```python
podium = positionOrder.between(1, 3).astype(int)
```

The target follows the dataset's recorded final classification. The task does not predict exact finishing positions or require exactly three positive predictions per race.

## Repository structure

```text
ACL_Project/
├── acl-ms1.ipynb     # Main analysis, cleaning, experiments, models, inference, and XAI
├── README.md        # Project documentation
├── pyproject.toml   # Python requirements and project dependencies
├── uv.lock          # Dependency lockfile
├── .python-version  # Python 3.12 selection
├── main.py          # Starter script; not the training entry point
└── data/            # Local dataset CSVs; excluded from version control
```

`tmp/` is also excluded from version control and is used for temporary working files. Running `main.py` does not execute the ML pipeline; open the notebook instead.

## Dataset

The project uses the [Formula 1 World Championship dataset on Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020), derived from Ergast data. Although the source URL contains `1950-2020`, this project's intended snapshot extends through 2024. Verify the years in your downloaded copy before reproducing the results.

The dataset consists of 14 related CSV tables:

| File | Main information |
|---|---|
| `circuits.csv` | Circuit identity, location, country, and coordinates |
| `constructors.csv` | Constructor identities and metadata |
| `constructor_results.csv` | Per-race constructor summaries |
| `constructor_standings.csv` | Constructor standings after each race |
| `drivers.csv` | Driver identities, nationality, and date of birth |
| `driver_standings.csv` | Driver standings after each race |
| `races.csv` | Race identity, season, round, circuit, and date |
| `results.csv` | Driver–race results, grid, classification, points, and status |
| `qualifying.csv` | Qualifying positions and Q1/Q2/Q3 times |
| `lap_times.csv` | Recorded lap-by-lap timings |
| `pit_stops.csv` | Recorded pit stops and durations |
| `sprint_results.csv` | Sprint classifications and timings |
| `seasons.csv` | Championship seasons |
| `status.csv` | Descriptions of result status codes |

Tables are joined through identifiers such as `raceId`, `driverId`, `constructorId`, `circuitId`, and `statusId`. Primary keys, foreign keys, and join cardinality are checked explicitly.

The documented snapshot contains **26,759 raw result entries**. Shared-drive consolidation produces **26,668 unique driver–race records**. These counts depend on the dataset snapshot and are validated in the notebook.

## Setup and execution

### 1. Prepare the Python environment

Use Python **3.12** to match `.python-version`; the project declares Python `>=3.12`.

With `uv` installed, run from the repository root:

```bash
uv sync --locked
```

The dependencies include NumPy, pandas, SciPy, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras, SHAP, LIME, and the IPython kernel. `pyproject.toml` declares the requirements; `uv.lock` records the resolved environment.

### 2. Obtain the CSV files

Download the dataset and place all 14 CSVs directly inside `data/`. They are not included in a normal clone because `data/` is ignored by Git.

In the notebook's first code cell, set `DATA_DIR` to your dataset folder. For a local checkout, replace the existing machine-specific absolute path with:

```python
DATA_DIR = Path("data")
```

The current code falls back to `archive/` if the configured folder is unavailable. Run the notebook from the repository root when using a relative path.

### 3. Open the notebook

Use VS Code, JupyterLab, or another Jupyter-compatible editor and select the project's `.venv` Python kernel.

To start JupyterLab with `uv` without adding it to the project dependencies:

```bash
uv run --with jupyterlab jupyter lab acl-ms1.ipynb
```

Restart the kernel and select **Run All**. The initial setup cell checks imports and displays package versions. If LIME is missing, it attempts to install `lime==0.2.0.1`; that installation requires network access.

Training includes multiple model versions, hyperparameter searches, independent feature-group ablations, and SHAP/LIME explanations. A full run therefore takes longer than fitting a single model. No fixed runtime is guaranteed.

### Running on Kaggle

1. Import `acl-ms1.ipynb` into a Kaggle notebook.
2. Attach the dataset.
3. Set `DATA_DIR` to the actual mounted CSV directory shown in Kaggle's input browser. The saved local path is not portable to Kaggle.
4. Enable Internet if the setup needs to install LIME.
5. Restart the session and run all cells in order.

The notebook's saved metadata and outputs do not certify that a fresh Kaggle session will run successfully. Check the displayed dependency versions and the complete execution in the target environment.

## Notebook workflow

| Section | Contents |
|---|---|
| 1 | Environment checks and data loading |
| 2 | Raw EDA, profiling, missingness, key checks, and target diagnostics |
| 3 | Table-specific cleaning, historical investigations, and final validation |
| 4 | Three analytical questions requiring relational joins |
| 5 | Candidate features, coverage, associations, and pre-race availability |
| 6 | Modeling setup, feature experiments, preprocessing, and shared helpers |
| 7 | TensorFlow/Keras neural-network experiments |
| 8 | Random-forest experiments and tuning |
| 9 | Logistic-regression experiments and tuning |
| 10 | Earlier experimental model comparison and ablation |
| 11 | Final comparable models, evaluation, group ablations, and inference |
| 12 | Global and local explanations for all three final models |

Sections 7–10 compare experimental versions. Section 11 trains the final three models under one common configuration; it does not automatically adopt a different winning training window for each model.

## Data cleaning and validation

### Raw-data preservation

Cleaning operates on separate `_clean` DataFrames. The notebook records fingerprints of the raw tables and CSV files and checks that cleaning has not modified the originals. Before/after tables and plots make the effects of each treatment visible.

### Main checks and treatments

- Detect literal `\N` sentinels, missing values, blanks, and semantic type problems.
- Parse dates and convert lap/race durations to numeric seconds where applicable.
- Distinguish elapsed durations, gaps to the winner, and time-of-day readings.
- Audit duplicate records, primary-key uniqueness, non-null identifiers, and foreign-key coverage.
- Investigate qualifying coverage by season and distinguish absent records from missing later qualifying sessions.
- Document supported historical corrections and retain explicit limitations where evidence is incomplete.
- Check row counts, schema changes, target consistency, and the final driver–race grain.

### Historical shared drives

Some early races contain multiple car entries for the same driver. To construct one modeling row per driver and race, the notebook:

1. Counts the original entries.
2. Sums the driver's points across those entries.
3. Retains the remaining fields from the entry with the best, lowest `positionOrder`.
4. Uses `resultId` to break ties deterministically.

This preserves the driver's best credited outcome, but the retained grid and constructor may describe a car acquired during the race rather than the original starting car. That is an important limitation of the historical representation, not a reconstruction of pre-race information.

### Position-column comparison

The results and sprint-results sections independently compare `position`, `positionText`, and `positionOrder` using:

- Pearson and Spearman correlations on available numeric pairs;
- pair counts and coverage;
- exact agreement percentages and mismatch counts;
- a correlation heatmap.

Outcome letters in `positionText` are excluded from numeric correlations. The notebook drops `position` from the relevant cleaned table only when both correlations with the other representations are at least **0.95**, numeric `positionText` exactly preserves every observed position, and `positionOrder` is complete.

`positionText` remains available for outcome interpretation; `positionOrder` remains available for the target. The correlation threshold is a documented project rule, not a universal feature-selection test.

### Existing missing-value conventions

The current notebook still uses documented `-1` placeholders for two unresolved event car numbers and unavailable sprint timing fields. Their validation checks and timing plots account for those conventions. These fields are not part of the final 14-feature model input list. The position-column comparison does not change those separate cleaning decisions.

## Data-engineering questions

The notebook answers three questions with joins, plots, and documented denominators:

1. **Front-row conversion:** which circuits most often convert a start from grid positions 1–2 into a podium, and how does the ranking differ before and from 2014 onward?
2. **Home-race association:** do drivers podium more often in their home country when comparing similar starting positions?
3. **Mechanical reliability:** which constructors have the highest mechanical-retirement rates in 2014–2021 versus 2022–2024?

Qualifying position and starting grid are treated separately. Nationality-to-country mappings and mechanical-failure groupings are documented. The recorded-activity screens used in individual analyses do not define the final modeling population.

## Feature engineering and leakage control

Only information available before the Grand Prix is eligible as a predictor. Current-race classification, points, laps, retirement status, pit stops, and fastest laps are excluded from current-race model inputs.

Historical features use earlier races. Constructor histories are calculated at race level before being joined back to driver records, preventing one teammate's current-race result from entering another teammate's history. Championship standings are joined from earlier rounds, with documented season-opening fallbacks.

The final **E3** configuration contains 14 inputs:

| Group | Features |
|---|---|
| Starting grid | `grid`, `no_grid_slot` |
| Prior championship performance | `driver_standing_before`, `constructor_standing_before`, `driver_unranked`, `constructor_unranked`, `driver_gp_points_before` |
| Recent form | `driver_podium_last5`, `constructor_podium_last5` |
| Experience and age | `driver_age`, `driver_prior_appearances`, `constructor_prior_recorded_races` |
| Context | `is_home`, `year` |

Recorded grid zero is not interpreted as a position ahead of pole. For modeling, it becomes the largest positive recorded grid in that race plus one, with `no_grid_slot = 1`. This is a shared fallback value, not an inferred ordering among drivers without numbered slots.

Feature experiments compare alternatives such as 5- versus 10-race form, prior wins, reliability, circuit history, standings, and context. E3 is the documented final design choice; the notebook explicitly acknowledges that an alternative feature set can achieve a higher experimental score.

## Splitting and preprocessing

The final evaluation follows time rather than a random row split:

| Partition | Seasons | Purpose |
|---|---|---|
| Training | 1950–2019 | Fit models and preprocessing |
| Validation | 2020–2021 | Select settings, thresholds, and the default inference model |
| Test | 2022–2024 | Report held-out performance |

All driver records from a race belong to the same partition. The model code expresses the test mask as `year >= 2022`, so the downloaded snapshot must be checked if it contains seasons beyond 2024.

Median imputers are fitted on training inputs only. The neural network and logistic regression also use a training-fitted `StandardScaler`; random forests use imputed, unscaled inputs.

Preprocessing experiments compare training-only versus deliberately leaky train-plus-validation fitting, imputation before/after scaling, and clipping before/after scaling. The deliberately leaky variant is an illustrative experiment, not the final preprocessing policy.

## Models and experiments

### Logistic regression

An interpretable statistical baseline fitted to scaled inputs. Regularization settings are compared using validation data. It provides a useful reference for whether additional model complexity improves performance.

### Random forest

An ensemble that captures nonlinear relationships and feature interactions. The notebook compares a default forest with tuned settings, including depth, leaf size, feature sampling, and class weighting. Final forests use 500 trees.

### TensorFlow/Keras shallow neural network

A feed-forward binary classifier with two hidden layers:

```text
14 inputs
  → Dense(128) → Batch normalization → ReLU → Dropout(0.4)
  → Dense(64)  → Batch normalization → ReLU → Dropout(0.3)
  → Dense(1, sigmoid)
```

Training uses Adam with learning rate `0.001`, binary cross-entropy, batch size 256, and up to 100 epochs. Early stopping monitors validation loss with patience 8 and restores the best weights.

### Training-data versions and ablations

Each model is explored with four versions:

| Version | Training data | Inputs |
|---|---|---|
| Baseline (1950+) | 1950–2019 | E3 |
| 2003+ | 2003–2019 | E3 |
| 2003+ + Q1 | 2003–2019 | E3 plus raw Q1 seconds |
| 2003+ + Q1 gap | 2003–2019 | E3 plus Q1 time relative to the fastest Q1 in the race |

Validation and test rows remain the same across versions. The final comparison deliberately uses the common 1950+ training window and E3 inputs for all three models.

Final ablations retrain each model after independently removing grid, standings, recent form, experience/age, or context. Section 11 keeps each model's full-feature threshold fixed across its group-removal runs, separating threshold changes from changes in the feature set.

## Evaluation

The primary metric is **PR-AUC**, reported using scikit-learn's **average precision** calculation. This differs from a trapezoidal integral under the precision–recall curve. ROC-AUC, F1, precision, recall, and accuracy are also reported.

Probability thresholds are selected by maximizing validation F1 over `0.05`–`0.95` in steps of `0.01`. Test results do not select the threshold or the default inference model. Accuracy is interpreted alongside the always-no-podium baseline because podiums are a minority class.

### Saved example results

The following values are transcribed from the notebook's saved **section 11.5** output for the final 14-feature models. They are not a new training run or a guarantee of identical reproduction; rerun the notebook to obtain current results.

| Model | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1 | Test precision | Test recall |
|---|---:|---:|---:|---:|---:|---:|
| Neural network | 0.7423 | 0.6940 | 0.9364 | 0.6653 | 0.5830 | 0.7745 |
| Random forest | 0.7247 | 0.7291 | 0.9403 | 0.6863 | 0.6015 | 0.7990 |
| Logistic regression | 0.7143 | 0.6912 | 0.9334 | 0.6523 | 0.5830 | 0.7402 |

In that saved run, the neural network is selected for default inference because it has the highest validation PR-AUC. The forest's higher test PR-AUC does not change the validation-based selection rule.

## Inference

Section **11.6** defines `predict_podium(record)` after the final models have been trained. It accepts a dictionary or pandas Series containing the engineered E3 features in their original units, before imputation and scaling.

```python
# Run after the preceding notebook cells create complete_record and the fitted models.
result = predict_podium(complete_record)
print(result)
```

The returned dictionary contains the selected model name, podium probability, saved threshold, podium/no-podium prediction, missing input names, and the values substituted by the training-fitted imputer.

The notebook demonstrates both a complete record and a copy with missing inputs. Omitted fields, `None`, and `NaN` are treated as missing. Unknown feature names, nonnumeric input, infinite values, and entirely missing records are rejected.

The function expects prepared historical features; it does not fetch race data or construct a driver's history from an ID. Supply the engineered grid representation described above, not an unadjusted recorded zero. No fitted model export or standalone serving API is provided by this repository.

## Explainability

Section 12 explains **all three final models**, using common validation records and a common training background:

- **Permutation importance:** change in validation average precision when each input is shuffled.
- **Global SHAP:** distributions and magnitudes of feature contributions across shared validation examples.
- **Local SHAP:** a waterfall explanation of the same individual driver–race record for each model.
- **LIME:** a local surrogate explanation for the same record, including the surrogate's fit score.

The notebook checks that SHAP contributions reconstruct the predicted probabilities and provides a model-agnostic fallback when necessary. Explanations describe how the fitted model uses its inputs; they do not establish causal effects on podium finishes.

## Limitations and reproducibility

- **Historical drift:** regulations, reliability, team identities, field sizes, and qualifying formats change substantially across the dataset.
- **Population scope:** the final historical modeling table includes Indianapolis entries and recorded non-starts. Its population is broader than a modern confirmed starting grid.
- **Shared-drive representation:** choosing the best-finishing historical entry can make retained grid/team information depend on events after the pre-race cutoff. This limits strict prospective interpretation for those records.
- **Qualifying coverage:** older records are sparse, and absent Q2/Q3 times may reflect session progression rather than random missingness.
- **Identity changes:** constructor renaming and driver/team movement complicate historical form features.
- **Validation reuse:** extensive feature and model comparisons reuse the validation period; small gains should not be treated as statistically proven improvements.
- **Descriptive EDA:** some exploratory plots use all years. Treat them as descriptive evidence rather than independent held-out evaluation.
- **Remaining sentinels:** current car-number and sprint-timing conventions require care if those fields are added to future models.
- **Probabilities:** estimates are not demonstrated to be calibrated, and independent binary predictions do not enforce three podium finishers per race.
- **Execution:** neural-network outcomes can vary with software versions and hardware despite fixed seeds. Saved outputs may come from an earlier session.

The notebook includes the analysis and figures needed for a milestone report. A separately exported PDF report is not currently included in the repository.
