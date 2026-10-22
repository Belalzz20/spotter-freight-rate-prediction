# Spotter Freight Rate Assessment 

This folder preserves the supplied source files and provides an updated reproducible training script. Generated outputs go to `outputs/` so the supplied files are not overwritten.

## Setup (Python 3.10+)

```bash
python -m pip install -r requirements.txt
python train_model.py --data-dir data --output-dir outputs
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_predictions.csv --output-dir outputs/scorer_results
```

The training script recomputes October holdout metrics and writes them to `outputs/holdout_metrics.json`, plus row-level holdout predictions. Do not copy old metrics into a report unless they match this run.

## Important
- `data/december_chart_inputs.csv` is treated as fixed December input features; any existing `predicted_rate` column is overwritten in the generated output.
- Negative weights are retained as observed in the source data; missing weights are imputed with the median from the applicable training data. This is an explicit modeling choice to revisit after the first reproducible baseline.
- The official `score.py` is kept unchanged.

## Files

- `train_model.py` — chronological holdout evaluation, final model training, and prediction generation.
- `data/train_test.csv` — labeled development dataset.
- `data/validation.csv` — unlabeled loads requiring predictions.
- `data/validation_predictions_template.csv` — required submission template.
- `data/december_chart_inputs.csv` — fixed-route December input data.
- `score.py` — supplied output validator and December chart generator.
- `outputs/validation_predictions.csv` — final predictions for 12,000 validation loads.
- `outputs/december_predictions.csv` — predictions for the 31 December dates.
- `outputs/scorer_results/candidate_december.png` — generated December prediction chart.
- `technical_report.pdf` — technical report.

## October Holdout Results

- MAE: $117.01
- RMSE: $647.14
- MAPE: 5.675%
- R²: 0.8208
- Selected iterations: 466

These metrics describe the October 2025 chronological holdout only. Ground-truth labels are unavailable for the final validation dataset, so its prediction accuracy cannot yet be measured.