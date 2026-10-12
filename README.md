[README.md](https://github.com/user-attachments/files/33327032/README.md)
# NBA Talent Intelligence — Early-Career Screening Prototype

## What this is
A research prototype that tests whether first- and second-season NBA box-score statistics and available biographical attributes can help rank players who later appeared in an NBA All-Star Game during career seasons 3–7.

The project includes a reproducible Jupyter notebook, a Random Forest modelling pipeline, temporal test-cohort evaluation results, and a Streamlit dashboard for exploring model performance and ranked player results.

The intended use is **decision support for further scouting and development review**, not automated recruitment or player selection. Results should be interpreted as exploratory evidence, with limitations in historical data, outcome definition, model generalisability, and the distinction between All-Star selection and broader player potential.

## Included
- `NBA_Talent_Intelligence.ipynb` — reproducible analysis notebook.
- `app.py` — Streamlit retrospective evaluation dashboard.
- `evaluation_metrics.json` and `model_comparison.csv` — evaluation outputs.
- `scored_test_players.csv` — compact temporal test-cohort results for the dashboard.
- `requirements.txt` — Python dependencies.

## Source data
The source datasets are not bundled because `PlayerStatistics.csv` is approximately 390 MB. Keep the original CSV files beside the notebook or rename them to:
- `PlayerStatistics.csv`
- `Players.csv`

The notebook also recognises the uploaded filenames `PlayerStatistics(1).csv` and `Players(1).csv`.

## Run the notebook
1. Install Python 3.10+.
2. Install dependencies: `pip install -r requirements.txt`
3. Open `NBA_Talent_Intelligence.ipynb` in Jupyter.
4. Run all cells from top to bottom.

The notebook loads the full game-by-game file and may take a few minutes depending on hardware.

## Run the dashboard
1. Install dependencies: `pip install -r requirements.txt`
2. From this folder, run: `streamlit run app.py`
3. Open the local URL shown by Streamlit.

The dashboard uses the included compact historical test cohort and metrics file. It does not retrain the model when launched.

## Target and validation
- Features: season-level averages and totals from the first two observed regular seasons, plus available age, height, weight, position indicators and draft information.
- Target: All-Star Game appearance in career seasons 3–7, excluding players with an All-Star appearance in seasons 1–2.
- Train cohort: debut season through 2010.
- Temporal test cohort: debut seasons 2011–2019.
- Baseline: rank by mean points per game across seasons 1 and 2.
- Metrics: ROC-AUC, average precision, Brier score (model only), Precision@10% and Recall@10%.

## Preliminary retrospective results
Random Forest: ROC-AUC 0.949; average precision 0.558; Brier score 0.037; Precision@10% 0.373; Recall@10% 0.694.
Mean-PPG baseline: ROC-AUC 0.905; average precision 0.445; Precision@10% 0.328; Recall@10% 0.611.

These results are exploratory. The test cohort contains only 36 positive outcomes. All-Star selection is a proxy for future success, not a complete measure of player value. Era differences, selection processes, survivorship/coverage and missing non-box-score information limit generalisation. Further robustness checks, calibration review and scout validation are required before any operational use.
