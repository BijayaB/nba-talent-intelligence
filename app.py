from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA Talent Intelligence", page_icon="🏀", layout="wide")
BASE = Path(__file__).parent
DATA_PATH = BASE / "scored_test_players.csv"
METRICS_PATH = BASE / "evaluation_metrics.json"

st.title("NBA Talent Intelligence")
st.caption("Early-career talent screening prototype | Retrospective evaluation dashboard")
st.warning("Demonstrator only: this dashboard shows a historical temporal test cohort. It is not a live recruitment recommendation system.")

if not DATA_PATH.exists() or not METRICS_PATH.exists():
    st.error("Required files are missing. Keep scored_test_players.csv and evaluation_metrics.json beside app.py.")
    st.stop()

df = pd.read_csv(DATA_PATH)
with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

model_metrics = metrics.get("random_forest", metrics.get("model", {}))
baseline_metrics = metrics.get("baseline_mean_points_per_game", metrics.get("baseline", {}))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test-cohort players", f"{len(df):,}")
c2.metric("Future All-Star outcomes", f"{int(df['actual_future_allstar'].sum()):,}")
c3.metric("Random Forest ROC-AUC", f"{model_metrics.get('roc_auc', model_metrics.get('ROC-AUC', 0)):.3f}")
c4.metric("Precision@10%", f"{model_metrics.get('precision_at_k', model_metrics.get('Precision@10%', 0)):.1%}")

st.subheader("Shortlist exploration")
left, right = st.columns([1, 2])
with left:
    seasons = sorted(df["debut_season"].dropna().astype(int).unique())
    selected_seasons = st.multiselect("Debut season", seasons, default=seasons)
    positions = [("Guard", "guard"), ("Forward", "forward"), ("Center", "center")]
    selected_positions = st.multiselect("Position indicator", [p[0] for p in positions], default=[p[0] for p in positions])
    top_n = st.slider("Number of ranked players", min_value=10, max_value=100, value=30, step=5)
    show_outcome = st.checkbox("Show retrospective outcome (evaluation only)", value=False)

filtered = df[df["debut_season"].astype(int).isin(selected_seasons)].copy()
position_mask = pd.Series(False, index=filtered.index)
for label, col in positions:
    if label in selected_positions:
        position_mask = position_mask | (pd.to_numeric(filtered[col], errors="coerce").fillna(0) == 1)
filtered = filtered[position_mask]
filtered = filtered.sort_values("predicted_probability", ascending=False).head(top_n)

with right:
    st.markdown("**How to interpret this view**")
    st.write("Players are ranked by the model's retrospective score. The score is a ranking aid, not a verified probability of player value. Scouts should review role, minutes, context and qualitative evidence before any decision.")
    display_cols = ["player_name", "debut_season", "predicted_probability", "model_rank",
                    "baseline_mean_points_pg", "baseline_rank"]
    if show_outcome:
        display_cols.append("actual_future_allstar")
    st.dataframe(
        filtered[display_cols].rename(columns={
            "player_name":"Player", "debut_season":"Debut season",
            "predicted_probability":"Model score", "model_rank":"Model rank",
            "baseline_mean_points_pg":"Mean PPG baseline score", "baseline_rank":"Baseline rank",
            "actual_future_allstar":"Future All-Star (evaluation)"
        }).style.format({"Model score":"{:.3f}", "Mean PPG baseline score":"{:.2f}"}),
        use_container_width=True, hide_index=True
    )

st.subheader("Model versus baseline")
compare = pd.DataFrame({
    "Metric": ["ROC-AUC", "Average precision", "Precision@10%", "Recall@10%"],
    "Random Forest": [
        model_metrics.get("roc_auc", model_metrics.get("ROC-AUC")),
        model_metrics.get("average_precision", model_metrics.get("Average precision")),
        model_metrics.get("precision_at_k", model_metrics.get("Precision@10%")),
        model_metrics.get("recall_at_k", model_metrics.get("Recall@10%"))
    ],
    "Mean PPG baseline": [
        baseline_metrics.get("roc_auc", baseline_metrics.get("ROC-AUC")),
        baseline_metrics.get("average_precision", baseline_metrics.get("Average precision")),
        baseline_metrics.get("precision_at_k", baseline_metrics.get("Precision@10%")),
        baseline_metrics.get("recall_at_k", baseline_metrics.get("Recall@10%"))
    ]
})
st.dataframe(compare.style.format({"Random Forest":"{:.3f}", "Mean PPG baseline":"{:.3f}"}), use_container_width=True, hide_index=True)
st.bar_chart(compare.set_index("Metric")[["Random Forest", "Mean PPG baseline"]])

with st.expander("Target definition, caveats and governance"):
    st.write("Target: at least one All-Star Game appearance in career seasons 3–7, excluding players who already appeared in seasons 1–2. Only players with a complete observable outcome window are included.")
    st.write("Limitations: All-Star selection is influenced by factors beyond player performance; historical eras differ; box-score data omits tracking, health, scouting and contract context; and only 36 positive outcomes appear in the temporal test cohort. The prototype must not be used as an automated recruitment decision.")
    st.write("Recommended next steps: validate with scouts, perform era-aware and subgroup checks, review calibration, document a baseline scouting process, and assess operational benefit in a controlled pilot.")

st.caption("Built for academic demonstration. Model scores and retrospective outcomes should be interpreted with caution.")
