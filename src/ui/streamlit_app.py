"""Streamlit interactive results explorer.

Run with: streamlit run src/ui/streamlit_app.py
Or: make ui
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import settings


def load_data() -> pd.DataFrame:
    """Load aggregated.parquet from results dir."""
    parquet_path = settings.results_dir / "aggregated.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(parquet_path)


def main() -> None:
    """Render the Streamlit explorer app."""
    st.set_page_config(page_title="tiny-llm-edge-bench", layout="wide")
    st.title("tiny-llm-edge-bench - Results Explorer")

    df = load_data()
    if df.empty:
        st.warning("No results found. Run `make bench-all` first.")
        return

    with st.sidebar:
        st.header("Filters")
        models = st.multiselect("Models", df["model_id"].unique(), default=list(df["model_id"].unique()))
        quants = st.multiselect("Quants", df["quant_name"].unique(), default=list(df["quant_name"].unique()))
        tasks = ["All"] + list(df["task_id"].dropna().unique())
        task_filter = st.selectbox("Task", tasks)

    mask = df["model_id"].isin(models) & df["quant_name"].isin(quants)
    if task_filter != "All":
        mask &= df["task_id"] == task_filter
    view = df[mask]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Throughput (tok/s median)")
        if "tps_median" in view.columns:
            st.bar_chart(view.set_index("model_id")[["tps_median"]])
    with col2:
        st.subheader("Quality Score")
        if "quality_score" in view.columns:
            st.bar_chart(view.dropna(subset=["quality_score"]).set_index("model_id")[["quality_score"]])

    st.subheader("Full Results Table")
    st.dataframe(view, use_container_width=True)

    st.subheader("Compare Two Runs")
    run_ids = view["run_id"].tolist()
    if len(run_ids) >= 2:
        col_a, col_b = st.columns(2)
        run_a = col_a.selectbox("Run A", run_ids, key="a")
        run_b = col_b.selectbox("Run B", run_ids, index=1, key="b")
        row_a = view[view["run_id"] == run_a].iloc[0]
        row_b = view[view["run_id"] == run_b].iloc[0]
        compare = pd.DataFrame({"Run A": row_a, "Run B": row_b}).T
        st.dataframe(compare)


if __name__ == "__main__":
    main()
