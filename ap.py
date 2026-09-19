import streamlit as st

# Force wide mode so map and chart fit comfortably side-by-side
st.set_page_config(layout="wide", page_title="UK Cryptosporidium Risk Portal")

# Header section
st.title("UK Cryptosporidium Risk Monitoring Portal")
st.caption("Latest Model Run: October 2026 | Updated Monthly")

# Top KPI Metric Cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Catchments Monitored", "42")
col2.metric("High Risk Catchments", "3", delta="1 from last month", delta_color="inverse")
col3.metric("Model Confidence Score", "94.2%")

st.divider()

# Main Interactive Area: Left = Map, Right = Selected Catchment Graph
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("Interactive Catchment Risk Map")
    # Render your Plotly Mapbox map here
    st.plotly_chart(map_fig, use_container_width=True)

with right_col:
    st.subheader("Risk Trend & Prediction")
    # Render your Plotly line chart here
    st.plotly_chart(trend_fig, use_container_width=True)