import streamlit as st
import pandas as pd
import pyproj
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. UI Password Guard
# ---------------------------------------------------------
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔒 UK Water Intake Risk Portal")
        st.subheader("Restricted Access")
        
        entered_password = st.text_input("Enter Access Password:", type="password")
        if st.button("Log In"):
            # Fetch password from Streamlit secrets (or fallback to default)
            correct_password = st.secrets.get("APP_PASSWORD", "demo123")
            if entered_password == correct_password:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Access denied.")
        return False
    return True

# Stop script execution until the user logs in
if not check_password():
    st.stop()

# ---------------------------------------------------------
# 2. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="UK Cryptosporidium Risk Map",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 3. Secure Data Loading & Coordinate Transformation
# ---------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    # Read URLs securely from Streamlit Secrets (falls back to local files if secrets aren't set)
    meta_url = st.secrets.get("META_DATA_URL", "zs_combined_(all)_REORDERED.csv")
    pred_url = st.secrets.get("PRED_DATA_URL", "compiled_site_adaptive_predictions_with_risk_levels.csv")

    meta_df = pd.read_csv(meta_url)
    pred_df = pd.read_csv(pred_url)

    # Convert OSGB36 Eastings/Northings (EPSG:27700) to WGS84 Lat/Lon (EPSG:4326)
    transformer = pyproj.Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)
    meta_df['lon'], meta_df['lat'] = transformer.transform(
        meta_df['Eastings'].values, 
        meta_df['Northings'].values
    )

    # Handle missing values
    pred_df['predicted_risk_level'] = pred_df['predicted_risk_level'].fillna('No Data')
    if 'observed_risk_level' in pred_df.columns:
        pred_df['observed_risk_level'] = pred_df['observed_risk_level'].fillna('No Data')

    # Merge spatial metadata with risk predictions
    merged_df = pd.merge(
        pred_df, 
        meta_df[['Site', 'Site_No', 'Region', 'lat', 'lon']], 
        on='Site', 
        how='left'
    )
    
    return merged_df

df = load_and_prepare_data()

# ---------------------------------------------------------
# 4. Sidebar Controls & Metrics
# ---------------------------------------------------------
st.sidebar.title("Navigation & Filters")
available_months = sorted(df['YearMonth'].unique(), reverse=True)
selected_month = st.sidebar.selectbox("Select Model Calendar Month:", available_months)

latest_df = df[df['YearMonth'] == selected_month].copy()

st.sidebar.divider()
st.sidebar.markdown(f"### Summary for **{selected_month}**")
st.sidebar.metric("Total Active Sites", len(latest_df))

high_count = (latest_df['predicted_risk_level'] == 'High').sum()
med_count = (latest_df['predicted_risk_level'] == 'Medium').sum()
low_count = (latest_df['predicted_risk_level'] == 'Low').sum()

st.sidebar.write(f"🔴 **High Risk:** {high_count}")
st.sidebar.write(f"🟡 **Medium Risk:** {med_count}")
st.sidebar.write(f"🟢 **Low Risk:** {low_count}")

# ---------------------------------------------------------
# 5. Dashboard View (Map & Trend Inspector)
# ---------------------------------------------------------
st.title("UK Cryptosporidium Catchment Risk Portal")
st.caption("Interactive Risk Predictions & Historical Water Intake Monitoring")

col_map, col_chart = st.columns([1.3, 1])

# Column 1: Mapbox/MapLibre Risk Map
with col_map:
    st.subheader(f"Catchment Risk Levels ({selected_month})")
    
    risk_color_map = {
        "High": "#E63946", 
        "Medium": "#FFB703", 
        "Low": "#2A9D8F",
        "No Data": "#94A3B8"
    }
    
    # Version-safe check for Plotly 6+ (scatter_map) vs Plotly 5 (scatter_mapbox)
    if hasattr(px, "scatter_map"):
        fig_map = px.scatter_map(
            latest_df,
            lat="lat",
            lon="lon",
            color="predicted_risk_level",
            color_discrete_map=risk_color_map,
            category_orders={"predicted_risk_level": ["High", "Medium", "Low", "No Data"]},
            hover_name="Site",
            hover_data={
                "predicted_risk_level": True,
                "predicted_risk": ":.4f",
                "lat": False,
                "lon": False
            },
            zoom=5,
            center={"lat": 55.0, "lon": -3.5},
            map_style="carto-positron"
        )
    else:
        fig_map = px.scatter_mapbox(
            latest_df,
            lat="lat",
            lon="lon",
            color="predicted_risk_level",
            color_discrete_map=risk_color_map,
            category_orders={"predicted_risk_level": ["High", "Medium", "Low", "No Data"]},
            hover_name="Site",
            hover_data={
                "predicted_risk_level": True,
                "predicted_risk": ":.4f",
                "lat": False,
                "lon": False
            },
            zoom=5,
            center={"lat": 55.0, "lon": -3.5},
            mapbox_style="carto-positron"
        )
    
    fig_map.update_traces(marker={"size": 13, "opacity": 0.85})
    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend_title_text="Predicted Risk Level"
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

# Column 2: Catchment Inspector Chart
with col_chart:
    st.subheader("Site Risk Inspector")
    
    site_list = sorted(df['Site'].unique())
    selected_site = st.selectbox("Select Catchment to View Historical Trend:", site_list)
    site_df = df[df['Site'] == selected_site].sort_values('YearMonth')
    
    fig_chart = go.Figure()
    fig_chart.add_trace(go.Scatter(
        x=site_df['YearMonth'], 
        y=site_df['predicted_risk'], 
        mode='lines+markers', 
        name='Predicted Risk', 
        line=dict(color='#2563EB', width=2.5)
    ))
    
    if 'observed_risk' in site_df.columns:
        fig_chart.add_trace(go.Scatter(
            x=site_df['YearMonth'], 
            y=site_df['observed_risk'], 
            mode='lines+markers', 
            name='Observed Risk', 
            line=dict(color='#475569', width=2, dash='dot')
        ))
    
    fig_chart.update_layout(
        title=f"Risk Score Trend: {selected_site}",
        xaxis_title="Month", 
        yaxis_title="Risk Score",
        hovermode="x unified", 
        template="plotly_white",
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_chart, use_container_width=True)

    latest_site_row = site_df[site_df['YearMonth'] == selected_month]
    if not latest_site_row.empty:
        curr_pred = latest_site_row['predicted_risk_level'].values[0]
        curr_model = latest_site_row['Selected_Model'].values[0] if 'Selected_Model' in latest_site_row.columns else 'Adaptive'
        st.info(f"**Latest Status ({selected_month}):** Risk Level = **{curr_pred}** | Selected Model = **{curr_model}**")