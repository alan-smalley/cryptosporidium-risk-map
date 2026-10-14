import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyproj
import streamlit as st
from shapely import ops, wkb

# ---------------------------------------------------------
# 1. UI Password Guard
# ---------------------------------------------------------
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔒 UK Water Intake Risk Portal")
        st.subheader("Restricted Access")

        entered_password = st.text_input(
            "Enter Access Password:", type="password"
        )
        if st.button("Log In"):
            correct_password = st.secrets.get("APP_PASSWORD", "demo123")
            if entered_password == correct_password:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Access denied.")
        return False
    return True


if not check_password():
    st.stop()

# ---------------------------------------------------------
# 2. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="UK Cryptosporidium Risk Map",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize coordinate transformer (EPSG:27700 OSGB36 -> EPSG:4326 WGS84)
transformer = pyproj.Transformer.from_crs(
    "EPSG:27700", "EPSG:4326", always_xy=True
)

# 2a. Reload / Reset Data Button in Sidebar
if st.sidebar.button("Reload Data"):
    st.session_state.clear()
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

# ---------------------------------------------------------
# 3. Secure Data Loading & Helper Functions
# ---------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    meta_url = st.secrets.get("META_DATA_URL", "zs_combined_(all)_REORDERED.csv")
    pred_url = st.secrets.get(
        "PRED_DATA_URL", "compiled_site_adaptive_predictions_with_risk_levels.csv"
    )

    try:
        meta_df = pd.read_csv(meta_url)
        pred_df = pd.read_csv(pred_url)
    except Exception as e:
        st.error(
            f"⚠️ Failed to load CSV data. Please check file paths or Streamlit Secrets. Details: {e}"
        )
        st.stop()

    # Convert OSGB36 Eastings/Northings to WGS84 Lat/Lon
    meta_df["lon"], meta_df["lat"] = transformer.transform(
        meta_df["Eastings"].values, meta_df["Northings"].values
    )

    # Handle missing values
    pred_df["predicted_risk_level"] = pred_df["predicted_risk_level"].fillna(
        "No Data"
    )
    if "observed_risk_level" in pred_df.columns:
        pred_df["observed_risk_level"] = pred_df["observed_risk_level"].fillna(
            "No Data"
        )

    # Merge spatial metadata with risk predictions
    merged_df = pd.merge(
        pred_df,
        meta_df[["Site", "Site_No", "Region", "lat", "lon"]],
        on="Site",
        how="left",
    )

    return merged_df


def parse_gpkg_geom(blob):
    """Parses binary GeoPackage BLOB into Shapely geometry."""
    if blob is None:
        return None
    flags = blob[3]
    envelope_type = (flags >> 1) & 0x07
    env_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    header_len = 8 + env_sizes.get(envelope_type, 0)
    return wkb.loads(blob[header_len:])


@st.cache_data
def get_site_spatial_data(site_name, amber_thresh=0.5, red_thresh=2.0):
    """Loads upstream river network vectors and scored CSO sites for a target intake."""
    # A. Extract Feature Importance for Site CSOs
    try:
        df_feat = pd.read_csv("all_site_feature_importance.csv")
        site_feats = df_feat[df_feat["Site"] == site_name]

        cso_importance = {}
        for _, row in site_feats.iterrows():
            feat = str(row["Feature"])
            if feat.startswith("CSO_Individual_"):
                cso_id = feat.replace("CSO_Individual_", "")
                cso_importance[cso_id] = row["Gain_Percent"]
    except Exception:
        cso_importance = {}

    # B. Load CSO Geometries
    try:
        conn_cso = sqlite3.connect("riversage_52_catchments__cso_sites.gpkg")
        cso_df = pd.read_sql_query(
            "SELECT canonical_cso_id, company, canonical_name, canonical_easting, canonical_northing FROM riversage_52_catchments__cso_sites",
            conn_cso,
        )
        conn_cso.close()

        site_csos = cso_df[
            cso_df["canonical_cso_id"].isin(cso_importance.keys())
        ].copy()

        if not site_csos.empty:
            site_csos["Gain_Percent"] = (
                site_csos["canonical_cso_id"].map(cso_importance).fillna(0.0)
            )
            lons, lats = transformer.transform(
                site_csos["canonical_easting"].values,
                site_csos["canonical_northing"].values,
            )
            site_csos["lon"] = lons
            site_csos["lat"] = lats

            def assign_color(gain):
                if gain >= red_thresh:
                    return "Red (High Impact)"
                elif gain >= amber_thresh:
                    return "Amber (Moderate Impact)"
                else:
                    return "Grey (Low/None)"

            site_csos["Impact_Level"] = site_csos["Gain_Percent"].apply(
                assign_color
            )
    except Exception:
        site_csos = pd.DataFrame()

    # C. Load Upstream River Reach Geometry Vectors
    river_lons, river_lats = [], []
    try:
        conn_net = sqlite3.connect(
            "riversage_52_catchments__upstream_networks.gpkg"
        )
        cursor = conn_net.cursor()
        cursor.execute(
            "SELECT geom FROM riversage_52_catchments__upstream_networks WHERE target_name = ?",
            (site_name,),
        )
        net_rows = cursor.fetchall()
        conn_net.close()

        for (geom_blob,) in net_rows:
            if geom_blob:
                geom = parse_gpkg_geom(geom_blob)
                geom_wgs84 = ops.transform(transformer.transform, geom)
                geoms = (
                    geom_wgs84.geoms
                    if geom_wgs84.geom_type == "MultiLineString"
                    else [geom_wgs84]
                )
                for g in geoms:
                    coords = list(g.coords)
                    river_lons.extend([c[0] for c in coords] + [None])
                    river_lats.extend([c[1] for c in coords] + [None])
    except Exception:
        pass

    return site_csos, river_lons, river_lats


df = load_and_prepare_data()

# ---------------------------------------------------------
# 4. Sidebar Controls & Metrics
# ---------------------------------------------------------
st.sidebar.title("Navigation & Filters")
available_months = sorted(df["YearMonth"].unique(), reverse=True)
selected_month = st.sidebar.selectbox(
    "Select Model Calendar Month:", available_months
)

latest_df = df[df["YearMonth"] == selected_month].copy()

st.sidebar.divider()
st.sidebar.markdown(f"### Summary for **{selected_month}**")
st.sidebar.metric("Total Active Sites", len(latest_df))

high_count = (latest_df["predicted_risk_level"] == "High").sum()
med_count = (latest_df["predicted_risk_level"] == "Medium").sum()
low_count = (latest_df["predicted_risk_level"] == "Low").sum()

st.sidebar.write(f"🔴 **High Risk:** {high_count}")
st.sidebar.write(f"🟡 **Medium Risk:** {med_count}")
st.sidebar.write(f"🟢 **Low Risk:** {low_count}")

st.sidebar.divider()
st.sidebar.markdown("### CSO Model Settings")
amber_thresh = st.sidebar.slider(
    "Amber Threshold (% Gain)", 0.0, 2.0, 0.5, step=0.1
)
red_thresh = st.sidebar.slider(
    "Red Threshold (% Gain)", 1.0, 10.0, 2.0, step=0.5
)

st.sidebar.divider()
with st.sidebar.expander("ℹ️ Portal Disclaimer & Risk Definitions"):
    st.warning(
        "**Trial Version:** Values are hypothetical and for demonstration purposes only."
    )
    st.markdown("""
    **Monthly Mean Concentrations:**
    * 🟢 **Low Risk:** < 0.25 oocysts/L
    * 🟡 **Medium Risk:** 0.25 – 0.50 oocysts/L
    * 🔴 **High Risk:** > 0.50 oocysts/L
    """)

# Initialize active site in session state
site_list = sorted(df["Site"].unique())
if "selected_site" not in st.session_state:
    st.session_state["selected_site"] = site_list[0]

selected_site = st.session_state["selected_site"]

# ---------------------------------------------------------
# 5. Dashboard View (Interactive Map & Inspector)
# ---------------------------------------------------------
st.title("UK Cryptosporidium Catchment Risk Portal")
st.caption("Interactive Risk Predictions & Historical Water Intake Monitoring")

col_map, col_chart = st.columns([1.3, 1])

# Column 1: UK Catchments Overview Map
with col_map:
    st.subheader(f"Catchment Risk Levels ({selected_month})")
    st.caption("💡 *Click any catchment marker on the map to inspect its trend.*")

    risk_color_map = {
        "High": "#E63946",
        "Medium": "#FFB703",
        "Low": "#2A9D8F",
        "No Data": "#94A3B8",
    }

    if hasattr(px, "scatter_map"):
        fig_map = px.scatter_map(
            latest_df,
            lat="lat",
            lon="lon",
            color="predicted_risk_level",
            color_discrete_map=risk_color_map,
            category_orders={
                "predicted_risk_level": ["High", "Medium", "Low", "No Data"]
            },
            hover_name="Site",
            custom_data=["Site"],
            hover_data={
                "predicted_risk_level": True,
                "predicted_risk": ":.4f",
                "lat": False,
                "lon": False,
            },
            zoom=5,
            center={"lat": 55.0, "lon": -3.5},
            map_style="carto-positron",
        )
    else:
        fig_map = px.scatter_mapbox(
            latest_df,
            lat="lat",
            lon="lon",
            color="predicted_risk_level",
            color_discrete_map=risk_color_map,
            category_orders={
                "predicted_risk_level": ["High", "Medium", "Low", "No Data"]
            },
            hover_name="Site",
            custom_data=["Site"],
            hover_data={
                "predicted_risk_level": True,
                "predicted_risk": ":.4f",
                "lat": False,
                "lon": False,
            },
            zoom=5,
            center={"lat": 55.0, "lon": -3.5},
            mapbox_style="carto-positron",
        )

    fig_map.update_traces(
        marker={"size": 13, "opacity": 0.85},
        unselected={"marker": {"opacity": 0.85}},
        selected={"marker": {"opacity": 0.85}},
    )

    # Highlight currently selected site with a central black dot
    selected_row = latest_df[latest_df["Site"] == selected_site]
    if not selected_row.empty:
        ScatterTrace = go.Scattermap if hasattr(go, "Scattermap") else go.Scattermapbox
        fig_map.add_trace(
            ScatterTrace(
                lat=selected_row["lat"],
                lon=selected_row["lon"],
                mode="markers",
                marker=dict(size=5, color="black"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend_title_text="Predicted Risk Level",
    )

    map_event = st.plotly_chart(
        fig_map,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="map_plot",
    )

    if map_event and "selection" in map_event and map_event["selection"].get("points"):
        points = map_event["selection"]["points"]
        if points and "customdata" in points[0]:
            clicked_site = points[0]["customdata"][0]
            if clicked_site != st.session_state["selected_site"]:
                st.session_state["selected_site"] = clicked_site
                st.rerun()

# Column 2: Historical Trend Line Chart
with col_chart:
    st.subheader(f"Site Risk Inspector: {selected_site}")

    site_df = df[df["Site"] == selected_site].sort_values("YearMonth")

    fig_chart = go.Figure()
    fig_chart.add_trace(
        go.Scatter(
            x=site_df["YearMonth"],
            y=site_df["predicted_risk"],
            mode="lines+markers",
            name="Predicted Risk",
            line=dict(color="#2563EB", width=2.5),
        )
    )

    if "observed_risk" in site_df.columns:
        fig_chart.add_trace(
            go.Scatter(
                x=site_df["YearMonth"],
                y=site_df["observed_risk"],
                mode="lines+markers",
                name="Observed Risk",
                line=dict(color="#475569", width=2, dash="dot"),
            )
        )

    fig_chart.update_layout(
        title=dict(text=f"Risk Score Trend: {selected_site}", pad=dict(b=15)),
        xaxis_title="Year",
        yaxis_title="Mean Crypto conc (oocysts/L)",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(t=100, b=30, l=20, r=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )
    st.plotly_chart(fig_chart, use_container_width=True)

    latest_site_row = site_df[site_df["YearMonth"] == selected_month]
    if not latest_site_row.empty:
        curr_pred = latest_site_row["predicted_risk_level"].values[0]
        curr_model = (
            latest_site_row["Selected_Model"].values[0]
            if "Selected_Model" in latest_site_row.columns
            else "Adaptive"
        )
        st.info(
            f"**Latest Status ({selected_month}):** Risk Level = **{curr_pred}** | Selected Model = **{curr_model}**"
        )

# ---------------------------------------------------------
# 6. Upstream River Network & CSO Spatial Inspector
# ---------------------------------------------------------
st.divider()
st.subheader(f"🌊 Upstream Catchment & CSO Network: {selected_site}")
st.caption(
    "Interactive spatial tracing of river reaches and contributing Combined Sewer Overflows (CSOs) styled by XGBoost model gain contribution."
)

site_csos, river_lons, river_lats = get_site_spatial_data(
    selected_site, amber_thresh=amber_thresh, red_thresh=red_thresh
)

if river_lons or not site_csos.empty:
    fig_cso_map = go.Figure()

    # 1. Add River Reach Line Network
    if river_lons:
        fig_cso_map.add_trace(
            go.Scattermapbox(
                lon=river_lons,
                lat=river_lats,
                mode="lines",
                line=dict(width=2, color="#1D3557"),
                name="Upstream River Network",
                hoverinfo="none",
            )
        )

    # 2. Add CSO Markers Grouped by Importance Threshold
    cso_color_map = {
        "Red (High Impact)": "#E63946",
        "Amber (Moderate Impact)": "#FFB703",
        "Grey (Low/None)": "#8D99AE",
    }

    if not site_csos.empty:
        for level, df_group in site_csos.groupby("Impact_Level"):
            fig_cso_map.add_trace(
                go.Scattermapbox(
                    lon=df_group["lon"],
                    lat=df_group["lat"],
                    mode="markers",
                    marker=dict(
                        size=(
                            13
                            if "High" in level
                            else (10 if "Moderate" in level else 7)
                        ),
                        color=cso_color_map.get(level, "#8D99AE"),
                    ),
                    name=level,
                    text=(
                        "<b>"
                        + df_group["canonical_name"]
                        + "</b><br>"
                        + "Company: "
                        + df_group["company"].astype(str)
                        + "<br>"
                        + "CSO ID: "
                        + df_group["canonical_cso_id"]
                        + "<br>"
                        + "Gain Contribution: "
                        + df_group["Gain_Percent"].round(3).astype(str)
                        + "%"
                    ),
                    hoverinfo="text",
                )
            )

    # Determine Map Center
    if not site_csos.empty:
        c_lat, c_lon = site_csos["lat"].mean(), site_csos["lon"].mean()
    elif river_lats:
        valid_lats = [la for la in river_lats if la is not None]
        valid_lons = [lo for lo in river_lons if lo is not None]
        c_lat = sum(valid_lats) / len(valid_lats)
        c_lon = sum(valid_lons) / len(valid_lons)
    else:
        c_lat, c_lon = 55.0, -3.5

    fig_cso_map.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=c_lat, lon=c_lon),
            zoom=10,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=550,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    st.plotly_chart(fig_cso_map, use_container_width=True)

    # Optional Breakdown Table for Contributing CSOs
    if not site_csos.empty:
        with st.expander("📋 View Contributing CSO Feature Breakdown"):
            cso_display = site_csos[
                [
                    "canonical_cso_id",
                    "canonical_name",
                    "company",
                    "Gain_Percent",
                    "Impact_Level",
                ]
            ].sort_values("Gain_Percent", ascending=False)
            cso_display.columns = [
                "CSO ID",
                "CSO Station Name",
                "Water Company",
                "Model Gain %",
                "Impact Level",
            ]
            st.dataframe(cso_display, use_container_width=True)
else:
    st.info(
        f"No upstream network or CSO features found for intake site: **{selected_site}**."
    )