import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyproj
import re
import streamlit as st
from shapely import ops, wkb

# Detect Plotly version capabilities for map traces
if hasattr(go, "Scattermap"):
    ScatterMapTrace = go.Scattermap
    MAP_DICT_KEY = "map"
else:
    ScatterMapTrace = go.Scattermapbox
    MAP_DICT_KEY = "mapbox"

# ---------------------------------------------------------
# Site Selection & Initial Load Configuration
# ---------------------------------------------------------
# Default site that loads first when opening the portal
INITIAL_SITE = "River_Thames_Laleham"

# Toggle custom filtering:
# Set to True to restrict the app to SITES_TO_DISPLAY.
# Set to False to display ALL available sites in the dataset.
ENABLE_CUSTOM_SITE_FILTER = False

SITES_TO_DISPLAY = [
    "River_Deveron_TurriffMuiresk",
    "River_Ugie_Forehill",
    "River_Deveron_CairnfordBridgeHuntly",
    "Allt_a_Choire_Burn_Tomnavoulin",
    "River_Dee_Invercannie",
    "River_Tay_Perth",
    "River_Wear_Lumley",
    "River_Gelt",
    "River_Eden",
    "River_Duddon",
    "River_Dee_3",
    "River_Dee_2",
    "River_Esk_Whitby",
    "River_Ure_Masham",
    "River_Ouse_York",
    "River_Hull_Driffield",
    "River_Derwent_Elvington_2",
    "River_Derwent_Wressle",
    "Afon_Ddu_BrynCowlyd",
    "River_Dee_1",
    "River_Teifi_Llechryd",
    "River_Severn_Shelton",
    "River_Severn_TrimpleyWTW",
    "River_Bure_Belaugh",
    "River_Wensum_Costessey",
    "River_Nar_Marham",
    "River_Wensum_Heigham",
    "River_Wissey_StokeFerry",
    "River_Waveney",
    "River_Great_Ouse_GrafhamSystem",
    "River_Great_Ouse_Bedford",
    "River_Stour_Langham",
    "River_Blackwater_Langford",
    "River_Lee_NewGauge",
    "River_Chelmer_Langford",
    "River_Lee_Girling",
    "River_Cherwell_Grimsbury",
    "River_Great_Ouse_Foxcote",
    "River_Thames_Farmoor",
    "River_Thames_Datchet",
    "River_Thames_HytheEnd",
    "River_Thames_Walton",
    "River_Thames_Laleham",
    "RIver_Kennet_Fobney",
    "River_Tillingbourne_Shalford",
    "River_Wey_Shalford",
    "River_Rother_Robertsbridge",
    "River_Itchen_Otterbourne",
    "River_Rother_Hardham",
    "River_Test_Testwood",
    "River_Exe_AllersWTW",
    "RIver_Dart_Littlehempston",
]

CUSTOM_SITE_LIST = SITES_TO_DISPLAY if ENABLE_CUSTOM_SITE_FILTER else None

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
# 2. Page Configuration & Coordinate Transformer
# ---------------------------------------------------------
st.set_page_config(
    page_title="UK Cryptosporidium Risk Map",
    layout="wide",
    initial_sidebar_state="expanded",
)

transformer = pyproj.Transformer.from_crs(
    "EPSG:27700", "EPSG:4326", always_xy=True
)

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

    meta_df["lon"], meta_df["lat"] = transformer.transform(
        meta_df["Eastings"].values, meta_df["Northings"].values
    )

    pred_df["predicted_risk_level"] = pred_df["predicted_risk_level"].fillna(
        "No Data"
    )
    if "observed_risk_level" in pred_df.columns:
        pred_df["observed_risk_level"] = pred_df["observed_risk_level"].fillna(
            "No Data"
        )

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


def normalize_name(s):
    """Normalizes site names for case- and symbol-insensitive matching."""
    return str(s).lower().replace(" ", "_").replace("-", "_")


@st.cache_data
def get_site_spatial_data(site_name, amber_thresh=0.5, red_thresh=2.0):
    """Loads upstream river network vectors and scored CSO sites for a target intake."""
    norm_site_name = normalize_name(site_name)

    # A. Feature Importance Parsing
    cso_importance = {}
    try:
        df_feat = pd.read_csv("all_site_feature_importance.csv")
        df_feat["norm_site"] = df_feat["Site"].apply(normalize_name)
        site_feats = df_feat[df_feat["norm_site"] == norm_site_name]

        for _, row in site_feats.iterrows():
            feat = str(row["Feature"])
            if "CSO" in feat and not any(k in feat for k in ["count", "mean", "max", "min"]):
                gain_val = float(row["Mean_Gain_Percent"])
                
                digits = re.sub(r"\D", "", feat)
                if digits:
                    cso_importance[digits] = gain_val
                    cso_importance[str(int(digits))] = gain_val
                    cso_importance[f"CSO_{digits}"] = gain_val
                    cso_importance[f"CSO_{int(digits)}"] = gain_val
                    cso_importance[feat] = gain_val
    except Exception as e:
        st.warning(f"Feature importance load warning: {e}")
        cso_importance = {}

    # B. CSO Geometries from GeoPackage
    site_csos = pd.DataFrame()
    try:
        conn_cso = sqlite3.connect("riversage_52_catchments__cso_sites.gpkg")
        cso_df = pd.read_sql_query(
            "SELECT canonical_cso_id, company, canonical_name, canonical_easting, canonical_northing FROM riversage_52_catchments__cso_sites",
            conn_cso,
        )
        conn_cso.close()

        def find_gain(raw_id):
            s_id = str(raw_id).strip()
            digits = re.sub(r"\D", "", s_id)
            candidates = [
                s_id, 
                digits, 
                str(int(digits)) if digits else "", 
                f"CSO_{digits}", 
                f"CSO_{int(digits)}" if digits else ""
            ]
            for candidate in candidates:
                if candidate in cso_importance:
                    return cso_importance[candidate]
            return None

        cso_df["Gain_Percent"] = cso_df["canonical_cso_id"].apply(find_gain)
        site_csos = cso_df[cso_df["Gain_Percent"].notna()].copy()

        if not site_csos.empty:
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
    except Exception as e:
        st.warning(f"CSO GPKG load warning: {e}")

    # C. Upstream River Reach Vectors
    river_lons, river_lats = [], []
    try:
        conn_net = sqlite3.connect(
            "riversage_52_catchments__upstream_networks.gpkg"
        )
        net_df = pd.read_sql_query(
            "SELECT target_name, geom FROM riversage_52_catchments__upstream_networks",
            conn_net,
        )
        conn_net.close()

        net_df["norm_target"] = net_df["target_name"].apply(normalize_name)
        matched_net = net_df[net_df["norm_target"] == norm_site_name]

        for geom_blob in matched_net["geom"]:
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
    except Exception as e:
        st.warning(f"River network load warning: {e}")

    return site_csos, river_lons, river_lats


df = load_and_prepare_data()

# Filter sites if a custom list is defined
all_available_sites = sorted(df["Site"].unique())
if CUSTOM_SITE_LIST and len(CUSTOM_SITE_LIST) > 0:
    norm_custom = [normalize_name(s) for s in CUSTOM_SITE_LIST]
    site_list = [s for s in all_available_sites if normalize_name(s) in norm_custom]
    if not site_list:
        site_list = all_available_sites
else:
    site_list = all_available_sites

# Initialize session state for initial load site
if "selected_site" not in st.session_state or st.session_state["selected_site"] not in site_list:
    if INITIAL_SITE in site_list:
        st.session_state["selected_site"] = INITIAL_SITE
    else:
        st.session_state["selected_site"] = site_list[0]

# ---------------------------------------------------------
# 4. Sidebar Controls & Metrics
# ---------------------------------------------------------
st.sidebar.title("Navigation & Filters")

# Site Selectbox Sync
site_idx = site_list.index(st.session_state["selected_site"])
chosen_site = st.sidebar.selectbox("Select Target Intake Site:", site_list, index=site_idx)
if chosen_site != st.session_state["selected_site"]:
    st.session_state["selected_site"] = chosen_site
    st.rerun()

selected_site = st.session_state["selected_site"]

filtered_df = df[df["Site"].isin(site_list)].copy()

available_months = sorted(filtered_df["YearMonth"].unique(), reverse=True)
selected_month = st.sidebar.selectbox(
    "Select Model Calendar Month:", available_months
)

latest_df = filtered_df[filtered_df["YearMonth"] == selected_month].copy()

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
st.sidebar.markdown("### CSO & Map Settings")
amber_thresh = st.sidebar.slider(
    "Amber Threshold (% Gain)", 0.0, 2.0, 0.5, step=0.1
)
red_thresh = st.sidebar.slider(
    "Red Threshold (% Gain)", 1.0, 10.0, 2.0, step=0.5
)

show_terrain = st.sidebar.checkbox("Enable Terrain / Topo Map Background", value=False)
base_map_style = "open-street-map" if show_terrain else "carto-positron"

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

# ---------------------------------------------------------
# 5. Dashboard View (Interactive Map & Inspector)
# ---------------------------------------------------------
st.title("UK Cryptosporidium Catchment Risk Portal")
st.caption("Interactive Risk Predictions & Historical Water Intake Monitoring")

col_map, col_chart = st.columns([1.3, 1])

with col_map:
    st.subheader(f"Catchment Risk Levels ({selected_month})")
    st.caption("💡 *Click any catchment marker on the map or select from the sidebar.*")

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
            map_style=base_map_style,
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
            mapbox_style=base_map_style,
        )

    fig_map.update_traces(
        marker={"size": 13, "opacity": 0.85},
        unselected={"marker": {"opacity": 0.85}},
        selected={"marker": {"opacity": 0.85}},
    )

    selected_row = latest_df[latest_df["Site"] == selected_site]
    if not selected_row.empty:
        fig_map.add_trace(
            ScatterMapTrace(
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

with col_chart:
    st.subheader(f"Site Risk Inspector: {selected_site}")

    site_df = df[df["Site"] == selected_site].sort_values("YearMonth")

    # --- Date Range Override Expander ---
    with st.expander("⚙️ Chart Settings (Filter Date Range)"):
        use_manual_x = st.checkbox("Override Date Range (X-Axis)", value=False)
        if use_manual_x:
            cx1, cx2 = st.columns(2)
            x_min = cx1.text_input("Start Month (YYYY-MM)", value="2018-01")
            x_max = cx2.text_input("End Month (YYYY-MM)", value="2023-12")
            if x_min and x_max:
                site_df = site_df[
                    (site_df["YearMonth"] >= x_min)
                    & (site_df["YearMonth"] <= x_max)
                ]

    # Map risk levels to categorical numerical values (0 to 3) for heatmap matrix
    risk_num_map = {"No Data": 0, "Low": 1, "Medium": 2, "High": 3}

    x_months = site_df["YearMonth"].tolist()

    z_pred = [
        risk_num_map.get(lvl, 0) for lvl in site_df["predicted_risk_level"]
    ]
    obs_col = (
        site_df["observed_risk_level"]
        if "observed_risk_level" in site_df.columns
        else ["No Data"] * len(site_df)
    )
    z_obs = [risk_num_map.get(lvl, 0) for lvl in obs_col]

    # 2 rows: Top = Predicted Risk, Bottom = Observed Risk
    z_matrix = [z_pred, z_obs]
    y_labels = ["Predicted Risk", "Observed Risk"]

    # Interactive hover labels
    text_pred = [
        f"<b>Predicted Risk</b><br>Month: {m}<br>Level: {lvl}"
        for m, lvl in zip(x_months, site_df["predicted_risk_level"])
    ]
    text_obs = [
        f"<b>Observed Risk</b><br>Month: {m}<br>Level: {lvl}"
        for m, lvl in zip(x_months, obs_col)
    ]
    hover_matrix = [text_pred, text_obs]

    # Color steps: 0 -> Grey, 1 -> Green, 2 -> Amber, 3 -> Red
    discrete_colorscale = [
        [0.00, "#94A3B8"],
        [0.25, "#94A3B8"],  # No Data (Grey)
        [0.25, "#2A9D8F"],
        [0.50, "#2A9D8F"],  # Low Risk (Green)
        [0.50, "#FFB703"],
        [0.75, "#FFB703"],  # Medium Risk (Amber)
        [0.75, "#E63946"],
        [1.00, "#E63946"],  # High Risk (Red)
    ]

    fig_chart = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=x_months,
            y=y_labels,
            hoverinfo="text",
            text=hover_matrix,
            colorscale=discrete_colorscale,
            zmin=0,
            zmax=3,
            showscale=False,
            xgap=2,  # Tile separation
            ygap=6,
        )
    )

    # Add explicit legend items matching map colors
    legend_items = [
        ("Low Risk (🟢)", "#2A9D8F"),
        ("Medium Risk (🟡)", "#FFB703"),
        ("High Risk (🔴)", "#E63946"),
        ("No Data (⚪)", "#94A3B8"),
    ]

    for label, color in legend_items:
        fig_chart.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=label,
                showlegend=True,
            )
        )

    fig_chart.update_layout(
        title=dict(text=f"Risk Level Timeline: {selected_site}", pad=dict(b=15)),
        xaxis=dict(title="Year / Month", type="category", tickangle=-45),
        yaxis=dict(autorange="reversed"),  # Predicted on top, Observed below
        template="plotly_white",
        height=320,
        margin=dict(t=80, b=50, l=20, r=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1
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

    if river_lons:
        fig_cso_map.add_trace(
            ScatterMapTrace(
                lon=river_lons,
                lat=river_lats,
                mode="lines",
                line=dict(width=2, color="#1D3557"),
                name="Upstream River Network",
                hoverinfo="none",
            )
        )

    cso_color_map = {
        "Red (High Impact)": "#E63946",
        "Amber (Moderate Impact)": "#FFB703",
        "Grey (Low/None)": "#8D99AE",
    }

    if not site_csos.empty:
        for level, df_group in site_csos.groupby("Impact_Level"):
            fig_cso_map.add_trace(
                ScatterMapTrace(
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
                        + df_group["canonical_cso_id"].astype(str)
                        + "<br>"
                        + "Gain Contribution: "
                        + df_group["Gain_Percent"].round(3).astype(str)
                        + "%"
                    ),
                    hoverinfo="text",
                )
            )

    if not site_csos.empty:
        c_lat, c_lon = site_csos["lat"].mean(), site_csos["lon"].mean()
    elif river_lats:
        valid_lats = [la for la in river_lats if la is not None]
        valid_lons = [lo for lo in river_lons if lo is not None]
        c_lat = sum(valid_lats) / len(valid_lats)
        c_lon = sum(valid_lons) / len(valid_lons)
    else:
        c_lat, c_lon = 55.0, -3.5

    map_config = {
        "style": base_map_style,
        "center": dict(lat=c_lat, lon=c_lon),
        "zoom": 10,
    }

    layout_kwargs = {
        MAP_DICT_KEY: map_config,
        "margin": dict(l=0, r=0, t=0, b=0),
        "height": 550,
        "legend": dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    }

    fig_cso_map.update_layout(**layout_kwargs)

    st.plotly_chart(fig_cso_map, use_container_width=True)

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
            f"ℹ️ Upstream river channels displayed. No upstream CSOs identified for intake site: **{selected_site}**."
        )
else:
    st.info(
        f"No upstream network or CSO features found for intake site: **{selected_site}**."
    )