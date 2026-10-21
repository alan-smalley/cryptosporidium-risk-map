import os
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyproj
import re
import streamlit as st
from shapely import ops, wkb
from shapely.geometry import Point, mapping

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
INITIAL_SITE = "River_Thames_Laleham"

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
        meta_df[["Site", "Site_No", "Region", "Eastings", "Northings", "lat", "lon"]],
        on="Site",
        how="left",
    )

    return merged_df


def parse_gpkg_geom(blob):
    """Parses binary GeoPackage BLOB into Shapely geometry."""
    if blob is None:
        return None
    try:
        flags = blob[3]
        envelope_type = (flags >> 1) & 0x07
        env_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
        header_len = 8 + env_sizes.get(envelope_type, 0)
        return wkb.loads(blob[header_len:])
    except Exception:
        return None


def normalize_name(s):
    """Normalizes site names for case- and symbol-insensitive matching."""
    return str(s).lower().replace(" ", "_").replace("-", "_")


@st.cache_data
def load_catchment_polygons(site_meta_df):
    """Generates full regional catchment basin polygons for all intake sites."""
    features = []
    
    # Map normalized site names to canonical names and intake coordinates
    site_dict = {}
    for _, row in site_meta_df.iterrows():
        s_name = row["Site"]
        norm_s = normalize_name(s_name)
        if norm_s not in site_dict:
            site_dict[norm_s] = {
                "canonical_name": s_name,
                "eastings": row.get("Eastings", None),
                "northings": row.get("Northings", None),
            }

    # A. Load OSGB river reach network geometries from GPKG
    site_osgb_geoms = {}
    gpkg_net_file = "riversage_52_catchments__upstream_networks.gpkg"
    if os.path.exists(gpkg_net_file):
        try:
            conn = sqlite3.connect(gpkg_net_file)
            net_df = pd.read_sql_query(
                "SELECT target_name, geom FROM riversage_52_catchments__upstream_networks",
                conn,
            )
            conn.close()

            for _, row in net_df.iterrows():
                raw_target = row["target_name"]
                norm_target = normalize_name(raw_target)
                geom_blob = row["geom"]
                if geom_blob and norm_target in site_dict:
                    geom_osgb = parse_gpkg_geom(geom_blob)
                    if geom_osgb and not geom_osgb.is_empty:
                        if norm_target not in site_osgb_geoms:
                            site_osgb_geoms[norm_target] = []
                        site_osgb_geoms[norm_target].append(geom_osgb)
        except Exception:
            pass

    # B. Build full catchment area polygons
    for norm_s, s_info in site_dict.items():
        canonical_name = s_info["canonical_name"]
        catchment_osgb = None

        if norm_s in site_osgb_geoms and len(site_osgb_geoms[norm_s]) > 0:
            # Combine all upstream river lines for this catchment
            from shapely.geometry import GeometryCollection
            all_reaches = GeometryCollection(site_osgb_geoms[norm_s])
            
            # Convex hull + 1.5km catchment drainage buffer in meters
            catchment_osgb = all_reaches.convex_hull.buffer(1500)
        
        # Fallback if no network vectors exist: 5km radius regional catchment zone around intake
        if (catchment_osgb is None or catchment_osgb.is_empty) and s_info["eastings"] is not None:
            try:
                intake_pt = Point(float(s_info["eastings"]), float(s_info["northings"]))
                catchment_osgb = intake_pt.buffer(5000)  # 5km radius catchment area
            except Exception:
                catchment_osgb = None

        if catchment_osgb and not catchment_osgb.is_empty:
            # Transform OSGB meter coordinates to WGS84 lat/lon
            catchment_wgs84 = ops.transform(transformer.transform, catchment_osgb)
            
            # Simplify polygon for web rendering performance
            catchment_simple = catchment_wgs84.simplify(0.0005, preserve_topology=True)

            features.append(
                {
                    "type": "Feature",
                    "id": canonical_name,
                    "properties": {"Site": canonical_name},
                    "geometry": mapping(catchment_simple),
                }
            )

    if features:
        return {"type": "FeatureCollection", "features": features}
    return None


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
            if "CSO" in feat and not any(
                k in feat for k in ["count", "mean", "max", "min"]
            ):
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

        sample_df = pd.read_sql_query(
            "SELECT * FROM riversage_52_catchments__cso_sites LIMIT 1",
            conn_cso,
        )
        cso_cols = [
            "canonical_cso_id",
            "company",
            "canonical_name",
            "canonical_easting",
            "canonical_northing",
        ]
        if "permit_number" in sample_df.columns:
            cso_cols.append("permit_number")

        cso_df = pd.read_sql_query(
            f"SELECT {', '.join(cso_cols)} FROM riversage_52_catchments__cso_sites",
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
                f"CSO_{int(digits)}" if digits else "",
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
                if geom:
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


def calculate_map_center_and_zoom(lats, lons):
    """Calculates center lat/lon and zoom level to fit all network elements initially."""
    valid_lats = [la for la in lats if la is not None and not np.isnan(la)]
    valid_lons = [lo for lo in lons if lo is not None and not np.isnan(lo)]

    if not valid_lats or not valid_lons:
        return 55.0, -3.5, 9.5

    min_lat, max_lat = min(valid_lats), max(valid_lats)
    min_lon, max_lon = min(valid_lons), max(valid_lons)

    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    max_diff = max(max_lat - min_lat, max_lon - min_lon)

    if max_diff < 0.02:
        zoom = 13.0
    elif max_diff < 0.05:
        zoom = 12.0
    elif max_diff < 0.1:
        zoom = 11.0
    elif max_diff < 0.2:
        zoom = 10.0
    elif max_diff < 0.4:
        zoom = 9.2
    elif max_diff < 0.8:
        zoom = 8.5
    elif max_diff < 1.5:
        zoom = 7.5
    else:
        zoom = 6.5

    return center_lat, center_lon, zoom


df = load_and_prepare_data()

# Filter sites if a custom list is defined
all_available_sites = sorted(df["Site"].unique())
if CUSTOM_SITE_LIST and len(CUSTOM_SITE_LIST) > 0:
    norm_custom = [normalize_name(s) for s in CUSTOM_SITE_LIST]
    site_list = [
        s for s in all_available_sites if normalize_name(s) in norm_custom
    ]
    if not site_list:
        site_list = all_available_sites
else:
    site_list = all_available_sites

# Initialize session state for initial load site
if (
    "selected_site" not in st.session_state
    or st.session_state["selected_site"] not in site_list
):
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
chosen_site = st.sidebar.selectbox(
    "Select Target Intake Site:", site_list, index=site_idx
)
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

st.sidebar.markdown(f"🔴 **High Risk:** {high_count}  \n*(>0.50 oocysts/L)*\n")
st.sidebar.markdown(
    f"🟡 **Medium Risk:** {med_count}  \n*(0.25 - 0.50 oocysts/L)*\n"
)
st.sidebar.markdown(f"🟢 **Low Risk:** {low_count}  \n*(<0.25 oocysts/L)*\n")

st.sidebar.divider()
st.sidebar.markdown("### CSO & Map Settings")
amber_thresh = st.sidebar.slider(
    "Amber Threshold (% Gain)", 0.0, 2.0, 0.5, step=0.1
)
red_thresh = st.sidebar.slider(
    "Red Threshold (% Gain)", 1.0, 10.0, 2.0, step=0.5
)

show_terrain = st.sidebar.checkbox(
    "Enable Terrain / Topo Map Background", value=False
)
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

col_map, col_chart = st.columns([2.2, 1])

with col_map:
    st.subheader(f"Catchment Risk Levels ({selected_month})")
    st.caption(
        "💡 *Click any catchment area or marker on the map to inspect its trend.*"
    )

    risk_color_map = {
        "High": "#E63946",
        "Medium": "#FFB703",
        "Low": "#2A9D8F",
        "No Data": "#94A3B8",
    }

    category_order = ["High", "Medium", "Low", "No Data"]

    # Load catchment area polygons
    site_meta_unique = latest_df.drop_duplicates(subset=["Site"])
    catchment_geojson = load_catchment_polygons(site_meta_unique)

    if catchment_geojson and len(catchment_geojson["features"]) > 0:
        if hasattr(px, "choropleth_map"):
            fig_map = px.choropleth_map(
                latest_df,
                geojson=catchment_geojson,
                locations="Site",
                featureidkey="properties.Site",
                color="predicted_risk_level",
                color_discrete_map=risk_color_map,
                category_orders={"predicted_risk_level": category_order},
                hover_name="Site",
                custom_data=["Site"],
                hover_data={
                    "predicted_risk_level": True,
                    "predicted_risk": ":.4f",
                },
                zoom=5.2,
                center={"lat": 54.5, "lon": -3.0},
                map_style=base_map_style,
            )
        else:
            fig_map = px.choropleth_mapbox(
                latest_df,
                geojson=catchment_geojson,
                locations="Site",
                featureidkey="properties.Site",
                color="predicted_risk_level",
                color_discrete_map=risk_color_map,
                category_orders={"predicted_risk_level": category_order},
                hover_name="Site",
                custom_data=["Site"],
                hover_data={
                    "predicted_risk_level": True,
                    "predicted_risk": ":.4f",
                },
                zoom=5.2,
                center={"lat": 54.5, "lon": -3.0},
                mapbox_style=base_map_style,
            )

        # Overlay intake location markers on top of catchment polygons
        fig_map.add_trace(
            ScatterMapTrace(
                lat=latest_df["lat"],
                lon=latest_df["lon"],
                mode="markers",
                marker=dict(size=7, color="#1D3557"),
                customdata=latest_df[["Site"]].values,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
        # Fallback to scatter map
        if hasattr(px, "scatter_map"):
            fig_map = px.scatter_map(
                latest_df,
                lat="lat",
                lon="lon",
                color="predicted_risk_level",
                color_discrete_map=risk_color_map,
                category_orders={"predicted_risk_level": category_order},
                hover_name="Site",
                custom_data=["Site"],
                zoom=5.2,
                center={"lat": 54.5, "lon": -3.0},
                map_style=base_map_style,
            )
        else:
            fig_map = px.scatter_mapbox(
                latest_df,
                lat="lat",
                lon="lon",
                color="predicted_risk_level",
                color_discrete_map=risk_color_map,
                category_orders={"predicted_risk_level": category_order},
                hover_name="Site",
                custom_data=["Site"],
                zoom=5.2,
                center={"lat": 54.5, "lon": -3.0},
                mapbox_style=base_map_style,
            )

    selected_row = latest_df[latest_df["Site"] == selected_site]
    if not selected_row.empty:
        fig_map.add_trace(
            ScatterMapTrace(
                lat=selected_row["lat"],
                lon=selected_row["lon"],
                mode="markers",
                marker=dict(size=12, color="black"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig_map.update_layout(
        height=680,
        showlegend=False,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    map_event = st.plotly_chart(
        fig_map,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="map_plot",
    )

    if (
        map_event
        and "selection" in map_event
        and map_event["selection"].get("points")
    ):
        points = map_event["selection"]["points"]
        if points and "customdata" in points[0]:
            clicked_site = points[0]["customdata"][0]
            if clicked_site != st.session_state["selected_site"]:
                st.session_state["selected_site"] = clicked_site
                st.rerun()

with col_chart:
    st.subheader(f"Site Risk Inspector: {selected_site}")

    graph_type = st.radio(
        "Select Graph Display Mode:",
        ["Heatmap (Risk Levels)", "Line Chart (Concentrations)"],
        horizontal=True,
        key="graph_type_selector",
    )

    site_df = df[df["Site"] == selected_site].sort_values("YearMonth")

    # --- Chart Settings / Range Filter Expander ---
    with st.expander("⚙️ Chart Range Settings"):
        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            use_manual_x = st.checkbox(
                "Override Date Range (X-Axis)", value=False
            )
            if use_manual_x:
                cx1, cx2 = st.columns(2)
                x_min = cx1.text_input("Start Month", value="2018-01")
                x_max = cx2.text_input("End Month", value="2023-12")
                if x_min and x_max:
                    site_df = site_df[
                        (site_df["YearMonth"] >= x_min)
                        & (site_df["YearMonth"] <= x_max)
                    ]

        with col_opt2:
            if graph_type == "Line Chart (Concentrations)":
                use_manual_y = st.checkbox(
                    "Override Y-Axis Range", value=False
                )
                if use_manual_y:
                    cy1, cy2 = st.columns(2)
                    y_min = cy1.number_input("Y Min", value=0.0, step=0.05)
                    y_max = cy2.number_input("Y Max", value=1.0, step=0.1)
                else:
                    use_manual_y = False

    if graph_type == "Heatmap (Risk Levels)":
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

        z_matrix = [z_pred, z_obs]
        y_labels = ["Predicted Risk", "Observed Risk"]

        text_pred = [
            f"<b>Predicted Risk</b><br>Month: {m}<br>Level: {lvl}"
            for m, lvl in zip(x_months, site_df["predicted_risk_level"])
        ]
        text_obs = [
            f"<b>Observed Risk</b><br>Month: {m}<br>Level: {lvl}"
            for m, lvl in zip(x_months, obs_col)
        ]
        hover_matrix = [text_pred, text_obs]

        discrete_colorscale = [
            [0.00, "#94A3B8"],
            [0.25, "#94A3B8"],
            [0.25, "#2A9D8F"],
            [0.50, "#2A9D8F"],
            [0.50, "#FFB703"],
            [0.75, "#FFB703"],
            [0.75, "#E63946"],
            [1.00, "#E63946"],
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
                xgap=2,
                ygap=6,
            )
        )

        fig_chart.update_layout(
            title=dict(
                text=f"Risk Level Timeline: {selected_site}", pad=dict(b=15)
            ),
            xaxis=dict(title="Year / Month", type="category", tickangle=-45),
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=300,
            showlegend=False,
            margin=dict(t=60, b=50, l=20, r=20),
        )

    else:
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

        line_layout = dict(
            title=dict(
                text=f"Risk Score Trend: {selected_site}", pad=dict(b=15)
            ),
            xaxis_title="Year / Month",
            yaxis_title="Mean Crypto conc (oocysts/L)",
            hovermode="x unified",
            template="plotly_white",
            height=350,
            margin=dict(t=80, b=40, l=20, r=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        if "use_manual_y" in locals() and use_manual_y:
            line_layout["yaxis"] = dict(
                title="Mean Crypto conc (oocysts/L)", range=[y_min, y_max]
            )

        fig_chart.update_layout(**line_layout)

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
st.subheader(
    f"🌊 Upstream Catchment Network & Contributing CSOs: {selected_site}"
)
st.caption(
    "Interactive spatial view showing the intake location, upstream river channels, and High/Medium impact Combined Sewer Overflows (CSOs)."
)

site_csos, river_lons, river_lats = get_site_spatial_data(
    selected_site, amber_thresh=amber_thresh, red_thresh=red_thresh
)

# Filter site_csos to show BOTH High (red) AND Medium (amber) Impact CSOs (>= amber_thresh)
if not site_csos.empty:
    high_med_csos = site_csos[site_csos["Gain_Percent"] >= amber_thresh].copy()
else:
    high_med_csos = pd.DataFrame()

if river_lons or not high_med_csos.empty:
    fig_cso_map = go.Figure()

    # A. Upstream River Reach Vectors
    if river_lons:
        fig_cso_map.add_trace(
            ScatterMapTrace(
                lon=river_lons,
                lat=river_lats,
                mode="lines",
                line=dict(width=2.5, color="#1D3557"),
                name="Upstream River Network",
                hoverinfo="none",
            )
        )

    # B. Target Intake Location
    selected_row = latest_df[latest_df["Site"] == selected_site]
    intake_lat, intake_lon = None, None
    if not selected_row.empty:
        intake_lat = selected_row["lat"].values[0]
        intake_lon = selected_row["lon"].values[0]
        fig_cso_map.add_trace(
            ScatterMapTrace(
                lon=[intake_lon],
                lat=[intake_lat],
                mode="markers",
                marker=dict(size=14, color="#1D3557"),
                name="Water Intake",
                text=f"<b>Target Intake:</b> {selected_site}",
                hoverinfo="text",
            )
        )

    # C. High & Medium Impact CSOs with Permit Number displayed on hover
    cso_color_map = {
        "Red (High Impact)": "#E63946",
        "Amber (Moderate Impact)": "#FFB703",
    }

    if not high_med_csos.empty:
        for level, df_group in high_med_csos.groupby("Impact_Level"):
            permit_ids = (
                df_group["permit_number"].astype(str)
                if "permit_number" in df_group.columns
                else df_group["canonical_cso_id"].astype(str)
            )

            hover_txt = (
                "<b>Permit No / CSO ID: "
                + permit_ids
                + "</b><br>"
                + "Name: "
                + df_group["canonical_name"].astype(str)
                + "<br>"
                + "Company: "
                + df_group["company"].astype(str)
                + "<br>"
                + "Model Gain Impact: "
                + df_group["Gain_Percent"].round(3).astype(str)
                + "%"
            )

            fig_cso_map.add_trace(
                ScatterMapTrace(
                    lon=df_group["lon"],
                    lat=df_group["lat"],
                    mode="markers",
                    marker=dict(
                        size=13 if "High" in level else 10,
                        color=cso_color_map.get(level, "#FFB703"),
                    ),
                    name=level,
                    text=hover_txt,
                    hoverinfo="text",
                )
            )

    # Collect all coordinates to calculate initial bounding box
    all_lats = list(river_lats)
    all_lons = list(river_lons)
    if not high_med_csos.empty:
        all_lats.extend(high_med_csos["lat"].tolist())
        all_lons.extend(high_med_csos["lon"].tolist())
    if intake_lat is not None:
        all_lats.append(intake_lat)
        all_lons.append(intake_lon)

    c_lat, c_lon, initial_zoom = calculate_map_center_and_zoom(
        all_lats, all_lons
    )

    map_config = {
        "style": base_map_style,
        "center": dict(lat=c_lat, lon=c_lon),
        "zoom": initial_zoom,
    }

    # Zoomable layout settings with initial auto-fit view
    layout_kwargs = {
        MAP_DICT_KEY: map_config,
        "margin": dict(l=0, r=0, t=0, b=0),
        "height": 420,
        "legend": dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    }

    fig_cso_map.update_layout(**layout_kwargs)

    st.plotly_chart(
        fig_cso_map,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )

    if not high_med_csos.empty:
        with st.expander(
            "📋 View Contributing High & Medium Impact CSO Breakdown"
        ):
            cso_display = high_med_csos[
                [
                    "canonical_cso_id",
                    "canonical_name",
                    "company",
                    "Gain_Percent",
                    "Impact_Level",
                ]
            ].sort_values("Gain_Percent", ascending=False)
            cso_display.columns = [
                "Permit / CSO ID",
                "CSO Station Name",
                "Water Company",
                "Model Gain %",
                "Impact Level",
            ]
            st.dataframe(cso_display, use_container_width=True)
    else:
        st.info(
            f"ℹ️ Upstream river channels displayed. No high or medium impact CSOs (≥ {amber_thresh}% gain) identified for intake site: **{selected_site}**."
        )
else:
    st.info(
        f"No upstream network or CSO features found for intake site: **{selected_site}**."
    )