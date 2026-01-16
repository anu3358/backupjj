import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import google.generativeai as genai

from data_engine import load_thermal_data, get_city_data, get_baseline_stats
from ml_model import (
    apply_intervention_to_dataframe,
    get_intervention_summary,
    estimate_trees_required,
    estimate_paint_layers_and_color,
)
from agent_logic import get_ai_recommendation
from config import DEFAULT_CITIES

# ==========================================================
# GEMINI CONFIG
# ==========================================================
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    gemini_api_key = os.getenv("GEMINI_API_KEY")

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    gemini_api_key = None


@st.cache_resource
def get_available_gemini_model():
    if not gemini_api_key:
        return None
    try:
        for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                return model.name.replace("models/", "")
    except Exception:
        return None
    return None


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="AuraCool — Urban Heat AI",
    layout="wide",
    page_icon="🏙️",
)

st.title("🏙️ AuraCool — Urban Heat AI Optimizer")
st.caption("🌍 Real satellite data + AI-driven cooling strategies")

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.header("⚙️ Control Panel")

    city_name = st.selectbox(
        "📍 Select City",
        list(DEFAULT_CITIES.keys()),
        index=0,
    )

    st.markdown("### 🌱 Interventions")

    green_inc_pct = st.slider(
        "🌿 Increase Vegetation (%)",
        0, 60, 20, 5
    )

    refl_inc_pct = st.slider(
        "🏙️ Increase Roof Reflectivity (%)",
        0, 60, 15, 5
    )

    # 🔥 NEW — IMPACT STRENGTH SLIDER (KEY FIX)
    intervention_strength = st.slider(
        "⚡ Intervention Strength",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Scales real-world implementation intensity"
    )

    budget_musd = st.slider(
        "💰 Climate Budget (Million USD)",
        5, 200, 40, 10
    )

    if st.button("🔄 Refresh Analysis", use_container_width=True):
        st.rerun()


# ==========================================================
# MAIN LOGIC
# ==========================================================
try:
    df_city = get_city_data(city_name)
    baseline_stats = get_baseline_stats(city_name)

    # 🔥 APPLY STRENGTH SCALING HERE
    effective_green = int(green_inc_pct * intervention_strength)
    effective_refl = int(refl_inc_pct * intervention_strength)

    df_modified = apply_intervention_to_dataframe(
        df_city,
        effective_green,
        effective_refl,
    )

    summary = get_intervention_summary(df_city, df_modified)

    base_temp = summary["base_temp"]
    new_temp = summary["new_temp"]
    reduction = max(0.0, base_temp - new_temp)

    area_km2 = baseline_stats.get("area_km2", 50)

    estimated_trees = estimate_trees_required(area_km2, reduction)
    layers_needed, paint_color = estimate_paint_layers_and_color(reduction)

    # ==========================================================
    # TOP METRICS
    # ==========================================================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌡️ Baseline Temp", f"{base_temp:.1f}°C")
    col2.metric(
        "❄️ Optimized Temp",
        f"{new_temp:.1f}°C",
        delta=f"-{reduction:.1f}°C",
    )
    col3.metric("🔥 Heat Hotspots", f"{baseline_stats['hotspot_pct']:.1f}%")
    col4.metric("🏢 Urban Density", f"{baseline_stats['urban_density']:.2f}")

    # ==========================================================
    # TREES + PAINT
    # ==========================================================
    st.markdown("## 🌳 Cooling Infrastructure Estimate")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Estimated Trees Needed", f"{estimated_trees:,}")
        st.caption("≈1000 trees / km² / °C cooling")

    with c2:
        st.metric("Cool Roof Layers", layers_needed)
        st.write(f"Recommended color: **{paint_color}**")

    # ==========================================================
    # 3D MAP
    # ==========================================================
    st.markdown("## 🗺️ 3D Heat Map (Before vs After)")

    view_state = pdk.ViewState(
        latitude=df_city.latitude.mean(),
        longitude=df_city.longitude.mean(),
        zoom=11,
        pitch=50,
    )

    col_b, col_a = st.columns(2)

    with col_b:
        st.subheader("🔴 Before")
        layer_before = pdk.Layer(
            "HexagonLayer",
            df_city,
            get_position="[longitude, latitude]",
            get_elevation="temperature_c * 40",
            radius=120,
            extruded=True,
            coverage=0.9,
            get_fill_color="[255, 70, 70, 180]",
        )
        st.pydeck_chart(pdk.Deck([layer_before], view_state))

    with col_a:
        st.subheader("🟢 After")
        layer_after = pdk.Layer(
            "HexagonLayer",
            df_modified,
            get_position="[longitude, latitude]",
            get_elevation="temperature_c * 40",
            radius=120,
            extruded=True,
            coverage=0.9,
            get_fill_color="[70, 200, 120, 180]",
        )
        st.pydeck_chart(pdk.Deck([layer_after], view_state))

    # ==========================================================
    # AI STRATEGY
    # ==========================================================
    st.markdown("## 🤖 AI Strategy")

    report = get_ai_recommendation(
        city_name,
        base_temp,
        new_temp,
        reduction,
        budget_musd,
        baseline_stats["hotspot_pct"],
    )

    st.markdown(report)

except Exception as e:
    st.error("❌ Error loading application")
    st.exception(e)
