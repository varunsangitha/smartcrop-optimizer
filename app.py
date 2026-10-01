"""
SmartCrop Optimizer - Streamlit Web Dashboard
AI-Based Crop & Agricultural Resource Optimization System
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.recommendation import (
    recommend_crops,
    estimate_crop_yield,
    estimate_crop_price,
    get_crop_profile,
    run_farm_planning_pipeline,
    load_models,
    BENCHMARK_YIELDS_Q_PER_ACRE
)
from backend.resource_calculator import (
    get_resource_requirements,
    calculate_cultivation_cost,
    calculate_water_requirement,
    calculate_fertilizer_requirement,
    calculate_expected_revenue,
    calculate_expected_profit,
    get_all_supported_crops,
    RESOURCE_REQUIREMENTS
)
from backend.optimizer import optimize_crop_allocation
from utils.helpers import (
    format_currency,
    format_acres,
    create_land_allocation_chart,
    create_profit_chart,
    create_resource_utilization_chart,
    create_revenue_vs_cost_chart,
    get_text
)

# Page configuration
st.set_page_config(
    page_title="SmartCrop Optimizer | AI Farm Planning",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern farmer-friendly aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #065f46 0%, #047857 50%, #059669 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(6, 95, 70, 0.15);
    }
    .main-header h1 {
        color: white !important;
        font-weight: 700;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #d1fae5 !important;
        font-size: 1.05rem;
        margin: 6px 0 0 0;
    }

    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .kpi-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .kpi-sub {
        color: #10b981;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 4px;
    }

    .crop-badge {
        display: inline-block;
        background: #ecfdf5;
        color: #047857;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid #a7f3d0;
        margin-right: 6px;
    }

    .explanation-box {
        background: #f8fafc;
        border-left: 4px solid #10b981;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-optimal {
        background: #dcfce7;
        color: #15803d;
    }
    .status-warning {
        background: #fef3c7;
        color: #b45309;
    }
</style>
""", unsafe_allow_html=True)

# Language Selection
language = st.sidebar.selectbox("🌐 Language / భాష", ["English", "తెలుగు (Telugu)"], index=0)
lang_code = "te" if "తెలుగు" in language else "en"

# Sidebar: Farmer Input Panel
st.sidebar.markdown(f"### {get_text('soil_weather_header', lang_code)}")

st.sidebar.markdown("##### 🧪 Soil Parameters")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    n_input = st.number_input("Nitrogen (N)", min_value=0.0, max_value=250.0, value=90.0, step=1.0, help="Soil Nitrogen content (ratio/ppm)")
    k_input = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=43.0, step=1.0, help="Soil Potassium content (ratio/ppm)")
with col_s2:
    p_input = st.number_input("Phosphorus (P)", min_value=0.0, max_value=250.0, value=42.0, step=1.0, help="Soil Phosphorus content (ratio/ppm)")
    ph_input = st.number_input("pH Level", min_value=3.0, max_value=11.0, value=6.5, step=0.1, help="Soil pH (acidity/alkalinity)")

st.sidebar.markdown("##### 🌦️ Weather Parameters")
col_w1, col_w2 = st.sidebar.columns(2)
with col_w1:
    temp_input = st.number_input("Temperature (°C)", min_value=5.0, max_value=55.0, value=25.0, step=0.5)
    rainfall_input = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1200.0, value=200.0, step=10.0)
with col_w2:
    humidity_input = st.number_input("Humidity (%)", min_value=5.0, max_value=100.0, value=80.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### {get_text('farm_resources_header', lang_code)}")

land_input = st.sidebar.number_input("Available Land (Acres)", min_value=0.5, max_value=500.0, value=5.0, step=0.5, help="Total cultivable land")
water_input = st.sidebar.number_input("Available Water (Units / m³)", min_value=100.0, max_value=100000.0, value=4000.0, step=100.0, help="Irrigation water supply available")
fert_input = st.sidebar.number_input("Available Fertilizer (kg)", min_value=10.0, max_value=50000.0, value=500.0, step=25.0, help="Total fertilizer stock available")
budget_input = st.sidebar.number_input("Available Budget (₹ INR)", min_value=1000.0, max_value=10000000.0, value=100000.0, step=5000.0, help="Total financial budget for the season")

st.sidebar.markdown("##### 📍 Regional Context (Optional)")
state_options = [
    "Andhra Pradesh", "Telangana", "Karnataka", "Tamil Nadu", "Maharashtra",
    "Gujarat", "Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh",
    "Bihar", "West Bengal", "Odisha", "Rajasthan", "Assam"
]
season_options = ["Kharif", "Rabi", "Summer", "Whole Year", "Autumn", "Winter"]

col_reg1, col_reg2 = st.sidebar.columns(2)
with col_reg1:
    state_input = st.selectbox("State", state_options, index=0)
with col_reg2:
    season_input = st.selectbox("Season", season_options, index=0)

# Main Page Header
title_text = get_text("title", lang_code)
sub_text = get_text("subtitle", lang_code)
st.markdown(f"""
<div class="main-header">
    <h1>🌱 {title_text}</h1>
    <p>{sub_text}</p>
</div>
""", unsafe_allow_html=True)

# Load Models Safely
try:
    crop_model, yield_data, price_data = load_models()
    models_ready = True
except Exception as e:
    st.error(f"⚠️ Error loading ML models: {e}. Please ensure model pickle files exist in the `models/` directory.")
    models_ready = False

if models_ready:
    # -------------------------------------------------------------
    # 1. AI CROP RECOMMENDATION
    # -------------------------------------------------------------
    try:
        rec_data = recommend_crops(
            N=n_input,
            P=p_input,
            K=k_input,
            temperature=temp_input,
            humidity=humidity_input,
            ph=ph_input,
            rainfall=rainfall_input,
            top_k=4
        )
    except Exception as e:
        st.error(f"Error evaluating crop recommendation: {e}")
        rec_data = {
            "recommended_crop": "rice",
            "confidence": 50.0,
            "candidates": [{"crop": "rice", "confidence": 50.0}],
            "explanation": "Default fallback recommendation."
        }

    rec_crop = rec_data["recommended_crop"]
    rec_conf = rec_data["confidence"]
    top_candidates = rec_data["candidates"]

    # Recommendation Summary Card
    st.markdown("### 🌾 1. AI Crop Recommendation")
    c_rec1, c_rec2, c_rec3 = st.columns([1.5, 1.2, 1.3])

    with c_rec1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 5px solid #10b981;">
            <div class="kpi-title">{get_text('recommended_crop', lang_code)}</div>
            <div class="kpi-value" style="color: #065f46;">{rec_crop.title()}</div>
            <div class="kpi-sub">Confidence Score: <b>{rec_conf}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    # Estimate Single Crop Profile for top recommendation
    top_profile = get_crop_profile(rec_crop, annual_rainfall=rainfall_input, state=state_input, season=season_input)

    with c_rec2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{get_text('yield_prediction', lang_code)}</div>
            <div class="kpi-value">{top_profile['yield']} <span style="font-size:1rem;font-weight:500;color:#64748b;">Q/Acre</span></div>
            <div class="kpi-sub">Tonnes/Acre: ~{top_profile['yield']/10.0:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c_rec3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{get_text('market_price', lang_code)}</div>
            <div class="kpi-value">{format_currency(top_profile['price'])} <span style="font-size:1rem;font-weight:500;color:#64748b;">/ Q</span></div>
            <div class="kpi-sub">🏷️ <i>Estimated market value</i></div>
        </div>
        """, unsafe_allow_html=True)

    # Agronomic Explanation
    st.markdown(f"""
    <div class="explanation-box">
        <b>💡 Agronomic Assessment:</b> {rec_data['explanation']}
        <br><span style="color:#64748b; font-size:0.85rem;">Recommended because the provided soil and weather conditions are most similar to the learned conditions for this crop in historical trials.</span>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. RESOURCE REQUIREMENTS & CANDIDATE CROPS
    # -------------------------------------------------------------
    st.markdown("### 📋 2. Candidate Crops & Resource Benchmark")
    st.caption("Review expected economics and per-acre inputs. *Note: Values are planning estimates / demo assumptions.*")

    # Crop Selection for Optimizer
    available_candidate_names = [c["crop"] for c in top_candidates]
    # Add any extra popular crops if farmer wants to compare
    for extra in ["rice", "maize", "cotton", "banana", "jute"]:
        if extra not in available_candidate_names:
            available_candidate_names.append(extra)

    candidate_selection = st.multiselect(
        "Select candidate crop(s) for farm optimization:",
        options=available_candidate_names,
        default=[c["crop"] for c in top_candidates[:2]] if len(top_candidates) >= 2 else [rec_crop],
        format_func=lambda x: f"{x.title()} (Suitability: {[c['confidence'] for c in top_candidates if c['crop']==x][0]}%)" if any(c['crop']==x for c in top_candidates) else x.title()
    )

    if not candidate_selection:
        candidate_selection = [rec_crop]

    # Build profiles for candidate crops
    candidate_profiles = {}
    candidate_rows = []
    for c_name in candidate_selection:
        prof = get_crop_profile(c_name, annual_rainfall=rainfall_input, state=state_input, season=season_input)
        candidate_profiles[c_name] = prof
        candidate_rows.append({
            "Crop": prof["crop_display"],
            "Expected Yield (Q/Acre)": prof["yield"],
            "Estimated Price (₹/Q)": format_currency(prof["price"]),
            "Water Req (Units/Acre)": prof["water"],
            "Fertilizer Req (kg/Acre)": prof["fertilizer"],
            "Cultivation Cost (₹/Acre)": format_currency(prof["cost"]),
            "Expected Revenue (₹/Acre)": format_currency(prof["revenue_per_acre"]),
            "Expected Profit (₹/Acre)": format_currency(prof["profit_per_acre"])
        })

    candidate_df = pd.DataFrame(candidate_rows)
    st.dataframe(candidate_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 3. PU LP FARM OPTIMIZATION
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚙️ 3. Optimal Farm Resource Allocation")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        optimize_clicked = st.button("🚀 " + get_text("optimize_btn", lang_code), type="primary", use_container_width=True)

    with col_info:
        st.markdown(
            "Solves constrained linear programming: **Max Total Profit** subject to available **Land**, **Water**, **Fertilizer**, and **Budget** constraints."
        )

    # Run Optimization
    opt_result = optimize_crop_allocation(
        candidate_crops=candidate_profiles,
        available_land=land_input,
        available_water=water_input,
        available_fertilizer=fert_input,
        available_budget=budget_input
    )

    if opt_result["success"]:
        # KPI Cards Section
        st.markdown("#### 📊 Farm Plan Financial & Operational KPIs")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #10b981;">
                <div class="kpi-title">{get_text('total_profit', lang_code)}</div>
                <div class="kpi-value" style="color: #047857;">{format_currency(opt_result['total_profit'])}</div>
                <div class="kpi-sub">Net Earnings</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #06b6d4;">
                <div class="kpi-title">{get_text('total_revenue', lang_code)}</div>
                <div class="kpi-value">{format_currency(opt_result['total_revenue'])}</div>
                <div class="kpi-sub">Gross Output Value</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #f59e0b;">
                <div class="kpi-title">{get_text('total_cost', lang_code)}</div>
                <div class="kpi-value">{format_currency(opt_result['total_cost'])}</div>
                <div class="kpi-sub">Budget Used: {opt_result['utilization']['budget_pct']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #8b5cf6;">
                <div class="kpi-title">{get_text('land_used', lang_code)}</div>
                <div class="kpi-value">{opt_result['total_land_used']} <span style="font-size:1rem;color:#64748b;">/ {land_input} Ac</span></div>
                <div class="kpi-sub">Land Used: {opt_result['utilization']['land_pct']}%</div>
            </div>
            """, unsafe_allow_html=True)

        kpi_sub1, kpi_sub2, kpi_sub3, kpi_sub4 = st.columns(4)
        with kpi_sub1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Water Utilized</div>
                <div class="kpi-value" style="font-size:1.3rem;">{opt_result['total_water_used']:,.1f} <span style="font-size:0.85rem;color:#64748b;">/ {water_input:,.0f}</span></div>
                <div class="kpi-sub">Used: {opt_result['utilization']['water_pct']}% (Unused: {opt_result['unused_resources']['water']:,.1f})</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_sub2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Fertilizer Utilized</div>
                <div class="kpi-value" style="font-size:1.3rem;">{opt_result['total_fertilizer_used']:,.1f} <span style="font-size:0.85rem;color:#64748b;">kg / {fert_input:,.0f} kg</span></div>
                <div class="kpi-sub">Used: {opt_result['utilization']['fertilizer_pct']}% (Unused: {opt_result['unused_resources']['fertilizer']:,.1f} kg)</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_sub3:
            total_production_q = sum(item.get("Total Yield (Q)", 0.0) for item in opt_result["crop_details"])
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Expected Production</div>
                <div class="kpi-value" style="font-size:1.3rem;">{total_production_q:,.1f} <span style="font-size:0.85rem;color:#64748b;">Quintals</span></div>
                <div class="kpi-sub">~{total_production_q/10.0:,.1f} Metric Tonnes</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_sub4:
            limiting = opt_result["limiting_resources"]
            limiting_text = ", ".join(limiting) if limiting else "None (Balanced)"
            color = "#ef4444" if limiting else "#10b981"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Limiting Bottleneck</div>
                <div class="kpi-value" style="font-size:1.2rem; color:{color};">{limiting_text}</div>
                <div class="kpi-sub">Reached Capacity Limit</div>
            </div>
            """, unsafe_allow_html=True)

        # Optimal Allocation Table
        st.markdown("#### 🗺️ Optimal Crop Allocation Table")
        disp_details = []
        for item in opt_result["crop_details"]:
            disp_details.append({
                "Crop": item["Crop"],
                "Land (Acres)": f"{item['Allocated Land (Acres)']:.3f}",
                "Expected Yield (Q)": f"{item['Total Yield (Q)']:,.2f}",
                "Price (₹/Q)": format_currency(item["Price (₹/Q)"]),
                "Cultivation Cost (₹)": format_currency(item["Cultivation Cost (₹)"]),
                "Expected Revenue (₹)": format_currency(item["Expected Revenue (₹)"]),
                "Expected Profit (₹)": format_currency(item["Expected Profit (₹)"]),
                "Water Used": f"{item['Water Used']:,.1f}",
                "Fertilizer Used (kg)": f"{item['Fertilizer Used (kg)']:,.1f}"
            })
        st.dataframe(pd.DataFrame(disp_details), use_container_width=True, hide_index=True)

        # -------------------------------------------------------------
        # 4. CHARTS & VISUALIZATIONS
        # -------------------------------------------------------------
        st.markdown("#### 📈 Visual Analysis")
        ch_col1, ch_col2 = st.columns(2)

        with ch_col1:
            fig_land = create_land_allocation_chart(opt_result["crop_details"], opt_result["unused_resources"]["land"])
            st.plotly_chart(fig_land, use_container_width=True)

        with ch_col2:
            fig_util = create_resource_utilization_chart(opt_result["utilization"])
            st.plotly_chart(fig_util, use_container_width=True)

        ch_col3, ch_col4 = st.columns(2)
        with ch_col3:
            fig_profit = create_profit_chart(opt_result["crop_details"])
            st.plotly_chart(fig_profit, use_container_width=True)

        with ch_col4:
            fig_rev_cost = create_revenue_vs_cost_chart(opt_result["crop_details"])
            st.plotly_chart(fig_rev_cost, use_container_width=True)

        # -------------------------------------------------------------
        # 5. WHY THIS PLAN? EXPLANATION SECTION
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown(f"### ❓ 4. {get_text('why_this_plan', lang_code)}")

        with st.expander("🔍 Click to view step-by-step reasoning for this allocation", expanded=True):
            st.markdown(f"""
            - **Candidate Crops Considered**: {', '.join([c.title() for c in candidate_selection])}.
            - **Limiting Resource Constraint**: **{', '.join(opt_result['limiting_resources']) if opt_result['limiting_resources'] else 'None (All resources sufficient)'}**.
            - **How the Optimizer Allocated Land**:
              Linear programming evaluates the marginal profit per unit of the most restricted resource. When a constraint (e.g. water or budget) binds, land is assigned to crops that deliver the highest financial return per unit of that scarce resource.
            - **Unused Resource Buffer**:
              - Remaining Unallocated Land: **{opt_result['unused_resources']['land']} Acres**
              - Surplus Water: **{opt_result['unused_resources']['water']} Units**
              - Surplus Fertilizer: **{opt_result['unused_resources']['fertilizer']} kg**
              - Surplus Budget: **{format_currency(opt_result['unused_resources']['budget'])}**
            """)

        # -------------------------------------------------------------
        # 6. WHAT-IF SCENARIO ANALYSIS
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown(f"### 🔄 5. {get_text('what_if_header', lang_code)}")
        st.write("Simulate changes in water availability, budget, land, or rainfall to evaluate risk and see how the optimal plan adapts.")

        # Preset scenarios
        c_p1, c_p2, c_p3 = st.columns(3)
        preset = None
        with c_p1:
            if st.button("🏜️ 20% Water Shortage (Drought)", use_container_width=True):
                preset = "drought"
        with c_p2:
            if st.button("💰 +30% Capital Infusion", use_container_width=True):
                preset = "capital"
        with c_p3:
            if st.button("🌧️ +25% High Rainfall", use_container_width=True):
                preset = "rain"

        # Sliders for What-if
        default_water_mod = 0.8 if preset == "drought" else 1.0
        default_budget_mod = 1.3 if preset == "capital" else 1.0
        default_rain_mod = 1.25 if preset == "rain" else 1.0

        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            whatif_water = st.slider(
                "Simulated Water Availability (Units)",
                min_value=float(max(100.0, water_input * 0.3)),
                max_value=float(water_input * 2.0),
                value=float(water_input * default_water_mod),
                step=100.0
            )
        with col_w2:
            whatif_budget = st.slider(
                "Simulated Budget (₹ INR)",
                min_value=float(max(5000.0, budget_input * 0.3)),
                max_value=float(budget_input * 2.5),
                value=float(budget_input * default_budget_mod),
                step=5000.0
            )
        with col_w3:
            whatif_land = st.slider(
                "Simulated Land (Acres)",
                min_value=float(max(0.5, land_input * 0.5)),
                max_value=float(land_input * 2.0),
                value=float(land_input),
                step=0.5
            )

        # Run What-If Optimization
        whatif_result = optimize_crop_allocation(
            candidate_crops=candidate_profiles,
            available_land=whatif_land,
            available_water=whatif_water,
            available_fertilizer=fert_input,
            available_budget=whatif_budget
        )

        if whatif_result["success"]:
            # Display What-if Comparison
            w_c1, w_c2, w_c3 = st.columns(3)
            profit_delta = whatif_result["total_profit"] - opt_result["total_profit"]
            land_delta = whatif_result["total_land_used"] - opt_result["total_land_used"]

            with w_c1:
                st.metric(
                    "Simulated Expected Profit",
                    format_currency(whatif_result["total_profit"]),
                    delta=f"{format_currency(profit_delta)} ({'+' if profit_delta >= 0 else ''}{(profit_delta/opt_result['total_profit']*100) if opt_result['total_profit'] > 0 else 0:.1f}%)"
                )
            with w_c2:
                st.metric(
                    "Simulated Land Utilization",
                    f"{whatif_result['total_land_used']} Acres",
                    delta=f"{'+' if land_delta >= 0 else ''}{land_delta:.2f} Acres"
                )
            with w_c3:
                w_lim = whatif_result["limiting_resources"]
                st.metric(
                    "New Limiting Factor",
                    ", ".join(w_lim) if w_lim else "None"
                )

            # Allocation Comparison table
            comp_rows = []
            for crop in candidate_selection:
                orig_alloc = opt_result["allocations"].get(crop, 0.0)
                sim_alloc = whatif_result["allocations"].get(crop, 0.0)
                comp_rows.append({
                    "Crop": crop.title(),
                    "Original Plan (Acres)": f"{orig_alloc:.3f}",
                    "Simulated Plan (Acres)": f"{sim_alloc:.3f}",
                    "Change (Acres)": f"{'+' if sim_alloc >= orig_alloc else ''}{sim_alloc - orig_alloc:.3f}"
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            # Plain english summary of what-if delta
            if whatif_water < water_input:
                water_pct_drop = int((1 - whatif_water / water_input) * 100)
                st.info(f"💡 **Scenario Insight**: With **{water_pct_drop}% less water**, the optimizer adjusts total cultivated land from **{opt_result['total_land_used']:.2f}** to **{whatif_result['total_land_used']:.2f} acres**, shifting resources to maintain maximum possible profitability.")
            elif whatif_budget > budget_input:
                budget_pct_gain = int((whatif_budget / budget_input - 1) * 100)
                st.success(f"💡 **Scenario Insight**: With **{budget_pct_gain}% more budget**, net expected farm profit changes by **{format_currency(profit_delta)}**.")

    else:
        st.error(f"Optimization could not be completed: {opt_result['message']}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 12px;'>"
    "🌱 SmartCrop Optimizer | AI-Based Agricultural Decision Support System | Hackathon MVP"
    "</div>",
    unsafe_allow_html=True
)
