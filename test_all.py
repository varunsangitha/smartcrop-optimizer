"""
Verification script for SmartCrop Optimizer
Tests all backend modules, models, optimization, and visualization components.
"""

import sys
import os

# Set UTF-8 output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("========================================")
print("  SmartCrop Optimizer Full Test Suite   ")
print("========================================")

# 1. Test Model Loading
print("\n[1/7] Testing Model Loading...")
from backend.recommendation import load_models
crop_m, yield_m, price_m = load_models()
assert crop_m is not None, "Crop model is None"
assert yield_m is not None, "Yield model is None"
assert price_m is not None, "Price model is None"
print("  --> All 3 models loaded successfully.")

# 2. Test Crop Recommendation Model
print("\n[2/7] Testing Crop Recommendation with prompt sample inputs...")
from backend.recommendation import recommend_crops
# Prompt test input:
# N = 90, P = 42, K = 43, temperature = 25, humidity = 80, ph = 6.5, rainfall = 200
rec = recommend_crops(N=90, P=42, K=43, temperature=25, humidity=80, ph=6.5, rainfall=200)
print(f"  --> Top Recommended Crop: {rec['recommended_crop']}")
print(f"  --> Confidence: {rec['confidence']}%")
print(f"  --> Candidates: {[c['crop'] for c in rec['candidates']]}")
assert rec['recommended_crop'] == "rice", f"Expected 'rice', got {rec['recommended_crop']}"
print("  --> Crop recommendation verified: 'rice' predicted accurately.")

# 3. Test Yield Prediction
print("\n[3/7] Testing Yield Prediction...")
from backend.recommendation import estimate_crop_yield
rice_yield = estimate_crop_yield("rice", annual_rainfall=200, state="Andhra Pradesh", season="Kharif")
print(f"  --> Rice Expected Yield: {rice_yield} Q/Acre (~{rice_yield/10:.2f} Tonnes/Acre)")
assert rice_yield > 0, "Rice yield must be > 0"

maize_yield = estimate_crop_yield("maize", annual_rainfall=200, state="Andhra Pradesh", season="Kharif")
print(f"  --> Maize Expected Yield: {maize_yield} Q/Acre (~{maize_yield/10:.2f} Tonnes/Acre)")
assert maize_yield > 0, "Maize yield must be > 0"

# 4. Test Price Estimation
print("\n[4/7] Testing Market Price Estimation...")
from backend.recommendation import estimate_crop_price
rice_price = estimate_crop_price("rice")
print(f"  --> Rice Estimated Modal Price: ₹{rice_price:,.2f} / Quintal")
assert rice_price > 1000, "Rice price should be realistic Mandi price (> ₹1000/Q)"

# 5. Test Resource Calculator
print("\n[5/7] Testing Resource Calculator...")
from backend.resource_calculator import (
    get_resource_requirements,
    calculate_cultivation_cost,
    calculate_water_requirement,
    calculate_fertilizer_requirement,
    RESOURCE_REQUIREMENTS
)
for crop in ["rice", "maize", "cotton", "banana", "apple", "coconut", "coffee", "grapes", "jute", "mango", "orange", "papaya", "pomegranate"]:
    req = get_resource_requirements(crop)
    assert req["water"] > 0 and req["fertilizer"] > 0 and req["cost"] > 0
print(f"  --> Verified planning assumptions for all {len(RESOURCE_REQUIREMENTS)} crops.")

# 6. Test PuLP Optimizer with prompt constraints:
# Land = 5 acres, Water = 4000, Fertilizer = 500, Budget = 100000
print("\n[6/7] Testing PuLP Optimizer with prompt test inputs...")
from backend.recommendation import get_crop_profile
from backend.optimizer import optimize_crop_allocation

rice_prof = get_crop_profile("rice", annual_rainfall=200)
opt_rice = optimize_crop_allocation(
    candidate_crops={"rice": rice_prof},
    available_land=5.0,
    available_water=4000.0,
    available_fertilizer=500.0,
    available_budget=100000.0
)
print(f"  --> Status: {opt_rice['status']}")
print(f"  --> Rice Allocation: {opt_rice['allocations']['rice']} Acres (Expected ~3.333 due to 4000 water limit)")
print(f"  --> Water Used: {opt_rice['total_water_used']} / 4000.0 (Util: {opt_rice['utilization']['water_pct']}%)")
print(f"  --> Limiting Resource: {opt_rice['limiting_resources']}")
print(f"  --> Expected Profit: ₹{opt_rice['total_profit']:,.2f}")
assert opt_rice["allocations"]["rice"] > 3.0, "Expected ~3.333 acres allocated to rice"
assert "Irrigation Water" in opt_rice["limiting_resources"], "Water should be the limiting resource"

# 7. Test Visualization Charts
print("\n[7/7] Testing Visualizations and Localization...")
from utils.helpers import (
    create_land_allocation_chart,
    create_profit_chart,
    create_resource_utilization_chart,
    create_revenue_vs_cost_chart,
    get_text
)
fig1 = create_land_allocation_chart(opt_rice["crop_details"], opt_rice["unused_resources"]["land"])
fig2 = create_resource_utilization_chart(opt_rice["utilization"])
fig3 = create_profit_chart(opt_rice["crop_details"])
fig4 = create_revenue_vs_cost_chart(opt_rice["crop_details"])
assert len(fig1.data) > 0 and len(fig2.data) > 0
print("  --> All Plotly figures generated successfully.")
print(f"  --> English Title: {get_text('title', 'en')}")
print(f"  --> Telugu Title: {get_text('title', 'te')}")

print("\n========================================")
print("  ALL 7 VERIFICATION TESTS PASSED!       ")
print("========================================")
