"""
Recommendation Pipeline Module
Connects the AI models for:
  1. Crop Recommendation (Random Forest Classifier)
  2. Yield Prediction (Random Forest Regressor)
  3. Market Price Estimation (Random Forest Regressor)
  4. Resource & Cost Calculation
  5. Optimal Allocation via PuLP
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Attempt Streamlit resource caching for fast reruns on Streamlit Cloud
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

from backend.resource_calculator import (
    get_resource_requirements,
    calculate_cultivation_cost,
    calculate_water_requirement,
    calculate_fertilizer_requirement,
    calculate_expected_revenue,
    calculate_expected_profit,
    RESOURCE_REQUIREMENTS
)
from backend.optimizer import optimize_crop_allocation

# Crop name mappings across datasets
CROP_TO_MANDI = {
    "apple": "Apple",
    "banana": "Banana",
    "coconut": "Coconut",
    "coffee": "Coffee",
    "cotton": "Cotton",
    "grapes": "Grapes",
    "jute": "Jute",
    "maize": "Maize",
    "mango": "Mango",
    "orange": "Orange",
    "papaya": "Papaya",
    "pomegranate": "Pomegranate",
    "rice": "Paddy(Common)"
}

# Baseline Mandi price bounds (min_price, max_price in INR/Quintal) from Mandi historical data
MANDI_PRICE_BOUNDS = {
    "apple": (12000.0, 18000.0),
    "banana": (4000.0, 4800.0),
    "coconut": (4800.0, 5000.0),
    "coffee": (21400.0, 21600.0),
    "cotton": (7700.0, 8600.0),
    "grapes": (9000.0, 10000.0),
    "jute": (14400.0, 14700.0),
    "maize": (2000.0, 2100.0),
    "mango": (4000.0, 5000.0),
    "orange": (12000.0, 14000.0),
    "papaya": (3000.0, 3500.0),
    "pomegranate": (12000.0, 16000.0),
    "rice": (2360.0, 2400.0),
    "chickpea": (5600.0, 6000.0),
    "lentil": (6200.0, 6600.0),
    "blackgram": (7000.0, 7400.0),
    "mungbean": (7300.0, 7800.0),
    "mothbeans": (5800.0, 6200.0),
    "pigeonpeas": (9200.0, 9800.0),
    "kidneybeans": (8200.0, 8800.0),
    "watermelon": (1200.0, 1600.0),
    "muskmelon": (2000.0, 2400.0)
}

# Crop name mapping to Yield dataset column name
CROP_TO_YIELD_COL = {
    "rice": "Crop_Rice",
    "maize": "Crop_Maize",
    "cotton": "Crop_Cotton(lint)",
    "banana": "Crop_Banana",
    "coconut": "Crop_Coconut",
    "grapes": "Crop_Grapes",
    "jute": "Crop_Jute",
    "mango": "Crop_Mango"
}

# Benchmark yields (in Quintals per Acre) for fallbacks and domain validation
BENCHMARK_YIELDS_Q_PER_ACRE = {
    "rice": 9.5,
    "maize": 11.0,
    "cotton": 6.2,
    "jute": 12.5,
    "banana": 105.0,
    "grapes": 92.0,
    "mango": 27.5,
    "coconut": 35.0,
    "apple": 45.0,
    "coffee": 5.2,
    "orange": 42.0,
    "papaya": 110.0,
    "pomegranate": 34.0,
    "chickpea": 5.5,
    "lentil": 4.5,
    "blackgram": 3.8,
    "mungbean": 3.5,
    "mothbeans": 2.8,
    "pigeonpeas": 4.8,
    "kidneybeans": 4.5,
    "watermelon": 85.0,
    "muskmelon": 65.0
}


def _load_models_from_disk(models_dir: str = "models") -> Tuple[Any, Any, Any]:
    """
    Safely resolve the models directory relative to project root and load pickles.
    """
    if not os.path.isabs(models_dir):
        # Resolve path relative to project root (parent directory of backend/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resolved_dir = os.path.join(project_root, models_dir)
    else:
        resolved_dir = models_dir

    crop_path = os.path.join(resolved_dir, "crop_model.pkl")
    yield_path = os.path.join(resolved_dir, "yield_model.pkl")
    price_path = os.path.join(resolved_dir, "price_model.pkl")

    if not os.path.exists(crop_path):
        raise FileNotFoundError(f"Crop model not found at: {crop_path}")
    if not os.path.exists(yield_path):
        raise FileNotFoundError(f"Yield model not found at: {yield_path}")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Price model not found at: {price_path}")

    crop_model = joblib.load(crop_path)
    yield_model_data = joblib.load(yield_path)
    price_model_data = joblib.load(price_path)

    return crop_model, yield_model_data, price_model_data


if _HAS_STREAMLIT:
    # Use Streamlit's cache_resource to cache models in memory across reruns and user sessions
    @st.cache_resource(show_spinner=False)
    def load_models(models_dir: str = "models") -> Tuple[Any, Any, Any]:
        """Load crop, yield, and price models with Streamlit resource caching."""
        return _load_models_from_disk(models_dir)
else:
    _LOCAL_CACHE = None
    def load_models(models_dir: str = "models") -> Tuple[Any, Any, Any]:
        """Load crop, yield, and price models with in-memory singleton fallback."""
        global _LOCAL_CACHE
        if _LOCAL_CACHE is None:
            _LOCAL_CACHE = _load_models_from_disk(models_dir)
        return _LOCAL_CACHE


def recommend_crops(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Predict top suitable crop(s) using the trained Random Forest classifier.
    """
    crop_model, _, _ = load_models()

    input_df = pd.DataFrame([{
        "N": float(N),
        "P": float(P),
        "K": float(K),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "ph": float(ph),
        "rainfall": float(rainfall)
    }])

    best_crop = crop_model.predict(input_df)[0]
    probabilities = crop_model.predict_proba(input_df)[0]
    classes = crop_model.classes_

    # Sort crops by predicted probability
    sorted_indices = np.argsort(probabilities)[::-1]

    candidates = []
    for idx in sorted_indices:
        prob = probabilities[idx]
        crop_name = classes[idx].lower().strip()
        # Include if probability is > 0.01 or if we need top_k
        if prob > 0.005 or len(candidates) < top_k:
            candidates.append({
                "crop": crop_name,
                "crop_display": crop_name.title(),
                "confidence": round(float(prob) * 100, 2),
                "probability": float(prob)
            })
        if len(candidates) >= top_k:
            break

    # Confidence of best crop
    best_idx = np.where(classes == best_crop)[0][0]
    best_confidence = round(float(probabilities[best_idx]) * 100, 2)

    explanation = (
        f"Recommended '{best_crop.title()}' with {best_confidence}% confidence because the provided "
        f"soil nutrients (N={N}, P={P}, K={K}, pH={ph}) and weather conditions "
        f"(Temp={temperature}°C, Humidity={humidity}%, Rainfall={rainfall}mm) "
        f"closely align with the agronomic conditions learned for this crop."
    )

    return {
        "recommended_crop": best_crop.lower(),
        "confidence": best_confidence,
        "candidates": candidates,
        "explanation": explanation
    }


def estimate_crop_yield(
    crop: str,
    annual_rainfall: float = 1200.0,
    state: str = "Andhra Pradesh",
    season: str = "Kharif",
    year: int = 2024
) -> float:
    """
    Estimate expected yield (in Quintals per Acre) using the trained Random Forest regressor.
    Falls back gracefully to verified agronomic benchmarks if crop is not in yield dataset.
    """
    _, yield_model_data, _ = load_models()
    model = yield_model_data["model"]
    features = yield_model_data["features"]

    crop_lower = crop.lower().strip()
    col_name = CROP_TO_YIELD_COL.get(crop_lower)

    # If crop is in yield model features, use the ML model
    if col_name and col_name in features:
        try:
            row = pd.DataFrame(0.0, index=[0], columns=features)
            row["Year"] = float(year)
            # Use typical agricultural production area medians
            row["Area"] = 250000.0
            row["Production"] = 550000.0
            row["Annual_Rainfall"] = float(annual_rainfall)
            row["Fertilizer"] = 25000000.0
            row["Pesticide"] = 50000.0

            row[col_name] = 1.0

            state_col = f"State_{state}"
            if state_col in features:
                row[state_col] = 1.0

            season_col = f"Season_{season}"
            if season_col in features:
                row[season_col] = 1.0

            pred_yield_tha = float(model.predict(row)[0])
            # Ensure prediction is physically plausible (greater than 0)
            if pred_yield_tha > 0:
                # Convert Tonnes per Hectare to Quintals per Acre:
                # 1 ha = 2.471 acres, 1 Tonne = 10 Quintals -> Q/acre = (T/ha / 2.471) * 10
                q_per_acre = (pred_yield_tha / 2.471) * 10.0
                return round(float(q_per_acre), 2)
        except Exception:
            pass

    # Benchmark fallback if crop feature is not in yield model or on error
    fallback = BENCHMARK_YIELDS_Q_PER_ACRE.get(crop_lower, 8.0)
    # Slight dynamic adjustment based on rainfall
    rainfall_factor = max(0.85, min(1.15, annual_rainfall / 1000.0))
    return round(float(fallback * rainfall_factor), 2)


def estimate_crop_price(
    crop: str,
    year: int = 2024,
    month: int = 10,
    day: int = 1
) -> float:
    """
    Estimate modal market price (INR per Quintal) using the trained Random Forest regressor.
    Label: 'Estimated market value'
    """
    _, _, price_model_data = load_models()
    p_model = price_model_data["model"]

    crop_lower = crop.lower().strip()
    bounds = MANDI_PRICE_BOUNDS.get(crop_lower, (2500.0, 3200.0))
    min_p, max_p = bounds

    try:
        price_row = pd.DataFrame([{
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "min_price": float(min_p),
            "max_price": float(max_p)
        }])
        pred_modal = float(p_model.predict(price_row)[0])
        # Bound predicted price within realistic mandi limits
        pred_modal = max(min_p * 0.9, min(max_p * 1.1, pred_modal))
        return round(pred_modal, 2)
    except Exception:
        # Fallback to midpoint of historical bounds
        return round((min_p + max_p) / 2.0, 2)


def get_crop_profile(
    crop: str,
    annual_rainfall: float = 1200.0,
    state: str = "Andhra Pradesh",
    season: str = "Kharif",
    year: int = 2024
) -> Dict[str, Any]:
    """
    Assemble the complete agronomic & economic profile for a candidate crop.
    """
    crop_lower = crop.lower().strip()
    expected_yield = estimate_crop_yield(crop_lower, annual_rainfall, state, season, year)
    expected_price = estimate_crop_price(crop_lower, year)
    resources = get_resource_requirements(crop_lower)

    unit_revenue = round(expected_yield * expected_price, 2)
    unit_cost = round(float(resources["cost"]), 2)
    unit_profit = round(unit_revenue - unit_cost, 2)

    return {
        "crop": crop_lower,
        "crop_display": crop_lower.title(),
        "yield": expected_yield,         # Quintals / acre
        "price": expected_price,         # INR / Quintal
        "water": resources["water"],      # m3 or mm equivalent / acre
        "fertilizer": resources["fertilizer"], # kg / acre
        "cost": unit_cost,               # INR / acre
        "revenue_per_acre": unit_revenue,
        "profit_per_acre": unit_profit,
        "water_per_acre": resources["water"],
        "fertilizer_per_acre": resources["fertilizer"]
    }


def run_farm_planning_pipeline(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    available_land: float,
    available_water: float,
    available_fertilizer: float,
    available_budget: float,
    state: str = "Andhra Pradesh",
    season: str = "Kharif",
    selected_crops: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes the entire end-to-end pipeline:
      User Inputs -> Crop Recommendation -> Yield & Price Estimation
      -> Resource & Cost Calculation -> PuLP Optimization -> Results & Analysis.
    """
    # Step 1: Crop Recommendation
    rec_result = recommend_crops(N, P, K, temperature, humidity, ph, rainfall)

    # Step 2: Determine candidate crops
    if selected_crops and len(selected_crops) > 0:
        candidates_to_use = [c.lower().strip() for c in selected_crops]
    else:
        # Use top recommended crops from the ML model
        top_candidates = [c["crop"] for c in rec_result["candidates"]]
        # Ensure the top recommendation is present
        if rec_result["recommended_crop"] not in top_candidates:
            top_candidates.insert(0, rec_result["recommended_crop"])
        candidates_to_use = list(dict.fromkeys(top_candidates))

    # Step 3: Compute crop profiles (Yield, Price, Cost, Water, Fertilizer)
    candidate_profiles = {}
    for crop in candidates_to_use:
        candidate_profiles[crop] = get_crop_profile(
            crop=crop,
            annual_rainfall=rainfall,
            state=state,
            season=season
        )

    # Step 4: Run PuLP Optimization
    opt_result = optimize_crop_allocation(
        candidate_crops=candidate_profiles,
        available_land=available_land,
        available_water=available_water,
        available_fertilizer=available_fertilizer,
        available_budget=available_budget
    )

    return {
        "recommendation": rec_result,
        "candidate_profiles": candidate_profiles,
        "optimization": opt_result
    }
