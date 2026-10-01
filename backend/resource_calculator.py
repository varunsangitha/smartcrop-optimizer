"""
Resource Calculator Module
Provides planning estimates for water, fertilizer, and cost per acre for crops.

NOTE:
These values are demo planning estimates / demo assumptions for the hackathon MVP.
They should be interpreted as planning guidelines, not universal scientific constants.
"""

# Resource requirements per acre (Planning estimates / demo assumptions)
RESOURCE_REQUIREMENTS = {
    "rice": {
        "water": 1200,      # Irrigation water in m3 or mm equivalent / acre
        "fertilizer": 120,   # NPK fertilizer in kg / acre
        "cost": 18000       # Cultivation cost in INR / acre
    },
    "maize": {
        "water": 650,
        "fertilizer": 90,
        "cost": 14000
    },
    "cotton": {
        "water": 700,
        "fertilizer": 100,
        "cost": 16000
    },
    "banana": {
        "water": 900,
        "fertilizer": 150,
        "cost": 22000
    },
    "apple": {
        "water": 600,
        "fertilizer": 100,
        "cost": 20000
    },
    "coconut": {
        "water": 500,
        "fertilizer": 80,
        "cost": 15000
    },
    "coffee": {
        "water": 450,
        "fertilizer": 70,
        "cost": 13000
    },
    "grapes": {
        "water": 750,
        "fertilizer": 110,
        "cost": 20000
    },
    "jute": {
        "water": 550,
        "fertilizer": 80,
        "cost": 12000
    },
    "mango": {
        "water": 500,
        "fertilizer": 70,
        "cost": 14000
    },
    "orange": {
        "water": 550,
        "fertilizer": 80,
        "cost": 15000
    },
    "papaya": {
        "water": 700,
        "fertilizer": 100,
        "cost": 17000
    },
    "pomegranate": {
        "water": 500,
        "fertilizer": 80,
        "cost": 16000
    },
    # Extended planning estimates for additional crops in crop_model dataset
    "chickpea": {
        "water": 350,
        "fertilizer": 60,
        "cost": 11000
    },
    "lentil": {
        "water": 320,
        "fertilizer": 55,
        "cost": 10500
    },
    "blackgram": {
        "water": 300,
        "fertilizer": 50,
        "cost": 10000
    },
    "mungbean": {
        "water": 300,
        "fertilizer": 50,
        "cost": 10000
    },
    "mothbeans": {
        "water": 250,
        "fertilizer": 45,
        "cost": 9000
    },
    "pigeonpeas": {
        "water": 400,
        "fertilizer": 65,
        "cost": 11500
    },
    "kidneybeans": {
        "water": 400,
        "fertilizer": 70,
        "cost": 12500
    },
    "watermelon": {
        "water": 450,
        "fertilizer": 85,
        "cost": 13500
    },
    "muskmelon": {
        "water": 400,
        "fertilizer": 80,
        "cost": 13000
    }
}


def get_resource_requirements(crop: str) -> dict:
    """
    Return resource requirements (water, fertilizer, cost) per acre for a crop.
    """
    crop = crop.lower().strip()
    if crop not in RESOURCE_REQUIREMENTS:
        # Fallback to general grain/pulse average if not found
        return {
            "water": 500,
            "fertilizer": 75,
            "cost": 13500
        }
    return RESOURCE_REQUIREMENTS[crop]


def calculate_water_requirement(crop: str, acres: float) -> float:
    """Calculate total water requirement for the given crop and acres."""
    req = get_resource_requirements(crop)
    return round(float(req["water"] * acres), 2)


def calculate_fertilizer_requirement(crop: str, acres: float) -> float:
    """Calculate total fertilizer requirement (kg) for the given crop and acres."""
    req = get_resource_requirements(crop)
    return round(float(req["fertilizer"] * acres), 2)


def calculate_cultivation_cost(crop: str, acres: float) -> float:
    """Calculate total cultivation cost (INR) for the given crop and acres."""
    req = get_resource_requirements(crop)
    return round(float(req["cost"] * acres), 2)


def calculate_expected_revenue(acres: float, yield_per_acre: float, price_per_unit: float) -> float:
    """
    Calculate expected revenue.
    Revenue = allocated_land * predicted_yield * predicted_price
    """
    return round(float(acres * yield_per_acre * price_per_unit), 2)


def calculate_expected_profit(revenue: float, total_cost: float) -> float:
    """
    Calculate expected profit.
    Profit = Expected Revenue - Cultivation Cost
    """
    return round(float(revenue - total_cost), 2)


def get_all_supported_crops() -> list:
    """Return list of all crops with resource definitions."""
    return sorted(list(RESOURCE_REQUIREMENTS.keys()))