"""
Agricultural Resource Allocation Optimizer using PuLP Linear Programming.

Formulation:
  Decision Variables:
    x_i >= 0 (acres allocated to crop i, continuous / fractional acres allowed)

  Objective:
    Maximize Expected Total Profit = SUM(x_i * (predicted_yield_i * predicted_price_i - crop_cost_i))

  Constraints:
    1. Land:       SUM(x_i) <= available_land
    2. Water:      SUM(x_i * water_i) <= available_water
    3. Fertilizer: SUM(x_i * fertilizer_i) <= available_fertilizer
    4. Budget:     SUM(x_i * cost_i) <= available_budget
    5. Non-negativity: x_i >= 0
"""

import pulp
from typing import Dict, Any, List, Optional


def get_pulp_solver():
    """
    Detect and return the appropriate PuLP solver instance.
    Supports PuLP 4.x (HiGHS) and PuLP 2.x (CBC / default).
    """
    # Try HiGHS first (ideal for PuLP 4.x)
    if hasattr(pulp, "HiGHS"):
        try:
            solver = pulp.HiGHS(msg=False)
            if solver.available():
                return solver
        except Exception:
            pass

    # Try PULP_CBC_CMD (standard for PuLP 2.x)
    if hasattr(pulp, "PULP_CBC_CMD"):
        try:
            solver = pulp.PULP_CBC_CMD(msg=False)
            if solver.available():
                return solver
        except Exception:
            pass

    # Try default solver
    try:
        if hasattr(pulp, "LpSolverDefault") and pulp.LpSolverDefault is not None:
            return pulp.LpSolverDefault
    except Exception:
        pass

    return None


def optimize_crop_allocation(
    candidate_crops: Dict[str, Dict[str, float]],
    available_land: float,
    available_water: float,
    available_fertilizer: float,
    available_budget: float,
    min_allocation_pct: float = 0.0
) -> Dict[str, Any]:
    """
    Solves the linear programming problem to maximize farm profit given constraints.

    Parameters:
      candidate_crops: dict mapping crop_name -> {
          "yield": float,        # Expected yield per acre (Quintals/acre)
          "price": float,        # Estimated modal price (INR / Quintal)
          "water": float,        # Water requirement per acre (m3 or mm equivalent)
          "fertilizer": float,   # Fertilizer requirement per acre (kg)
          "cost": float          # Cultivation cost per acre (INR)
      }
      available_land: total land in acres
      available_water: total water available
      available_fertilizer: total fertilizer available (kg)
      available_budget: total capital / budget available (INR)
      min_allocation_pct: optional minimum land percentage for diversification

    Returns:
      Dict with status, success, allocations, financial summary, and resource usage.
    """
    # 1. Validation
    if not candidate_crops:
        return {
            "success": False,
            "status": "No Candidates",
            "message": "No candidate crops were provided for optimization.",
            "allocations": {},
            "total_land_used": 0.0,
            "total_cost": 0.0,
            "total_revenue": 0.0,
            "total_profit": 0.0,
            "total_water_used": 0.0,
            "total_fertilizer_used": 0.0,
            "unused_resources": {
                "land": available_land,
                "water": available_water,
                "fertilizer": available_fertilizer,
                "budget": available_budget,
            },
            "limiting_resources": [],
            "crop_details": []
        }

    if available_land <= 0 or available_budget <= 0:
        return {
            "success": False,
            "status": "Invalid Constraints",
            "message": "Available land and budget must be strictly greater than zero.",
            "allocations": {c: 0.0 for c in candidate_crops},
            "total_land_used": 0.0,
            "total_cost": 0.0,
            "total_revenue": 0.0,
            "total_profit": 0.0,
            "total_water_used": 0.0,
            "total_fertilizer_used": 0.0,
            "unused_resources": {
                "land": available_land,
                "water": available_water,
                "fertilizer": available_fertilizer,
                "budget": available_budget,
            },
            "limiting_resources": ["Land/Budget"],
            "crop_details": []
        }

    # 2. Build PuLP Problem
    prob = pulp.LpProblem("Crop_Allocation_Optimization", pulp.LpMaximize)

    # Create decision variables x_i >= 0
    crop_vars = {}
    for crop in candidate_crops:
        safe_name = f"acres_{crop.replace(' ', '_').replace('(', '_').replace(')', '_')}"
        if hasattr(prob, "add_variable"):
            # PuLP 4.x syntax
            crop_vars[crop] = prob.add_variable(safe_name, lowBound=0.0, cat="Continuous")
        else:
            # PuLP 2.x syntax
            crop_vars[crop] = pulp.LpVariable(safe_name, lowBound=0.0, cat="Continuous")

    # Objective Function: Maximize Total Profit
    # Profit_i = (yield_i * price_i) - cost_i
    profit_terms = []
    unit_profits = {}
    for crop, data in candidate_crops.items():
        y = float(data.get("yield", 0.0))
        p = float(data.get("price", 0.0))
        c = float(data.get("cost", 0.0))
        revenue_per_acre = y * p
        profit_per_acre = revenue_per_acre - c
        unit_profits[crop] = profit_per_acre
        profit_terms.append(crop_vars[crop] * profit_per_acre)

    prob += pulp.lpSum(profit_terms), "Total_Farm_Profit"

    # Constraint 1: Total Land Constraint
    prob += pulp.lpSum([crop_vars[c] for c in candidate_crops]) <= float(available_land), "Land_Constraint"

    # Constraint 2: Total Water Constraint
    prob += (
        pulp.lpSum([crop_vars[c] * float(candidate_crops[c].get("water", 0.0)) for c in candidate_crops])
        <= float(available_water),
        "Water_Constraint"
    )

    # Constraint 3: Total Fertilizer Constraint
    prob += (
        pulp.lpSum([crop_vars[c] * float(candidate_crops[c].get("fertilizer", 0.0)) for c in candidate_crops])
        <= float(available_fertilizer),
        "Fertilizer_Constraint"
    )

    # Constraint 4: Total Budget Constraint
    prob += (
        pulp.lpSum([crop_vars[c] * float(candidate_crops[c].get("cost", 0.0)) for c in candidate_crops])
        <= float(available_budget),
        "Budget_Constraint"
    )

    # Optional minimum allocation constraint if farmer desires crop diversification
    if min_allocation_pct > 0 and len(candidate_crops) > 1:
        min_acres = (min_allocation_pct / 100.0) * available_land
        for c in candidate_crops:
            prob += crop_vars[c] >= min_acres, f"Min_Diversification_{c}"

    # 3. Solve Model
    solver = get_pulp_solver()
    try:
        if solver is not None:
            solve_result = prob.solve(solver)
        else:
            solve_result = prob.solve()
    except Exception as e:
        # Fallback if solver threw error
        return {
            "success": False,
            "status": "Solver Error",
            "message": f"Optimization solver encountered an error: {str(e)}",
            "allocations": {c: 0.0 for c in candidate_crops},
            "total_land_used": 0.0,
            "total_cost": 0.0,
            "total_revenue": 0.0,
            "total_profit": 0.0,
            "total_water_used": 0.0,
            "total_fertilizer_used": 0.0,
            "unused_resources": {
                "land": available_land,
                "water": available_water,
                "fertilizer": available_fertilizer,
                "budget": available_budget,
            },
            "limiting_resources": [],
            "crop_details": []
        }

    # Status check
    status_str = "Optimal"
    if hasattr(prob, "status"):
        if isinstance(prob.status, str):
            status_str = prob.status
        elif hasattr(pulp, "LpStatus") and prob.status in pulp.LpStatus:
            status_str = pulp.LpStatus[prob.status]

    if "Infeasible" in status_str:
        return {
            "success": False,
            "status": "Infeasible",
            "message": "Resource constraints are too tight to support any crop plan. Try increasing budget, water, or fertilizer.",
            "allocations": {c: 0.0 for c in candidate_crops},
            "total_land_used": 0.0,
            "total_cost": 0.0,
            "total_revenue": 0.0,
            "total_profit": 0.0,
            "total_water_used": 0.0,
            "total_fertilizer_used": 0.0,
            "unused_resources": {
                "land": available_land,
                "water": available_water,
                "fertilizer": available_fertilizer,
                "budget": available_budget,
            },
            "limiting_resources": ["Infeasible Constraints"],
            "crop_details": []
        }

    # 4. Extract Solution
    allocations = {}
    total_land_used = 0.0
    total_water_used = 0.0
    total_fertilizer_used = 0.0
    total_cost = 0.0
    total_revenue = 0.0
    total_profit = 0.0
    crop_details = []

    for crop in candidate_crops:
        var = crop_vars[crop]
        val = var.value() if hasattr(var, "value") else getattr(var, "varValue", 0.0)
        acres = max(0.0, float(val or 0.0))
        acres = round(acres, 3)
        allocations[crop] = acres

        data = candidate_crops[crop]
        y = float(data.get("yield", 0.0))
        p = float(data.get("price", 0.0))
        c = float(data.get("cost", 0.0))
        w = float(data.get("water", 0.0))
        f = float(data.get("fertilizer", 0.0))

        crop_yield = round(acres * y, 2)
        crop_cost = round(acres * c, 2)
        crop_rev = round(acres * y * p, 2)
        crop_prof = round(crop_rev - crop_cost, 2)
        crop_water = round(acres * w, 2)
        crop_fert = round(acres * f, 2)

        total_land_used += acres
        total_water_used += crop_water
        total_fertilizer_used += crop_fert
        total_cost += crop_cost
        total_revenue += crop_rev
        total_profit += crop_prof

        crop_details.append({
            "Crop": crop.title(),
            "Allocated Land (Acres)": acres,
            "Yield/Acre (Q)": round(y, 2),
            "Total Yield (Q)": crop_yield,
            "Price (₹/Q)": round(p, 2),
            "Cultivation Cost (₹)": crop_cost,
            "Expected Revenue (₹)": crop_rev,
            "Expected Profit (₹)": crop_prof,
            "Water Used": crop_water,
            "Fertilizer Used (kg)": crop_fert,
            "Profit/Acre (₹)": round(unit_profits[crop], 2)
        })

    total_land_used = round(total_land_used, 3)
    total_water_used = round(total_water_used, 2)
    total_fertilizer_used = round(total_fertilizer_used, 2)
    total_cost = round(total_cost, 2)
    total_revenue = round(total_revenue, 2)
    total_profit = round(total_profit, 2)

    unused_resources = {
        "land": round(max(0.0, available_land - total_land_used), 3),
        "water": round(max(0.0, available_water - total_water_used), 2),
        "fertilizer": round(max(0.0, available_fertilizer - total_fertilizer_used), 2),
        "budget": round(max(0.0, available_budget - total_cost), 2),
    }

    utilization = {
        "land_pct": round((total_land_used / available_land) * 100, 1) if available_land > 0 else 0.0,
        "water_pct": round((total_water_used / available_water) * 100, 1) if available_water > 0 else 0.0,
        "fertilizer_pct": round((total_fertilizer_used / available_fertilizer) * 100, 1) if available_fertilizer > 0 else 0.0,
        "budget_pct": round((total_cost / available_budget) * 100, 1) if available_budget > 0 else 0.0,
    }

    # Identify limiting (binding) resources: utilization >= 99.0%
    limiting_resources = []
    if utilization["land_pct"] >= 99.0:
        limiting_resources.append("Available Land")
    if utilization["water_pct"] >= 99.0:
        limiting_resources.append("Irrigation Water")
    if utilization["fertilizer_pct"] >= 99.0:
        limiting_resources.append("Fertilizer Stock")
    if utilization["budget_pct"] >= 99.0:
        limiting_resources.append("Budget / Capital")

    # Generate friendly explanation
    if limiting_resources:
        limit_desc = ", ".join(limiting_resources)
        explanation = (
            f"The optimizer maximized expected farm profit at ₹{total_profit:,.2f}. "
            f"The primary constraint(s) restricting further expansion are: {limit_desc}. "
            f"Land was allocated to crops yielding the highest net return per unit of the limiting resource."
        )
    else:
        explanation = (
            f"The optimizer allocated land to achieve ₹{total_profit:,.2f} expected profit without exhausting any single resource completely."
        )

    return {
        "success": True,
        "status": status_str,
        "message": "Optimization completed successfully.",
        "allocations": allocations,
        "total_land_used": total_land_used,
        "total_water_used": total_water_used,
        "total_fertilizer_used": total_fertilizer_used,
        "total_cost": total_cost,
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "unused_resources": unused_resources,
        "utilization": utilization,
        "limiting_resources": limiting_resources,
        "crop_details": crop_details,
        "explanation": explanation
    }
