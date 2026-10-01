"""
Helper utilities and Plotly visualization builders for SmartCrop Optimizer.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List


def format_currency(amount: float) -> str:
    """Format number as INR currency string."""
    return f"₹{amount:,.2f}"


def format_acres(acres: float) -> str:
    """Format acres with unit."""
    return f"{acres:,.2f} Acres"


def create_land_allocation_chart(crop_details: List[Dict[str, Any]], unused_land: float) -> go.Figure:
    """
    Donut chart showing Land Allocation by Crop plus Unused Land.
    """
    labels = []
    values = []
    colors = ["#10b981", "#06b6d4", "#f59e0b", "#8b5cf6", "#ec4899", "#3b82f6"]

    for item in crop_details:
        acres = item.get("Allocated Land (Acres)", 0.0)
        if acres > 0:
            labels.append(item.get("Crop", "Crop"))
            values.append(acres)

    if unused_land > 0.01:
        labels.append("Unused Land")
        values.append(round(unused_land, 2))

    if not values or sum(values) == 0:
        # Empty placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="No land allocated under current constraints",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(height=320, margin=dict(t=30, b=20, l=20, r=20))
        return fig

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        textinfo="label+percent",
        marker=dict(colors=colors[:len(labels)], line=dict(color="#ffffff", width=2)),
        hoverinfo="label+value+percent",
        textfont=dict(size=13, family="sans-serif")
    )])

    fig.update_layout(
        title=dict(text="<b>Land Allocation by Crop (Acres)</b>", font=dict(size=15, color="#1e293b")),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=340,
        margin=dict(t=40, b=30, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_profit_chart(crop_details: List[Dict[str, Any]]) -> go.Figure:
    """
    Bar chart showing Expected Profit per crop.
    """
    crops = []
    profits = []
    colors = []

    for item in crop_details:
        acres = item.get("Allocated Land (Acres)", 0.0)
        if acres > 0:
            crops.append(item.get("Crop", "Crop"))
            prof = item.get("Expected Profit (₹)", 0.0)
            profits.append(prof)
            colors.append("#10b981" if prof >= 0 else "#ef4444")

    if not crops:
        fig = go.Figure()
        fig.add_annotation(
            text="No active crop allocation to display profit",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(height=320, margin=dict(t=30, b=20, l=20, r=20))
        return fig

    fig = go.Figure(data=[go.Bar(
        x=crops,
        y=profits,
        text=[f"₹{p:,.0f}" for p in profits],
        textposition="auto",
        marker_color=colors,
        marker_line=dict(color="#065f46", width=1.5)
    )])

    fig.update_layout(
        title=dict(text="<b>Expected Profit by Crop (₹)</b>", font=dict(size=15, color="#1e293b")),
        xaxis=dict(title="Crop"),
        yaxis=dict(title="Profit (INR)", tickprefix="₹"),
        height=340,
        margin=dict(t=40, b=30, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_resource_utilization_chart(utilization: Dict[str, float]) -> go.Figure:
    """
    Horizontal bar chart showing percentage utilization of each farm resource.
    Highlights bottleneck / limiting resources nearing 100%.
    """
    resources = ["Budget", "Fertilizer", "Water", "Land"]
    keys = ["budget_pct", "fertilizer_pct", "water_pct", "land_pct"]
    values = [utilization.get(k, 0.0) for k in keys]

    bar_colors = []
    for val in values:
        if val >= 99.0:
            bar_colors.append("#ef4444") # Red for binding / exhausted constraint
        elif val >= 75.0:
            bar_colors.append("#f59e0b") # Amber for high utilization
        else:
            bar_colors.append("#10b981") # Green for comfortable headroom

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=resources,
        x=values,
        orientation="h",
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        marker=dict(color=bar_colors, line=dict(color="#1e293b", width=1))
    ))

    # Add 100% capacity threshold line
    fig.add_vline(x=100, line_dash="dash", line_color="#ef4444", annotation_text="100% Limit", annotation_position="top right")

    fig.update_layout(
        title=dict(text="<b>Resource Utilization (%)</b>", font=dict(size=15, color="#1e293b")),
        xaxis=dict(title="Utilization Percentage (%)", range=[0, max(115, max(values) + 15)]),
        yaxis=dict(title="Resource"),
        height=340,
        margin=dict(t=40, b=30, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_revenue_vs_cost_chart(crop_details: List[Dict[str, Any]]) -> go.Figure:
    """
    Grouped bar chart comparing Cultivation Cost vs Expected Revenue.
    """
    crops = []
    costs = []
    revenues = []

    for item in crop_details:
        acres = item.get("Allocated Land (Acres)", 0.0)
        if acres > 0:
            crops.append(item.get("Crop", "Crop"))
            costs.append(item.get("Cultivation Cost (₹)", 0.0))
            revenues.append(item.get("Expected Revenue (₹)", 0.0))

    if not crops:
        fig = go.Figure()
        fig.add_annotation(
            text="No active crop allocation to compare revenue and cost",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(height=320, margin=dict(t=30, b=20, l=20, r=20))
        return fig

    fig = go.Figure(data=[
        go.Bar(name="Cost (₹)", x=crops, y=costs, marker_color="#f87171"),
        go.Bar(name="Revenue (₹)", x=crops, y=revenues, marker_color="#34d399")
    ])

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Revenue vs Cultivation Cost by Crop (₹)</b>", font=dict(size=15, color="#1e293b")),
        xaxis=dict(title="Crop"),
        yaxis=dict(title="INR (₹)", tickprefix="₹"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=340,
        margin=dict(t=40, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# English - Telugu translation glossary for farmer accessibility
TRANSLATIONS = {
    "en": {
        "title": "SmartCrop Optimizer",
        "subtitle": "AI-Based Crop & Agricultural Resource Optimization System",
        "soil_weather_header": "1. Soil & Weather Conditions",
        "farm_resources_header": "2. Farm Resources & Budget",
        "recommended_crop": "AI Recommended Crop",
        "confidence": "Confidence",
        "yield_prediction": "Expected Yield",
        "market_price": "Estimated Market Price",
        "water_req": "Water Requirement",
        "fertilizer_req": "Fertilizer Requirement",
        "cultivation_cost": "Estimated Cultivation Cost",
        "optimize_btn": "Optimize Farm Plan",
        "why_this_plan": "Why This Plan?",
        "total_profit": "Expected Total Profit",
        "total_revenue": "Expected Revenue",
        "total_cost": "Total Investment Cost",
        "land_used": "Land Allocated",
        "what_if_header": "What-If Scenario Simulation"
    },
    "te": {
        "title": "స్మార్ట్ క్రాప్ ఆప్టిమైజర్ (SmartCrop Optimizer)",
        "subtitle": "AI ఆధారిత పంట మరియు వనరుల సమతుల్య ప్రణాళికా వ్యవస్థ",
        "soil_weather_header": "1. నేల & వాతావరణ వివరాలు",
        "farm_resources_header": "2. అందుబాటులో ఉన్న వనరులు & బడ్జెట్",
        "recommended_crop": "సిఫార్సు చేయబడిన పంట",
        "confidence": "ఖచ్చితత్వం",
        "yield_prediction": "అంచనా దిగుబడి",
        "market_price": "మార్కెట్ ధర (క్వింటాల్ కు)",
        "water_req": "నీటి అవసరం",
        "fertilizer_req": "ఎరువుల అవసరం",
        "cultivation_cost": "సాగు ఖర్చు",
        "optimize_btn": "వ్యవసాయ ప్రణాళికను ఆప్టిమైజ్ చేయండి",
        "why_this_plan": "ఈ ప్రణాళికను ఎందుకు ఎంచుకోవాలి?",
        "total_profit": "మొత్తం ఆశించే లాభం",
        "total_revenue": "మొత్తం ఆదాయం",
        "total_cost": "మొత్తం సాగు ఖర్చు",
        "land_used": "కేటాయించిన భూమి",
        "what_if_header": "పరిస్థితులు మారితే? (What-If విశ్లేషణ)"
    }
}


def get_text(key: str, lang: str = "en") -> str:
    """Retrieve localized text string."""
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return lang_dict.get(key, TRANSLATIONS["en"].get(key, key))
