# 🌱 SmartCrop Optimizer
### AI-Based Crop & Agricultural Resource Optimization System

---

## 📌 1. Project Overview & Problem Statement

Smallholder farmers often make critical planting and investment decisions based on intuition, legacy habits, or incomplete market signals. This frequently leads to:
1. **Suboptimal Crop Selection**: Cultivating crops poorly suited to soil chemical composition or seasonal weather patterns.
2. **Resource Inefficiencies**: Overuse or underuse of critical irrigation water and chemical fertilizers.
3. **Financial Vulnerability**: Planting decisions made without visibility into market price trends or cultivation costs, resulting in slim profit margins or severe losses during droughts.
4. **Lack of Risk & Scenario Planning**: Inability to anticipate how farm profitability and land allocation should shift under constrained conditions (e.g., 20% water deficit or reduced operating capital).

**SmartCrop Optimizer** bridges this gap by combining **Machine Learning** with **Linear Programming (PuLP)** into an end-to-end, decision-support dashboard for farmers, agricultural extension workers, and agronomists.

---

## 🚀 2. System Architecture & Complete Pipeline

The application connects four distinct analytical tiers into a unified workflow:

```
                      [ Farmer Inputs ]
         Soil (N, P, K, pH) + Weather (Temp, Humidity, Rain)
                   + Resources (Land, Water, Fert, Budget)
                                ↓
               [ Tier 1: AI Crop Recommendation ]
          Random Forest Classifier (Accuracy: ~99.99%)
             Outputs: Suitable crop(s) + Confidence %
                                ↓
             [ Tier 2: Yield & Price Estimation ]
    Yield Prediction (RF Regressor) + Mandi Price Model (RF Regressor)
      Outputs: Expected Yield (Q/Acre) & Market Value (₹/Quintal)
                                ↓
           [ Tier 3: Resource & Cost Calculation ]
           Per-acre Water, Fertilizer, and Cost Estimates
            Outputs: Net Profit Margin per Acre for Each Crop
                                ↓
            [ Tier 4: PuLP Linear Optimization ]
    Maximize Total Farm Profit Subject to Land, Water, Fert, Budget
      Outputs: Optimal Land Allocation, Financial KPIs, Bottlenecks
                                ↓
        [ Tier 5: Interactive Dashboard & What-If Analysis ]
           KPI Cards + 4 Plotly Charts + Scenario Simulation
```

---

## 🛠️ 3. Technology Stack (Locked)

| Domain | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core application logic |
| **Frontend** | Streamlit | Farmer-friendly interactive dashboard |
| **Machine Learning** | Scikit-learn | Random Forest Classifier & Regressors |
| **Optimization** | PuLP + HiGHS / CBC | Constrained Linear Programming (LP) |
| **Data Processing** | Pandas & NumPy | Tabular data manipulation and vectorization |
| **Visualization** | Plotly | Interactive responsive charts (Donut, Bar, Grouped) |
| **Model Persistence** | Joblib | Serialization and unpickling of ML models |
| **Data Formats** | CSV & Parquet | Historical mandi prices and yield training records |

---

## 📊 4. Datasets & Machine Learning Models

### A. Crop Recommendation Model
- **File**: `backend/crop_model.py` | **Artifact**: `models/crop_model.pkl`
- **Dataset**: `data/crop_recommendation.csv` (2,200 rows, 22 crop classes)
- **Features**: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature (°C), Humidity (%), pH level, Rainfall (mm)
- **Target**: `label` (Crop name)
- **Algorithm**: `RandomForestClassifier(n_estimators=200, random_state=42)`
- **Accuracy**: ~99.99% on stratified test set

### B. Yield Prediction Model
- **File**: `backend/yield_model.py` | **Artifact**: `models/yield_model.pkl`
- **Dataset**: `data/train-00000-of-00001.parquet` (17,400 agricultural records)
- **Features**: Year, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, State (one-hot), Crop (one-hot), Season (one-hot)
- **Target**: `Yield` (Metric Tonnes per Hectare, converted to Quintals per Acre in pipeline)
- **Algorithm**: `RandomForestRegressor(n_estimators=200, random_state=42)`
- **Evaluation**: MAE = 10.18, $R^2$ = 0.9814

### C. Mandi Market Price Model
- **File**: `backend/price_model.py` | **Artifact**: `models/price_model.pkl`
- **Dataset**: `data/crop_prices.csv` (~388,895 APMC Mandi market records)
- **Features**: Year, Month, Day, Historical Min Price, Historical Max Price
- **Target**: `modal_price` (₹ / Quintal)
- **Algorithm**: `RandomForestRegressor(n_estimators=150, random_state=42)`
- **Evaluation**: MAE = 128.59, $R^2$ = 0.9889
- *Notice*: This is an MVP market-value estimation model calibrated on APMC transactions. It predicts the expected modal transaction price rather than long-range speculative futures.

---

## 📐 5. Mathematical Optimization Formulation (PuLP)

Let $C$ be the set of candidate crops. For each crop $i \in C$, let $x_i$ be the continuous decision variable representing the **acres allocated to crop $i$**.

### Objective Function:
$$\max \sum_{i \in C} x_i \times \Big( \text{Yield}_i \times \text{Price}_i - \text{Cost}_i \Big)$$

Where:
- $\text{Yield}_i$: Expected yield in Quintals per Acre
- $\text{Price}_i$: Estimated modal price in ₹ per Quintal
- $\text{Cost}_i$: Cultivation cost in ₹ per Acre

### Subject to Operational Constraints:

1. **Total Land Constraint**:
   $$\sum_{i \in C} x_i \le \text{Available Land (Acres)}$$

2. **Irrigation Water Constraint**:
   $$\sum_{i \in C} x_i \times \text{Water Requirement}_i \le \text{Available Water (Units / m}^3\text{)}$$

3. **Fertilizer Constraint**:
   $$\sum_{i \in C} x_i \times \text{Fertilizer Requirement}_i \le \text{Available Fertilizer (kg)}$$

4. **Working Capital / Budget Constraint**:
   $$\sum_{i \in C} x_i \times \text{Cultivation Cost}_i \le \text{Available Budget (₹)}$$

5. **Non-negativity & Fractional Allocation**:
   $$x_i \ge 0 \quad \forall i \in C$$

---

## 🏷️ 6. Transparency: Distinguishing Data Categories

To maintain scientific integrity and realistic expectations, the system distinguishes three tiers of metrics:

| Category | Source / Mechanism | Label in Application | Example |
| :--- | :--- | :--- | :--- |
| **Trained ML Predictions** | Random Forest models fitted on historical agronomic and market datasets | *"AI Recommended Crop"* & *"Expected Yield"* | Rice (52% confidence), Yield: 9.01 Q/Acre |
| **Market Value Estimation** | Random Forest regressor fitted on APMC Mandi price distributions | *"Estimated market value"* | ₹2,383.00 / Quintal |
| **Planning Estimates** | Benchmark agro-economic assumptions for hackathon MVP | *"Planning estimates / demo assumptions"* | Water: 1200 units, Fert: 120 kg, Cost: ₹18,000/acre |
| **Optimization Outputs** | Deterministic mathematical solution from PuLP linear solver | *"Optimal Crop Allocation"* | Rice: 3.333 Acres, Surplus Land: 1.667 Acres |

---

## 💻 7. Installation & How to Run

### Step 1: Clone or Navigate to Project
```bash
cd C:\SmartCropOptimizer
```

### Step 2: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### Step 4: Run the Test Suite
```bash
python test_all.py
```

### Step 5: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 8. Example Input & Expected Output

### Input Scenario (Prompt Benchmark):
- **Soil**: Nitrogen = 90, Phosphorus = 42, Potassium = 43, pH = 6.5
- **Weather**: Temperature = 25°C, Humidity = 80%, Rainfall = 200 mm
- **Resources**:
  - Land = 5.0 Acres
  - Water = 4000.0 Units
  - Fertilizer = 500.0 kg
  - Budget = ₹100,000.00

### Expected System Outputs:
1. **AI Crop Recommendation**: **Rice** (Confidence: ~52%, Runner-up: Jute).
2. **Expected Yield**: ~9.01 Quintals / Acre (~0.90 Metric Tonnes / Acre).
3. **Estimated Market Price**: ~₹2,383.00 / Quintal.
4. **Optimal Allocation (Rice)**:
   - **Allocated Land**: **3.333 Acres**
   - **Unused Land**: **1.667 Acres** (Remaining fallow)
   - **Water Used**: **3999.6 / 4000.0 Units (100% capacity)**
   - **Fertilizer Used**: **400.0 / 500.0 kg (80%)**
   - **Total Cultivation Cost**: **₹60,000.00** (Surplus Budget: ₹40,000.00)
   - **Expected Total Profit**: **₹11,568.28**
   - **Limiting Resource**: **Irrigation Water** (explaining why only 3.333 acres could be planted).
5. **What-If Scenario**:
   - If water is reduced by 20% (from 4000 to 3200 units), the optimizer dynamically reduces rice cultivation to 2.667 acres, recalculating expected profit to ₹9,255.

---

## 🌐 9. Farmer-Friendly Accessibility & Features
- **Bilingual Interface**: Seamless toggle between **English** and **తెలుగు (Telugu)**.
- **KPI Summary Cards**: Immediate visibility into Net Earnings, Revenue, Cost, and Resource Utilization.
- **Plotly Visuals**:
  1. *Land Allocation Donut Chart* (Allocated vs Surplus Land)
  2. *Resource Utilization Bar Chart* with 100% limit line
  3. *Expected Profit by Crop*
  4. *Revenue vs Cultivation Cost Comparison*
- **What-If Simulator**: Preset buttons for Drought (-20% Water), Capital Infusion (+30% Budget), and Monsoonal Rain (+25% Rainfall) plus real-time continuous sliders.
- **Robust Error Handling**: Never crashes on invalid numbers, negative bounds, or missing features.

---

## ⚠️ 10. Limitations & Future Enhancements

### Current Limitations:
1. **Fixed Planning Estimates**: Per-acre water and fertilizer requirements are baseline planning estimates rather than dynamic daily sensor streams.
2. **Crop Pairing in Mandi Data**: Mandi price data matches 13 primary crops directly; secondary crops use benchmark APMC price averages.
3. **Single Period Model**: Linear programming currently optimizes for a single seasonal cycle (e.g., Kharif) rather than multi-year rotational intercropping.

### Future Enhancements:
- **IoT & Sensor Integration**: Real-time soil moisture sensor and weather station API integration.
- **Dynamic Risk Penalties**: Incorporating Quadratic Programming or CVaR (Conditional Value at Risk) for price volatility.
- **Satellite Remote Sensing**: Ingesting NDVI vegetation indices for post-planting yield tracking.
- **Voice Interface**: Audio-based input for local regional languages.
