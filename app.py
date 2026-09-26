"""
House Price Predictor
Python for Data Science (BE05000231) - Mini Project
Covers: file I/O, pandas/numpy, descriptive statistics, data preparation,
        scikit-learn regression modeling, and data visualization.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Predictor", layout="wide")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (5, 3)
plt.rcParams["font.size"] = 8

st.title("🏠 House Price Predictor")
st.caption("Upload a housing dataset, train a model, and predict prices for new houses.")

# ---------------------------------------------------------
# 1. DATA SOURCE
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a housing CSV (must include a price column)", type=["csv"])

if "house_df" not in st.session_state:
    st.session_state.house_df = None

if uploaded_file is not None:
    st.session_state.house_df = pd.read_csv(uploaded_file)

if st.session_state.house_df is None:
    st.info("No file uploaded — using a built-in synthetic Ahmedabad housing dataset so you can try the app right away.")
    rng = np.random.default_rng(7)
    n = 300

    # Areas around Ahmedabad, each with a different price premium (₹ per sqft add-on)
    area_premium = {
        "Nikol": 0,
        "Naroda": -500,
        "Sindhu Bhavan Road": 3500,
        "South Bopal": 1500,
        "Sola": 800,
        "Gota": 600,
    }
    locations = rng.choice(list(area_premium.keys()), n)

    # Property type, each with its own price multiplier
    type_multiplier = {"Flat": 1.0, "Villa": 1.25, "Bungalow": 1.4}
    property_types = rng.choice(list(type_multiplier.keys()), n, p=[0.6, 0.25, 0.15])

    # Furnishing status, each with its own flat price add-on
    furnish_premium = {"Unfurnished": 0, "Semi-Furnished": 150000, "Furnished": 350000}
    furnishing = rng.choice(list(furnish_premium.keys()), n, p=[0.4, 0.35, 0.25])

    area_sqft = rng.normal(1500, 500, n).clip(400, None)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    age = rng.integers(0, 40, n)

    loc_premium = np.array([area_premium[loc] for loc in locations])
    type_mult = np.array([type_multiplier[t] for t in property_types])
    furnish_add = np.array([furnish_premium[f] for f in furnishing])

    price = (
        (area_sqft * (2500 + loc_premium) * type_mult)  # base rate + location premium, scaled by property type
        + bedrooms * 80000
        + bathrooms * 50000
        - age * 9000
        + furnish_add
        + rng.normal(0, 150000, n)
    ).clip(200000, None).round(0)

    st.session_state.house_df = pd.DataFrame({
        "area_sqft": area_sqft.round(0),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "location": locations,
        "property_type": property_types,
        "furnishing": furnishing,
        "price": price,
    })

df = st.session_state.house_df

st.subheader("1️⃣ Dataset Preview")
st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
st.dataframe(df.head(10), use_container_width=True)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

if not numeric_cols:
    st.error("This dataset has no numeric columns — a regression model needs a numeric price column.")
    st.stop()

target_col = st.selectbox(
    "Which column is the price / target you want to predict?",
    numeric_cols,
    index=numeric_cols.index("price") if "price" in numeric_cols else len(numeric_cols) - 1,
)

available_features = [c for c in df.columns if c != target_col]
feature_cols = st.multiselect(
    "Which columns should be used as features? (place/location is a text column — that's fine)",
    available_features,
    default=available_features,
)

if not feature_cols:
    st.warning("Select at least one feature column to continue.")
    st.stop()

numeric_features = [c for c in feature_cols if c in numeric_cols]
categorical_features = [c for c in feature_cols if c in categorical_cols]

# ---------------------------------------------------------
# 2. DATA QUALITY & CLEANING
# ---------------------------------------------------------
st.subheader("2️⃣ Data Quality")
work_df = df[feature_cols + [target_col]].copy()

missing = work_df.isna().sum().sum()
if missing > 0:
    st.write(f"Found {int(missing)} missing values.")
    if numeric_features:
        work_df[numeric_features] = work_df[numeric_features].fillna(work_df[numeric_features].mean())
    for c in categorical_features:
        mode = work_df[c].mode()
        if not mode.empty:
            work_df[c] = work_df[c].fillna(mode.iloc[0])
else:
    st.write("No missing values found.")

dups = work_df.duplicated().sum()
if dups > 0:
    st.write(f"Found {int(dups)} duplicate rows — removing them.")
    work_df = work_df.drop_duplicates()

# ---------------------------------------------------------
# 3. DESCRIPTIVE STATS & CORRELATION
# ---------------------------------------------------------
st.subheader("3️⃣ Explore the Data")

c1, c2 = st.columns(2)
with c1:
    st.write("Descriptive statistics (numeric columns):")
    if numeric_features:
        st.dataframe(work_df[numeric_features + [target_col]].describe().T, use_container_width=True)
    else:
        st.info("No numeric feature columns selected.")
with c2:
    if len(numeric_features) >= 1:
        fig, ax = plt.subplots()
        sns.heatmap(work_df[numeric_features + [target_col]].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig, use_container_width=False)

if categorical_features:
    loc_col = st.selectbox("See average price by category", categorical_features)
    fig_loc, ax_loc = plt.subplots()
    avg_by_loc = work_df.groupby(loc_col)[target_col].mean().sort_values(ascending=False)
    avg_by_loc.plot(kind="bar", ax=ax_loc, color="#4C72B0")
    ax_loc.set_ylabel(f"Average {target_col}")
    ax_loc.set_title(f"Average {target_col} by {loc_col}")
    st.pyplot(fig_loc, use_container_width=False)

if numeric_features:
    scatter_feature = st.selectbox("Plot a numeric feature against the target", numeric_features)
    fig2, ax2 = plt.subplots()
    ax2.scatter(work_df[scatter_feature], work_df[target_col], alpha=0.6)
    ax2.set_xlabel(scatter_feature)
    ax2.set_ylabel(target_col)
    st.pyplot(fig2, use_container_width=False)

# ---------------------------------------------------------
# 4. TRAIN MODEL
# ---------------------------------------------------------
st.subheader("4️⃣ Train a Model")

model_choice = st.radio("Choose a model", ["Linear Regression", "Random Forest"], horizontal=True)
test_size = st.slider("Test set size (%)", 10, 40, 20) / 100

# One-hot encode categorical (e.g. location/place) columns
X = pd.get_dummies(work_df[feature_cols], columns=categorical_features, drop_first=False)
encoded_columns = X.columns.tolist()
y = work_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if model_choice == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
else:
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)  # tree models don't need scaling
    preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:,.0f}")
m2.metric("RMSE", f"{rmse:,.0f}")
m3.metric("R² score", f"{r2:.3f}")

fig3, ax3 = plt.subplots()
ax3.scatter(y_test, preds, alpha=0.6)
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
ax3.plot(lims, lims, color="red", linestyle="--")
ax3.set_xlabel("Actual price")
ax3.set_ylabel("Predicted price")
ax3.set_title("Actual vs Predicted")
st.pyplot(fig3, use_container_width=False)

if model_choice == "Linear Regression":
    st.write("Feature importance (regression coefficients, standardized):")
    coef_df = pd.DataFrame({"feature": encoded_columns, "coefficient": model.coef_}).sort_values(
        "coefficient", key=abs, ascending=False
    )
    st.dataframe(coef_df, use_container_width=True)
else:
    st.write("Feature importance:")
    imp_df = pd.DataFrame({"feature": encoded_columns, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    st.dataframe(imp_df, use_container_width=True)

# ---------------------------------------------------------
# 5. PREDICT A NEW HOUSE
# ---------------------------------------------------------
st.subheader("5️⃣ Predict a New House Price")
st.write("Enter details for the house:")

input_vals = {}
cols = st.columns(min(3, len(feature_cols)))
for i, feat in enumerate(feature_cols):
    with cols[i % len(cols)]:
        if feat in categorical_features:
            options = sorted(work_df[feat].dropna().unique().tolist())
            input_vals[feat] = st.selectbox(feat, options)
        else:
            default_val = float(work_df[feat].mean())
            input_vals[feat] = st.number_input(feat, value=round(default_val, 2))

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_vals])
    input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=False)
    # align columns with training data (fills missing dummy columns with 0)
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

    if model_choice == "Linear Regression":
        input_scaled = scaler.transform(input_encoded)
        prediction = model.predict(input_scaled)[0]
    else:
        prediction = model.predict(input_encoded)[0]
    st.success(f"Predicted price: **₹{prediction:,.0f}**")

st.divider()
st.caption("Built for BE05000231 - Python for Data Science | GTU Semester 5")
