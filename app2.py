
import streamlit as st

st.set_page_config(page_title="Streamlit Test")

# st.title("Streamlit is Working!")
# st.success("Your application is running correctly.")
st.write("Waste Classification Project - SDG 12")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Page configuration
st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Waste Classification System")
st.subheader("SDG 12: Responsible Consumption and Production")
st.write(
    "This Machine Learning project predicts waste categories "
    "using waste properties."
)

# Project folder and dataset path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "waste_data.csv"

# Generate a sample dataset if it doesn't exist
@st.cache_data
def generate_dataset():
    rng = np.random.default_rng(42)

    categories = {
        "Organic": {
            "moisture": (55, 95),
            "biodegradable": (75, 100),
            "plastic": (0, 10),
            "paper": (0, 15),
            "metal": (0, 5)
        },
        "Plastic": {
            "moisture": (0, 25),
            "biodegradable": (0, 15),
            "plastic": (70, 100),
            "paper": (0, 15),
            "metal": (0, 10)
        },
        "Paper": {
            "moisture": (5, 35),
            "biodegradable": (65, 100),
            "plastic": (0, 15),
            "paper": (70, 100),
            "metal": (0, 5)
        },
        "Metal": {
            "moisture": (0, 15),
            "biodegradable": (0, 5),
            "plastic": (0, 10),
            "paper": (0, 10),
            "metal": (70, 100)
        }
    }

    records = []

    for category, ranges in categories.items():
        for _ in range(300):
            records.append({
                "moisture_percent": rng.uniform(*ranges["moisture"]),
                "biodegradable_score": rng.uniform(*ranges["biodegradable"]),
                "plastic_content": rng.uniform(*ranges["plastic"]),
                "paper_content": rng.uniform(*ranges["paper"]),
                "metal_content": rng.uniform(*ranges["metal"]),
                "weight_grams": rng.uniform(5, 2000),
                "category": category
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    return df

# Load or create dataset
try:
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        df = generate_dataset()
    else:
        df = pd.read_csv(DATA_PATH)

    required_columns = [
        "moisture_percent",
        "biodegradable_score",
        "plastic_content",
        "paper_content",
        "metal_content",
        "weight_grams",
        "category"
    ]

    if df.empty or any(c not in df.columns for c in required_columns):
        df = generate_dataset()

    df = df[required_columns].dropna()

    if df.empty:
        raise ValueError("Dataset contains no valid records.")

except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()

# Features and target
FEATURES = [
    "moisture_percent",
    "biodegradable_score",
    "plastic_content",
    "paper_content",
    "metal_content",
    "weight_grams"
]

X = df[FEATURES]
y = df["category"]

# Train the ML model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, len(X_train), len(X_test), y_test, predictions

try:
    model, accuracy, train_count, test_count, y_test, predictions = (
        train_model(X, y)
    )
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

# User input form
st.markdown("### Enter Waste Properties")

with st.form("waste_form"):
    moisture = st.slider("Moisture (%)", 0, 100, 70)
    biodegradable = st.slider(
        "Biodegradability score (%)", 0, 100, 80
    )
    plastic = st.slider("Plastic content (%)", 0, 100, 5)
    paper = st.slider("Paper content (%)", 0, 100, 5)
    metal = st.slider("Metal content (%)", 0, 100, 2)
    weight = st.number_input(
        "Weight (grams)",
        min_value=1,
        max_value=10000,
        value=500,
        step=10
    )

    submitted = st.form_submit_button("Classify Waste")

# Prediction
if submitted:
    sample = pd.DataFrame([{
        "moisture_percent": moisture,
        "biodegradable_score": biodegradable,
        "plastic_content": plastic,
        "paper_content": paper,
        "metal_content": metal,
        "weight_grams": weight
    }])

    prediction = model.predict(sample)[0]

    icons = {
        "Organic": "🍃",
        "Plastic": "🧴",
        "Paper": "📄",
        "Metal": "🔩"
    }

    advice = {
        "Organic": "Consider suitable organic waste collection or composting.",
        "Plastic": "Check local rules for plastic recycling.",
        "Paper": "Keep paper dry and recycle where accepted.",
        "Metal": "Use an appropriate metal recycling collection."
    }

    st.success(
        f"Predicted Waste Category: {icons.get(prediction, '♻️')} {prediction}"
    )
    st.write("**Suggested action:**", advice.get(prediction, "Check local rules."))

    # Show model scores
    probabilities = model.predict_proba(sample)[0]

    score_df = pd.DataFrame({
        "Category": model.classes_,
        "Model score (%)": np.round(probabilities * 100, 2)
    }).sort_values("Model score (%)", ascending=False)

    st.markdown("### Model scores")
    st.bar_chart(score_df.set_index("Category"))

    st.caption(
        "Model scores are not guaranteed to represent real-world probabilities. "
        "This is an educational model trained on synthetic data."
    )

# Model evaluation
st.divider()
st.markdown("### Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Training Records", train_count)
col3.metric("Testing Records", test_count)

with st.expander("View Classification Report"):
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose())

with st.expander("View Dataset"):
    st.dataframe(df.head(30), use_container_width=True)
    st.write("Total records:", len(df))

with st.expander("About the Project"):
    st.write("**Algorithm:** Decision Tree Classifier")
    st.write("**Dataset:** 1,200 synthetic sample records")
    st.write("**Inputs:** Six numerical waste properties")
    st.write("**Output:** Organic, Plastic, Paper, or Metal")
    st.write("**SDG:** 12 - Responsible Consumption and Production")

    st.warning(
        "The dataset is artificially generated for demonstration. "
        "Real-world waste classification requires a real, labelled dataset "
        "and testing with actual samples."
    )
