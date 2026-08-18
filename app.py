from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
TARGET = "Revenue"

NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAL_FEATURES = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}
MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Linear classifier using scaled numerical features and one-hot encoded categorical features.",
    "Decision Tree": "Single decision tree using unscaled numerical features and one-hot encoded categorical features.",
    "kNN": "Distance-based classifier using scaled numerical features.",
    "Gaussian Naive Bayes": "Gaussian Naive Bayes on scaled numerical and one-hot encoded categorical features.",
    "Random Forest (Ensemble)": "Ensemble of 300 decision trees; numerical features are not scaled.",
}

st.set_page_config(page_title="Purchase Intent Model Lab", page_icon="🛒", layout="wide")

@st.cache_resource
def load_models():
    return {name: joblib.load(ROOT / "model" / filename) for name, filename in MODEL_FILES.items()}


def parse_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    true_values = {"true", "1", "1.0", "yes", "y"}
    false_values = {"false", "0", "0.0", "no", "n"}
    parsed = []
    invalid = []
    for value in series:
        if pd.isna(value):
            parsed.append(pd.NA)
            continue
        text = str(value).strip().lower()
        if text in true_values:
            parsed.append(True)
        elif text in false_values:
            parsed.append(False)
        else:
            invalid.append(str(value))
            parsed.append(pd.NA)
    if invalid:
        raise ValueError(f"Column '{column}' has invalid Boolean values: {sorted(set(invalid))[:5]}")
    return pd.Series(parsed, index=series.index, dtype="boolean")


def validate_csv(frame: pd.DataFrame):
    if frame.empty:
        raise ValueError("The CSV contains no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")

    has_target = TARGET in frame.columns
    allowed = set(FEATURES + ([TARGET] if has_target else []))
    missing = [c for c in FEATURES if c not in frame.columns]
    extra = [c for c in frame.columns if c not in allowed]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if extra:
        raise ValueError("Unexpected columns: " + ", ".join(extra))

    X = frame[FEATURES].copy()
    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(X[column], errors="coerce")
        if (converted.isna() & X[column].notna()).any():
            raise ValueError(f"Column '{column}' must be numeric.")
        X[column] = converted

    for column in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        converted = pd.to_numeric(X[column], errors="coerce")
        if (converted.isna() & X[column].notna()).any() or converted.dropna().mod(1).ne(0).any():
            raise ValueError(f"Column '{column}' must contain integer category codes.")
        X[column] = converted

    X["Weekend"] = parse_bool(X["Weekend"], "Weekend")
    X["Month"] = X["Month"].where(X["Month"].isna(), X["Month"].astype(str).str.strip())
    X["VisitorType"] = X["VisitorType"].where(
        X["VisitorType"].isna(), X["VisitorType"].astype(str).str.strip()
    )

    y = None
    if has_target:
        y_bool = parse_bool(frame[TARGET], TARGET)
        if y_bool.isna().any():
            raise ValueError("Revenue must contain a true label for every row in evaluation mode.")
        y = y_bool.astype(int)
    return X, y


def positive_scores(model, X):
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The fitted model does not contain positive class 1.")
    return probabilities[:, classes.index(1)]


def evaluate(model, X, y):
    pred = model.predict(X)
    score = positive_scores(model, X)
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, score),
        "Precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "F1": f1_score(y, pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    cm = confusion_matrix(y, pred, labels=[0, 1])
    report = classification_report(
        y, pred, labels=[0, 1],
        target_names=["No Purchase (0)", "Purchase (1)"],
        output_dict=True, zero_division=0,
    )
    return metrics, cm, report


models = load_models()

st.title("🛒 Purchase Intent Model Lab")
st.write("Compare five classification models that predict whether an online shopping session ends in a purchase.")
a, b, c = st.columns(3)
a.metric("Dataset", "UCI Online Shoppers")
b.metric("Input features", "17")
c.metric("Positive class", "Purchase (Revenue = 1)")

with st.sidebar:
    st.header("Controls")
    selected = st.selectbox("Select model", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    st.download_button(
        "Download bundled test_data.csv",
        data=(ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

try:
    frame = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(ROOT / "test_data.csv")
    X, y = validate_csv(frame)
except Exception as exc:
    st.error(f"CSV validation failed: {exc}")
    st.stop()

st.success(f"Validated {len(X):,} rows.")

if y is not None:
    st.subheader("Model comparison on the supplied labeled test data")
    rows = []
    detailed = {}
    for name, model in models.items():
        metrics, cm, report = evaluate(model, X, y)
        rows.append({"Model": name, **metrics})
        detailed[name] = (metrics, cm, report)
    comparison = pd.DataFrame(rows)
    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        comparison[col] = comparison[col].round(4)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    metrics, cm, report = detailed[selected]
    st.subheader(f"Selected model: {selected}")
    cols = st.columns(6)
    for col, key in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        col.metric(key, f"{metrics[key]:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Purchase", "Actual: Purchase"],
            columns=["Predicted: No Purchase", "Predicted: Purchase"],
        )
        st.dataframe(cm_df, use_container_width=True)
    with right:
        st.markdown("#### Classification report")
        report_df = pd.DataFrame(report).T.loc[
            ["No Purchase (0)", "Purchase (1)", "macro avg", "weighted avg"],
            ["precision", "recall", "f1-score", "support"],
        ].round(4)
        st.dataframe(report_df, use_container_width=True)

    st.info(MODEL_DESCRIPTIONS[selected])
else:
    st.info("Revenue is absent, so evaluation metrics cannot be computed. Prediction-only mode is active.")
    model = models[selected]
    pred = model.predict(X).astype(int)
    score = positive_scores(model, X)
    output = frame.copy()
    output["Predicted_Revenue"] = pred
    output["Purchase_Probability"] = score
    st.dataframe(output.head(25), use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="purchase_intent_predictions.csv",
        mime="text/csv",
    )

st.caption("BITS Pilani WILP Machine Learning Assignment 2 • Richa Tripathi • 2025AC05424")
