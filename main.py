import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide", page_title="House Price Prediction Demo")

st.title("House Price Prediction System")

# -----------------------
# Friendly feature descriptions
# -----------------------
FEATURE_DESCRIPTIONS = {
    "CRIM": "Per capita crime rate by town",
    "ZN": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
    "INDUS": "Proportion of non-retail business acres per town",
    "CHAS": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
    "NOX": "Nitric oxides concentration (parts per 10 million)",
    "RM": "Average number of rooms per dwelling",
    "AGE": "Proportion of owner-occupied units built prior to 1940",
    "DIS": "Weighted distances to five Boston employment centres",
    "RAD": "Index of accessibility to radial highways",
    "TAX": "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B": "1000(Bk - 0.63)^2 where Bk is proportion of blacks by town",
    "LSTAT": "% lower status of the population",
    "MEDV": "Median value of owner-occupied homes in $1000's (target)",
}

def nice_label(col):
    desc = FEATURE_DESCRIPTIONS.get(col, "")
    return f"{col} — {desc}" if desc else col

# -----------------------
# Load dataset
# -----------------------
try:
    df = pd.read_csv("data.csv")
except Exception:
    df = None

# -----------------------
# Load model
# -----------------------
MODEL = None
MODEL_PATHS = ["housepriceprediction.joblib"]
MODEL_PATH_USED = None

for p in MODEL_PATHS:
    try:
        MODEL = joblib.load(p)
        MODEL_PATH_USED = p
        break
    except Exception as e:
        MODEL = None
        model_load_err = (p, e)

# -----------------------
# Feature inference
# -----------------------
def infer_feature_names(model):
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        if hasattr(model, "get_feature_names_out"):
            return list(model.get_feature_names_out())
    except Exception:
        pass

    try:
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
        if hasattr(model, "steps"):
            final = model.steps[-1][1]
            if hasattr(final, "feature_names_in_"):
                return list(final.feature_names_in_)
    except Exception:
        pass

    return None

MODEL_FEATURES = infer_feature_names(MODEL) if MODEL else None

# -----------------------
# Dynamic EDA
# -----------------------
st.markdown("## Dynamic EDA")
if df is None:
    st.info("No dataset loaded for EDA. Upload a CSV in the app folder named 'data.csv'.")
else:
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    c1, _ = st.columns([2, 1])
    with c1:
        st.dataframe(df.head())

    with st.form("eda_controls"):
        plot_type = st.selectbox("Select plot type", ["Histogram", "Bar (counts)", "Boxplot", "Scatter", "Correlation heatmap"])
        if plot_type in ("Histogram", "Bar (counts)", "Boxplot"):
            feat = st.selectbox("Feature", cols, format_func=nice_label)
        elif plot_type == "Scatter":
            x_feat = st.selectbox("X feature (numeric)", numeric_cols, index=numeric_cols.index("RM") if "RM" in numeric_cols else 0, format_func=nice_label)
            y_feat = st.selectbox("Y feature (numeric)", numeric_cols, index=numeric_cols.index("MEDV") if "MEDV" in numeric_cols else (1 if len(numeric_cols) > 1 else 0), format_func=nice_label)
            color_feat = st.selectbox("Optional color / category (or 'None')", ["None"] + cols, index=0, format_func=lambda x: "None" if x=="None" else nice_label(x))
        else:
            corr_subset = st.multiselect("Select features for correlation (empty = use all numeric)", numeric_cols, default=numeric_cols[:10], format_func=nice_label)

        bins = st.slider("Bins (for histogram)", min_value=5, max_value=100, value=30)
        submit_eda = st.form_submit_button("Show plot")

    if submit_eda:
        try:
            if plot_type == "Histogram":
                if feat not in numeric_cols:
                    st.error("Histogram requires numeric feature.")
                else:
                    fig = px.histogram(df, x=feat, nbins=bins, title=f"Histogram of {nice_label(feat)}")
                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Bar (counts)":
                vc = df[feat].value_counts()
                fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={'x': feat, 'y':'count'}, title=f"Counts of {nice_label(feat)}")
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Boxplot":
                group_by = st.selectbox("Group by (optional) - for multiple boxes (or 'None')", ["None"] + cols, index=0, format_func=lambda x: "None" if x=="None" else nice_label(x))
                if group_by == "None" or group_by == feat:
                    fig = px.box(df, y=feat, points="all", title=f"Boxplot of {nice_label(feat)}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.box(df, x=group_by, y=feat, points="outliers", title=f"Boxplot of {nice_label(feat)} by {nice_label(group_by)}")
                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Scatter":
                if x_feat == y_feat:
                    st.error("Pick two different numeric features for scatter.")
                else:
                    if color_feat != "None":
                        fig = px.scatter(df, x=x_feat, y=y_feat, color=color_feat, title=f"{nice_label(y_feat)} vs {nice_label(x_feat)} colored by {nice_label(color_feat)}", trendline="ols")
                    else:
                        fig = px.scatter(df, x=x_feat, y=y_feat, title=f"{nice_label(y_feat)} vs {nice_label(x_feat)}", trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Correlation heatmap":
                use_feats = corr_subset if corr_subset else numeric_cols
                if len(use_feats) < 2:
                    st.error("Select at least 2 numeric features for correlation.")
                else:
                    corr = df[use_feats].corr()
                    fig, ax = plt.subplots(figsize=(8,6))
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", ax=ax, vmin=-1, vmax=1)
                    st.pyplot(fig)
        except Exception:
            st.error("Error creating plot. See app logs for details.")
            with st.expander("Plot error traceback"):
                st.text(traceback.format_exc())



# -----------------------
# Prediction section
# -----------------------
st.markdown("## Prediction")
if MODEL is None:
    st.info("Model not loaded. Place 'housepriceprediction.joblib' in the app folder.")
else:
    # exclude MEDV
    suggested_features = [f for f in MODEL_FEATURES if f != 'MEDV'] if MODEL_FEATURES else [f for f in df.select_dtypes(include=[np.number]).columns if f != 'MEDV']

    selected_features = st.multiselect("Select features for prediction (order should match model)", options=suggested_features, default=suggested_features)

    if selected_features:
        with st.form("predict_form"):
            st.write("Enter feature values (defaults = column median if dataset available):")
            user_data = {}
            for i in range(0, len(selected_features), 2):
                colA, colB = st.columns(2)
                for j, col_widget in enumerate((colA, colB)):
                    try:
                        f = selected_features[i+j]
                    except IndexError:
                        break

                    # numeric input defaults
                    if df is not None and f in df.columns:
                        series = df[f].dropna()
                        if pd.api.types.is_numeric_dtype(series):
                            col_median = float(series.median())
                            col_min = float(series.min())
                            col_max = float(series.max())
                        else:
                            col_median = 0.0
                            col_min, col_max = -1e9, 1e9
                    else:
                        col_median, col_min, col_max = 0.0, -1e9, 1e9

                    # small-cardinality or binary
                    if df is not None and f in df.columns:
                        uniq = np.unique(df[f].dropna())
                        if set(uniq).issubset({0,1}) or len(uniq)<=6:
                            opts = list(map(float, np.unique(uniq)))
                            user_data[f] = col_widget.selectbox(nice_label(f), options=opts, index=0, key=f)
                            continue

                    user_data[f] = col_widget.number_input(nice_label(f), value=col_median, min_value=col_min, max_value=col_max, format="%.6f", key=f)

            submit_pred = st.form_submit_button("Predict")

        if submit_pred:
            X_input = pd.DataFrame([user_data], columns=selected_features)
            st.write("Input values:")
            st.table(X_input.T.rename(columns={0:"value"}))

            # try prediction and show error traceback if fails
            try:
                try:
                    pred = MODEL.predict(X_input)
                except Exception:
                    pred = MODEL.predict(X_input.values)
                pred_val = pred[0] if hasattr(pred, "__len__") else pred
            except Exception:
                st.error("Model prediction failed. See details below.")
                with st.expander("Prediction traceback"):
                    st.text(traceback.format_exc())
            else:
                st.success(f"Predicted value: **{pred_val:.4f}**" if isinstance(pred_val, (int,float,np.integer,np.floating)) else f"Prediction: **{pred_val}**")

