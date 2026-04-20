import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
pip install plotly
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="MeritAI - Fair Decision Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


SENSITIVE_PATTERNS = {
    "Gender": [
        "gender",
        "sex",
        "male",
        "female",
        "pronoun",
    ],
    "Religion": [
        "religion",
        "faith",
        "belief",
        "caste",
        "sect",
    ],
    "Ethnicity / Race": [
        "ethnicity",
        "ethnic",
        "race",
        "racial",
        "tribe",
        "nationality",
        "origin",
    ],
    "Location / Region": [
        "location",
        "region",
        "city",
        "state",
        "country",
        "zip",
        "postal",
        "district",
        "branch",
        "office",
        "geography",
    ],
}

MERIT_PATTERNS = [
    "performance",
    "experience",
    "skill",
    "competency",
    "assessment",
    "score",
    "rating",
    "tenure",
    "project",
    "achievement",
    "certification",
    "qualification",
    "salary",
    "income",
    "compensation",
    "role",
    "grade",
    "band",
    "productivity",
    "revenue",
    "education",
    "training",
    "kpi",
]

IDENTIFIER_PATTERNS = [
    "id",
    "identifier",
    "employee_number",
    "candidate_number",
    "serial",
    "uuid",
]

POSITIVE_TARGET_HINTS = [
    "hire",
    "hired",
    "selected",
    "promot",
    "retain",
    "approved",
    "shortlist",
    "success",
    "decision",
    "offer",
]

NEGATIVE_TARGET_HINTS = [
    "attrition",
    "leave",
    "left",
    "terminate",
    "resign",
    "exit",
    "churn",
]

NEGATIVE_MERIT_HINTS = [
    "absence",
    "lateness",
    "complaint",
    "warning",
    "disciplinary",
    "error",
    "risk",
]


@dataclass
class RunArtifacts:
    df: pd.DataFrame
    cleaned_df: pd.DataFrame
    sensitive_map: Dict[str, str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    training_features: List[str]
    merit_features: List[str]
    merit_weights: Dict[str, float]
    results_df: pd.DataFrame
    fairness_report: pd.DataFrame
    tradeoff_df: pd.DataFrame
    baseline_threshold: float
    optimized_threshold: float
    baseline_accuracy: float
    optimized_accuracy: float
    baseline_gap: float
    optimized_gap: float


def slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].astype(str).str.strip()
            cleaned[col] = cleaned[col].replace(
                {"": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan, "na": np.nan}
            )
            numeric_candidate = pd.to_numeric(cleaned[col], errors="coerce")
            if numeric_candidate.notna().mean() >= 0.8:
                cleaned[col] = numeric_candidate
    return cleaned.drop_duplicates().reset_index(drop=True)


def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def detect_sensitive_attributes(df: pd.DataFrame) -> Dict[str, str]:
    sensitive_map: Dict[str, str] = {}
    for col in df.columns:
        lowered = slug(col)
        for label, patterns in SENSITIVE_PATTERNS.items():
            if any(pattern in lowered for pattern in patterns) and label not in sensitive_map:
                sensitive_map[label] = col
    return sensitive_map


def infer_positive_label(series: pd.Series, target_name: str):
    non_null = series.dropna()
    unique_vals = list(pd.unique(non_null))
    if len(unique_vals) != 2:
        return None

    lowered_target = slug(target_name)
    string_map = {val: str(val).strip().lower() for val in unique_vals}
    positive_tokens = {
        "1",
        "true",
        "yes",
        "y",
        "selected",
        "hired",
        "hire",
        "promoted",
        "retained",
        "approved",
        "accepted",
        "offer",
        "placed",
        "success",
    }
    negative_tokens = {
        "0",
        "false",
        "no",
        "n",
        "rejected",
        "denied",
        "declined",
        "left",
        "attrition",
        "exit",
        "terminated",
        "failed",
    }

    if any(token in lowered_target for token in NEGATIVE_TARGET_HINTS):
        for val, txt in string_map.items():
            if txt in negative_tokens or txt in {"1", "true", "yes", "y"}:
                return val

    for val, txt in string_map.items():
        if txt in positive_tokens:
            return val

    numeric_unique = pd.to_numeric(pd.Series(unique_vals), errors="coerce")
    if numeric_unique.notna().all():
        return max(unique_vals)
    return unique_vals[-1]


def encode_binary_target(series: pd.Series, target_name: str) -> Tuple[pd.Series, str]:
    positive_label = infer_positive_label(series, target_name)
    if positive_label is None:
        raise ValueError("Target variable must be binary for classification.")
    encoded = (series == positive_label).astype(int)
    return encoded, str(positive_label)


def auto_detect_merit_features(
    df: pd.DataFrame,
    candidate_cols: List[str],
    target: pd.Series,
) -> Tuple[List[str], Dict[str, float]]:
    merit_cols = []
    for col in candidate_cols:
        lowered = slug(col)
        if any(pattern in lowered for pattern in IDENTIFIER_PATTERNS):
            continue
        if any(pattern in lowered for pattern in MERIT_PATTERNS):
            merit_cols.append(col)

    if not merit_cols:
        numeric_candidates = [
            col
            for col in candidate_cols
            if pd.api.types.is_numeric_dtype(df[col]) and not any(pattern in slug(col) for pattern in IDENTIFIER_PATTERNS)
        ]
        fallback_count = min(6, len(numeric_candidates))
        merit_cols = numeric_candidates[:fallback_count]
        if not merit_cols:
            merit_cols = [
                col for col in candidate_cols if not any(pattern in slug(col) for pattern in IDENTIFIER_PATTERNS)
            ][: min(6, len(candidate_cols))]

    weights: Dict[str, float] = {}
    for col in merit_cols:
        series = df[col]
        score = 0.05
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce").fillna(series.median() if series.notna().any() else 0)
            if numeric_series.nunique() > 1:
                corr = np.corrcoef(numeric_series, target)[0, 1]
                if np.isfinite(corr):
                    score = abs(float(corr))
        else:
            factorized = pd.factorize(series.fillna("Missing"))[0]
            if len(np.unique(factorized)) > 1:
                corr = np.corrcoef(factorized, target)[0, 1]
                if np.isfinite(corr):
                    score = abs(float(corr))
        weights[col] = max(score, 0.05)

    weight_sum = sum(weights.values()) or 1.0
    normalized = {col: value / weight_sum for col, value in weights.items()}
    return merit_cols, normalized


def calculate_merit_score(df: pd.DataFrame, merit_features: List[str], merit_weights: Dict[str, float]) -> pd.Series:
    if not merit_features:
        return pd.Series(np.zeros(len(df)), index=df.index)

    merit_matrix = pd.DataFrame(index=df.index)
    for col in merit_features:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").astype(float)
            fill_value = values.median() if values.notna().any() else 0.0
            values = values.fillna(fill_value)
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                scaled = (values - min_val) / (max_val - min_val)
            else:
                scaled = pd.Series(0.5, index=df.index)
        else:
            codes = pd.Series(pd.factorize(series.fillna("Missing"))[0], index=df.index).astype(float)
            min_val, max_val = codes.min(), codes.max()
            if max_val > min_val:
                scaled = (codes - min_val) / (max_val - min_val)
            else:
                scaled = pd.Series(0.5, index=df.index)

        lowered = slug(col)
        if any(token in lowered for token in NEGATIVE_MERIT_HINTS):
            scaled = 1 - scaled
        merit_matrix[col] = scaled

    weighted = sum(merit_matrix[col] * merit_weights.get(col, 0) for col in merit_features)
    return (100 * weighted).round(2)


def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series, merit_score: pd.Series) -> pd.DataFrame:
    metric_rows = []
    data = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "group": sensitive.fillna("Missing").astype(str),
            "merit_score": merit_score,
        }
    )
    high_merit_cutoff = data["merit_score"].quantile(0.75)

    for group, frame in data.groupby("group"):
        selection_rate = frame["y_pred"].mean() if len(frame) else np.nan
        positives = frame[frame["y_true"] == 1]
        true_positive_rate = positives["y_pred"].mean() if len(positives) else np.nan
        high_merit = frame[frame["merit_score"] >= high_merit_cutoff]
        high_merit_rate = high_merit["y_pred"].mean() if len(high_merit) else np.nan
        metric_rows.append(
            {
                "group": group,
                "count": len(frame),
                "selection_rate": selection_rate,
                "equal_opportunity_rate": true_positive_rate,
                "high_merit_selection_rate": high_merit_rate,
            }
        )
    return pd.DataFrame(metric_rows)


def summarize_gap(metrics_df: pd.DataFrame, metric_col: str) -> float:
    valid = metrics_df[metric_col].dropna()
    if valid.empty:
        return 0.0
    return float(valid.max() - valid.min())


def evaluate_fairness_for_attribute(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    merit_score: pd.Series,
    sensitive: pd.Series,
    threshold: float,
    attribute_label: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics_df = compute_group_metrics(y_true, y_pred, sensitive, merit_score)
    gaps = {
        "Demographic Parity Gap": summarize_gap(metrics_df, "selection_rate"),
        "Equal Opportunity Gap": summarize_gap(metrics_df, "equal_opportunity_rate"),
        "Merit-Based Gap": summarize_gap(metrics_df, "high_merit_selection_rate"),
    }
    metrics_df.insert(0, "attribute", attribute_label)
    return metrics_df, gaps


def aggregate_fairness_gap(fairness_summary: pd.DataFrame) -> float:
    cols = ["Demographic Parity Gap", "Equal Opportunity Gap", "Merit-Based Gap"]
    return float(fairness_summary[cols].mean().mean()) if not fairness_summary.empty else 0.0


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    merit_score: pd.Series,
    sensitive_map: Dict[str, pd.Series],
) -> Tuple[float, pd.DataFrame]:
    thresholds = np.round(np.linspace(0.2, 0.8, 25), 2)
    records = []

    baseline_pred = (y_prob >= 0.5).astype(int)
    baseline_accuracy = accuracy_score(y_true, baseline_pred)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        gap_values = []
        for label, sensitive in sensitive_map.items():
            _, gaps = evaluate_fairness_for_attribute(y_true, y_prob, merit_score, sensitive, threshold, label)
            gap_values.append(np.mean(list(gaps.values())))

        overall_gap = float(np.mean(gap_values)) if gap_values else 0.0
        objective = overall_gap + 0.35 * max(0.0, baseline_accuracy - accuracy)
        records.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "fairness_gap": overall_gap,
                "objective": objective,
            }
        )

    tradeoff_df = pd.DataFrame(records)
    best_row = tradeoff_df.sort_values(["objective", "fairness_gap", "accuracy"], ascending=[True, True, False]).iloc[0]
    return float(best_row["threshold"]), tradeoff_df


def build_demo_dataset(rows: int = 350) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    genders = rng.choice(["Female", "Male", "Non-Binary"], rows, p=[0.45, 0.48, 0.07])
    religions = rng.choice(["Hindu", "Muslim", "Christian", "Sikh", "Other"], rows)
    ethnicities = rng.choice(["Group A", "Group B", "Group C", "Group D"], rows)
    locations = rng.choice(["North", "South", "East", "West", "Central"], rows)
    experience = rng.integers(0, 16, rows)
    performance = np.clip(rng.normal(75, 12, rows), 30, 100).round(1)
    skill_score = np.clip(rng.normal(72, 15, rows), 20, 100).round(1)
    assessment = np.clip(rng.normal(70, 14, rows), 25, 100).round(1)
    salary = rng.integers(28000, 145000, rows)
    role_level = rng.choice(["Analyst", "Associate", "Senior Associate", "Lead", "Manager"], rows)
    education = rng.choice(["Bachelor", "Master", "MBA", "PhD"], rows, p=[0.46, 0.31, 0.15, 0.08])
    tenure = np.clip(experience + rng.normal(1.5, 2.0, rows), 0, None).round(1)

    merit_signal = (
        0.30 * (performance / 100)
        + 0.22 * (skill_score / 100)
        + 0.20 * (assessment / 100)
        + 0.14 * np.clip(experience / 15, 0, 1)
        + 0.07 * np.clip(tenure / 15, 0, 1)
        + 0.07 * MinMaxScaler().fit_transform(salary.reshape(-1, 1)).flatten()
    )
    hired_prob = 1 / (1 + np.exp(-(merit_signal * 5 - 2.6)))
    hired = (rng.random(rows) < hired_prob).astype(int)
    attrition_prob = 1 / (1 + np.exp(-((1 - merit_signal) * 4 - 1.5)))
    attrition = (rng.random(rows) < attrition_prob).astype(int)

    return pd.DataFrame(
        {
            "Candidate_ID": [f"CAND-{1000 + i}" for i in range(rows)],
            "Gender": genders,
            "Religion": religions,
            "Ethnicity": ethnicities,
            "Location": locations,
            "Years_Experience": experience,
            "Performance_Rating": performance,
            "Skill_Score": skill_score,
            "Assessment_Score": assessment,
            "Annual_Income": salary,
            "Role_Level": role_level,
            "Education": education,
            "Tenure_Years": tenure,
            "Hiring_Decision": hired,
            "Attrition": attrition,
        }
    )


def run_pipeline(raw_df: pd.DataFrame, target_col: str) -> RunArtifacts:
    cleaned_df = clean_dataframe(raw_df)
    sensitive_map = detect_sensitive_attributes(cleaned_df)
    numeric_cols, categorical_cols = detect_column_types(cleaned_df)

    if target_col not in cleaned_df.columns:
        raise ValueError("Selected target column was not found in the dataset.")

    y, _ = encode_binary_target(cleaned_df[target_col], target_col)

    excluded = set(sensitive_map.values()) | {target_col}
    feature_candidates = [col for col in cleaned_df.columns if col not in excluded]
    if not feature_candidates:
        raise ValueError("No usable training features remain after removing sensitive attributes and target.")

    merit_features, merit_weights = auto_detect_merit_features(cleaned_df, feature_candidates, y)
    training_features = merit_features if merit_features else feature_candidates

    preprocessor, _, _ = build_preprocessor(cleaned_df, training_features)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        cleaned_df[training_features],
        y,
        cleaned_df.index,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    merit_score = calculate_merit_score(cleaned_df, merit_features, merit_weights)
    test_merit = merit_score.loc[idx_test]

    audit_sensitive_map = {
        label: cleaned_df.loc[idx_test, col].reset_index(drop=True)
        for label, col in sensitive_map.items()
    }
    y_test_array = y_test.reset_index(drop=True).to_numpy()
    y_prob_array = pd.Series(y_prob).reset_index(drop=True).to_numpy()
    test_merit = test_merit.reset_index(drop=True)

    optimized_threshold, tradeoff_df = optimize_threshold(
        y_test_array,
        y_prob_array,
        test_merit,
        audit_sensitive_map,
    )

    fairness_rows = []
    baseline_pred = (y_prob_array >= 0.5).astype(int)
    optimized_pred = (y_prob_array >= optimized_threshold).astype(int)

    for label, sensitive_series in audit_sensitive_map.items():
        detail_df, base_gaps = evaluate_fairness_for_attribute(
            y_test_array, y_prob_array, test_merit, sensitive_series, 0.5, label
        )
        _, opt_gaps = evaluate_fairness_for_attribute(
            y_test_array, y_prob_array, test_merit, sensitive_series, optimized_threshold, label
        )
        fairness_rows.append(
            {
                "Sensitive Attribute": label,
                "Column Used Internally": sensitive_map[label],
                "Demographic Parity Gap": round(base_gaps["Demographic Parity Gap"], 4),
                "Equal Opportunity Gap": round(base_gaps["Equal Opportunity Gap"], 4),
                "Merit-Based Gap": round(base_gaps["Merit-Based Gap"], 4),
                "Optimized DP Gap": round(opt_gaps["Demographic Parity Gap"], 4),
                "Optimized EO Gap": round(opt_gaps["Equal Opportunity Gap"], 4),
                "Optimized Merit Gap": round(opt_gaps["Merit-Based Gap"], 4),
            }
        )
    fairness_report = pd.DataFrame(fairness_rows)
    baseline_gap = (
        float(fairness_report[["Demographic Parity Gap", "Equal Opportunity Gap", "Merit-Based Gap"]].mean().mean())
        if not fairness_report.empty
        else 0.0
    )
    optimized_gap = (
        float(fairness_report[["Optimized DP Gap", "Optimized EO Gap", "Optimized Merit Gap"]].mean().mean())
        if not fairness_report.empty
        else 0.0
    )

    results_df = cleaned_df.loc[idx_test].copy()
    results_df["Actual"] = y_test_array
    results_df["Prediction_Probability"] = np.round(y_prob_array, 4)
    results_df["Baseline_Prediction"] = baseline_pred
    results_df["Optimized_Prediction"] = optimized_pred
    results_df["Merit_Score"] = test_merit.to_numpy()

    return RunArtifacts(
        df=raw_df,
        cleaned_df=cleaned_df,
        sensitive_map=sensitive_map,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        training_features=training_features,
        merit_features=merit_features,
        merit_weights=merit_weights,
        results_df=results_df.reset_index(drop=True),
        fairness_report=fairness_report,
        tradeoff_df=tradeoff_df,
        baseline_threshold=0.5,
        optimized_threshold=optimized_threshold,
        baseline_accuracy=float(accuracy_score(y_test_array, baseline_pred)),
        optimized_accuracy=float(accuracy_score(y_test_array, optimized_pred)),
        baseline_gap=baseline_gap,
        optimized_gap=optimized_gap,
    )


def render_header() -> None:
    st.markdown(
        """
        <style>
            .main {
                background:
                    radial-gradient(circle at top left, rgba(31, 111, 235, 0.08), transparent 28%),
                    radial-gradient(circle at top right, rgba(14, 165, 140, 0.12), transparent 30%),
                    linear-gradient(180deg, #f7fbff 0%, #eef7f4 100%);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
            }
            .hero {
                padding: 1.6rem 1.8rem;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(9, 44, 76, 0.97), rgba(17, 94, 89, 0.96));
                color: white;
                box-shadow: 0 24px 70px rgba(9, 44, 76, 0.18);
                margin-bottom: 1.2rem;
            }
            .mini-card {
                background: rgba(255, 255, 255, 0.88);
                border: 1px solid rgba(15, 76, 117, 0.1);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 28px rgba(15, 76, 117, 0.08);
            }
            .ethics-note {
                padding: 0.9rem 1rem;
                border-left: 4px solid #0f766e;
                background: rgba(15, 118, 110, 0.08);
                border-radius: 12px;
                margin: 0.8rem 0 1rem 0;
            }
        </style>
        <div class="hero">
            <h1 style="margin:0; font-size:2.2rem;">MeritAI</h1>
            <p style="margin:0.45rem 0 0 0; font-size:1.05rem;">
                Multi-Dimensional Fair Decision Intelligence Platform for hiring, promotions, retention, and workforce analytics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="ethics-note">
            <strong>Ethical design guarantee:</strong> sensitive attributes are automatically detected and excluded from model decisions.
            They are retained only inside the audit layer to measure fairness and support responsible AI governance.
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div style="font-size:0.86rem; color:#355070;">{title}</div>
            <div style="font-size:1.7rem; font-weight:700; color:#0b2545; margin-top:0.2rem;">{value}</div>
            <div style="font-size:0.82rem; color:#5c677d; margin-top:0.25rem;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_fairness_chart(fairness_df: pd.DataFrame) -> go.Figure:
    melted = fairness_df.melt(
        id_vars="Sensitive Attribute",
        value_vars=["Demographic Parity Gap", "Equal Opportunity Gap", "Merit-Based Gap"],
        var_name="Metric",
        value_name="Gap",
    )
    fig = px.bar(
        melted,
        x="Sensitive Attribute",
        y="Gap",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#1d4ed8", "#0f766e", "#f97316"],
    )
    fig.update_layout(
        title="Baseline Fairness Metrics Across Sensitive Dimensions",
        yaxis_title="Gap (lower is better)",
        xaxis_title="Sensitive Attribute",
        legend_title="Metric",
    )
    return fig


def build_tradeoff_chart(tradeoff_df: pd.DataFrame, optimized_threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tradeoff_df["threshold"],
            y=tradeoff_df["accuracy"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#1d4ed8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tradeoff_df["threshold"],
            y=tradeoff_df["fairness_gap"],
            mode="lines+markers",
            name="Average Fairness Gap",
            line=dict(color="#0f766e", width=3),
            yaxis="y2",
        )
    )
    fig.add_vline(x=optimized_threshold, line_dash="dash", line_color="#f97316")
    fig.update_layout(
        title="Fairness vs Accuracy Threshold Trade-off",
        xaxis_title="Decision Threshold",
        yaxis=dict(title="Accuracy"),
        yaxis2=dict(title="Fairness Gap", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1, x=0.02),
    )
    return fig


def build_selection_chart(selection_df: pd.DataFrame, sensitive_col: str) -> go.Figure:
    chart_df = (
        selection_df.groupby(sensitive_col, dropna=False)["Selected"]
        .mean()
        .reset_index()
        .rename(columns={"Selected": "Selection Rate", sensitive_col: "Group"})
    )
    chart_df["Selection Rate"] = chart_df["Selection Rate"].fillna(0)
    fig = px.bar(
        chart_df,
        x="Group",
        y="Selection Rate",
        color="Selection Rate",
        color_continuous_scale=["#dbeafe", "#60a5fa", "#1d4ed8"],
    )
    fig.update_layout(
        title=f"Selection Distribution by {sensitive_col}",
        xaxis_title=sensitive_col,
        yaxis_title="Selected Share",
        coloraxis_showscale=False,
    )
    return fig


render_header()

with st.sidebar:
    st.header("Data Input")
    use_demo = st.toggle("Use built-in demo HR dataset", value=True)
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"], disabled=use_demo)
    st.caption("MeritAI adapts to HR-style datasets and keeps sensitive attributes hidden from model training.")

if use_demo:
    input_df = build_demo_dataset()
else:
    if uploaded_file is None:
        st.info("Upload a CSV file or enable the demo dataset to start the fairness audit.")
        st.stop()
    input_df = pd.read_csv(uploaded_file)

cleaned_preview = clean_dataframe(input_df)
auto_sensitive = detect_sensitive_attributes(cleaned_preview)
eligible_targets = [col for col in cleaned_preview.columns if col not in auto_sensitive.values()]

if not eligible_targets:
    st.error("No target candidates are available because every column was identified as sensitive.")
    st.stop()

default_target_index = 0
for idx, candidate in enumerate(eligible_targets):
    lowered = slug(candidate)
    if any(token in lowered for token in POSITIVE_TARGET_HINTS + NEGATIVE_TARGET_HINTS):
        default_target_index = idx
        break

st.subheader("Data Input")
left, right = st.columns([1.2, 1])
with left:
    st.dataframe(cleaned_preview.head(12), use_container_width=True)
with right:
    st.markdown("### Dataset Profile")
    st.write(f"Rows: **{len(cleaned_preview):,}**")
    st.write(f"Columns: **{len(cleaned_preview.columns)}**")
    numeric_cols, categorical_cols = detect_column_types(cleaned_preview)
    st.write(f"Numeric features: **{len(numeric_cols)}**")
    st.write(f"Categorical features: **{len(categorical_cols)}**")
    target_col = st.selectbox(
        "Select target variable",
        options=eligible_targets,
        index=default_target_index,
        help="Sensitive columns are automatically excluded from this list.",
    )

target_unique = cleaned_preview[target_col].dropna().nunique()
if target_unique != 2:
    st.error(
        f"The selected target `{target_col}` has {target_unique} unique values. MeritAI currently requires a binary target for classification."
    )
    st.stop()

artifacts = run_pipeline(cleaned_preview, target_col)

st.subheader("Governance Analysis")
gov_cols = st.columns(4)
with gov_cols[0]:
    metric_card("Sensitive attributes detected", str(len(artifacts.sensitive_map)), "Automatically governed inside the audit layer.")
with gov_cols[1]:
    metric_card("Model training features", str(len(artifacts.training_features)), "Sensitive attributes excluded from learning.")
with gov_cols[2]:
    metric_card("Merit features used", str(len(artifacts.merit_features)), "Automatically weighted to build merit score.")
with gov_cols[3]:
    metric_card("Target outcome", target_col, "User-selected decision variable.")

gov_left, gov_right = st.columns([1.1, 1])
with gov_left:
    st.markdown("### Sensitive Attribute Governance")
    if artifacts.sensitive_map:
        governance_df = pd.DataFrame(
            [
                {
                    "Sensitive Dimension": label,
                    "Detected Column": column,
                    "Decision Usage": "Excluded from training",
                    "Audit Usage": "Used internally for fairness metrics",
                }
                for label, column in artifacts.sensitive_map.items()
            ]
        )
        st.dataframe(governance_df, use_container_width=True)
    else:
        st.warning("No sensitive attributes were automatically detected from column names. The audit will be limited until such fields are present in the CSV.")

with gov_right:
    st.markdown("### Merit Score Design")
    merit_weight_df = pd.DataFrame(
        {
            "Merit Feature": list(artifacts.merit_weights.keys()),
            "Weight": [round(value, 4) for value in artifacts.merit_weights.values()],
        }
    ).sort_values("Weight", ascending=False)
    st.dataframe(merit_weight_df, use_container_width=True)
    st.caption("Weights are inferred automatically from feature relevance while keeping sensitive attributes fully separated from the model.")

st.subheader("Fairness Metrics")
metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card("Baseline accuracy", f"{artifacts.baseline_accuracy:.3f}", "Classification performance at threshold 0.50.")
with metric_cols[1]:
    metric_card("Optimized accuracy", f"{artifacts.optimized_accuracy:.3f}", "Performance after fairness-aware threshold tuning.")
with metric_cols[2]:
    metric_card("Baseline fairness gap", f"{artifacts.baseline_gap:.3f}", "Average of all fairness gaps before optimization.")
with metric_cols[3]:
    metric_card("Optimized fairness gap", f"{artifacts.optimized_gap:.3f}", "Average of all fairness gaps after optimization.")

fair_left, fair_right = st.columns([1.15, 1])
with fair_left:
    if not artifacts.fairness_report.empty:
        st.plotly_chart(build_fairness_chart(artifacts.fairness_report), use_container_width=True)
    else:
        st.info("Fairness charts will appear when the uploaded dataset includes detectable sensitive attributes.")
with fair_right:
    st.markdown("### Multi-Dimensional Bias Audit")
    if not artifacts.fairness_report.empty:
        st.dataframe(artifacts.fairness_report, use_container_width=True)
    else:
        st.info("No governed sensitive columns were detected, so the bias audit report is currently empty.")

st.subheader("Optimization")
opt_left, opt_right = st.columns([1.25, 0.75])
with opt_left:
    st.plotly_chart(build_tradeoff_chart(artifacts.tradeoff_df, artifacts.optimized_threshold), use_container_width=True)
with opt_right:
    metric_card("Optimized threshold", f"{artifacts.optimized_threshold:.2f}", "Threshold chosen to minimize fairness gap with bounded accuracy loss.")
    metric_card(
        "Bias reduction",
        f"{max(0.0, artifacts.baseline_gap - artifacts.optimized_gap):.3f}",
        "Average fairness-gap improvement after optimization.",
    )
    st.markdown(
        """
        **How optimization works**

        MeritAI scans candidate thresholds and selects the point that reduces average fairness gaps across all detected sensitive dimensions while preserving predictive accuracy as much as possible.
        """
    )

st.subheader("Decision Simulation")
sim_left, sim_right = st.columns([0.9, 1.1])
with sim_left:
    select_count = st.slider(
        "Candidates to select",
        min_value=5,
        max_value=min(100, len(artifacts.results_df)),
        value=min(25, len(artifacts.results_df)),
        step=1,
    )
    audit_dimension = st.selectbox(
        "Audit simulation by sensitive dimension",
        options=list(artifacts.sensitive_map.keys()) if artifacts.sensitive_map else ["No sensitive dimensions detected"],
        disabled=not artifacts.sensitive_map,
    )

results_for_sim = artifacts.results_df.copy()
results_for_sim["Selection_Score"] = (
    0.75 * (results_for_sim["Merit_Score"] / 100)
    + 0.25 * results_for_sim["Prediction_Probability"]
)
results_for_sim = results_for_sim.sort_values(["Selection_Score", "Merit_Score"], ascending=False).reset_index(drop=True)
results_for_sim["Selected"] = 0
results_for_sim.loc[: select_count - 1, "Selected"] = 1

with sim_right:
    st.markdown("### Simulated Candidate Selection")
    display_cols = [col for col in ["Candidate_ID", "Merit_Score", "Prediction_Probability", "Selected"] if col in results_for_sim.columns]
    if not display_cols:
        display_cols = ["Merit_Score", "Prediction_Probability", "Selected"]
    st.dataframe(results_for_sim[display_cols].head(select_count), use_container_width=True)

if artifacts.sensitive_map:
    selected_dimension_col = artifacts.sensitive_map[audit_dimension]
    selection_fig = build_selection_chart(results_for_sim, selected_dimension_col)
    sim_metric_df = (
        results_for_sim.groupby(selected_dimension_col, dropna=False)["Selected"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "Selection Rate"})
    )
    sim_metric_df["Selection Rate"] = sim_metric_df["Selection Rate"].round(4)

    sim_chart_col, sim_table_col = st.columns([1.1, 0.9])
    with sim_chart_col:
        st.plotly_chart(selection_fig, use_container_width=True)
    with sim_table_col:
        st.markdown("### Selection Fairness Snapshot")
        st.dataframe(sim_metric_df, use_container_width=True)

report_buffer = io.BytesIO()
artifacts.fairness_report.to_csv(report_buffer, index=False)

st.subheader("Output")
out_left, out_right = st.columns([1, 1])
with out_left:
    st.markdown("### Decision Intelligence Summary")
    st.write(f"Model decisions are based on **{len(artifacts.training_features)} merit-oriented, non-sensitive features**.")
    st.write(f"Accuracy at baseline threshold `0.50`: **{artifacts.baseline_accuracy:.3f}**")
    st.write(f"Accuracy at optimized threshold `{artifacts.optimized_threshold:.2f}`: **{artifacts.optimized_accuracy:.3f}**")
    st.write("Fairness metrics include Demographic Parity, Equal Opportunity, and high-merit gap analysis across every detected sensitive dimension.")
with out_right:
    st.download_button(
        label="Download fairness report (CSV)",
        data=report_buffer.getvalue(),
        file_name="meritai_fairness_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "MeritAI supports fair hiring practices, ethical AI deployment, and scalable workforce analytics by separating decision intelligence from protected-attribute auditing."
)
