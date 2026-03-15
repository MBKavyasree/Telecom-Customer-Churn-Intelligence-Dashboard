import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
C_BG        = "#0D1B2A"
C_SURFACE   = "#132338"
C_BORDER    = "#1E3A5F"
C_GREEN     = "#22C55E"
C_RED       = "#EF4444"
C_AMBER     = "#F59E0B"
C_BLUE      = "#38BDF8"
C_TEXT      = "#E2EAF4"
C_MUTED     = "#7F9DB8"
C_CHART_BG  = "#0D1B2A"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C_CHART_BG,
    plot_bgcolor=C_CHART_BG,
    font=dict(family="Inter, sans-serif", color=C_TEXT, size=13),
    title_font=dict(size=15, color=C_TEXT),
    margin=dict(l=16, r=16, t=52, b=16),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=C_BORDER,
        borderwidth=1,
    ),
)

AXIS_STYLE = dict(
    gridcolor=C_BORDER,
    zerolinecolor=C_BORDER,
    tickfont=dict(color=C_MUTED),
    title_font=dict(color=C_MUTED),
    linecolor=C_BORDER,
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background-color: {C_BG} !important;
        font-family: 'Inter', sans-serif;
        color: {C_TEXT};
    }}
    [data-testid="stSidebar"] {{
        background-color: #0a1623 !important;
        border-right: 1px solid {C_BORDER};
    }}
    [data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}

    .block-container {{ padding: 1.4rem 2rem 2rem 2rem; }}

    .hero {{
        background: linear-gradient(135deg, #0f2a45 0%, #0a3d3a 100%);
        border: 1px solid {C_BORDER};
        border-radius: 20px;
        padding: 2rem 2.2rem 1.8rem 2.2rem;
        margin-bottom: 1.5rem;
    }}
    .hero h1 {{
        margin: 0 0 0.4rem 0;
        font-size: 1.95rem;
        font-weight: 700;
        color: {C_TEXT};
        letter-spacing: -0.5px;
    }}
    .hero p {{
        margin: 0;
        color: {C_MUTED};
        font-size: 0.97rem;
        max-width: 800px;
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: {C_GREEN};
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 100px;
        padding: 0.18rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    .kpi-card {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 16px;
        padding: 1.15rem 1.3rem;
        height: 100%;
        margin-bottom: 0.5rem;
    }}
    .kpi-label {{
        font-size: 0.75rem;
        font-weight: 600;
        color: {C_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.5rem;
    }}
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        color: {C_TEXT};
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        color: {C_MUTED};
        margin-top: 0.35rem;
    }}
    .kpi-icon {{
        font-size: 1.5rem;
        float: right;
        margin-top: -0.2rem;
    }}
    .kpi-red   .kpi-value {{ color: {C_RED}; }}
    .kpi-green .kpi-value {{ color: {C_GREEN}; }}
    .kpi-blue  .kpi-value {{ color: {C_BLUE}; }}
    .kpi-amber .kpi-value {{ color: {C_AMBER}; }}

    .insight {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-left: 4px solid {C_AMBER};
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        font-size: 0.92rem;
        color: {C_TEXT};
        line-height: 1.55;
    }}
    .insight strong {{ color: {C_AMBER}; }}

    .section-heading {{
        font-size: 1rem;
        font-weight: 600;
        color: {C_TEXT};
        border-bottom: 1px solid {C_BORDER};
        padding-bottom: 0.5rem;
        margin: 1.2rem 0 0.9rem 0;
    }}

    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: {C_MUTED} !important;
        border-radius: 10px 10px 0 0 !important;
        font-weight: 500;
        font-size: 0.92rem;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {C_BLUE} !important;
        border-bottom: 2px solid {C_BLUE} !important;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {C_BORDER} !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── DATA HELPERS ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/Telco_Customer_Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)
    df = df.drop_duplicates().copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    df_ml = df.drop(columns=["customerid"]).copy()
    df_ml["charges_per_month"] = df_ml["totalcharges"] / df_ml["tenure"].replace(0, 1)
    df_ml["is_long_tenure"] = (df_ml["tenure"] >= 24).astype(int)
    binary_cols = [
        "partner", "dependents", "phoneservice", "paperlessbilling",
        "onlinesecurity", "onlinebackup", "deviceprotection",
        "techsupport", "streamingtv", "streamingmovies",
    ]
    for col in binary_cols:
        df_ml[col] = df_ml[col].replace(
            {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
        )
    enc = LabelEncoder()
    for col in ["gender", "multiplelines", "internetservice", "contract", "paymentmethod"]:
        df_ml[col] = enc.fit_transform(df_ml[col].astype(str))
    df_ml["churn"] = (df_ml["churn"] == "Yes").astype(int)
    for col in df_ml.columns:
        if df_ml[col].isnull().any():
            df_ml[col] = (
                df_ml[col].fillna(df_ml[col].median())
                if pd.api.types.is_numeric_dtype(df_ml[col])
                else df_ml[col].fillna(df_ml[col].mode().iloc[0])
            )
    return df_ml


@st.cache_data
def train_models(df: pd.DataFrame) -> dict:
    df_ml = prepare_model_data(df)
    X, y = df_ml.drop(columns=["churn"]), df_ml["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_tr_sc, y_tr)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    feat_imp = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "lr_acc": accuracy_score(y_te, lr.predict(X_te_sc)),
        "rf_acc": accuracy_score(y_te, rf.predict(X_te)),
        "lr_cm":  confusion_matrix(y_te, lr.predict(X_te_sc)),
        "rf_cm":  confusion_matrix(y_te, rf.predict(X_te)),
        "feat_imp": feat_imp,
    }


# ─── SIDEBAR FILTERS ──────────────────────────────────────────────────────────
def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown(
        f"<div style='font-size:1.1rem;font-weight:700;color:{C_TEXT};"
        "padding:0.5rem 0 1rem 0;letter-spacing:-0.3px;'>🎛️ Filters</div>",
        unsafe_allow_html=True,
    )
    contracts = st.sidebar.multiselect(
        "Contract Type", options=sorted(df["contract"].unique()),
        default=sorted(df["contract"].unique()),
    )
    services = st.sidebar.multiselect(
        "Internet Service", options=sorted(df["internetservice"].unique()),
        default=sorted(df["internetservice"].unique()),
    )
    payments = st.sidebar.multiselect(
        "Payment Method", options=sorted(df["paymentmethod"].unique()),
        default=sorted(df["paymentmethod"].unique()),
    )
    churn_filter = st.sidebar.multiselect(
        "Churn Status", options=["No", "Yes"], default=["No", "Yes"],
    )
    tenure_range = st.sidebar.slider(
        "Tenure (months)", int(df["tenure"].min()), int(df["tenure"].max()),
        (int(df["tenure"].min()), int(df["tenure"].max())),
    )
    monthly_range = st.sidebar.slider(
        "Monthly Charges ($)",
        float(df["monthlycharges"].min()), float(df["monthlycharges"].max()),
        (float(df["monthlycharges"].min()), float(df["monthlycharges"].max())),
    )
    mask = (
        df["contract"].isin(contracts)
        & df["internetservice"].isin(services)
        & df["paymentmethod"].isin(payments)
        & df["churn"].isin(churn_filter)
        & df["tenure"].between(*tenure_range)
        & df["monthlycharges"].between(*monthly_range)
    )
    return df[mask].copy()


# ─── REUSABLE COMPONENTS ──────────────────────────────────────────────────────
def kpi(col, label: str, value: str, sub: str, icon: str, accent: str) -> None:
    with col:
        st.markdown(
            f"""<div class="kpi-card kpi-{accent}">
                <span class="kpi-icon">{icon}</span>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    if fig.data and fig.data[0].type not in ("pie", "heatmap"):
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
    return fig


def cm_fig(matrix: np.ndarray, title: str) -> go.Figure:
    labels = ["No Churn", "Churn"]
    fig = go.Figure(
        go.Heatmap(
            z=matrix[::-1],
            x=labels,
            y=labels[::-1],
            colorscale=[[0, C_SURFACE], [0.5, "#0C4A6E"], [1, C_BLUE]],
            showscale=False,
            text=matrix[::-1],
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=20, color=C_TEXT),
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=320, title=title)
    fig.update_xaxes(title="Predicted", **AXIS_STYLE)
    fig.update_yaxes(title="Actual", **AXIS_STYLE)
    return fig


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    df      = load_data()
    fdf     = sidebar_filters(df)
    models  = train_models(df)

    n_total     = len(fdf)
    churn_rate  = (fdf["churn"] == "Yes").mean() * 100 if n_total else 0.0
    n_retained  = int(n_total * (1 - churn_rate / 100))
    avg_charge  = fdf["monthlycharges"].mean() if n_total else 0.0
    avg_tenure  = fdf["tenure"].mean() if n_total else 0.0

    # Hero
    st.markdown(
        """<div class="hero">
            <div class="hero-badge">📡 Portfolio Project</div>
            <h1>Telecom Customer Churn Analytics</h1>
            <p>End-to-end churn intelligence platform — explore customer segments,
            understand churn drivers, and evaluate predictive model performance
            on 7,043 real telecom customers.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Total Customers",   f"{n_total:,}",          "after filters",     "👥", "blue")
    kpi(k2, "Churn Rate",        f"{churn_rate:.1f}%",    "churned customers", "📉", "red")
    kpi(k3, "Retained",          f"{n_retained:,}",       "customers staying", "✅", "green")
    kpi(k4, "Avg Monthly Charge",f"${avg_charge:.0f}",    "per customer",      "💳", "amber")
    kpi(k5, "Avg Tenure",        f"{avg_tenure:.0f} mo",  "months on service", "📆", "blue")

    st.markdown("<br>", unsafe_allow_html=True)

    if fdf.empty:
        st.error("No records match the current filters.")
        st.stop()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊  Overview", "🔍  Churn Drivers", "🤖  ML Models", "🗂️  Data"]
    )

    # ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns((1, 1.6))

        counts = fdf["churn"].value_counts().reset_index()
        counts.columns = ["churn", "count"]
        donut = px.pie(counts, names="churn", values="count", hole=0.62,
                       color="churn",
                       color_discrete_map={"No": C_GREEN, "Yes": C_RED})
        donut.update_traces(textposition="outside", textfont_size=13,
                            marker=dict(line=dict(color=C_BG, width=3)))
        donut = style(donut, 360)
        donut.update_layout(title="Churn Split",
                            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))

        contract_agg = (
            fdf.groupby(["contract", "churn"]).size()
            .reset_index(name="count")
        )
        bar_ct = px.bar(
            contract_agg, x="contract", y="count", color="churn", barmode="group",
            color_discrete_map={"No": C_GREEN, "Yes": C_RED},
            category_orders={"contract": ["Month-to-month", "One year", "Two year"]},
        )
        bar_ct.update_traces(marker_line_width=0)
        bar_ct = style(bar_ct, 360)
        bar_ct.update_layout(title="Customers by Contract & Churn",
                             xaxis_title=None, yaxis_title="Customers")

        with c1:
            st.plotly_chart(donut, use_container_width=True)
        with c2:
            st.plotly_chart(bar_ct, use_container_width=True)

        c3, c4 = st.columns(2)

        box = px.box(fdf, x="churn", y="monthlycharges", color="churn",
                     color_discrete_map={"No": C_GREEN, "Yes": C_RED},
                     points=False,
                     labels={"churn": "Churn", "monthlycharges": "Monthly Charge ($)"})
        box.update_traces(line_width=2)
        box = style(box, 340)
        box.update_layout(title="Monthly Charges by Churn Status")

        violin = px.violin(fdf, x="churn", y="tenure", color="churn", box=True,
                           color_discrete_map={"No": C_GREEN, "Yes": C_RED},
                           labels={"churn": "Churn", "tenure": "Tenure (months)"})
        violin = style(violin, 340)
        violin.update_layout(title="Tenure Distribution by Churn Status")

        with c3:
            st.plotly_chart(box, use_container_width=True)
        with c4:
            st.plotly_chart(violin, use_container_width=True)

    # ── TAB 2: CHURN DRIVERS ──────────────────────────────────────────────────
    with tab2:
        def churn_pct(col: str) -> pd.DataFrame:
            return (
                fdf.groupby(col)["churn"]
                .apply(lambda s: (s == "Yes").mean() * 100)
                .reset_index(name="churn_rate")
                .sort_values("churn_rate")
            )

        scale = [[0, C_GREEN], [0.5, C_AMBER], [1, C_RED]]

        c5, c6 = st.columns(2)

        is_df = churn_pct("internetservice")
        bar_is = px.bar(is_df, x="churn_rate", y="internetservice", orientation="h",
                        color="churn_rate", color_continuous_scale=scale,
                        text=is_df["churn_rate"].map("{:.1f}%".format))
        bar_is.update_traces(textposition="outside", marker_line_width=0)
        bar_is = style(bar_is, 285)
        bar_is.update_layout(title="Churn Rate — Internet Service",
                             coloraxis_showscale=False,
                             xaxis_title="Churn Rate (%)", yaxis_title=None)

        pm_df = churn_pct("paymentmethod")
        bar_pm = px.bar(pm_df, x="churn_rate", y="paymentmethod", orientation="h",
                        color="churn_rate", color_continuous_scale=scale,
                        text=pm_df["churn_rate"].map("{:.1f}%".format))
        bar_pm.update_traces(textposition="outside", marker_line_width=0)
        bar_pm = style(bar_pm, 285)
        bar_pm.update_layout(title="Churn Rate — Payment Method",
                             coloraxis_showscale=False,
                             xaxis_title="Churn Rate (%)", yaxis_title=None)

        with c5:
            st.plotly_chart(bar_is, use_container_width=True)
        with c6:
            st.plotly_chart(bar_pm, use_container_width=True)

        c7, c8 = st.columns(2)

        ct_df = churn_pct("contract").sort_values("churn_rate", ascending=False)
        bar_c = px.bar(ct_df, x="contract", y="churn_rate",
                       color="churn_rate", color_continuous_scale=scale,
                       text=ct_df["churn_rate"].map("{:.1f}%".format))
        bar_c.update_traces(textposition="outside", marker_line_width=0)
        bar_c = style(bar_c, 320)
        bar_c.update_layout(title="Churn Rate by Contract Type",
                            coloraxis_showscale=False,
                            xaxis_title=None, yaxis_title="Churn Rate (%)")

        hist = px.histogram(fdf, x="tenure", color="churn", nbins=24,
                            opacity=0.85, barmode="overlay",
                            color_discrete_map={"No": C_GREEN, "Yes": C_RED})
        hist.update_traces(marker_line_width=0)
        hist = style(hist, 320)
        hist.update_layout(title="Tenure Histogram by Churn",
                           xaxis_title="Tenure (months)", yaxis_title="Customers",
                           bargap=0.05)

        with c7:
            st.plotly_chart(bar_c, use_container_width=True)
        with c8:
            st.plotly_chart(hist, use_container_width=True)

        st.markdown('<div class="section-heading">💡 Data-Driven Insights</div>',
                    unsafe_allow_html=True)

        top_ct = ct_df.sort_values("churn_rate", ascending=False).iloc[0]
        top_pm = pm_df.sort_values("churn_rate", ascending=False).iloc[0]
        top_is = is_df.sort_values("churn_rate", ascending=False).iloc[0]
        cc = fdf.loc[fdf["churn"] == "Yes", "monthlycharges"].mean()
        rc = fdf.loc[fdf["churn"] == "No",  "monthlycharges"].mean()
        ct = fdf.loc[fdf["churn"] == "Yes", "tenure"].mean()
        rt = fdf.loc[fdf["churn"] == "No",  "tenure"].mean()

        for ins in [
            f"<strong>{top_ct['contract']}</strong> contracts have the highest churn rate at "
            f"<strong>{top_ct['churn_rate']:.1f}%</strong>.",
            f"<strong>{top_pm['paymentmethod']}</strong> is the highest-risk payment method "
            f"at <strong>{top_pm['churn_rate']:.1f}%</strong> — migrating to auto-pay can reduce this.",
            f"<strong>{top_is['internetservice']}</strong> internet users churn at "
            f"<strong>{top_is['churn_rate']:.1f}%</strong> — competitive pricing or quality improvements may help.",
            f"Churned customers pay <strong>${cc:.2f}/mo</strong> vs "
            f"<strong>${rc:.2f}/mo</strong> for retained customers.",
            f"Churned customers average only <strong>{ct:.1f} months</strong> of tenure vs "
            f"<strong>{rt:.1f} months</strong> — early engagement is critical.",
        ]:
            st.markdown(f'<div class="insight">{ins}</div>', unsafe_allow_html=True)

    # ── TAB 3: ML MODELS ──────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-heading">Model Performance Summary</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        best = "Random Forest" if models["rf_acc"] >= models["lr_acc"] else "Logistic Regression"
        kpi(m1, "Logistic Regression", f"{models['lr_acc']*100:.2f}%", "test accuracy", "📈", "blue")
        kpi(m2, "Random Forest",        f"{models['rf_acc']*100:.2f}%", "test accuracy", "🌲", "green")
        kpi(m3, "Best Model",           best.split()[0],                "higher accuracy","🏆", "amber")
        kpi(m4, "Test Set",             "1,409",                        "customers",      "🧪", "blue")

        st.markdown("<br>", unsafe_allow_html=True)

        c9, c10 = st.columns(2)
        with c9:
            st.plotly_chart(cm_fig(models["lr_cm"],
                "Logistic Regression — Confusion Matrix"), use_container_width=True)
        with c10:
            st.plotly_chart(cm_fig(models["rf_cm"],
                "Random Forest — Confusion Matrix"), use_container_width=True)

        top15 = models["feat_imp"].head(15).sort_values("importance")
        feat_fig = go.Figure(
            go.Bar(
                x=top15["importance"],
                y=top15["feature"],
                orientation="h",
                marker=dict(
                    color=top15["importance"],
                    colorscale=[[0, C_AMBER], [1, C_BLUE]],
                    line=dict(width=0),
                ),
                text=(top15["importance"] * 100).map("{:.2f}%".format),
                textposition="outside",
                textfont=dict(color=C_MUTED, size=11),
            )
        )
        feat_fig = style(feat_fig, 480)
        feat_fig.update_layout(title="Top 15 Feature Importances — Random Forest",
                               xaxis_title="Importance Score", yaxis_title=None)
        st.plotly_chart(feat_fig, use_container_width=True)

    # ── TAB 4: DATA ───────────────────────────────────────────────────────────
    with tab4:
        preview_cols = [
            "customerid", "gender", "seniorcitizen", "tenure",
            "contract", "internetservice", "paymentmethod",
            "monthlycharges", "totalcharges", "churn",
        ]
        st.markdown(f'<div class="section-heading">Filtered Records ({n_total:,} rows)</div>',
                    unsafe_allow_html=True)
        st.dataframe(fdf[preview_cols], use_container_width=True, height=400)

        st.markdown('<div class="section-heading">Churn Rate by Contract × Internet Service</div>',
                    unsafe_allow_html=True)
        pivot = (
            fdf.groupby(["contract", "internetservice"])["churn"]
            .apply(lambda s: f"{(s=='Yes').mean()*100:.1f}%")
            .unstack(fill_value="—")
        )
        st.dataframe(pivot, use_container_width=True)

        st.markdown('<div class="section-heading">Average Spend & Tenure by Churn</div>',
                    unsafe_allow_html=True)
        agg = (
            fdf.groupby("churn")[["monthlycharges", "totalcharges", "tenure"]]
            .mean().round(2).reset_index()
        )
        st.dataframe(agg, use_container_width=True, height=120)


if __name__ == "__main__":
    main()
