# =============================================================================
# TELECOM CUSTOMER CHURN ANALYSIS
# -----------------------------------------------------------------------------
# Author  : Senior Data Analyst
# Project : Portfolio – Telecom Customer Churn Prediction
# Dataset : Telco Customer Churn (IBM Sample Dataset)
# =============================================================================

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – saves figures to disk
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
DATA_PATH    = os.path.join("data", "Telco_Customer_Churn.csv")
OUTPUT_DIR   = "outputs"
CLEAN_CSV    = os.path.join(OUTPUT_DIR, "cleaned_churn_data.csv")
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# consistent plot style
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
PALETTE = {"No": "#2ecc71", "Yes": "#e74c3c"}


# =============================================================================
# STEP 1 – LOAD DATA
# =============================================================================
print("=" * 65)
print("STEP 1 | Loading dataset")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nFirst five rows:\n{df.head()}")


# =============================================================================
# STEP 2 – DATA EXPLORATION
# =============================================================================
print("\n" + "=" * 65)
print("STEP 2 | Data exploration")
print("=" * 65)

print("\n--- df.info() ---")
df.info()

print("\n--- Descriptive statistics (numeric) ---")
print(df.describe())

print("\n--- Descriptive statistics (object columns) ---")
print(df.describe(include="object"))


# =============================================================================
# STEP 3 – MISSING VALUE AUDIT
# =============================================================================
print("\n" + "=" * 65)
print("STEP 3 | Missing value audit")
print("=" * 65)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"Missing": missing, "Pct (%)": missing_pct})
missing_report = missing_report[missing_report["Missing"] > 0]

if missing_report.empty:
    print("  No NaN values detected via isnull().")
else:
    print(missing_report)


# =============================================================================
# STEP 4 – CONVERT TotalCharges TO NUMERIC
# =============================================================================
print("\n" + "=" * 65)
print("STEP 4 | TotalCharges → numeric")
print("=" * 65)

# TotalCharges is read as object; blank strings must become NaN first
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

newly_missing = df["TotalCharges"].isnull().sum()
print(f"  Rows with NaN TotalCharges after coercion : {newly_missing}")

if newly_missing > 0:
    # These customers have just joined (tenure == 0); fill with MonthlyCharges
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)
    print(f"  Filled {newly_missing} NaN(s) with the corresponding MonthlyCharges value.")

print(f"  TotalCharges dtype now : {df['TotalCharges'].dtype}")


# =============================================================================
# STEP 5 – REMOVE DUPLICATE ROWS
# =============================================================================
print("\n" + "=" * 65)
print("STEP 5 | Duplicate removal")
print("=" * 65)

dupes = df.duplicated().sum()
print(f"  Duplicate rows found : {dupes}")
if dupes:
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Removed {dupes} duplicate rows.  New shape : {df.shape}")
else:
    print("  No duplicates – nothing removed.")


# =============================================================================
# STEP 6 – STANDARDISE COLUMN NAMES
# =============================================================================
print("\n" + "=" * 65)
print("STEP 6 | Standardising column names")
print("=" * 65)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)
print("  Columns after standardisation:")
print("  ", list(df.columns))


# =============================================================================
# STEP 7 – EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 65)
print("STEP 7 | Exploratory Data Analysis")
print("=" * 65)

# --- 7a. Churn rate overview ---
churn_counts = df["churn"].value_counts()
churn_rate   = churn_counts["Yes"] / len(df) * 100
print(f"\n  Overall churn rate : {churn_rate:.2f}%")
print(f"  Churned customers  : {churn_counts['Yes']:,}")
print(f"  Retained customers : {churn_counts['No']:,}")

# --- 7b. Churn by key categorical features ---
cat_features = ["contract", "internetservice", "paymentmethod",
                "partner", "dependents", "seniorcitizen"]

print("\n  Churn rate by selected features:")
for feat in cat_features:
    rates = (
        df.groupby(feat)["churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .round(2)
    )
    print(f"\n  [{feat}]\n{rates.to_string()}")

# --- 7c. Numeric summary by churn label ---
num_cols = ["tenure", "monthlycharges", "totalcharges"]
print("\n  Numeric feature means by churn label:")
print(df.groupby("churn")[num_cols].mean().round(2))


# =============================================================================
# STEP 8 – VISUALISATIONS
# =============================================================================
print("\n" + "=" * 65)
print("STEP 8 | Generating visualisations")
print("=" * 65)

# ── 8-1.  Churn distribution (pie + bar) ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Churn Distribution", fontsize=14, fontweight="bold")

# bar chart
churn_counts.plot(
    kind="bar", ax=axes[0],
    color=[PALETTE["No"], PALETTE["Yes"]], edgecolor="white", width=0.5
)
axes[0].set_title("Count of Customers")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Number of Customers")
axes[0].set_xticklabels(["No", "Yes"], rotation=0)
for bar in axes[0].patches:
    axes[0].annotate(
        f"{int(bar.get_height()):,}",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center", va="bottom", fontsize=11
    )

# pie chart
axes[1].pie(
    churn_counts,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    colors=[PALETTE["No"], PALETTE["Yes"]],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
axes[1].set_title("Churn Proportion")

plt.tight_layout()
path_1 = os.path.join(OUTPUT_DIR, "01_churn_distribution.png")
plt.savefig(path_1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_1}")

# ── 8-2.  Churn by Contract Type ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ct_data = (
    df.groupby("contract")["churn"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("pct")
    .reset_index()
)
sns.barplot(data=ct_data, x="contract", y="pct", hue="churn",
            palette=PALETTE, edgecolor="white", ax=ax)
ax.set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Percentage of Customers (%)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(title="Churn")
plt.tight_layout()
path_2 = os.path.join(OUTPUT_DIR, "02_churn_by_contract.png")
plt.savefig(path_2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_2}")

# ── 8-3.  Churn by Internet Service ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
is_data = (
    df.groupby("internetservice")["churn"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("pct")
    .reset_index()
)
sns.barplot(data=is_data, x="internetservice", y="pct", hue="churn",
            palette=PALETTE, edgecolor="white", ax=ax)
ax.set_title("Churn Rate by Internet Service", fontsize=13, fontweight="bold")
ax.set_xlabel("Internet Service Type")
ax.set_ylabel("Percentage of Customers (%)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(title="Churn")
plt.tight_layout()
path_3 = os.path.join(OUTPUT_DIR, "03_churn_by_internet_service.png")
plt.savefig(path_3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_3}")

# ── 8-4.  Monthly Charges vs Churn (violin + strip) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Monthly Charges vs Churn", fontsize=14, fontweight="bold")

sns.violinplot(data=df, x="churn", y="monthlycharges",
               palette=PALETTE, inner="quartile", ax=axes[0])
axes[0].set_title("Distribution (Violin)")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Monthly Charges ($)")

sns.boxplot(data=df, x="churn", y="monthlycharges",
            palette=PALETTE, width=0.4, ax=axes[1])
axes[1].set_title("Distribution (Box Plot)")
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Monthly Charges ($)")

plt.tight_layout()
path_4 = os.path.join(OUTPUT_DIR, "04_churn_vs_monthly_charges.png")
plt.savefig(path_4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_4}")

# ── 8-5.  Tenure distribution by churn ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for label, colour in PALETTE.items():
    subset = df[df["churn"] == label]
    ax.hist(subset["tenure"], bins=30, alpha=0.65,
            color=colour, label=f"Churn = {label}", edgecolor="white")
ax.set_title("Tenure Distribution by Churn", fontsize=13, fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Number of Customers")
ax.legend()
plt.tight_layout()
path_5 = os.path.join(OUTPUT_DIR, "05_tenure_distribution.png")
plt.savefig(path_5, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_5}")

# ── 8-6.  Correlation heatmap (numeric features) ─────────────────────────────
df_for_corr          = df.copy()
df_for_corr["churn_bin"] = (df_for_corr["churn"] == "Yes").astype(int)
corr_cols            = ["tenure", "monthlycharges", "totalcharges", "churn_bin"]
corr_matrix          = df_for_corr[corr_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True, ax=ax)
ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=13, fontweight="bold")
plt.tight_layout()
path_6 = os.path.join(OUTPUT_DIR, "06_correlation_heatmap.png")
plt.savefig(path_6, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_6}")


# =============================================================================
# STEP 9 – KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 65)
print("STEP 9 | Key Business Insights")
print("=" * 65)

insights = """
  1. HIGH CHURN RATE (~26 %)
     Nearly 1 in 4 customers churns – a significant revenue risk that demands
     proactive retention initiatives.

  2. CONTRACT TYPE IS THE STRONGEST DRIVER
     Month-to-month customers churn at ~42 %, versus ~11 % (one-year) and
     ~3 % (two-year).  Incentivising longer contracts (discounts, loyalty
     perks) can substantially reduce churn.

  3. FIBRE-OPTIC CUSTOMERS CHURN MORE
     Customers on Fibre Optic internet churn at nearly double the rate of
     DSL customers.  Price sensitivity and service-quality perception are
     likely contributing factors.

  4. SHORT-TENURE CUSTOMERS ARE HIGH RISK
     Churn is heavily concentrated in the first 12 months.  Onboarding
     experience and early engagement programmes are critical retention levers.

  5. HIGHER MONTHLY CHARGES CORRELATE WITH CHURN
     Churned customers pay ~$15 more per month on average.  Competitive
     pricing, bundled discounts, and tailored plans can reduce price-driven
     churn.

  6. SINGLE / NO-DEPENDENTS CUSTOMERS CHURN MORE
     Customers without a partner or dependents have fewer switching barriers,
     making them more price-sensitive and easier to win back with targeted
     offers.

  7. ELECTRONIC CHECK PAYERS CHURN MORE
     Customers paying by electronic check have the highest churn rate.
     Migrating customers to auto-pay (credit card / bank transfer) improves
     retention.
"""
print(insights)


# =============================================================================
# STEP 10 – FEATURE ENGINEERING & ENCODING
# =============================================================================
print("=" * 65)
print("STEP 10 | Feature engineering & encoding")
print("=" * 65)

df_ml = df.drop(columns=["customerid"]).copy()   # ID column carries no signal

# ── 10a. Derived features ─────────────────────────────────────────────────────
df_ml["charges_per_month"] = (
    df_ml["totalcharges"] / df_ml["tenure"].replace(0, 1)
)  # avoid /0 for brand-new customers

df_ml["is_long_tenure"] = (df_ml["tenure"] >= 24).astype(int)

# ── 10b. Binary columns with Yes / No → 1 / 0 ───────────────────────────────
binary_cols = [
    "partner", "dependents", "phoneservice", "paperlessbilling",
    "onlinesecurity", "onlinebackup", "deviceprotection",
    "techsupport", "streamingtv", "streamingmovies",
]
for col in binary_cols:
    df_ml[col] = df_ml[col].replace({"Yes": 1, "No": 0,
                                     "No internet service": 0,
                                     "No phone service": 0})

# ── 10c. Label-encode remaining object columns ───────────────────────────────
le = LabelEncoder()
label_enc_cols = ["gender", "multiplelines", "internetservice",
                  "contract", "paymentmethod"]
for col in label_enc_cols:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# ── 10d. Target variable ──────────────────────────────────────────────────────
df_ml["churn"] = (df_ml["churn"] == "Yes").astype(int)

# Final validation before modeling: unexpected category values can still leave
# gaps, so impute defensively if any feature contains missing values.
feature_cols = [col for col in df_ml.columns if col != "churn"]
missing_after_encoding = df_ml[feature_cols].isnull().sum()
missing_after_encoding = missing_after_encoding[missing_after_encoding > 0]

if not missing_after_encoding.empty:
    print("\n  Missing values detected after encoding:")
    print(missing_after_encoding.to_string())

    for col in missing_after_encoding.index:
        if pd.api.types.is_numeric_dtype(df_ml[col]):
            df_ml[col] = df_ml[col].fillna(df_ml[col].median())
        else:
            df_ml[col] = df_ml[col].fillna(df_ml[col].mode().iloc[0])

    print("  Filled post-encoding missing values using median/mode imputation.")

print(f"  Final feature matrix shape : {df_ml.shape}")
print(f"  Target distribution (churn) :\n{df_ml['churn'].value_counts().to_string()}")


# =============================================================================
# STEP 11 – TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "=" * 65)
print("STEP 11 | Train / Test split (80 / 20, stratified)")
print("=" * 65)

X = df_ml.drop(columns=["churn"])
y = df_ml["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"  Training set   : {X_train.shape[0]:,} samples")
print(f"  Test set       : {X_test.shape[0]:,} samples")
print(f"  Features used  : {X.shape[1]}")

# Scale features (critical for Logistic Regression)
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# =============================================================================
# STEP 12 – MODEL TRAINING
# =============================================================================
print("\n" + "=" * 65)
print("STEP 12 | Model training")
print("=" * 65)

# ── Logistic Regression ───────────────────────────────────────────────────────
lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
lr_model.fit(X_train_sc, y_train)
print("  [✓] Logistic Regression trained.")

# ── Random Forest ─────────────────────────────────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=5,
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_model.fit(X_train, y_train)   # tree-based models don't need scaling
print("  [✓] Random Forest Classifier trained.")


# =============================================================================
# STEP 13 – MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 65)
print("STEP 13 | Model evaluation")
print("=" * 65)

def evaluate_model(name, model, X_eval, y_eval, scaled=False):
    """Print accuracy, classification report and save confusion matrix."""
    y_pred = model.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred)
    cm  = confusion_matrix(y_eval, y_pred)

    print(f"\n  ─── {name} ─────────────────────────────────────")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_eval, y_pred, target_names=["No Churn", "Churn"]))

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix – {name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    path  = os.path.join(OUTPUT_DIR, f"cm_{fname}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {path}")
    return acc


lr_acc = evaluate_model("Logistic Regression", lr_model, X_test_sc, y_test)
rf_acc = evaluate_model("Random Forest",        rf_model, X_test,    y_test)

print(f"\n  ┌─────────────────────────────────┐")
print(f"  │  Summary                        │")
print(f"  │  Logistic Regression : {lr_acc*100:6.2f}%  │")
print(f"  │  Random Forest       : {rf_acc*100:6.2f}%  │")
print(f"  └─────────────────────────────────┘")


# =============================================================================
# STEP 14 – FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 65)
print("STEP 14 | Feature importance (Random Forest)")
print("=" * 65)

feature_names = X.columns.tolist()
importances   = rf_model.feature_importances_

feat_imp_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\n  Top 10 features by importance:")
print(feat_imp_df.head(10).to_string(index=False))

# Plot top-15 features
fig, ax = plt.subplots(figsize=(10, 7))
top15 = feat_imp_df.head(15)
sns.barplot(data=top15, x="importance", y="feature",
            palette="viridis_r", edgecolor="white", ax=ax)
ax.set_title("Top 15 Features – Random Forest Importance",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
for bar in ax.patches:
    ax.text(
        bar.get_width() + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.4f}",
        va="center", fontsize=8,
    )
plt.tight_layout()
path_fi = os.path.join(OUTPUT_DIR, "07_feature_importance.png")
plt.savefig(path_fi, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Feature importance chart saved → {path_fi}")


# Logistic Regression coefficients (top absolute coefficients)
lr_coef_df = (
    pd.DataFrame({"feature": feature_names,
                  "coefficient": lr_model.coef_[0]})
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=False)
    .reset_index(drop=True)
)
print("\n  Top 10 Logistic Regression coefficients (by magnitude):")
print(lr_coef_df[["feature", "coefficient"]].head(10).to_string(index=False))


# =============================================================================
# STEP 15 – SAVE CLEANED DATASET
# =============================================================================
print("\n" + "=" * 65)
print("STEP 15 | Saving cleaned dataset")
print("=" * 65)

# Save the pre-ML, human-readable cleaned frame (original column values)
df.to_csv(CLEAN_CSV, index=False)
print(f"  Cleaned dataset saved → {CLEAN_CSV}")
print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  PROJECT COMPLETE")
print("=" * 65)
print(f"""
  Dataset       : {DATA_PATH}
  Rows          : {df.shape[0]:,}
  Features used : {X.shape[1]}
  Churn rate    : {churn_rate:.2f}%

  Models trained:
    • Logistic Regression  → Accuracy : {lr_acc*100:.2f}%
    • Random Forest        → Accuracy : {rf_acc*100:.2f}%

  Outputs saved to : {OUTPUT_DIR}/
    • Cleaned CSV
    • 7 visualisation charts
    • 2 confusion matrix charts
""")
