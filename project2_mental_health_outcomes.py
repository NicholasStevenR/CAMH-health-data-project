"""
Digital Mental Health Service Outcomes Dashboard
Author: Nicholas Steven
Target Role: Research Analyst, Krembil Centre for Neuroinformatics — CAMH
Repo: github.com/nicholasstevenr/CAMH-health-data-project

Integrates CAMH encounter data with OHIP MH claims to compute:
symptom trajectories (PANSS/PHQ-9/GAD-7), 90-day readmission rates,
logistic regression predictors, and geographic utilization by FSA.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ── OHIP Mental Health Billing Codes ─────────────────────────────────────────

OHIP_MH_CODES = {
    "290": "Psychiatric assessment",
    "291": "Repeat psychiatric assessment",
    "292": "Psychiatric consultation",
    "K030": "Individual psychotherapy 20-45 min",
    "K031": "Individual psychotherapy 45-75 min",
    "K032": "Individual psychotherapy 75+ min",
    "K033": "Group psychotherapy",
    "K034": "Family psychotherapy",
}

CAMH_PROGRAM_TYPES = [
    "structured_day_program",
    "outpatient_individual",
    "group_therapy",
    "assertive_community_treatment",
    "crisis_stabilization",
]

SYMPTOM_SCALES = {
    "PANSS": {"min": 30, "max": 210, "direction": "lower_better"},
    "PHQ9":  {"min": 0,  "max": 27,  "direction": "lower_better"},
    "GAD7":  {"min": 0,  "max": 21,  "direction": "lower_better"},
}


# ── Load ──────────────────────────────────────────────────────────────────────

def load(encounter_path: str, ohip_path: str, population_path: str):
    enc  = pd.read_csv(encounter_path, parse_dates=["admission_date", "discharge_date",
                                                     "program_start_date", "program_end_date"])
    ohip = pd.read_csv(ohip_path, parse_dates=["service_date"])
    pop  = pd.read_csv(population_path)   # fsa, fiscal_year, population
    print(f"Encounters: {len(enc):,}  |  OHIP MH claims: {len(ohip):,}")
    return enc, ohip, pop


# ── 1. Symptom Trajectory Analysis ───────────────────────────────────────────

def symptom_trajectory(enc: pd.DataFrame) -> pd.DataFrame:
    """
    Paired pre/post analysis for each program type × symptom scale.
    Requires columns: program_type, {scale}_pre, {scale}_post
    Returns: delta, t-stat, p-value, Cohen's d, effect size category.
    """
    results = []
    for program in CAMH_PROGRAM_TYPES:
        prog_df = enc[enc["program_type"] == program].copy()
        for scale in SYMPTOM_SCALES:
            pre_col  = f"{scale.lower()}_pre"
            post_col = f"{scale.lower()}_post"
            if pre_col not in prog_df.columns or post_col not in prog_df.columns:
                continue
            complete = prog_df[[pre_col, post_col]].dropna()
            if len(complete) < 10:
                continue
            pre  = complete[pre_col]
            post = complete[post_col]
            delta = post.mean() - pre.mean()
            t, p  = stats.ttest_rel(pre, post)
            d     = delta / np.std(post - pre, ddof=1)   # Cohen's d (paired)
            effect_cat = ("negligible" if abs(d) < 0.2 else
                          "small"      if abs(d) < 0.5 else
                          "medium"     if abs(d) < 0.8 else "large")
            results.append({
                "program_type":     program,
                "scale":            scale,
                "n":                len(complete),
                "pre_mean":         round(pre.mean(), 2),
                "post_mean":        round(post.mean(), 2),
                "delta_mean":       round(delta, 2),
                "t_stat":           round(t, 3),
                "p_value":          round(p, 4),
                "significant":      p < 0.05,
                "cohens_d":         round(d, 3),
                "effect_size":      effect_cat,
            })
    return pd.DataFrame(results)


# ── 2. 90-Day Readmission Analysis ────────────────────────────────────────────

def readmission_analysis(enc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flags 90-day returns to CAMH inpatient or ED after index discharge.
    Returns: subject-level readmission flags + summary by diagnosis/program.
    """
    enc = enc.sort_values(["pseudo_hcn", "admission_date"])
    enc["prev_discharge"] = enc.groupby("pseudo_hcn")["discharge_date"].shift(1)
    enc["days_since_discharge"] = (
        enc["admission_date"] - enc["prev_discharge"]
    ).dt.days
    enc["is_readmission_90d"] = enc["days_since_discharge"].between(1, 90)

    # Summary by diagnosis
    diag_summary = (
        enc.groupby(["primary_diagnosis_category", "fiscal_year"])
        .agg(
            n_index_discharges = ("is_readmission_90d", "count"),
            n_readmissions_90d = ("is_readmission_90d", "sum"),
        )
        .reset_index()
    )
    diag_summary["readmission_rate_pct"] = (
        diag_summary["n_readmissions_90d"] / diag_summary["n_index_discharges"] * 100
    ).round(1)

    return enc, diag_summary


# ── 3. Readmission Predictors — Logistic Regression ──────────────────────────

def readmission_predictor_model(enc: pd.DataFrame) -> dict:
    """
    Logistic regression predicting 90-day readmission from modifiable factors.
    Features: prior_admission_count, program_completed, discharge_phq9,
              unstable_housing_at_discharge, age_at_discharge, primary_dx_cat (encoded)
    """
    feature_cols = [
        "prior_admission_count", "program_completed",
        "discharge_phq9", "unstable_housing_at_discharge", "age_at_discharge",
    ]
    target = "is_readmission_90d"

    # Encode diagnosis category
    if "primary_diagnosis_category" in enc.columns:
        enc["dx_encoded"] = pd.Categorical(enc["primary_diagnosis_category"]).codes
        feature_cols.append("dx_encoded")

    model_df = enc[feature_cols + [target]].dropna()
    if len(model_df) < 50:
        return {"error": "Insufficient data for modelling"}

    X = model_df[feature_cols].values
    y = model_df[target].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    cv_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc")
    lr.fit(X_scaled, y)

    coef_df = pd.DataFrame({
        "feature":    feature_cols,
        "coefficient": lr.coef_[0].round(3),
        "odds_ratio": np.exp(lr.coef_[0]).round(3),
    }).sort_values("odds_ratio", ascending=False)

    print(f"\n── Readmission Predictor Model ──")
    print(f"  5-fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(coef_df.to_string(index=False))

    return {
        "cv_auc_mean":  round(cv_scores.mean(), 3),
        "cv_auc_std":   round(cv_scores.std(), 3),
        "coefficients": coef_df,
    }


# ── 4. OHIP MH Utilization by FSA ────────────────────────────────────────────

def ohip_utilization_by_fsa(ohip: pd.DataFrame, pop: pd.DataFrame,
                              camh_enc: pd.DataFrame) -> pd.DataFrame:
    """
    OHIP MH billing rate per 1,000 population by FSA.
    Stratified by care type (primary care vs. specialist) and
    overlaid with CAMH direct service volume.
    """
    # Filter to MH codes
    ohip["is_mh"] = ohip["fee_code"].astype(str).isin(OHIP_MH_CODES.keys())
    mh_claims = ohip[ohip["is_mh"]].copy()

    # Extract FSA from postal code (first 3 chars)
    if "patient_postal_code" in mh_claims.columns:
        mh_claims["fsa"] = mh_claims["patient_postal_code"].str[:3].str.upper()

    # Aggregate OHIP by FSA + fiscal year
    ohip_by_fsa = (
        mh_claims.groupby(["fsa", "fiscal_year"])
        .agg(
            total_mh_claims        = ("is_mh", "count"),
            primary_care_claims    = ("provider_type_primary_care", "sum") if "provider_type_primary_care" in mh_claims.columns else ("is_mh", "count"),
        )
        .reset_index()
        .merge(pop, on=["fsa", "fiscal_year"], how="left")
    )
    ohip_by_fsa["mh_rate_per_1000"] = (
        ohip_by_fsa["total_mh_claims"] / ohip_by_fsa["population"] * 1_000
    ).round(2)

    # Overlay CAMH direct service volume by FSA
    if "patient_fsa" in camh_enc.columns:
        camh_vol = (
            camh_enc.groupby(["patient_fsa", "fiscal_year"])
            .size().reset_index(name="camh_encounters")
            .rename(columns={"patient_fsa": "fsa"})
        )
        ohip_by_fsa = ohip_by_fsa.merge(camh_vol, on=["fsa", "fiscal_year"], how="left")
        ohip_by_fsa["camh_encounters"] = ohip_by_fsa["camh_encounters"].fillna(0)

    return ohip_by_fsa


# ── Export ────────────────────────────────────────────────────────────────────

def export_all(results: dict, outdir: str = "output") -> None:
    import os; os.makedirs(outdir, exist_ok=True)
    for name, obj in results.items():
        if isinstance(obj, pd.DataFrame):
            path = f"{outdir}/{name}.csv"
            obj.to_csv(path, index=False)
            print(f"  Exported {len(obj)} rows → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    enc, ohip, pop = load(
        "data/camh_encounters_synthetic.csv",
        "data/ohip_mh_claims_synthetic.csv",
        "data/catchment_fsa_population.csv",
    )

    symptom_traj            = symptom_trajectory(enc)
    enc_with_flags, readmit = readmission_analysis(enc)
    predictor_results       = readmission_predictor_model(enc_with_flags)
    ohip_fsa                = ohip_utilization_by_fsa(ohip, pop, enc)

    print("\n── Symptom Trajectory (significant programs) ──")
    sig = symptom_traj[symptom_traj["significant"]]
    print(sig[["program_type", "scale", "n", "delta_mean", "cohens_d", "effect_size"]].to_string(index=False))

    print("\n── Readmission by Diagnosis ──")
    print(readmit.sort_values("readmission_rate_pct", ascending=False).to_string(index=False))

    export_all({
        "symptom_trajectories":     symptom_traj,
        "readmission_by_diagnosis": readmit,
        "ohip_utilization_by_fsa":  ohip_fsa,
    })

    if isinstance(predictor_results.get("coefficients"), pd.DataFrame):
        predictor_results["coefficients"].to_csv("output/readmission_model_coefficients.csv", index=False)
        print("  Exported model coefficients → output/readmission_model_coefficients.csv")
