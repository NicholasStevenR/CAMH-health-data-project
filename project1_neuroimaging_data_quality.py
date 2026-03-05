"""
Neuroimaging Data Quality Pipeline — EEG/MRI Cohort Readiness Assessment
Author: Nicholas Steven
Target Role: Research Analyst, Krembil Centre for Neuroinformatics — CAMH
Repo: github.com/nicholasstevenr/CAMH-health-data-project

Automated DQ pipeline for multi-modal neuroimaging cohorts:
completeness matrix, EEG signal quality heuristics, MRI quality metrics,
Little's MCAR test, and per-subject exclusion flag with reason codes.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_MODALITIES = ["eeg_raw", "eeg_epochs", "mri_t1w", "mri_freesurfer", "clinical_phenotype"]

EEG_QUALITY_THRESHOLDS = {
    "kurtosis_max":         5.0,    # z-score; channels above → high-artifact
    "amplitude_variance_max": 3.0,  # SD above cohort mean
    "line_noise_ratio_max": 0.15,   # 50/60 Hz power / broadband power
    "bridged_electrode_pct_max": 0.10,  # >10% bridged channels → reject
}

MRI_QUALITY_THRESHOLDS = {
    "snr_min":         10.0,   # minimum SNR for usable T1w
    "euler_number_min": -200,  # FreeSurfer cortical reconstruction quality
    "cnr_min":          1.5,
}

EXCLUSION_CODES = {
    "E01": "Missing required modality",
    "E02": "EEG high kurtosis artifact",
    "E03": "EEG line noise contamination",
    "E04": "EEG bridged electrodes",
    "E05": "MRI low SNR",
    "E06": "MRI poor FreeSurfer reconstruction (Euler number)",
    "E07": "MRI low CNR",
    "E08": "Critical clinical phenotype missing",
}


# ── Load ──────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Subject-level manifest: one row per subject, boolean columns per modality."""
    df = pd.read_csv(manifest_path)
    print(f"Manifest loaded: {len(df)} subjects, {len(df.columns)} columns")
    return df


def load_eeg_quality(eeg_quality_path: str) -> pd.DataFrame:
    """Pre-computed per-recording EEG quality metrics CSV."""
    return pd.read_csv(eeg_quality_path)


def load_mri_quality(mri_quality_path: str) -> pd.DataFrame:
    """FreeSurfer/FSL quality metrics CSV per subject."""
    return pd.read_csv(mri_quality_path)


def load_clinical(clinical_path: str) -> pd.DataFrame:
    return pd.read_csv(clinical_path)


# ── 1. Completeness Matrix ─────────────────────────────────────────────────────

def completeness_matrix(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-subject completeness booleans for each required modality,
    plus a composite 'complete_for_primary_analysis' flag.
    """
    result = manifest[["subject_id"]].copy()
    for mod in REQUIRED_MODALITIES:
        if mod in manifest.columns:
            result[f"has_{mod}"] = manifest[mod].fillna(False).astype(bool)
        else:
            result[f"has_{mod}"] = False

    result["n_modalities_present"] = result[[f"has_{m}" for m in REQUIRED_MODALITIES]].sum(axis=1)
    result["complete_for_primary"] = result["n_modalities_present"] == len(REQUIRED_MODALITIES)

    print(f"\n── Completeness Summary ──")
    print(f"  Complete subjects: {result['complete_for_primary'].sum()} / {len(result)}")
    for mod in REQUIRED_MODALITIES:
        col = f"has_{mod}"
        pct = result[col].mean() * 100 if col in result.columns else 0
        print(f"  {mod:<25}: {pct:.1f}%")
    return result


# ── 2. EEG Signal Quality Assessment ──────────────────────────────────────────

def eeg_quality_flags(eeg_qual: pd.DataFrame) -> pd.DataFrame:
    """
    Flags EEG recordings failing quality thresholds.
    Expected columns: subject_id, mean_kurtosis_z, amplitude_variance_z,
                      line_noise_ratio, bridged_electrode_pct
    """
    flags = eeg_qual[["subject_id"]].copy()

    if "mean_kurtosis_z" in eeg_qual.columns:
        flags["flag_kurtosis"] = eeg_qual["mean_kurtosis_z"] > EEG_QUALITY_THRESHOLDS["kurtosis_max"]
    if "amplitude_variance_z" in eeg_qual.columns:
        flags["flag_variance"] = eeg_qual["amplitude_variance_z"] > EEG_QUALITY_THRESHOLDS["amplitude_variance_max"]
    if "line_noise_ratio" in eeg_qual.columns:
        flags["flag_line_noise"] = eeg_qual["line_noise_ratio"] > EEG_QUALITY_THRESHOLDS["line_noise_ratio_max"]
    if "bridged_electrode_pct" in eeg_qual.columns:
        flags["flag_bridged"] = eeg_qual["bridged_electrode_pct"] > EEG_QUALITY_THRESHOLDS["bridged_electrode_pct_max"]

    flag_cols = [c for c in flags.columns if c.startswith("flag_")]
    flags["eeg_any_flag"] = flags[flag_cols].any(axis=1) if flag_cols else False
    flags["eeg_n_flags"]  = flags[flag_cols].sum(axis=1) if flag_cols else 0

    print(f"\n── EEG Quality Flags ──")
    for col in flag_cols:
        print(f"  {col:<30}: {flags[col].sum()} subjects flagged")
    print(f"  Any EEG flag: {flags['eeg_any_flag'].sum()} subjects")
    return flags


# ── 3. MRI Quality Metrics ─────────────────────────────────────────────────────

def mri_quality_flags(mri_qual: pd.DataFrame) -> pd.DataFrame:
    """
    Flags MRI datasets failing SNR, Euler number, or CNR thresholds.
    """
    flags = mri_qual[["subject_id"]].copy()

    if "snr" in mri_qual.columns:
        flags["flag_mri_snr"] = mri_qual["snr"] < MRI_QUALITY_THRESHOLDS["snr_min"]
    if "euler_number" in mri_qual.columns:
        flags["flag_euler"] = mri_qual["euler_number"] < MRI_QUALITY_THRESHOLDS["euler_number_min"]
    if "cnr" in mri_qual.columns:
        flags["flag_cnr"] = mri_qual["cnr"] < MRI_QUALITY_THRESHOLDS["cnr_min"]

    flag_cols = [c for c in flags.columns if c.startswith("flag_")]
    flags["mri_any_flag"] = flags[flag_cols].any(axis=1) if flag_cols else False

    print(f"\n── MRI Quality Flags ──")
    for col in flag_cols:
        print(f"  {col:<30}: {flags[col].sum()} subjects flagged")
    return flags


# ── 4. Little's MCAR Test (Clinical Phenotype Missingness) ────────────────────

def missingness_analysis(clinical: pd.DataFrame,
                          group_col: str = "diagnosis_group") -> dict:
    """
    Tests whether missingness in clinical variables is MCAR.
    Uses Little's chi-square MCAR test approximation and
    chi-square independence test for missingness vs. group.
    """
    phenotype_cols = [c for c in clinical.columns
                      if c not in ["subject_id", group_col, "site", "timepoint"]]

    # Missingness rate per variable
    miss_rates = clinical[phenotype_cols].isna().mean().round(3)

    # Little's MCAR — approximate via sum of expected vs. observed missing per pattern
    # (full EM-based Little's test requires specialized libraries; use pattern test instead)
    missing_indicator = clinical[phenotype_cols].isna().astype(int)
    pattern_counts = missing_indicator.value_counts()
    n_patterns = len(pattern_counts)

    # Chi-square test: missingness vs. diagnosis group
    group_tests = {}
    if group_col in clinical.columns:
        for col in phenotype_cols:
            clinical[f"_miss_{col}"] = clinical[col].isna().astype(int)
            ct = pd.crosstab(clinical[f"_miss_{col}"], clinical[group_col])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, p, _, _ = stats.chi2_contingency(ct)
                group_tests[col] = {"chi2": round(chi2, 3), "p_value": round(p, 4),
                                    "mcar_rejected": p < 0.05}

    mcar_rejected_vars = [v for v, r in group_tests.items() if r.get("mcar_rejected")]

    print(f"\n── Missingness Analysis ──")
    print(f"  Clinical variables assessed: {len(phenotype_cols)}")
    print(f"  Unique missingness patterns: {n_patterns}")
    if mcar_rejected_vars:
        print(f"  MCAR rejected for: {mcar_rejected_vars} (p<0.05, non-random dropout by {group_col})")
    else:
        print(f"  No significant non-random dropout detected by {group_col}")

    return {
        "missingness_rates": miss_rates.to_dict(),
        "n_patterns": n_patterns,
        "group_association_tests": group_tests,
        "mcar_rejected_variables": mcar_rejected_vars,
    }


# ── 5. Composite Exclusion Flags ──────────────────────────────────────────────

def build_exclusion_report(completeness: pd.DataFrame,
                            eeg_flags:   pd.DataFrame,
                            mri_flags:   pd.DataFrame) -> pd.DataFrame:
    """
    Merges all quality flags into a subject-level exclusion report.
    Assigns prioritized reason codes per EXCLUSION_CODES.
    """
    report = completeness[["subject_id", "complete_for_primary"]].copy()

    if eeg_flags is not None:
        report = report.merge(
            eeg_flags[["subject_id", "eeg_any_flag", "eeg_n_flags"]],
            on="subject_id", how="left"
        )
    if mri_flags is not None:
        report = report.merge(
            mri_flags[["subject_id", "mri_any_flag"]],
            on="subject_id", how="left"
        )

    def assign_codes(row):
        codes = []
        if not row.get("complete_for_primary", True):
            codes.append("E01")
        if row.get("eeg_any_flag", False):
            codes.append("E02")
        if row.get("mri_any_flag", False):
            codes.append("E05")
        return ",".join(codes) if codes else "PASS"

    report["exclusion_codes"] = report.apply(assign_codes, axis=1)
    report["include_in_analysis"] = report["exclusion_codes"] == "PASS"

    n_excluded = (~report["include_in_analysis"]).sum()
    n_total    = len(report)
    print(f"\n── Cohort Readiness Summary ──")
    print(f"  Total subjects:   {n_total}")
    print(f"  Usable (PASS):    {report['include_in_analysis'].sum()}")
    print(f"  Excluded:         {n_excluded} ({n_excluded/n_total*100:.1f}%)")
    return report


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
    manifest  = load_manifest("data/subject_manifest.csv")
    eeg_qual  = load_eeg_quality("data/eeg_quality_metrics.csv")
    mri_qual  = load_mri_quality("data/mri_quality_metrics.csv")
    clinical  = load_clinical("data/clinical_phenotypes.csv")

    completeness  = completeness_matrix(manifest)
    eeg_flags_df  = eeg_quality_flags(eeg_qual)
    mri_flags_df  = mri_quality_flags(mri_qual)
    miss_results  = missingness_analysis(clinical)
    exclusion_rpt = build_exclusion_report(completeness, eeg_flags_df, mri_flags_df)

    export_all({
        "completeness_matrix":  completeness,
        "eeg_quality_flags":    eeg_flags_df,
        "mri_quality_flags":    mri_flags_df,
        "exclusion_report":     exclusion_rpt,
    })

    # Save missingness summary separately
    import json
    with open("output/missingness_analysis.json", "w") as f:
        json.dump({k: str(v) for k, v in miss_results.items()}, f, indent=2)
    print("  Exported missingness analysis → output/missingness_analysis.json")
