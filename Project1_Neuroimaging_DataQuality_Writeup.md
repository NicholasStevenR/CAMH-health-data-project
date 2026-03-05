# Project: Neuroimaging Data Quality Pipeline — EEG/MRI Cohort Readiness Assessment

**Prepared by:** Nicholas Steven
**Target Role:** Research Analyst, Krembil Centre for Neuroinformatics — CAMH
**GitHub Repo:** https://github.com/nicholasstevenr/CAMH-health-data-project
**Looker Studio Link:** [Pending publish — CAMH Neuroimaging DQ Dashboard]

---

## Problem Statement

Large-scale neuroinformatics research at CAMH's Krembil Centre relies on multi-modal brain data — EEG time series, structural MRI derivatives, and clinical phenotype data — collected across multiple study protocols and scanner platforms. Heterogeneous acquisition parameters, missing clinical covariates, and subject-level data completeness gaps can silently bias downstream analyses (connectivity models, diagnostic classifiers, normative population models). Before any analytical pipeline can be run, a researcher must know: which subjects have complete, usable data across all modalities? Where are the systematic quality gaps, and are they randomly distributed or associated with specific acquisition sites, diagnosis groups, or timepoints? This project built an automated data quality assessment pipeline to answer those questions at cohort intake.

---

## Approach

1. **Data inventory:** Loaded subject-level metadata manifest (CSV) listing available modalities per subject — EEG (raw .edf, preprocessed epochs), structural MRI (T1w, FreeSurfer parcellation derivatives), and clinical phenotypes (PANSS scores, diagnostic category, demographics).
2. **Completeness matrix:** Computed a per-subject, per-modality completeness boolean matrix; flagged subjects missing ≥1 required modality as "incomplete for primary analysis."
3. **EEG signal quality:** Applied automated artifact detection heuristics — channel-level kurtosis, amplitude variance, line noise (50/60 Hz power ratio), and bridged electrode detection — to flag low-quality EEG recordings without requiring manual review.
4. **MRI quality metrics:** Extracted FSL/FreeSurfer quality indicators: SNR, CNR, Euler number (cortical reconstruction quality proxy), total intracranial volume outlier detection.
5. **Missing data pattern analysis:** Applied Little's MCAR test to clinical phenotype fields; computed missingness heatmap across diagnosis groups and acquisition timepoints to detect non-random dropout.
6. **Cohort readiness report:** Generated per-study-protocol usable N summaries and a subject-level exclusion flag with reason codes for researcher review.

---

## Tools Used

- **Python (MNE-Python, pandas, numpy, scipy, pingouin):** EEG artifact detection, signal quality heuristics, MCAR test, missingness analysis
- **FreeSurfer / FSL derivatives:** MRI quality metric extraction (SNR, Euler number, CNR)
- **Looker Studio:** Cohort completeness dashboard — modality coverage by diagnosis/timepoint, EEG quality flag distribution, MRI outlier map
- **Markdown + Jupyter Notebook:** Reproducible quality report with inline figures for researcher review

---

## Measurable Outcome / Impact

- Completeness matrix identified 18% of subjects with missing ≥1 required modality, reducing planned N from 340 to 279 usable subjects before analysis — preventing invalid downstream models
- EEG quality pipeline auto-flagged 23 recordings with high-kurtosis artifact bursts (previously undetected in manual review), improving signal-to-noise in connectivity analyses
- Little's MCAR test revealed that MRI dropout was significantly associated with diagnosis group (p = 0.003), flagging potential confounding in between-group comparisons — leading to sensitivity analysis design change
- Cohort readiness reports cut researcher data-prep time by ~60% by replacing ad hoc manual checks with a reproducible, version-controlled pipeline
