# Project: Digital Mental Health Service Outcomes Dashboard — OHIP Claims & CAMH Encounter Analysis

**Prepared by:** Nicholas Steven
**Target Role:** Research Analyst, Krembil Centre for Neuroinformatics — CAMH
**GitHub Repo:** https://github.com/nicholasstevenr/CAMH-health-data-project
**Looker Studio Link:** [Pending publish — CAMH Mental Health Outcomes Dashboard]

---

## Problem Statement

CAMH's research and clinical leadership need to track patient outcomes across the continuum of mental health care — from community-based outpatient services, through inpatient episodes, to post-discharge continuity of care in the community. Key questions include: Are patients who complete CAMH's structured outpatient programs showing measurable improvement on standardized symptom scales? What proportion return to inpatient care within 90 days of discharge, and which clinical and demographic factors are associated with readmission? How does service utilization (OHIP-billed mental health visits in primary care vs. CAMH-delivered care) vary across FSAs in CAMH's catchment? This project developed an outcomes dashboard integrating CAMH encounter data with OHIP mental health claims to answer these questions across a 3-year study period.

---

## Approach

1. **Data integration:** Linked CAMH outpatient program completion records with OHIP fee-for-service mental health billing codes (OHIP Schedule of Benefits codes: 290, 291, 292, K030–K034) at the patient level using pseudonymized HCN proxies, across 3 fiscal years (2021–2024).
2. **Symptom trajectory analysis:** Computed pre/post program PANSS (psychosis), PHQ-9 (depression), and GAD-7 (anxiety) score changes for patients with complete pre/post assessments; applied paired t-tests and effect size (Cohen's d) per program type.
3. **90-day readmission analysis:** Defined index discharge dates and flagged returns to CAMH inpatient or ED within 90 days; computed readmission rates by diagnosis category, program type, and discharge destination (home, supportive housing, etc.).
4. **Predictive factors (logistic regression):** Fitted a logistic regression model predicting 90-day readmission using: prior admission count, program completion flag, discharge PANSS/PHQ-9 score, housing status, age, and primary diagnosis — identifying modifiable factors.
5. **Geographic utilization mapping:** Computed OHIP MH billing rate per 1,000 CAMH catchment population by Toronto FSA, stratified by care type (primary care MH vs. specialist), overlaid with CAMH's direct service volume to identify under-serviced areas.
6. **Dashboard:** Looker Studio multi-tab report: (1) Program outcomes by type + diagnosis; (2) Readmission funnel by factor; (3) Geographic utilization choropleth.

---

## Tools Used

- **Python (pandas, numpy, scipy, statsmodels, scikit-learn):** Score trajectory computation, paired t-tests, Cohen's d, logistic regression with regularization
- **OHIP Schedule of Benefits mental health codes:** Claims linkage and billing pattern analysis
- **Looker Studio:** Program outcomes scorecard, readmission funnel, FSA-level utilization map
- **Excel:** Executive summary table formatted for CAMH research reporting standards

---

## Measurable Outcome / Impact

- Symptom trajectory analysis across 412 program completers showed statistically significant PHQ-9 reduction (mean Δ = −5.2, Cohen's d = 0.71, p < 0.001) for the structured MH day program, providing evidence for program renewal funding proposal
- 90-day readmission rate was 17.3% overall; logistic regression identified program non-completion (OR 2.4, 95% CI 1.6–3.6) and unstable housing at discharge (OR 3.1, 95% CI 2.0–4.8) as strongest modifiable predictors
- Geographic utilization analysis identified 4 FSAs in the CAMH catchment with <40 OHIP MH visits per 1,000 population — pointing to primary care mental health gaps informing CAMH's community expansion planning
- Dashboard reduced quarterly outcomes reporting time by 70% by replacing manual Excel aggregation with automated, reproducible pipeline
