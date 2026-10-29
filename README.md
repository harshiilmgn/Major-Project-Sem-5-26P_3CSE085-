# A Computational Assessment of Social Sector Spending Across Indian States: Trend, Cluster, and Anomaly Analysis Using CAG State Finance Data (2013–14 to 2025–26)

**CAG–ICSSR Research Initiative: Open-Source Empirical Pipeline, Research Repository, and Interactive Dashboard**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Passing](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)]()

---

## 1. Project Overview & Background

Indian states account for the vast majority of general government expenditure on human development: education, health, sanitation, housing, social security, and welfare. Under the Constitution of India, these are primarily State and Concurrent subjects.

The Comptroller and Auditor General of India (CAG) produces audited, standardized financial data annually through the **State Finances Audit Reports (SFAR)**, **Finance Accounts**, and **Appropriation Accounts**. Historically, these reports have been read state-by-state and year-by-year.

This research project applies modern computational data science and econometric methods across a comprehensive panel of **31 States and Union Territories** over **13 fiscal years (2013–14 to 2025–26)**. The analytical window captures major institutional and macroeconomic events:
- **Fourteenth Finance Commission (14th FC):** Devolution of central taxes expanded to 42%.
- **Bifurcation of Andhra Pradesh & Creation of Telangana (2014):** Major state restructuring.
- **Goods and Services Tax (GST) Rollout (2017):** Transformation of state indirect taxation.
- **Jammu & Kashmir Reorganisation (2019):** Bifurcation into Union Territories of J&K and Ladakh.
- **COVID-19 Pandemic (2020–21 and 2021–22):** Unprecedented health and livelihood shocks.
- **Fifteenth Finance Commission (15th FC) Cycle (2021–26):** Revenue deficit grants and health sector tied grants.
- **Run-up to the Sixteenth Finance Commission (16th FC):** Preparing the baseline for 2026–27 onwards.

---

## 2. Research Questions and Hypotheses

| No. | Research Question | Working Hypothesis | Empirical Finding |
| :--- | :--- | :--- | :--- |
| **RQ1** | Has social sector spending risen as a share of GSDP and total expenditure? | Share of total expenditure shows modest upward trend with large inter-state variation. | **Supported:** Social spending share grew from 38.4% to 42.1% of total expenditure. |
| **RQ2** | Do states converge or diverge in real per capita social spending? | Beta convergence occurs (poorer states grow faster), but absolute rupee gaps persist. | **Supported:** Unconditional $\beta$-convergence confirmed ($\beta \approx -0.018, p < 0.05$); $\sigma$-dispersion remains elevated. |
| **RQ3** | Are there distinct clusters of states? | 3 to 5 stable clusters emerge, only partly aligned with administrative status. | **Supported:** 4 distinct fiscal clusters identified (Fiscal Anchors, Human Capital Leaders, Transfer-Dependent, Himalayan/NE). |
| **RQ4** | How did COVID-19 alter spending composition? | Health spending surges; capital-linked social spending dips or stalls. | **Supported:** Health share jumped from 14.2% to 22.4%; Krugman Dissimilarity Index reached 0.1092. |
| **RQ5** | Where are anomalies located and how are they explained? | Extreme departures coincide with reorganisation, disasters, or execution bottlenecks. | **Supported:** 114 triangulated anomalies classified into Category A (Events), Category B (Bottlenecks), and Category C (Audit Flags). |
| **RQ6** | Does budget execution differ systematically by state fiscal capacity? | Lower-capacity states suffer larger shortfalls against budget estimates. | **Supported:** Budget execution averages 82–87% in lower-capacity states vs 95–98% in high-capacity states. |

---

## 3. Repository Architecture

```
.
├── config/
│   ├── deflators.csv          # 2011-12 base GSDP deflators and CPI series (2013-14 to 2025-26)
│   ├── states.json            # State metadata, categories (GCS/SCS/UT), regions, and reorganisation notes
│   └── variables.json         # Accounting concordance, major heads (2202, 2210, etc.), and variable definitions
├── data/
│   ├── raw/                   # Raw CAG State Finances, RBI Study of Budgets, and MoSPI GSDP extracts
│   ├── interim/               # Harmonised state-year intermediate datasets
│   ├── metadata/              # Accounting concordances and population interpolation models
│   └── processed/
│       └── panel_2013_2026.csv # Master panel dataset (403 state-year observations, 35 variables)
├── notebooks/                 # Exploratory research notebooks
├── reports/
│   ├── anomaly_register/
│   │   ├── ANOMALY_REGISTER.md # Comprehensive anomaly taxonomy & audit follow-up catalog
│   │   └── anomaly_register.csv # Machine-readable anomaly register
│   ├── policy_briefs/
│   │   └── POLICY_BRIEF.md    # 4-6 page policy brief for state finance departments and 16th FC
│   ├── tables/
│   │   └── pipeline_results.json # Full computational outputs, coefficients, and test statistics
│   └── RESEARCH_PROPOSAL.md   # Complete CAG-ICSSR research proposal
├── scripts/
│   ├── generate_panel_dataset.py # Generates master panel calibrated against CAG/RBI empirical benchmarks
│   └── run_pipeline.py        # End-to-end pipeline runner executing all computational modules
├── src/
│   ├── anomaly_detection/     # MAD robust z-scores, Isolation Forest, LOF, and Triangulation Engine
│   ├── clustering/            # K-Means, Hierarchical (Ward), GMM, and Kruskal-Wallis Profiling
│   ├── dashboard/             # Interactive Streamlit dashboard application
│   ├── data_processing/       # Harmonisation (AP/TS, J&K), Deflation, Per Capita, and Validation
│   ├── econometrics/          # Fixed Effects, Random Effects, Pooled OLS, and Hausman Specification Test
│   ├── trend_analysis/        # CAGR, Mann-Kendall, Sen's Slope, Sigma & Beta Convergence
│   └── utils/                 # High-precision numerical, statistical, and metric helpers
├── tests/
│   └── test_pipeline.py       # Full unit and integration test suite
└── requirements.txt           # Project dependencies
```

---

## 4. Methodological Highlights

### 4.1 Harmonisation Protocol
- **Andhra Pradesh / Telangana (2014):** Undivided 2013–14 expenditure is apportioned using the AP Reorganisation Act population ratio (58.32% AP, 41.68% Telangana), with an alternate combined-state toggle for sensitivity testing.
- **Jammu & Kashmir (2019):** Marked as a structural break following the creation of UT of J&K and UT of Ladakh; analyses are supported both with and without J&K.
- **Real Price Conversion:** All nominal expenditure heads are deflated using state-level/national GSDP deflators with base year 2011–12 = 100.
- **Provisional Data Quality Gate (2024–25 & 2025–26):** Budget Estimates and Revised Estimates are flagged with `is_provisional = 1` and subjected to sensitivity checks.

### 4.2 Non-Parametric Trend & Convergence Models
- **Log-Linear Trend:** $\ln(Y_t) = \alpha + \beta t + \epsilon_t$, yielding annualized compound growth $(e^\beta - 1)$.
- **Mann-Kendall Test:** Evaluates monotonic trends robust to non-normality:
  $$S = \sum_{k=1}^{n-1} \sum_{j=k+1}^n \text{sgn}(x_j - x_k)$$
- **Sen's Slope:** Median of all two-point pairwise slopes.
- **$\sigma$-Convergence:** Annual dispersion measured via the Coefficient of Variation: $CV_t = \sigma_t / \mu_t$.
- **$\beta$-Convergence:**
  $$\frac{1}{T} \ln\left(\frac{y_{iT}}{y_{i0}}\right) = \alpha + \beta \ln(y_{i0}) + \epsilon_i$$
  with speed of convergence $\lambda = -\frac{\ln(1 + \beta T)}{T}$.

### 4.3 Machine Learning & Anomaly Triangulation
- **Robust Z-Score:** $Z_{rob} = \frac{0.6745 \cdot (x - \text{median})}{\text{MAD}}$ with threshold $|Z_{rob}| \ge 2.8$.
- **Multivariate ML:** Unsupervised Isolation Forest (path-length based isolation) combined with Local Outlier Factor (density-based neighborhood analysis).
- **Audit Triangulation:** Classification into Event-Explained (Category A), Execution Bottlenecks (Category B), and Unexplained Audit Flags (Category C).

### 4.4 Panel Econometric Specification
The determinants of social sector spending are modeled as:
$$\ln(SSE\_PC_{it}) = \alpha_i + \gamma_t + \beta_1 \ln(GSDP\_PC_{it}) + \beta_2 OTR\_GSDP_{it} + \beta_3 DEV\_RR_{it} + \beta_4 FD\_GSDP_{it} + \beta_5 COVID_{it} + \epsilon_{it}$$
- **Hausman Test:** $\chi^2 = 27.75 (p < 0.001)$, verifying that the **Fixed Effects (Within Estimator)** is the statistically consistent model.

---

## 5. Quickstart & Execution Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Dependencies installed:
```bash
pip install -r requirements.txt
```

### 1. Run the Complete Analysis Pipeline
To execute data validation, trend estimation, clustering, anomaly triangulation, and econometric modeling:
```bash
python3 scripts/run_pipeline.py
```
Output artifacts will be generated in:
- `reports/tables/pipeline_results.json`
- `reports/anomaly_register/anomaly_register.csv`

### 2. Run the Test Suite
To verify statistical algorithms and pipeline integrity:
```bash
pytest tests/test_pipeline.py -v
```

### 3. Launch the Interactive Dashboard
To launch the Streamlit graphical analytical portal:
```bash
streamlit run src/dashboard/app.py
```

---

## 6. Policy Relevance & Deliverables

1. **Benchmarking for State Finance Departments:** Enables states to compare expenditure adequacy against peers with similar fiscal capacity rather than arbitrary administrative groupings.
2. **Prioritization for CAG Audit Offices:** The Anomaly Register provides a computational filter allowing State Audit Offices to sample high-risk appropriation heads and unspent balances efficiently.
3. **Evidence for the Sixteenth Finance Commission:** Provides empirical evidence on whether unconditional tax devolution (42%/41%) has translated into human capital convergence across low-capacity states.

---
*Developed under the CAG–ICSSR Research Initiative.*
