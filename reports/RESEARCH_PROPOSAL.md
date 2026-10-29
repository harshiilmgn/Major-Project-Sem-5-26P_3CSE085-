# A Computational Assessment of Social Sector Spending Across Indian States: Trend, Cluster, and Anomaly Analysis Using CAG State Finance Data (2013–14 to 2025–26)

## CAG–ICSSR Research Initiative: Project Proposal and Detailed Plan

---

## 1. Introduction and Background
Indian states are the main spenders on the social sector. Education, health, water supply and sanitation, housing, nutrition, social security and welfare of marginalised groups are mostly State subjects under the Constitution, and states account for the bulk of general government spending on them. How much each state spends, how steadily, and how well that spending follows the budget therefore shapes human development outcomes across the country.

The Comptroller and Auditor General of India (CAG) publishes audited, standardised state finance data every year through the State Finances Audit Reports, the Finance Accounts and the Appropriation Accounts of each state. Together they form a consistent, long-running series, but they are usually read state by state and year by year. This project applies computational and data-science methods to the whole panel of states over thirteen fiscal years (2013–14 to 2025–26) to answer three questions at once: how social sector spending has moved over time (trend), which states behave alike (cluster), and where spending departs sharply from what would be expected (anomaly).

The period is analytically rich. It covers the Fourteenth Finance Commission (higher tax devolution to 42%), the bifurcation of Andhra Pradesh and creation of Telangana (2014), the introduction of GST (2017), the reorganisation of Jammu and Kashmir (2019), the COVID-19 shock (2020–21 and 2021–22), the Fifteenth Finance Commission cycle (2021–26) and the run-up to the Sixteenth Finance Commission period (from 2026–27).

---

## 2. Problem Statement
Existing analysis of state social spending tends to rely on a few headline ratios, covers short periods, or compares only a handful of states. It rarely distinguishes durable differences in spending patterns from one-off shocks, and it does not systematically flag unusual values for audit follow-up. There is a need for a reproducible, data-driven framework that:
1. Measures long-run trends consistently,
2. Groups states by the shape and level of their spending rather than by administrative labels alone, and
3. Detects anomalies that merit closer scrutiny.

---

## 3. Objectives

### Main Objective
To computationally assess social sector spending across Indian states from 2013–14 to 2025–26 using CAG state finance data, through trend, cluster and anomaly analysis.

### Specific Objectives
1. To build a clean, harmonised state-year panel of social sector expenditure from CAG sources, with per capita, share-of-GSDP and share-of-total-expenditure measures in real terms.
2. To estimate and test trends in total social sector spending and its main sub-sectors (education, health, water supply and sanitation, housing and urban development, social security and welfare, nutrition, SC/ST/OBC welfare).
3. To identify groups of states with similar spending levels and trajectories using unsupervised learning, and to profile those groups.
4. To detect statistical anomalies in spending levels, growth rates and budget-versus-actual gaps, and to examine them against known events and audit observations.
5. To test how far fiscal capacity, the Finance Commission cycle, COVID-19 and state category (general category or special category) explain differences in spending.
6. To draw policy and audit-relevant conclusions and to deliver a reproducible code base and dashboard.

---

## 4. Research Questions and Working Hypotheses

| No. | Research Question | Working Hypothesis |
| :--- | :--- | :--- |
| **RQ1** | Has social sector spending risen as a share of GSDP and of total expenditure across states? | The share of total expenditure shows a modest upward trend, with large variation between states. |
| **RQ2** | Do states converge or diverge in per capita social spending? | Absolute per capita gaps widen even if growth rates converge (sigma and beta convergence tested). |
| **RQ3** | Are there distinct clusters of states? | Three to five stable clusters emerge, only partly aligned with income level or special category status. |
| **RQ4** | How did COVID-19 change the composition of spending? | Health spending jumps in 2020–21 and 2021–22 while education and capital-linked social spending dip or stall. |
| **RQ5** | Where are the anomalies, and are they explained by shocks, reorganisation or reporting changes? | Most extreme values coincide with reorganisation, disasters, scheme changes, or large savings and excess over grants. |
| **RQ6** | Does budget execution (actual versus budget estimate) differ systematically by state group? | Lower-capacity states show larger shortfalls against budget estimates in social sub-sectors. |

---

## 5. Scope and Definitions
- **Coverage:** All 28 states, with Delhi and Puducherry added where their data are comparable. Jammu and Kashmir is treated carefully for the post-2019 break. Telangana and Andhra Pradesh are handled with a harmonisation rule.
- **Social Sector Definition:** Following the classification in CAG's State Finances reports, social services expenditure under the revenue and capital accounts covers:
  - Education, sports, art and culture (2202–2205, 4202)
  - Health and family welfare (2210–2211, 4210–4211)
  - Water supply, sanitation, housing and urban development (2215–2217, 4215–4217)
  - Information and broadcasting (2220, 4220)
  - Welfare of SC, ST, OBC and Minorities (2225, 4225)
  - Labour and labour welfare (2230, 4230)
  - Social security and welfare (2235, 4235)
  - Nutrition (2236, 4236)
  - Relief on account of natural calamities (2245)
- **Time Period:** 2013–14 to 2025–26.

---

## 6. Data Sources and Harmonisation Rules

### 6.1 Sources
- **CAG of India:** State Finances Audit Reports (SFAR), Finance Accounts, Appropriation Accounts.
- **Reserve Bank of India (RBI):** State Finances: A Study of Budgets (annual).
- **MoSPI / DES:** Gross State Domestic Product (GSDP), NSDP, and price indices.
- **Census of India / National Commission on Population:** Population projections.
- **Finance Commission Reports:** 14th and 15th FC reports.

### 6.2 Treatment of 2024–25 and 2025–26
Audited accounts for recent years become available after a lag. The recommended protocol is to maintain 2013–14 through the latest audited year as the core audited panel, and treat subsequent years as a provisional extension built from revised and budget estimates flagged separately (`is_provisional = 1`).

### 6.3 Harmonisation Rules
1. **Andhra Pradesh and Telangana:** Apportion 2013–14 using the AP Reorganisation Act population ratio (58.32% AP, 41.68% TS), and test with a combined-unit sensitivity model.
2. **Jammu & Kashmir:** Treat 2019–20 as a structural break; evaluate with and without J&K.
3. **Price Deflation:** Convert to constant 2011–12 prices using GSDP deflators.
4. **Special Category Classification:** Maintain the pre-2015 11 Special Category States as a fixed group dummy.

---

## 7. Detailed Methodology Pipeline

```
Raw CAG/RBI Data -> Harmonisation & Validation ->master panel -> Trend & Convergence Analysis
                                                              -> Unsupervised Clustering (Level & Trajectory)
                                                              -> Anomaly Detection & Audit Triangulation
                                                              -> Panel Econometric Models (Fixed & Random Effects)
                                                              -> Interactive Dashboard & Anomaly Register
```

---

## 8. 12-Month Work Plan

| Phase | Months | Core Activities | Deliverable Output |
| :--- | :--- | :--- | :--- |
| **Phase 1** | 1–2 | Literature review, definitions, data inventory, accounting head concordances | Literature review, data dictionary, harmonisation protocol |
| **Phase 2** | 2–4 | PDF extraction (pdfplumber/camelot), data cleaning, deflating, per capita computation | Clean master state-year panel & validation audit log |
| **Phase 3** | 4–5 | Exploratory analysis, CAGR, Mann-Kendall tests, Sigma & Beta convergence | Trend and convergence analytical report |
| **Phase 4** | 5–7 | Clustering (K-Means, Hierarchical, GMM), silhouette validation, profiling | Cluster profiles and stability report |
| **Phase 5** | 7–8 | Anomaly detection (MAD, IsoForest, LOF), audit triangulation | Triangulated Anomaly Register |
| **Phase 6** | 8–9 | Fixed and Random Effects panel models, Hausman test, robustness checks | Econometric regression tables |
| **Phase 7** | 9–10 | Streamlit interactive dashboard development & user testing | Interactive visual dashboard portal |
| **Phase 8** | 10–12 | Final report drafting, 4–6 page policy brief, code repository packaging | Final report, policy brief, reproducible repository |

---

## 9. Deliverables and Expected Contributions
1. A harmonised, open-source state-year panel of social sector spending (2013–14 to 2025–26).
2. A comprehensive final research report formatted for the CAG–ICSSR initiative.
3. A 4-to-6 page policy brief for state finance secretaries and audit practitioners.
4. An anomaly register listing flagged state-years with explanations and audit recommendations.
5. An open, reproducible code repository and interactive analytical dashboard.
6. Research papers for submission to peer-reviewed public finance and development journals.
