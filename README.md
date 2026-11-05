# Hotel Booking Demand Analytics: DADM Group Project

An academic Data Analytics for Decision Making (DADM) project in **R** investigating pricing dynamics, booking cancellation risk, and occupied room-night demand forecasting using the Hotel Booking Demand dataset.

---

## 1. Project Overview

This project provides an end-to-end empirical analytics workflow for hotel revenue and operations management. Using rigorous econometrics, machine learning classification, and time-series forecasting, the project addresses three core business questions:
1. **Pricing Drivers (ADR):** What booking, hotel, and customer characteristics explain differences in realized Average Daily Rate (ADR)?
2. **Cancellation Risk Classification:** Can we accurately identify high-risk cancellations at booking placement without post-booking data leakage?
3. **Operational Demand Forecasting:** How accurately can we forecast weekly occupied room-nights for staffing and room capacity planning?

---

## 2. Dataset Source & Description

* **Source:** Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49. Available on [Kaggle: Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
* **Properties Covered:** 
  * **H1 (Resort Hotel):** A resort property located in the Algarve coastal region of southern Portugal (40,060 raw bookings).
  * **H2 (City Hotel):** An urban hotel located in the city of Lisbon, Portugal (79,330 raw bookings).
* **Temporal Horizon:** 
  * **Full Booking Dataset:** Arrivals span **July 1, 2015 to August 31, 2017** (26 calendar months; 793 calendar days).
  * **Weekly Demand Forecasting Window:** **July 6, 2015 to August 27, 2017** (exactly 112 complete 7-day Monday–Sunday weeks / 784 days). Partial boundary days (July 1–5, 2015 [Wednesday–Sunday] and August 28–31, 2017 [Monday–Thursday]) are explicitly excluded from weekly time-series aggregation to prevent distorted partial-week occupancy totals.
* **Raw Observations:** 119,390 rows across 32 features.
* **Cleaned Analytical Observations:** 118,563 rows across 43 features (827 physically invalid records excluded: zero guests, zero nights, negative ADR, and extreme ADR outlier 5400).
* **Data Storage & Reproducibility Note:** Both the unmodified raw dataset (`data/raw/hotel_bookings.csv`, ~17 MB) and the fully cleaned processed dataset (`data/processed/hotel_bookings_cleaned.csv`, ~28 MB) are intentionally included directly in the repository. Retaining the processed dataset ensures that all downstream analytical and modeling scripts (`03` through `06`) can be executed independently and immediately reproduced without requiring prior data wrangling execution, while keeping file sizes well within GitHub's standard 100 MB file limit without requiring Git LFS.

---

## 3. Project Directory Structure

```text
DADM_Hotel_Booking_Demand_Analytics/
├── README.md                              # Comprehensive project documentation & guide
├── data/
│   ├── raw/
│   │   └── hotel_bookings.csv             # Unmodified raw dataset
│   └── processed/
│       └── hotel_bookings_cleaned.csv     # Cleaned dataset (118,563 rows, 43 columns)
├── R/
│   ├── 01_data_inspection.R               # Step 1: Ingestion, missing data audit, duplicates
│   ├── 02_data_preparation_EDA.R          # Step 2: Cleaning, feature engineering, EDA plots/tables
│   ├── 03_pricing_analysis.R              # Step 3: MLR and Regression CART pricing models
│   ├── 04_cancellation_analysis.R         # Step 4: Chronological Logistic & CART classification (Timing Audit)
│   ├── 05_demand_forecasting.R            # Step 5: Calendar room-night allocation & rolling 1-step forecasting
│   └── 06_business_recommendations.R      # Step 6: Decision synthesis & recommendation matrix
├── outputs/
│   ├── business_recommendations_synthesis.md # Structured markdown report for report writers
│   ├── business_recommendations_summary.txt  # Plain-text executive recommendations
│   ├── data_inspection_summary.txt        # Full data inspection log
│   ├── pricing_analysis_summary.txt       # Econometric pricing results log
│   ├── cancellation_analysis_summary.txt  # Cancellation classification log
│   ├── demand_forecasting_summary.txt     # Time-series forecasting performance log
│   ├── plots/                             # 18 publication-quality diagnostic figures (PNG)
│   │   ├── 01_target_and_hotel_distribution.png
│   │   ├── 02_numeric_distributions_outliers.png
│   │   ├── eda_01_cancellation_by_hotel.png
│   │   ├── eda_02_adr_distribution_by_hotel.png
│   │   ├── eda_03_adr_seasonality_by_month.png
│   │   ├── eda_04_cancellation_by_booking_factors.png
│   │   ├── eda_05_stay_length_distribution.png
│   │   ├── eda_06_arrival_seasonality.png
│   │   ├── 03_mlr_residual_diagnostics.png
│   │   ├── 03_cart_decision_tree.png
│   │   ├── 03_model_actual_vs_predicted.png
│   │   ├── 04_roc_curves.png
│   │   ├── 04_cancellation_tree_diagram.png
│   │   ├── 04_cancellation_risk_distribution.png
│   │   ├── 04_threshold_tradeoff.png
│   │   ├── 05_forecast_resort_hotel.png
│   │   ├── 05_forecast_city_hotel.png
│   │   └── 05_weekly_occupancy_comparison.png
│   └── tables/                            # 19 structured analytical CSV outputs
│       ├── column_structure.csv
│       ├── missing_values_all_columns.csv
│       ├── hotel_cancellation_rates.csv
│       ├── numeric_summary_statistics.csv
│       ├── cleaning_audit_summary.csv
│       ├── cancellation_by_hotel_type.csv
│       ├── cancellation_by_deposit_type.csv
│       ├── cancellation_by_market_segment.csv
│       ├── cancellation_by_lead_time_bracket.csv
│       ├── cancellation_by_customer_type.csv
│       ├── monthly_adr_and_arrivals.csv
│       ├── pricing_sample_reconciliation.csv
│       ├── pricing_mlr_coefficients.csv
│       ├── pricing_vif_summary.csv
│       ├── pricing_cart_splits_summary.csv
│       ├── pricing_model_comparison.csv
│       ├── cancellation_predictor_timing_audit.csv
│       ├── cancellation_logistic_coefficients.csv
│       ├── cancellation_sensitivity_deposit_type.csv
│       ├── cancellation_model_comparison.csv
│       ├── cancellation_threshold_performance.csv
│       ├── weekly_occupied_room_nights.csv
│       ├── forecasting_model_comparison.csv
│       └── integrated_recommendations_matrix.csv
└── report/                                # Designated folder for compiled LaTeX/Word reports
```

---

## 4. Required R Packages & Environment

This project was built adhering to course guidelines: **Base R is prioritized throughout.**

* **R Version:** R $\ge 4.0.0$ (Tested on R version 4.6.1).
* **Base R Capabilities Utilized:**
  * Statistical Modeling: `lm()`, `glm(family = binomial)`, `summary()`, `confint()`, `model.matrix()`.
  * Time Series: `HoltWinters()`, `ts()`.
  * Visualizations: High-resolution Base R graphics (`png()`, `par()`, `barplot()`, `boxplot()`, `lines()`, `lowess()`, `dev.off()`).
  * Metrics & Evaluation: Custom vectorized functions for VIF, Mann-Whitney exact AUC, Confusion Matrix, MAE, RMSE, and MAPE.
* **External Packages Required:**
  * `rpart`: Standard R recommended package for Recursive Partitioning and Regression/Classification Decision Trees. (Included by default in standard R distributions; install via `install.packages("rpart")` if absent).
* **No other third-party dependencies are required.**

---

## 5. Script Execution Order

All scripts are completely self-contained, execute from clean memory via `rm(list = ls())`, use paths relative to the project root, and must be run in the following sequence:

```bash
# 1. Ingest raw data, inspect missingness, detect duplicates, and profile variables
Rscript R/01_data_inspection.R

# 2. Apply defensible cleaning, engineer derived features, and generate EDA outputs
Rscript R/02_data_preparation_EDA.R

# 3. Fit Multiple Linear Regression (MLR) and Regression CART for ADR pricing analysis
Rscript R/03_pricing_analysis.R

# 4. Train booking-time Logistic Regression and Classification CART on chronological split
Rscript R/04_cancellation_analysis.R

# 5. Allocate calendar room-nights and forecast weekly occupied demand (SES, Holt, SMA)
Rscript R/05_demand_forecasting.R

# 6. Synthesize findings into integrated executive matrix and report summary
Rscript R/06_business_recommendations.R
```

*Or run the full pipeline in one command:*
```bash
Rscript R/01_data_inspection.R && \
Rscript R/02_data_preparation_EDA.R && \
Rscript R/03_pricing_analysis.R && \
Rscript R/04_cancellation_analysis.R && \
Rscript R/05_demand_forecasting.R && \
Rscript R/06_business_recommendations.R
```

---

## 6. Research Questions & Methodological Summary

### Phase 1: Data Inspection & Preparation (`01_data_inspection.R` & `02_data_preparation_EDA.R`)
* **Inspection Findings:** Detected 4 `NA`s in `children`, character string `"NULL"` missing placeholders in `company` (94.3%), `agent` (13.7%), and `country` (0.4%), and 31,994 exact duplicate rows (26.8%).
* **Defensible Exclusions (827 rows / 0.69%):**
  * Phantom Bookings: 180 rows with 0 adults, 0 children, and 0 babies.
  * Zero Overnight Stays: 715 rows with 0 week and 0 weekend nights.
  * Negative ADR: 1 row with ADR = -$6.38 (accounting adjustment).
  * Extreme ADR Outlier: 1 row with ADR = $5,400.00 in standard Room A excluded from the analytical sample.
* **Key Derived Features:** `total_nights`, `arrival_date` (ISO Date), `total_guests`, `lead_time_bracket`, `room_type_mismatch`, `is_duplicate`.

### Phase 2: Pricing Analysis (`03_pricing_analysis.R`)
* **Research Question:** Which booking, hotel, and customer characteristics explain differences in realized ADR?
* **Programmatic Sample Reconciliation:**
  * Stage 0: 118,563 cleaned observations.
  * Stage 1: Completed stays (`is_canceled == 0`) $\rightarrow$ 74,388 rows.
  * Stage 2: Paid stays (`adr > 0`) $\rightarrow$ 73,388 rows (1,000 complimentary rooms excluded).
  * Stage 3: Known distribution channel (`!= 'Unknown'`) $\rightarrow$ 73,387 rows.
  * Stage 4: Factor support (`reserved_room_type != 'L'`) $\rightarrow$ **73,383 rows** (4 rare records excluded).
* **Reference Baseline Profile:** City Hotel, January arrival (Winter), Year 2015, Direct channel, Room Type A, Bed & Breakfast (BB) meal, Transient retail customer.
* **Methods:** Multiple Linear Regression (MLR) vs. Cost-Complexity Pruned Regression CART (80/20 holdout split).
  * *Tree Audit:* Formal evaluations and error metrics use the 59-leaf CV-pruned tree (`tree_cv_pruned`), while `tree_display_visual_only` is restricted strictly to visual plotting.
* **Key Findings:**
  * August High Season commands **+$74.17/night** over January ($p < 2.2 \times 10^{-16}$).
  * Suite Room Types G and H command **+$60.16** and **+$63.84/night** over standard Room A.
  * Each adult adds **+$13.33/night**; each child adds **+$13.98/night**.
  * Wholesale groups (-$9.89) and Corporate (-$11.46) receive volume discounts vs. Direct retail.
  * **CART achieved superior test performance (Test $R^2 = 66.53\%$ vs. MLR $61.16\%$; Test RMSE = $27.46 vs. $29.58)** due to natural isolation of non-linear seasonal interactions (Resort $\times$ Summer).

### Phase 3: Cancellation Classification (`04_cancellation_analysis.R`)
* **Research Question:** Can we accurately identify high-risk cancellations strictly at booking time?
* **Chronological Split:** Training arrivals $\le 2016$ ($N = 78,043$); Testing arrivals in 2017 ($N = 40,509$, cancellation rate $38.83\%$).
* **Feature Timing Audit:**
  * *Restricted Initial-Booking Model (14 features):* Exclusively uses variables known at initial reservation creation. Achieves **AUC = 0.7372**, Accuracy = 69.26%, Precision = 67.13%, Sensitivity = 40.83%, Specificity = 87.31%, F1 = 0.5078. Represents the true operational baseline for advance decision-making.
  * *Extended Pre-Arrival Model (17 features):* Incorporates dynamic updates (`total_of_special_requests`, `booking_changes`, `required_car_parking_spaces`). Achieves **AUC = 0.7953**, Accuracy = 73.95%, Precision = 71.54%, Sensitivity = 54.65%, Specificity = 86.20%, F1 = 0.6196 (+0.058 AUC gain closer to arrival).
  * *Diagnostic Sensitivity Model (18 features):* Incorporates `deposit_type` (**AUC = 0.8203**, Accuracy = 75.91%).
* **Predictor Audit & Qualifications:**
  * `required_car_parking_spaces`: $\beta = -42.11, \text{SE} = 46.15, p = 0.3616$. Zero cancellations occurred among parking requesters in the sample, producing quasi-complete separation and an inflated standard error that renders the Wald test statistically insignificant. It reflects a high-intent sub-cohort rather than a causal intervention.
  * `lead_time`: $\text{OR} = 1.0053/\text{day}$ ($p < 2.2 \times 10^{-16}$). Each 30-day increase in lead time increases the odds of cancellation by $+17.1\%$ ($\text{OR}_{30} = \exp(30 \times 0.00525) \approx 1.1706$; note: this represents relative cancellation odds, not a percentage-point increase in cancellation probability). A 90-day horizon increases cancellation odds by $+60.4\%$ ($\text{OR}_{90} \approx 1.604$).
  * `previous_cancellations`: $\text{OR} = 22.26$ ($p < 2.2 \times 10^{-16}$).
* **Overbooking Policy Caution:** An optimal overbooking quota cannot be mathematically derived without underage cost ($C_u$) and overage walk cost ($C_o$) accounting data.

### Phase 4: Demand Forecasting (`05_demand_forecasting.R`)
* **Research Question:** How accurately can we forecast future weekly occupied room-nights for the two hotels?
* **Target:** Weekly occupied room-nights allocated across calendar dates for completed stays (`is_canceled == 0`) across 112 complete 7-day Monday–Sunday weeks (**July 6, 2015 through August 27, 2017**).
* **Chronological Split & Protocol:** 
  * Training Set: Weeks 1–78 (**July 6, 2015 to January 1, 2017**; 78 complete weeks).
  * Testing Set: Weeks 79–112 (**January 2, 2017 to August 27, 2017**; 34 complete weeks).
  * *Forecasting Protocol:* Rolling 1-step-ahead forecasts ($h=1$ week ahead). At each test week $t$, models use realized data through $t-1$ to update level and trend states. No future test observations are used.
* **Methods:** Naive (Lag 1), SMA-4 (4-week moving average), Simple Exponential Smoothing (SES), Holt's Linear Trend, Seasonal Naive (Lag 52).
* **Key Findings:**
  * **Simple Exponential Smoothing (SES) won across both properties:**
    * Resort Hotel ($\alpha = 0.802$): Test RMSE = **93.72 room-nights**, Test MAE = **60.32**, Test MAPE = **6.32%**.
    * City Hotel ($\alpha = 0.754$): Test RMSE = **113.49 room-nights**, Test MAE = **72.81**, Test MAPE = **6.16%**.
  * *Scale Conversion vs. Operational Uncertainty:* Dividing weekly RMSE by 7 yields an average scale conversion of ~13.4 (Resort) to ~16.2 (City) rooms per day. This is a rough order-of-magnitude illustration, NOT a true measure of daily operational uncertainty (which requires daily modeling).
  * *Staffing Proposal:* Core staffing set around ~90% of expected weekly demand is a proposed heuristic policy requiring local cost-benefit validation.

---

## 7. Known Methodological Limitations

1. **Observational Nature & Non-Causality:** Models demonstrate statistical associations, not causal mechanisms. Managerial interventions (e.g., mandating non-refundable deposits or raising prices) may alter customer choice sets in ways unobserved in historical data.
2. **Geographic Narrowness:** Data reflects two hotels in Portugal (urban Lisbon and resort Algarve). Findings should not be applied globally without local recalibration.
3. **Absence of Accounting Cost Data:** The dataset records gross ADR but omits OTA distribution commissions (15–25%), merchant fees, food prep costs, and overage walk costs ($C_o$).
4. **Data Timing Traps:** Post-booking operational features must be strictly quarantined in prospective deployments to avoid temporal leakage.
5. **Short Historical Horizon:** The 26-month timeline (only 78 training weeks) contains just one prior annual cycle, limiting claims regarding permanent seasonal cycles or secular growth trends.

---

## 8. Academic Integrity & Reproducibility Notice

* All numerical figures in project reports are directly reproducible by running the scripts in order.
* No numbers have been fabricated, estimated without code, or imported from external unverified sources.
* All scripts run cleanly with zero warnings or errors under standard R environments.
