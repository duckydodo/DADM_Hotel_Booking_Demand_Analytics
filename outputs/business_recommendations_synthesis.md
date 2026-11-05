# DADM Hotel Booking Demand Analytics: Executive Decision Synthesis

> **Target Audience:** DADM Project Evaluators, Hotel General Managers, and Academic Report Writers  
> **Source Data:** Kaggle Hotel Booking Demand Dataset (Antonio, de Almeida, & Nunes, 2019)  
> **Sample Scope:** 118,563 cleaned observations across City Hotel and Resort Hotel (2015–2017)

---

## 1. Executive Summary & Core Insights

This project applied rigorous econometrics, machine learning classification, and time-series forecasting to evaluate three core operational dimensions of hotel performance:
1. **Pricing Drivers (ADR):** Modeled on 73,383 completed, paid reservations (reconciled programmatically from 118,563 cleaned records) via Multiple Linear Regression (MLR, $R^2 = 61.16\%$) and Regression CART ($R^2 = 66.53\%$). Seasonality, room tiering, party size, and channel contracts dominate rate realization.
2. **Cancellation Risk (`is_canceled`):** Evaluated strictly on an out-of-sample 2017 chronological test set ($N = 40,509$). The **Restricted Initial-Booking Logistic Regression** (14 predictors known at booking creation) achieved an **AUC of 0.7372** (Accuracy $69.26\%$, Precision $67.13\%$, Specificity $87.31\%$, F1 $0.5078$), representing the true advance operational baseline. Incorporating dynamic pre-arrival updates (special requests, booking changes) in the **Extended Model** lifted **AUC to 0.7953** (Accuracy $73.95\%$, Precision $71.54\%$, F1 $0.6196$), outperforming CART (AUC $0.7277$).
3. **Occupied Room-Night Forecasting:** Weekly occupied room-nights were reconstructed across calendar dates for completed stays ($N = 112$ complete 7-day weeks, July 6, 2015 to August 27, 2017). Simple Exponential Smoothing (SES) achieved the lowest Test RMSE for both properties (**Resort: 93.72 room-nights, MAPE 6.32% | City: 113.49 room-nights, MAPE 6.16%**). Dividing weekly RMSE by 7 yields a rough scale conversion of ~13–16 rooms/day (not true daily operational uncertainty).

---

## 2. Pillar 1: Pricing Decisions & ADR Associations

### Programmatic Analytical Sample Reconciliation
To ensure total transparency, every stage of data filtering from the 118,563 cleaned rows to the analytical sample was tracked programmatically:

| Filtering Stage | Retained Count | Excluded Count | % of Cleaned Data | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Stage 0: Processed Cleaned Dataset** | 118,563 | 0 | 100.00% | Verified baseline dataset. |
| **Stage 1: Completed Stays (`is_canceled == 0`)** | 74,388 | 44,175 | 62.74% | Restrict to stays where room was occupied and rate realized. |
| **Stage 2: Paid Stays (`adr > 0`)** | 73,388 | 1,000 | 61.90% | Exclude zero-rate non-commercial/complimentary rooms. |
| **Stage 3: Known Channel (`!= 'Unknown'`)** | 73,387 | 1 | 61.90% | Exclude unclassifiable distribution channel record. |
| **Stage 4: Factor Support (exclude Room 'L')** | 73,383 | 4 | 61.89% | Exclude ultra-rare room type lacking train/test balance. |
| **FINAL ANALYTICAL SAMPLE ($N$)** | **73,383** | **45,180** | **61.89%** | Robust sample with full factor support across splits. |

> [!NOTE]
> **Survivorship Bias Caution:** Restricting analysis to completed stays introduces selection bias. Canceled reservations may have had different pricing sensitivity. Findings reflect conditional observational associations of realized demand rather than unconditional market willingness-to-pay.

### Statistical Drivers of ADR (Multiple Linear Regression)
*Reference Baseline Profile:* City Hotel, January arrival (Winter), Year 2015, Direct booking channel, Standard Room Type A, Bed & Breakfast (BB) meal, Transient retail customer.

| Characteristic / Factor | MLR Estimate ($\beta$) | $t$-statistic | $p$-value | Observational Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **August High Season** | **+$74.17** | 111.04 | $< 2.2 \times 10^{-16}$ | Peak annual rate markup over January baseline; July follows at +$62.62/night. |
| **Resort Property Baseline** | **-$21.82** | -76.10 | $< 2.2 \times 10^{-16}$ | Resort baseline winter rate sits $21.82 below City Hotel before summer surge. |
| **Annual Inflation (2017 vs 2015)** | **+$29.21** | 61.93 | $< 2.2 \times 10^{-16}$ | Reflects macroeconomic/destination rate appreciation over the sample window. |
| **Suite Room Type H** | **+$63.84** | 34.69 | $< 2.2 \times 10^{-16}$ | Top-tier executive suite markup over standard entry-level Room A. |
| **Suite Room Type G** | **+$60.16** | 56.49 | $< 2.2 \times 10^{-16}$ | Multi-room suite markup over Room A. |
| **Deluxe Room Type D** | **+$15.54** | 44.08 | $< 2.2 \times 10^{-16}$ | Mid-tier room upgrade premium over Room A. |
| **Full Board (FB) Meal** | **+$52.12** | 26.70 | $< 2.2 \times 10^{-16}$ | 3-meal full package premium over Bed & Breakfast. |
| **Half Board (HB) Meal** | **+$27.00** | 68.01 | $< 2.2 \times 10^{-16}$ | Breakfast + dinner package premium over Bed & Breakfast. |
| **Adult Occupants** | **+$13.33** | 47.86 | $< 2.2 \times 10^{-16}$ | Incremental rate associated with each additional adult guest. |
| **Child Occupants** | **+$13.98** | 35.62 | $< 2.2 \times 10^{-16}$ | Incremental rate associated with each child occupant. |
| **Online Travel Agency (OTA)** | **+$9.47** | 9.63 | $< 2.2 \times 10^{-16}$ | Dynamic retail distribution associated with higher unit ADR than Direct retail. |
| **Corporate Segment** | **-$11.46** | -9.26 | $< 2.2 \times 10^{-16}$ | Negotiated corporate contract concession below Direct retail. |
| **Wholesale Groups Segment** | **-$9.89** | -10.02 | $< 2.2 \times 10^{-16}$ | Bulk volume concession granted to tour group organizers. |
| **Lead Time (per day)** | **-$0.097** | -57.51 | $< 2.2 \times 10^{-16}$ | Early-booking pattern: booking 100 days early is associated with ~$9.70 lower ADR. |
| **Length of Stay (per night)** | **-$0.173** | -3.14 | $1.66 \times 10^{-3}$ | Minor extended-stay rate concession. |

### Model Architecture Comparison
- **Regression Tree (Pruned CART):** Test RMSE **$27.46** | Test MAE **$19.41** | Test $R^2$ **66.53%**
- **Multiple Linear Regression (MLR):** Test RMSE **$29.58** | Test MAE **$21.81** | Test $R^2$ **61.16%**
- *Why CART Outperformed MLR:* CART inherently splits on complex non-linear interactions without manual feature engineering (e.g., Resort Hotel rates drop to $61.47 in winter but surge past $188 in summer, an effect MLR averages into constant additive terms unless interaction terms are specified).

---

## 3. Pillar 2: Cancellation Risk Classification

### Chronological Test Evaluation & Feature Timing Audit ($N = 40,509$, 2017 Arrivals)

| Model Architecture | Feature Timing Scope | Accuracy | Precision | Sensitivity | Specificity | F1-Score | AUC-ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Restricted Logistic Regression** | **Strict Initial Booking (14 features)** | **69.26%** | **67.13%** | **40.83%** | **87.31%** | **0.5078** | **0.7372** |
| 2. Restricted CART (CV Pruned) | Strict Initial Booking (14 features) | 69.89% | 79.24% | 30.45% | 94.94% | 0.4399 | 0.7339 |
| **3. Extended Logistic Regression** | **Extended: Initial + Pre-Arrival (17 features)** | **73.95%** | **71.54%** | **54.65%** | **86.20%** | **0.6196** | **0.7953** |
| 4. Extended CART (CV Pruned) | Extended: Initial + Pre-Arrival (17 features) | 73.62% | 76.05% | 46.79% | 90.65% | 0.5793 | 0.7277 |
| *5. Sensitivity Logistic Regression* | *Diagnostic: Extended + deposit_type (18)* | *75.91%* | *79.46%* | *51.18%* | *91.60%* | *0.6226* | *0.8203* |

### Operational Significance of Prediction Timing
1. **Initial Booking Horizon (100–30 Days Out):** Only the Restricted Model is legitimate at this stage. Special requests, booking amendments, and parking requirements have not yet occurred. The Restricted Model provides an **AUC of 0.7372**, enabling management to identify inherently risky reservation cohorts (long lead times, wholesale groups, repeat cancellers) for deposit rules.
2. **Dynamic Pre-Arrival Window (14–3 Days Out):** Incorporating customer engagement (special requests, booking changes) lifts discrimination to **AUC 0.7953** (+0.058 AUC). Guest communication closer to arrival provides substantial predictive lift.
3. **Qualification of the Parking Predictor (`required_car_parking_spaces`):** In the sample, zero guests who requested parking cancelled. However, because of quasi-complete separation, the Wald test standard error is heavily inflated ($\beta = -42.11, \text{SE} = 46.15, z = -0.91, p = 0.3616$), rendering the coefficient **statistically insignificant**. While parking requesters reflect an empirical sub-cohort with high intent, this cannot be interpreted as a causal mechanism or an operational panacea.
4. **Lead Time Effect:** Each 30 days of lead time increases cancellation odds by **+17.1%** ($\text{OR}_{30} = \exp(30 \times 0.00525) \approx 1.1706$; note: this represents relative odds of canceling versus not canceling, not a percentage-point increase in cancellation probability). A booking made 90 days in advance carries +60.4% higher odds of cancellation ($\text{OR}_{90} \approx 1.604$) than a same-day arrival.
5. **Prior History:** Guests with prior cancellations exhibit a **22.3x higher odds** of re-canceling (OR = 22.26).

### Why Direct Optimal Overbooking Cannot Be Claimed
Determining an optimal overbooking quota requires minimizing the expected financial cost function via the classic Newsvendor critical fractile:
$$\text{Critical Fractile } P^* = \frac{C_u}{C_u + C_o}$$
- **Underage Cost ($C_u$):** Lost contribution margin of an unrented empty room (ADR minus marginal cleaning/amenity cost).
- **Overage Cost ($C_o$):** Total financial, operational, and reputational penalty of 'walking' a booked guest with a confirmed reservation (rebooking cost at rival property, taxi transport, meal vouchers, and severe customer lifetime value destruction).
- *Crucial Missing Data:* The dataset contains **zero accounting cost data**, no hotel room capacity limits, and no record of walk expenses. Recommending a hard numerical overbooking margin without $C_u$ and $C_o$ is methodologically unsound.

---

## 4. Pillar 3: Operational Demand Forecasting

### 34-Week Holdout Test Performance (Weeks 79–112: Jan 2, 2017 – Aug 27, 2017)

| Hotel Property | Forecasting Model | Test MAE (Room-Nights) | Test RMSE (Room-Nights) | Test MAPE (%) | Model Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Resort Hotel** | **Simple Exponential Smoothing (SES, $\alpha = 0.802$)** | **60.32** | **93.72** | **6.32%** | **Best for Resort (Lowest RMSE)** |
| Resort Hotel | Holt's Linear Trend ($\alpha = 0.850, \beta = 0.105$) | 64.20 | 98.85 | 6.61% | Close runner-up |
| Resort Hotel | Simple Moving Average (SMA-4) | 63.29 | 103.67 | 6.42% | Smooths weekly noise |
| Resort Hotel | Naive Benchmark (Lag 1) | 72.82 | 117.16 | 8.15% | High volatility lag |
| Resort Hotel | Seasonal Naive (Lag 52) | 104.38 | 165.80 | 10.93% | Under-predicts growth |
| | | | | | |
| **City Hotel** | **Simple Exponential Smoothing (SES, $\alpha = 0.754$)** | **72.81** | **113.49** | **6.16%** | **Best for City (Lowest RMSE)** |
| City Hotel | Holt's Linear Trend ($\alpha = 0.733, \beta = 0.080$) | 72.09 | 121.63 | 6.06% | Strong trend tracker |
| City Hotel | Simple Moving Average (SMA-4) | 80.40 | 126.31 | 6.63% | Robust 4-week lag |
| City Hotel | Naive Benchmark (Lag 1) | 83.32 | 128.90 | 7.19% | Baseline lag |
| City Hotel | Seasonal Naive (Lag 52) | 145.12 | 205.09 | 12.19% | Under-predicts growth |

### Operational Significance for Hotel Staffing and Inventory
1. **Scale Conversion vs Daily Uncertainty:** Dividing weekly RMSE by 7 yields an average scale conversion of ~13.4 (Resort) to ~16.2 (City) rooms per day. It is critical to recognize that this is a rough order-of-magnitude scaling for intuitive comparison, NOT a true estimate of daily operational forecast uncertainty. Intra-week demand variance, day-of-week arrival surges, and weekend spikes are smoothed out by weekly aggregation. Day-level operational staffing requires dedicated daily-level modeling.
2. **Housekeeping Rostering Policy:** Housekeeping shifts depend directly on daily occupied rooms rather than arrival counts. Management can set core permanent staffing around a proposed heuristic policy (such as ~90% of expected weekly demand or baseline minus 1 SD), utilizing flexible on-call agency labor to absorb demand peaks. This proposal represents an operational heuristic requiring local empirical validation (comparing idle permanent wages vs agency overtime premiums) rather than an optimal mathematical solution.
3. **Food & Beverage Procurement:** Buffet culinary ordering (accounting for $>75\%$ of guest meal packages) can be placed 1 week in advance using the SES projection, reducing perishable waste.
4. **Maintenance Windows:** The sharp drop in Resort demand during November–February (down to 350 room-nights/week vs >1,250 in summer) identifies the exact window to take entire guest wings offline for refurbishment without displacing paying guests.

---

## 5. Integrated Recommendations (Finding → Evidence → Action → Limitation)

### Recommendation 1: Dynamic Seasonal Rate Differentiation
- **Finding:** Demand and ADR surge dramatically during summer, creating intense seasonal rate concentration.
- **Evidence:** August ADR commands a +$74.17/night premium over January (MLR $p < 2.2 \times 10^{-16}$). Resort occupancy surges past 1,270 room-nights/week.
- **Possible Action:** Implement dynamic minimum rate floors for June–August arrivals. Discontinue deep wholesale OTA discounting during high summer; shift inventory to Direct channels.
- **Limitation:** The dataset lacks competitor rate data and unconstrained demand counts (days turned away due to 100% capacity are unobserved).

### Recommendation 2: Tiered Lead-Time Cancellation Protections
- **Finding:** Long booking horizons are the single strongest predictor of cancellation.
- **Evidence:** Each 30 days of lead time increases cancellation odds by +17.1% (OR = 1.0053/day, OR_30 = 1.1706; distinguishing relative odds from probability). Group bookings carry 3.0x higher cancellation odds.
- **Possible Action:** Introduce a tiered cancellation window: bookings made $>60$ days out should receive automated re-confirmation nudges and modest prepaid non-refundable discounts (e.g. 10% off for non-refundable commitment).
- **Limitation:** Causal relationship is unproven; forced non-refundable policies on price-sensitive retail segments could deter booking placement entirely.

### Recommendation 3: Proactive Guest Engagement & Commitment Anchoring
- **Finding:** Engaged guests who submit requests or travel with vehicles exhibit lower cancellation rates in historical data.
- **Evidence:** Each special request cuts cancellation odds by 59.0% (OR = 0.4099). Guests requesting parking had 0 cancellations in the sample, though the parking coefficient is statistically insignificant ($p = 0.362$) due to quasi-complete separation.
- **Possible Action:** Send automated digital pre-arrival concierge forms (bed configuration, dietary preferences, estimated arrival time) 14 days prior to arrival. Engaging the guest confirms arrival intent.
- **Limitation:** Submitting a request may be an indicator of existing commitment rather than an intervention that creates commitment.

### Recommendation 4: Forecast-Driven Flexible Labor Scheduling
- **Finding:** Simple Exponential Smoothing (SES) reliably forecasts weekly occupied room-nights with ~6.2% error.
- **Evidence:** Out-of-sample Test RMSE was 93.72 for Resort and 113.49 for City Hotel, translating to an average scale conversion of ~13–16 rooms/day.
- **Possible Action:** Proposed Heuristic Policy: Calibrate core permanent housekeeping and F&B culinary rosters to 90% of the SES forecast; utilize variable-hour agency staff to flex upward when realized occupancy tracks higher, validating against local labor costs.
- **Limitation:** The model forecasts total room-nights, not daily checkout turnover volume (which drives departure deep-cleaning workloads).

---

## 6. Comprehensive Limitations & Methodological Cautions

1. **Observational Data & Non-Causality:** All regression and classification models demonstrate statistical associations, not causal mechanisms. Interventions (such as altering deposit rules or raising prices) may trigger unobserved behavioral shifts not captured in historical passive data.
2. **Geographic & Institutional Narrowness:** The dataset originates from two specific hotels in Portugal (an urban hotel in Lisbon and a resort in the Algarve). Results cannot be generalized globally without local recalibration.
3. **Absence of Cost & Profitability Accounting:** The data records gross room revenue (ADR) without deducting travel agent commissions (often 15–25% for OTAs), merchant processing fees, food preparation costs, or cancellation forfeiture fee collections.
4. **Data Timing & Leakage Traps:** Operational variables (`reservation_status`, `assigned_room_type`, `room_type_mismatch`, `booking_changes`) must be strictly audited in prospective deployments to avoid temporal leakage.
5. **Short Historical Time Span:** The dataset spans 26 calendar months (arrivals July 1, 2015 – August 31, 2017; forecasting window July 6, 2015 – August 27, 2017). A single prior seasonal cycle is statistically insufficient to confirm stable multi-year seasonality or permanent secular trends.

---
*Document compiled automatically from project analytical outputs. All cited values are verifiable in `outputs/tables/` and `outputs/plots/`.*
