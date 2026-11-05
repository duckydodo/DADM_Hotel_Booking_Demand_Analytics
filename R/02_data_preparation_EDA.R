# ==============================================================================
# Script: 02_data_preparation_EDA.R
# Project: Hotel Booking Demand Analytics (DADM Group Project)
# Purpose: Clean the raw dataset, create defensible derived variables, 
#          perform comprehensive Exploratory Data Analysis (EDA),
#          export publication-ready diagnostic plots and summary tables,
#          and save the cleaned dataset for downstream analytics.
# Author: DADM Analytics Team
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Environment Setup & Data Loading
# ------------------------------------------------------------------------------
rm(list = ls())

# Define input path (from data/raw/ as per specification)
raw_data_file <- "data/raw/hotel_bookings.csv"
if (!file.exists(raw_data_file)) {
  if (file.exists("hotel_bookings.csv")) {
    raw_data_file <- "hotel_bookings.csv"
  } else {
    stop("Error: 'hotel_bookings.csv' not found in data/raw/ or project root.")
  }
}

# Ensure destination directories exist
dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs/plots", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs/tables", showWarnings = FALSE, recursive = TRUE)

cat("Loading raw dataset from:", raw_data_file, "...\n")
raw_df <- read.csv(raw_data_file, stringsAsFactors = FALSE)
initial_n <- nrow(raw_df)
cat(sprintf("Loaded %s observations across %d variables.\n\n", 
            format(initial_n, big.mark = ","), ncol(raw_df)))

# ------------------------------------------------------------------------------
# 2. Documented & Defensible Data Cleaning
# ------------------------------------------------------------------------------
cat("====================================================================\n")
cat("STEP 1: DOCUMENTED DATA CLEANING & QUALITY AUDIT\n")
cat("====================================================================\n")

# A. Missing Value Recoding & Imputation
# -------------------------------------
# 1. 'children': 4 NA values.
#    Rationale: Median and mode are 0. In these 4 bookings, adults >= 1. 
#    Imputing 0 children is logically sound and preserves the observations.
na_children_count <- sum(is.na(raw_df$children))
raw_df$children[is.na(raw_df$children)] <- 0

# 2. 'country': 488 string "NULL" entries.
#    Rationale: Country was unrecorded by the booking channel/reception. 
#    Recoding to "Unknown" preserves the row without misrepresenting nationality.
null_country_count <- sum(raw_df$country == "NULL")
raw_df$country[raw_df$country == "NULL"] <- "Unknown"

# 3. 'agent': 16,340 string "NULL" entries.
#    Rationale: Missing agent ID means the booking was made directly by the guest 
#    without travel agency intermediation. Recoding to "No Agent".
null_agent_count <- sum(raw_df$agent == "NULL")
raw_df$agent[raw_df$agent == "NULL"] <- "No Agent"

# 4. 'company': 112,593 string "NULL" entries (94.3% of records).
#    Rationale: Indicates non-corporate, individual consumer reservations.
#    Recoding to "No Company".
null_company_count <- sum(raw_df$company == "NULL")
raw_df$company[raw_df$company == "NULL"] <- "No Company"

# 5. 'meal': 1,169 "Undefined" entries.
#    Rationale: Per dataset authors (Antonio et al., 2019), "Undefined" maps to "SC" 
#    (Self-Catering: room only, no meals included).
undefined_meal_count <- sum(raw_df$meal == "Undefined")
raw_df$meal[raw_df$meal == "Undefined"] <- "SC"

# 6. 'market_segment' and 'distribution_channel': 2 and 5 "Undefined" entries.
raw_df$market_segment[raw_df$market_segment == "Undefined"] <- "Unknown"
raw_df$distribution_channel[raw_df$distribution_channel == "Undefined"] <- "Unknown"

cat("Missing values and placeholder categories successfully recoded.\n")

# B. Identification of Defensible Removal Criteria
# -------------------------------------------------
# Condition 1: Phantom Bookings (0 adults, 0 children, 0 babies)
# Rationale: A reservation with 0 total guests is physically impossible and 
# represents a database glitch or abandoned reservation attempt.
cond_zero_guests <- (raw_df$adults + raw_df$children + raw_df$babies) == 0

# Condition 2: Zero-Night Stays (0 weekend nights AND 0 week nights)
# Rationale: The core scope of hotel demand analytics is overnight accommodation.
# Same-day check-in/check-out (day use) or zero-duration aborted stays distort stay
# duration metrics and daily rate calculations.
cond_zero_nights <- (raw_df$stays_in_weekend_nights + raw_df$stays_in_week_nights) == 0

# Condition 3: Negative Average Daily Rate (ADR < 0)
# Rationale: ADR cannot be negative in real commerce. ADR = -6.38 is an accounting 
# credit/refund adjustment entry, not a genuine daily rate.
cond_neg_adr <- raw_df$adr < 0

# Condition 4: Extreme ADR Outlier (ADR > 5000)
# Rationale: A single entry in City Hotel room type A has ADR = 5,400.00 while the 
# median is 94.58 and the 99th percentile is 252.00. This is an undeniable data entry 
# typo that would severely distort linear models and variance estimators.
cond_outlier_adr <- raw_df$adr > 5000

# Combined removal filter
rows_to_remove <- cond_zero_guests | cond_zero_nights | cond_neg_adr | cond_outlier_adr

# Build Cleaning Audit Table
cleaning_audit <- data.frame(
  Filter_Reason = c(
    "Zero Total Guests (Phantom Bookings: adults+children+babies == 0)",
    "Zero Overnight Stays (stays_in_weekend + stays_in_week == 0)",
    "Negative ADR (Accounting adjustment / error: ADR < 0)",
    "Extreme ADR Outlier (Typo: ADR == 5400 in standard room)",
    "Overlap between criteria (multi-condition matches)",
    "TOTAL UNIQUE ROWS EXCLUDED"
  ),
  Affected_Count = c(
    sum(cond_zero_guests),
    sum(cond_zero_nights),
    sum(cond_neg_adr),
    sum(cond_outlier_adr),
    sum(cond_zero_guests & cond_zero_nights),
    sum(rows_to_remove)
  ),
  Percent_of_Raw = round(c(
    sum(cond_zero_guests) / initial_n * 100,
    sum(cond_zero_nights) / initial_n * 100,
    sum(cond_neg_adr) / initial_n * 100,
    sum(cond_outlier_adr) / initial_n * 100,
    sum(cond_zero_guests & cond_zero_nights) / initial_n * 100,
    sum(rows_to_remove) / initial_n * 100
  ), 3),
  stringsAsFactors = FALSE
)

write.csv(cleaning_audit, "outputs/tables/cleaning_audit_summary.csv", row.names = FALSE)
cat("\nCleaning Audit Summary:\n")
print(cleaning_audit)

# Apply removal
cleaned_df <- raw_df[!rows_to_remove, ]
cat(sprintf("\nRetained %s observations (%.2f%% of raw data).\n\n",
            format(nrow(cleaned_df), big.mark = ","),
            (nrow(cleaned_df) / initial_n) * 100))

# C. Zero ADR Investigation (Retained with Documentation)
# -------------------------------------------------------
# ADR == 0 occurs in 1,939 cleaned rows (1.64%).
# Investigation: 672 are explicitly in 'Complementary' market segment; 
# others are promotional stays, corporate comp rooms, or tour guide allocations.
# Business Rationale: These are real, authorized hotel stays (free rewards/comps) 
# and not data entry corruption. They are retained, but flagged with a derived column.
zero_adr_count <- sum(cleaned_df$adr == 0)
cat(sprintf("Zero ADR observations retained: %d (%.2f%% of cleaned data) - flagged as complimentary/zero-rate.\n",
            zero_adr_count, (zero_adr_count / nrow(cleaned_df)) * 100))

# D. Duplicate Records Identification & Flagging
# ----------------------------------------------
# In hotel property management systems (PMS), exact duplicate rows can occur when:
# (a) A family or organizer books multiple identical rooms on the same date with identical specs.
# (b) Travel agency batch imports create multiple identical entries.
# Rather than arbitrarily dropping ~32k rows (which would eliminate legitimate multi-room demand),
# we explicitly IDENTIFY and FLAG exact duplicates using an 'is_duplicate' logical column.
# This allows researchers to evaluate models both with and without duplicates.
cleaned_df$is_duplicate <- duplicated(cleaned_df)
dup_count <- sum(cleaned_df$is_duplicate)
cat(sprintf("Identified %s exact duplicate rows (%.2f%% of cleaned data) - flagged via 'is_duplicate'.\n\n",
            format(dup_count, big.mark = ","), (dup_count / nrow(cleaned_df)) * 100))


# ------------------------------------------------------------------------------
# 3. Creation of Defensible Derived Variables
# ------------------------------------------------------------------------------
cat("====================================================================\n")
cat("STEP 2: FEATURE ENGINEERING & DERIVED VARIABLES\n")
cat("====================================================================\n")

# 1. total_nights = stays_in_weekend_nights + stays_in_week_nights
cleaned_df$total_nights <- cleaned_df$stays_in_weekend_nights + cleaned_df$stays_in_week_nights

# 2. arrival_date constructed correctly from year, month, and day-of-month
# Month names are standardized and parsed via as.Date
month_str <- match(cleaned_df$arrival_date_month, month.name)
cleaned_df$arrival_date <- as.Date(
  paste(cleaned_df$arrival_date_year, sprintf("%02d", month_str), 
        sprintf("%02d", cleaned_df$arrival_date_day_of_month), sep = "-")
)

# 3. Factor representations for categorical clarity
cleaned_df$is_canceled_factor <- factor(cleaned_df$is_canceled, 
                                        levels = c(0, 1), 
                                        labels = c("Not Canceled", "Canceled"))
cleaned_df$hotel_factor <- factor(cleaned_df$hotel)

# 4. total_guests
cleaned_df$total_guests <- cleaned_df$adults + cleaned_df$children + cleaned_df$babies

# 5. Room assignment mismatch indicator (Operational upgrade or reassignment)
# 1 if assigned room != reserved room, 0 otherwise
cleaned_df$room_type_mismatch <- as.integer(cleaned_df$reserved_room_type != cleaned_df$assigned_room_type)

# 6. Lead time brackets for intuitive segmentation
cleaned_df$lead_time_bracket <- cut(
  cleaned_df$lead_time,
  breaks = c(-1, 7, 30, 90, 180, 365, Inf),
  labels = c("0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", ">365 days")
)

# 7. Channel indicators
cleaned_df$has_agent   <- as.integer(cleaned_df$agent != "No Agent")
cleaned_df$has_company <- as.integer(cleaned_df$company != "No Company")
cleaned_df$is_zero_adr <- as.integer(cleaned_df$adr == 0)

cat("Derived variables created successfully:\n")
cat(" - total_nights (Stay duration in nights)\n")
cat(" - arrival_date (ISO Date object: YYYY-MM-DD)\n")
cat(" - total_guests (adults + children + babies)\n")
cat(" - room_type_mismatch (Binary indicator: reserved != assigned)\n")
cat(" - lead_time_bracket (Categorical booking horizon)\n")
cat(" - is_canceled_factor & hotel_factor (Clean factors for modeling/plotting)\n")
cat(" - has_agent, has_company, is_zero_adr (Binary flags)\n\n")


# ------------------------------------------------------------------------------
# 4. Exploratory Data Analysis & Summary Tables
# ------------------------------------------------------------------------------
cat("====================================================================\n")
cat("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)\n")
cat("====================================================================\n")

# A. Cancellation Rates Overall & by Hotel Type
# ---------------------------------------------
hotel_stats <- aggregate(is_canceled ~ hotel, data = cleaned_df, FUN = function(x) {
  c(Total = length(x), 
    Canceled = sum(x), 
    Rate = mean(x) * 100)
})
hotel_summary_tbl <- data.frame(
  Hotel = hotel_stats$hotel,
  Total_Bookings = hotel_stats$is_canceled[, "Total"],
  Canceled_Bookings = hotel_stats$is_canceled[, "Canceled"],
  Cancellation_Rate_Pct = round(hotel_stats$is_canceled[, "Rate"], 2),
  Mean_ADR = round(tapply(cleaned_df$adr, cleaned_df$hotel, mean), 2),
  Median_ADR = round(tapply(cleaned_df$adr, cleaned_df$hotel, median), 2)
)
write.csv(hotel_summary_tbl, "outputs/tables/cancellation_by_hotel_type.csv", row.names = FALSE)
cat("Cancellation Summary by Hotel Type:\n")
print(hotel_summary_tbl)

# B. Monthly ADR & Arrivals Analysis
# ----------------------------------
month_levels <- c("January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December")
cleaned_df$arrival_date_month <- factor(cleaned_df$arrival_date_month, levels = month_levels)

monthly_summary <- aggregate(
  cbind(is_canceled, adr) ~ arrival_date_month + hotel, 
  data = cleaned_df, 
  FUN = function(x) c(Count = length(x), Mean = mean(x))
)
monthly_summary_df <- data.frame(
  Month = monthly_summary$arrival_date_month,
  Hotel = monthly_summary$hotel,
  Total_Arrivals = monthly_summary$is_canceled[, "Count"],
  Cancellation_Rate_Pct = round(monthly_summary$is_canceled[, "Mean"] * 100, 2),
  Average_ADR = round(monthly_summary$adr[, "Mean"], 2)
)
monthly_summary_df <- monthly_summary_df[order(monthly_summary_df$Hotel, match(monthly_summary_df$Month, month_levels)), ]
write.csv(monthly_summary_df, "outputs/tables/monthly_adr_and_arrivals.csv", row.names = FALSE)

# C. Cancellation Rate by Market Segment
# --------------------------------------
market_stats <- aggregate(is_canceled ~ market_segment, data = cleaned_df, FUN = function(x) {
  c(Total = length(x), Canceled = sum(x), Rate = mean(x) * 100)
})
market_tbl <- data.frame(
  Market_Segment = market_stats$market_segment,
  Total_Bookings = market_stats$is_canceled[, "Total"],
  Canceled_Count = market_stats$is_canceled[, "Canceled"],
  Cancellation_Rate_Pct = round(market_stats$is_canceled[, "Rate"], 2)
)
market_tbl <- market_tbl[order(-market_tbl$Total_Bookings), ]
write.csv(market_tbl, "outputs/tables/cancellation_by_market_segment.csv", row.names = FALSE)

# D. Cancellation Rate by Deposit Type
# ------------------------------------
deposit_stats <- aggregate(is_canceled ~ deposit_type, data = cleaned_df, FUN = function(x) {
  c(Total = length(x), Canceled = sum(x), Rate = mean(x) * 100)
})
deposit_tbl <- data.frame(
  Deposit_Type = deposit_stats$deposit_type,
  Total_Bookings = deposit_stats$is_canceled[, "Total"],
  Canceled_Count = deposit_stats$is_canceled[, "Canceled"],
  Cancellation_Rate_Pct = round(deposit_stats$is_canceled[, "Rate"], 2)
)
write.csv(deposit_tbl, "outputs/tables/cancellation_by_deposit_type.csv", row.names = FALSE)

# E. Cancellation Rate by Lead Time Bracket
# -----------------------------------------
lead_stats <- aggregate(is_canceled ~ lead_time_bracket, data = cleaned_df, FUN = function(x) {
  c(Total = length(x), Canceled = sum(x), Rate = mean(x) * 100)
})
lead_tbl <- data.frame(
  Lead_Time_Bracket = lead_stats$lead_time_bracket,
  Total_Bookings = lead_stats$is_canceled[, "Total"],
  Canceled_Count = lead_stats$is_canceled[, "Canceled"],
  Cancellation_Rate_Pct = round(lead_stats$is_canceled[, "Rate"], 2)
)
write.csv(lead_tbl, "outputs/tables/cancellation_by_lead_time_bracket.csv", row.names = FALSE)

# F. Cancellation Rate by Customer Type
# -------------------------------------
cust_stats <- aggregate(is_canceled ~ customer_type, data = cleaned_df, FUN = function(x) {
  c(Total = length(x), Canceled = sum(x), Rate = mean(x) * 100)
})
cust_tbl <- data.frame(
  Customer_Type = cust_stats$customer_type,
  Total_Bookings = cust_stats$is_canceled[, "Total"],
  Canceled_Count = cust_stats$is_canceled[, "Canceled"],
  Cancellation_Rate_Pct = round(cust_stats$is_canceled[, "Rate"], 2)
)
write.csv(cust_tbl, "outputs/tables/cancellation_by_customer_type.csv", row.names = FALSE)

cat("EDA summary tables saved to outputs/tables/.\n\n")


# ------------------------------------------------------------------------------
# 5. Production of Purpose-Driven Visualizations (Base R)
# ------------------------------------------------------------------------------
cat("====================================================================\n")
cat("STEP 4: GENERATING PUBLICATION-READY EDA VISUALIZATIONS\n")
cat("====================================================================\n")

# Plot 1: Cancellation Rates Overall and by Hotel Type
# Business Question: How severe is cancellation risk between property types?
png("outputs/plots/eda_01_cancellation_by_hotel.png", width = 1200, height = 700, res = 130)
par(mar = c(5, 5, 4, 2))
rates <- c(
  "Overall\n(All Bookings)" = mean(cleaned_df$is_canceled) * 100,
  "City Hotel\n(Urban Transient)" = hotel_summary_tbl$Cancellation_Rate_Pct[hotel_summary_tbl$Hotel == "City Hotel"],
  "Resort Hotel\n(Leisure & Resort)" = hotel_summary_tbl$Cancellation_Rate_Pct[hotel_summary_tbl$Hotel == "Resort Hotel"]
)
bp1 <- barplot(rates, col = c("#2B5C8F", "#D95F02", "#1B9E77"),
               ylim = c(0, 50), ylab = "Cancellation Rate (%)",
               main = "Hotel Booking Cancellation Rates by Property Type",
               font.main = 2, cex.names = 1.0, cex.lab = 1.1)
grid(nx = NA, ny = NULL, col = "gray85", lty = "dotted")
bp1 <- barplot(rates, col = c("#2B5C8F", "#D95F02", "#1B9E77"), add = TRUE)
text(bp1, rates + 2, labels = paste0(round(rates, 1), "%"), font = 2, cex = 1.1)
dev.off()
cat(" - Saved: outputs/plots/eda_01_cancellation_by_hotel.png\n")

# Plot 2: ADR Distribution by Hotel Type
# Business Question: What does the pricing structure look like across property types?
png("outputs/plots/eda_02_adr_distribution_by_hotel.png", width = 1200, height = 700, res = 130)
par(mar = c(5, 5, 4, 2))
# Filter ADR <= 350 for clear visualization of the core 99% distribution
boxplot(adr ~ hotel, data = cleaned_df[cleaned_df$adr <= 350, ],
        col = c("#FDBF6F", "#A6CEE3"),
        ylab = "Average Daily Rate (EUR / $)",
        xlab = "Hotel Type",
        main = "Average Daily Rate (ADR) Distribution by Hotel Type (ADR <= $350)",
        notch = TRUE, outline = TRUE, pch = 20, col.symbol = "gray50")
grid(nx = NA, ny = NULL, col = "gray85", lty = "dotted")
dev.off()
cat(" - Saved: outputs/plots/eda_02_adr_distribution_by_hotel.png\n")

# Plot 3: ADR Seasonality by Month and Hotel Type
# Business Question: When do properties experience pricing power peaks?
resort_monthly <- monthly_summary_df[monthly_summary_df$Hotel == "Resort Hotel", ]
city_monthly   <- monthly_summary_df[monthly_summary_df$Hotel == "City Hotel", ]

png("outputs/plots/eda_03_adr_seasonality_by_month.png", width = 1300, height = 750, res = 130)
par(mar = c(6, 5, 4, 2))
plot(1:12, resort_monthly$Average_ADR, type = "b", pch = 19, col = "#1B9E77", lwd = 3,
     ylim = c(40, 200), xaxt = "n", xlab = "", ylab = "Mean ADR ($ / EUR)",
     main = "Seasonal Pricing Trajectory: Monthly Mean ADR by Hotel Type")
axis(1, at = 1:12, labels = substr(month_levels, 1, 3), las = 2, font = 2)
lines(1:12, city_monthly$Average_ADR, type = "b", pch = 15, col = "#D95F02", lwd = 3)
grid(col = "gray85", lty = "dotted")
legend("topleft", legend = c("Resort Hotel (Peak Summer July-Aug)", "City Hotel (Stable Spring/Autumn)"),
       col = c("#1B9E77", "#D95F02"), pch = c(19, 15), lwd = 3, bty = "o", bg = "white")
dev.off()
cat(" - Saved: outputs/plots/eda_03_adr_seasonality_by_month.png\n")

# Plot 4: Cancellation Rates by Key Booking Dimensions (2x2 Panel)
# Business Question: What booking attributes present the highest cancellation risk?
png("outputs/plots/eda_04_cancellation_by_booking_factors.png", width = 1400, height = 1100, res = 130)
par(mfrow = c(2, 2), mar = c(6, 5, 3.5, 2))

# Subplot 1: Deposit Type (The Non-Refundable Deposit Paradox)
dep_order <- order(-deposit_tbl$Cancellation_Rate_Pct)
bp_dep <- barplot(deposit_tbl$Cancellation_Rate_Pct[dep_order], 
                  names.arg = deposit_tbl$Deposit_Type[dep_order],
                  col = c("#E41A1C", "#377EB8", "#4DAF4A"), ylim = c(0, 110),
                  ylab = "Cancellation Rate (%)", main = "Cancellation by Deposit Type",
                  font.main = 2, las = 1)
text(bp_dep, deposit_tbl$Cancellation_Rate_Pct[dep_order] + 5, 
     labels = paste0(round(deposit_tbl$Cancellation_Rate_Pct[dep_order], 1), "%"), font = 2)

# Subplot 2: Market Segment
mkt_order <- order(-market_tbl$Cancellation_Rate_Pct)
bp_mkt <- barplot(market_tbl$Cancellation_Rate_Pct[mkt_order], 
                  names.arg = market_tbl$Market_Segment[mkt_order],
                  col = "#4A7BB0", ylim = c(0, 75), las = 2,
                  ylab = "Cancellation Rate (%)", main = "Cancellation by Market Segment",
                  font.main = 2, cex.names = 0.85)
text(bp_mkt, market_tbl$Cancellation_Rate_Pct[mkt_order] + 3,
     labels = paste0(round(market_tbl$Cancellation_Rate_Pct[mkt_order], 1), "%"), font = 2, cex = 0.8)

# Subplot 3: Customer Type
cust_order <- order(-cust_tbl$Cancellation_Rate_Pct)
bp_cust <- barplot(cust_tbl$Cancellation_Rate_Pct[cust_order],
                   names.arg = cust_tbl$Customer_Type[cust_order],
                   col = "#8DA0CB", ylim = c(0, 50), las = 2,
                   ylab = "Cancellation Rate (%)", main = "Cancellation by Customer Type",
                   font.main = 2, cex.names = 0.9)
text(bp_cust, cust_tbl$Cancellation_Rate_Pct[cust_order] + 2,
     labels = paste0(round(cust_tbl$Cancellation_Rate_Pct[cust_order], 1), "%"), font = 2)

# Subplot 4: Lead Time Bracket
lead_order <- 1:nrow(lead_tbl)
bp_lead <- barplot(lead_tbl$Cancellation_Rate_Pct[lead_order],
                   names.arg = lead_tbl$Lead_Time_Bracket[lead_order],
                   col = "#FC8D62", ylim = c(0, 75), las = 2,
                   ylab = "Cancellation Rate (%)", main = "Cancellation by Lead Time Horizon",
                   font.main = 2, cex.names = 0.85)
text(bp_lead, lead_tbl$Cancellation_Rate_Pct[lead_order] + 3,
     labels = paste0(round(lead_tbl$Cancellation_Rate_Pct[lead_order], 1), "%"), font = 2, cex = 0.85)

dev.off()
cat(" - Saved: outputs/plots/eda_04_cancellation_by_booking_factors.png\n")

# Plot 5: Stay Length Distributions by Hotel Type
# Business Question: How do typical stay lengths differ between resort and city properties?
png("outputs/plots/eda_05_stay_length_distribution.png", width = 1200, height = 700, res = 130)
par(mar = c(5, 5, 4, 2))
# Filter stays between 1 and 14 nights for actionable operational focus
stays_sub <- cleaned_df[cleaned_df$total_nights >= 1 & cleaned_df$total_nights <= 14, ]
stay_table <- prop.table(table(stays_sub$total_nights, stays_sub$hotel), 2) * 100

bp_stay <- barplot(t(stay_table), beside = TRUE, col = c("#386CB0", "#7FC97F"),
                   xlab = "Length of Stay (Total Nights: Week + Weekend)",
                   ylab = "Percentage of Stays within Hotel Type (%)",
                   main = "Stay Length Distribution (1 to 14 Nights) by Hotel Type",
                   ylim = c(0, 45))
grid(nx = NA, ny = NULL, col = "gray85", lty = "dotted")
legend("topright", legend = c("City Hotel (Dominated by 1-3 Nights)", "Resort Hotel (Longer stays with 7-night peak)"),
       fill = c("#386CB0", "#7FC97F"), bty = "o", bg = "white")
dev.off()
cat(" - Saved: outputs/plots/eda_05_stay_length_distribution.png\n")

# Plot 6: Seasonal Patterns in Monthly Arrivals
# Business Question: How does arrival volume fluctuate month-by-month over the calendar year?
arrivals_by_month <- table(cleaned_df$arrival_date_month, cleaned_df$hotel)
png("outputs/plots/eda_06_arrival_seasonality.png", width = 1300, height = 700, res = 130)
par(mar = c(6, 5, 4, 2))
bp_arr <- barplot(t(arrivals_by_month), beside = TRUE, col = c("#2B5C8F", "#E6AB02"),
                  ylab = "Total Arrival Bookings", ylim = c(0, 10000),
                  main = "Monthly Arrival Volume by Hotel Type (Calendar Seasonality)",
                  las = 2, cex.names = 0.95)
grid(nx = NA, ny = NULL, col = "gray85", lty = "dotted")
legend("topleft", legend = c("City Hotel", "Resort Hotel"),
       fill = c("#2B5C8F", "#E6AB02"), bty = "o", bg = "white")
dev.off()
cat(" - Saved: outputs/plots/eda_06_arrival_seasonality.png\n\n")


# ------------------------------------------------------------------------------
# 6. Save Cleaned Dataset
# ------------------------------------------------------------------------------
cat("====================================================================\n")
cat("STEP 5: SAVING CLEANED DATASET\n")
cat("====================================================================\n")

output_csv <- "data/processed/hotel_bookings_cleaned.csv"
cat("Writing cleaned dataset to:", output_csv, "...\n")
write.csv(cleaned_df, output_csv, row.names = FALSE)
cat(sprintf("Successfully written %s rows and %d columns to %s.\n\n",
            format(nrow(cleaned_df), big.mark = ","), ncol(cleaned_df), output_csv))

cat("All processing and exploratory analysis completed successfully!\n")
