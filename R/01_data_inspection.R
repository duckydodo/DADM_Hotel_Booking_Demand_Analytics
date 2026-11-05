# ==============================================================================
# Script: 01_data_inspection.R
# Project: Hotel Booking Demand Analytics (DADM Group Project)
# Purpose: Comprehensive data inspection, sanity checks, missing value audit,
#          duplicate detection, and univariate/bivariate checks using Base R.
# Author: DADM Analytics Team
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Configuration & Setup
# ------------------------------------------------------------------------------
# Clear workspace for a clean run environment
rm(list = ls())

# Define file paths
# Note: Checks both data/raw/ and root directory to ensure portability
raw_data_path <- if (file.exists("data/raw/hotel_bookings.csv")) {
  "data/raw/hotel_bookings.csv"
} else if (file.exists("hotel_bookings.csv")) {
  "hotel_bookings.csv"
} else {
  stop("Error: 'hotel_bookings.csv' was not found in data/raw/ or project root.")
}

# Ensure output directories exist
dir.create("outputs", showWarnings = FALSE)
dir.create("outputs/tables", showWarnings = FALSE)
dir.create("outputs/plots", showWarnings = FALSE)

summary_output_file <- "outputs/data_inspection_summary.txt"

# ------------------------------------------------------------------------------
# 2. Data Import
# ------------------------------------------------------------------------------
# read.csv() with stringsAsFactors = FALSE ensures character strings are preserved.
# We do not modify the raw data file at all (Rule 7).
cat("Loading raw dataset from:", raw_data_path, "...\n")
hotel_df <- read.csv(raw_data_path, stringsAsFactors = FALSE)
cat("Data successfully loaded.\n\n")

# Open a connection to log output both to the console and to a summary file
log_con <- file(summary_output_file, open = "wt")
# Helper function to print to both console and log file
log_print <- function(...) {
  msg <- paste0(...)
  cat(msg)
  cat(msg, file = log_con)
}

log_print("================================================================================\n")
log_print("         HOTEL BOOKING DEMAND ANALYTICS - DATA INSPECTION REPORT         \n")
log_print("================================================================================\n\n")

# ------------------------------------------------------------------------------
# 3. Dataset Dimensions and Structure
# ------------------------------------------------------------------------------
n_rows <- nrow(hotel_df)
n_cols <- ncol(hotel_df)

log_print("--- 1. BASIC DIMENSIONS & STRUCTURE ---\n")
log_print("Total Observations (Rows): ", format(n_rows, big.mark = ","), "\n")
log_print("Total Features (Columns) : ", n_cols, "\n\n")

log_print("Column Names and Data Types:\n")
col_summary <- data.frame(
  Column_Name = names(hotel_df),
  Data_Type   = sapply(hotel_df, typeof),
  stringsAsFactors = FALSE
)
# Write structure table to outputs/tables
write.csv(col_summary, "outputs/tables/column_structure.csv", row.names = FALSE)

# Display compact summary
for (i in seq_along(hotel_df)) {
  col_name <- names(hotel_df)[i]
  col_type <- typeof(hotel_df[[i]])
  col_class <- class(hotel_df[[i]])
  log_print(sprintf("  [%02d] %-30s : %-10s (class: %s)\n", i, col_name, col_type, col_class))
}
log_print("\n")

# Display first 5 rows (sample preview)
log_print("Sample Rows (First 5 observations):\n")
print_head <- capture.output(print(head(hotel_df, 5)))
log_print(paste(print_head, collapse = "\n"), "\n\n")

# ------------------------------------------------------------------------------
# 4. Missing Values Audit (Explicit NA and Implicit 'NULL' Strings)
# ------------------------------------------------------------------------------
log_print("--- 2. MISSING VALUE AUDIT ---\n")

# Standard R NA counts
na_counts <- colSums(is.na(hotel_df))

# The dataset uses the literal string "NULL" for missing values in several columns
null_string_counts <- sapply(hotel_df, function(col) {
  if (is.character(col)) sum(trimws(col) == "NULL", na.rm = TRUE) else 0
})

# Empty string counts
empty_string_counts <- sapply(hotel_df, function(col) {
  if (is.character(col)) sum(trimws(col) == "", na.rm = TRUE) else 0
})

total_missing <- na_counts + null_string_counts + empty_string_counts
pct_missing <- round((total_missing / n_rows) * 100, 3)

missing_df <- data.frame(
  Column_Name       = names(hotel_df),
  Explicit_NA       = na_counts,
  String_NULL       = null_string_counts,
  Empty_Strings     = empty_string_counts,
  Total_Missing     = total_missing,
  Percent_Missing   = pct_missing,
  stringsAsFactors  = FALSE
)

# Filter to columns that actually have missing values
missing_columns <- missing_df[missing_df$Total_Missing > 0, ]
missing_columns <- missing_columns[order(-missing_columns$Total_Missing), ]

write.csv(missing_df, "outputs/tables/missing_values_all_columns.csv", row.names = FALSE)

log_print("Columns with Missing/Unrecorded Data:\n")
missing_table_print <- capture.output(print(missing_columns, row.names = FALSE))
log_print(paste(missing_table_print, collapse = "\n"), "\n\n")

log_print("Key Missing Data Insights:\n")
log_print(" - 'company' has ", format(missing_columns$Total_Missing[missing_columns$Column_Name == "company"], big.mark = ","),
          " missing values (", missing_columns$Percent_Missing[missing_columns$Column_Name == "company"], "% missing).\n")
log_print(" - 'agent' has ", format(missing_columns$Total_Missing[missing_columns$Column_Name == "agent"], big.mark = ","),
          " missing values (", missing_columns$Percent_Missing[missing_columns$Column_Name == "agent"], "% missing, represents non-agency/direct bookings).\n")
log_print(" - 'country' has ", format(missing_columns$Total_Missing[missing_columns$Column_Name == "country"], big.mark = ","),
          " missing values (", missing_columns$Percent_Missing[missing_columns$Column_Name == "country"], "% missing).\n")
log_print(" - 'children' has ", format(missing_columns$Total_Missing[missing_columns$Column_Name == "children"], big.mark = ","),
          " explicit NA values (0.003% missing).\n\n")

# ------------------------------------------------------------------------------
# 5. Exact Duplicate Rows Detection
# ------------------------------------------------------------------------------
log_print("--- 3. DUPLICATE ROWS DETECTION ---\n")
num_duplicates <- sum(duplicated(hotel_df))
pct_duplicates <- round((num_duplicates / n_rows) * 100, 2)

log_print("Exact Duplicate Rows: ", format(num_duplicates, big.mark = ","), " (", pct_duplicates, "% of total dataset)\n")
log_print("Note on Duplicates: In hotel reservation systems, identical rows can arise from:\n")
log_print(" (a) Technical duplication/re-entry during database collation, OR\n")
log_print(" (b) Multi-room bookings by families, event organizers, or tour groups sharing identical attributes.\n")
log_print(" Data cleaning strategy should carefully evaluate deduplication impact on class distribution.\n\n")

# ------------------------------------------------------------------------------
# 6. Target Variable Analysis: is_canceled
# ------------------------------------------------------------------------------
log_print("--- 4. TARGET VARIABLE: is_canceled ---\n")
cancel_freq <- table(hotel_df$is_canceled)
cancel_prop <- prop.table(cancel_freq) * 100

cancel_summary <- data.frame(
  Status     = c("Not Canceled (0)", "Canceled (1)"),
  Count      = as.vector(cancel_freq),
  Percentage = round(as.vector(cancel_prop), 2)
)
log_print(paste(capture.output(print(cancel_summary, row.names = FALSE)), collapse = "\n"), "\n\n")

# ------------------------------------------------------------------------------
# 7. Hotel Types Distribution and Cancellation Rates
# ------------------------------------------------------------------------------
log_print("--- 5. HOTEL TYPES DISTRIBUTION & CANCELLATIONS ---\n")
hotel_freq <- table(hotel_df$hotel)
hotel_prop <- prop.table(hotel_freq) * 100

hotel_cancel_ct <- table(hotel_df$hotel, hotel_df$is_canceled)
hotel_cancel_rates <- prop.table(hotel_cancel_ct, 1) * 100

hotel_summary <- data.frame(
  Hotel_Type          = names(hotel_freq),
  Total_Bookings      = as.vector(hotel_freq),
  Share_Percent       = round(as.vector(hotel_prop), 2),
  Not_Canceled_Count  = as.vector(hotel_cancel_ct[, "0"]),
  Canceled_Count      = as.vector(hotel_cancel_ct[, "1"]),
  Cancellation_Rate   = round(as.vector(hotel_cancel_rates[, "1"]), 2)
)
write.csv(hotel_summary, "outputs/tables/hotel_cancellation_rates.csv", row.names = FALSE)
log_print(paste(capture.output(print(hotel_summary, row.names = FALSE)), collapse = "\n"), "\n\n")

# ------------------------------------------------------------------------------
# 8. Suspicious Values & Outliers in Key Numeric Features
# ------------------------------------------------------------------------------
log_print("--- 6. SUSPICIOUS VALUES & OUTLIER AUDIT ---\n")

# Helper function for five-number summary + mean + sd
stat_summary <- function(x, name) {
  clean_x <- x[!is.na(x)]
  qs <- quantile(clean_x, probs = c(0, 0.25, 0.5, 0.75, 0.95, 0.99, 1))
  data.frame(
    Variable = name,
    Min      = round(qs[1], 2),
    Q25      = round(qs[2], 2),
    Median   = round(qs[3], 2),
    Mean     = round(mean(clean_x), 2),
    Q75      = round(qs[4], 2),
    P95      = round(qs[5], 2),
    P99      = round(qs[6], 2),
    Max      = round(qs[7], 2),
    SD       = round(sd(clean_x), 2),
    stringsAsFactors = FALSE
  )
}

numeric_vars <- c("adr", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
                  "adults", "children", "babies", "days_in_waiting_list",
                  "previous_cancellations", "booking_changes", "total_of_special_requests")

numeric_stats_list <- lapply(numeric_vars, function(v) stat_summary(hotel_df[[v]], v))
numeric_stats_df <- do.call(rbind, numeric_stats_list)
write.csv(numeric_stats_df, "outputs/tables/numeric_summary_statistics.csv", row.names = FALSE)

log_print("Summary Statistics of Key Numeric Variables:\n")
log_print(paste(capture.output(print(numeric_stats_df, row.names = FALSE)), collapse = "\n"), "\n\n")

# Specific Anomalies Detection
log_print("Specific Anomalies Identified:\n")

# (A) ADR Anomalies
adr_neg <- sum(hotel_df$adr < 0)
adr_zero <- sum(hotel_df$adr == 0)
adr_high <- sum(hotel_df$adr > 1000)
log_print(" [ADR]:\n")
log_print("   - Negative ADR count: ", adr_neg, " (Value: ", hotel_df$adr[hotel_df$adr < 0], ")\n")
log_print("   - Zero ADR count    : ", format(adr_zero, big.mark = ","), " (", round(adr_zero / n_rows * 100, 2), "% of total bookings - likely complimentary/promotional)\n")
log_print("   - ADR > 1000 count  : ", adr_high, " (Extreme outlier value: ", max(hotel_df$adr), ")\n")

# (B) Zero Guests (Adults + Children + Babies == 0)
zero_guests <- sum((hotel_df$adults + ifelse(is.na(hotel_df$children), 0, hotel_df$children) + hotel_df$babies) == 0)
log_print("\n [Guest Counts]:\n")
log_print("   - Bookings with 0 adults, 0 children, and 0 babies: ", zero_guests, " (Phantom bookings)\n")
log_print("   - Max adults in a single booking: ", max(hotel_df$adults), " (Group booking or data error)\n")
log_print("   - Max babies: ", max(hotel_df$babies), ", Max children: ", max(hotel_df$children, na.rm = TRUE), "\n")

# (C) Zero Stay Duration (Week nights + Weekend nights == 0)
zero_nights <- sum((hotel_df$stays_in_weekend_nights + hotel_df$stays_in_week_nights) == 0)
log_print("\n [Stay Duration]:\n")
log_print("   - Bookings with 0 weekend AND 0 week nights: ", zero_nights, " (Same-day check-in/out or aborted bookings)\n")
log_print("   - Max weekend nights: ", max(hotel_df$stays_in_weekend_nights), ", Max week nights: ", max(hotel_df$stays_in_week_nights), "\n")

# (D) Lead Time
lead_gt_365 <- sum(hotel_df$lead_time > 365)
log_print("\n [Lead Time]:\n")
log_print("   - Bookings with lead time > 1 year (> 365 days): ", format(lead_gt_365, big.mark = ","),
          " (", round(lead_gt_365 / n_rows * 100, 2), "%)\n")
log_print("   - Maximum lead time: ", max(hotel_df$lead_time), " days (~2.02 years)\n\n")

# ------------------------------------------------------------------------------
# 9. Categorical Sanity Checks
# ------------------------------------------------------------------------------
log_print("--- 7. CATEGORICAL ANOMALIES & DISTRIBUTION ---\n")

# Meal distribution
meal_tbl <- table(hotel_df$meal)
log_print("Meal Package Distribution:\n")
log_print(paste(capture.output(print(meal_tbl)), collapse = "\n"), "\n")
log_print("Note: 'Undefined' meal occurs ", meal_tbl["Undefined"], " times (maps to 'SC' / Self-Catering in dataset documentation).\n\n")

# Market Segment
market_tbl <- table(hotel_df$market_segment)
log_print("Market Segment Distribution:\n")
log_print(paste(capture.output(print(market_tbl)), collapse = "\n"), "\n")
log_print("Note: 'Undefined' market segment occurs ", sum(hotel_df$market_segment == "Undefined"), " times.\n\n")

# Distribution Channel
dist_tbl <- table(hotel_df$distribution_channel)
log_print("Distribution Channel Distribution:\n")
log_print(paste(capture.output(print(dist_tbl)), collapse = "\n"), "\n")
log_print("Note: 'Undefined' distribution channel occurs ", sum(hotel_df$distribution_channel == "Undefined"), " times.\n\n")

# Room type mismatch (Reserved vs Assigned)
room_mismatch <- sum(hotel_df$reserved_room_type != hotel_df$assigned_room_type)
log_print("Room Assignment Mismatches (Reserved != Assigned):\n")
log_print(" - Total occurrences: ", format(room_mismatch, big.mark = ","),
          " (", round(room_mismatch / n_rows * 100, 2), "% of all bookings)\n")
log_print("   (Indicates complimentary upgrades, operational room reallocation, or overbooking reassignments)\n\n")

# ------------------------------------------------------------------------------
# 10. Summary Conclusions and Next Steps
# ------------------------------------------------------------------------------
log_print("================================================================================\n")
log_print("SUMMARY OF DATA INSPECTION FOR DECISION MAKING:\n")
log_print("1. Data Volume: 119,390 rows and 32 variables.\n")
log_print("2. Missingness: 'company' (94.3% NULL), 'agent' (13.7% NULL), 'country' (0.4% NULL), 'children' (4 NAs).\n")
log_print("3. Duplicates: 31,994 duplicate rows (26.8%) require an explicit project-level handling decision.\n")
log_print("4. Outliers/Anomalies: ADR has a negative value (-6.38) and an extreme outlier (5,400.00).\n")
log_print("5. Logical Inconsistencies: 180 rows with 0 guests, 715 rows with 0 night stays, and undefined categories.\n")
log_print("6. Class Balance: 37.04% cancellation rate overall (City Hotel: 41.73%, Resort Hotel: 27.76%).\n")
log_print("================================================================================\n")

# ------------------------------------------------------------------------------
# 11. Base R Visual Inspection Plots (Diagnostic Visualizations)
# ------------------------------------------------------------------------------
cat("Generating visual inspection diagnostic plots with Base R...\n")

# Plot 1: Target variable and Hotel Type distributions
png("outputs/plots/01_target_and_hotel_distribution.png", width = 1000, height = 500, res = 120)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# Barplot for hotel types
barplot(hotel_prop, col = c("#4E79A7", "#F28E2B"),
        main = "Distribution of Hotel Types",
        ylab = "Percentage (%)", ylim = c(0, 100))

# Barplot for cancellation rate by hotel
barplot(hotel_cancel_rates[, "1"], col = c("#E15759", "#76B7B2"),
        main = "Cancellation Rate by Hotel Type",
        ylab = "Cancellation Rate (%)", ylim = c(0, 50))
dev.off()

# Plot 2: Boxplots for ADR and Lead Time (Diagnostic Outlier Check)
png("outputs/plots/02_numeric_distributions_outliers.png", width = 1000, height = 500, res = 120)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

boxplot(hotel_df$adr, col = "#59A14F",
        main = "ADR Distribution (Shows 5400 Outlier)",
        ylab = "Average Daily Rate ($)")

boxplot(hotel_df$lead_time, col = "#EDC948",
        main = "Lead Time Distribution",
        ylab = "Lead Time (Days)")
dev.off()
cat("Diagnostic plots saved to outputs/plots/\n")

# Close file connection
close(log_con)

cat("\n[SUCCESS] Data inspection completed. Output written to:\n")
cat(" - Summary text: ", summary_output_file, "\n")
cat(" - Diagnostic plots: outputs/plots/01_target_and_hotel_distribution.png, outputs/plots/02_numeric_distributions_outliers.png\n")
cat(" - Table: outputs/tables/column_structure.csv\n")
cat(" - Table: outputs/tables/missing_values_all_columns.csv\n")
cat(" - Table: outputs/tables/hotel_cancellation_rates.csv\n")
cat(" - Table: outputs/tables/numeric_summary_statistics.csv\n")
