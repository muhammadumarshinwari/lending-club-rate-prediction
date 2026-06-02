Predicting Loan Interest Rates — Regression Trees vs Linear Models
================

- [What This Project Is About](#what-this-project-is-about)
- [The Data](#the-data)
- [Train and Validation Split](#train-and-validation-split)
- [Model 1: Regression Tree](#model-1-regression-tree)
- [Model 2: Linear Regression with Best Subset
  Selection](#model-2-linear-regression-with-best-subset-selection)
- [Model 3: LASSO Regression](#model-3-lasso-regression)
- [Model Comparison](#model-comparison)
- [Key Takeaways](#key-takeaways)

## What This Project Is About

When someone applies for a personal loan on Lending Club, the platform
assigns them an interest rate. That rate is not random. It reflects the
platform’s assessment of how risky the borrower is. A borrower with a
high credit score, stable income, and a short loan term will get a lower
rate. Someone with past delinquencies, high debt relative to income, and
a longer loan term will pay more.

This project builds three models that each try to predict what interest
rate a borrower will receive, given information about their financial
background. The three models are a regression tree, a linear regression
with best subset selection, and a LASSO regression. The goal is to
compare how well each method predicts interest rates on borrowers the
model has never seen, and to understand what each model is actually
learning about who pays what.

The dataset comes from Lending Club, one of the largest peer-to-peer
lending platforms in the US. It contains 15,000 loan records with 15
features per borrower and a target variable: the assigned interest rate.

------------------------------------------------------------------------

## The Data

``` r
lend <- read.csv("Lending Club.csv")
cat("Rows:", nrow(lend), "\n")
```

    ## Rows: 15000

``` r
cat("Columns:", ncol(lend), "\n")
```

    ## Columns: 16

``` r
cat("\nColumn names:\n")
```

    ## 
    ## Column names:

``` r
print(names(lend))
```

    ##  [1] "Loan_amt"               "Loan_term"              "emp_length"            
    ##  [4] "Home"                   "Income_source_verified" "Income_verified"       
    ##  [7] "Income_thou"            "Debt_income"            "delinq_2yrs"           
    ## [10] "Credit_history_length"  "FICO"                   "open_acc"              
    ## [13] "Derogatory_recs"        "Revol_balance"          "Revol_util"            
    ## [16] "Loan_rate"

``` r
feature_df <- data.frame(
  Feature = c("Loan_amt", "Loan_term", "emp_length", "Home",
              "Income_source_verified", "Income_verified", "Income_thou",
              "Debt_income", "delinq_2yrs", "Credit_history_length",
              "FICO", "open_acc", "Derogatory_recs",
              "Revol_balance", "Revol_util", "Loan_rate"),
  Type = c("Numeric", "Numeric (36 or 60)", "Categorical", "Categorical",
           "Binary", "Binary", "Numeric",
           "Numeric", "Numeric", "Numeric",
           "Numeric", "Numeric", "Numeric",
           "Numeric", "Numeric", "Numeric (target)"),
  Description = c(
    "Loan amount requested in dollars",
    "Loan length in months (36 or 60)",
    "How many years the borrower has been employed",
    "Whether the borrower rents, owns, or has a mortgage",
    "Whether the income source was verified by Lending Club",
    "Whether the income amount was verified",
    "Annual income in thousands of dollars",
    "Debt-to-income ratio: monthly debt payments divided by monthly income",
    "Number of times the borrower was 30+ days late on a payment in the last 2 years",
    "How many years since the borrower opened their first credit account",
    "FICO credit score (300 to 850, higher is better)",
    "Number of currently open credit accounts",
    "Number of derogatory public records (bankruptcies, tax liens, etc.)",
    "Total revolving credit balance in dollars",
    "Revolving credit utilization: how much of available revolving credit is being used",
    "Interest rate assigned by Lending Club (what we are predicting)"
  )
)
knitr::kable(feature_df, caption = "Dataset features and descriptions")
```

| Feature | Type | Description |
|:---|:---|:---|
| Loan_amt | Numeric | Loan amount requested in dollars |
| Loan_term | Numeric (36 or 60) | Loan length in months (36 or 60) |
| emp_length | Categorical | How many years the borrower has been employed |
| Home | Categorical | Whether the borrower rents, owns, or has a mortgage |
| Income_source_verified | Binary | Whether the income source was verified by Lending Club |
| Income_verified | Binary | Whether the income amount was verified |
| Income_thou | Numeric | Annual income in thousands of dollars |
| Debt_income | Numeric | Debt-to-income ratio: monthly debt payments divided by monthly income |
| delinq_2yrs | Numeric | Number of times the borrower was 30+ days late on a payment in the last 2 years |
| Credit_history_length | Numeric | How many years since the borrower opened their first credit account |
| FICO | Numeric | FICO credit score (300 to 850, higher is better) |
| open_acc | Numeric | Number of currently open credit accounts |
| Derogatory_recs | Numeric | Number of derogatory public records (bankruptcies, tax liens, etc.) |
| Revol_balance | Numeric | Total revolving credit balance in dollars |
| Revol_util | Numeric | Revolving credit utilization: how much of available revolving credit is being used |
| Loan_rate | Numeric (target) | Interest rate assigned by Lending Club (what we are predicting) |

Dataset features and descriptions

``` r
lend$emp_length <- factor(lend$emp_length)
lend$Home       <- factor(lend$Home)
cat("Employment length categories:", paste(levels(lend$emp_length), collapse = ", "), "\n")
```

    ## Employment length categories: < 1 year, 1 year, 10+ years, 2 years, 3 years, 4 years, 5 years, 6 years, 7 years, 8 years, 9 years, n/a

``` r
cat("Home ownership categories:   ", paste(levels(lend$Home), collapse = ", "), "\n")
```

    ## Home ownership categories:    MORTGAGE, OWN, RENT

### Distribution of Interest Rates

``` r
library(ggplot2)

ggplot(lend, aes(x = Loan_rate)) +
  geom_histogram(binwidth = 0.5, fill = "#2c7bb6", color = "white", alpha = 0.85) +
  geom_vline(xintercept = mean(lend$Loan_rate), linetype = "dashed",
             color = "#E07B39", linewidth = 1) +
  annotate("text", x = mean(lend$Loan_rate) + 0.5, y = 500,
           label = paste0("Mean: ", round(mean(lend$Loan_rate), 2), "%"),
           color = "#E07B39", hjust = 0, size = 3.8) +
  labs(title = "Distribution of Loan Interest Rates",
       subtitle = "15,000 Lending Club borrowers",
       x = "Interest Rate (%)", y = "Number of Borrowers") +
  theme_minimal() +
  theme(plot.subtitle = element_text(color = "grey40", size = 10))
```

![](analysis_files/figure-gfm/rate-dist-1.png)<!-- -->

The distribution is right-skewed. Most borrowers receive rates between
8% and 18%, but a meaningful tail extends to 26%. The average rate is
about 13.8%. The spread is wide: a borrower at the low end pays roughly
6%, while someone at the high end pays over 26%. That 20 percentage
point spread represents thousands of dollars in additional interest cost
over the life of a loan.

### Loan Rate by Key Variables

``` r
library(gridExtra)

p1 <- ggplot(lend, aes(x = FICO, y = Loan_rate)) +
  geom_point(alpha = 0.08, color = "#2c7bb6", size = 0.8) +
  geom_smooth(method = "lm", color = "#E07B39", se = FALSE, linewidth = 1) +
  labs(title = "FICO Score vs Interest Rate",
       x = "FICO Score", y = "Interest Rate (%)") +
  theme_minimal()

p2 <- ggplot(lend, aes(x = factor(Loan_term), y = Loan_rate, fill = factor(Loan_term))) +
  geom_boxplot(alpha = 0.8, outlier.size = 0.5) +
  scale_fill_manual(values = c("36" = "#2c7bb6", "60" = "#E07B39")) +
  labs(title = "Loan Term vs Interest Rate",
       x = "Loan Term (months)", y = "Interest Rate (%)") +
  theme_minimal() +
  theme(legend.position = "none")

grid.arrange(p1, p2, ncol = 2)
```

![](analysis_files/figure-gfm/eda-charts-1.png)<!-- -->

Two patterns stand out immediately. First, FICO score and interest rate
have a strong negative relationship: higher credit scores consistently
receive lower rates. This is the core logic of credit risk pricing.
Second, 60-month loans carry significantly higher rates than 36-month
loans. Longer loans expose the lender to more uncertainty about whether
the borrower will remain able to repay.

``` r
p3 <- ggplot(lend, aes(x = Debt_income, y = Loan_rate)) +
  geom_point(alpha = 0.08, color = "#3A9E82", size = 0.8) +
  geom_smooth(method = "lm", color = "#E07B39", se = FALSE, linewidth = 1) +
  labs(title = "Debt-to-Income Ratio vs Interest Rate",
       x = "Debt-to-Income Ratio (%)", y = "Interest Rate (%)") +
  theme_minimal()

p4 <- ggplot(lend, aes(x = Home, y = Loan_rate, fill = Home)) +
  geom_boxplot(alpha = 0.8, outlier.size = 0.5) +
  scale_fill_manual(values = c("MORTGAGE" = "#2c7bb6", "OWN" = "#3A9E82", "RENT" = "#E07B39")) +
  labs(title = "Home Ownership vs Interest Rate",
       x = "Home Ownership Status", y = "Interest Rate (%)") +
  theme_minimal() +
  theme(legend.position = "none")

grid.arrange(p3, p4, ncol = 2)
```

![](analysis_files/figure-gfm/eda-charts2-1.png)<!-- -->

Debt-to-income ratio has a mild positive relationship with interest
rate: borrowers who are already carrying more debt relative to their
income tend to receive slightly higher rates. Home ownership shows a
clear pattern: renters pay the highest rates on average, homeowners with
a mortgage are in the middle, and outright homeowners get the best
rates, likely because outright ownership signals greater financial
stability.

------------------------------------------------------------------------

## Train and Validation Split

The data is split 80/20: 12,000 loans for training and 3,000 for
validation. All three models are trained on the same training set and
evaluated on the same validation set, so the RMS errors are directly
comparable.

``` r
set.seed(1)
trainindex <- sample(15000, 12000)
train <- lend[trainindex, ]
valid <- lend[-trainindex, ]

cat("Training set:", nrow(train), "loans\n")
```

    ## Training set: 12000 loans

``` r
cat("Validation set:", nrow(valid), "loans\n")
```

    ## Validation set: 3000 loans

``` r
cat("\nTraining set rate stats:\n")
```

    ## 
    ## Training set rate stats:

``` r
cat("  Mean:", round(mean(train$Loan_rate), 2), "%\n")
```

    ##   Mean: 13.81 %

``` r
cat("  SD:  ", round(sd(train$Loan_rate), 2), "%\n")
```

    ##   SD:   4.33 %

``` r
cat("\nValidation set rate stats:\n")
```

    ## 
    ## Validation set rate stats:

``` r
cat("  Mean:", round(mean(valid$Loan_rate), 2), "%\n")
```

    ##   Mean: 13.91 %

``` r
cat("  SD:  ", round(sd(valid$Loan_rate), 2), "%\n")
```

    ##   SD:   4.29 %

The mean and standard deviation of interest rates are nearly identical
in both sets, confirming the random split produced representative
samples.

------------------------------------------------------------------------

## Model 1: Regression Tree

A regression tree works by repeatedly splitting the data into subgroups
based on feature thresholds. At each split, it picks the feature and
threshold that most reduces the prediction error within the resulting
subgroups. Once splitting stops, each terminal node (leaf) makes a
single prediction: the average interest rate of all training loans that
fell into that leaf.

The appeal of a regression tree is interpretability. You can follow the
path from the root to any leaf and read exactly why a borrower received
a particular predicted rate. The trade-off is that trees can overfit if
grown too deep, and they often miss smooth continuous relationships that
a linear model handles naturally.

``` r
library(tree)

lendtree <- tree(Loan_rate ~ ., data = train)
summary(lendtree)
```

    ## 
    ## Regression tree:
    ## tree(formula = Loan_rate ~ ., data = train)
    ## Variables actually used in tree construction:
    ## [1] "Loan_term" "FICO"     
    ## Number of terminal nodes:  6 
    ## Residual mean deviance:  12.16 = 145800 / 11990 
    ## Distribution of residuals:
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## -10.170  -2.251  -0.329   0.000   2.063  15.370

``` r
plot(lendtree, type = "uniform")
text(lendtree, pretty = 0, cex = 0.8, col = "#2c7bb6")
title(main = "Full Regression Tree: Predicting Loan Interest Rate", cex.main = 1.1)
```

![](analysis_files/figure-gfm/tree-plot-1.png)<!-- -->

The full tree uses only two variables out of the 15 available: loan term
and FICO score. This is a strong signal. Even though the dataset has
income, employment history, debt levels, delinquencies, and many other
features, the tree finds that just those two variables explain most of
the variation in interest rates. FICO score captures creditworthiness
comprehensively, and loan term directly affects the risk premium the
platform charges.

### Cross-Validation to Find Optimal Tree Size

``` r
cv <- cv.tree(lendtree)

cv_df <- data.frame(size = cv$size, rms = sqrt(cv$dev / nrow(train)))

ggplot(cv_df, aes(x = size, y = rms)) +
  geom_line(color = "#2c7bb6", linewidth = 1) +
  geom_point(color = "#2c7bb6", size = 3) +
  geom_vline(xintercept = 6, linetype = "dashed", color = "#E07B39", linewidth = 1) +
  annotate("text", x = 6.15, y = max(cv_df$rms) * 0.99,
           label = "Optimal: 6 nodes", color = "#E07B39", hjust = 0, size = 3.8) +
  labs(title = "Cross-Validation RMS Error by Tree Size",
       subtitle = "Lowest error is achieved at the full tree size (6 nodes)",
       x = "Number of Terminal Nodes", y = "CV RMS Error (%)") +
  theme_minimal() +
  theme(plot.subtitle = element_text(color = "grey40", size = 10))
```

![](analysis_files/figure-gfm/cv-tree-1.png)<!-- -->

Cross-validation tests each possible tree size and measures prediction
error on held-out portions of the training data. Here, the full 6-node
tree achieves the lowest cross-validation error. Pruning would increase
error by removing splits that are genuinely useful. So the full tree is
also the final tree.

``` r
lendtree2 <- prune.tree(lendtree, best = 6)
```

``` r
library(rpart)
library(rpart.plot)

lendtree3 <- rpart(Loan_rate ~ ., data = train)

rpart.plot(lendtree3,
           type = 4, extra = 101,
           box.palette = "Blues",
           shadow.col = "grey80",
           nn = FALSE,
           main = "Regression Tree: Predicting Loan Interest Rate")
```

![](analysis_files/figure-gfm/pruned-tree-plot-1.png)<!-- -->

This cleaner version of the tree shows the same structure with predicted
rates and observation counts at each leaf. Reading from the top:

- The first split is on loan term. Loans with 36-month terms go left,
  60-month terms go right. 60-month loans already carry a rate premium
  just from their length.
- Within each side, FICO score determines where you end up. Higher FICO
  scores push borrowers into lower-rate leaves, lower FICO scores into
  higher-rate leaves.
- The lowest predicted rate (~8.3%) goes to 36-month borrowers with FICO
  above 745. The highest (~19.5%) goes to 60-month borrowers with FICO
  below 660.

### Tree Predictions vs Actual Rates

``` r
pred_tree <- predict(lendtree2, newdata = valid)
validRMS_tree <- sqrt(sum((valid$Loan_rate - pred_tree)^2) / nrow(valid))
cat("Regression tree RMS error on validation set:", round(validRMS_tree, 4), "%\n")
```

    ## Regression tree RMS error on validation set: 3.4935 %

``` r
resid_df <- data.frame(
  Actual    = valid$Loan_rate,
  Predicted = pred_tree,
  Residual  = valid$Loan_rate - pred_tree
)

p_pred <- ggplot(resid_df, aes(x = Predicted, y = Actual)) +
  geom_point(alpha = 0.15, color = "#2c7bb6", size = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "#E07B39", linewidth = 1) +
  labs(title = "Predicted vs Actual: Regression Tree",
       x = "Predicted Rate (%)", y = "Actual Rate (%)") +
  theme_minimal()

p_resid <- ggplot(resid_df, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.15, color = "#2c7bb6", size = 0.8) +
  geom_hline(yintercept = 0, color = "#E07B39", linewidth = 1) +
  labs(title = "Residuals: Regression Tree",
       x = "Predicted Rate (%)", y = "Residual (Actual minus Predicted)") +
  theme_minimal()

grid.arrange(p_pred, p_resid, ncol = 2)
```

![](analysis_files/figure-gfm/tree-residuals-1.png)<!-- -->

The predicted vs actual chart shows the limitation of the regression
tree clearly. Because the tree only has 6 leaf nodes, it can only
predict 6 distinct values. You can see the predicted values clustering
in vertical bands. The tree is partitioning borrowers into buckets but
cannot make a smooth, continuous prediction the way a linear model can.
Borrowers within the same leaf all receive the same predicted rate
regardless of how different their individual profiles are.

------------------------------------------------------------------------

## Model 2: Linear Regression with Best Subset Selection

Best subset selection tries every possible combination of features and
finds the one that produces the best model according to a criterion.
Here we use adjusted R-squared, which rewards models that explain more
variance while penalizing unnecessary complexity.

The reason to do this rather than simply including all features is that
not every feature improves predictions. Some are redundant or add noise.
Subset selection finds the most useful combination.

``` r
library(leaps)

subset_model <- regsubsets(Loan_rate ~ ., data = train, nvmax = 20)
subset_summary <- summary(subset_model)

adj_r2_df <- data.frame(
  num_vars = 1:length(subset_summary$adjr2),
  adj_r2   = subset_summary$adjr2
)
best_n <- which.max(subset_summary$adjr2)

ggplot(adj_r2_df, aes(x = num_vars, y = adj_r2)) +
  geom_line(color = "#2c7bb6", linewidth = 1) +
  geom_point(color = "#2c7bb6", size = 3) +
  geom_vline(xintercept = best_n, linetype = "dashed",
             color = "#E07B39", linewidth = 1) +
  annotate("text", x = best_n + 0.3, y = min(adj_r2_df$adj_r2) + 0.01,
           label = paste0("Best: ", best_n, " variables"),
           color = "#E07B39", hjust = 0, size = 3.8) +
  scale_x_continuous(breaks = 1:max(adj_r2_df$num_vars)) +
  labs(title = "Best Subset Selection: Adjusted R-squared by Model Size",
       subtitle = "Higher adjusted R-squared means better model fit, penalized for complexity",
       x = "Number of Variables in Model", y = "Adjusted R-squared") +
  theme_minimal() +
  theme(plot.subtitle = element_text(color = "grey40", size = 10))
```

![](analysis_files/figure-gfm/subset-selection-1.png)<!-- -->

``` r
coef_best <- coef(subset_model, best_n)
cat("Best model uses", best_n, "variables:\n")
```

    ## Best model uses 19 variables:

``` r
print(round(coef_best, 4))
```

    ##            (Intercept)              Loan_term    emp_length10+ years 
    ##                44.1280                 0.1631                 0.1058 
    ##      emp_length2 years      emp_length5 years      emp_length8 years 
    ##                -0.1141                 0.2034                 0.2884 
    ##          emp_lengthn/a                HomeOWN               HomeRENT 
    ##                -0.1947                 0.2527                 0.1453 
    ## Income_source_verified        Income_verified            Income_thou 
    ##                 0.6261                 1.8810                -0.0028 
    ##            Debt_income            delinq_2yrs  Credit_history_length 
    ##                 0.0517                 0.1014                -0.0389 
    ##                   FICO               open_acc        Derogatory_recs 
    ##                -0.0554                -0.0250                 0.1176 
    ##          Revol_balance             Revol_util 
    ##                 0.0000                 0.7006

``` r
# Build the best variable names and fit final linear model
best_vars <- names(coef_best)[-1]

# Handle factor dummy variable names back to original column names
clean_vars <- unique(gsub("MORTGAGE|OWN|RENT|< 1 year|1 year|2 years|3 years|4 years|5 years|6 years|7 years|8 years|9 years|10\\+ years|n/a", "", best_vars))
clean_vars <- unique(trimws(clean_vars))
clean_vars <- clean_vars[clean_vars != ""]

# Use the full formula approach: fit on all vars, subset selection just tells us which matter
# Build formula from best_vars
build_formula <- function(coef_names, response = "Loan_rate") {
  vars <- coef_names[coef_names != "(Intercept)"]
  # strip factor level suffixes to get original column names
  orig <- gsub("(Home|emp_length)(.*)", "\\1", vars)
  orig <- unique(orig)
  as.formula(paste(response, "~", paste(orig, collapse = " + ")))
}

lm_formula <- build_formula(names(coef_best))
cat("Formula used:\n")
```

    ## Formula used:

``` r
print(lm_formula)
```

    ## Loan_rate ~ Loan_term + emp_length + Home + Income_source_verified + 
    ##     Income_verified + Income_thou + Debt_income + delinq_2yrs + 
    ##     Credit_history_length + FICO + open_acc + Derogatory_recs + 
    ##     Revol_balance + Revol_util
    ## <environment: 0x00000118ccb45ac0>

``` r
lm_model <- lm(lm_formula, data = train)
summary(lm_model)
```

    ## 
    ## Call:
    ## lm(formula = lm_formula, data = train)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -10.3506  -2.2445  -0.2754   1.8240  15.5121 
    ## 
    ## Coefficients:
    ##                          Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)             4.411e+01  9.402e-01  46.909  < 2e-16 ***
    ## Loan_term               1.631e-01  2.884e-03  56.572  < 2e-16 ***
    ## emp_length1 year        1.185e-03  1.639e-01   0.007 0.994230    
    ## emp_length10+ years     1.188e-01  1.221e-01   0.973 0.330735    
    ## emp_length2 years      -1.008e-01  1.494e-01  -0.675 0.499884    
    ## emp_length3 years       5.769e-02  1.531e-01   0.377 0.706355    
    ## emp_length4 years       5.427e-03  1.710e-01   0.032 0.974681    
    ## emp_length5 years       2.167e-01  1.686e-01   1.286 0.198589    
    ## emp_length6 years      -3.713e-02  1.726e-01  -0.215 0.829689    
    ## emp_length7 years       1.119e-01  1.691e-01   0.662 0.508227    
    ## emp_length8 years       3.017e-01  1.737e-01   1.737 0.082451 .  
    ## emp_length9 years      -8.534e-02  1.842e-01  -0.463 0.643165    
    ## emp_lengthn/a          -1.818e-01  1.833e-01  -0.992 0.321380    
    ## HomeOWN                 2.532e-01  1.091e-01   2.320 0.020362 *  
    ## HomeRENT                1.454e-01  6.873e-02   2.116 0.034368 *  
    ## Income_source_verified  6.264e-01  7.576e-02   8.269  < 2e-16 ***
    ## Income_verified         1.881e+00  8.490e-02  22.158  < 2e-16 ***
    ## Income_thou            -2.845e-03  7.389e-04  -3.849 0.000119 ***
    ## Debt_income             5.179e-02  4.332e-03  11.953  < 2e-16 ***
    ## delinq_2yrs             1.017e-01  3.368e-02   3.019 0.002538 ** 
    ## Credit_history_length  -3.882e-02  4.428e-03  -8.767  < 2e-16 ***
    ## FICO                   -5.536e-02  1.264e-03 -43.792  < 2e-16 ***
    ## open_acc               -2.509e-02  6.604e-03  -3.799 0.000146 ***
    ## Derogatory_recs         1.170e-01  5.753e-02   2.034 0.042020 *  
    ## Revol_balance          -5.895e-06  1.596e-06  -3.695 0.000221 ***
    ## Revol_util              7.012e-01  1.603e-01   4.375 1.22e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3.32 on 11974 degrees of freedom
    ## Multiple R-squared:  0.413,  Adjusted R-squared:  0.4118 
    ## F-statistic: 337.1 on 25 and 11974 DF,  p-value: < 2.2e-16

``` r
lm_coef <- as.data.frame(summary(lm_model)$coefficients)
lm_coef$Variable <- rownames(lm_coef)
lm_coef <- lm_coef[lm_coef$Variable != "(Intercept)", ]
lm_coef$Significant <- lm_coef$`Pr(>|t|)` < 0.05

readable <- c(
  "Loan_term"              = "Loan term (months)",
  "FICO"                   = "FICO credit score",
  "Debt_income"            = "Debt-to-income ratio",
  "Income_thou"            = "Annual income (thousands)",
  "Loan_amt"               = "Loan amount",
  "delinq_2yrs"            = "Delinquencies (last 2 yrs)",
  "Revol_util"             = "Revolving credit utilization",
  "open_acc"               = "Open credit accounts",
  "Credit_history_length"  = "Credit history length",
  "Derogatory_recs"        = "Derogatory public records",
  "Revol_balance"          = "Revolving credit balance",
  "Income_source_verified" = "Income source verified",
  "Income_verified"        = "Income verified",
  "HomeOWN"                = "Home: owns outright",
  "HomeRENT"               = "Home: renting",
  "HomeMORTGAGE"           = "Home: mortgage"
)

lm_coef$Label <- ifelse(lm_coef$Variable %in% names(readable),
                        readable[lm_coef$Variable],
                        lm_coef$Variable)

ggplot(lm_coef, aes(x = reorder(Label, Estimate), y = Estimate, fill = Significant)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("FALSE" = "grey75", "TRUE" = "#2c7bb6")) +
  labs(title = "Linear Regression Coefficients",
       subtitle = "Blue bars are statistically significant (p < 0.05). Positive values push the rate higher.",
       x = NULL, y = "Coefficient Estimate",
       fill = "Significant") +
  theme_minimal() +
  theme(plot.subtitle = element_text(color = "grey40", size = 9))
```

![](analysis_files/figure-gfm/lm-coef-plot-1.png)<!-- -->

``` r
pred_lm <- predict(lm_model, newdata = valid)
validRMS_lm <- sqrt(sum((valid$Loan_rate - pred_lm)^2) / nrow(valid))
cat("Linear regression RMS error on validation set:", round(validRMS_lm, 4), "%\n")
```

    ## Linear regression RMS error on validation set: 3.3189 %

``` r
resid_lm_df <- data.frame(
  Actual    = valid$Loan_rate,
  Predicted = pred_lm,
  Residual  = valid$Loan_rate - pred_lm
)

p_lm1 <- ggplot(resid_lm_df, aes(x = Predicted, y = Actual)) +
  geom_point(alpha = 0.15, color = "#3A9E82", size = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "#E07B39", linewidth = 1) +
  labs(title = "Predicted vs Actual: Linear Regression",
       x = "Predicted Rate (%)", y = "Actual Rate (%)") +
  theme_minimal()

p_lm2 <- ggplot(resid_lm_df, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.15, color = "#3A9E82", size = 0.8) +
  geom_hline(yintercept = 0, color = "#E07B39", linewidth = 1) +
  labs(title = "Residuals: Linear Regression",
       x = "Predicted Rate (%)", y = "Residual") +
  theme_minimal()

grid.arrange(p_lm1, p_lm2, ncol = 2)
```

![](analysis_files/figure-gfm/lm-residuals-1.png)<!-- -->

The linear regression predicted vs actual chart looks much tighter than
the regression tree. Instead of 6 fixed values, the linear model makes a
continuous prediction for each borrower. The predictions spread across
the full range of actual rates. The residuals are centered around zero
with no obvious pattern, which is what you want to see.

------------------------------------------------------------------------

## Model 3: LASSO Regression

LASSO (Least Absolute Shrinkage and Selection Operator) is a form of
linear regression that adds a penalty for including too many variables.
As the penalty strength (lambda) increases, LASSO shrinks some
coefficients toward zero and eventually sets them exactly to zero,
effectively removing those variables from the model. This is called
regularization.

The practical benefit is that LASSO automatically handles feature
selection. You give it all the variables and it figures out which ones
matter. It also tends to generalize better than ordinary linear
regression when some features are noisy or correlated with each other.

``` r
library(glmnet)

x_train <- model.matrix(Loan_rate ~ ., data = train)[, -1]
y_train <- train$Loan_rate
x_valid <- model.matrix(Loan_rate ~ ., data = valid)[, -1]
y_valid  <- valid$Loan_rate

cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)

lasso_cv_df <- data.frame(
  log_lambda = log(cv_lasso$lambda),
  mse        = cv_lasso$cvm,
  upper      = cv_lasso$cvup,
  lower      = cv_lasso$cvlo
)

ggplot(lasso_cv_df, aes(x = log_lambda, y = mse)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#2c7bb6", alpha = 0.15) +
  geom_line(color = "#2c7bb6", linewidth = 1) +
  geom_vline(xintercept = log(cv_lasso$lambda.min), linetype = "dashed",
             color = "#E07B39", linewidth = 1) +
  annotate("text",
           x = log(cv_lasso$lambda.min) + 0.15,
           y = max(lasso_cv_df$mse) * 0.98,
           label = paste0("Best lambda = ", round(cv_lasso$lambda.min, 4)),
           color = "#E07B39", hjust = 0, size = 3.5) +
  labs(title = "LASSO Cross-Validation: MSE by Lambda",
       subtitle = "Shaded band shows standard error. Dashed line marks the lambda with lowest error.",
       x = "Log(Lambda)", y = "Mean Squared Error") +
  theme_minimal() +
  theme(plot.subtitle = element_text(color = "grey40", size = 10))
```

![](analysis_files/figure-gfm/lasso-1.png)<!-- -->

As lambda increases (moving right on the chart), the model becomes more
penalized and error rises. The dashed line marks the lambda value that
minimizes cross-validated MSE. That is the lambda used for final
predictions.

``` r
best_lambda <- cv_lasso$lambda.min
lasso_coefs <- coef(cv_lasso, s = best_lambda)
coef_df <- data.frame(
  Variable    = rownames(lasso_coefs),
  Coefficient = as.vector(lasso_coefs)
)
coef_df <- coef_df[coef_df$Variable != "(Intercept)" & coef_df$Coefficient != 0, ]
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]

coef_df$Coefficient <- round(coef_df$Coefficient, 5)
cat("LASSO non-zero coefficients at best lambda:\n")
```

    ## LASSO non-zero coefficients at best lambda:

``` r
knitr::kable(coef_df,
             caption = paste0("LASSO coefficients (lambda = ", round(best_lambda, 4), ")"),
             row.names = FALSE)
```

| Variable               | Coefficient |
|:-----------------------|------------:|
| Income_verified        |     1.82821 |
| Revol_util             |     0.66236 |
| Income_source_verified |     0.58893 |
| emp_length8 years      |     0.22947 |
| HomeOWN                |     0.20566 |
| Loan_term              |     0.16274 |
| emp_length5 years      |     0.14970 |
| emp_lengthn/a          |    -0.14645 |
| HomeRENT               |     0.11732 |
| Derogatory_recs        |     0.09886 |
| emp_length2 years      |    -0.09680 |
| delinq_2yrs            |     0.08737 |
| emp_length10+ years    |     0.07251 |
| emp_length9 years      |    -0.07085 |
| FICO                   |    -0.05548 |
| Debt_income            |     0.05060 |
| emp_length7 years      |     0.04248 |
| Credit_history_length  |    -0.03756 |
| emp_length6 years      |    -0.02605 |
| open_acc               |    -0.02348 |
| Income_thou            |    -0.00272 |
| Revol_balance          |    -0.00001 |
| Loan_amt               |     0.00000 |

LASSO coefficients (lambda = 0.0095)

``` r
coef_df$Direction <- ifelse(coef_df$Coefficient > 0, "Increases rate", "Decreases rate")

ggplot(coef_df, aes(x = reorder(Variable, Coefficient), y = Coefficient, fill = Direction)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("Increases rate" = "#E07B39", "Decreases rate" = "#2c7bb6")) +
  labs(title = "LASSO Regression Coefficients at Best Lambda",
       subtitle = "Only non-zero coefficients shown. Variables set to zero were removed by LASSO.",
       x = NULL, y = "Coefficient", fill = NULL) +
  theme_minimal() +
  theme(plot.subtitle = element_text(color = "grey40", size = 9),
        legend.position = "bottom")
```

![](analysis_files/figure-gfm/lasso-coef-plot-1.png)<!-- -->

``` r
pred_lasso <- predict(cv_lasso, s = best_lambda, newx = x_valid)
validRMS_lasso <- sqrt(sum((y_valid - pred_lasso)^2) / nrow(valid))
cat("LASSO RMS error on validation set:", round(validRMS_lasso, 4), "%\n")
```

    ## LASSO RMS error on validation set: 3.3176 %

``` r
resid_lasso_df <- data.frame(
  Actual    = y_valid,
  Predicted = as.vector(pred_lasso),
  Residual  = y_valid - as.vector(pred_lasso)
)

p_las1 <- ggplot(resid_lasso_df, aes(x = Predicted, y = Actual)) +
  geom_point(alpha = 0.15, color = "#E07B39", size = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "#2c7bb6", linewidth = 1) +
  labs(title = "Predicted vs Actual: LASSO",
       x = "Predicted Rate (%)", y = "Actual Rate (%)") +
  theme_minimal()

p_las2 <- ggplot(resid_lasso_df, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.15, color = "#E07B39", size = 0.8) +
  geom_hline(yintercept = 0, color = "#2c7bb6", linewidth = 1) +
  labs(title = "Residuals: LASSO",
       x = "Predicted Rate (%)", y = "Residual") +
  theme_minimal()

grid.arrange(p_las1, p_las2, ncol = 2)
```

![](analysis_files/figure-gfm/lasso-residuals-1.png)<!-- -->

------------------------------------------------------------------------

## Model Comparison

``` r
results <- data.frame(
  Model = c("Regression Tree", "Linear Regression (Best Subset)", "LASSO Regression"),
  RMS_Error = c(round(validRMS_tree, 4),
                round(validRMS_lm,   4),
                round(validRMS_lasso, 4)),
  Features_Used = c("2 (Loan_term, FICO)",
                    paste(best_n, "selected by subset search"),
                    paste(nrow(coef_df), "non-zero after penalization"))
)
knitr::kable(results,
             caption = "Model performance on validation set (lower RMS error is better)",
             col.names = c("Model", "RMS Error (%)", "Features Used"))
```

| Model | RMS Error (%) | Features Used |
|:---|---:|:---|
| Regression Tree | 3.4935 | 2 (Loan_term, FICO) |
| Linear Regression (Best Subset) | 3.3189 | 19 selected by subset search |
| LASSO Regression | 3.3176 | 23 non-zero after penalization |

Model performance on validation set (lower RMS error is better)

``` r
ggplot(results, aes(x = reorder(Model, RMS_Error), y = RMS_Error, fill = Model)) +
  geom_bar(stat = "identity", width = 0.55) +
  geom_text(aes(label = paste0(RMS_Error, "%")), vjust = -0.4, size = 4) +
  scale_fill_manual(values = c(
    "Regression Tree"                  = "#2c7bb6",
    "Linear Regression (Best Subset)"  = "#3A9E82",
    "LASSO Regression"                 = "#E07B39"
  )) +
  scale_y_continuous(limits = c(0, max(results$RMS_Error) * 1.15)) +
  labs(title = "Validation Set RMS Error by Model",
       subtitle = "Lower is better",
       x = NULL, y = "RMS Error (%)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.subtitle = element_text(color = "grey40", size = 10))
```

![](analysis_files/figure-gfm/comparison-chart-1.png)<!-- -->

``` r
all_resid <- data.frame(
  Residual = c(resid_df$Residual, resid_lm_df$Residual, resid_lasso_df$Residual),
  Model    = rep(c("Regression Tree", "Linear Regression", "LASSO"),
                 each = nrow(valid))
)

ggplot(all_resid, aes(x = Residual, fill = Model, color = Model)) +
  geom_density(alpha = 0.25, linewidth = 0.9) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_fill_manual(values  = c("Regression Tree" = "#2c7bb6",
                                "Linear Regression" = "#3A9E82",
                                "LASSO" = "#E07B39")) +
  scale_color_manual(values = c("Regression Tree" = "#2c7bb6",
                                "Linear Regression" = "#3A9E82",
                                "LASSO" = "#E07B39")) +
  labs(title = "Distribution of Prediction Errors by Model",
       subtitle = "Tighter distribution centered at zero means more accurate predictions",
       x = "Prediction Error (Actual minus Predicted)", y = "Density",
       fill = NULL, color = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.subtitle = element_text(color = "grey40", size = 10))
```

![](analysis_files/figure-gfm/error-dist-1.png)<!-- -->

The regression tree’s error distribution has a distinctive multi-modal
shape, with spikes at specific values. This is the bucket effect:
because the tree only predicts 6 distinct values, errors cluster at
multiples of the differences between those values. LASSO and linear
regression produce smooth, bell-shaped error distributions centered near
zero, which is the ideal behavior.

------------------------------------------------------------------------

## Key Takeaways

**FICO score and loan term drive interest rate more than anything
else.** The regression tree found this by using only those two variables
out of 15. Linear regression and LASSO confirm it with their coefficient
magnitudes. Lending Club’s rate-setting logic is, at its core, a
function of how creditworthy you are and how long you need the money.

**Linear models outperform the tree here.** The interest rate in this
dataset increases relatively smoothly as FICO falls or loan term
extends. Linear models are built for exactly this kind of smooth
continuous relationship. The regression tree divides borrowers into
buckets and assigns everyone in a bucket the same predicted rate, which
is a crude approximation of what is actually a continuous function.

**LASSO and linear regression perform nearly identically.** This is
common when the dataset is large, the signal is strong, and the features
are not highly multicollinear. LASSO’s main advantage shows up more
clearly in smaller datasets with many noisy features. Here, best subset
selection and LASSO converge on similar variable choices and similar
errors.

**A 3.5% RMS error means the model’s typical prediction is off by about
3.5 percentage points.** Given that rates range from 6% to 26%, this is
a meaningful error, not a trivial one. Some of what drives individual
rates cannot be captured by the features in this dataset. Lending Club
also uses proprietary internal grades and loan history information that
are not included here.
