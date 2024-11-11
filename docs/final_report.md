# Olist Marketplace Causal Analysis


Understanding the causal relationship between delivery delays and customer satisfaction is crucial for e-commerce platforms, as it directly impacts customer retention, brand reputation, and long-term profitability. While delivery performance metrics are routinely collected, establishing a true causal link between delays and customer satisfaction presents unique challenges.

Although randomized controlled trials (RCTs) represent the gold standard in causal inference, deliberately delaying customer deliveries for experimental purposes would be both unethical and potentially damaging to business operations. This necessitates the use of observational study methods to draw reliable causal conclusions from existing data.

This study leverages the rich dataset from Olist, Brazil's largest department store marketplace, to estimate the causal effect of delivery delays on customer satisfaction ratings. We employ two complementary approaches: propensity score matching to simulate experimental conditions using observational data, and graphical causal models to understand the underlying data-generating process. This dual methodology allows us to not only quantify the impact of delays but also to understand the complex mechanisms through which delivery performance affects customer satisfaction.



### 1.2 Research Questions

Before we formalize out research questions, we need to be clear about how we define a late delivery, this is important when analyzing the results

- Primary research question
- Definition of key terms (delayed delivery, customer satisfaction)
- Scope and limitations

## 2. Data Overview
### 2.1 Dataset Description
- Source and time period
- Key variables
- Summary statistics

### 2.2 Missing Data Analysis
- Pattern of missing data
- Treatment of missing values
- Potential impact on analysis

## 3. Methodology
### 3.1 Causal Framework
- Potential outcomes framework
- Key assumptions
- Identification strategy

### 3.2 Propensity Score Matching (Late Delivery --> Ratings)
- Matching methodology
- Covariate selection
- Balance diagnostics
- Results
- Robustness Checks
- Interpretation of Results

### 3.3 Graphical Causal Models
- DAG specification
- Model assumptions
- Identification strategy
- Results
- Robustness Checks
- Interpretation of Results

### 3.3 Graphical Causal Models
- DAG specification
- Model assumptions
- Identification strategy
- Results
- Robustness Checks
- Interpretation of Results

### 3.2 Propensity Score Matching (Late Delivery --> Revenue)
- Matching methodology
- Logic behind lags
- Covariate selection
- Balance diagnostics
- Results
- Robustness Checks
- Interpretation of Results


### 6.2 Practical Implications
- Recommendations for e-commerce operations
- Potential interventions
- Cost-benefit considerations



### References

let's include some literature



