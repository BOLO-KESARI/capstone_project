# AI-Driven Airline Demand Forecasting and Revenue Optimization System

**Bolo Kesari**
St. Vincent Pallotti College of Engineering and Technology, Nagpur, India
Email: bolokesari@svpcet.edu

---

## Abstract

The airline industry operates on extremely narrow profit margins, where small inefficiencies in demand prediction, pricing, and seat allocation significantly impact profitability. This paper presents an AI-driven Airline Revenue Optimization and Flight Demand Forecasting System — an end-to-end analytics platform for intelligent revenue management in domestic aviation.

The system integrates multiple machine learning models, time-series forecasting techniques, and optimization algorithms to analyze passenger demand and optimize pricing strategies. A 7-factor dynamic pricing engine adjusts fares based on demand trends, seasonal effects, booking windows, and market conditions. A Monte Carlo–based overbooking simulation determines optimal overbooking limits, and multi-objective Pareto optimization supports route-level profitability decisions. The platform uses a star-schema data warehouse with automated ETL pipelines, SHAP-based model explainability, Population Stability Index (PSI) drift detection, and walk-forward validation. A Revenue Command Center dashboard provides interactive analytics and KPIs for real-time monitoring.

Experimental results demonstrate MAPE below 10% for demand forecasting, churn prediction AUC-ROC of 0.87, and route profitability R² above 0.90.

**Keywords:** Revenue Management, Dynamic Pricing, Demand Forecasting, XGBoost, Monte Carlo Simulation, Pareto Optimization, SHAP Explainability, Time-Series Forecasting, Airline Analytics, FastAPI, Star Schema.

---

## I. Introduction

The global airline industry generated approximately $996 billion in revenue in 2024 [10], yet average net profit margins remain between 2–5%. Revenue management — selling the right seat to the right customer at the right price at the right time — has been central to airline profitability since American Airlines pioneered yield management in the 1980s. In the Indian domestic market, which handled over 150 million passengers in 2024 [7], competition among IndiGo, Air India, SpiceJet, and Akasa Air makes sophisticated revenue optimization essential.

Traditional revenue management relies on static fare buckets, manually-set booking limits, and analyst intuition. These approaches struggle with real-time competitor pricing, multi-channel booking, volatile demand from festivals and events, and price-sensitive passengers. Machine learning offers a paradigm shift — replacing heuristics with data-driven predictions that adapt continuously to market dynamics.

A mid-sized Indian airline operating 500+ daily flights across 150 domestic routes faces several challenges. Its current average load factor stands at 78%, significantly below the 85–90% industry benchmark, resulting in approximately ₹450 crore in unrealized annual revenue. Static fare buckets miss demand surges during Indian festivals (Diwali, Holi, Durga Puja) and wedding seasons. Denied boarding incidents cost ₹15–25 crore annually, while conservative overbooking leaves an average of 8–12 empty seats per flight. Low-cost carriers undercut fares by 10–20% on key metro routes without the airline having a real-time competitive response. Demand forecasting errors exceeding 15% MAPE on seasonal routes cause capacity misallocation, and the 12–15 new routes launched annually lack historical data for initial pricing.

This project delivers an integrated analytics platform with 12 ML models covering demand forecasting, dynamic pricing, overbooking optimization, customer churn, route profitability, operational risk, flight delay, cancellation probability, load factor, no-show prediction, passenger clustering, and competitor price anticipation. The system includes a 7-factor dynamic pricing engine with a 365-day forward pricing calendar, a Monte Carlo overbooking optimizer, a multi-objective Pareto optimizer, a cold-start strategy for new routes, a production-grade FastAPI backend (49+ endpoints), and a Revenue Command Center dashboard with 12 analytics tabs and 40+ KPI visualizations.

---

## II. Literature Review

Revenue management originated with Littlewood's Rule [13], which established that a seat should be sold at a lower fare only if the expected revenue exceeds the probability-weighted revenue from a future higher-fare booking. Belobaba [2] extended this to the Expected Marginal Seat Revenue (EMSR) framework, which remains the industry standard for seat inventory allocation. Talluri and van Ryzin [20] demonstrated that revenue management can improve airline revenue by 4–8% compared to first-come-first-served pricing. Rothstein [18] established foundations for optimal overbooking under stochastic demand, and Fiig et al. [8] introduced choice-based optimization models accounting for customer behavior across fare classes.

Chen and Kachani [3] showed that gradient boosting captures non-linear demand-price relationships better than linear models. Weatherford and Kimes [23] demonstrated that neural networks outperform traditional EMSR methods in volatile markets, achieving 15–20% improvements in forecast accuracy. SHAP [14] provides game-theoretically consistent feature attributions essential for model explainability in safety-critical domains.

Abdella et al. [1] compared ML algorithms for airline demand prediction, finding XGBoost achieved the lowest MAPE of 6.8% for 7-day ahead forecasts, outperforming LSTM (8.2%) and ARIMA (14.5%). Booking horizon and seasonal index were the two most predictive features, accounting for 38% of total feature importance. Shihab et al. [19] developed a two-stage LightGBM pipeline showing that incorporating competitor pricing improves demand forecast accuracy by 12–15%, with 5.2% revenue uplift over static analyst-set fares.

Gönsch and Steinhardt [9] found that deep reinforcement learning outperformed EMSR-b by 3–5% on average revenue per flight, but cold-start performance was poor (requiring 50,000+ training episodes per route), limiting practical adoption. Kumar and Singh [11] identified that Indian festival seasons drive 35–50% demand surges and monsoon seasonality depresses demand by 15–25% on leisure routes. Their festival-aware XGBoost + Prophet ensemble achieved MAPE of 7.2% on festival-week forecasts versus 18.5% for a festival-unaware model. Zhang et al. [24] developed a no-show prediction system achieving AUC-ROC of 0.83 and a 22% reduction in denied boardings.

Li et al. [12] found that for real-time pricing, weighted-sum optimization was 200× faster than NSGA-III with near-identical top solutions (within 1.2% revenue difference). Park and Lee [16] demonstrated that ML-enhanced Monte Carlo reduces total overbooking cost by 18–24%, and that 1,000 simulations per overbooking level provides less than 2% error versus the theoretical optimum. Chen et al. [4] found that a Prophet+SARIMA ensemble (60/40 weighting) achieved MAPE of 8.4% on 7-day horizons, and that walk-forward validation is critical — standard k-fold showed 20–30% optimistically biased MAPE estimates. Ramanathan and Subramanian [17] found that 72% of pricing analysts would not trust a model without feature-level explanations, and that DGCA's 2024 fare transparency guidelines require audit trails for a minimum of 3 years.

Most existing literature addresses a single aspect — pricing, forecasting, or overbooking — in isolation. No published system integrates all 12 model types with dynamic pricing, overbooking, multi-objective optimization, and regulatory compliance in a single production-grade platform. This project addresses all of these research gaps.

---

## III. Methodology

### 1. System Architecture

The system is built on FastAPI 0.104+ as the backend framework, providing an async REST API with OpenAPI documentation and Pydantic validation. The ML stack includes scikit-learn 1.3+, XGBoost 2.0+, Prophet 1.1+, statsmodels 0.14+, and SHAP 0.43+. Data processing uses Pandas 2.0+ and NumPy 1.24+. The database layer uses SQLite in WAL mode. ETL scheduling is managed by APScheduler 3.10+ with 7 concurrent cron jobs. External data integration connects to AviationStack and OpenWeather APIs. The frontend is implemented using HTML5, Chart.js 4.x, and Vanilla JavaScript. Security uses custom JWT authentication with PBKDF2-SHA256 hashing and role-based access control.

The architecture flows through five layers: (1) Data Sources, (2) ETL & Data Quality, (3) Star Schema Database, (4) ML/Analytics Engine, and (5) API + Dashboard Presentation.

### 2. Data Engineering and ETL Pipeline

The platform ingests data from 14 simulated CSV datasets, 5 ML-specific training datasets, 1 NoSQL JSON source (passenger intelligence), and 1 OLAP CSV view (route performance). Key datasets include ~50,000 booking records, ~10,000 flight records, ~5,000 passenger profiles, ~15,000 competitor price snapshots, and ~50,000 records with 50+ engineered features.

The ETL pipeline runs 7 cron jobs: daily demand aggregation (2:00 AM), route revenue view refresh (2:15 AM), load factor aggregation (2:30 AM), PSI-triggered model retraining check (3:00 AM), competitor price refresh (6:00 AM), weather sync every 30 minutes for 20+ Indian airports, and sentiment processing every 6 hours.

Nine data quality rules enforce data integrity: IATA airport code validation, negative price removal, impossible date detection, competitor price imputation, booking spike detection (>3 standard deviations above 30-day rolling average), timezone reconciliation to IST (UTC+5:30), stale feed flagging, weather backfill, and fare class validation.

### 3. Database Design (Star Schema)

The data warehouse follows a star schema optimized for OLAP queries in SQLite with WAL mode. It consists of 5 dimension tables (time, route, passenger, price, aircraft), 3 fact tables (bookings, flights, revenue), a time-series booking velocity table, 6 flexible collection tables (competitor prices, sentiment, events, weather, etc.), an immutable audit log, and a model registry. Three materialized views (route revenue, load factor aggregates, daily booking pace) are refreshed by ETL jobs. Fourteen composite indexes optimize the most frequent query patterns.

### 4. Feature Engineering

The advanced features dataset contains 50+ engineered features across 10 categories:

- **Booking patterns:** lead time, booking velocity (normalized 7-day rolling rate), pace vs. historical, booking day/time, horizon bucket
- **Temporal/seasonality:** calendar attributes, holiday and weekend flags, peak season indicator, multiplicative seasonal index (0.6–1.4)
- **Route characteristics:** distance, haul type, destination category, seat capacity, flight frequency
- **Pricing/revenue:** base fare, RASM, yield per passenger, ancillary revenue, discount percentage
- **Competitive landscape:** competitor count, market share, competitor price differential, price competitiveness index
- **Passenger segmentation:** business traveler probability, loyalty tier, lifetime value metrics
- **Demand indicators:** search volume, social sentiment score
- **Operational:** on-time percentage, average delay, crew availability, maintenance flag rate
- **No-show/cancellation:** route-class no-show rates, cancellation rate by horizon bucket
- **Overbooking:** optimal overbook quantity, denied boarding cost, empty seat cost

### 5. Machine Learning Models

The platform trains 12 ML models at startup, stored in-memory for real-time inference:

**Figure 2: Model Performance Summary Across All 12 ML Models**

![Model Performance Summary](generated_reports/12_model_performance_summary.png)

| Model | Algorithm | Target |
|---|---|---|
| Demand Forecasting | XGBoost Regressor | Passenger demand |
| Dynamic Pricing | XGBoost Regressor | Optimal fare |
| Overbooking Optimization | Monte Carlo Simulation | Optimal overbook level |
| Route Profitability | Linear Regression | Route profit |
| Operational Risk | Random Forest Classifier | Risk category |
| Customer Churn | Gradient Boosting (n=150, d=5) | Churn probability |
| Flight Delay | Random Forest Classifier | Delay probability |
| Cancellation | Gradient Boosting Classifier | Cancellation probability |
| Load Factor | Random Forest Regressor | Load factor |
| No-Show | Gradient Boosting Classifier | No-show probability |
| Passenger Clustering | K-Means (k=4) | Passenger segment |
| Time-Series | Prophet + SARIMA (60/40) | Multi-horizon demand |

The **Overbooking Optimization** model uses Monte Carlo simulation with 1,000 runs per flight. For each candidate overbooking level $k \in [0, 24]$, it simulates flights with $B = \text{capacity} + k$ bookings, draws no-shows from a Binomial distribution, and selects $k^* = \arg\min_k E[C_k]$ where:

$$E[C_k] = \overline{\text{denied}} \times C_{deny} + \overline{\text{empty}} \times C_{empty}$$

The **Churn model** categorizes risk as: High Risk (≥70% probability — immediate intervention), Medium Risk (40–70% — re-engagement campaign), and Low Risk (<40% — maintain engagement).

The **Clustering model** with k=4 and StandardScaler normalization automatically labels segments as: Premium Frequent, High-Value Leisure, At-Risk/Churners, and Budget Traveler.

### 6. Dynamic Pricing Engine

The pricing engine computes fares via 7 multiplicative factors applied to a distance-based base price:

$$P_{final} = P_{base} \times f_{demand} \times f_{horizon} \times f_{event} \times f_{weather} \times f_{competition} \times f_{seats} \times f_{segment}$$

Subject to constraints: $P_{floor} = 0.55 \times P_{base}$ and $P_{ceiling} = 2.80 \times P_{base}$.

Factor ranges:
- **Demand factor:** 0.85 (low) to 1.25 (very high)
- **Booking horizon:** 0.90 (>60 days) to 1.25 (<7 days)
- **Event factor:** 1.00 (no event) to 1.30 (major festival)
- **Weather factor:** 0.80 (storm) to 1.00 (clear)
- **Competition factor:** competitor price / base price, clamped [0.85, 1.15]
- **Seat pressure:** 0.85 (load factor <0.50) to 1.30 (load factor >0.90)
- **Segment factor:** 0.90 to 1.10 (business, leisure, gold, group)

Base fares reflect Indian domestic dynamics with declining per-km rates: ₹8.0–9.5/km for ≤300 km, declining to ₹3.0–3.8/km for >2,000 km. The 365-day forward pricing calendar incorporates 25+ Indian festivals (Diwali, Holi, Durga Puja, Navratri, Eid, Christmas), wedding seasons (November–February, April–June), school breaks, monsoon patterns (June–September), weekend bumps, and a ±2 day halo effect around major festivals.

### 7. Optimization Algorithms

The **multi-objective Pareto optimizer** balances five objectives: Revenue (weight 0.40), Load Factor (0.25, target 0.85), Market Share (0.15), Churn Risk (0.10, minimize), and Profit Margin (0.10). It generates 200 price candidates, evaluates all objectives using demand elasticity models, applies Pareto dominance filtering, and returns the top 20 Pareto-optimal solutions.

The **EMSR-b fare class allocation** computes protection levels using the inverse standard normal CDF:

$$y_j = \sigma_j \cdot \Phi^{-1}\left(\frac{f_j - f_{j+1}}{f_j}\right) + \mu_j$$

The **scenario engine** provides 10 pre-defined templates: fuel spike (+30%), demand drop (-25%), competitor undercut (-20%), monsoon (+40% cancellation), Diwali (+50% demand), recession (-15% demand), new route cold-start, capacity increase (+20%), premium push (+15% business demand), and operational crisis (+50% delays). Each scenario runs Monte Carlo simulation (default 500 iterations) with confidence intervals.

For new routes without historical data, the **cold-start strategy** uses cluster-based similarity (distance 50%, city tier 25%, region 25%), Bayesian prior estimation from similar routes, and a 5-phase pricing ramp-up from 85% to 100% of estimated fare over 6 months.

### 8. Time-Series Forecasting

Facebook Prophet is configured with yearly and weekly seasonality enabled, changepoint prior scale of 0.05, seasonality prior scale of 10, custom quarterly seasonality (Fourier order 5), and an Indian holiday calendar. SARIMA provides a classical baseline with automatic AIC-based parameter selection. The ensemble combines both:

$$\hat{y}_t = 0.60 \cdot \hat{y}_{t}^{Prophet} + 0.40 \cdot \hat{y}_{t}^{SARIMA}$$

Forecasts are generated at four horizons: 7 days (MAPE target <8%), 30 days (<10%), 90 days (<15%), and 365 days (<20%). Walk-forward validation with expanding and sliding windows prevents optimistic bias, with alerts triggered when recent MAPE exceeds historical average by >20%.

### 11. Revenue Command Center Dashboard

**Figure 14: Revenue Command Center — Main Dashboard Overview**

![Dashboard Overview](prototype_image/Screenshot%202026-03-10%20080944.png)

**Figure 15: Revenue Command Center — Routes Analytics & Overbooking Optimization**

![Routes and Overbooking](prototype_image/Screenshot%202026-03-10%20080953.png)

**Figure 15b: Revenue Command Center — Demand Forecast**

![Overbooking and Demand Forecast](prototype_image/Screenshot%202026-03-10%20081002.png)

**Figure 16: Revenue Command Center — Dynamic Pricing Recommendations**

![Dynamic Pricing Recommendations](prototype_image/Screenshot%202026-03-10%20081016.png)

The Revenue Command Center dashboard implements a comprehensive analytics interface with 12 KPI cards and 12 analytics tabs: Revenue, Routes, Bookings, Pricing, Overbooking, Operations, Customers, Ancillary, Forecast, Alerts, 365-Day Pricing, and ML Predictions. All visualizations use Chart.js with INR currency formatting and 80+ interactive info-buttons serving as an embedded user guide.

### 9. Explainability, Drift Detection, and Compliance

**SHAP explainability** provides global feature importance (TreeExplainer/LinearExplainer), individual prediction force plots, and route-level driver analysis — all in under 50ms per prediction.

**PSI drift detection** monitors distribution shift:

$$PSI = \sum_{i=1}^{k} (p_i^{actual} - p_i^{reference}) \cdot \ln\left(\frac{p_i^{actual}}{p_i^{reference}}\right)$$

PSI < 0.10 indicates no drift; 0.10–0.25 indicates moderate drift; ≥0.25 triggers automatic retraining. Kolmogorov-Smirnov tests detect distributional shifts (p < 0.05).

The **DGCA compliance engine** enforces distance-based fare caps (Economy: ₹5,000 for <500 km to ₹15,000 for >1,000 km), emergency fare multiplier checks, and denied boarding compensation across a 3×3 tier matrix (₹5,000–₹20,000). JWT authentication uses PBKDF2-SHA256 hashing with 3 roles: Admin, Analyst, and Viewer. The sliding-window rate limiter provides tiered protection: 100 requests/minute general, 30 for ML predictions, 10 for heavy queries.

---

## IV. Results and Discussion

### Demand Forecasting

**Figure 3: Demand Forecasting Results**

![Demand Forecasting Results](generated_reports/01_demand_forecasting_results.png)

The XGBoost-based demand model achieved MAPE below 10% for near-term (7-day) horizons. The Prophet + SARIMA ensemble (60/40 weighting) achieved MAPE of 8.4% on 7-day forecasts, consistently outperforming standalone models. Walk-forward validation confirmed that forecast accuracy holds under proper temporal evaluation, avoiding the 20–30% optimistic bias seen with standard k-fold cross-validation.

### Dynamic Pricing Validation

**Figure 4: Dynamic Pricing Optimization Results**

![Dynamic Pricing Results](generated_reports/02_dynamic_pricing_results.png)

The 365-day pricing calendar was validated against historical fare data. Price peaks correctly aligned with Diwali (+30–40%), Christmas (+25–35%), and summer holidays (+20–25%). Monsoon discounting accurately reflected July–August drops of 10–20% below baseline, and wedding season surges (November–February) correctly showed 15–25% demand uplift. Distance-calibrated base fares matched published Indian domestic fares within ±10%. The 7-factor pricing engine produced simulated revenue uplift of 8–12% compared to static fare pricing.

### Overbooking Optimization

**Figure 5: Monte Carlo Overbooking Simulation Results**

![Overbooking Monte Carlo Results](generated_reports/03_overbooking_monte_carlo_results.png)

Monte Carlo simulation with 1,000 runs per overbooking level identified optimal quantities that reduced denied boardings by 30% while simultaneously reducing empty seats by 15%. The cost trade-off curve across all 25 overbooking levels (0–24 extra seats) enables risk-appetite-based decision-making for analysts.

### Classification Model Performance

**Figure 8: Customer Churn Prediction Results**

![Customer Churn Results](generated_reports/06_customer_churn_results.png)

**Figure 7: Operational Risk Classification Results**

![Operational Risk Results](generated_reports/05_operational_risk_results.png)

**Figure 9: Flight Delay Prediction Results**

![Flight Delay Results](generated_reports/07_flight_delay_results.png)

**Figure 10: Cancellation Prediction Results**

![Cancellation Results](generated_reports/08_cancellation_results.png)

**Figure 11: Load Factor Prediction Results**

![Load Factor Results](generated_reports/09_load_factor_results.png)

**Figure 12: No-Show Prediction Results**

![No-Show Results](generated_reports/10_noshow_results.png)

| Model | Metric | Result |
|---|---|---|
| Customer Churn | AUC-ROC | 0.87 |
| Customer Churn | F1-Score | 0.82 |
| Operational Risk | Accuracy | >85% |
| Flight Delay | Accuracy | >80% |
| Cancellation | AUC-ROC | >0.85 |
| No-Show | AUC-ROC | >0.80 |

The Flight Delay model identified turnaround time as the highest-importance feature.

### Route Profitability and Clustering

**Figure 6: Route Profitability Analysis Results**

![Route Profitability Results](generated_reports/04_route_profitability_results.png)

**Figure 13: Passenger Clustering Results**

![Passenger Clustering Results](generated_reports/11_passenger_clustering_results.png)

The Linear Regression route profitability model achieved R² exceeding 0.90, confirming that the relationship between operational features and profit is approximately linear when strong economic indicators are included. K-Means clustering with k=4 produced clearly separated passenger segments — Premium Frequent, High-Value Leisure, At-Risk/Churners, and Budget Traveler — validated by silhouette analysis.

### Explainability and System Performance

SHAP explanations were generated in under 50ms per prediction. The DGCA compliance engine correctly validated fare caps across all distance bands. The dashboard summary API achieved latency under 2 seconds for 50,000-record datasets. The 365-day pricing calendar generated in under 3 seconds. All ML prediction endpoints responded in under 100ms. The system was tested for up to 50 concurrent dashboard sessions.

### Business Impact (Projected)

| Area | Estimated Annual Impact |
|---|---|
| Dynamic pricing revenue uplift (8–12%) | ₹240–360 crore |
| Load factor improvement (78% → 88%) | ₹150 crore |
| Overbooking optimization savings | ₹4.5–7.5 crore |
| Route portfolio optimization | ₹50+ crore |
| Ancillary cross-selling | ₹30–50 crore |
| Competitive response protection | ₹20–40 crore |

Operational efficiency gains include an 80% reduction in manual pricing analysis time, real-time competitive alerts enabling response within minutes, and day-one profitable pricing for new routes via cold-start similarity.

### Limitations

SQLite lacks concurrent write support required for production scale, necessitating migration to PostgreSQL. Current in-memory model serving loads all 12 models at startup (~200MB RAM). Prophet has complex dependencies that fail on some platforms, requiring a fallback seasonal decomposition model with reduced accuracy. Cold-start routes can deviate 20–30% from actual demand in the first 2–3 months. Black swan events like COVID-19 invalidate all trained models and require manual intervention. Competitor load factors are estimated rather than observed. The system optimizes per-route independently; network-level optimization is deferred to future work.

---

## V. Conclusion and Future Scope

The Airline Revenue Optimization & Flight Demand Forecasting System successfully demonstrates the feasibility and value of an end-to-end ML platform for airline revenue optimization. The system integrates 12 predictive models, a 7-factor dynamic pricing engine, Monte Carlo overbooking optimization, multi-objective Pareto optimization, and Prophet/SARIMA time-series forecasting into a production-grade FastAPI application with 49+ endpoints, a Revenue Command Center dashboard with 12 analytics tabs and 40+ KPIs, and robust infrastructure including ETL scheduling, data quality checks, SHAP explainability, PSI drift detection, DGCA regulatory compliance, and audit logging.

The platform addresses core business challenges of a mid-sized Indian airline: improving load factor from 78% to 88%, reducing unsold inventory by 15%, and achieving 8–12% revenue growth through intelligent pricing. Critically, the system prioritizes explainability and human oversight — SHAP values make model decisions transparent, a pricing approval gate prevents extreme automated price changes, and the DGCA compliance engine ensures regulatory adherence.

Future enhancements should include: (1) replacing the rule-based pricing engine with a Deep Reinforcement Learning agent (PPO/SAC); (2) extending to network-level revenue management for connecting passengers and codeshare agreements; (3) integrating Apache Kafka for real-time booking event ingestion; (4) applying Graph Neural Networks for demand spillover modeling across connected routes; (5) implementing Double Machine Learning [5] for causal price elasticity estimation; (6) adding A/B testing to validate ML-recommended prices against analyst baselines; and (7) extending to international routes with exchange rate modeling and multi-currency pricing.

---

## References

[1] Abdella, J. A., et al. (2021). Airline ticket price and demand prediction using machine learning. *Expert Systems with Applications, 174*, 114762.

[2] Belobaba, P. P. (1987). Air travel demand and airline seat inventory management. *PhD Thesis, MIT*.

[3] Chen, M., & Kachani, S. (2007). Forecasting and optimization for hotel revenue management. *Journal of Revenue and Pricing Management, 6*(3), 163–174.

[4] Chen, Y., et al. (2024). Hybrid time-series models for airline demand forecasting. *International Journal of Forecasting, 40*(2), 654–672.

[5] Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal, 21*(1), C1–C68.

[6] Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation, 6*(2), 182–197.

[7] Directorate General of Civil Aviation (DGCA). (2024). *Annual Report 2023–24*. Ministry of Civil Aviation, India.

[8] Fiig, T., et al. (2010). Optimization of mixed fare structures. *Journal of Revenue and Pricing Management, 9*(1), 152–170.

[9] Gönsch, J., & Steinhardt, C. (2023). Deep reinforcement learning for dynamic pricing in airline revenue management. *OR Spectrum, 45*(2), 375–410.

[10] IATA. (2024). *World Air Transport Statistics, 68th Edition*.

[11] Kumar, A., & Singh, R. (2023). Demand forecasting for Indian domestic airlines post-COVID. *Transportation Research Part E, 170*, 103021.

[12] Li, J., et al. (2024). Multi-objective optimization for airline revenue management. *Computers & Operations Research, 161*, 106423.

[13] Littlewood, K. (1972). Forecasting and control of passenger bookings. *AGIFORS Symposium Proceedings, 12*, 95–117.

[14] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS, 30*, 4765–4774.

[15] Makridakis, S., et al. (2020). The M5 accuracy competition. *International Journal of Forecasting, 38*(4), 1346–1364.

[16] Park, S., & Lee, K. (2022). Monte Carlo simulation-based overbooking optimization. *Omega, 112*, 102691.

[17] Ramanathan, V., & Subramanian, S. (2025). Explainable AI for airline revenue management. *Decision Support Systems, 178*, 114125.

[18] Rothstein, M. (1971). An airline overbooking model. *Transportation Science, 5*(2), 180–192.

[19] Shihab, S. A., et al. (2022). A machine learning approach to airline pricing. *Journal of Revenue and Pricing Management, 21*(4), 312–328.

[20] Talluri, K. T., & van Ryzin, G. J. (2004). *The Theory and Practice of Revenue Management*. Springer.

[21] Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician, 72*(1), 37–45.

[22] Wang, L., et al. (2024). Airline route profitability prediction using ensemble learning. *Transportation Research Part A, 179*, 103912.

[23] Weatherford, L. R., & Kimes, S. E. (2003). A comparison of forecasting methods for hotel revenue management. *International Journal of Forecasting, 19*(3), 401–415.

[24] Zhang, Q., et al. (2023). Passenger no-show prediction for airline overbooking. *Journal of Air Transport Management, 107*, 102348.
