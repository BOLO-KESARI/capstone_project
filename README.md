# Airline Revenue Management Platform

> Enterprise-grade airline revenue management system with ML-powered dynamic pricing, demand forecasting, and regulatory compliance.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Dashboard)                       │
│  Bootstrap 5.3 + Chart.js 4.4 + Vanilla JS                 │
├─────────────────────────────────────────────────────────────┤
│                    FastAPI Backend (main.py)                  │
│  49+ API endpoints │ JWT Auth │ Rate Limiting                │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│   ML     │  Optim.  │  ETL     │  Reg.    │  Reports        │
│ Pipeline │  Engine  │ Scheduler│ Compliance│ Generator       │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│             SQLite (Star Schema) + CSV Sim Data              │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Data Engineering
- **Star Schema Database** — 6 dimension tables, 3 fact tables, materialized views
- **ETL Pipeline** — APScheduler with 7 automated jobs (demand agg, route revenue, load factor, competitor refresh, weather sync, sentiment batch, model retrain check)
- **Data Quality Engine** — 9 validation rules, spike detection, timezone reconciliation
- **CSV Migration** — Automated migration from simulated CSV/JSON data

### Machine Learning (12+ Models)
- **Demand Forecasting** — XGBoost + Prophet + SARIMA ensemble
- **Dynamic Pricing** — Rule-based + ML-optimized pricing engine
- **Overbooking Optimization** — No-show prediction + DGCA compliance
- **Route Profitability** — Multi-factor route scoring
- **Customer Churn** — Behavioral prediction with SHAP explanations
- **Flight Delay / Cancellation** — Operational risk models
- **Load Factor Prediction** — Capacity optimization
- **Passenger Clustering** — K-Means segmentation (5 segments)
- **Time Series Forecasting** — Prophet + SARIMA + multi-horizon ensemble (7/30/90/365 days)

### MLOps
- **Walk-Forward Validation** — Time-series cross-validation with MAPE tracking
- **Model Drift Detection** — PSI + KS-test for data drift, performance degradation monitoring
- **Model Registry** — Version tracking, promotion/retirement, artifact persistence
- **SHAP Explainability** — Feature importance, prediction explanations, route drivers

### Optimization
- **Multi-Objective Optimizer** — Pareto-optimal pricing (revenue + load factor + market share + churn)
- **EMSR-b Fare Class Allocation** — Seat inventory optimization across fare classes
- **Scenario Simulator** — 10 pre-built scenarios + Monte Carlo simulation + sensitivity analysis
- **Cold Start Strategy** — Cluster-based similarity for new routes, Bayesian priors, ramp-up plans

### Security & Compliance
- **JWT Authentication** — RBAC with admin/analyst/viewer roles
- **Rate Limiting** — Per-IP sliding window, tiered limits, X-RateLimit headers
- **DGCA Fare Compliance** — Fare caps, denied boarding compensation, overbooking limits
- **Audit Logging** — Immutable trail for all pricing decisions and model events

### Reporting
- **Automated Reports** — Weekly revenue, monthly summary, route performance, model perf, pricing audit
- **Export** — JSON reports saved to `generated_reports/`

## Quick Start

```bash
# 1. Clone & install
pip install -r requirements.txt

# 2. Initialize database
python -c "from database.schema import init_database; init_database()"
python -c "from database.migrate import run_full_migration; run_full_migration()"

# 3. Start server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. Open dashboard
# http://localhost:8000/dashboard
```

## API Endpoints

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| Dashboard | `/dashboard` | GET | Revenue Command Center |
| Health | `/health` | GET | System health check |
| Revenue | `/api/revenue-summary` | GET | Revenue analytics |
| ML | `/api/predict/{model}` | POST | ML model predictions |
| Pricing | `/api/pricing/365` | POST | 365-day pricing engine |
| Scenarios | `/api/scenarios` | GET/POST | What-if analysis |
| Reports | `/api/reports/generate` | POST | Generate reports |
| Auth | `/api/auth/login` | POST | JWT authentication |
| Audit | `/api/audit/log` | GET | Audit trail |
| DQ | `/api/data-quality/run` | POST | Data quality checks |
| ETL | `/api/etl/status` | GET | ETL scheduler status |
| Drift | `/api/models/drift` | GET | Model drift status |

## Project Structure

```
├── main.py                     # FastAPI app (49+ endpoints, 12 ML models)
├── aviation_client.py           # AviationStack API client
├── config.py                   # API key configuration
├── database/
│   ├── schema.py               # SQLite star schema
│   └── migrate.py              # CSV → SQLite migration
├── data_quality/
│   └── rules.py                # 9 DQ validation rules
├── etl/
│   └── scheduler.py            # APScheduler ETL jobs
├── ml_advanced/
│   ├── time_series.py          # Prophet + SARIMA
│   ├── shap_explainer.py       # SHAP model explanations
│   ├── validation.py           # Walk-forward validation
│   ├── drift_detector.py       # Data/concept drift detection
│   └── model_registry.py       # Model versioning
├── optimization/
│   ├── scenario_engine.py      # What-if simulator
│   ├── multi_objective.py      # Pareto optimizer
│   └── cold_start.py           # New route strategy
├── middleware/
│   ├── rate_limiter.py         # Rate limiting
│   └── auth.py                 # JWT + RBAC
├── regulatory/
│   ├── compliance.py           # DGCA fare rules
│   └── audit_log.py            # Audit logging
├── reports/
│   └── generator.py            # Report automation
├── simulated_data/             # CSV simulation files
├── db_simulation/              # RDBMS/NoSQL/OLAP simulation
├── new_models/                 # ML training datasets
├── static/                     # Frontend assets
├── templates/                  # Jinja2 HTML templates
├── tests/                      # Test suite
├── .github/workflows/ci.yml    # CI/CD pipeline
└── requirements.txt            # Python dependencies
```

## Default Credentials (Development)

| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Full access |
| analyst | analyst123 | ML + pricing |
| viewer | viewer123 | Read-only |

## Technology Stack

- **Backend**: FastAPI, Flask
- **ML**: scikit-learn, XGBoost, Prophet, SARIMA, SHAP
- **Database**: SQLite (production: MySQL/PostgreSQL)
- **Scheduling**: APScheduler
- **Frontend**: Bootstrap 5.3, Chart.js 4.4
- **CI/CD**: GitHub Actions

## License

Internal use only. All rights reserved.
