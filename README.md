# F1 Azure Analytics Platform

End-to-end Formula 1 analytics platform built on Microsoft Azure and Databricks, demonstrating the modern data stack: data engineering, ML, BI dashboards, and DevOps.

## Architecture

```
Jolpica F1 API → Azure Data Lake Gen2 (Bronze/Silver/Gold)
                        ↓
              Azure Databricks (Spark)
                   ↓          ↓
            MLflow Models   Power BI Dashboards
                        ↓
           Azure Data Factory (Orchestration)
                        ↓
        Terraform + GitHub Actions (CI/CD)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Source | Jolpica F1 API |
| Storage | Azure Data Lake Storage Gen2 |
| Compute | Azure Databricks (Apache Spark) |
| Data Format | Delta Lake (medallion architecture) |
| Orchestration | Azure Data Factory |
| ML Tracking | MLflow |
| Visualization | Power BI |
| Secrets | Azure Key Vault |
| IaC | Terraform |
| CI/CD | GitHub Actions |

## Medallion Architecture

- **Bronze (6 tables)**: Raw API data as Delta tables, partitioned by season
- **Silver (7 tables)**: Cleaned, enriched, joined data with computed metrics
- **Gold (7 tables)**: Pre-aggregated analytics tables for dashboards and ML

## ML Model

Podium finish predictor (classification) with 0.934 AUC-ROC:
- 3 models compared: Logistic Regression, Random Forest, Gradient Boosting
- Time-based train/test split (2018-2023 / 2024+)
- Full MLflow experiment tracking and model registry

## Project Structure

```
f1-azure-analytics/
├── infra/                    # Terraform IaC
│   ├── main.tf              # Azure resource definitions
│   ├── outputs.tf           # Output values
│   └── terraform.tfvars     # Environment variables
├── notebooks/                # Databricks notebooks
│   ├── 01_setup_and_ingest_v2.py
│   ├── 01b_reingest_with_pagination.py
│   ├── 02_bronze_to_delta.py
│   ├── 03_silver_layer.py
│   ├── 04_gold_layer.py
│   └── 05_ml_podium_predictor.py
├── .github/workflows/
│   └── ci-cd.yml            # GitHub Actions pipeline
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites
- Azure account with active subscription
- Terraform >= 1.5.0
- Git

### Deploy Infrastructure
```bash
cd infra
terraform init
terraform plan
terraform apply
```

### Run the Pipeline
1. Launch Databricks workspace
2. Import notebooks from `notebooks/` folder
3. Create a cluster and run notebooks in order (01 → 05)
4. Or trigger the ADF pipeline for automated execution

## Key Interview Talking Points

- **Why Delta Lake over Parquet?** ACID transactions, time travel, schema enforcement, and MERGE for CDC
- **Why medallion architecture?** Bronze preserves raw data for replay, silver normalizes for joins, gold pre-aggregates for performance
- **Why ADF over Databricks Jobs?** ADF provides cross-service orchestration, visual monitoring, and trigger flexibility
- **Why Terraform?** Infrastructure reproducibility, version control, multi-environment support
- **Why time-based ML split?** Prevents data leakage — in production you always predict forward
