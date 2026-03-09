# Housing Price Prediction — End-to-End MLOps Pipeline

I built this project to go through the full lifecycle of a machine learning system — from raw data to a live deployed model on AWS. The goal was to predict US housing median list prices and practice every layer of an MLOps pipeline: data engineering, model development, productionization, containerization, and cloud deployment.

---

## What I Built

A complete end-to-end MLOps pipeline that:
- Cleans and engineers features from raw US housing data
- Trains and tunes an XGBoost model tracked with MLflow
- Serves predictions through a FastAPI REST API
- Visualizes results in a Streamlit dashboard
- Stores models and data in AWS S3
- Deploys automatically to AWS ECS via GitHub Actions CI/CD

---

## Full Pipeline Flow

```
Raw Data (untouched_raw_original.csv)
        │
        ▼
┌───────────────────┐
│  00. Data Split   │  Time-based split → Train / Eval / Holdout
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  01. EDA &        │  City normalization, geo merge, outlier removal
│      Cleaning     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  02. Feature      │  Date features, frequency encoding (zipcode),
│      Engineering  │  target encoding (city) — fitted on train only
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  03–04. Baseline Models                        │
│         Linear Regression → Ridge → Lasso      │
└────────────────────┬───────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────┐
│  05–06. XGBoost + Hyperparameter Tuning        │
│         15 Optuna trials tracked in MLflow     │
└────────────────────┬───────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────┐
│  07. Push to AWS S3                            │
│      Best model + processed data               │
└────────────────────┬───────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  ┌───────────────┐    ┌─────────────────────┐
  │  FastAPI API  │    │  Streamlit Dashboard │
  │  /predict     │    │  Predictions vs      │
  │  /run_batch   │    │  Actuals, MAE/RMSE   │
  └───────┬───────┘    └─────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────┐
│  Docker → GitHub Actions → AWS ECR → AWS ECS  │
└───────────────────────────────────────────────┘
```

---

## Project Structure

```
Regression_Model/
├── notebooks/                        # Exploratory work done in order
│   ├── 00_data_split.ipynb           # Time-based train/eval/holdout split
│   ├── 01_EDA_cleaning.ipynb         # EDA, city normalization, outlier removal
│   ├── 02_feature_eng_encoding.ipynb # Feature engineering & encoding
│   ├── 03_baseline.ipynb             # Linear regression baseline models
│   ├── 04_linear_regression_regularization.ipynb  # Ridge & Lasso
│   ├── 05_XGBoost.ipynb              # XGBoost model
│   ├── 06_hyperparameter_tuning_MFLow.ipynb       # Optuna + MLflow
│   └── 07_S3_push_datasets_AWS.ipynb # Push model & data to S3
│
├── src/
│   ├── feature_pipeline/
│   │   ├── load.py                   # Time-based data splitting
│   │   ├── preprocess.py             # Cleaning, city normalization, geo merge
│   │   └── feature_engineering.py   # Encoding, date features
│   ├── training_pipeline/
│   │   ├── train.py                  # Baseline XGBoost training
│   │   ├── eval.py                   # Model evaluation (MAE, RMSE, R²)
│   │   └── tune.py                   # Optuna hyperparameter tuning + MLflow
│   ├── inference_pipeline/
│   │   └── inference.py              # End-to-end prediction pipeline
│   ├── api/
│   │   └── main.py                   # FastAPI service
│   └── batch/
│       └── run_monthly.py            # Monthly batch inference runner
│
├── data/
│   ├── raw/                          # Original + time-split CSVs
│   ├── processed/                    # Cleaned & feature-engineered CSVs
│   └── predictions/                  # Monthly batch prediction outputs
│
├── models/
│   ├── xgb_best_model.pkl            # Tuned production model
│   ├── xgb_model.pkl                 # Baseline model
│   ├── freq_encoder.pkl              # Zipcode frequency encoder
│   └── target_encoder.pkl            # City target encoder
│
├── tests/                            # Unit & integration tests
├── configs/                          # App & MLflow config files
├── app.py                            # Streamlit dashboard
├── Dockerfile                        # FastAPI container
├── Dockerfile.streamlit              # Streamlit container
├── pyproject.toml                    # Dependencies (uv)
└── .github/workflows/ci.yml         # CI/CD pipeline
```

---

## Step 1 — Data Cleaning & EDA

The first thing I did was understand the raw data and get it into a usable state. The dataset had housing listing information across US metro areas — prices, zip codes, cities, and dates.

**What I did:**
- Split the data by time to avoid data leakage: train (before 2020), eval (2020–2022), holdout (2022+). Using a random split here would have let future data leak into training, which would give falsely optimistic results.
- Explored price distributions across metros and identified heavily skewed data
- Found a major issue with city names — the same city appeared in dozens of different formats (`new york`, `New-York`, `NewYork`, `new york city`). I wrote manual correction mappings and normalization logic to unify them
- Merged a US metros reference file to attach latitude and longitude to each record
- Removed exact duplicates and dropped extreme outliers (listings above $19M) that would distort the model

**Difficulties:**
- The city name inconsistency was the most painful part. There was no automated way to handle all the edge cases — I had to manually map corrections for dozens of cities. One mismatch here causes the target encoder to generate a new unknown category at inference time, which breaks predictions silently.
- Deciding where to draw the outlier cutoff was tricky. Setting it too low removes real expensive markets (NYC, SF); too high and you're training on data points the model will never generalize from.

---

## Step 2 — Feature Engineering & Encoding

After cleaning, I needed to turn the remaining categorical and temporal columns into numeric features the model could use — without leaking information from the eval or holdout sets.

**What I did:**
- Extracted `year`, `quarter`, and `month` from the date column to give the model temporal awareness
- Applied **frequency encoding** to zipcode: replaced each zip with how often it appears in the training set. This captures how well-represented each area is without creating thousands of one-hot columns
- Applied **target encoding** to city: replaced each city with the mean `median_list_price` from the training set. This lets the model understand price levels by city without treating it as a raw string
- Saved both encoders (`freq_encoder.pkl`, `target_encoder.pkl`) so inference uses the exact same transformations as training

**Difficulties:**
- The biggest risk with target encoding is leakage — if you fit it on the full dataset, you're embedding future price information into the features. I had to be careful to fit only on the training split and then apply (transform only) to eval and holdout.
- At inference time, unseen cities or zip codes need a fallback value. Handling these unknown categories gracefully without crashing the API took some iteration.

---

## Step 3 — Baseline Models

Before jumping to a complex model, I trained simple linear models to establish a performance floor. This gives a meaningful comparison point — if XGBoost barely beats linear regression, that tells you something.

**What I did:**
- Trained a plain Linear Regression as the true baseline
- Tried Ridge (L2 regularization) and Lasso (L1 regularization) to see if penalizing large coefficients helped
- Evaluated all models on MAE, RMSE, and R²

**Difficulties:**
- Linear models struggled with the non-linear relationships in the data. Price doesn't scale linearly with location or time — housing markets have complex interactions that linear models can't capture. The R² scores were mediocre, which made it clear I needed a tree-based model.

---

## Step 4 — XGBoost Model

With a baseline established, I moved to XGBoost as my main model.

**What I did:**
- Trained a baseline XGBoost with standard hyperparameters (500 estimators, lr=0.05, max_depth=6)
- Evaluated against the same metrics as the linear models
- XGBoost significantly outperformed the linear baselines, confirming the non-linear structure in the data

**Difficulties:**
- XGBoost with default settings can overfit on tabular data with high-cardinality features. The encoded city and zipcode features had lots of unique values and the model needed regularization to not memorize training patterns.

---

## Step 5 — Hyperparameter Tuning with Optuna + MLflow

With XGBoost confirmed as the right model class, I ran a proper hyperparameter search to find the best configuration.

**What I did:**
- Used **Optuna** to run 15 trials searching over 9 hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- Logged every trial to **MLflow** — each run tracked the hyperparameters used, the RMSE/MAE/R² achieved, and the trained model artifact
- The best trial was registered in the MLflow model registry as `best_xgb_model`

**Difficulties:**
- Getting MLflow and Optuna to integrate cleanly took some work. Optuna runs trials in its own loop and MLflow needs a run context — I had to nest the MLflow run inside each Optuna trial callback carefully so experiments didn't bleed into each other.
- 15 trials felt like a reasonable tradeoff between search quality and time, but with a larger search space some trials landed in clearly bad regions. Pruning would have helped here.
<img width="2150" height="2122" alt="Screenshot 2026-03-08 at 4 23 03 PM" src="https://github.com/user-attachments/assets/9ecdf447-7158-4e74-9eed-e8c659aff8fe" />

<img width="2114" height="2160" alt="Screenshot 2026-03-08 at 4 22 42 PM" src="https://github.com/user-attachments/assets/d25fd238-6e4f-42ca-9334-33cb977725a1" />


---

## Step 6 — Converting Notebooks to Production Code

Once I was happy with the model in notebooks, I rewrote everything as proper Python modules under `src/`. This was about making the code reusable, testable, and deployable rather than keeping it in a one-off notebook format.

**What I built:**
- `src/feature_pipeline/` — load, preprocess, and feature_engineering as separate importable modules
- `src/training_pipeline/` — train, eval, and tune functions with clean interfaces
- `src/inference_pipeline/inference.py` — a single end-to-end function: raw input → preprocessing → encoding → schema alignment → predictions
- `src/batch/run_monthly.py` — groups holdout data by year/month and runs inference on each period, saving results to `data/predictions/`

**Difficulties:**
- The inference pipeline had to replicate the exact same transformations as training — same encoders, same feature order, same column drops. Any mismatch between training and inference causes silent prediction errors. I spent time making sure the saved encoders were loaded and applied identically, and added schema alignment (reindex to training columns) as a safeguard.

---

## Step 7 — FastAPI + Streamlit

With the code modularized, I built a REST API to serve predictions and a dashboard to explore them.

**FastAPI** (`src/api/main.py`) — on startup it downloads the model and training features from S3 if not already cached locally.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Model status |
| `POST` | `/predict` | Batch prediction — list of records → predicted prices |
| `POST` | `/run_batch` | Trigger monthly batch inference |
| `GET` | `/latest_predictions` | Retrieve latest prediction file |

<img width="726" height="88" alt="Screenshot 2026-03-08 at 8 55 21 PM" src="https://github.com/user-attachments/assets/2fb2dd3c-7614-4521-a7bf-f4165b8d5fc4" />


**Streamlit** (`app.py`) pulls holdout data from S3, calls the FastAPI `/predict` endpoint, and displays predictions vs actuals with MAE, RMSE, and % error metrics. Users can filter by year, month, and region.

**Difficulties:**
- The API needed to be stateless but also fast — loading a model from S3 on every request would be too slow. I added local caching so the model is downloaded once on startup and reused for all subsequent requests.
- Getting the Streamlit app to talk to the FastAPI service across Docker containers required configuring the `API_URL` environment variable properly — localhost doesn't work when services are in separate containers.

---

## Step 8 — Push Model & Data to AWS S3

Before deployment, I pushed everything the deployed services would need up to S3.

**What I uploaded:**
- `models/xgb_best_model.pkl` → `model-regression-data` bucket
- `processed/feature_engineered_train.csv` → used by the API for schema alignment
- `processed/feature_engineered_holdout.csv` → used by the Streamlit dashboard

**Difficulties:**
- The FastAPI and Streamlit services use different S3 buckets and different AWS regions in the current setup (`us-east-2` vs `eu-west-2`). This was something I'd unify in a cleaner version but worked for the scope of this project.

---

## Step 9 — Docker

I containerized both services so they can run anywhere without environment setup.

- **`Dockerfile`** — builds the FastAPI service, exposes port 8000, runs with `uvicorn`
- **`Dockerfile.streamlit`** — builds the Streamlit dashboard, exposes port 8501, uses `--platform=$BUILDPLATFORM` for cross-architecture compatibility (M1/M2 Mac → Linux on AWS)

Both use `uv` for fast, reproducible dependency installation from `pyproject.toml`.

**Difficulties:**
- Cross-architecture builds were a headache. Building on an Apple Silicon Mac and deploying to AWS Linux (x86) required the `--platform` flag and multi-arch build support. Without it, the containers silently failed on ECS.

---

## Step 10 — CI/CD Pipeline with GitHub Actions

I set up a fully automated deployment pipeline so every push to `main` builds, pushes, and deploys both services without manual steps.

**Pipeline jobs:**
1. **Build & push `housing-api`** — builds Docker image, tags with `$GITHUB_SHA` and `latest`, pushes to AWS ECR
2. **Build & push `housing-streamlit`** — same process for the dashboard image
3. **Deploy API** — triggers ECS service update to pull the new image
4. **Deploy Streamlit** — same for the dashboard service

AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) are stored as GitHub secrets.

<img width="1708" height="734" alt="Screenshot 2026-03-08 at 8 55 12 PM" src="https://github.com/user-attachments/assets/350dd7e3-7bf1-4193-867a-eb4ce7610d27" />


**Difficulties:**
- The first few pipeline runs failed because of IAM permission issues. The GitHub Actions role didn't have the right policies to push to ECR or update ECS services. I had to create and attach the correct IAM policies, which required understanding the AWS permission model for cross-service access.
- ECS `update-service` triggers a rolling deployment but doesn't wait for the new task to become healthy before the pipeline completes. Early on I thought deployments succeeded when they hadn't — added health check monitoring to verify.

---

## Step 11 — AWS Infrastructure

### S3 — Model & Data Storage
Two buckets store everything the deployed services need at runtime, so containers stay lightweight.

### ECR — Container Registry
Both Docker images live in ECR and are tagged per-commit for rollback capability.

<img width="1460" height="606" alt="Screenshot 2026-03-08 at 8 55 58 PM" src="https://github.com/user-attachments/assets/90180e6d-5f4a-4dfa-8008-3fce743ed54a" />



### IAM Roles
I created custom roles to give ECS tasks the minimum permissions needed to read from S3:

<img width="1576" height="860" alt="Screenshot 2026-03-08 at 8 55 37 PM" src="https://github.com/user-attachments/assets/716b4cc1-3a03-4c70-bc03-9cd938ef82bc" />


### ECS Cluster — `regression-model-cluster-for-project`
Two services running in the same cluster, both Active:

| Service | Role |
|---------|------|
| `regression-model-cluster-for-project-service-07233mgp` | FastAPI prediction API |
| `housing-streamlit-service-5cvxvvhd` | Streamlit dashboard |

<img width="1526" height="1756" alt="Screenshot 2026-03-08 at 8 56 18 PM" src="https://github.com/user-attachments/assets/3746af61-db7f-429e-a39e-c6a2dd0a7140" />

### Application Load Balancer
An internet-facing ALB (`housing-price-prediction`) routes incoming traffic across two availability zones (us-east-2a, us-east-2b) to the ECS tasks.


<img width="1612" height="1530" alt="Screenshot 2026-03-08 at 8 56 44 PM" src="https://github.com/user-attachments/assets/07797732-a8b7-4dea-b41b-440e0edcefff" />

**Difficulties with AWS setup:**
- Setting up the ECS task definitions to inject AWS credentials as environment variables (so the containers can access S3) without hardcoding them took several iterations through IAM roles and task execution policies.
- The ALB target group health checks initially failed because the FastAPI startup takes a few seconds to download the model from S3. I had to increase the health check grace period so ECS didn't kill the task before it was ready.
<img width="1214" height="788" alt="Screenshot 2026-03-09 at 12 07 17 AM" src="https://github.com/user-attachments/assets/f5e9e730-d240-4187-9ddd-b61dc9e0f6f7" />

<img width="1180" height="742" alt="Screenshot 2026-03-09 at 12 07 30 AM" src="https://github.com/user-attachments/assets/02a06f1f-8279-4b67-b3a7-6c85603bc193" />
---

## Tech Stack

| Category | Tools |
|----------|-------|
| **ML / Modeling** | XGBoost, Scikit-learn, LightGBM |
| **Experiment Tracking** | MLflow, Optuna |
| **Feature Engineering** | category-encoders, Pandas, Polars |
| **API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit |
| **Cloud Storage** | AWS S3 (boto3) |
| **Containerization** | Docker |
| **Container Registry** | AWS ECR |
| **Orchestration** | AWS ECS (Fargate) |
| **Load Balancing** | AWS ALB |
| **CI/CD** | GitHub Actions |
| **Package Manager** | uv |
| **Testing** | pytest |
| **Data Quality** | Great Expectations, Evidently |
