# Telecom Customer Churn Prediction & Retention System

## Project Overview
This project is an end-to-end Machine Learning solution designed to predict telecom customer churn and generate actionable, ROI-positive retention leads. Going beyond simple classification, the system calculates the probability of churn, evaluates the expected revenue loss using Customer Lifetime Value (CLTV), and recommends targeted discount offers to maximize Net ROI.

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework, transitioning from exploratory data analysis in Jupyter Notebooks to a modular, object-oriented production pipeline in Python.

##  Project Structure

The repository is divided into two main components: **Exploration/Development** (Jupyter Notebooks) and the **Production Pipeline** (Python scripts).

### Exploratory & Development Notebooks (`.ipynb`)
* `data_understanding.ipynb`: Initial data exploration and integration of external data (web scraping).
* `data_preparation.ipynb`: Feature engineering, handling missing values, and encoding categorical variables.
* `model.ipynb`: Model selection and hyperparameter tuning (Logistic Regression, Decision Trees, Random Forest) and identifying the top business drivers of churn.
* `evalution.ipynb`: Evaluation of the model against technical metrics and business logic (proving positive Net ROI for retention campaigns).
* `Deployment.ipynb`: Simulation of the monthly batch prediction process and pipeline finalization.

### Production Pipeline Modules (`.py` in `src/`)
* `data_enrichment.py` (`DataScraper`): Fetches live population data from Wikipedia to enrich the base dataset.
* `data_preparation.py` (`DataPreprocessor`): Automates data cleaning, handling nulls (mode/median imputation), dropping identifiers, and one-hot encoding.
* `model_training.py` (`ModelTrainer`): Constructs and trains a scikit-learn `Pipeline` utilizing a tuned `RandomForestClassifier` (`max_depth=10`, `class_weight='balanced'`) to prevent overfitting.
* `evaluation.py` (`ModelEvaluator`): Generates key technical metrics including Precision, Recall, and F1-Score.
* `deployment.py` (`BatchPredictor`): The business engine. It identifies high-risk customers (>75% churn probability), applies a tiered discount strategy (5%, 10%, 15%), calculates expected loss via CLTV, and outputs a prioritized retention list.

##  Methodology & Business Logic

1.  **Data Enrichment:** We enhance internal customer data with external demographic data (state populations) via web scraping (`BeautifulSoup`).
2.  **Robust Modeling:** A Random Forest model is utilized to capture complex, non-linear relationships. Class imbalance is handled natively via balanced class weights.
3.  **Business-Driven Output:** Instead of just outputting `1` (Churn) or `0` (Stay), the deployment script:
    * Filters for customers with a **>75% churn probability**.
    * Assigns a variable **Discount Rate** based on the severity of the churn risk.
    * Calculates the **Annual Offer Cost** vs. the **Expected Loss** (Probability $\times$ CLTV).
    * Ranks the final leads by **Net ROI**, ensuring the retention team calls the most valuable, at-risk customers first.

