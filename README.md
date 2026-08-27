# REP — Security Incident Grade Prediction

REP is a machine-learning project that predicts the grade of a security incident from Microsoft security-alert data. The project converts raw incident records into model-ready features, trains classification models, evaluates their performance, stores the best model, and exposes the prediction through a Streamlit user interface.

The target variable is `IncidentGrade`, with the following classes:

- `BenignPositive`
- `FalsePositive`
- `TruePositive`

---

## 1. Problem Statement

Security operations teams receive a large number of alerts and incidents. Reviewing every alert manually is slow and can lead to inconsistent triage decisions. The objective of this project is to build a classification system that can automatically estimate whether an incident is benign, a false positive, or a true positive.

### Why this problem matters

- It reduces repetitive manual investigation.
- It helps analysts prioritize likely true-positive incidents.
- It provides a consistent first-level classification.
- It can support faster incident response and better use of SOC resources.

### Machine-learning objective

Given the available incident attributes, learn a function:

\[
f(X) \rightarrow \text{IncidentGrade}
\]

where `X` contains incident, detector, entity, operating-system, location, and time-based features.

---

## 2. Project Goals

This project is designed to:

1. Load and inspect the NYC security-incident dataset.
2. Clean missing and duplicate records.
3. Create useful time-based features from `Timestamp`.
4. Encode the target and categorical input variables.
5. Train baseline, Random Forest, and XGBoost classifiers.
6. Compare the models using accuracy, multiclass ROC-AUC, and macro F1 score.
7. Save the trained Random Forest model and preprocessing artifacts.
8. Provide a Streamlit interface for interactive predictions.

---

## 3. Repository Structure

```text
REP/
├── README.md
├── .gitignore
├── requirements.txt
├── check_imports.py
├── data/
│   ├── NYC_Train.csv
│   ├── NYC_Test.csv
│   └── raw/
│       └── NYC_Train_raw.csv
├── feature_store/
│   └── feature_store.db
├── features/
│   ├── build_feature.py
│   └── build_feature1.py
├── model_store/
│   ├── feature_columns.json
│   ├── rf_model.pkl
│   ├── feature_encoder.pkl
│   └── target_encoder.pkl
├── serving/
│   └── api.py
├── training/
│   ├── train_model.py
│   ├── train_model_from_store.py
│   └── mlflow_train_model.py
├── ui/
│   └── app.py
├── validation/
│   └── validate_data.py
└── .venv-1/
```

### Folder responsibilities

- `data/`: Raw and processed project datasets.
- `feature_store/`: SQLite feature store containing the model feature table.
- `features/`: Centralized feature-engineering module used as the single source of truth.
- `training/`: Model-training scripts, including the feature-store training version and the MLflow-tracked version.
- `model_store/`: Serialized model, encoders, and feature schema used during prediction.
- `ui/`: Streamlit application for interactive predictions.
- `serving/`: Reserved API-serving layer.
- `validation/`: Input-schema validation checks for the raw source dataset.
- `requirements.txt`: Python package dependencies.
- `check_imports.py`: Dependency verification script.

---

## 4. Technology Stack

- Python 3.10+
- Pandas — data loading and transformation
- NumPy — numerical operations
- Scikit-learn — preprocessing, models, metrics, and validation split
- XGBoost — gradient-boosted classification model
- Joblib — model serialization
- Streamlit — interactive user interface
- Matplotlib and Seaborn — charting and evaluation plots
- SQLite — feature-store persistence
- MLflow — experiment tracking, parameters, tags, metrics, and model logging
- FastAPI and Uvicorn — reserved API-serving dependencies

---

## 5. Environment Setup

The commands below are for Windows PowerShell and were used from the project root.

### Step 1: Open the project directory

```powershell
cd "C:\Kapil\BITS Mini Project\REP"
```

### Step 2: Activate the existing virtual environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
(& "C:\Kapil\BITS Mini Project\REP\.venv-1\Scripts\Activate.ps1")
```

### Step 3: Install or upgrade dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Verify imports

```powershell
python check_imports.py
```

### Step 5: Validate raw data before training

```powershell
& "C:\Kapil\BITS Mini Project\REP\.venv-1\Scripts\python.exe" "C:\Kapil\BITS Mini Project\REP\validation\validate_data.py" --data "data\NYC_Train.csv"
```

This validation step checks required columns, timestamps, missing target values, and non-negative fields before model training.

---

## 6. Current Feature-Store Workflow

The project now uses a centralized feature-engineering flow:

1. Build the feature store from the raw CSV.
2. Train models from the SQLite feature store rather than directly from raw CSV files.
3. Log metrics and metadata to MLflow.
4. Run Streamlit for interactive prediction.

### Build the feature store

```powershell
& "C:\Kapil\BITS Mini Project\REP\.venv-1\Scripts\python.exe" "C:\Kapil\BITS Mini Project\REP\features\build_feature.py"
```

This creates the SQLite table in `feature_store/feature_store.db` and stores the model-ready feature set.

### Train from the feature store

```powershell
& "C:\Kapil\BITS Mini Project\REP\.venv-1\Scripts\python.exe" "C:\Kapil\BITS Mini Project\REP\training\train_model_from_store.py"
```

This script loads feature rows from the feature store, splits them into train/validation sets, compares baseline, Random Forest, and XGBoost models, and saves the trained artifacts.

### Train with MLflow tracking

```powershell
& "C:\Kapil\BITS Mini Project\REP\.venv-1\Scripts\python.exe" "C:\Kapil\BITS Mini Project\REP\training\mlflow_train_model.py"
```

This version records:

- model parameters
- experiment tags such as `git_commit`, `data_file`, and `data_md5`
- ROC-AUC, accuracy, and macro-F1 metrics
- MLflow model artifacts for the trained Random Forest and XGBoost models

### Open the MLflow UI

```powershell
mlflow ui
```

Or if you want to run it as a Python module:

```powershell
python -m mlflow ui
```

---

## 7. Commands Used in This Workspace

These are the commands that were used during the setup and execution of this project in the current environment.

```powershell
# Activate the environment
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& "c:\Kapil\BITS Mini Project\REP\.venv-1\Scripts\Activate.ps1")

# Validate raw data
& "C:\Kapil\BITS Mini Project\REP\.venv-1\Scripts\python.exe" "C:\Kapil\BITS Mini Project\REP\validation\validate_data.py" --data "data\NYC_Train.csv"

# Syntax check the MLflow training script
& "c:/Kapil/BITS Mini Project/REP/.venv-1/Scripts/python.exe" -m py_compile "training/mlflow_train_model.py"

# Run the MLflow training workflow
& "c:/Kapil/BITS Mini Project/REP/.venv-1/Scripts/python.exe" "c:/Kapil/BITS Mini Project/REP/training/mlflow_train_model.py"

# Build the feature store
& "c:/Kapil/BITS Mini Project/REP/.venv-1/Scripts/python.exe" "c:/Kapil/BITS Mini Project/REP/features/build_feature.py"

# Run the Streamlit app from the project root
streamlit run app.py

# Or run via the environment explicitly
& "c:\Kapil\BITS Mini Project\REP\.venv-1\Scripts\python.exe" -m streamlit run ui\app.py
```

---

## 8. Step-by-Step Data and Model Implementation

### Step 1: Load the data

The training script loads `data/NYC_Train.csv` using Pandas. It prints the shape, schema, data types, and summary information so that the input data can be understood before modeling.

### Step 2: Inspect missing values

The script counts missing values in every column. The target column, `IncidentGrade`, cannot be used for supervised training when it is missing, so rows without a target value are removed.

Other missing feature values are handled later during feature preparation.

### Step 3: Remove duplicates

Duplicate records can give a model repeated evidence for the same observation. Duplicate rows are identified and removed to reduce data leakage and improve the reliability of the evaluation.

### Step 4: Create time features

The original `Timestamp` field is converted into useful numerical fields, including:

- `Hour`
- `DayOfWeek`
- `Month`
- `IsWeekend`

These features allow the model to learn time-related patterns without using the raw timestamp string directly.

### Step 5: Remove unsuitable or high-cardinality identifiers

Several identifier columns are dropped because they are mainly record IDs or can create noise and leakage. The remaining columns are used as model features.

### Step 6: Encode the target

`LabelEncoder` converts the text labels in `IncidentGrade` into numeric class IDs so that the classifiers can process them.

The target encoder is saved as `model_store/target_encoder.pkl`. It is required to convert a predicted numeric class back to its original label.

### Step 7: Split the data

The dataset is divided into training and validation sets using an 80/20 split. `stratify=y` preserves the class distribution in both sets.

```python
train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)
```

### Step 8: Encode categorical features

Categorical columns are transformed with `OrdinalEncoder`.

The encoder is configured as follows:

- Known categories receive numeric values.
- Unknown categories receive `-1`.
- Missing values are filled with `0` after encoding.

The fitted encoder is saved as `model_store/feature_encoder.pkl` so that the UI uses the same transformation as training.

### Step 9: Train models

The training script fits three models:

1. `DummyClassifier` — a simple baseline.
2. `RandomForestClassifier` — the model saved for application use.
3. `XGBClassifier` — a boosted-tree comparison model.

The Random Forest uses balanced class weights to reduce the impact of class imbalance.

### Step 10: Evaluate the models

The models are compared using:

- Accuracy
- Multiclass ROC-AUC with one-vs-rest (`multi_class="ovr"`)
- Macro F1 score
- Classification report
- Confusion matrix

For multiclass ROC-AUC, the model must provide class probabilities with `predict_proba()`, not class labels from `predict()`.

```python
y_probability_rf = rf_model.predict_proba(X_val)
rf_roc_score = roc_auc_score(
    y_val,
    y_probability_rf,
    multi_class="ovr",
)
```

### Step 11: Save model artifacts

After training, the following files are created in `model_store/`:

- `rf_model.pkl`: Trained Random Forest classifier.
- `feature_encoder.pkl`: Fitted categorical feature encoder.
- `target_encoder.pkl`: Fitted target-label encoder.
- `feature_columns.json`: Exact feature order expected by the model.

Saving the feature order and encoders is important because inference must use the same preprocessing logic as training.

---

## 9. Train the Models

From the project root, the main training patterns are:

```powershell
# feature-store training path
& "C:\Kapil\BITS Mini Project\REP\.venv-1\Scripts\python.exe" "C:\Kapil\BITS Mini Project\REP\training\train_model_from_store.py"

# MLflow-enabled training path
& "C:\Kapil\BITS Mini Project\REP\.venv-1\Scripts\python.exe" "C:\Kapil\BITS Mini Project\REP\training\mlflow_train_model.py"
```

The MLflow script logs metrics and artifacts to the local MLflow tracking server, while the feature-store variant retains the same baseline, Random Forest, and XGBoost evaluation logic.

Run training before starting the UI if the model files do not already exist.

---

## 10. Run the Streamlit Application

Start the user interface from the project root:

```powershell
streamlit run ui\app.py
```

Or use the virtual-environment executable explicitly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui\app.py
```

Streamlit normally opens a browser automatically. If it does not, open:

```text
http://localhost:8501
```

### How to use the UI

1. Enter the incident and detector attributes.
2. Enter categorical values such as `Category`, `EntityType`, and `EvidenceRole`.
3. Set the time and location values.
4. Select **Predict incident grade**.
5. Review the predicted incident grade.
6. Review the probability for each class.
7. Download the prediction as a JSON file if required.

Unknown categorical values are handled by the saved encoder and mapped to the configured unknown-category value.

---

## 11. Optional Streamlit Commands

Run on a different port:

```powershell
streamlit run ui\app.py --server.port 8502
```

Run without automatically opening a browser:

```powershell
streamlit run ui\app.py --server.headless true
```

Stop the running application with:

```text
Ctrl+C
```

---

## 12. Inference Flow

The prediction process follows this sequence:

```text
User input
   ↓
Create one-row DataFrame
   ↓
Reorder columns using feature_columns.json
   ↓
Convert numeric values
   ↓
Apply feature_encoder.pkl
   ↓
Fill missing values
   ↓
rf_model.pkl → predict() and predict_proba()
   ↓
Decode class using target_encoder.pkl
   ↓
Display grade and probabilities
```

This sequence prevents a common production problem: training and inference using different feature orders or different category encodings.

---

## 13. Expected Model Results

On the current validation split, the project produced results approximately in the following range:

| Model | Accuracy | ROC-AUC | Macro F1 |
|---|---:|---:|---:|
| Baseline | 0.43 | 0.50 | 0.20 |
| Random Forest | 0.78 | 0.92 | 0.77 |
| XGBoost | 0.78 | 0.93 | 0.77 |

Exact values can change if the dataset, dependency versions, random seeds, or preprocessing logic changes.

---

## 14. Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and reinstall dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Missing model artifacts

Run training first:

```powershell
python training\train_model.py
```

Confirm that `model_store/` contains the model, encoders, and feature schema files.

### Streamlit is not recognized

Use the module form:

```powershell
python -m streamlit run ui\app.py
```

### Port already in use

Use another port:

```powershell
python -m streamlit run ui\app.py --server.port 8502
```

### ROC-AUC shape error

For multiclass ROC-AUC, use probability output:

```python
model.predict_proba(X)
```

Do not pass the one-dimensional output from `model.predict(X)` to `roc_auc_score` when there are multiple target classes.

### Different Python environments

Make sure training and Streamlit use the same environment. Check the executable with:

```powershell
python -c "import sys; print(sys.executable)"
```

If multiple environments exist, run both commands with the same explicit interpreter, for example:

```powershell
.\.venv\Scripts\python.exe training\train_model.py
.\.venv\Scripts\python.exe -m streamlit run ui\app.py
```

---

## 15. Future Improvements

- Add a complete FastAPI endpoint in `serving/api.py`.
- Add request validation and structured response schemas.
- Add model versioning and experiment tracking with MLflow.
- Add automated unit tests for preprocessing and inference.
- Add batch CSV prediction.
- Add monitoring for data drift and prediction distributions.
- Save the training configuration and evaluation metrics with each model version.
- Add authentication and access control before deploying the application.

---

## 16. Important Note

This model is a decision-support tool, not a replacement for security analysts. Predictions should be reviewed together with the underlying evidence, organizational policies, and incident-response procedures before taking action.

---

## 17. Reflection Questions

The following answers apply to the current REP implementation, which trains a Random Forest classifier in `training/train_model.py` and performs inference in `ui/app.py` using saved model artifacts.

### Technical Reflection

#### 1. Why is the feature encoding logic written twice—once in `train_model.py` and once in `api.py` or `app.py`? What risk does this create?

The logic exists in two places because training and prediction happen at different times and in different execution contexts:

- During training, the data is converted into the numerical format expected by the model.
- During prediction, a new user or API record must be converted into exactly the same format before it is passed to the model.

In REP, the training code fits an `OrdinalEncoder` on the training data. The UI then loads the saved `feature_encoder.pkl` and applies it to new input. The UI also loads `feature_columns.json` so that the feature order matches the order used during training.

Duplicating preprocessing rules creates a serious risk of training-serving skew. For example, if training treats a category as value `3` but inference uses a different mapping, the model receives a different meaning for the same number. Differences in missing-value handling, column order, dropped columns, or data types can also produce incorrect predictions without an obvious runtime error.

The safer design is to place preprocessing in one reusable pipeline, save the fitted pipeline with the model, and use that same pipeline for both training and inference. REP partially addresses this risk by saving the fitted feature encoder, target encoder, and feature schema, but the feature construction rules should eventually be centralized as well.

#### 2. What happens when a new category, such as `PayPal`, appears in production traffic? Why does no error appear?

The REP encoder is configured with `handle_unknown="use_encoded_value"` and `unknown_value=-1`. Therefore, if a category was not present when the model was trained, the encoder converts it to `-1` instead of raising an exception.

This prevents the application from crashing, but it does not mean that the model understands the new category. The Random Forest sees an unknown numeric code and applies the behavior learned for that encoded value. The result may therefore be less accurate, especially if the new category is common or has a meaning that differs from the training categories.

No error appears because unknown categories are deliberately accepted by the encoder. Production should still record these values, monitor their frequency, and trigger a review or retraining process when unknown-category rates increase.

#### 3. If you wanted to retrain this model tomorrow with the same accuracy, what information would you need? Is that information captured anywhere?

At minimum, the following information would be required:

1. The same version of `NYC_Train.csv`, or a precisely defined replacement dataset.
2. The target definition and the exact class labels in `IncidentGrade`.
3. The columns that were dropped and the feature-engineering rules for `Timestamp`.
4. The missing-value and duplicate-removal rules.
5. The categorical and target encoding configuration.
6. The train/validation split settings, including `random_state=42`, `test_size=0.2`, and stratification.
7. The model type and hyperparameters for the Random Forest and comparison models.
8. The dependency versions and Python environment.
9. The evaluation code and the validation metrics.

Some of this information is captured in `training/train_model.py`, `requirements.txt`, `feature_columns.json`, and the saved encoder artifacts. The README documents the workflow and commands. However, the project does not yet capture a dataset version or hash, a complete configuration file, the exact installed package lockfile, or a formal experiment record. Consequently, exact reproduction of the same accuracy is not guaranteed if the data or environment changes.

#### 4. The model always returns a number between 0 and 1, even for garbage inputs. Why is this dangerous in a production setting?

`predict_proba()` returns the model's estimated class probabilities for any row that has the expected shape and data types. It does not prove that the input is valid, realistic, complete, or similar to the training data.

For example, an out-of-range detector ID, nonsensical hour, invalid category combination, or entirely missing record may still produce probabilities. Users can incorrectly interpret those values as reliable confidence. A high probability for `TruePositive` does not necessarily mean the incident is genuinely a true positive; it may only mean that the input resembles one of the model's learned regions, or that the model is extrapolating poorly.

In production, REP should validate ranges and required fields, identify out-of-distribution inputs, expose model and data versions, log predictions, and provide an abstain or manual-review path when input quality is poor. Probabilities should be calibrated and described as model estimates, not guarantees.

### Engineering Mindset Reflection

#### 5. What is the difference between “the system works” and “the system is trustworthy”?

“The system works” means that the application starts, accepts an input, loads the model, and returns a prediction. The current Streamlit application satisfies this basic functional requirement.

“The system is trustworthy” requires much more evidence. It should also have:

- Reliable and reproducible preprocessing.
- Validated inputs and safe handling of missing or unknown values.
- Evaluation on representative, recent, and unseen data.
- Transparent metrics for every incident class.
- Monitoring for data drift, model drift, and service failures.
- Versioned models, datasets, encoders, and configurations.
- Logging, access control, auditability, and rollback procedures.
- A clear human-review process for uncertain or high-impact decisions.

A prediction appearing on the screen demonstrates functionality; it does not demonstrate correctness, fairness, stability, or operational safety.

#### 6. If the security team started using this system tomorrow, what would happen in one month when the incident population changes?

The distribution of incidents could change because of new threats, new devices, new applications, changes in security policies, or seasonal activity. This is data drift and concept drift.

The model would continue producing predictions because the saved Random Forest does not automatically know that the environment has changed. Its accuracy and class-level recall could decline, unknown categories could increase, and the model could become overconfident on patterns that were not present in the training data. A change in the class balance could also make the original accuracy misleading.

Without monitoring, the team might not discover the degradation until analysts notice a growing number of incorrect triage decisions. The system therefore needs prediction logging, input-distribution monitoring, unknown-category counts, periodic labeled evaluation, and a retraining policy.

#### 7. Which single addition would most improve this system's reliability? Justify your answer.

The single most valuable addition would be a production validation and monitoring layer around the saved model.

It should validate required fields and reasonable ranges before prediction, detect unknown and out-of-distribution values, log the model version and input summary, and monitor prediction distributions and later-confirmed outcomes. Invalid or uncertain records should be sent to manual review instead of being presented as normal high-confidence predictions.

This addition would improve reliability more than simply adding another model because the largest current risks are silent failures: training-serving skew, garbage inputs receiving plausible probabilities, and performance degradation after the incident population changes. Monitoring and validation make those failures visible and provide the evidence needed for safe retraining.
