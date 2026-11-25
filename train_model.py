
# ============================================
# PHISHING WEBSITE DETECTION SYSTEM
# MCA PBL PROJECT - SECTION E
# ============================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import arff

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer
)

warnings.filterwarnings("ignore")

# ============================================
# STEP 1: LOAD DATASET
# ============================================

print("\n================================")
print("PHISHING WEBSITE DETECTION")
print("================================")

print("\nLoading Dataset...")

data, meta = arff.loadarff(
    "dataset/Training Dataset.arff"
)

df = pd.DataFrame(data)

# Convert byte values into normal values
for col in df.columns:
    df[col] = df[col].apply(
        lambda x: x.decode("utf-8")
        if isinstance(x, bytes) else x
    )

# Convert columns into numeric format
for col in df.columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

print("\nDataset Loaded Successfully!")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nDataset Description:")
print(df.describe())

# ============================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================

print("\n================================")
print("EXPLORATORY DATA ANALYSIS")
print("================================")

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nTarget Class Distribution:")
print(df["Result"].value_counts())

# Class Distribution Graph
plt.figure(figsize=(7, 5))

sns.countplot(
    x="Result",
    data=df
)

plt.title("Phishing vs Legitimate Websites")
plt.xlabel("Website Class (-1: Phishing, 1: Legitimate)")
plt.ylabel("Number of Websites")

plt.tight_layout()
plt.savefig("dataset/class_distribution.png")
plt.close()

# Feature Correlation Heatmap
plt.figure(figsize=(16, 10))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig("dataset/correlation_heatmap.png")
plt.close()

print("\nEDA Completed Successfully!")

# ============================================
# STEP 3: DATA PREPROCESSING
# ============================================

print("\n================================")
print("DATA PREPROCESSING")
print("================================")

# Remove Duplicate Records
df = df.drop_duplicates()

# Separate Input Features and Target
X = df.drop("Result", axis=1)
y = df["Result"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ============================================
# STEP 4: TRAIN TEST SPLIT
# ============================================

print("\n================================")
print("TRAIN TEST SPLIT")
print("================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ============================================
# STEP 5: DEFINE THREE ML MODELS
# ============================================

print("\n================================")
print("MODEL TRAINING")
print("================================")

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            random_state=42
        ))
    ])
}

# ============================================
# STEP 6: MODEL EVALUATION
# ============================================

print("\n================================")
print("MODEL EVALUATION")
print("================================")

results = []

# Phishing class is -1
phishing_recall = make_scorer(
    recall_score,
    pos_label=-1,
    zero_division=0
)

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        pos_label=-1,
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        pos_label=-1,
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        pos_label=-1,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[-1, 1]
        )
    )

# Model Comparison Table
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# ============================================
# STEP 7: 5-FOLD CROSS VALIDATION
# ============================================

print("\n================================")
print("5-FOLD CROSS VALIDATION")
print("================================")

cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=phishing_recall,
        n_jobs=-1
    )

    cv_results[name] = scores

    print("\nModel:", name)
    print("Five Validation Scores:", scores)
    print("Average CV Score:", scores.mean())

# ============================================
# STEP 8: HYPERPARAMETER TUNING
# ============================================

print("\n================================")
print("GRID SEARCH CV")
print("================================")

param_grids = {

    "Logistic Regression": {
        "classifier__C": [0.1, 1, 10]
    },

    "Decision Tree": {
        "classifier__max_depth": [None, 10, 20]
    },

    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 20]
    }
}

best_models = {}
tuning_results = []

for name, model in models.items():

    print("\nTuning:", name)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=5,
        scoring=phishing_recall,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_models[name] = grid_search.best_estimator_

    tuning_results.append({
        "Model": name,
        "Best Parameters": grid_search.best_params_,
        "Best CV Recall": grid_search.best_score_
    })

    print("Best Parameters:")
    print(grid_search.best_params_)

    print("Best CV Recall:")
    print(grid_search.best_score_)

# ============================================
# STEP 9: SELECT BEST MODEL
# ============================================

print("\n================================")
print("BEST MODEL SELECTION")
print("================================")

best_model_name = max(
    tuning_results,
    key=lambda x: x["Best CV Recall"]
)["Model"]

best_model = best_models[best_model_name]

print("\nSelected Best Model:", best_model_name)

# ============================================
# STEP 10: FINAL MODEL EVALUATION
# ============================================

print("\n================================")
print("FINAL TEST EVALUATION")
print("================================")

final_predictions = best_model.predict(X_test)

print("\nAccuracy:",
      accuracy_score(y_test, final_predictions))

print("\nPrecision:",
      precision_score(
          y_test,
          final_predictions,
          pos_label=-1,
          zero_division=0
      ))

print("\nRecall:",
      recall_score(
          y_test,
          final_predictions,
          pos_label=-1,
          zero_division=0
      ))

print("\nF1 Score:",
      f1_score(
          y_test,
          final_predictions,
          pos_label=-1,
          zero_division=0
      ))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        final_predictions,
        labels=[-1, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    )
)

print("\nFinal Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        final_predictions,
        labels=[-1, 1]
    )
)

# ============================================
# STEP 11: SAVE BEST MODEL
# ============================================

print("\n================================")
print("SAVING BEST MODEL")
print("================================")

os.makedirs("model", exist_ok=True)

joblib.dump(
    best_model,
    "model/best_phishing_model.joblib"
)

print("\nBest Model Saved Successfully!")

print("\nSaved File:")
print("model/best_phishing_model.joblib")

# ============================================
# PROJECT COMPLETED
# ============================================

print("\n================================")
print("PROJECT TRAINING COMPLETED")
print("================================")

print("Best Model:", best_model_name)
print("Dataset Records:", len(df))
print("Features:", X.shape[1])

print("\nPhishing Website Detection System Ready!")
