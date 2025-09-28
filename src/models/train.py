#!/usr/bin/env python3
# Minimal training script (same as notebook)
import argparse, os, json, joblib
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import mlflow, mlflow.sklearn

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", ohe)])
    pre = ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)], remainder="drop")
    return pre

def train_and_eval(train_path, test_path, output_path, experiment="tourism-package-prediction"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target = "ProdTaken"
    features = [c for c in train_df.columns if c != target]
    X_train = train_df[features]; y_train = train_df[target].astype(int)
    X_test = test_df[features]; y_test = test_df[target].astype(int)
    pre = build_preprocessor(X_train)
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    pipeline = Pipeline([("preproc", pre), ("clf", clf)])
    param_grid = {"clf__n_estimators":[100,200],"clf__max_depth":[6,12],"clf__class_weight":[None,"balanced"]}
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="rf-gridsearch"):
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        probs = best.predict_proba(X_test)[:,1]
        roc = float(roc_auc_score(y_test, probs))
        report = classification_report(y_test, (probs>=0.5).astype(int), output_dict=True)
        cm = confusion_matrix(y_test, (probs>=0.5).astype(int)).tolist()
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("test_roc_auc", roc)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, output_path)
        mlflow.sklearn.log_model(best, artifact_path="sklearn-best-model")
        metrics = {"test_roc_auc":roc,"best_params":grid.best_params_,"confusion_matrix":cm,"classification_report":report}
        metrics_path = os.path.join(Path(output_path).parent, "metrics.json")
        with open(metrics_path,"w") as f:
            json.dump(metrics,f,indent=2)
    print("Saved model to", output_path)
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--experiment", default="tourism-package-prediction")
    args = p.parse_args()
    train_and_eval(args.train, args.test, args.output, args.experiment)
