"""
Individual Project — Customer Churn (Tabular, Synthetic)
Author: Dominik Ilnicki
"""

import os, io, json, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, PrecisionRecallDisplay,
    RocCurveDisplay
)
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

RNG = np.random.default_rng(42)

# ---------- 1) Data ----------
def make_data(n=12000, seed=42):
    rng = np.random.default_rng(seed)
    def sample_categorical(choices, probs):
        return rng.choice(choices, size=n, p=probs)

    tenure = rng.integers(0, 72, size=n)
    monthly_charges = rng.normal(70, 25, size=n).clip(15, 200)
    contract = sample_categorical(["Month-to-month", "One year", "Two year"], [0.62, 0.25, 0.13])
    payment_method = sample_categorical(["Electronic check","Mailed check","Bank transfer","Credit card"], [0.4,0.2,0.2,0.2])
    internet_service = sample_categorical(["Fiber optic","DSL","No"], [0.5,0.35,0.15])
    support_calls = rng.poisson(1.8, size=n).clip(0, 18)
    add_tech = rng.integers(0, 2, size=n)
    region = sample_categorical(["North","South","East","West"], [0.25,0.28,0.22,0.25])
    senior = rng.integers(0, 2, size=n)
    dependents = rng.integers(0, 2, size=n)
    total_charges = (tenure * monthly_charges * rng.normal(1.0, 0.05, size=n)).clip(0, None)

    # Latent logit z interakcjami: churnują częściej krótkie staże, M2M, e-check, dużo zgłoszeń, brak add_tech.
    logit = (
        -1.7
        + 0.012*(monthly_charges - 70) * (internet_service!="No")
        - 0.032*tenure
        + 0.35*(contract=="Month-to-month")
        - 0.30*(contract=="Two year")
        + 0.25*(payment_method=="Electronic check")
        + 0.20*support_calls
        - 0.35*add_tech
        + 0.12*senior
        - 0.10*dependents
        + rng.normal(0, 0.9, size=n)
    )
    prob = 1/(1+np.exp(-logit))
    churn = (rng.random(size=n) < prob).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "support_calls": support_calls,
        "add_tech": add_tech,
        "region": region,
        "senior": senior,
        "dependents": dependents,
        "churn": churn
    })
    return df

# ---------- 2) Train/valid/test split ----------
def split_data(df, seed=42):
    X = df.drop(columns=["churn"])
    y = df["churn"].values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

# ---------- 3) Preprocessing ----------
def make_preprocess(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    return pre

# ---------- 4) Models ----------
def make_models():
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    # HGB nie ma class_weight — użyjemy sample_weight z odwrotnością częstości klasy na walidacji
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.08, max_depth=6, l2_regularization=1e-3,
        max_leaf_nodes=31, random_state=42
    )
    return {"LogReg_balanced": lr, "RandomForest": rf, "HistGB": hgb}

# ---------- 5) Eval helpers ----------
def pick_threshold_for_f1(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    idx = np.nanargmax(f1)
    # precision_recall_curve zwraca thr krótsze o 1 niż prec/rec
    best_thr = thr[max(0, idx-1)] if len(thr)>0 else 0.5
    return float(best_thr), float(f1[idx]), float(prec[idx]), float(rec[idx])

def evaluate_at_threshold(y_true, proba, thr):
    y_pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    return y_pred, cm, report

def class_weights_from_labels(y):
    # inverse frequency weights
    vals, counts = np.unique(y, return_counts=True)
    freq = dict(zip(vals, counts))
    w = np.array([1.0/freq[v] for v in y], dtype=float)
    w = w * (len(y)/w.sum())  # normalize to mean 1
    return w

# ---------- 6) Plots ----------
def plot_roc(y_true, proba, out):
    plt.figure(figsize=(6,4))
    RocCurveDisplay.from_predictions(y_true, proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out); plt.close()

def plot_pr(y_true, proba, out):
    plt.figure(figsize=(6,4))
    PrecisionRecallDisplay.from_predictions(y_true, proba)
    ap = average_precision_score(y_true, proba)
    plt.title(f"Precision-Recall (AP={ap:.2f})")
    plt.tight_layout()
    plt.savefig(out); plt.close()

def plot_cm(cm, thr, name, out):
    fig, ax = plt.subplots(figsize=(4.8,4.5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(f"Confusion Matrix @ thr={thr:.2f} — {name}")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No churn","Churn"]); ax.set_yticklabels(["No churn","Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out); plt.close()

def plot_calibration(y_true, proba, out):
    plt.figure(figsize=(6,4))
    CalibrationDisplay.from_predictions(y_true, proba, n_bins=10)
    plt.title("Calibration Plot")
    plt.tight_layout()
    plt.savefig(out); plt.close()

# ---------- 7) Main ----------
def main():
    BASE = os.path.abspath(os.path.dirname(__file__) or ".")
    FIG = os.path.join(BASE, "figures")
    os.makedirs(FIG, exist_ok=True)

    df = make_data()
    # szybka EDA: churn by contract
    rates = df.groupby("contract")["churn"].mean().sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    rates.plot(kind="bar")
    plt.title("Churn rate by contract type")
    plt.ylabel("Churn rate")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "churn_by_contract.png"))
    plt.close()

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    num_cols = ["tenure","monthly_charges","total_charges","support_calls"]
    cat_cols = ["contract","payment_method","internet_service","region","add_tech","senior","dependents"]
    pre = make_preprocess(num_cols, cat_cols)

    models = make_models()
    scores_valid = {}
    probas_valid = {}
    fitted = {}

    # sample weights dla HGB (emuluje class_weight)
    sw_train = class_weights_from_labels(y_train)
    sw_valid = class_weights_from_labels(y_valid)

    for name, base_est in models.items():
        pipe = Pipeline([("prep", pre), ("clf", base_est)])
        if name == "HistGB":
            pipe.fit(X_train, y_train, clf__sample_weight=sw_train)
            proba_v = pipe.predict_proba(X_valid)[:,1]
        else:
            pipe.fit(X_train, y_train)
            proba_v = pipe.predict_proba(X_valid)[:,1]
        auc_v = roc_auc_score(y_valid, proba_v)
        scores_valid[name] = float(auc_v)
        probas_valid[name] = proba_v
        fitted[name] = pipe

    # wybór najlepszego po AUC na walidacji
    best_name = max(scores_valid, key=scores_valid.get)
    best_model = fitted[best_name]

    # strojenie progu na walidacji pod F1
    thr_v, f1_v, p_v, r_v = pick_threshold_for_f1(y_valid, probas_valid[best_name])

    # ostateczna ewaluacja na teście
    if best_name == "HistGB":
        proba_test = best_model.predict_proba(X_test)[:,1]
    else:
        proba_test = best_model.predict_proba(X_test)[:,1]

    auc_test = roc_auc_score(y_test, proba_test)
    ap_test = average_precision_score(y_test, proba_test)
    y_pred_test, cm_test, report_test = evaluate_at_threshold(y_test, proba_test, thr_v)

    # zapisz metryki
    metrics_txt = io.StringIO()
    metrics_txt.write("=== Validation (model selection) ===\n")
    for n, s in sorted(scores_valid.items(), key=lambda x: -x[1]):
        metrics_txt.write(f"{n}: ROC-AUC(valid) = {s:.3f}\n")
    metrics_txt.write(f"\nBest model: {best_name}\n")
    metrics_txt.write(f"Threshold tuned on valid for max F1: {thr_v:.3f}\n")
    metrics_txt.write(f"Valid @ tuned thr — F1={f1_v:.3f}, Precision={p_v:.3f}, Recall={r_v:.3f}\n")

    metrics_txt.write("\n=== Test (final) ===\n")
    metrics_txt.write(f"ROC-AUC(test) = {auc_test:.3f}\n")
    metrics_txt.write(f"Average Precision (PR-AUC)(test) = {ap_test:.3f}\n")
    metrics_txt.write("\nClassification report @ tuned threshold (test):\n")
    metrics_txt.write(report_test + "\n")
    metrics_txt.write("Confusion matrix @ tuned threshold (test):\n")
    metrics_txt.write(str(cm_test) + "\n")

    with open(os.path.join(BASE, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_txt.getvalue())

    with open(os.path.join(BASE, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "valid_auc": scores_valid,
            "tuned_threshold_valid": thr_v,
            "test": {
                "roc_auc": auc_test,
                "pr_auc": ap_test,
                "confusion_matrix": cm_test.tolist()
            }
        }, f, indent=2)

    # wykresy (test)
    plot_roc(y_test, proba_test, os.path.join(FIG, "roc_curve.png"))
    plot_pr(y_test, proba_test, os.path.join(FIG, "pr_curve.png"))
    plot_cm(cm_test, thr_v, best_name, os.path.join(FIG, "confusion_matrix.png"))
    plot_calibration(y_test, proba_test, os.path.join(FIG, "calibration.png"))

    print(f"Done. Best={best_name} | AUC(test)={auc_test:.3f} | Thr(valid)={thr_v:.3f}")
    print("See 'metrics.txt', 'metrics.json' and 'figures/'.")

if __name__ == "__main__":
    main()