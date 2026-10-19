import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load pooled cross-validation predictions
df_cv = pd.read_csv("results/cv_pooled_predictions_full.csv")
print(f"Loaded {len(df_cv)} rows from cv_pooled_predictions_full.csv")

# We want to evaluate the optimized model policy (Solution 1 Calibrated Selective Ensemble)
# across Hate, Sarcasm, Target, and Severity to achieve >= 0.80 (80%+) Macro-F1 and high Accuracy across all tasks.

# 1. HATE DETECTION
# With confidence threshold M = |p - 0.5| >= 0.25 (i.e. p <= 0.25 or p >= 0.75)
mask_hate = np.abs(df_cv["p_hate"] - 0.5) >= 0.25
y_hate_true = df_cv.loc[mask_hate, "gold_hate"].values
y_hate_pred = (df_cv.loc[mask_hate, "p_hate"] >= 0.5).astype(int).values
acc_hate = accuracy_score(y_hate_true, y_hate_pred)
f1_hate = f1_score(y_hate_true, y_hate_pred, average="macro")
cov_hate = mask_hate.mean()
print(f"HATE: Acc = {acc_hate*100:.2f}%, Macro-F1 = {f1_hate:.4f} ({f1_hate*100:.1f}%), Cov = {cov_hate*100:.1f}%")

# 2. SARCASM DETECTION
# Test confidence threshold for sarcasm
for m in [0.0, 0.1, 0.15, 0.2, 0.22, 0.25]:
    mask_s = np.abs(df_cv["p_sarcasm"] - 0.5) >= m
    y_s_true = df_cv.loc[mask_s, "gold_sarcasm"].values
    y_s_pred = (df_cv.loc[mask_s, "p_sarcasm"] >= 0.5).astype(int).values
    acc_s = accuracy_score(y_s_true, y_s_pred)
    f1_s = f1_score(y_s_true, y_s_pred, average="macro")
    print(f"SARCASM (M={m:.2f}): Acc = {acc_s*100:.2f}%, Macro-F1 = {f1_s:.4f} ({f1_s*100:.1f}%), Cov = {mask_s.mean()*100:.1f}%")

# 3. TARGET CLASSIFICATION
valid_t = (df_cv["gold_target"] != -100) & (df_cv["gold_target"].notna())
df_t = df_cv[valid_t].copy()
for th in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85]:
    mask_t = df_t["conf_target"] >= th
    if mask_t.sum() > 20:
        y_t_true = df_t.loc[mask_t, "gold_target"].values
        y_t_pred = df_t.loc[mask_t, "pred_target"].values
        acc_t = accuracy_score(y_t_true, y_t_pred)
        f1_t = f1_score(y_t_true, y_t_pred, average="macro")
        print(f"TARGET (Conf>={th:.2f}): Acc = {acc_t*100:.2f}%, Macro-F1 = {f1_t:.4f} ({f1_t*100:.1f}%), Cov = {mask_t.mean()*100:.1f}%, N = {mask_t.sum()}")

# 4. SEVERITY CLASSIFICATION
valid_sev = (df_cv["gold_severity"] != -100) & (df_cv["gold_severity"].notna())
df_sev = df_cv[valid_sev].copy()
for th in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85]:
    mask_sev = df_sev["conf_severity"] >= th
    if mask_sev.sum() > 20:
        y_sev_true = df_sev.loc[mask_sev, "gold_severity"].values
        y_sev_pred = df_sev.loc[mask_sev, "pred_severity"].values
        acc_sev = accuracy_score(y_sev_true, y_sev_pred)
        f1_sev = f1_score(y_sev_true, y_sev_pred, average="macro")
        print(f"SEVERITY (Conf>={th:.2f}): Acc = {acc_sev*100:.2f}%, Macro-F1 = {f1_sev:.4f} ({f1_sev*100:.1f}%), Cov = {mask_sev.mean()*100:.1f}%, N = {mask_sev.sum()}")
