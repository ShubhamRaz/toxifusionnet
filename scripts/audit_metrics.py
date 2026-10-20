"""Reproduction script for Stage 3 Tri-Dataset global test metrics."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

# Search candidates for preds_test.csv
candidates = [
    Path(__file__).resolve().parent.parent / "results" / "stage3_tri_dataset" / "preds_test.csv",
    Path(__file__).resolve().parent.parent / "runs" / "exp_tri_dataset_xlmr" / "preds_test.csv",
    Path("results/stage3_tri_dataset/preds_test.csv"),
    Path("runs/exp_tri_dataset_xlmr/preds_test.csv"),
]
preds_path = next((p for p in candidates if p.exists()), None)
if preds_path is None:
    raise FileNotFoundError("Could not locate preds_test.csv in results/stage3_tri_dataset or runs/exp_tri_dataset_xlmr")

print(f"Loading test predictions from: {preds_path}")
df = pd.read_csv(preds_path)

print("=== 1. HATE SPEECH DETECTION REPRODUCTION ===")
mask_hate = df["gold_hate"] != -100
df_hate = df[mask_hate]
y_true_hate = df_hate["gold_hate"].astype(int)
y_pred_hate = df_hate["pred_hate"].astype(int)
y_prob_hate = df_hate["p_hate"]

cm_hate = confusion_matrix(y_true_hate, y_pred_hate)
tn, fp, fn, tp = cm_hate.ravel()
print(f"Hate Evaluated Samples: {len(df_hate)}")
print(f"Hate Confusion Matrix:\n  TN={tn}, FP={fp}\n  FN={fn}, TP={tp}")
acc_hate = accuracy_score(y_true_hate, y_pred_hate)
bal_acc_hate = balanced_accuracy_score(y_true_hate, y_pred_hate)
f1_hate = f1_score(y_true_hate, y_pred_hate, average="macro")
auc_hate = roc_auc_score(y_true_hate, y_prob_hate)
prec_hate = precision_score(y_true_hate, y_pred_hate, average="macro")
rec_hate = recall_score(y_true_hate, y_pred_hate, average="macro")
weighted_f1_hate = f1_score(y_true_hate, y_pred_hate, average="weighted")

print(f"Accuracy:          {acc_hate:.6f}")
print(f"Balanced Accuracy: {bal_acc_hate:.6f}")
print(f"Macro Precision:   {prec_hate:.6f}")
print(f"Macro Recall:      {rec_hate:.6f}")
print(f"Macro-F1:          {f1_hate:.6f}")
print(f"Weighted F1:       {weighted_f1_hate:.6f}")
print(f"AUROC:             {auc_hate:.6f}")

true_fnr_hate = fn / (fn + tp)
error_rate_hate = 1.0 - acc_hate
print(f"True Miss Rate / FNR [FN / (FN+TP)]: {true_fnr_hate:.6f} ({true_fnr_hate*100:.2f}%)")
print(f"Overall Error Rate [1 - Acc]:        {error_rate_hate:.6f} ({error_rate_hate*100:.2f}%)")

print("\n=== 2. SARCASM DETECTION REPRODUCTION ===")
mask_sarc = df["gold_sarcasm"] != -100
df_sarc = df[mask_sarc]
y_true_sarc = df_sarc["gold_sarcasm"].astype(int)
y_pred_sarc = df_sarc["pred_sarcasm"].astype(int)
y_prob_sarc = df_sarc["p_sarcasm"]

cm_sarc = confusion_matrix(y_true_sarc, y_pred_sarc)
tn_s, fp_s, fn_s, tp_s = cm_sarc.ravel()
print(f"Sarcasm Evaluated Samples: {len(df_sarc)}")
print(f"Sarcasm Excluded (-100):   {len(df) - len(df_sarc)}")
print(f"Sarcasm Confusion Matrix:\n  TN={tn_s}, FP={fp_s}\n  FN={fn_s}, TP={tp_s}")
acc_sarc = accuracy_score(y_true_sarc, y_pred_sarc)
bal_acc_sarc = balanced_accuracy_score(y_true_sarc, y_pred_sarc)
f1_sarc = f1_score(y_true_sarc, y_pred_sarc, average="macro")
auc_sarc = roc_auc_score(y_true_sarc, y_prob_sarc)
prec_sarc = precision_score(y_true_sarc, y_pred_sarc, average="macro")
rec_sarc = recall_score(y_true_sarc, y_pred_sarc, average="macro")
weighted_f1_sarc = f1_score(y_true_sarc, y_pred_sarc, average="weighted")

print(f"Accuracy:          {acc_sarc:.6f}")
print(f"Balanced Accuracy: {bal_acc_sarc:.6f}")
print(f"Macro Precision:   {prec_sarc:.6f}")
print(f"Macro Recall:      {rec_sarc:.6f}")
print(f"Macro-F1:          {f1_sarc:.6f}")
print(f"Weighted F1:       {weighted_f1_sarc:.6f}")
print(f"AUROC:             {auc_sarc:.6f}")

true_fnr_sarc = fn_s / (fn_s + tp_s)
error_rate_sarc = 1.0 - acc_sarc
print(f"True Miss Rate / FNR [FN / (FN+TP)]: {true_fnr_sarc:.6f} ({true_fnr_sarc*100:.2f}%)")
print(f"Overall Error Rate [1 - Acc]:        {error_rate_sarc:.6f} ({error_rate_sarc*100:.2f}%)")

print("\n=== 3. TARGET IDENTIFICATION REPRODUCTION ===")
mask_target = df["gold_target"] != -100
df_target = df[mask_target]
y_true_target = df_target["gold_target"].astype(int)
y_pred_target = df_target["pred_target"].astype(int)

cm_target = confusion_matrix(y_true_target, y_pred_target)
print(f"Target Evaluated Samples: {len(df_target)}")
print("Target Confusion Matrix (classes 0:individual, 1:org, 2:community, 3:society):\n", cm_target)
acc_target = accuracy_score(y_true_target, y_pred_target)
bal_acc_target = balanced_accuracy_score(y_true_target, y_pred_target)
f1_target = f1_score(y_true_target, y_pred_target, average="macro")
prec_target = precision_score(y_true_target, y_pred_target, average="macro")
rec_target = recall_score(y_true_target, y_pred_target, average="macro")
weighted_f1_target = f1_score(y_true_target, y_pred_target, average="weighted")

print(f"Accuracy:          {acc_target:.6f}")
print(f"Balanced Accuracy: {bal_acc_target:.6f}")
print(f"Macro Precision:   {prec_target:.6f}")
print(f"Macro Recall:      {rec_target:.6f}")
print(f"Macro-F1:          {f1_target:.6f}")
print(f"Weighted F1:       {weighted_f1_target:.6f}")

print("\n=== 4. SEVERITY PREDICTION REPRODUCTION ===")
mask_sev = df["gold_severity"] != -100
df_sev = df[mask_sev]
y_true_sev = df_sev["gold_severity"].astype(int)
y_pred_sev = df_sev["pred_severity"].astype(int)

cm_sev = confusion_matrix(y_true_sev, y_pred_sev)
print(f"Severity Evaluated Samples: {len(df_sev)}")
print("Severity Confusion Matrix (classes 0:low, 1:moderate, 2:high):\n", cm_sev)
acc_sev = accuracy_score(y_true_sev, y_pred_sev)
bal_acc_sev = balanced_accuracy_score(y_true_sev, y_pred_sev)
f1_sev = f1_score(y_true_sev, y_pred_sev, average="macro")
prec_sev = precision_score(y_true_sev, y_pred_sev, average="macro")
rec_sev = recall_score(y_true_sev, y_pred_sev, average="macro")
weighted_f1_sev = f1_score(y_true_sev, y_pred_sev, average="weighted")

print(f"Accuracy:          {acc_sev:.6f}")
print(f"Balanced Accuracy: {bal_acc_sev:.6f}")
print(f"Macro Precision:   {prec_sev:.6f}")
print(f"Macro Recall:      {rec_sev:.6f}")
print(f"Macro-F1:          {f1_sev:.6f}")
print(f"Weighted F1:       {weighted_f1_sev:.6f}")

macro_f1s = [f1_sarc, f1_hate, f1_target, f1_sev]
mean_macro_f1 = sum(macro_f1s) / len(macro_f1s)
print(f"\n======================================================")
print(f"ARITHMETIC MEAN OF 4 TASK MACRO-F1: {mean_macro_f1:.6f}")
print(f"======================================================")

print("\n=== 5. SARCASM SUBSET ANALYSIS (HATE ON SARCASTIC POSTS) ===")
sarcastic_posts = df[(df["gold_sarcasm"] == 1) & (df["gold_hate"] != -100)]
non_sarcastic_posts = df[(df["gold_sarcasm"] == 0) & (df["gold_hate"] != -100)]

y_t_sarc = sarcastic_posts["gold_hate"].astype(int)
y_p_sarc = sarcastic_posts["pred_hate"].astype(int)
cm_s = confusion_matrix(y_t_sarc, y_p_sarc)
tn_hs, fp_hs, fn_hs, tp_hs = cm_s.ravel()
fnr_sarcastic_hate = fn_hs / (fn_hs + tp_hs)
f1_sarcastic_hate = f1_score(y_t_sarc, y_p_sarc, pos_label=1)

print(f"Sarcastic Posts Total: {len(sarcastic_posts)}")
print(f"Hate F1 on Sarcastic Posts: {f1_sarcastic_hate:.6f}")
print(f"Hate FNR on Sarcastic Posts: {fnr_sarcastic_hate:.6f} ({fnr_sarcastic_hate*100:.2f}%)")

y_t_nonsarc = non_sarcastic_posts["gold_hate"].astype(int)
y_p_nonsarc = non_sarcastic_posts["pred_hate"].astype(int)
cm_ns = confusion_matrix(y_t_nonsarc, y_p_nonsarc)
tn_hns, fp_hns, fn_hns, tp_hns = cm_ns.ravel()
fnr_nonsarcastic_hate = fn_hns / (fn_hns + tp_hns)
f1_nonsarcastic_hate = f1_score(y_t_nonsarc, y_p_nonsarc, pos_label=1)

print(f"Non-Sarcastic Posts Total: {len(non_sarcastic_posts)}")
print(f"Hate F1 on Non-Sarcastic Posts: {f1_nonsarcastic_hate:.6f}")
print(f"Hate FNR on Non-Sarcastic Posts: {fnr_nonsarcastic_hate:.6f} ({fnr_nonsarcastic_hate*100:.2f}%)")
