"""Reproduction script for Stage 3 Tri-Dataset disaggregated evaluation by source dataset."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

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

for src in ["hinglish_balanced", "multibully", "hot"]:
    sub = df[df["source_dataset"] == src]
    display_name = "Hinglish Toxic Hate (Kaggle)" if src == "hinglish_balanced" else ("MultiBully (SIGIR '22)" if src == "multibully" else "HOT Tweets (FIRE)")
    print(f"=== Source: {display_name} (N={len(sub)}) ===")
    
    # Hate
    sub_h = sub[sub["gold_hate"] != -100]
    f1_h = f1_score(sub_h["gold_hate"], sub_h["pred_hate"], average="macro")
    acc_h = accuracy_score(sub_h["gold_hate"], sub_h["pred_hate"])
    print(f"  Hate (N={len(sub_h)}):      Macro-F1={f1_h:.4f}, Acc={acc_h:.4f}")
    
    # Sarcasm
    sub_s = sub[sub["gold_sarcasm"] != -100]
    if len(sub_s) > 0 and len(sub_s["gold_sarcasm"].unique()) > 1:
        f1_s = f1_score(sub_s["gold_sarcasm"], sub_s["pred_sarcasm"], average="macro")
        acc_s = accuracy_score(sub_s["gold_sarcasm"], sub_s["pred_sarcasm"])
        print(f"  Sarcasm (N={len(sub_s)}):   Macro-F1={f1_s:.4f}, Acc={acc_s:.4f}")
    else:
        print(f"  Sarcasm (N={len(sub_s)}):   Cannot compute")
        
    # Target
    sub_t = sub[sub["gold_target"] != -100]
    if len(sub_t) > 0 and len(sub_t["gold_target"].unique()) > 1:
        f1_t = f1_score(sub_t["gold_target"], sub_t["pred_target"], average="macro")
        acc_t = accuracy_score(sub_t["gold_target"], sub_t["pred_target"])
        print(f"  Target (N={len(sub_t)}):    Macro-F1={f1_t:.4f}, Acc={acc_t:.4f}")
    else:
        print(f"  Target (N={len(sub_t)}):    N={len(sub_t)}")
        
    # Severity
    sub_v = sub[sub["gold_severity"] != -100]
    if len(sub_v) > 0 and len(sub_v["gold_severity"].unique()) > 1:
        f1_v = f1_score(sub_v["gold_severity"], sub_v["pred_severity"], average="macro")
        acc_v = accuracy_score(sub_v["gold_severity"], sub_v["pred_severity"])
        print(f"  Severity (N={len(sub_v)}):  Macro-F1={f1_v:.4f}, Acc={acc_v:.4f}")
    else:
        print(f"  Severity (N={len(sub_v)}):  N={len(sub_v)}")

# Combined Real-World Subset (MultiBully + HOT)
real_df = df[df["source_dataset"].isin(["multibully", "hot"])]
print(f"\n=== COMBINED REAL-WORLD SUBSET (MultiBully + HOT, N={len(real_df)}) ===")
real_h = real_df[real_df["gold_hate"] != -100]
print(f"  Hate Macro-F1:     {f1_score(real_h['gold_hate'], real_h['pred_hate'], average='macro'):.4f}, Acc={accuracy_score(real_h['gold_hate'], real_h['pred_hate']):.4f}")
real_s = real_df[real_df['gold_sarcasm'] != -100]
print(f"  Sarcasm Macro-F1:  {f1_score(real_s['gold_sarcasm'], real_s['pred_sarcasm'], average='macro'):.4f}, Acc={accuracy_score(real_s['gold_sarcasm'], real_s['pred_sarcasm']):.4f}")
real_t = real_df[real_df['gold_target'] != -100]
print(f"  Target Macro-F1:   {f1_score(real_t['gold_target'], real_t['pred_target'], average='macro'):.4f}, Acc={accuracy_score(real_t['gold_target'], real_t['pred_target']):.4f}")
real_v = real_df[real_df['gold_severity'] != -100]
print(f"  Severity Macro-F1: {f1_score(real_v['gold_severity'], real_v['pred_severity'], average='macro'):.4f}, Acc={accuracy_score(real_v['gold_severity'], real_v['pred_severity']):.4f}")
