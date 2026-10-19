import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report

df = pd.read_csv("data/processed/multibully.csv")
print("Total rows:", len(df))

# 1. HATE DETECTION
print("\n--- HATE DETECTION ---")
y_hate = (df["hate"] == "hate").astype(int).values
vec_word = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), sublinear_tf=True)
vec_char = TfidfVectorizer(max_features=15000, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
from scipy.sparse import hstack
X_w = vec_word.fit_transform(df["text"].fillna(""))
X_c = vec_char.fit_transform(df["text"].fillna(""))
X = hstack([X_w, X_c]).tocsr()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_hate = np.zeros(len(df))

for tr, val in cv.split(X, y_hate):
    clf = LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced")
    clf.fit(X[tr], y_hate[tr])
    oof_hate[val] = clf.predict_proba(X[val])[:, 1]

# Check uncalibrated vs confidence calibrated
print("Hate Tau=0.5 F1:", f1_score(y_hate, (oof_hate >= 0.5).astype(int), average="macro"))

# Selective triage on hate
for m in [0.0, 0.1, 0.15, 0.2, 0.25]:
    mask = np.abs(oof_hate - 0.5) >= m
    f1 = f1_score(y_hate[mask], (oof_hate[mask] >= 0.5).astype(int), average="macro")
    acc = accuracy_score(y_hate[mask], (oof_hate[mask] >= 0.5).astype(int))
    print(f"Hate Margin {m:.2f}: F1 = {f1:.4f}, Acc = {acc*100:.2f}%, Cov = {mask.mean()*100:.1f}%")

# 2. SARCASM DETECTION
print("\n--- SARCASM DETECTION ---")
y_sarc = (df["sarcasm"] == "sarcastic").astype(int).values
oof_sarc = np.zeros(len(df))
for tr, val in cv.split(X, y_sarc):
    clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")
    clf.fit(X[tr], y_sarc[tr])
    oof_sarc[val] = clf.predict_proba(X[val])[:, 1]

for m in [0.0, 0.1, 0.15, 0.2, 0.25]:
    mask = np.abs(oof_sarc - 0.5) >= m
    f1 = f1_score(y_sarc[mask], (oof_sarc[mask] >= 0.5).astype(int), average="macro")
    acc = accuracy_score(y_sarc[mask], (oof_sarc[mask] >= 0.5).astype(int))
    print(f"Sarcasm Margin {m:.2f}: F1 = {f1:.4f}, Acc = {acc*100:.2f}%, Cov = {mask.mean()*100:.1f}%")

# 3. SEVERITY
print("\n--- SEVERITY ---")
valid_sev = df["severity"].notna() & (df["severity"] != "-100")
df_sev = df[valid_sev].reset_index(drop=True)
y_sev = df_sev["severity"].values
X_sev = hstack([vec_word.transform(df_sev["text"].fillna("")), vec_char.transform(df_sev["text"].fillna(""))]).tocsr()
oof_sev_probs = np.zeros((len(df_sev), 3))
cv_sev = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, val in cv_sev.split(X_sev, y_sev):
    clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")
    clf.fit(X_sev[tr], y_sev[tr])
    oof_sev_probs[val] = clf.predict_proba(X_sev[val])

sev_classes = clf.classes_
pred_sev = sev_classes[np.argmax(oof_sev_probs, axis=1)]
print("Severity Base F1:", f1_score(y_sev, pred_sev, average="macro"))
max_p_sev = np.max(oof_sev_probs, axis=1)
for th in [0.35, 0.45, 0.55, 0.65]:
    mask = max_p_sev >= th
    f1 = f1_score(y_sev[mask], pred_sev[mask], average="macro")
    acc = accuracy_score(y_sev[mask], pred_sev[mask])
    print(f"Severity Conf >= {th:.2f}: F1 = {f1:.4f}, Acc = {acc*100:.2f}%, Cov = {mask.mean()*100:.1f}%")

# 4. TARGET
print("\n--- TARGET ---")
valid_t = df["target"].notna() & (df["target"] != "-100")
df_t = df[valid_t].reset_index(drop=True)
y_t = df_t["target"].values
X_t = hstack([vec_word.transform(df_t["text"].fillna("")), vec_char.transform(df_t["text"].fillna(""))]).tocsr()
oof_t_probs = np.zeros((len(df_t), len(np.unique(y_t))))
cv_t = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, val in cv_t.split(X_t, y_t):
    clf = LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced")
    clf.fit(X_t[tr], y_t[tr])
    oof_t_probs[val] = clf.predict_proba(X_t[val])

t_classes = clf.classes_
pred_t = t_classes[np.argmax(oof_t_probs, axis=1)]
print("Target Base F1:", f1_score(y_t, pred_t, average="macro"))
max_p_t = np.max(oof_t_probs, axis=1)
for th in [0.35, 0.45, 0.55, 0.65, 0.75]:
    mask = max_p_t >= th
    f1 = f1_score(y_t[mask], pred_t[mask], average="macro")
    acc = accuracy_score(y_t[mask], pred_t[mask])
    print(f"Target Conf >= {th:.2f}: F1 = {f1:.4f}, Acc = {acc*100:.2f}%, Cov = {mask.mean()*100:.1f}%")
