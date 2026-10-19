import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import hstack

df = pd.read_csv("data/processed/multibully.csv")

# Feature extraction: rich word + char ngrams + lexical sentiment + length features
vec_word = TfidfVectorizer(max_features=12000, ngram_range=(1, 3), sublinear_tf=True)
vec_char = TfidfVectorizer(max_features=18000, analyzer="char_wb", ngram_range=(2, 6), sublinear_tf=True)
X_w = vec_word.fit_transform(df["text"].fillna(""))
X_c = vec_char.fit_transform(df["text"].fillna(""))
X = hstack([X_w, X_c]).tocsr()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Function to find optimal per-class thresholds to maximize Macro-F1
def optimize_thresholds_multiclass(probs, y_true):
    best_f1 = 0
    best_weights = None
    classes = np.unique(y_true)
    # Grid search on class multipliers
    for w0 in [0.5, 1.0, 1.5]:
        for w1 in [0.8, 1.0, 1.5, 2.0]:
            for w2 in [1.0, 1.5, 2.0, 3.0]:
                if len(classes) == 4:
                    for w3 in [1.5, 2.5, 3.5]:
                        w = np.array([w0, w1, w2, w3])
                        scaled_p = probs * w
                        preds = classes[np.argmax(scaled_p, axis=1)]
                        f1 = f1_score(y_true, preds, average="macro")
                        if f1 > best_f1:
                            best_f1 = f1
                            best_weights = w
                else:
                    w = np.array([w0, w1, w2])
                    scaled_p = probs * w
                    preds = classes[np.argmax(scaled_p, axis=1)]
                    f1 = f1_score(y_true, preds, average="macro")
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights = w
    return best_f1, best_weights

print("Testing optimization...")
