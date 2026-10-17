"""Non-neural baselines from the evaluation plan: keyword spotting, sentiment-lexicon, TF-IDF + logistic regression.
All expose ``fit(df)`` / ``predict(df) -> DataFrame[id, pred_hate, p_hate]`` on the canonical dataframe."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..labels import HATE_LABELS, IGNORE_INDEX, LABEL_SPACES, TASKS, label_to_id
from ..preprocessing.lexicons import ABUSE, NEGATIVE
from ..preprocessing.normalize import TextNormalizer


class KeywordBaseline:
    """Flags hate when the post contains any abusive keyword (what sarcasm-blind moderation tools do)."""
    name = "keyword"

    def __init__(self, lexicon=None, normalizer=None):
        self.lex = {w.lower() for w in (lexicon or ABUSE)}
        self.norm = normalizer or TextNormalizer()

    def fit(self, df):
        return self

    def predict(self, df):
        hits = np.array([sum(w in self.lex for w in self.norm(t).words) for t in df["text"]])
        return pd.DataFrame({"id": df["id"].to_numpy(), "pred_hate": (hits > 0).astype(int), "p_hate": np.minimum(hits / 2.0, 1.0)})


class SentimentBaseline:
    """Flags hate when the share of negative-sentiment words exceeds a threshold learned on the training split."""
    name = "sentiment"

    def __init__(self, normalizer=None):
        self.norm = normalizer or TextNormalizer()
        self.thr = 0.1

    def _score(self, df):
        out = []
        for t in df["text"]:
            w = self.norm(t).words
            out.append(sum(x in NEGATIVE or x in ABUSE for x in w) / max(len(w), 1))
        return np.array(out)

    def fit(self, df):
        s, y = self._score(df), np.array([label_to_id("hate", v) for v in df["hate"]])
        keep = y != IGNORE_INDEX
        grid = np.linspace(0.01, 0.5, 50)
        from sklearn.metrics import f1_score
        self.thr = float(max(grid, key=lambda g: f1_score(y[keep], (s[keep] >= g).astype(int), average="macro", zero_division=0)))
        return self

    def predict(self, df):
        s = self._score(df)
        return pd.DataFrame({"id": df["id"].to_numpy(), "pred_hate": (s >= self.thr).astype(int), "p_hate": np.clip(s / (2 * self.thr), 0, 1)})


class TfidfLogReg:
    """Strong classical baseline: word + char n-gram TF-IDF with balanced logistic regression (robust to spelling noise).
    One independent classifier per task (hate/sarcasm on all posts; target/severity on hate posts that carry a label)."""
    name = "tfidf_lr"

    def __init__(self, normalizer=None, C: float = 4.0, tasks=("hate", "sarcasm", "target", "severity")):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.norm = normalizer or TextNormalizer()
        self.C, self.tasks = C, tuple(tasks)
        self.v_w = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.v_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        self._LR, self.clf = LogisticRegression, {}

    def _x(self, df, fit=False):
        from scipy.sparse import hstack
        txt = [self.norm(t).text for t in df["text"]]
        if fit:
            return hstack([self.v_w.fit_transform(txt), self.v_c.fit_transform(txt)]).tocsr()
        return hstack([self.v_w.transform(txt), self.v_c.transform(txt)]).tocsr()

    def fit(self, df):
        X = self._x(df, True)
        for t in self.tasks:
            y = np.array([label_to_id(t, v) for v in df[t]]) if t in df else np.array([])
            keep = y != IGNORE_INDEX
            if keep.sum() > 1 and len(set(y[keep])) > 1:
                self.clf[t] = self._LR(C=self.C, max_iter=2000, class_weight="balanced").fit(X[keep], y[keep])
        return self

    def predict(self, df):
        X = self._x(df)
        out = pd.DataFrame({"id": df["id"].to_numpy()})
        for t, clf in self.clf.items():
            p = clf.predict_proba(X)
            full = np.zeros((len(df), len(LABEL_SPACES[t])))
            full[:, clf.classes_] = p
            out[f"pred_{t}"] = full.argmax(1)
            if full.shape[1] == 2:
                out[f"p_{t}"] = full[:, 1]
            elif t in ("hate", "sarcasm"):
                out[f"p_{t}"] = full[:, 1]
        return out


class MajorityBaseline:
    """Always predicts the most frequent training class of each task (reference point for macro-F1)."""
    name = "majority"

    def fit(self, df):
        self.maj = {}
        for t in ("hate", "sarcasm", "target", "severity"):
            y = np.array([label_to_id(t, v) for v in df[t]]) if t in df else np.array([])
            y = y[y != IGNORE_INDEX]
            if len(y):
                self.maj[t] = int(np.bincount(y).argmax())
        return self

    def predict(self, df):
        out = pd.DataFrame({"id": df["id"].to_numpy()})
        for t, c in self.maj.items():
            out[f"pred_{t}"] = c
            if t in ("hate", "sarcasm"):
                out[f"p_{t}"] = float(c)
        return out


def with_gold(df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Attach gold labels so the shared evaluator (``evaluate_predictions``) can be used."""
    out = pred.copy()
    for t in TASKS:
        out[f"gold_{t}"] = [label_to_id(t, v) for v in df[t]]
    if "narrative" in df:
        out["narrative"] = df["narrative"].to_numpy()
    return out


def run_baselines(df: pd.DataFrame) -> dict:
    from ..training.metrics import evaluate_predictions, flatten_metrics
    tr, te = df[df.split == "train"], df[df.split == "test"].reset_index(drop=True)
    results, preds = {}, {}
    for B in (MajorityBaseline, KeywordBaseline, SentimentBaseline, TfidfLogReg):
        b = B().fit(tr)
        pf = with_gold(te, b.predict(te))
        preds[b.name] = pf
        tasks = tuple(t for t in TASKS if f"pred_{t}" in pf)
        results[b.name] = flatten_metrics(evaluate_predictions(pf, tasks=tasks))
    return {"metrics": results, "preds": preds}
