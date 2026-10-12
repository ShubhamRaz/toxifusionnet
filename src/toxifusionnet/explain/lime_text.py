"""LIME explanations (Ribeiro et al., 2016) for the hate decision, as used by Mamun et al. (2024)."""
from __future__ import annotations

import numpy as np


def lime_explain(pipeline, text: str, num_features: int = 8, num_samples: int = 300, seed: int = 0) -> list[tuple[str, float]]:
    from lime.lime_text import LimeTextExplainer  # optional dependency
    nt = pipeline.normalizer(text)
    explainer = LimeTextExplainer(class_names=["non_hate", "hate"], split_expression=r"\s+", bow=False, random_state=seed)

    def clf(strings):
        p = pipeline.hate_probability([s if s.strip() else "<empty>" for s in strings])
        return np.stack([1 - p, p], axis=1)

    exp = explainer.explain_instance(nt.text, clf, num_features=num_features, num_samples=num_samples, labels=(1,))
    return [(str(w), float(v)) for w, v in exp.as_list(label=1)]
