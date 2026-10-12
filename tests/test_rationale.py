import numpy as np

from toxifusionnet.explain.metrics import faithfulness, iou_f1, plausibility, top_k_mask
from toxifusionnet.explain.rationale import (aggregate_annotators, combine_rationales, ground_truth_attention,
                                             temperature_softmax, to_binary_masks)


def test_combine_matches_paper_text_values():
    h = np.array([0, 1, 1, 0], dtype=np.float32)
    s = np.array([1, 0, 1, 0], dtype=np.float32)
    assert combine_rationales(h, s).tolist() == [0.5, 1.0, 1.5, 0.0]
    assert combine_rationales(h, s, mode="algorithm").tolist() == [0.5, 1.0, 2.0, 0.0]


def test_mask_specs():
    words = ["a", "b", "c"]
    assert to_binary_masks(words, [0, 1, 0])[0].tolist() == [0, 1, 0]
    assert to_binary_masks(words, '["A", "c"]')[0].tolist() == [1, 0, 1]
    assert len(to_binary_masks(words, [["a"], ["b"], ["zzz"]])) == 2   # annotator with no match dropped
    assert to_binary_masks(words, None) == []


def test_annotator_vote_threshold():
    m = [np.array([1, 0], np.float32)]
    assert aggregate_annotators(m, 2, min_votes=2).sum() == 0
    assert aggregate_annotators(m + m, 2, min_votes=2).tolist() == [1, 0]


def test_ground_truth_attention_is_distribution_over_non_special_tokens():
    words = ["wah", "us", "dharm", "log", "lol"]
    wid = [None, 0, 1, 2, 2, 3, 4, None]  # "dharm" split in two sub-words
    ht, st = ground_truth_attention(words, wid, ["dharm", "log"], ["wah", "lol"], tau=0.5)
    for t in (ht, st):
        assert abs(t.sum() - 1) < 1e-5 and t[0] == 0 and t[-1] == 0
    assert ht[3] > ht[1]  # rationale sub-word outweighs a non-rationale word
    none_h, none_s = ground_truth_attention(words, wid, None, None)
    assert none_h is None and none_s is None


def test_temperature_sharpens():
    v = np.array([0, 1.0, 0], dtype=np.float32); valid = np.array([True, True, True])
    assert temperature_softmax(v, valid, 0.1)[1] > temperature_softmax(v, valid, 1.0)[1]


def test_explainability_metrics():
    gold = np.array([0, 1, 1, 0, 0], dtype=np.float32)
    good = np.array([.1, .9, .8, .1, .0]); bad = np.array([.9, .0, .1, .8, .7])
    assert plausibility([good], [gold], k=2)["token_f1"] == 1.0
    assert plausibility([bad], [gold], k=2)["token_f1"] == 0.0
    assert iou_f1(top_k_mask(good, 2), gold) == 1.0
    f = faithfulness(lambda w: 0.9 if "x" in w else 0.1, ["x", "y"], np.array([1, 0]))
    assert f["comprehensiveness"] > 0.5 and abs(f["sufficiency"]) < 1e-9
