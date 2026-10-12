# Dataset report

Rows: **5718**  |  splits: {'train': 4002, 'test': 858, 'val': 858}  |  label sources: `{"hate": "gold-derived", "target": "gold", "sarcasm": "gold", "severity": "gold"}`

**sarcasm**: not_sarcastic 3589 (62.8%), sarcastic 2129 (37.2%)

**hate**: non_hate 4138 (72.4%), hate 1580 (27.6%)

**target**: (missing) 4281 (74.9%), individual 1037 (18.1%), community 251 (4.4%), organization 81 (1.4%), society 68 (1.2%)

**severity**: (missing) 4138 (72.4%), moderate 1409 (24.6%), low 141 (2.5%), high 30 (0.5%)


## Hate x sarcasm by split

| split | posts | hate | sarcastic | sarcastic hate | sarcastic non-hate |
|---|---|---|---|---|---|
| test | 858 | 237 | 312 | **131** | 181 |
| train | 4002 | 1106 | 1517 | **599** | 918 |
| val | 858 | 237 | 300 | **117** | 183 |

## Language mix (heuristic)

- mean_cmi: 9.4
- pct_posts_cmi_ge_10: 27.7
- pct_posts_with_any_hindi_token: 38.3
- pct_posts_majority_hindi: 24.9
- pct_posts_with_devanagari: 2.2
- note: token-level language ID is heuristic (seed lexicons); treat as an estimate

## Length (words)

{'mean': 16.2, 'p50': 14, 'p95': 35, 'max': 184}

## Warnings

- Only 38.3% of posts contain a detected Hindi token: the corpus is predominantly English; describe it as 'code-mixed' only for its Hinglish portion.