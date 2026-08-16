# Results v2, full 41k-sample bilingual run

Data: `modeling/data_v2/` (hard edge negatives) and `modeling/data_v2_random/` (v1-style random negatives), both from the 41,315 accepted samples of the full pipeline run (20,982 EN / 20,333 DE), participant-independent 50/30/20 split stratified by language, seed 42. Models: TF-IDF (word 1 to 2 grams) with (OvR) logistic regression, class-weight balanced, dev-tuned threshold, matching the v1 setup.

## Node prediction (micro/macro-F1, dev-tuned threshold, common concepts)

| Model / eval | Micro-F1 | Macro-F1 | Micro-P | Micro-R | n test |
|---|---:|---:|---:|---:|---:|
| v1 combined (1.2k entries, 71 common labels) | 0.6670 | 0.6672 | 0.6470 | 0.6882 | 227 |
| v2 combined EN+DE, full (130 labels) | 0.8134 | 0.8007 | 0.8138 | 0.8129 | 8246 |
| v2 combined EN+DE, common (130) | 0.8134 | 0.8007 | 0.8138 | 0.8129 | 8246 |
| v2 combined, scored on EN test only | 0.8466 | 0.8367 | 0.8461 | 0.8471 | 4166 |
| v2 combined, scored on DE test only | 0.7793 | 0.7647 | 0.7807 | 0.7779 | 4080 |
| v2 EN-only model on EN test | 0.8490 | 0.8399 | 0.8419 | 0.8562 | 4166 |
| v2 DE-only model on DE test | 0.7837 | 0.7698 | 0.7797 | 0.7878 | 4080 |

## Edge prediction (binary connected, dev-tuned threshold)

| Model / negatives | F1 | ROC-AUC | Accuracy |
|---|---:|---:|---:|
| v1 random negatives, full model | 0.8757 | 0.9432 | 0.8641 |
| v1 random negatives, C1 pair-only (no text) | 0.8721 | 0.9408 | 0.8574 |
| v1 random negatives, C2 type+polarity | 0.8550 | 0.9322 | 0.8424 |
| v2 random negatives, full model | 0.7400 | 0.7797 | 0.7104 |
| **v2 HARD negatives, full model** | 0.7125 | 0.7744 | 0.6973 |
| v2 hard, C1 pair-only (no text) | 0.7161 | 0.7738 | 0.7059 |
| v2 hard, C2 type+polarity only | 0.7017 | 0.7376 | 0.6568 |

## Node prediction, transformer + LLM + floor (full 130-label vocab)

| Model / eval | Micro-F1 | Macro-F1 | Micro-P | Micro-R | n test |
|---|---:|---:|---:|---:|---:|
| XLM-R combined EN+DE | 0.9106 | 0.8493 | 0.9199 | 0.9015 | 8246 |
| XLM-R EN-only on EN test | 0.8728 | 0.7282 | 0.9044 | 0.8434 | 4166 |
| XLM-R DE-only on DE test | 0.7994 | 0.6317 | 0.8477 | 0.7562 | 4080 |
| string-match floor | 0.3638 | 0.2893 | 0.6698 | 0.2498 | 8246 |
| Llama-3.2-3B zero-shot (8246 ex, 5 parse errors, max_new_tokens 400) | 0.2790 | 0.2419 | 0.2986 | 0.2617 | 8246 |

## Edge prediction, transformer + LLM (hard negatives)

| Model | F1 | ROC-AUC | Accuracy | n test |
|---|---:|---:|---:|---:|
| XLM-R pair-aware (marker tokens) | 0.7492 | 0.8123 | 0.7653 | 103798 |
| XLM-R C3 no-text ablation | 0.7410 | 0.8038 | 0.7452 | 103798 |
| Llama-3.2-3B zero-shot yes/no (4000 of 103798 pairs) | 0.6668 | 0.5833 | 0.5005 | 4000 |

Hard-negative type-pair match (total variation distance vs positives): train 0.265, dev 0.258, test 0.272. Residual mismatch is structural: type pairs with P(connected) > 0.5 lack enough unconnected co-occurrences at neg_ratio 1.0; details in `modeling/data_v2/report.md`.

## Takeaways

- **(a) Scale.** 34× more training data (602 → 20,679 train entries) moved node micro-F1 from 0.667 to 0.813 (+0.146) and macro-F1 from 0.667 to 0.801, with the label space growing 93 → 130 concepts (all now ≥ 20 train support).
- **(b) Language.** The combined EN+DE model scores micro-F1 0.847 on EN and 0.779 on DE; monolingual models reach 0.849 (EN-only) and 0.784 (DE-only). Combined vs monolingual differs by -0.002 (EN) and -0.004 (DE), so one bilingual model delivers the same accuracy as two monolingual ones.
- **(c) Shortcut.** Hard negatives removed most of the structural shortcut: the pure type+polarity model (C2) fell from AUC 0.932 (v1 random negatives) to 0.738, and every model got much harder. The full model dropped from F1 0.876/AUC 0.943 to 0.713/0.774. **The entry text stays inert for this baseline family**: pair features alone (C1, with concept ids) already match the full text+pair model (-0.004 F1, +0.001 AUC for adding text). Two reasons: (i) the residual signal lives in concept-PAIR statistics, which C1 sees directly; (ii) TF-IDF text is identical for every candidate pair from the same entry, so a bag-of-words model only sees pair features vary within an entry. C2 also stays above chance (AUC 0.738 vs 0.5) because the type-pair match is capped by candidate availability (TV ≈ 0.26), which forbids exact matching at neg_ratio 1.0 for pairs with P(connected) > 0.5. Verdict: the type-pair shortcut is largely closed; showing that *text* matters for edges needs a pair-aware text model (e.g. BERT with node-span markers), a class that entry-level TF-IDF sits outside of.

## Reproduction

```bash
python modeling/prepare_data.py --negatives hard   --output modeling/data_v2
python modeling/prepare_data.py --negatives random --output modeling/data_v2_random
python modeling/baselines/node_baseline.py --language all
python modeling/baselines/node_baseline.py --language en
python modeling/baselines/node_baseline.py --language de
python modeling/baselines/edge_baseline.py --data-dir modeling/data_v2/edge_prediction --tag hard
python modeling/baselines/edge_baseline.py --data-dir modeling/data_v2_random/edge_prediction --tag random
python modeling/baselines/edge_shortcut.py
# GPU (RTX 4090 used for the reported numbers):
python modeling/baselines/node_bert.py --language all   # + en, de
python modeling/baselines/edge_bert.py --tag hard        # + --no-text
python modeling/baselines/llama_v2.py node               # + edge
python modeling/baselines/string_match_floor.py
python modeling/baselines/make_results_v2.py
```
