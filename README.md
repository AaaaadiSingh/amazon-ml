# Business Entity Resolution — ML Challenge

Match each **Source 1** business record to zero, one, or many records in **Source 2 / Source 3**.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data files in data/raw/
#    - source1.csv, source2.csv, source3.csv
#    - train_ground_truth.tsv
#    - test source files (same format)

# 3. Run the full pipeline (train + predict)
python -m src.pipeline --mode train --data-dir data/raw --output-dir output

# 4. Validate submission
python utils/validate_submission.py --matching output/matching_results.tsv --candidates output/candidate_pairs.tsv
```

## Architecture

```
Normalization → Multi-strategy Blocking → Feature Engineering → XGBoost Scorer → Calibration + F0.5 Threshold → Output
```

- **Blocking**: Lexical (MinHash/LSH) + Embedding (multilingual bi-encoder + FAISS) + Address (token overlap). Union of all three.
- **Scoring**: XGBoost on pairwise features (Stage A). Optional fine-tuned sequence-pair classifier on gray zone (Stage B, stretch).
- **Decision**: Isotonic calibration → F0.5-optimal threshold search → abstain-to-empty for low-confidence → collective consistency.

## Repository Structure

```
├── data/
│   └── raw/                    # Drop challenge data here
├── src/
│   ├── data/                   # Loading, normalization, abbreviation dictionaries
│   ├── blocking/               # Lexical, embedding, address blockers + union
│   ├── features/               # Pairwise feature computation
│   ├── models/                 # XGBoost training/inference
│   ├── decision/               # Calibration, F0.5 threshold, abstain logic
│   ├── collective/             # Conflict resolution
│   ├── eval/                   # Custom per-entity macro F0.5 scorer
│   └── pipeline.py             # End-to-end orchestration
├── utils/
│   └── validate_submission.py  # Submission format validator
├── output/                     # Generated TSV files
├── requirements.txt
└── README.md
```

## Metric

**F₀.₅** per Source-1 entity, macro-averaged. Precision matters ~2x more than recall.
A correctly predicted empty list scores 1.0; a wrong match on a true singleton scores 0.0.

## Constraints

- No external data/API lookups
- Final model ≤ 8B params, MIT/Apache-2.0 licensed
- All components: XGBoost (Apache-2.0), FAISS (MIT), sentence-transformers (Apache-2.0), rapidfuzz (MIT)
