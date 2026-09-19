# Experiment 2D Results: Candidate Connection Ranking

## Validation Metrics (Held-out Cold-start Edges)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9796 |
| PR-AUC | 0.9740 |
| Recall@10 | 0.0000 (9.58e-06) |
| Recall@50 | 0.0000 (4.79e-05) |
| Recall@100 | 0.0001 (9.58e-05) |
| Hits@10 | 10 |
| Hits@50 | 50 |
| Hits@100 | 100 |
| Precision@10 | 1.0000 |
| Precision@50 | 1.0000 |
| Precision@100 | 1.0000 |

Recall is near-zero because the test set has 1,043,974 positive edges. Hits@K (the raw count) is more informative.

## Candidate Generation

| Parameter | Value |
|-----------|-------|
| Sources sampled | 10,000 |
| Targets per source | 100 |
| Total candidates scored | 1,000,000 |
| Final top candidates saved | 100,000 |
| Score mean | 0.1346 |
| Score median | 0.0300 |
| Score P90 / P95 / P99 | 0.48 / 0.68 / 0.91 |

## Top 20 Candidates

| Rank | Source ID | Target ID | Score | Source Type | Target Type | Source NT | Target NT | Source Flow | Target Flow |
|------|-----------|-----------|-------|-------------|-------------|-----------|-----------|-------------|-------------|
| 1 | ...11292622 | ...14140522 | 1.0 | CL029b | CB1301 | GLUT | ACH | intrinsic | intrinsic |
| 1 | ...39428174 | ...20842111 | 1.0 | DNp23 | PS013 | ACH | ACH | efferent | intrinsic |
| 1 | ...25011600 | ...27281594 | 1.0 | AVLP316 | VES079 | ACH | ACH | intrinsic | intrinsic |
| 1 | ...32964423 | ...37899197 | 1.0 | AVLP572 | AOTU046 | ACH | GLUT | intrinsic | intrinsic |
| 1 | ...39417728 | ...25019696 | 1.0 | AVLP053 | LCe06 | ACH | ACH | intrinsic | intrinsic |
| 1 | ...11292622 | ...18364480 | 1.0 | CL029b | pC1e | GLUT | ACH | intrinsic | intrinsic |
| 1 | ...24023188 | ...16262261 | 1.0 | CB0824 | CB1444 | ACH | -- | intrinsic | intrinsic |
| 1 | ...11665358 | ...14737704 | 1.0 | Tm2 | Pm04 | ACH | GABA | intrinsic | intrinsic |
| 1 | ...27438076 | ...14165201 | 1.0 | AVLP201 | DNge028 | GABA | ACH | intrinsic | efferent |
| 1 | ...27438076 | ...30535030 | 1.0 | AVLP201 | AVLP574 | GABA | ACH | intrinsic | intrinsic |
| 1 | ...30839371 | ...27728149 | 1.0 | DNge146 | CB0073 | GABA | ACH | efferent | intrinsic |
| 1 | ...30839371 | ...04009952 | 1.0 | DNge146 | PS061 | GABA | ACH | efferent | intrinsic |
| 1 | ...20326253 | ...21818203 | 1.0 | AOTU063a | LT63 | GLUT | ACH | intrinsic | intrinsic |
| 2 | ...39417728 | ...14140522 | 1.0 | AVLP053 | CB1301 | ACH | ACH | intrinsic | intrinsic |
| 2 | ...27438076 | ...14140522 | 1.0 | AVLP201 | CB1301 | GABA | ACH | intrinsic | intrinsic |
| 2 | ...39417728 | ...33094113 | 1.0 | AVLP053 | SMP179 | ACH | ACH | intrinsic | intrinsic |
| 2 | ...27341736 | ...27683588 | 1.0 | LPT21 | CL339 | ACH | ACH | intrinsic | intrinsic |
| 3 | ...39417728 | ...14140522 | 1.0 | AVLP053 | LC11 | ACH | ACH | intrinsic | intrinsic |
| 3 | ...27438076 | ...14671411 | 1.0 | AVLP201 | Li23 | GABA | GABA | intrinsic | intrinsic |
| 3 | ...27341736 | ...33614253 | 1.0 | LPT21 | LT72 | ACH | ACH | intrinsic | intrinsic |

Full list: `results/candidates/top20_annotated.csv`

## Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | area_nm | 0.0203 |
| 2 | length_nm | 0.0165 |
| 3 | size_nm | 0.0152 |
| 4 | coord_x | 0.0142 |
| 5 | coord_y | 0.0100 |
| 6 | gaba_avg | 0.0088 |
| 7 | coord_z | 0.0085 |
| 8 | ach_avg | 0.0073 |
| 9 | da_avg | 0.0070 |
| 10 | nt_type_score | 0.0059 |

## Feature Group Importance

| Group | Importance |
|-------|------------|
| Source biology | 0.0520 |
| Target biology | 0.0331 |
| Source-target interaction | 0.0253 |
| Source-target difference | 0.0228 |

## Peak RAM

1056 MB

## Tests

25/25 passed

- `tests/test_link_prediction.py`: 8/8
- `tests/test_cold_start.py`: 7/7
- `tests/test_experiment_2d.py`: 10/10
