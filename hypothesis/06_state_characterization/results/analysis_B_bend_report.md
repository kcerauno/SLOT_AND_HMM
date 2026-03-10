# 提案 B：B-end 特化状態の同定レポート

生成日時: 2026-03-08 21:42:12

## 概要

hypothesis/05 で同定した B-start Focus State と同一手法で、
B-end 出現率が最大の状態（B-end Focus State）を同定し、
B-start/B-end の役割が同一状態か別状態かを検証する。

---

## サマリー

| モデル | k | B-start Focus | B-end Focus | 同一? | B-end率(end_fs) | B-start率(end_fs) |
|--------|---|--------------|------------|------|----------------|-----------------|
| trigram | 7 | S2 | S5 | 異なる | 35.49% | 30.68% |
| trigram | 8 | S5 | S2 | 異なる | 25.79% | 11.24% |
| bigram | 7 | S2 | S0 | 異なる | 28.04% | 23.06% |
| bigram | 8 | S1 | S5 | 異なる | 36.97% | 7.82% |

---

## trigram k=7

- B-start Focus State: **S2** (B-start率=34.58%)
- B-end Focus State:   **S5**   (B-end率=35.49%)

### 各状態の B-end / B-start 出現率

| 状態 | B-end (%) | B-start (%) | 差 (end−start) | 判定 |
|-----|----------|-----------|--------------|------|
| S0 | 7.65% | 23.67% | -16.02% |  |
| S1 | 1.45% | 1.39% | +0.06% |  |
| S2 | 21.31% | 34.58% | -13.27% | ☆B-start Focus |
| S3 | 0.00% | 0.25% | -0.25% |  |
| S4 | 34.10% | 9.43% | +24.67% |  |
| S5 | 35.49% | 30.68% | +4.81% | ★B-end Focus |

### B-end Focus State の Fisher 検定

- **B-end vs S-mid (end特化？)**: 率A=35.49%, 率B=20.20%, p=2.24e-54
- **B-end vs S-head (頭部と比較)**: 率A=35.49%, 率B=10.95%, p=6.56e-161
- **B-end vs B-start (start非特化)**: 率A=35.49%, 率B=30.68%, p=1.62e-05

### B-start Focus State での B-end 集中度（逆方向確認）

- **B-start vs B-end (B-start FS の B-end 出現)**: 率A=34.58%, 率B=21.31%, p=3.70e-36

### 解釈

**【異なる状態】** B-end=S5, B-start=S2 → 境界末尾と境界先頭が別状態で表現されている（状態遷移による境界マーキング）。
  - B-start Focus S2 での B-end 率: 21.31%
  - B-end Focus S5 での B-start 率: 30.68%

---

## trigram k=8

- B-start Focus State: **S5** (B-start率=31.32%)
- B-end Focus State:   **S2**   (B-end率=25.79%)

### 各状態の B-end / B-start 出現率

| 状態 | B-end (%) | B-start (%) | 差 (end−start) | 判定 |
|-----|----------|-----------|--------------|------|
| S0 | 18.30% | 21.31% | -3.00% |  |
| S1 | 3.14% | 1.95% | +1.20% |  |
| S2 | 25.79% | 11.24% | +14.55% | ★B-end Focus |
| S3 | 13.30% | 25.51% | -12.21% |  |
| S4 | 6.76% | 1.45% | +5.31% |  |
| S5 | 22.89% | 31.32% | -8.43% | ☆B-start Focus |
| S7 | 9.82% | 7.23% | +2.59% |  |

### B-end Focus State の Fisher 検定

- **B-end vs S-mid (end特化？)**: 率A=25.79%, 率B=18.73%, p=1.55e-14
- **B-end vs S-head (頭部と比較)**: 率A=25.79%, 率B=26.86%, p=2.80e-01
- **B-end vs B-start (start非特化)**: 率A=25.79%, 率B=11.24%, p=8.27e-58

### B-start Focus State での B-end 集中度（逆方向確認）

- **B-start vs B-end (B-start FS の B-end 出現)**: 率A=31.32%, 率B=22.89%, p=1.02e-15

### 解釈

**【異なる状態】** B-end=S2, B-start=S5 → 境界末尾と境界先頭が別状態で表現されている（状態遷移による境界マーキング）。
  - B-start Focus S5 での B-end 率: 22.89%
  - B-end Focus S2 での B-start 率: 11.24%

---

## bigram k=7

- B-start Focus State: **S2** (B-start率=26.31%)
- B-end Focus State:   **S0**   (B-end率=28.04%)

### 各状態の B-end / B-start 出現率

| 状態 | B-end (%) | B-start (%) | 差 (end−start) | 判定 |
|-----|----------|-----------|--------------|------|
| S0 | 28.04% | 23.06% | +4.98% | ★B-end Focus |
| S1 | 15.86% | 12.52% | +3.34% |  |
| S2 | 8.34% | 26.31% | -17.97% | ☆B-start Focus |
| S4 | 14.21% | 1.34% | +12.88% |  |
| S5 | 22.61% | 14.46% | +8.15% |  |
| S6 | 10.93% | 22.31% | -11.38% |  |

### B-end Focus State の Fisher 検定

- **B-end vs S-mid (end特化？)**: 率A=28.04%, 率B=41.62%, p=6.43e-38
- **B-end vs S-head (頭部と比較)**: 率A=28.04%, 率B=1.21%, p=1.98e-323
- **B-end vs B-start (start非特化)**: 率A=28.04%, 率B=23.06%, p=1.46e-06

### B-start Focus State での B-end 集中度（逆方向確認）

- **B-start vs B-end (B-start FS の B-end 出現)**: 率A=26.31%, 率B=8.34%, p=1.67e-93

### 解釈

**【異なる状態】** B-end=S0, B-start=S2 → 境界末尾と境界先頭が別状態で表現されている（状態遷移による境界マーキング）。
  - B-start Focus S2 での B-end 率: 8.34%
  - B-end Focus S0 での B-start 率: 23.06%

---

## bigram k=8

- B-start Focus State: **S1** (B-start率=26.15%)
- B-end Focus State:   **S5**   (B-end率=36.97%)

### 各状態の B-end / B-start 出現率

| 状態 | B-end (%) | B-start (%) | 差 (end−start) | 判定 |
|-----|----------|-----------|--------------|------|
| S0 | 3.31% | 0.47% | +2.84% |  |
| S1 | 9.49% | 26.15% | -16.66% | ☆B-start Focus |
| S2 | 4.48% | 22.42% | -17.94% |  |
| S3 | 4.90% | 21.11% | -16.22% |  |
| S5 | 36.97% | 7.82% | +29.15% | ★B-end Focus |
| S6 | 21.59% | 12.52% | +9.07% |  |
| S7 | 19.28% | 9.51% | +9.76% |  |

### B-end Focus State の Fisher 検定

- **B-end vs S-mid (end特化？)**: 率A=36.97%, 率B=16.89%, p=1.45e-95
- **B-end vs S-head (頭部と比較)**: 率A=36.97%, 率B=20.62%, p=8.96e-61
- **B-end vs B-start (start非特化)**: 率A=36.97%, 率B=7.82%, p=8.59e-206

### B-start Focus State での B-end 集中度（逆方向確認）

- **B-start vs B-end (B-start FS の B-end 出現)**: 率A=26.15%, 率B=9.49%, p=2.71e-78

### 解釈

**【異なる状態】** B-end=S5, B-start=S1 → 境界末尾と境界先頭が別状態で表現されている（状態遷移による境界マーキング）。
  - B-start Focus S1 での B-end 率: 9.49%
  - B-end Focus S5 での B-start 率: 7.82%

---

_本レポートは analysis_BC_boundary_transition.py により自動生成。_