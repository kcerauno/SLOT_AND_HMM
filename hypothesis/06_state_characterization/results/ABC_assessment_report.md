# 提案 A・B・C 実施結果評価レポート

作成日時: 2026-03-08
根拠: [hypothesis/05 の次フェーズ提案](../../05_re-compound_hmm_trigram/results/next_analysis_proposals.md)
実施: [hypothesis/06_state_characterization/](../)

---

## 総括

提案 A・B・C はいずれも予定通り実施し、**3 つすべてで新たな知見を得た**。
うち 2 つ（B・C）は提案時の「期待される成果」を完全に確認し、
1 つ（A）は期待と一部異なる結果を生んだが、それ自体が重要な知見である。

| 提案 | 主結論 | 期待との整合 |
|-----|--------|------------|
| A：放射確率行列の文字レベル分析 | Focus State の放射文字は gallows 系ではなく vowel/c 系。Trigram k=7 のみ基先頭文字分布に近い | **部分的**（期待と異なる成分が重要な発見） |
| B：B-end 特化状態の同定 | 全モデルで B-end ≠ B-start Focus（別状態）。境界の方向性が符号化されている | **完全に確認** |
| C：境界通過時の状態遷移パターン | ±2 全位置で有意。Bigram の方がパターンが鮮明 | **完全に確認**（Bigram > Trigram という追加知見あり） |

---

## 提案 A：放射確率行列の文字レベル分析

### 提案の問い

> HMM 各状態が「どの文字を好んで放射するか」を明らかにし、
> Focus State の放射が V8 スロットの先頭文字に集中しているかを検定する。

### 実施スクリプト・出力ファイル

- スクリプト: [source/analysis_A_emission.py](../source/analysis_A_emission.py)
- レポート: [results/analysis_A_emission_report.md](analysis_A_emission_report.md)
- 主要図表:
  - ヒートマップ: [emission_heatmap_trigram_k7.png](emission_heatmap_trigram_k7.png) / [_k8](emission_heatmap_trigram_k8.png) / [bigram_k7](emission_heatmap_bigram_k7.png) / [_k8](emission_heatmap_bigram_k8.png)
  - 各状態上位文字: [emission_top_chars_trigram_k7.png](emission_top_chars_trigram_k7.png) / [_k8](emission_top_chars_trigram_k8.png) / [bigram_k7](emission_top_chars_bigram_k7.png) / [_k8](emission_top_chars_bigram_k8.png)
  - Focus State 詳細比較: [emission_focus_detail_trigram_k7.png](emission_focus_detail_trigram_k7.png) / [_k8](emission_focus_detail_trigram_k8.png) / [bigram_k7](emission_focus_detail_bigram_k7.png) / [_k8](emission_focus_detail_bigram_k8.png)

### 主要結果

#### 全モデルの状態別放射特性

| モデル | k | 状態 | 上位文字（prob） | 文字グループ |
|--------|---|------|----------------|------------|
| Bigram | 7 | S0 | e(0.485), d(0.180), o(0.118) | vowel |
| | | S1 | c(0.741), s(0.240) | other（c 集中） |
| | | **S2 ★Focus** | **a(0.357), y(0.342), o(0.231)** | **vowel** |
| | | S4 | o(0.317), l(0.111), t(0.096) | vowel |
| | | S5 | i(0.144), r(0.109), l(0.106) | liquid |
| | | S6 | h(0.859) | sibilant（h 独占） |
| Bigram | 8 | S2 | h(0.986) | sibilant（h 独占） |
| | | S3 | e(0.814), o(0.166) | vowel（e 集中） |
| | | S5 | d(0.318), k(0.255), t(0.172) | gallows |
| | | **S1 ★Focus** | **a(0.352), y(0.340), o(0.217)** | **vowel** |
| Trigram | 7 | S0 | m(0.407), v(0.285) | nasal |
| | | S1 | f(0.987) | gallows（f 独占） |
| | | **S2 ★Focus** | **c(0.586), a(0.186), m(0.106)** | **other（c 集中）** |
| | | S4 | h(0.214), r(0.189), i(0.166) | mixed |
| | | S5 | y(0.515), g(0.149), p(0.098) | vowel |
| Trigram | 8 | S1 | f(0.984) | gallows（f 独占） |
| | | S3 | c(0.724), a(0.140) | other（c 集中） |
| | | **S5 ★Focus** | **y(0.640), v(0.234), l(0.079)** | **vowel（y 独占）** |
| | | S7 | m(0.573), v(0.080) | nasal |

#### Focus State と基先頭文字分布との距離（KL ダイバージェンス）

| モデル | k | Focus State | KL(Focus ∥ base_init) | KL(全状態平均 ∥ base_init) | Focus が近い？ |
|--------|---|-------------|----------------------|--------------------------|-------------|
| Bigram | 7 | S2 | 1.133 | 0.133 | **否** |
| Bigram | 8 | S1 | 1.179 | 0.141 | **否** |
| **Trigram** | **7** | **S2** | **1.048** | **2.387** | **✓ YES** |
| Trigram | 8 | S5 | 6.705 | 2.358 | **否（最遠）** |

### 評価と解釈

#### 1. 「gallows 系文字 → V8 スロット相当」という期待への回答

**否定される**。Focus State が gallows 系（t/k/p/f）を好むモデルは存在しない。

- Bigram Focus States（S2 k=7, S1 k=8）は一貫して **vowel 系（a, y, o）** を放射する。
  これは Voynich 語における複合語境界に母音的文字が現れやすいことを反映している可能性があるが、
  gallows 系との対応仮説は支持されない。

- Trigram k=7 の Focus State S2 は **'c' に 58.6% 集中**する。
  'c' は V8 スロットの C_digraph（ch, sh）・C_trigraph（cth, ckh 等）の先頭文字であり、
  これらの複合子音クラスタが複合語の基先頭で頻出することを反映していると解釈できる。

#### 2. Focus State の放射分布は「基先頭文字分布に近い」か

**Trigram k=7 のみ確認、他は否定される**。

- Trigram k=7 S2 だけが基先頭文字分布に対して全状態平均より小さい KL（1.048 < 2.387）を示す。
  → このモデルにおいてのみ、Focus State が「語基の先頭で現れやすい文字を優先的に放射する」
  という仮説が成立する。

- Trigram k=8 の Focus State S5 は 'y' に 64% 集中しており、KL が最大（6.705）。
  'y' は語末的な文字であり、B-start 特化状態としては異質に見える。
  ただし、hypothesis/05 の役割分析では S5 は B-start に強く集中していたため、
  **「y で始まる基（例: y 単独の基）」が複合語境界先頭として頻出する**という
  データ的事実を反映していると考えられる。

#### 3. 各モデルの状態機能の特定

各モデルに機能的に特化した状態が確認された：

- **h 独占状態**（Bigram k=7 S6, k=8 S2）：'h' を 86〜99% 放射。V8 の digraph/trigraph 第2文字（ch の h）を担う専用状態と解釈できる。
- **f 独占状態**（Trigram k=7 S1, k=8 S1）：'f' を 98% 放射。gallows 文字のうち f 専用状態。
- **e 集中状態**（Bigram k=8 S3）：'e' を 81% 放射。V8 の V_multi スロット（eee/ee/e）に対応。
- **m/v 状態**（Trigram k=7 S0, k=8 S7）：Voynich 手稿での 'm'/'v' の共起パターンを捉えた状態。

---

## 提案 B：B-end 特化状態の同定

### 提案の問い

> B-end 出現率が最大の状態を同定し、B-start 特化状態との対称性・非対称性を確認する。
> B-end/B-start が異なる状態なら「語境界の方向性がモデルに符号化されている」。

### 実施スクリプト・出力ファイル

- スクリプト: [source/analysis_BC_boundary_transition.py](../source/analysis_BC_boundary_transition.py)
- レポート: [results/analysis_B_bend_report.md](analysis_B_bend_report.md)
- 主要図表:
  - Focus State 比較: [B_bend_focus_trigram_k7.png](B_bend_focus_trigram_k7.png) / [_k8](B_bend_focus_trigram_k8.png) / [bigram_k7](B_bend_focus_bigram_k7.png) / [_k8](B_bend_focus_bigram_k8.png)
  - 非対称性分析: [B_asymmetry_trigram_k7.png](B_asymmetry_trigram_k7.png) / [_k8](B_asymmetry_trigram_k8.png) / [bigram_k7](B_asymmetry_bigram_k7.png) / [_k8](B_asymmetry_bigram_k8.png)

### 主要結果

**全モデルで B-end Focus State ≠ B-start Focus State**（完全分離）

| モデル | k | B-start Focus | B-end Focus | B-end率 | B-start率(end_fs) | Fisher p (B-end vs B-start) |
|--------|---|--------------|------------|---------|------------------|-----------------------------|
| Trigram | 7 | S2 | S5 | 35.49% | 30.68% | 1.62e-05 |
| Trigram | 8 | S5 | S2 | 25.79% | 11.24% | 8.27e-58 |
| Bigram  | 7 | S2 | S0 | 28.04% | 23.06% | 1.46e-06 |
| Bigram  | 8 | S1 | S5 | 36.97% | 7.82%  | 8.59e-206 |

#### 注目点：Trigram k=7 / k=8 での Focus State の役割交代

Trigram において、k=7 と k=8 で B-start/B-end の担当状態が入れ替わっている：

| モデル | B-start Focus | 放射特性 | B-end Focus | 放射特性 |
|--------|--------------|---------|------------|---------|
| Trigram k=7 | S2 | c(0.586)、複合子音クラスタ先頭 | S5 | y(0.515)、語末的母音 |
| Trigram k=8 | S5 | y(0.640)、語末的母音 | S2 | h(0.291), r(0.227)、sibilant/liquid |

つまり「'c' 集中状態が B-start を担う（k=7）」と「'y' 集中状態が B-start を担う（k=8）」という
2 種類の解が存在し、どちらの解釈も統計的に有意に機能している。

### 評価と解釈

**「境界の方向性がモデルに符号化されている」という提案の期待は完全に確認された。**

- 4 モデル全てで B-end と B-start の特化状態が異なる（統計的に高有意）。
- これは HMM が「語基の末尾である」という情報と「語基の先頭である」という情報を
  **別の隠れ状態で符号化している**ことを意味する。
- 「境界検出器型」（同一状態が前後を担う）という帰無仮説は棄却された。

Bigram k=8 での非対称性が最も顕著（B-end 率 36.97% vs B-start 率 7.82%、p=8.59e-206）であり、
Bigram においても境界方向性の符号化が明確に生じていることが確認された。

---

## 提案 C：境界通過時の状態遷移パターン分析

### 提案の問い

> 複合語境界 ±2 位置の 5-gram (s_{-2}, s_{-1}, s_0, s_{+1}, s_{+2}) に規則的パターンがあるか。
> Trigram と Bigram でパターンの鮮明さを比較する。

### 実施スクリプト・出力ファイル

- スクリプト: [source/analysis_BC_boundary_transition.py](../source/analysis_BC_boundary_transition.py)
- レポート: [results/analysis_C_transition_report.md](analysis_C_transition_report.md)
- 主要図表:
  - ウィンドウ位置分布: [C_5gram_pos_dist_trigram_k7.png](C_5gram_pos_dist_trigram_k7.png) / [_k8](C_5gram_pos_dist_trigram_k8.png) / [bigram_k7](C_5gram_pos_dist_bigram_k7.png) / [_k8](C_5gram_pos_dist_bigram_k8.png)
  - 上位 5-gram: [C_top5gram_trigram_k7.png](C_top5gram_trigram_k7.png) / [_k8](C_top5gram_trigram_k8.png) / [bigram_k7](C_top5gram_bigram_k7.png) / [_k8](C_top5gram_bigram_k8.png)

### 主要結果

#### chi-square 検定（各ウィンドウ位置、境界 vs 単独語語中）

| モデル | k | t-2 p | t-1 p | t+0 p | t+1 p | t+2 p |
|--------|---|-------|-------|-------|-------|-------|
| Trigram B-end | 7 | 1.01e-47 | 2.04e-111 | **3.87e-187** | 8.24e-74 | 8.10e-37 |
| Trigram B-start | 7 | 1.46e-51 | 4.14e-127 | 5.95e-41 | 5.55e-18 | 8.72e-28 |
| Trigram B-end | 8 | 2.88e-49 | **1.66e-135** | 3.13e-128 | 6.03e-78 | 4.25e-79 |
| Bigram B-end | 7 | 0.00 | **0.00** | 7.37e-224 | 2.52e-133 | 1.54e-68 |
| Bigram B-end | 8 | 0.00 | **0.00** | 0.00 | 3.58e-173 | 8.01e-86 |

全モデル・全位置（±2）で p < 1e-10 以下。境界効果はウィンドウ全体に及ぶ。

#### Bigram vs Trigram のパターン鮮明さ比較

| モデル | 最大 chi2（B-end） | 最大 chi2（B-start） |
|--------|-----------------|------------------|
| Bigram k=7 | **1637.62**（t-1 位置） | **1931.30**（t-2 位置） |
| Bigram k=8 | **2313.37**（t-1 位置） | **2207.82**（t-2 位置） |
| Trigram k=7 | 876.15（t+0 位置） | 598.56（t-1 位置） |
| Trigram k=8 | 642.40（t-1 位置） | 781.18（t-2 位置） |

**Bigram の chi2 は Trigram の約 2〜3 倍**（より鮮明な境界特化パターン）。

#### 特徴的な境界 5-gram（一例）

Trigram k=8 B-start 最頻パターン：`(S5, S5, S5, S5, S5)` — B-start 8.04% vs neutral 0.72%
→ Focus State S5 が連続する遷移が B-start 前後で著しく過多。

Bigram k=8 B-end 最頻パターン：`(S5, S1, S7, S1, S7)` — B-end 7.47% vs neutral 0.20%
→ S5→S1→S7 の繰り返しが基末尾で特徴的。

### 評価と解釈

**「定型遷移パターンの存在 = HMM が複合語構造を内部的に表現している」という期待は完全に確認された。**

1. **境界効果は局所的でない**：t+0（境界位置そのもの）だけでなく、±2 の全位置が有意。
   複合語境界は「点」ではなく、前後 2 文字をまたぐ「広域な状態構造変化」として現れる。

2. **Bigram の方がパターンが鮮明**という予想外の結果が得られた。
   提案では「Trigram の方が文脈を捉えるはず」と暗示されていたが、逆の結果になった。
   考えられる解釈：
   - Trigram は s_{t-2} の情報を遷移確率に分散させるため、個々の状態が境界に特化しにくい
   - Bigram は 2 状態間の遷移のみで複合語構造を表現するため、状態が過特化する
   - 結果として Bigram の方が「境界で状態が大きく変わる」という現象が統計的に明確

3. **B-start と B-end の位置依存パターンの非対称性**：
   Trigram k=7 では B-end の最大 chi2 が t+0（境界位置）に現れ、
   B-start の最大 chi2 は t-1（境界直前）に現れる。
   これは「基の終わりは境界位置そのもので状態変化が最大」「基の始まりは直前位置での遷移変化が最大」
   という非対称な時間構造を示唆する。

---

## 総合評価：3 提案から得られた新たな仮説

提案 A・B・C の結果を統合すると、以下の新たな問いが浮かぶ。

### 確立されたこと

1. **HMM は複合語境界の方向性（B-end / B-start）を別状態で符号化している**（提案 B より確立）
2. **境界効果は ±2 位置全体に及ぶ広域な状態遷移変化として現れる**（提案 C より確立）
3. **各状態には特徴的な字種が対応する**（提案 A より確立）：
   h 独占状態、f 独占状態、e 集中状態、c 集中状態 など

### 未解決・次フェーズへの問い

1. **提案 D（基内位置勾配）への橋渡し**：提案 A で各状態の字種が判明したため、
   「語頭は c/f 系状態が多く、語末は y/h 系状態が多い」という勾配仮説を
   放射特性に基づいて事前予測した上で検証できる。

2. **Bigram > Trigram の chi2 謎の解明**：Trigram は「s_{t-2} に情報を分散させるために
   状態の特化度が下がる」という仮説を提案 F（曖昧性解消の情報量定量化）で検証できる。

3. **Trigram k=7/k=8 での Focus State 交代の頑健性**：k=7 で B-start=S2（c系）、
   k=8 で B-start=S5（y系）という解が両立することは、複合語境界の信号が
   文字レベルで複数の経路から表現できることを示唆する。提案 E の文法感度分析で検証できる。

---

_本レポートは hypothesis/06_state_characterization の分析結果に基づき作成。_
_使用スクリプト: [common.py](../source/common.py) / [analysis_A_emission.py](../source/analysis_A_emission.py) / [analysis_BC_boundary_transition.py](../source/analysis_BC_boundary_transition.py)_
