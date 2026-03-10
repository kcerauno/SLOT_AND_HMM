# ヴォイニッチ手稿 HMM 研究 — 全体概観

作成日: 2026-03-08
対象ディレクトリ: `hypothesis/00` 〜 `hypothesis/10`

---

## 研究の問いと全体方針

ヴォイニッチ手稿の単語列に対して **隠れマルコフモデル (HMM)** を適用し、
独立に設計された **V8 スロット文法**（複合語仮説）の構造と HMM が統計的に学習する構造が
一致するかを多段階で検証する。

中心的な問い：

> **ヴォイニッチ語の単語は「複数の基（ベース）を連結した複合語」として構成されており、
> HMM はその複合語境界を自律的に学習しているか？**

---

## 分析フェーズ一覧

| hypothesis | テーマ | 主な手法 | 状態 |
|-----------|--------|---------|------|
| [00](#h00) | V8 スロット文法の定義 | 規則文法 | 基盤 |
| [01](#h01) | Bigram HMM 学習・状態解釈 | Bigram HMM (k=7,8) | 確立 |
| [02](#h02) | 複合語境界 × HMM 相互検証（初版・バグあり） | 境界分析 | ★バグ修正済み |
| [03](#h03) | Trigram HMM + 役割曖昧性解消 | Trigram HMM (k=7,8) | バグあり |
| [04](#h04) | 分類バグ修正 — 正確版 Bigram 再検証 | Bigram 役割分析 | 確立 |
| [05](#h05) | 正確版 Trigram 再検証 | Trigram 役割分析 | 確立 |
| [06](#h06) | 状態特性化 (提案 A・B・C) | 放射分析・境界分析・遷移分析 | 確立 |
| [07](#h07) | 基内位置勾配 (提案 D) | 位置回帰分析 | 確立 |
| [08](#h08) | V8 文法感度 E-1・E-2（E-2 にバグ） | 置換検定・文法バリアント | ★バグ修正済み |
| [09](#h09) | 修正版 E-2 + 新規 E-3 | 貪欲分割・語集合シャッフル | 確立 |
| [10](#h10) | s_{t-2} 曖昧性解消の情報量 (提案 F) | 相互情報量 | 確立 |

---

## 確立された主要知見

### 1. HMM はヴォイニッチ語を 5〜6 の機能的クラスに分類する

k=7・8 の両モデルで再現的に出現する文字クラスター：

| クラス | 代表文字 | 機能的役割 |
|--------|---------|-----------|
| E-cluster | `e`, `ee` | 母音連続・コア音節核 |
| CH-onset | `c`, `ch`, `sh` | 語頭子音クラスター |
| I-run | `i`, `in`, `r`, `l` | 鼻音・流音連続（語末接尾辞候補） |
| O/Q-core | `o`, `q`, `l` | 語核音節（q-o 制約を内包） |
| H-medial | `h`, `hh` | 語中コーダ子音（語頭位置とは別機能） |
| Gallows/f | `f` | Trigram で独立した f 専用状態 |

### 2. 複合語境界には方向性があり、B-end と B-start を別状態が担う

全 4 モデル（Bigram k=7/8, Trigram k=7/8）で B-end Focus State ≠ B-start Focus State。

| モデル | k | B-start Focus | B-end Focus | Fisher p |
|--------|---|--------------|------------|----------|
| Trigram | 7 | S2 (c 系) | S5 (y 系) | 1.62e-05 |
| Trigram | 8 | S5 (y/v 系) | S2 (h/r 系) | 8.27e-58 |
| Bigram  | 7 | S2 (a/y/o 系) | S0 | 1.46e-06 |
| Bigram  | 8 | S1 (a/y/o 系) | S5 | 8.59e-206 |

### 3. 境界効果は ±2 位置全体に及ぶ

複合語境界の状態遷移パターンは境界点のみならず前後 2 文字にわたって有意 (p < 1e-10)。
Bigram の方が Trigram よりもパターンが鮮明（chi2 約 2〜3 倍）。

### 4. 境界シグナルは V8 文法定義に依存しない（頑健）

- **置換検定 (E-1)**: 4 モデル中 3 モデルで H₀ 棄却 (p < 0.001)
- **文法バリアント (E-2)**: SLOTS_V8 の ±1 文字変更でも Focus State・方向性が完全に不変
- **語集合検定 (E-3)**: 複合語集合は単独語プールと HMM 的に異質 (3/4 モデルで有意)

### 5. s_{t-2} は境界方向の曖昧性を実効的に解消する

Trigram k=8 で acc +4.2%（MI = 0.051 bits）。
Trigram のみ s_{t-2} によって B-end 予測への反転が発生する
（Bigram は常に B-start 多数）。

---

## バグ修正の系譜

分析の途中で発見された 2 件の重大バグと修正方法をまとめる。

| バグ | 発見箇所 | 影響 | 修正先 |
|-----|---------|------|--------|
| `find_v8_splits_first()` が単独語の 99.6% を複合語と誤判定 | hypothesis/02, 03 | Focus State の同定が誤り（S4→S2, S0→S1） | hypothesis/04, 05 |
| `find_splits_variant()` が first-fit アルゴリズムを使用（貪欲と異なる） | hypothesis/08 (E-2) | V0_baseline と compound_words.txt が 99.6% で不一致 → 「逆転」という人工産物 | hypothesis/09 |

---

## 各フェーズ詳細

---

<a name="h00"></a>
## hypothesis/00: V8 スロット文法の定義

**ディレクトリ**: [hypothesis/00_slot_model/](../hypothesis/00_slot_model/)

V8 スロット文法（`SLOTS_V8`）はヴォイニッチ語の1基（ベース）を最大 16 スロットの
文字クラス連鎖として定義する。スロット定義は `parse_v4b.py` の `SLOTS_V8` と同一。

```
基の先頭スロット: {l, r, o, y, s, v}   (slot 0)
基の末尾スロット: {k, t, p, f, l, r, o, y}  (slot 15)
               {s}              (slot 13)
               {y}              (slot 14)
```

**貪欲アルゴリズム** (`parse_base_greedy`) により全ユニーク単語を分割：

- compound_words.txt: **3,363 語**（2基以上の複合語）
- words_base_only.txt: **4,695 語**（単独ベース語）

主要スクリプト: [parse_v4b.py](../hypothesis/00_slot_model/source/parse_v4b.py)
複合語データ: [compound_words.txt](../hypothesis/00_slot_model/data/compound_words.txt)

---

<a name="h01"></a>
## hypothesis/01: Bigram HMM — 初期状態解釈

**ディレクトリ**: [hypothesis/01_bigram/](../hypothesis/01_bigram/)
**主要レポート**: [HMM_state_analysis_report.md](../hypothesis/01_bigram/results/hmm_state_analysis/HMM_state_analysis_report.md)

### モデル仕様

| パラメータ | 値 |
|---|---|
| モデル種別 | Full HMM（Full 遷移行列） |
| 状態数 | k = 7, 8 |
| 学習試行数 | 20 回ランダム再スタート（最大尤度採用） |
| アルゴリズム | Baum-Welch（PyTorch GPU バッチ並列化） |

BIC は k=7→8 で単調減少。Full HMM は Left-to-Right より k=7 で約 8,000 nats 優位
→ ヴォイニッチ語の文字遷移は単純な一方向連鎖に収まらない。

### 発見された機能的クラス (k=7)

| 状態 | 主要文字 | 命名 | 位置傾向 |
|------|---------|------|---------|
| S0 | e, ee | E-cluster | 語頭・語中優位、語末回避 |
| S1 | c, ch, sh | C-onset | 語頭優位 |
| S2 | a, l, r, y | A-frame | 語頭・語末の両方 |
| S3 | *(0%)* | Phantom | 縮退 |
| S4 | o, q, l | O/Q-cluster | 語頭・語中優位 |
| S5 | i, n, r | I-run | 全位置均等 |
| S6 | h, hh | H-coda | 語中・語末優位 |

**k=8 では C-onset が語頭 ch（S6）と語中 h（S2）に分化**する。

モデルキャッシュ: [hmm_model_cache/](../hypothesis/01_bigram/results/hmm_model_cache/)

---

<a name="h02"></a>
## hypothesis/02: 複合語境界 × HMM 相互検証（★バグあり版）

**ディレクトリ**: [hypothesis/02_compound_hmm/](../hypothesis/02_compound_hmm/)
**主要レポート**: [compound_boundary_report.md](../hypothesis/02_compound_hmm/results/compound_boundary_report.md)
**解釈ノート**: [interpretation_notes.md](../hypothesis/02_compound_hmm/results/interpretation_notes.md)

### 検証仮説

> Phantom State（k=7: S3, k=8: S4）は複合語境界の遷移ハブとして機能するか？

### 結果（バグあり版・後に修正）

- Phantom State は Viterbi に一切出現しない → 仮説は直接棄却
- `find_v8_splits_first()` のバグにより複合語数が誤って **8,058 語**（正確には 3,363 語）
- バグ版では k=7 S4、k=8 S0 が「境界集中状態」と誤同定された

> **注**: この分析の結果は hypothesis/04 で全面修正されている。
> 交絡検定（B-start vs S-head）の「逆転（B-start << S-head）」もバグに由来するアーティファクト。

---

<a name="h03"></a>
## hypothesis/03: Trigram HMM 開発

**ディレクトリ**: [hypothesis/03_trigram/](../hypothesis/03_trigram/)
**主要レポート**: [role_analysis_report.md](../hypothesis/03_trigram/results/role_analysis_report.md)

k=5〜8 の Trigram HMM を学習・比較。Trigram は遷移確率として
P(s_t | s_{t-2}, s_{t-1}) を用いるため k×k の状態ペアをキーとして保持する。

### Bigram との役割比較（バグあり版）

| モデル | k | Focus State | s_{t-2} で曖昧性解消 |
|--------|---|------------|---------------------|
| Trigram | 7 | S6 | なし（S6 は縮退状態と後に判明） |
| Trigram | 8 | S6 | なし |
| Bigram | 7 | S4 | あり（B-end と B-start で s_{t-1} 分布が有意に異なる） |
| Bigram | 8 | S1 | あり |

> **注**: Trigram の Focus State S6 は実際には Phantom（縮退）状態であり、
> 分析に使用したバグ版複合語分類が原因。hypothesis/05 で修正済み。

---

<a name="h04"></a>
## hypothesis/04: 分類バグ修正 — 正確版 Bigram 再検証

**ディレクトリ**: [hypothesis/04_re-compound_hmm/](../hypothesis/04_re-compound_hmm/)
**主要レポート**: [analysis_summary.md](../hypothesis/04_re-compound_hmm/results/analysis_summary.md)
**修正役割分析**: [corrected_role_report.md](../hypothesis/04_re-compound_hmm/results/corrected_role_report.md)

### バグの定量確認

| 指標 | 値 |
|-----|---|
| 正解・複合語数 | **3,363 語** |
| バグ版・複合語数 | 8,058 語（2.4 倍膨張） |
| 単独ベース語の偽陽性率 | **99.6%**（4,695 語中 4,695 語が誤判定） |

### 正確版 Bigram 役割分析

| k | Focus State | B-start | S-head | Fisher① p | 判定 |
|---|------------|---------|--------|----------|------|
| 7 | **S2** | 26.31% | 1.47% | 2.2e-284 | **構造効果あり** |
| 8 | **S1** | 26.15% | 1.00% | 8.8e-305 | **構造効果あり** |

**B-start >> S-head**（バグ版は逆転していた）という正しい結果が得られた。
複合語境界の独立効果が確認された。

---

<a name="h05"></a>
## hypothesis/05: 正確版 Trigram 再検証

**ディレクトリ**: [hypothesis/05_re-compound_hmm_trigram/](../hypothesis/05_re-compound_hmm_trigram/)
**主要レポート**: [role_analysis_report.md](../hypothesis/05_re-compound_hmm_trigram/results/role_analysis_report.md)
**次フェーズ提案**: [next_analysis_proposals.md](../hypothesis/05_re-compound_hmm_trigram/results/next_analysis_proposals.md)

hypothesis/03 の 5 件のバグを修正した上で Trigram の役割分析を再実施。

修正内容:
1. `find_v8_splits_first()` → compound_words.txt 直接参照
2. 近似 Viterbi（logsumexp） → 厳密な k×k delta 行列
3. バックトラッキング off-by-one (`psi_list[t_back-2]` → `t_back-1`)
4. `start_trans` の欠落 → 追加
5. ハードコードされた Focus State S6 → データ駆動で選択

### 正確版 Trigram 役割分析

| k | Focus State | B-start | S-head | Fisher① p | 判定 |
|---|------------|---------|--------|----------|------|
| 7 | **S2** | 34.58% | 10.18% | 2.30e-163 | **構造効果あり** |
| 8 | **S5** | 31.32% | 0.00%  | ≈0        | **構造効果あり** |

**Phantom States**: k=7 S6, k=8 S6（Viterbi 占有率 0%）

**s_{t-2} の文脈分離**: k=7 で 4/7 状態が有意、k=8 で 2/8 状態が有意
→ Trigram は B-end/B-start を部分的に区別できる。

---

<a name="h06"></a>
## hypothesis/06: 状態特性化（提案 A・B・C）

**ディレクトリ**: [hypothesis/06_state_characterization/](../hypothesis/06_state_characterization/)
**総合評価レポート**: [ABC_assessment_report.md](../hypothesis/06_state_characterization/results/ABC_assessment_report.md)

hypothesis/05 終了後に策定された次フェーズ提案 A〜C を実施した。

---

### 提案 A: 放射確率行列の文字レベル分析

**詳細レポート**: [analysis_A_emission_report.md](../hypothesis/06_state_characterization/results/analysis_A_emission_report.md)

各状態が「どの文字を好んで放射するか」を emiss 行列（k × 32）から分析。

#### Focus State の放射特性

| モデル | k | Focus State | 上位文字 | 文字グループ |
|--------|---|------------|---------|------------|
| Bigram | 7 | S2 | a(0.357), y(0.342), o(0.231) | **vowel 系** |
| Bigram | 8 | S1 | a(0.352), y(0.340), o(0.217) | **vowel 系** |
| Trigram | 7 | S2 | c(0.586), a(0.186), m(0.106) | **other（c 集中）** |
| Trigram | 8 | S5 | y(0.640), v(0.234), l(0.079) | **vowel（y 独占）** |

KL ダイバージェンス（Focus ∥ 基先頭文字分布）：
- **Trigram k=7 のみ** 全状態平均より小さい（1.048 < 2.387）→ 基先頭文字分布に最も近い
- Trigram k=8 S5 は最遠（KL=6.705）

特化した単機能状態も確認：
- **h 独占状態**（Bigram k=7 S6: 86%、k=8 S2: 99%）
- **f 独占状態**（Trigram k=7/k=8 S1: 98%）
- **e 集中状態**（Bigram k=8 S3: 81%）

---

### 提案 B: B-end 特化状態の同定

**詳細レポート**: [analysis_B_bend_report.md](../hypothesis/06_state_characterization/results/analysis_B_bend_report.md)

#### 結果：全モデルで B-end Focus ≠ B-start Focus

| モデル | k | B-start Focus | B-end Focus | Fisher p (B-end vs B-start) |
|--------|---|--------------|------------|----------------------------|
| Trigram | 7 | S2 (c 系) | S5 (y 系) | 1.62e-05 |
| Trigram | 8 | S5 (y/v 系) | S2 (h/r 系) | 8.27e-58 |
| Bigram  | 7 | S2 (vowel) | S0 | 1.46e-06 |
| Bigram  | 8 | S1 (vowel) | S5 | 8.59e-206 |

Trigram k=7 と k=8 で B-start/B-end の担当状態が入れ替わる（S2 ⇔ S5）。
複合語境界は「方向性のある境界マーキング」として HMM に符号化されている。

---

### 提案 C: 境界通過時の状態遷移パターン分析

**詳細レポート**: [analysis_C_transition_report.md](../hypothesis/06_state_characterization/results/analysis_C_transition_report.md)

複合語境界 ±2 位置の 5-gram 状態列を集計・検定。

- **全モデル・全位置（±2）で p < 1e-10**
- 境界効果は点ではなく「前後 2 文字をまたぐ広域な状態構造変化」

| モデル | 最大 chi2（B-end） | 最大 chi2（B-start） |
|--------|-----------------|------------------|
| Bigram k=8 | **2313** (t-1) | **2208** (t-2) |
| Bigram k=7 | **1638** (t-1) | **1931** (t-2) |
| Trigram k=7 | 876 (t+0) | 599 (t-1) |
| Trigram k=8 | 642 (t-1) | 781 (t-2) |

**Bigram の chi2 が Trigram の約 2〜3 倍**（予想に反して Bigram の方が境界特化が鮮明）。
解釈：Trigram は s_{t-2} に情報を分散させるため個々の状態の境界特化度が下がる。

---

<a name="h07"></a>
## hypothesis/07: 基内位置勾配（提案 D）

**ディレクトリ**: [hypothesis/07_base_position_gradient/](../hypothesis/07_base_position_gradient/)
**詳細レポート**: [analysis_D_gradient_report.md](../hypothesis/07_base_position_gradient/results/analysis_D_gradient_report.md)

各基の内部で正規化位置 `pos = i / max(len(base)-1, 1)` を定義し、
線形回帰で位置勾配（head 集中 = 負傾き、tail 集中 = 正傾き）を定量化。

### 全 4 モデルで slope ≈ ±0.22〜0.25 の状態が存在（安定した位置符号化）

| モデル | k | 最強 head 集中状態 | slope | 最強 tail 集中状態 | slope |
|--------|---|-----------------|-------|-----------------|-------|
| Trigram | 7 | S0 (m/v) | −0.223 | **S5 (y/g)** | +0.252 |
| Trigram | 8 | S3 (c/a) | −0.138 | **S5 (y/v)** | +0.242 |
| Bigram  | 7 | S4 (o/l) | −0.245 | S5 (i/r) | +0.243 |
| Bigram  | 8 | S0 (e/d) | −0.170 | S7 | +0.248 |

### 事前仮説との照合

**Trigram k=7 のみ完全一致**：
- B-start Focus S2（c 系）→ head 集中（slope=−0.030 ***）
- B-end Focus S5（y 系）→ tail 集中（slope=+0.252 ***）

他 3 モデルでは B-start Focus が tail 集中（slope ≈ +0.17〜+0.24）。
vowel 系（a/y/o）を放射する B-start Focus は語末にも多く出現するためと解釈できる。
B-start 特化（境界シグナル）と基内 tail 集中（放射特性）は矛盾しない。

---

<a name="h08"></a>
## hypothesis/08: V8 文法感度分析（E-1・E-2）★E-2 にバグあり

**ディレクトリ**: [hypothesis/08_v8_sensitivity/](../hypothesis/08_v8_sensitivity/)
**レポート（旧版・一部誤り）**: [analysis_report.md](../hypothesis/08_v8_sensitivity/results/analysis_report.md)

### E-1: 置換検定（境界ラベルシャッフル）

N=1000 回のシャッフルで帰無分布を構築。統計量: chi2（2×2 分割表）。

| モデル | k | 観測 chi2 | 帰無中央値 | 置換 p | 判定 |
|--------|---|---------|---------|--------|------|
| Trigram | 7 | 736.5 | 472.0 | < 0.001 | H₀ 棄却 |
| Trigram | 8 | 1701.7 | 1384.6 | < 0.001 | H₀ 棄却 |
| Bigram  | 7 | 1169.7 | 1116.5 | 0.077 | 棄却失敗 |
| Bigram  | 8 | 1227.5 | 1101.0 | < 0.001 | H₀ 棄却 |

→ E-1 の結果は hypothesis/09 でもそのまま採用されている。

### E-2: 文法バリアント（★バグ版：first-fit アルゴリズム使用）

`find_splits_variant()` が first-fit（最短先頭基優先）を使用したため、
compound_words.txt（貪欲）と 99.6% の語で分割点が異なる人工産物が発生。
→ hypothesis/09 で修正版を再実施。

---

<a name="h09"></a>
## hypothesis/09: 修正版 E-2 + 新規 E-3（語集合シャッフル）

**ディレクトリ**: [hypothesis/09_v8_variants_greedy/](../hypothesis/09_v8_variants_greedy/)
**総合レポート（修正版）**: [analysis_report.md](../hypothesis/09_v8_variants_greedy/results/analysis_report.md)
**E-2 修正版レポート**: [variant_report_greedy.md](../hypothesis/09_v8_variants_greedy/results/variant_report_greedy.md)
**E-3 レポート**: [wordset_report.md](../hypothesis/09_v8_variants_greedy/results/e3_wordset/wordset_report.md)
**HMM 派生文法提案**: [hmm_derived_grammar_proposal.md](../hypothesis/09_v8_variants_greedy/results/hmm_derived_grammar_proposal.md)

### E-2（修正版）: 文法バリアント感度分析

`parse_base_greedy`（`parse_v4b.py` と等価）で再実装し、V0_baseline が compound_words.txt と 100% 一致することを確認。

| バリアント | 変更内容 | 複合語数 | k=7 Focus/B-start | k=8 Focus/B-start |
|-----------|---------|--------|-------------------|--------------------|
| V0_baseline | 基準 | 3,363 | **S2** / 34.6% | **S5** / 31.3% |
| V1_expand_vow | slot2 に 'i' 追加 | 3,357 | **S2** / 34.6% | **S5** / 31.4% |
| V2_revert_z | slot1 から 'z' 除去 | 3,361 | **S2** / 34.5% | **S5** / 31.3% |
| V3_expand_va2 | slot10 に 'e' 追加 | 3,229 | **S2** / **37.3%** | **S5** / 30.7% |

→ **全バリアントで Focus State（S2/S5）・方向性が完全に不変**。完全に頑健。

### E-3: 語集合シャッフル検定

単独語プール（4,695 語）から 3,363 語をランダムサンプリング + ランダム境界を N=1000 回。

| モデル | k | 観測/帰無比 | 語集合 p | 判定 |
|--------|---|-----------|---------|------|
| Trigram | 7 | 1.51× | < 0.001 | 語集合効果あり |
| **Trigram** | **8** | **778×** | **< 0.001** | **語集合効果あり（極大）** |
| Bigram  | 7 | 1.03× | 0.172 | なし |
| Bigram  | 8 | 1.12× | 0.001 | あり |

特に **Trigram k=8 の 778×** は S5（y/v 系）が複合語集合の本質的な性質であることを示す。

#### E-1 × E-3 組み合わせ解釈

| モデル | E-1（境界位置効果）| E-3（語集合効果）| 解釈 |
|--------|-----------------|----------------|------|
| Trigram k=7 | 有意 | 有意 | 境界位置と語集合の両方が独立に寄与 |
| Trigram k=8 | 有意 | **有意（778×）** | 語集合効果が圧倒的、境界位置も追加寄与 |
| Bigram k=7  | 非有意 | 非有意 | HMM 状態空間構造のみ（語頭母音優位性） |
| Bigram k=8  | 有意 | 有意 | 両方が寄与（ほぼ同程度） |

**Bigram k=7 の解釈**: S2（a/y/o 系）はヴォイニッチ語彙全体に広く分布する語頭母音優位性を
学習しており、複合語境界に特有の現象ではない。
複合語の基と単独語は先頭文字分布がほぼ同一→ bigram では 2 者を区別できない。

---

<a name="h10"></a>
## hypothesis/10: s_{t-2} 曖昧性解消の情報量（提案 F）

**ディレクトリ**: [hypothesis/10_stm2_disambiguation_info/](../hypothesis/10_stm2_disambiguation_info/)
**詳細レポート**: [analysis_F_report.md](../hypothesis/10_stm2_disambiguation_info/results/analysis_F_report.md)

Focus State 出現時の境界方向（B-start / B-end）を s_{t-2} によってどれだけ区別できるかを
相互情報量 MI = I(label; s_{t-2} | s_t = Focus) で定量化。

### サマリー比較表

| モデル | k | Focus | H(label) [bits] | MI [bits] | acc_without | acc_with | 向上 |
|--------|---|-------|----------------|----------|------------|---------|------|
| Bigram  | 7 | S2 | 0.795 | 0.090 | 76.0% | 76.2% | +0.2% |
| Bigram  | 8 | S1 | 0.836 | 0.054 | 73.4% | 74.2% | +0.8% |
| Trigram | 7 | S2 | 0.957 | 0.069 | 62.2% | 65.4% | +3.2% |
| **Trigram** | **8** | **S5** | **0.982** | **0.051** | **57.9%** | **62.1%** | **+4.2%** |

- Trigram の H(label) ≈ 1.0 bit（B-start/B-end がほぼ 50/50 の汎用境界マーカー）
- Bigram の H(label) ≈ 0.8 bit（B-start 偏重）
- **実効的な曖昧性解消は Trigram k=8 が最大（+4.2%）**
- Trigram のみ s_{t-2} = 特定状態で B-end 予測への反転が生じる
  - Trigram k=7: s_{t-2} = S0 のとき P(B-end) = 0.544
  - Trigram k=8: s_{t-2} = S2 のとき P(B-end) = 0.578

---

## 結論の統合

一連の分析を通じて確立された結論を階層的に整理する。

### 階層 1: ヴォイニッチ語の文字構造（hypothesis/01）

HMM は監督なしで 5〜6 の機能的文字クラスを学習する。
これらは k を変えても再現的に出現し、ランダム文字列では生まれない位置特化型の機能分化を示す。

### 階層 2: 複合語構造の統計的実在（hypothesis/04, 05）

V8 スロット文法が定義する 3,363 語の複合語境界において、
HMM の Focus State は B-start に顕著に集中する（B-start >> S-head, p < 1e-160 以上）。
これは文法設計と統計学習の独立した二手法が同じ構造を捉えている相互検証である。

### 階層 3: 境界の方向性符号化（hypothesis/06 提案 B）

全 4 モデルで B-end と B-start の担当状態が分離しており（p < 1e-5 以上）、
HMM は「基の終わり」と「基の始まり」を区別して学習している。
境界は「検出器」型ではなく「方向性マーカー」型として機能する。

### 階層 4: 頑健性の確認（hypothesis/09）

- 境界ラベルをシャッフルしても有意（E-1: 3/4 モデル）
- SLOTS_V8 の ±1 文字変更に対して完全不変（E-2）
- 単独語プールからのランダムサンプルでは再現できない（E-3: 3/4 モデル）

### 階層 5: 文脈的曖昧性解消（hypothesis/10）

Trigram の 2 ステップ前文脈は境界方向の予測精度を +3〜4% 向上させる（MI = 0.05〜0.09 bits）。
Bigram では全 s_{t-2} で B-start が多数のまま逆転しないが、
Trigram では特定の s_{t-2} 状態で B-end 予測に反転する。

---

## 今後の方向性

hypothesis/09 の `hmm_derived_grammar_proposal.md` で提案された
**HMM 派生スロット文法（状態数 k=16〜20 への拡張）** が次候補として示されている。

HMM の放射分布（B[state, char]）と遷移確率（A[i,j]）から
V8 スロット構造をデータ駆動で精緻化することで、
複合語構造の形式文法的記述を改善できる可能性がある。

ただし根本的な制約として、V8 はポジション型テンプレート（「基の何番目か」を定義する）
であるのに対し、HMM は文脈依存型（「直前の状態」のみを参照する）という
アーキテクチャ上の差異は k を増やしても解消されない。

---

*本ドキュメントは `hypothesis/00` 〜 `hypothesis/10` の全マークダウンレポートを統合して作成。*
*各詳細リンクは `hypothesis/` ディレクトリからの相対パス。*
