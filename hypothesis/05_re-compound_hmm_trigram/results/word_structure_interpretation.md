# ヴォイニッチ語彙の内部構造についての HMM 分析解釈

**作成日**: 2026-03-07
**対象**: hypothesis/04（Bigram 修正追試）および hypothesis/05（Trigram 正確版）の結果に基づく解釈

---

## 目次

1. [前提：分析の枠組みと正確性の担保](#1-前提分析の枠組みと正確性の担保)
2. [Bigram 修正後追試から言えること](#2-bigram-修正後追試から言えることhypothesis04)
   - 2.1 [Phantom State は真に縮退した無効状態](#21-phantom-state-は真に縮退した無効状態)
   - 2.2 [複合語の基先頭位置は独立した状態遷移パターンを持つ](#22-複合語の基先頭位置は独立した状態遷移パターンを持つ)
   - 2.3 [基末尾位置と基先頭位置は非対称](#23-基末尾位置と基先頭位置は非対称)
   - 2.4 [複数の状態が組み合わさって境界を符号化する](#24-複数の状態が組み合わさって境界を符号化する)
3. [Trigram 正確版から言えること](#3-trigram-正確版から言えることhypothesis05)
   - 3.1 [Phantom State の一貫性](#31-phantom-state-の一貫性bigramとtrigramで対応)
   - 3.2 [境界集中は Bigram と同様に有意](#32-境界集中は-bigram-と同様に有意)
   - 3.3 [Focus State の B-start 優位はより顕著](#33-focus-state-の-b-start-優位はより顕著)
   - 3.4 [Trigram の s_{t-2} 文脈が役割曖昧性を部分的に解消する（新発見）](#34-trigram-の-st-2-文脈が役割曖昧性を部分的に解消する新発見)
4. [総合解釈：語内構造についての統合的命題](#4-総合解釈語内構造についての統合的命題)
5. [言えること・言えないこと](#5-言えること言えないこと)

---

## 1. 前提：分析の枠組みと正確性の担保

### V8 文法による語分類

ヴォイニッチ語彙は **V8 文法**（16 スロットの SLOTS\_V8 定義）によって形式的に分類される。

| 分類 | 語数 |
|------|------|
| 単独ベース語（1基のみ） | **4,714 語** |
| 複合語（2基以上の連接） | **3,363 語** |
| 全ユニーク語型 | **8,060 語強** |

複合語が全語型の約 **41%** を占める。これは偶然ではなく、形式的な連接規則（V8 が定義する基を 2 個以上連接したもの）によって生成されている。

> **根拠**: [hypothesis/00\_slot\_model/data/compound\_words.txt](../../00_slot_model/data/compound_words.txt) および [words\_base\_only.txt](../../00_slot_model/data/words_base_only.txt)
> **生成プログラム**: [hypothesis/00\_slot\_model/source/analyze\_slot\_grammar\_v8.py](../../00_slot_model/source/analyze_slot_grammar_v8.py)

### 境界位置の定義

複合語 `base1 + base2 + ...` において：

- **B-end**: 各基（最後の基を除く）の最終文字の位置
- **B-start**: 各基（最初の基を除く）の先頭文字の位置

例：複合語 `"daiin"` が `"da" + "iin"` に分割されるなら、B-end は index=1（`a`）、B-start は index=2（`i`）。

### バグ修正と正確性確保

過去の分析（hypothesis/02, 03）では `find_v8_splits_first()` による動的分類を用いていたため、**単独ベース語の 99.6% が誤って複合語と判定**され、複合語数が 8,058 語（正解の 2.4 倍）に膨張していた。

> **根拠**: [hypothesis/04\_re-compound\_hmm/results/classification\_discrepancy\_report.md](../../04_re-compound_hmm/results/classification_discrepancy_report.md)
> **検証プログラム**: [hypothesis/04\_re-compound\_hmm/source/verify\_classification.py](../../04_re-compound_hmm/source/verify_classification.py)

本解釈は **正確版データのみ**（hypothesis/04 および hypothesis/05）に基づく。

---

## 2. Bigram 修正後追試から言えること（hypothesis/04）

> **結果ファイル**:
> - [corrected\_boundary\_report.txt](../../04_re-compound_hmm/results/corrected_boundary_report.txt)
> - [corrected\_role\_report.md](../../04_re-compound_hmm/results/corrected_role_report.md)
> - [analysis\_summary.md](../../04_re-compound_hmm/results/analysis_summary.md)
>
> **実行プログラム**:
> - [correct\_boundary\_analysis.py](../../04_re-compound_hmm/source/correct_boundary_analysis.py)
> - [correct\_role\_analysis.py](../../04_re-compound_hmm/source/correct_role_analysis.py)

### 2.1 Phantom State は真に縮退した無効状態

Bigram HMM の k=7, k=8 それぞれで、Viterbi パスに一度も現れない状態（Phantom State）が存在する。

| モデル | Phantom State | 境界での出現率 | 非境界での出現率 |
|--------|-------------|-----------|------------|
| Bigram k=7 | **S3** | 0.000% | 0.000% |
| Bigram k=8 | **S4** | 0.000% | 0.000% |

これは偽陽性データでも正確なデータでも同様であり（hypothesis/02 でも 0%）、**仮説の変化に対して頑健な事実**である。

**解釈**: EM（Baum-Welch）アルゴリズムは局所最適解に収束し、一部の状態が「使われない」結果になりうる。 k=7 または k=8 という状態数の設定が、実際の語彙変化に対してわずかに過剰であることを示唆している。ただし、この縮退状態が「存在すること自体」は、モデルが残りの状態を使って十分な表現を達成できていることの逆説的な証拠でもある。

### 2.2 複合語の基先頭位置は独立した状態遷移パターンを持つ

4グループ（B-start / B-end / S-head / S-mid）の役割分析において、最も重要な発見は **B-start と S-head（単独語語頭）が統計的に明確に区別される**ことである。

| k | Focus State | B-start 率 | S-head 率 | Fisher① p |
|---|------------|-----------|---------|---------|
| 7 | S2 | **26.31%** | **1.47%** | **2.21e-284** |
| 8 | S1 | **26.15%** | **1.00%** | **8.82e-305** |

**Fisher①検定（B-start vs S-head）の解釈**：

- 帰無仮説：「B-start が Focus State に多く現れるのは、単独語語頭（S-head）と同じ文字種制約（交絡）によるものである」
- 結果：p < 10⁻²⁸⁰ で棄却 → **交絡仮説は否定される**
- 結論：複合語における基先頭（B-start）は、**単独語の語頭とは独立した位置として、HMM が固有の状態を割り当てている**

これは、ヴォイニッチ語の語内部に「基の連接境界」という構造が存在し、その境界が **文字レベルの情報のみに依存しない** ことを示す。言い換えれば、HMM は語の「構成上の位置」（複合語内の2番目の基の先頭か、単独語の先頭か）を区別する特徴を文字列から抽出している。

さらに比較②（B-start vs S-mid）も有意（k=7: p=3.49e-10）であり、Focus State が「語中間の一般的な位置」とも区別される。

> **データ根拠**: [corrected\_role\_report.md](../../04_re-compound_hmm/results/corrected_role_report.md) §k=7 および §k=8

### 2.3 基末尾位置と基先頭位置は非対称

**B-end と B-start は同じ「境界」に関わりながら、まったく異なる出現パターンを示す**。

| 比較 | k=7 | k=8 | 解釈 |
|-----|-----|-----|------|
| B-start（Focus State 率） | 26.31% | 26.15% | 境界先頭は特異的 |
| B-end（Focus State 率） | **8.34%** | **9.49%** | 境界末尾は希薄 |
| S-mid（Focus State 率） | 20.43% | 20.32% | 語中央は中程度 |
| Fisher③: B-end vs B-start | p=1.67e-93 | p=2.71e-78 | 両者は明確に異なる |
| Fisher④: B-end vs S-mid | p=7.75e-55 | p=6.10e-43 | B-end は語中央より少ない |

**解釈**：

Focus State（S2 for k=7, S1 for k=8）は「基先頭専用状態」であり、基末尾（B-end）では逆に希薄化する。B-end は語中央（S-mid）よりも Focus State 率が低い（8% < 20%）。

これは「境界の直前（基末尾）では Focus State に移行するための準備的な遷移が起きている」という解釈が可能である。すなわち、**語の内部構造は「先頭マーキング」（base-initial marking）によって符号化**されており、末尾マーキング（base-final marking）は顕著ではない。

これは日本語の接頭辞（re-, un- など）と同様の「左端を示す標識」として機能している可能性を示唆する。

### 2.4 複数の状態が組み合わさって境界を符号化する

境界集中分析では、**単一状態ではなく複数の状態が統計的に境界位置に集中**していることが分かる。

| k | chi-square | p値 | 境界集中状態 |
|---|-----------|-----|------------|
| 7 | 989.47 | **1.14e-211** | S6, S0, S1, S5（差分 +4%〜+5.5%） |
| 8 | 1415.44 | **1.10e-302** | S5, S6, S2, S3（差分 +1%〜+7.7%） |

これほど多数の状態が境界に集中するということは、**境界という情報が 1 つの専用状態に一元化されているのではなく、複数の状態間での遷移パターンとして分散して符号化されている**ことを意味する。

逆説的なことに、境界から**最も離れた状態**（Phantom State と、境界で大きく枯渇する状態）も存在する。k=7 では S4 が境界で -15.6%、k=8 では S0 が -16.0% という大きな枯渇を示す。これらは「語の中央部や単独語でよく使われる状態」であり、**境界近傍では抑制される**というパターンを形成している。

> **データ根拠**: [corrected\_boundary\_report.txt](../../04_re-compound_hmm/results/corrected_boundary_report.txt)（k=7 S4 の -15.6%、k=8 S0 の -16.0% を参照）

---

## 3. Trigram 正確版から言えること（hypothesis/05）

> **結果ファイル**:
> - [boundary\_report.md](boundary_report.md)
> - [role\_analysis\_report.md](role_analysis_report.md)
>
> **実行プログラム**:
> - [trigram\_boundary\_analysis.py](../source/trigram_boundary_analysis.py)
> - [trigram\_role\_analysis\_corrected.py](../source/trigram_role_analysis_corrected.py)

### 3.1 Phantom State の一貫性（Bigram と Trigram で対応）

Trigram においても Phantom State が存在し、かつ両 k 値で **同一のラベル（S6）** が縮退する。

| モデル | Phantom State | Viterbi 占有率 |
|--------|-------------|------------|
| Trigram k=7 | **S6** | 0.000% |
| Trigram k=8 | **S6** | 0.000% |

Bigram（k=7: S3, k=8: S4）と Trigram（k=7, k=8: S6）で Phantom State の**ラベルは異なる**が、これはモデルの独立学習による状態の置換（label permutation）によるものであり、**現象としては完全に一致している**。

> 「k 状態で学習すると必ず 1 状態が縮退する」という現象は、
> Bigram × 2条件・Trigram × 2条件の **計4条件で独立に再現**された。

これは偶然ではなく、**ヴォイニッチ語彙が実質的に k-1 個の有効な隠れ状態で記述できる**ことを示唆する可能性がある。ただし、k を減らせば別の状態が縮退するかどうかは未検証である。

> **データ根拠**: [boundary\_report.md §Viterbi占有率サマリー](boundary_report.md)

### 3.2 境界集中は Bigram と同様に有意

| モデル | chi-square | p値 | 境界集中主要状態 |
|--------|-----------|-----|------------|
| Bigram k=7 | 989.47 | 1.14e-211 | S6(+5.5%), S0(+4.6%), S1(+4.5%), S5(+4.0%) |
| Bigram k=8 | 1415.44 | 1.10e-302 | S5(+7.7%), S6(+6.8%), S2(+3.0%) |
| **Trigram k=7** | **609.76** | **1.57e-129** | S5(Enr=1.42), S1(Enr=4.84), S2(Enr=1.23) |
| **Trigram k=8** | **504.29** | **1.00e-105** | S5(Enr=1.29), S0(Enr=1.31), S1(Enr=2.11) |

Bigram と Trigram で chi-square の絶対値は異なる（Trigram の方が小さい）が、**いずれも p < 10⁻¹⁰⁰ であり、境界集中パターンは極めて頑健に存在する**。

Trigram の chi-square が Bigram より小さい理由として考えられるのは、Trigram がより多くの遷移情報を 3-gram で分散させるため、個々の状態への集中が相対的に緩和されるからである。しかしその分、後述の s_{t-2} 情報が役割分担を助けている。

> **Trigram の境界集中状態における特記事項**：
> - S1 は両 k 値で Enrichment 比が 4.84（k=7）・2.11（k=8）と高く、境界での出現率が非境界の 2〜5 倍
> - S5 は両 k 値で境界集中が有意（k=7: p=3.57e-54, k=8: p=1.36e-24）
>
> **データ根拠**: [boundary\_report.md §各状態 Fisher 正確検定](boundary_report.md)

### 3.3 Focus State の B-start 優位はより顕著

Trigram の 4グループ役割分析では、Focus State の B-start 優位が Bigram より**さらに強い**。

| モデル | Focus State | B-start 率 | S-head 率 | Fisher① p |
|--------|------------|-----------|---------|---------|
| Bigram k=7 | S2 | 26.31% | 1.47% | 2.21e-284 |
| Bigram k=8 | S1 | 26.15% | 1.00% | 8.82e-305 |
| **Trigram k=7** | **S2** | **34.58%** | **10.18%** | **2.30e-163** |
| **Trigram k=8** | **S5** | **31.32%** | **0.00%** | **≈ 0** |

Trigram k=7 の Focus State S2 は、B-start で **34.58%** という出現率を示す（Bigram k=7 の 26.31% より高い）。
Trigram k=8 の Focus State S5 に至っては、S-head での出現率が **0.00%**（4,695 語・4,695 観測で 0 件）である。

> **注**: Trigram k=7 と Bigram k=7 で Focus State のラベルが同一（S2）になっているが、
> 独立に学習された別のモデルであり、ラベルの一致は偶然の可能性がある。
> 重要なのは現象（B-start >> S-head）が両モデルで再現されることである。

**Fisher①検定の解釈（Trigram k=8）**：
S-head = 0.00% とは、単独語の語頭 4,695 観測で Focus State S5 が一度も現れなかった、ということである。Fisher の p 値は浮動小数点のアンダーフローで 0.00e+00 と表示されており、実際には約 p < 10⁻⁴⁰⁰ 相当と考えられる。これは **Trigram が複合語の基先頭と単独語の語頭を完全に分離している**という極めて強い証拠である。

比較②（B-start vs S-mid）も有意（k=7: p=8.08e-13, k=8: p=4.68e-89）であり、Focus State は単純な語中間位置（S-mid）とも区別される。

> **データ根拠**: [role\_analysis\_report.md §サマリー](role_analysis_report.md)

### 3.4 Trigram の s_{t-2} 文脈が役割曖昧性を部分的に解消する（新発見）

**これが hypothesis/05 の最も重要な新発見である。**

#### 背景

Bigram でも Trigram でも、Focus State は B-start と B-end の**両方**に現れる（ただし B-start に偏る）。1 ステップ前の状態 s_{t-1} のみを参照する Bigram には、この曖昧性を解消する手段がない。

Trigram は s_{t-2}（2 ステップ前の状態）を参照できるため、Focus State が登場した際に「2 ステップ前に何の状態があったか」でその位置が B-end か B-start かを予測できる可能性がある。

#### 検証結果

Focus State 登場位置での **s_{t-2} 分布を B-end と B-start で比較**した結果：

**Trigram k=7（Focus State S2、B-end 766件 / B-start 1232件）**

| s_{t-2} | B-end での S2 率 | B-start での S2 率 | Fisher p | 方向 |
|---------|-----------------|-------------------|---------|------|
| S0 | **36.9%** | 19.1% | **2.50e-18** | s_{t-2}=S0 → B-end 予測 |
| S1 | 0.0% | 1.1% | 1.48e-03 | s_{t-2}=S1 → B-start 予測 |
| S3 | 0.0% | 0.6% | 4.85e-02 | s_{t-2}=S3 → B-start 予測（弱） |
| S5 | 5.2% | **21.2%** | **8.78e-25** | s_{t-2}=S5 → B-start 予測 |

→ **4/7 状態が有意**（p < 0.05）

**Trigram k=8（Focus State S5、B-end 823件 / B-start 1123件）**

| s_{t-2} | B-end での S5 率 | B-start での S5 率 | Fisher p | 方向 |
|---------|-----------------|-------------------|---------|------|
| S2 | **34.6%** | 18.3% | **4.39e-16** | s_{t-2}=S2 → B-end 予測 |
| S5 | 12.8% | **32.8%** | **1.69e-25** | s_{t-2}=S5 → B-start 予測 |

→ **2/8 状態が有意**（p < 0.05）

#### 解釈

この結果は以下を意味する：

1. **Trigram は 2 ステップ前の文脈を使って「今が基の末尾か先頭か」を部分的に予測できる**。
   具体的には、k=7 の場合、2 ステップ前に S0 があれば「今は基末尾（B-end）の直後」、
   2 ステップ前に S5 があれば「今は基先頭（B-start）の直後」というパターンが統計的に存在する。

2. **この文脈的役割分担は偶然ではない**。S0 と S5 という2状態が互いに「逆の予測」を与えており（S0: B-end 予測、S5: B-start 予測）、これは語の構成パターンに応じた系統的な文脈差を反映している。

3. **ただし解消は「部分的」にとどまる**。有意な状態は k=7 で 4/7、k=8 で 2/8 であり、
   B-end と B-start の全面的な分離は達成されていない。

4. **hypothesis/03 の「文脈差なし」という結論は誤りだった**。
   あの結論は偽陽性データ（8,058 語）＋近似 Viterbi（logsumexp 周辺化）の組み合わせによる
   アーティファクトであった。正確なデータと exact Viterbi を使うと文脈分離が現れる。

> **データ根拠**: [role\_analysis\_report.md §B. s\_{t-2}文脈曖昧性解消](role_analysis_report.md)

---

## 4. 総合解釈：語内構造についての統合的命題

以下の命題は、Bigram（k=7, k=8）と Trigram（k=7, k=8）の **4 条件で独立に再現**された知見に基づく。

### 命題 A：ヴォイニッチ語彙の内部には V8 文法境界と対応した構造が存在する

根拠：
- 境界 vs 非境界の状態分布差が chi-square 検定で全4条件 p < 10⁻¹⁰⁰（Bigram: p = 10⁻²¹¹〜10⁻³⁰²、Trigram: p = 10⁻¹⁰⁵〜10⁻¹²⁹）
- これは「V8 文法が定義する境界」が単なる記号的定義に留まらず、**実際の文字系列の統計的パターンに対応している**ことを意味する

### 命題 B：語の先頭マーキングが主要な符号化機構である

根拠：
- B-start >> S-head（p < 10⁻¹⁶³〜p ≈ 0）は、**交絡仮説（文字種制約の重複）を否定**する
- B-start >> B-end（Fisher③, p < 10⁻¹⁵〜10⁻⁹³）は、境界符号化が「先頭」に偏ることを示す
- B-end << S-mid（Fisher④, p < 10⁻¹⁰〜10⁻⁵⁵）は、基末尾が語中間より Focus State に少ないことを示す

**つまり「基の連接境界は先頭側でマークされる」という語構造上の非対称性が数値として現れている。**

### 命題 C：HMM の状態は語の「位置的役割」を捉える

根拠：
- 4グループ（B-start / B-end / S-head / S-mid）で状態分布が系統的に異なる
- 4 条件すべてで「複合語の基先頭に集中する状態」が独立に同定された
- これらは「文字の種類」だけでなく「語内の位置」に依存した確率的パターンを反映する

HMM は unsupervised（非教師あり）で学習されており、境界ラベルを一切見ていない。
にもかかわらず境界に対応した状態が現れることは、**語内部の統計的規則性が文字系列に埋め込まれている**ことの証拠である。

### 命題 D：Trigram の 2-ステップ文脈が役割分担を追加的に符号化する

根拠：
- s_{t-2} による B-end / B-start 予測が k=7 で 4/7 状態、k=8 で 2/8 状態で有意
- 特に k=7: S0（B-end 予測）と S5（B-start 予測）が逆方向の予測を示す

これは、ヴォイニッチ語の語内部に **局所的な文脈依存性**が存在することを示す。
すなわち「基の末尾」と「基の先頭」は、その直前 2 文字の状態列が統計的に異なる。

---

## 5. 言えること・言えないこと

### 言えること（統計的に示された事実）

| 命題 | 根拠の強度 |
|------|---------|
| V8 境界位置で状態分布が有意に偏る | 4条件全て p < 10⁻¹⁰⁰ |
| 基先頭（B-start）には特化した HMM 状態が割り当てられる | 4条件全て p < 10⁻¹⁶³ |
| 基先頭と語頭（単独語）は統計的に区別可能 | 同上 |
| 基先頭と基末尾は非対称（先頭マーキング優位） | Bigram 4条件で p < 10⁻¹⁵ |
| Trigram は s_{t-2} 文脈で境界役割を部分的に解消できる | k=7: 4/7, k=8: 2/8 状態有意 |
| k=7 または k=8 の HMM では常に 1 状態が縮退する | Bigram・Trigram 4条件で観測 |

### 言えないこと（未検証・推論の限界）

| 命題 | 理由 |
|------|------|
| V8 文法がヴォイニッチ語の「真の」形態論を反映する | V8 文法は研究者による仮説的定義であり、言語学的正当性は未証明 |
| Phantom State が言語構造と直接対応する | Phantom State は EM の数値的縮退であり、言語的意味の存在は不明 |
| Trigram の文脈分離が「意識的な文法規則」を反映する | 確率的パターンの存在は規則性の証拠だが、「意図的な文法」を意味しない |
| B-end のマーキングが存在しない | 分析は Focus State 中心であり、B-end を専門的に符号化する別の状態が存在する可能性は排除できない |
| この結果がヴォイニッチ語の「意味」に繋がる | HMM 分析は文字列の表層統計であり、意味構造への接続は別途必要 |

---

## 参照：根拠ファイルと作成プログラムの対応表

| 知見 | 結果ファイル | 作成プログラム |
|-----|------------|------------|
| 分類バグの定量（99.6% 偽陽性） | [classification\_discrepancy\_report.md](../../04_re-compound_hmm/results/classification_discrepancy_report.md) | [verify\_classification.py](../../04_re-compound_hmm/source/verify_classification.py) |
| Bigram 境界集中（chi-square, Fisher） | [corrected\_boundary\_report.txt](../../04_re-compound_hmm/results/corrected_boundary_report.txt) | [correct\_boundary\_analysis.py](../../04_re-compound_hmm/source/correct_boundary_analysis.py) |
| Bigram 4グループ役割分析 | [corrected\_role\_report.md](../../04_re-compound_hmm/results/corrected_role_report.md) | [correct\_role\_analysis.py](../../04_re-compound_hmm/source/correct_role_analysis.py) |
| Bigram 結果総合サマリ | [analysis\_summary.md](../../04_re-compound_hmm/results/analysis_summary.md) | 上記3プログラム |
| Trigram Phantom State・境界集中 | [boundary\_report.md](boundary_report.md) | [trigram\_boundary\_analysis.py](../source/trigram_boundary_analysis.py) |
| Trigram 4グループ役割分析・s_{t-2} 文脈 | [role\_analysis\_report.md](role_analysis_report.md) | [trigram\_role\_analysis\_corrected.py](../source/trigram_role_analysis_corrected.py) |
| V8 語彙分類（複合語・単独語リスト） | [compound\_words.txt](../../00_slot_model/data/compound_words.txt)・[words\_base\_only.txt](../../00_slot_model/data/words_base_only.txt) | [analyze\_slot\_grammar\_v8.py](../../00_slot_model/source/analyze_slot_grammar_v8.py) |

---

_本ドキュメントは hypothesis/04 および hypothesis/05 の数値結果を解釈するものであり、
新たな分析は含まない。各数値は上記結果ファイルから直接引用している。_
