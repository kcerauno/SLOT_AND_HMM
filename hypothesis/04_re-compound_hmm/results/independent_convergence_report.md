# V8文法と Bigram HMM の独立収束：修正後の解釈

**作成日時**: 2026-03-07
**関連ディレクトリ**: [`hypothesis/04_re-compound_hmm/`](../../../hypothesis/04_re-compound_hmm/)

---

## はじめに：2つの独立した調査とは何か

本調査は、全く異なる手法で設計された2つの解析が**同じ言語構造を独立に発見する**かを確認することを目的としていた。

| 手法 | 性質 | 設計方針 |
|-----|------|---------|
| **V8スロット文法**（ヒューリスティック） | 人間が書いたルールベース | 文字の出現パターンから帰納した16スロットの正規文法 |
| **Bigram HMM**（統計学習） | データ駆動 | ヴォイニッチ語の共起パターンをViterbi/EMで学習。文法知識を持たない |

V8文法は「この語は基Aと基Bに分割できる」という分割情報を持つ。
Bigram HMMはその分割を知らずに語の隠れ状態列を学習する。
**両者が独立に同じ位置を「特別」とみなすなら、そこに真の言語構造がある**、というのが検証の論理である。

---

## 問題の発端：分類バグ

### バグの所在

`hypothesis/02_compound_hmm` および `hypothesis/03_trigram` で使用していた
[`find_v8_splits_first()`](../../../hypothesis/02_compound_hmm/source/compound_boundary_analysis.py)
関数に分類バグが存在した。

```python
# compound_boundary_analysis.py および single_vs_compound_analysis.py (共通実装)
def find_v8_splits_first(word: str):
    for i in range(1, len(word)):
        p1 = word[:i]
        if not is_base(p1):
            continue
        rest = word[i:]
        if is_base(rest):
            return (p1, rest)   # ← word 自体が is_base(word)==True でもここに到達
        ...
    return None
```

**問題の核心**：`word` 自体が単独の有効基（`is_base(word) == True`）であっても、
前半 `word[:i]` と後半 `word[i:]` の両方が有効基なら**複合語と判定して返してしまう**。

### 具体例：`aiin`（実際は単独ベース語）

```
is_base("aiin") == True   ← 単独で完全なV8基

find_v8_splits_first("aiin"):
  i=1: p1="a"  → is_base("a")==True  ✓
       rest="iin" → is_base("iin")==True  ✓
       → return ("a", "iin")   ← 誤った複合語判定！
```

`a` も `iin` も有効基なので関数は分割を返すが、`aiin` 自体が1つの完全な基だった。

### スロット重複が偽陽性を増幅する

```
SLOTS_V8[0]  = {l, r, o, y, s, v}   ← 基の先頭文字
SLOTS_V8[15] = {k, t, p, f, l, r, o, y}   ← 基の末尾文字

重複: {l, r, o, y}
```

`l`, `r`, `o`, `y` で終わる基の後ろに、`l`, `r`, `o`, `y` で始まる別の有効基が続けば
「2つに分割できる」と判定される。1基の語の末尾がこれらの文字であれば高確率で偽陽性になる。

### 定量的影響

> **検証スクリプト**: [`verify_classification.py`](../source/verify_classification.py)
> **生成レポート**: [`classification_discrepancy_report.md`](classification_discrepancy_report.md)

```
正解・単独ベース語数  : 4,714 語  (words_base_only.txt)
偽陽性件数           : 4,695 語  → 99.6% が誤って複合語と判定
偽陰性件数           : 0 語      → 正解複合語の見逃しなし

正解・複合語数    : 3,363 語
動的分類・複合語数: 8,058 語  （2.4倍に膨張）
```

**偽陽性の代表例**（`words_base_only.txt` 収録の単独ベース語が誤判定された例）：

| 単独ベース語（正解） | 誤った動的分割 | 解説 |
|-------------------|-------------|------|
| `aiin` | `a + iin` | `a` も `iin` も有効基だが `aiin` 自体が1基 |
| `aly` | `a + ly` | `a` も `ly` も有効基だが `aly` 自体が1基 |
| `ol` | `o + l` | 2文字どちらも有効基だが `ol` は単独基 |
| `oky` | `o + ky` | 同上 |
| `sor` | `s + or` | `s` も `or` も有効基だが `sor` は単独基 |
| `lol` | `l + ol` | 同上 |

**正解複合語の例**（`compound_words.txt` 収録）：

```
[2基] adal   ->  ad + al          # "ad" と "al" は別々の基
[2基] adar   ->  ad + ar
[2基] aiidal ->  aiid + al
[3基] aeeodeey ->  a + eeod + eey  # 3基構造
[4基] ackaldy  ->  a + ck + al + dy # 4基構造
```

---

## Q1. 「V8が境界と定義した位置」に集中する HMM 固有の状態とは何か

### 問いの定義

> **検証スクリプト**: [`correct_boundary_analysis.py`](../source/correct_boundary_analysis.py)
> **生成レポート**: [`corrected_boundary_report.txt`](corrected_boundary_report.txt)

**境界位置** = 複合語の「基の末尾 OR 基の先頭」
（例：`adal` = `ad + al` なら、`d` の位置が base-end、`a`(2文字目) の位置が base-start）

```
  a  d  a  l
  0  1  2  3   (文字位置)
        ↑
        境界: pos=1が base-end、pos=2が base-start
```

HMMはこの境界情報を**知らずに**各位置に状態を割り当てる。
境界位置で特定の状態が過剰に現れれば、HMMが独立に境界を「感知」していたことになる。

### 正確版の結果（k=7）

| 状態 | 境界(%) | 非境界(%) | 差分 | カウント(境界) |
|-----|--------|---------|------|-------------|
| **S6** | 16.4% | 10.9% | **+5.5%** | 1,176件 |
| **S0** | 25.6% | 21.0% | **+4.6%** | 1,832件 |
| **S1** | 14.2% | 9.8%  | **+4.5%** | 1,020件 |
| **S5** | 18.6% | 14.6% | **+4.0%** | 1,333件 |
| S3 ★Phantom | 0.0% | 0.0% | 0% | 0件 |
| S2 | 17.4% | 20.3% | **-2.9%** | 1,245件 |
| S4 | 7.8% | 23.4% | **-15.6%** | 559件 |

全状態のカイ二乗検定: **χ² = 989.47, p = 1.14e-211**（偶然ではない）

### 正確版の結果（k=8）

| 状態 | 境界(%) | 非境界(%) | 差分 | カウント(境界) |
|-----|--------|---------|------|-------------|
| **S5** | 22.5% | 14.8% | **+7.7%** | 1,610件 |
| **S6** | 17.1% | 10.3% | **+6.8%** | 1,224件 |
| S2 | 13.2% | 10.2% | +3.0% | 947件 |
| S3 | 13.0% | 11.7% | +1.3% | 932件 |
| S4 ★Phantom | 0.0% | 0.0% | 0% | 0件 |
| S1 | 17.9% | 20.6% | -2.8% | 1,281件 |
| **S0** | 1.9% | 17.9% | **-16.0%** | 136件 |

全状態のカイ二乗検定: **χ² = 1415.44, p = 1.10e-302**

### 解釈

Q1 の答えは **k=7 では S6・S0・S1・S5、k=8 では S5・S6** である。

ただし注意点がある。この分析が使う「境界位置」は **「基の末尾」と「基の先頭」を合算した位置集合**（boundary_both）である。
たとえば `adal = ad + al` の場合：

```
  a  d  |  a  l
  0  1  |  2  3
     ↑     ↑
   base-end  base-start
   (どちらも "境界位置" としてカウント)
```

したがって Q1 の状態は「基の末尾寄り」か「基の先頭寄り」かまでは特定できていない。
その分離は Q2 の分析で扱う。

---

## Q2. 「複合語内部専用状態」とは何か（Q1 と同じか）

### Q1 と Q2 は**別の分析・別の問い・別の答え**

> **検証スクリプト**: [`correct_role_analysis.py`](../source/correct_role_analysis.py)
> **生成レポート**: [`corrected_role_report.md`](corrected_role_report.md)

Q2 の分析は4グループの**比較**に基づく：

| グループ | 定義 | 例 |
|---------|------|--|
| **B-start** | 複合語の第2基以降の先頭文字 | `adal = ad+al` → pos=2（`a`）|
| **B-end**   | 複合語の各基の最終文字 | `adal = ad+al` → pos=1（`d`）|
| **S-head**  | 単独ベース語の語頭（pos=0） | `aiin` → pos=0（`a`）|
| **S-mid**   | 単独ベース語の語中央（pos=L//2） | `aiin` → pos=2（`i`）|

**問い**：「複合語の内部基先頭（B-start）」に特化して現れ、単独語の語頭（S-head）には現れない状態はどれか。

### 結果（k=7 Focus State = S2）

| グループ | **S2** 出現率 | 件数 | 合計 |
|---------|------------|------|------|
| **B-start** | **26.3%** | 946 | 3,595 |
| S-mid | 20.4% | 959 | 4,695 |
| B-end | 8.3% | 300 | 3,595 |
| **S-head** | **1.5%** | 69 | 4,695 |

Fisher検定①（B-start vs S-head）: **p = 2.21e-284, odds = 23.9**

### 結果（k=8 Focus State = S1）

| グループ | **S1** 出現率 | 件数 | 合計 |
|---------|------------|------|------|
| **B-start** | **26.2%** | 940 | 3,595 |
| S-mid | 20.3% | 954 | 4,695 |
| B-end | 9.5% | 341 | 3,595 |
| **S-head** | **1.0%** | 47 | 4,695 |

Fisher検定①（B-start vs S-head）: **p = 8.82e-305, odds = 35.0**

### Q2 の状態（S2/S1）の性質を読む

S2（k=7）の分布パターン：

```
B-start: 26.3%  ████████████████████████████
S-mid:   20.4%  ████████████████████████
B-end:    8.3%  ████████
S-head:   1.5%  █
```

- **S-mid（語中）より B-start（複合語の内部基先頭）で高い** → 単純な「語の内部」ではなく「内部からの再スタート点」に反応
- **S-head（語頭）では極端に少ない** → 語の最初の文字には現れない
- **B-end（基の末尾）では低い** → 末尾ではなく先頭に特化

つまり S2/S1 は **「語の内側で基が新しく始まる位置」に特化した状態**である。

### Q1 と Q2 がなぜ異なる状態を指すのか

```
Q1 の "境界位置" = 基末尾 ∪ 基先頭（合算）
  → 末尾に強く現れる状態 + 先頭に強く現れる状態が両方カウントされる

Q2 の "B-start" = 基先頭のみ（単独語語頭 S-head と比較）
  → 「先頭」の中でも「内部的な先頭」vs「語全体の先頭」を分離する
```

**k=7 での S2 を例に**：

| 分析 | S2 の評価 | 理由 |
|-----|---------|------|
| Q1（境界集中分析） | **-2.9%（枯渇）** | B-start=26.3% だが B-end=8.3% と低く、合算すると非境界（S-mid:20%）に負ける |
| Q2（役割分析） | **最大特化（Focus State）** | B-start=26.3% vs S-head=1.5% の対比が最大 |

同じ状態 S2 が Q1 では「枯渇」、Q2 では「最大集中」と評価される。これは分析の問い方が違うためであり、矛盾ではない。

---

## Q3. 「同定が変わった」とは何から何へ

### hypothesis/02 が同定していた状態（偽陽性データ）

> **参照スクリプト**: [`single_vs_compound_analysis.py`](../../../hypothesis/02_compound_hmm/source/single_vs_compound_analysis.py)
> **参照レポート**: [`interpretation_notes.md`](../../../hypothesis/02_compound_hmm/results/interpretation_notes.md)

hypothesis/02 の役割分析（k=7）が見つけた Focus State のパターン：

```
S-head:  77.3%  ██████████████████████████████████████████████████████████████████████████
B-end:   54.3%  ██████████████████████████████████████████████████████
B-start: 28.3%  ████████████████████████████
S-mid:    3.8%  ███
```

**語頭（S-head: 77.3%）が最大**。語の最初の位置に強く現れる「語頭状態」。

そして B-start < S-head（28% < 77%）だったため、
「複合語の内部基先頭（B-start）では語頭状態が**抑制**される」という「交絡否定逆転」が主張された。

### なぜ偽陽性がこの結果を生んだか

```
偽陽性データにおける "B-start" の実態:

  真の B-start（複合語内部の基先頭）: 3,363語分 × 1件 ≈ 3,363件
  +
  偽 B-start（単独語が誤分割された「先頭付近の分割点」）: 4,695語分
    例: aiin → "a + iin" の分割点 = pos=1 (語の2文字目)
        aly  → "a + ly"  の分割点 = pos=1
        (...これらは実際には語内部ではなく語の2文字目)
```

偽分割の多くは `a + XXX` の形（`a` は最短有効基）なので、分割点 = pos=1 に集中する。
pos=1 は語の先頭に近い位置だが、語頭（pos=0 = S-head）ほどではない。
このため「語頭状態が S-head > B-start > S-mid > B-end の順で減衰する」という
**語頭特化パターンの人工物**が生成された。

### hypothesis/04 が同定した状態（正確データ）

> **検証スクリプト**: [`correct_role_analysis.py`](../source/correct_role_analysis.py)
> **生成レポート**: [`corrected_role_report.md`](corrected_role_report.md)

hypothesis/04 の役割分析（k=7）の Focus State（S2）のパターン：

```
B-start: 26.3%  ████████████████████████████
S-mid:   20.4%  ████████████████████████
B-end:    8.3%  ████████
S-head:   1.5%  █
```

**内部基先頭（B-start: 26.3%）が最大**、**語頭（S-head: 1.5%）が最小**。

B-start >> S-head（26% >> 1%）→「複合構造の独立効果（structural effect）」

### 変化の対照表

| 項目 | hypothesis/02（偽陽性） | hypothesis/04（正確版） |
|-----|----------------------|----------------------|
| 役割分析 Focus State (k=7) | S4相当（S-head 77%が最大） | **S2**（B-start 26%が最大） |
| 役割分析 Focus State (k=8) | S0相当（S-head 大） | **S1**（B-start 26%が最大） |
| B-start vs S-head | **B-start << S-head**（交絡否定逆転） | **B-start >> S-head**（複合構造の独立効果） |
| 境界集中分析の主役 (k=7) | S4（enrichment 10.4x と報告） | **S4 は境界で枯渇（-15.6%）**、S6/S0/S1/S5 が集中 |
| 境界集中分析の主役 (k=8) | S0（boundary enriched と報告） | **S0 は境界で枯渇（-16.0%）**、S5/S6 が集中 |
| 収束の解釈 | 「語頭状態が境界にも現れるが抑制される」 | 「複合内部専用状態が語頭とは独立に存在する」 |

---

## 修正後の「独立収束」の解釈

### 収束は存在するが、機構が変わった

**変わらない事実**（両分析で共通）：

1. **Phantom State は真に縮退**：S3（k=7）、S4（k=8）はViterbi占有率 = 0%。これは偽陽性の有無にかかわらず確認された。

2. **V8境界位置でHMM状態分布は有意に変化**：χ² p < 1e-200（正確版でも維持）。HMMはV8が定義した境界付近で状態の使い方を変えている。

**修正によって明確になった収束の内容**：

```
V8文法（ヒューリスティック）:
  "この語は ad + al に分割できる"
  "分割点の直前が基の末尾、直後が基の先頭"

Bigram HMM（統計学習）:
  Viterbi デコードで、"ad + al" の分割点直後（pos=2）に
  S2 という状態を配置する確率が 26.3%
  同じ文字が語全体の先頭（pos=0）に来る場合は 1.5%
  （p = 2.21e-284 で有意な差）
```

**2つの手法は独立に「複合語内部の基の再スタート点」を検出している**。

### 旧解釈との本質的な違い

| | 旧解釈（偽陽性由来） | 新解釈（正確版） |
|--|--|--|
| HMM が発見したもの | 「語頭」という位置の汎用マーカー | 「複合語内部での基の再スタート」という複合構造固有のマーカー |
| 証拠の強さ | 弱い（B-start < S-head: 語頭状態が境界でも現れるが抑制される） | 強い（B-start >> S-head: 語頭とは本質的に異なる状態が存在する） |
| 複合仮説との整合 | 間接的（語頭状態の延長として解釈） | 直接的（複合構造専用の状態として解釈） |

---

## 関連ファイル一覧

| 用途 | ファイル |
|-----|---------|
| 正解・単独ベース語 | [`hypothesis/00_slot_model/data/words_base_only.txt`](../../../hypothesis/00_slot_model/data/words_base_only.txt) |
| 正解・複合語（分割情報付き） | [`hypothesis/00_slot_model/data/compound_words.txt`](../../../hypothesis/00_slot_model/data/compound_words.txt) |
| バグ検証スクリプト | [`hypothesis/04_re-compound_hmm/source/verify_classification.py`](../source/verify_classification.py) |
| 境界分析スクリプト（正確版） | [`hypothesis/04_re-compound_hmm/source/correct_boundary_analysis.py`](../source/correct_boundary_analysis.py) |
| 役割分析スクリプト（正確版） | [`hypothesis/04_re-compound_hmm/source/correct_role_analysis.py`](../source/correct_role_analysis.py) |
| バグ検証レポート | [`classification_discrepancy_report.md`](classification_discrepancy_report.md) |
| 境界分析レポート（正確版） | [`corrected_boundary_report.txt`](corrected_boundary_report.txt) |
| 役割分析レポート（正確版） | [`corrected_role_report.md`](corrected_role_report.md) |
| 分析サマリ | [`analysis_summary.md`](analysis_summary.md) |
| 旧スクリプト（参照用） | [`hypothesis/02_compound_hmm/source/compound_boundary_analysis.py`](../../../hypothesis/02_compound_hmm/source/compound_boundary_analysis.py) |
| 旧スクリプト（参照用） | [`hypothesis/02_compound_hmm/source/single_vs_compound_analysis.py`](../../../hypothesis/02_compound_hmm/source/single_vs_compound_analysis.py) |
| 旧解釈ノート | [`hypothesis/02_compound_hmm/results/interpretation_notes.md`](../../../hypothesis/02_compound_hmm/results/interpretation_notes.md) |

---

## 実行方法

```bash
cd /home/practi/work_voy

# Step 1: バグの定量確認（約5秒）
PYTHONPATH=.venv/lib/python3.10/site-packages python3.10 \
    hypothesis/04_re-compound_hmm/source/verify_classification.py

# Step 2: 正確版 境界分析（約10秒）
PYTHONPATH=.venv/lib/python3.10/site-packages python3.10 \
    hypothesis/04_re-compound_hmm/source/correct_boundary_analysis.py

# Step 3: 正確版 役割分析（約15秒）
PYTHONPATH=.venv/lib/python3.10/site-packages python3.10 \
    hypothesis/04_re-compound_hmm/source/correct_role_analysis.py
```

---

_本レポートは `hypothesis/04_re-compound_hmm/results/independent_convergence_report.md`_
