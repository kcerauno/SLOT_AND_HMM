# 分析グループ用語解説：B-start / B-end / S-head / S-mid

**対象レポート**: [`independent_convergence_report.md`](independent_convergence_report.md) および [`corrected_role_report.md`](corrected_role_report.md)
**定義元スクリプト**: [`correct_role_analysis.py`](../source/correct_role_analysis.py)（旧: [`single_vs_compound_analysis.py`](../../../hypothesis/02_compound_hmm/source/single_vs_compound_analysis.py)）

---

## 1. これはプロジェクト固有の用語である

B-start / B-end / S-head / S-mid は**一般的な言語学・NLP用語ではない**。
この研究内で「交絡（confound）の有無を検証する」ために設計された4グループのラベルである。

---

## 2. プレフィックスの意味

| プレフィックス | 英語 | 意味 | 対象語種 |
|---|---|---|---|
| **B** | **B**oundary | 複合語の基と基の境界 | 複合語のみ |
| **S** | **S**ingle | 単独ベース語 | 単独ベース語のみ |

---

## 3. 各グループの定義

### B-start（Boundary-start）

> 複合語において、第2基以降の**先頭文字が置かれる位置**

複合語は複数の「基（ベース）」が連結した語である。
B-start は「2つ目以降の基が始まる地点」つまり**内部的な再スタート点**を指す。

**具体例：`adal = ad + al`（2基複合語）**

```
  a  d  a  l
  0  1  2  3   ← 文字位置（0始まり）

        ↑
      B-start = pos 2
      （基2「al」の先頭文字「a」の位置）
```

**具体例：`aeeodeey = a + eeod + eey`（3基複合語）**

```
  a  e  e  o  d  e  e  y
  0  1  2  3  4  5  6  7

     ↑              ↑
  B-start=1       B-start=5
  （基2の先頭）    （基3の先頭）
```

---

### B-end（Boundary-end）

> 複合語において、各基（最終基を除く）の**末尾文字が置かれる位置**

B-end は「基が終わる直前の地点」つまり**内部的な区切り点**を指す。

**具体例：`adal = ad + al`**

```
  a  d  a  l
  0  1  2  3

     ↑
   B-end = pos 1
   （基1「ad」の末尾文字「d」の位置）
```

**具体例：`aeeodeey = a + eeod + eey`**

```
  a  e  e  o  d  e  e  y
  0  1  2  3  4  5  6  7

  ↑           ↑
B-end=0     B-end=4
（基1の末尾）（基2の末尾）
```

---

### S-head（Single-head）

> 単独ベース語の**語頭文字（pos=0）の位置**

S-head は「語全体としての先頭」であり、複合語内部の基先頭（B-start）と対比するための**コントロールグループ**として機能する。

**具体例：`aiin`（単独ベース語）**

```
  a  i  i  n
  0  1  2  3

  ↑
S-head = pos 0
（語全体の先頭文字「a」の位置）
```

---

### S-mid（Single-middle）

> 単独ベース語の**語中央文字（pos = L // 2）の位置**

L は語長。S-mid は「どこにも特別な役割がない語の内部位置」として**ベースライン**に使われる。

**具体例：`aiin`（長さ 4）**

```
  a  i  i  n
  0  1  2  3

        ↑
     S-mid = pos 2（= 4 // 2）
     （語中央文字「i」の位置）
```

---

## 4. 4グループの対応関係

```
複合語:
  [基1]──────[基2]──────[基3]
         ↑ ↑      ↑ ↑
       B-end B-start  B-end B-start

単独ベース語:
  ↑            ↑
S-head        S-mid
（語頭）       （語中央）
```

---

## 5. なぜこの4グループを設計したか

### 検証したい問い

> HMM状態が「複合語の内部基先頭（B-start）」に集中するのは、
> **（A）複合構造を検出しているから**か、
> **（B）V8文法が基先頭文字を `{l,r,o,y,s,v}` などに制限しているだけ**か？

仮説（B）が正しいなら、単独語の語頭（S-head）も同じ文字集合から始まるため、
HMM は B-start と S-head を区別しないはずである。つまり B-start ≈ S-head になる。
これを **交絡（confounding）** と呼ぶ。

| 比較ペア | 何を測るか | 交絡なら | 複合構造効果があれば |
|---|---|---|---|
| **B-start vs S-head**（Fisher①） | 複合内部先頭 vs 語全体先頭 | B-start ≈ S-head | B-start ≠ S-head |
| B-start vs S-mid | 複合内部先頭 vs 語の中間 | B-start ≈ S-mid | B-start > S-mid |
| B-end vs S-mid | 複合内部末尾 vs 語の中間 | B-end ≈ S-mid | B-end ≠ S-mid |
| B-end vs B-start | 境界末尾 vs 境界先頭 | B-end ≈ B-start | 非対称 |

### S-mid はなぜ必要か

S-mid がない場合、「B-start が S-head と違う」だけでは
「複合語内部だから違う」のか「先頭以外の位置だから違う」のか区別できない。
S-mid を入れることで「語の内部位置は全般的にこの程度」という**参照ラインが得られる**。

---

## 6. hypothesis/04 での結果サマリ

> 根拠スクリプト: [`correct_role_analysis.py`](../source/correct_role_analysis.py)
> 数値レポート: [`corrected_role_report.md`](corrected_role_report.md)

Bigram HMM（k=7）の Focus State（S2）における各グループの出現率：

| グループ | S2 出現率 | 解釈 |
|---------|---------|------|
| **B-start** | **26.3%** | 複合語の内部基先頭に最も高く現れる |
| S-mid | 20.4% | 単独語の語中間でも相当数現れる（S2 は語内部に一般的） |
| B-end | 8.3% | 複合語の基末尾ではあまり現れない |
| **S-head** | **1.5%** | 単独語の語頭にはほぼ現れない |

Fisher① (B-start vs S-head): **p = 2.21e-284**（交絡仮説を強く否定）

→ **S2 は「複合語内部で基が新たに始まる位置」に特化した状態**であり、
文字種の制約（交絡）では説明できない。HMMが独立に複合構造を感知している証拠となる。

---

## 7. 一般的な概念との対応（参考）

強いて一般的な言語学・NLP の概念に対応させるなら：

| 本プロジェクト | 近い概念（厳密対応ではない） |
|---|---|
| B-start | 形態素境界直後位置 / 複合語構成素の語頭 |
| B-end | 形態素境界直前位置 / 複合語構成素の語末 |
| S-head | 語頭位置（word-initial position） |
| S-mid | 語内部位置（word-internal position） |

なお NLP では BIO タグ（Beginning / Inside / Outside）という系列ラベル体系があるが、
本プロジェクトの B-start / B-end はそれとは無関係である。

---

_本文書は `hypothesis/04_re-compound_hmm/results/terminology_guide.md`_
