# find_v8_splits_first() 分類バグ検証レポート

**作成日時**: 2026-03-08 21:36:40
**スクリプト**: `hypothesis/04_re-compound_hmm/source/verify_classification.py`

---

## 1. 問題の概要

`hypothesis/02_compound_hmm` および `hypothesis/03_trigram/source/trigram_role_analysis.py` で
使用されている `find_v8_splits_first()` 関数は、動的に語を「複合語 / 単独ベース語」に分類する。

この関数の問題点:
- `word` 自体が `is_base(word) == True`（単独ベース語）であっても、
  `word[:i]` と `word[i:]` の両方が valid base であれば**複合語と判定**してしまう。
- SLOTS_V8[0]（`{l,r,o,y,s,v}`）と SLOTS_V8[15]（`{k,t,p,f,l,r,o,y}`）の重複文字により、
  これらの文字で始まる単独ベース語の多くが誤って分割可能と判定される。

---

## 2. 分類数の比較

| 分類方法 | 複合語数 | 単独ベース語数 | 備考 |
|---------|---------|-------------|------|
| **正解** (`compound_words_sorted.txt` / `words_base_only.txt`) | **3,363** | **4,714** | `hypothesis/00_slot_model` が生成 |
| 動的分類 (`find_v8_splits_first()`) | **8,058** | 0 | hypothesis/02, 03 で使用 |
| **差分（過剰分類）** | **+4,695** | − | 約 **2.4倍** に膨張 |

動的分類による複合語数は正解の約 **2.4 倍**に膨張している。

---

## 3. 偽陽性分析（単独ベース語 → 誤って複合語と判定）

`words_base_only.txt`（正解・単独ベース語 4,714 語）に対して
`find_v8_splits_first()` を適用した結果:

| 指標 | 値 |
|-----|-----|
| 偽陽性件数 | **4,695 語** (99.6%) |
| SLOTS_V8[0] で始まる偽陽性 | **2,551 語** (54.3% of FP) |
| 偽陰性件数（正解複合語が未検出） | 0 語 |

**「処理が正しい場合 `words_base_only.txt` が抽出されるはずです」の検証結果**:
→ `words_base_only.txt` の 99.6% (4,695 語) が誤って複合語と判定される。
→ 正しい処理では偽陽性 = 0 件であるべき。

---

## 4. 偽陽性の先頭文字分布

SLOTS_V8[0] = `{l, r, o, y, s, v}` は基の先頭文字集合であり、
末尾文字集合 SLOTS_V8[15] との重複 `{l, r, o, y}` が偽陽性の主因となっている。

| 先頭文字 | 偽陽性件数 | 割合 | SLOTS_V8[0] |
|---------|----------|------|------------|
| `o` | 1,176 | 25.0% | ✓ |
| `q` | 597 | 12.7% |  |
| `s` | 561 | 11.9% | ✓ |
| `c` | 560 | 11.9% |  |
| `y` | 442 | 9.4% | ✓ |
| `d` | 338 | 7.2% |  |
| `l` | 282 | 6.0% | ✓ |
| `k` | 190 | 4.0% |  |
| `t` | 188 | 4.0% |  |
| `p` | 118 | 2.5% |  |
| `r` | 88 | 1.9% | ✓ |
| `e` | 54 | 1.2% |  |
| `f` | 49 | 1.0% |  |
| `a` | 35 | 0.7% |  |
| `x` | 5 | 0.1% |  |
| `i` | 4 | 0.1% |  |
| `g` | 3 | 0.1% |  |
| `h` | 3 | 0.1% |  |
| `v` | 2 | 0.0% | ✓ |

---

## 5. 偽陽性の代表例（最初の 30 件）

| 単独ベース語 | 誤った分割 | SLOTS_V8[0] |
|------------|-----------|------------|
| `ady` | `a + dy` | False |
| `aii` | `a + ii` | False |
| `aiidy` | `a + iidy` | False |
| `aiiin` | `a + iiin` | False |
| `aiil` | `a + iil` | False |
| `aiily` | `a + iily` | False |
| `aiim` | `a + iim` | False |
| `aiin` | `a + iin` | False |
| `aiiny` | `a + iiny` | False |
| `aiip` | `a + iip` | False |
| `aiir` | `a + iir` | False |
| `aiiry` | `a + iiry` | False |
| `aiis` | `a + iis` | False |
| `ail` | `a + il` | False |
| `aim` | `a + im` | False |
| `ain` | `a + in` | False |
| `ainy` | `a + iny` | False |
| `air` | `a + ir` | False |
| `airy` | `a + iry` | False |
| `ais` | `a + is` | False |
| `aiy` | `a + iy` | False |
| `al` | `a + l` | False |
| `alf` | `a + lf` | False |
| `alo` | `a + lo` | False |
| `als` | `a + ls` | False |
| `alsy` | `a + lsy` | False |
| `aly` | `a + ly` | False |
| `am` | `a + m` | False |
| `amy` | `a + my` | False |
| `an` | `a + n` | False |

---

## 6. 結論

`find_v8_splits_first()` は単独ベース語 `words_base_only.txt` の
**99.6% (4,695語)** を誤って複合語と判定する。

この偽陽性により、hypothesis/02 および 03 の分析では:
- 分析対象の複合語数が正解 3,363 語から **8,058 語** に膨張
- B-start / B-end のメトリクスが偽陽性含有の集合を対象に計算されている
- 偽陽性の境界位置は実際には単独ベース語の語中に設定されたノイズである

**修正方針**: `hypothesis/04_re-compound_hmm` では、動的分類を廃止し、
`compound_words.txt`（分割情報付き）を正解データとして直接使用する。

---

_本レポートは `verify_classification.py` により自動生成。_
