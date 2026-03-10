# E-2（修正版）: 文法バリアント感度分析レポート — 貪欲分割アルゴリズム

生成日時: 2026-03-08 21:48:35

## 修正内容

**hypothesis/08 e2_grammar_variants.py** では `find_splits_variant` が
「最短先頭基優先（first-fit）」だったため、compound_words.txt と 99.6% の分割点が異なっていた。

**本スクリプト（e2_grammar_variants_greedy.py）** では `parse_v4b.py` の
`parse_base_greedy` と等価な貪欲アルゴリズムを使用する。
これにより V0_baseline が compound_words.txt と同一の分割を再現する。

## V0_baseline 検証（compound_words.txt との一致確認）

| 項目 | 件数 |
|------|------|
| 元 compound_words.txt (V4/V8 文法) | 3,363 |
| V0_baseline (貪欲アルゴリズム) | 3,363 |
| 元ファイルのみ | 0 |
| V0_baseline のみ | 0 |
| 同一分割 | **3,363** |
| 分割点が異なる | 0 |

---

## 文法バリアント定義

| バリアント | 変更内容 |
|-----------|---------|
| V0_baseline | 基準 (SLOTS_V8 オリジナル) ← compound_words.txt と同一分割 |
| V1_expand_vow | 拡張: slot 2 (V_a) に 'i' 追加 → [o, y, i] |
| V2_revert_z | 制限: slot 1 (C2) から 'z' 除去 → v7 相当 |
| V3_expand_va2 | 拡張: slot 10 (V_a2) に 'e' 追加 → [o, a, y, e] |

---

## 複合語数の比較

| バリアント | 複合語数 | 差分 (vs V0) |
|-----------|---------|------------|
| V0_baseline | 3,363 |  |
| V1_expand_vow | 3,357 | -6 |
| V2_revert_z | 3,361 | -2 |
| V3_expand_va2 | 3,229 | -134 |

---

## 役割分析結果サマリー

| バリアント | k | Focus State | B-start 率 | S-head 率 | Fisher① p | 判定 |
|-----------|---|------------|-----------|---------|---------|------|
| V0_baseline | 7 | S2 | 34.6% | 10.2% | 2.30e-163 | **構造効果あり** |
| V0_baseline | 8 | S5 | 31.3% | 0.0% | 0.00e+00 | **構造効果あり** |
| V1_expand_vow | 7 | S2 | 34.6% | 10.2% | 5.26e-163 | **構造効果あり** |
| V1_expand_vow | 8 | S5 | 31.4% | 0.0% | 0.00e+00 | **構造効果あり** |
| V2_revert_z | 7 | S2 | 34.5% | 10.2% | 7.11e-163 | **構造効果あり** |
| V2_revert_z | 8 | S5 | 31.3% | 0.0% | 0.00e+00 | **構造効果あり** |
| V3_expand_va2 | 7 | S2 | 37.3% | 10.2% | 3.92e-189 | **構造効果あり** |
| V3_expand_va2 | 8 | S5 | 30.7% | 0.0% | 0.00e+00 | **構造効果あり** |

---

## 解釈

### V0_baseline と hypothesis/05 の比較

V0_baseline（貪欲分割）が compound_words.txt と同一分割であれば、
hypothesis/05 と同等の分析が再現できる。
Focus State・Fisher p 値が hypothesis/05 の結果と一致するかを確認する。

### バリアント感度の判断基準

- **全バリアントで同じ Focus State + 有意な Fisher p**: 文法小幅変更に対して頑健
- **Focus State が変わるが Fisher p は有意**: 状態割り当て変動、シグナルは保持
- **特定バリアントで Fisher p が非有意**: 該当スロット変更がシグナルに影響

---

## 参照

- 分割アルゴリズム原典: `hypothesis/00_slot_model/source/parse_v4b.py`
- 旧実装 (first-fit): `hypothesis/08_v8_sensitivity/source/e2_grammar_variants.py`
- 本スクリプト: `hypothesis/09_v8_variants_greedy/source/e2_grammar_variants_greedy.py`

_本レポートは e2_grammar_variants_greedy.py により自動生成。_