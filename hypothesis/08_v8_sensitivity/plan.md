# hypothesis/08_v8_sensitivity — 実施計画

作成日時: 2026-03-08
根拠: hypothesis/05_re-compound_hmm_trigram/results/next_analysis_proposals.md の提案 E

---

## 概要

提案 E（V8 文法感度分析・置換検定）の実装計画。
二つのサブ分析から構成される。

| サブ分析 | 目的 | スクリプト |
|---------|------|---------|
| E-1 | 境界ラベルのシャッフルによる置換検定 | `source/e1_permutation_test.py` |
| E-2 | V8 文法バリアント感度分析 | `source/e2_grammar_variants.py` |

---

## E-1: 置換検定（境界ラベルのシャッフル）

### 仮説

> 帰無仮説 H₀: 境界ラベルが語内でランダムに配置されても、同程度の B-start 特化シグナルが得られる

### 実装方針

**重要な効率化**: Viterbi デコードは一度だけ実行し、境界ラベルのみをシャッフルする。
これにより 1000 回の置換が数秒〜数分で完了する（再デコード不要）。

```
1. 全モデル（trigram k=7,8; bigram k=7,8）で Viterbi デコードを実行（一度）
2. 実際の compound_words.txt 境界ラベルで Fisher 統計量を計算
3. N=1000 回繰り返し:
   - 各複合語の境界位置をランダムに選択（基数は固定）
   - シャッフル後の境界ラベルで同じ Fisher 統計量を計算
4. 帰無分布を構築し、置換 p 値 = (null ≥ observed) / N を計算
5. 可視化・レポート生成
```

**統計量**: `-log10(Fisher p)` for B-start vs S-head（比較①）
- Focus State は各置換でも data-driven に選択（hypothesis/05 の手順と同一）
- B-start 出現率が最大の状態（Phantom 除外）を Focus State とする

**境界シャッフル方法**:
```python
# 元の基数（n_bases）を保持し、境界位置のみランダム化
split_positions = sorted(rng.choice(range(1, len(word)), size=n_bases-1, replace=False))
```
- 語長・基数を保持（分布的に公平な帰無仮説）
- シャッフル後の「基」は V8 的に有効でなくてもよい（純粋なラベル置換）

### 出力

- `results/e1_permutation/null_dist_{model_type}_k{k}.png` × 4（帰無分布ヒストグラム）
- `results/e1_permutation/permutation_report.md`（置換 p 値サマリー）

---

## E-2: 文法バリアント感度分析

### 仮説

> 分析結果は SLOTS_V8 の具体的なスロット定義に強く依存するか？
> 小さな文法変更で Focus State シグナルが消えるなら脆弱、安定するなら頑健。

### 実装方針

**注記**: 元の `compound_words.txt` は "V4 文法" (V8 より古い定義) で生成されており、
"ck" などの要素を単独基として含む。E-2 では SLOTS_V8 をベースラインとして
新たに複合語リストを生成し、バリアントと比較する。

**文法バリアント定義**:

| バリアント | 変更内容 | 根拠 |
|-----------|---------|------|
| V0_baseline | SLOTS_V8 オリジナル | 基準 |
| V1_expand_vow | slot 2 (V_a: o,y) に 'i' を追加 → [o,y,i] | i は母音的文字、V_a 位置への拡張 |
| V2_revert_z | slot 1 (C2) から 'z' を除去 (v8→v7 相当) | z は v8 で追加、感度確認 |
| V3_expand_va2 | slot 10 (V_a2: o,a,y) に 'e' を追加 → [o,a,y,e] | e 系の V_a2 位置への拡張 |

**複合語生成アルゴリズム**:
```
1. DB から全語彙を取得
2. 各語に対し、バリアント文法で貪欲分割を試みる（is_base_variant + find_splits_variant）
3. 2〜4 基に分割できる語を複合語とする
4. 単独基語（words_base_only.txt）は変更しない（S-head/S-mid 参照を固定）
```

**分析**: trigram k=7, k=8 のみ（hypothesis/05 の主分析と同一モデル）

### 出力

- `results/e2_variants/compounds_{variant}.txt` × 4（各バリアントの複合語リスト）
- `results/e2_variants/role_analysis_{variant}_k{k}.png` × 8
- `results/e2_variants/variant_report.md`（バリアント間比較表）

---

## ディレクトリ構成

```
hypothesis/08_v8_sensitivity/
├── plan.md                     ← 本ファイル
├── source/
│   ├── e1_permutation_test.py  ← E-1 実装
│   └── e2_grammar_variants.py  ← E-2 実装
└── results/
    ├── e1_permutation/
    │   ├── null_dist_trigram_k7.png
    │   ├── null_dist_trigram_k8.png
    │   ├── null_dist_bigram_k7.png
    │   ├── null_dist_bigram_k8.png
    │   └── permutation_report.md
    └── e2_variants/
        ├── compounds_V0_baseline.txt
        ├── compounds_V1_expand_vow.txt
        ├── compounds_V2_revert_z.txt
        ├── compounds_V3_expand_va2.txt
        ├── role_analysis_*.png
        └── variant_report.md
```

---

## 実行方法

```bash
cd /home/practi/work_voy

# E-1: 置換検定（Viterbi デコード 1 回 + 1000 回ラベルシャッフル、数分程度）
PYTHONPATH=/home/practi/work_voy/.venv/lib/python3.10/site-packages \
  python3.10 hypothesis/08_v8_sensitivity/source/e1_permutation_test.py

# E-2: 文法バリアント（複合語リスト再生成 + 4 バリアント × 2 モデル分析、数分〜10 分程度）
PYTHONPATH=/home/practi/work_voy/.venv/lib/python3.10/site-packages \
  python3.10 hypothesis/08_v8_sensitivity/source/e2_grammar_variants.py
```

---

## 解釈指針

### E-1 の判定基準

| 置換 p 値 | 解釈 |
|---------|------|
| < 0.05 | **帰無仮説棄却**: 境界シグナルは V8 文法が定義する実際の境界位置に依存 |
| ≥ 0.05 | **棄却失敗**: ランダムな境界でも同程度のシグナルが生まれる可能性あり |

### E-2 の判定基準

| 状況 | 解釈 |
|------|------|
| 全バリアントで同じ Focus State が同定される | **頑健**: 分析結果は文法定義に鈍感 |
| Focus State が変わっても Fisher p は有意 | **部分的頑健**: 状態割り当てが変化しても境界シグナルは存在 |
| 特定バリアントで Fisher p が非有意になる | **感度あり**: 該当スロット変更が境界シグナルに影響 |

---

_本計画は next_analysis_proposals.md の提案 E を根拠として作成。_
