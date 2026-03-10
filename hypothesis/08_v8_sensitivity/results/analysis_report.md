# hypothesis/08_v8_sensitivity — 総合分析レポート

作成日時: 2026-03-08
スクリプト: [e1_permutation_test.py](../../source/e1_permutation_test.py) / [e2_grammar_variants.py](../../source/e2_grammar_variants.py)
根拠データ: [E-1 レポート](e1_permutation/permutation_report.md) / [E-2 レポート](e2_variants/variant_report.md)

---

## 1. 目的と問い

本分析は次の問いに答えることを目的とする。

> **「hypothesis/05 で確認された複合語境界シグナルは V8 文法定義に依存するか？**
> それとも、文法によらない頑健な統計的実在として存在するか？」

この問いを 2 つのサブ問いに分解して検証した。

| サブ問い | 手法 | 対象データ |
|---------|------|---------|
| E-1: 境界「位置」の信憑性 | 境界ラベル置換検定（N=1000） | V4 compound_words.txt（3,363 語）|
| E-2: 境界「定義」の感度 | SLOTS_V8 文法バリアント分析 | SLOTS_V8 ベースライン + 3 バリアント |

---

## 2. E-1: 置換検定 — 境界位置は本物か

### 2.1 手法の設計

compound_words.txt の 3,363 語の「語」は固定したまま、**各語内の境界位置のみをランダム化**（基数固定）した帰無分布を 1,000 回構築した。
統計量には χ²（2×2 分割表 B-start vs S-head）を採用した。
→ 初回は −log₁₀(Fisher p) を使用したが trigram k=8 で float フロア(p=0)によるキャップ飽和が発生し、chi² に修正した。

### 2.2 結果

| モデル | k | Focus State | 観測 χ² | 帰無中央値 | 観測/帰無比 | 置換 p 値 | 判定 |
|--------|---|------------|--------|---------|----------|---------|------|
| trigram | 7 | S2 | **736.5** | 472.0 | **1.56×** | **< 0.001** | H₀ 棄却 |
| trigram | 8 | S5 | **1701.7** | 1384.6 | **1.23×** | **< 0.001** | H₀ 棄却 |
| bigram  | 7 | S2 | 1169.7 | 1116.5 | 1.05× | 0.077 | 棄却失敗 |
| bigram  | 8 | S1 | **1227.5** | 1101.0 | **1.12×** | **< 0.001** | H₀ 棄却 |

生成プログラム: [e1_permutation_test.py](../../source/e1_permutation_test.py)
帰無分布グラフ: [trigram k=7](e1_permutation/null_dist_trigram_k7.png) / [trigram k=8](e1_permutation/null_dist_trigram_k8.png) / [bigram k=7](e1_permutation/null_dist_bigram_k7.png) / [bigram k=8](e1_permutation/null_dist_bigram_k8.png)

### 2.3 考察

**3/4 モデルで H₀ 棄却（p < 0.001）**
V4 文法が定義した実際の境界位置は、同じ複合語内のランダムな位置よりも有意に大きな B-start 特化シグナルを生み出している。これは、V4 文法の複合語分割アルゴリズムが、HMM の状態遷移パターンと整合した言語的に有意な境界を捉えていることを示す。

**Bigram k=7 の例外（p = 0.077）**
帰無中央値（1116.5）と観測値（1169.7）の差はわずか 4.8%（他モデルは 12〜56%）。
これは「複合語という語のプール自体」がランダムな分割でも B-start 特化を生む構造を持つことを意味する。
Bigram k=7 の Focus State S2 は a/y/o 系母音を放射するため、複合語内の任意の母音位置が境界としてラベリングされれば同程度の偏りが生じる。
言い換えると、bigram k=7 では境界シグナルの一部は「複合語語集合の選択」に依存しており、「境界の正確な位置指定」への依存は弱い。

**Trigram の優位性**
trigram k=7 の観測/帰無比 1.56× は他の 3 モデル（1.05〜1.23×）を大きく上回る。
s_{t-2} の追加文脈が境界位置の「正確さ」をより鋭く識別していることを示す。

---

## 3. E-2: 文法バリアント分析 — 文法定義変更への感度

### 3.1 コア発見：V4 と SLOTS_V8 で分割点は 99.6% 異なる

E-2 の根本的な発見として、SLOTS_V8 ベースライン (V0_baseline) と元の compound_words.txt (V4) を比較すると、**語の集合は完全に一致（3,363 語 ← → 3,363 語、重複なし、排他なし）するが、分割点は 3,351 語（99.6%）で異なる**ことが判明した。

代表例：

| 語 | V4 分割 | SLOTS_V8 分割 |
|---|---------|--------------|
| ackaldy | a + **ck** + al + dy | a + **c** + ka + ldy |
| adal    | **ad** + al           | **a** + dal          |
| adam    | **ad** + am           | **a** + dam          |
| adchey  | **ad** + chey         | **a** + dchey        |

V4 文法は "ck", "ad" 等の 2 文字クラスタを単独基として認識するため、SLOTS_V8 とは体系的に異なる境界位置を生む。SLOTS_V8 は "a" 単独（slot 10: V_a2）が基として優先されるため、多くの語で先頭 1 文字 "a" を第 1 基として分離する。

### 3.2 SLOTS_V8 内部バリアントの結果

| バリアント | 変更内容 | 複合語数 | k=7 Focus / B-start / S-head | k=8 Focus / B-start / S-head |
|-----------|---------|--------|------------------------------|------------------------------|
| V0_baseline | 基準 | 3,363 | S4 / 38.7% / 53.1% (**逆転**) | S2 / 33.9% / 26.8% (構造効果) |
| V1_expand_vow | slot2 に 'i' 追加 | 3,357 (−6) | S4 / 38.3% / 53.1% (**逆転**) | S2 / 33.5% / 26.8% (構造効果) |
| V2_revert_z | slot1 から 'z' 除去 | 3,361 (−2) | S4 / 38.7% / 53.1% (**逆転**) | S2 / 33.9% / 26.8% (構造効果) |
| V3_expand_va2 | slot10 に 'e' 追加 | 3,229 (−134) | S4 / 38.8% / 53.1% (**逆転**) | S2 / 34.2% / 26.8% (構造効果) |

生成プログラム: [e2_grammar_variants.py](../../source/e2_grammar_variants.py)
複合語リスト: [V0](e2_variants/compounds_V0_baseline.txt) / [V1](e2_variants/compounds_V1_expand_vow.txt) / [V2](e2_variants/compounds_V2_revert_z.txt) / [V3](e2_variants/compounds_V3_expand_va2.txt)
役割分析グラフ: [V0 k=7](e2_variants/role_analysis_V0_baseline_k7.png) / [V0 k=8](e2_variants/role_analysis_V0_baseline_k8.png) 他

### 3.3 考察：2 層構造の頑健性

**【第 1 層: SLOTS_V8 内部の頑健性】**
V0〜V3 のすべてのバリアントで Focus State が完全に一致し、Fisher p 値も同水準（k=7: p ≈ 1e-53〜1e-57、k=8: p ≈ 1e-15〜1e-18）を維持している。複合語数が −134（V3、3.8%減少）しても結果は変わらない。この層においては、境界シグナルは SLOTS_V8 文法定義の小幅変更に**完全に頑健**である。

**【第 2 層: SLOTS_V8 vs V4 の感度】**
一方、SLOTS_V8 ベースライン（V0）と hypothesis/05 の V4 ベースを比較すると、k=7 で Focus State が S4（逆転）/ S2（B-start >> S-head）と大きく異なる。この差は語集合の違いではなく（同一の 3,363 語）、**分割点の体系的差異（99.6% 異なる）**に起因する。

**k=7 の「逆転」メカニズム**
SLOTS_V8 の複合語境界では、境界先頭（B-start）に出現する状態の最大率（S4: 38.7%）が単独語先頭（S-head: 53.1%）を下回る。つまり SLOTS_V8 境界は HMM が「語先頭」として認識するパターンとは整合しない。これは SLOTS_V8 の貪欲マッチが "a" 単独を第 1 基として頻繁に分離するためで、その次の境界（第 2 基先頭）は子音クラスタから始まり、S2（c/a 放射）の特化と合致しない。V4 境界では "ad", "ck" 等のクラスタ基の後に来る位置が境界先頭となり、そこで S2 が特化する。

---

## 4. 総合判断：境界シグナルは V8 文法定義に依存するか

### 4.1 判断

> **境界シグナルの「実在」は文法定義によらず確認される（3/4 モデルで置換検定有意）。
> しかし、シグナルの「具体的性質」（どの状態が特化するか、方向性）は文法の「分割アルゴリズム」に感度を持つ。SLOTS_V8 内部の小幅変更には頑健だが、V4 ← → SLOTS_V8 の体系的な分割点差異（99.6% の語で異なる）はシグナルの性質を変える。**

### 4.2 根拠の整理

```
境界シグナルの問い                結論              根拠
────────────────────────────────────────────────────────────────────────
ランダム境界でも同じシグナルが出るか？  NO (3/4 モデル)   E-1 置換検定
                                   例外: bigram k=7   p_perm=0.077

SLOTS_V8 の ±1 文字変更で変わるか？   変わらない         E-2 V0〜V3 完全一致

V4 → SLOTS_V8 の境界置換で変わるか？  変わる             V0 vs h05: Focus S2↔S4
                                                          方向も反転 (k=7)

その差の原因は語集合か、分割点か？     分割点（99.6%差）   compound diff 分析
```

### 4.3 モデル別の解釈

| モデル | 境界シグナルの実在 | シグナルの文法依存性 |
|--------|--------------|---------------|
| Trigram k=7 | **強い**（置換比 1.56×、p < 0.001）| V4 境界の S2(c 系)への特化は V4 分割に特有 |
| Trigram k=8 | **強い**（置換比 1.23×、p < 0.001）| S5(y/v 系)特化は V4 境界と整合 |
| Bigram k=7  | **弱い**（置換比 1.05×、p = 0.077）| 語集合選択への依存が高く、境界位置の精度依存は低い |
| Bigram k=8  | **中程度**（置換比 1.12×、p < 0.001）| 境界位置に実在シグナル |

---

## 5. 追加考察

### 5.1 V4 文法の言語的妥当性

E-2 で判明した「V4 と SLOTS_V8 は同じ語集合を複合語と判定するが、分割点は 99.6% 異なる」という事実は、hypothesis/05 の結果解釈に重要な意味を持つ。

hypothesis/05 で S2 が compound 境界（B-start）専用状態として機能したのは、V4 文法の分割アルゴリズムが HMM の状態遷移パターンと整合した境界を定義していたからである可能性がある。裏返せば、**HMM が内部的に学習している「基の区切り」の概念は、SLOTS_V8 よりも V4 文法の定義に近い**という仮説が導かれる。

### 5.2 置換検定の限界：語集合 vs 境界位置の交絡

E-1 の置換検定は「境界位置のランダム化」のみを検定しており、「複合語語集合の妥当性」は検定していない。もし「複合語と分類された語群自体」が HMM にとって境界シグナルを持ちやすい語集合であれば、bigram k=7 のような「語集合選択依存型」のシグナルが生まれる。

今後、語集合そのものをシャッフル（単独語集合から同数の語をランダムに「複合語」とみなす）する検定を追加すれば、語集合選択効果と境界位置効果を分離できる。

### 5.3 E-2 における SLOTS_V8 k=8 の一貫した「構造効果」

k=8 ではすべてのバリアントで B-start > S-head（S2: 33〜34% vs 26.8%）が維持された。SLOTS_V8 の貪欲分割でも k=8 の HMM は compound 内部境界で特化状態 S2 を呼び出す。k=7 と k=8 で結果が逆転している理由として、k が増えると状態が細分化され、SLOTS_V8 境界（子音クラスタ先頭）に特化した状態が k=8 の S2 として現れる可能性がある。

---

## 6. 結論

1. **境界シグナルの実在性（E-1 から）**
   V4 文法の複合語境界位置は、同語内のランダム位置より有意に大きな B-start 特化シグナルを持つ（4 モデル中 3 モデルで p < 0.001）。ランダム境界で説明できない「文法定義された境界の実在性」が統計的に確認された。

2. **SLOTS_V8 内部での頑健性（E-2 から）**
   slot 定義の ±1 文字変更（V1〜V3）では Focus State・Fisher p 値が全く変化しない。SLOTS_V8 の小幅修正に対して分析結果は完全に頑健である。

3. **文法分割アルゴリズムへの感度（E-2 compound diff から）**
   SLOTS_V8 と V4 は同じ語集合を複合語と分類するが、境界位置の 99.6%（3,351/3,363 語）が異なる。この体系的な分割点の差が Focus State の同定（S2 vs S4）と方向性（B-start >> S-head vs 逆転）を変える。

4. **主要判断**
   hypothesis/05 で確認された境界シグナルの**実在**は V8/V4 文法によらない（E-1）。しかし、**「S2 が B-start 専用状態として機能する」という具体的な結果**は V4 文法の分割アルゴリズムが定義した境界位置に依存しており、SLOTS_V8 境界では再現されない。この意味で、分析の定量的な結論は「どの文法で複合語を分割するか」に感度を持つ。

---

## 7. 参照ファイル一覧

| ファイル | 種別 | リンク |
|--------|------|------|
| 置換検定スクリプト | コード | [e1_permutation_test.py](../../source/e1_permutation_test.py) |
| 文法バリアントスクリプト | コード | [e2_grammar_variants.py](../../source/e2_grammar_variants.py) |
| E-1 置換検定レポート | 結果 | [permutation_report.md](e1_permutation/permutation_report.md) |
| E-2 バリアント分析レポート | 結果 | [variant_report.md](e2_variants/variant_report.md) |
| 帰無分布 trigram k=7 | グラフ | [null_dist_trigram_k7.png](e1_permutation/null_dist_trigram_k7.png) |
| 帰無分布 trigram k=8 | グラフ | [null_dist_trigram_k8.png](e1_permutation/null_dist_trigram_k8.png) |
| 帰無分布 bigram k=7 | グラフ | [null_dist_bigram_k7.png](e1_permutation/null_dist_bigram_k7.png) |
| 帰無分布 bigram k=8 | グラフ | [null_dist_bigram_k8.png](e1_permutation/null_dist_bigram_k8.png) |
| Focus State 安定性 trigram k=7 | グラフ | [focus_stability_trigram_k7.png](e1_permutation/focus_stability_trigram_k7.png) |
| Focus State 安定性 trigram k=8 | グラフ | [focus_stability_trigram_k8.png](e1_permutation/focus_stability_trigram_k8.png) |
| V0_baseline 複合語リスト | データ | [compounds_V0_baseline.txt](e2_variants/compounds_V0_baseline.txt) |
| V1 複合語リスト | データ | [compounds_V1_expand_vow.txt](e2_variants/compounds_V1_expand_vow.txt) |
| V2 複合語リスト | データ | [compounds_V2_revert_z.txt](e2_variants/compounds_V2_revert_z.txt) |
| V3 複合語リスト | データ | [compounds_V3_expand_va2.txt](e2_variants/compounds_V3_expand_va2.txt) |
| 元 compound_words.txt (V4) | 参照データ | [compound_words.txt](../../../00_slot_model/data/compound_words.txt) |
| 計画書 | 文書 | [plan.md](../../plan.md) |

---

_本レポートは hypothesis/08_v8_sensitivity の E-1・E-2 実行結果に基づき手動作成。_
