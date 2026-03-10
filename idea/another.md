検証立案：非複合語（単独ベース語）との比較
前提：直接比較ができない理由
V8単独ベース語が 0語 のため、レポート6.2の記述通りに「単独ベース語の中央位置」を比較対象にすることは不可能です。この制約を踏まえ、同じ目的（交絡要因の排除）をより鋭い設計で達成する代替案を提案します。

交絡要因の構造の整理
現在発見されている交絡仮説を明示的にします：
V8 is_base() は parse_greedy() を使う
  → 新ベースの先頭文字は {o, q, l, r, y} になりやすい（SLOTS_V8の先頭選択肢）
  
HMM S4(k=7) / S0(k=8) はこれらの文字クラスタを独立に学習
  
∴ 「V8境界 = ベース先頭 = {o,q,l,r,y}」「S4 = {o,q,l}クラスタ」が一致
  → S4の境界集中は文字レベルの一致であり、複合語構造の証拠ではないかもしれない

検証すべき問い：S4集中は「複合境界という構造的位置」に由来するか、それとも単に「その位置にベース先頭文字が来るから」か？

提案する検証設計：5位置グループ比較
単独ベース語の代わりに、複合語内の文字位置を構造的に5種類に分類して比較します。

| グループ | 定義 | 位置例（2基語 base1|base2） | 特性 |
|---|---|---|---|
| A: 語頭 | pos=0（base1の先頭文字） | c in ch·ol | ベース先頭 かつ 複合境界ではない |
| B: 複合境界・先頭 | 2番目以降のベースの先頭文字 | o in ch·ol | ベース先頭 かつ 複合境界（検証対象） |
| C: 複合境界・終端 | 最後以外のベースの末尾文字 | h in ch·ol | ベース終端 かつ 複合境界 |
| D: 語末 | pos=L-1（baseNの末尾文字） | l in ch·ol | ベース終端 かつ 複合境界ではない |
| E: ベース内部 | A/B/C/D 以外の全位置 | 中間文字 | 複合構造と無関係な純粋内部 |


各グループのS4率から交絡要因を判定する論理

【交絡要因説（文字制約による自明な帰結）が正しい場合】
  → ベース先頭（A, B）には常に {o,q,l} が来るため：
     rate(B) ≈ rate(A) >> rate(C) ≈ rate(D) ≈ rate(E)

【複合語構造説が正しい場合】
  → 複合境界という構造位置でS4が特異的に高い：
     rate(B) >> rate(A)（語頭より複合境界で有意に高い）
A（語頭）はBOS隣接のため、HMMの開始確率バイアスが乗る可能性があります。ただしこれは交絡要因説に有利な方向（BOSの直後も「ベース先頭っぽい文字」が来やすいため）なので、もし B >> A が確認できれば、より強い証拠になります。


付加検証：ランダム境界との比較（6.1との統合）
グループBに相当する位置を、V8境界ではなくランダムに選んだ境界で生成：
# 同じ語長の単語で、V8境界位置と同じ数のランダム位置を生成
for word in compound_words:
    rand_pos = rng.integers(1, len(word))  # V8境界とは無関係のランダム位置

実装方針
既存の compound_boundary_analysis.py をほぼそのまま流用して、analyze_boundaries() の位置分類ロジックを拡張します。
# 既存: boundary_both, non_boundary の2分類
# 変更後: A/B/C/D/E の5分類

def analyze_positions(info, compound_splits, char2idx, k):
    ...
    for word, splits in compound_splits.items():
        states = viterbi_path(word)  # 既存の関数を再利用
        bd_end, bd_start = get_boundary_positions(splits)
        L = len(states)
        
        for pos, state in enumerate(states):
            if pos == 0:                           # Group A: 語頭
                group_A.append(state)
            elif pos == L - 1:                     # Group D: 語末
                group_D.append(state)
            elif pos in bd_start:                  # Group B: 複合境界先頭
                group_B.append(state)
            elif pos in bd_end:                    # Group C: 複合境界終端
                group_C.append(state)
            else:                                  # Group E: ベース内部
                group_E.append(state)

