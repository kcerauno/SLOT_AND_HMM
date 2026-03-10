"""
verify_classification.py
=========================
find_v8_splits_first() の分類バグを定量化する検証スクリプト

問題:
  hypothesis/02_compound_hmm および hypothesis/03_trigram で使用している
  find_v8_splits_first() は、word 自体が is_base(word)==True であっても、
  部分文字列を2つの valid base に分割できれば複合語と判定してしまう。

  具体的には SLOTS_V8[0] ({l,r,o,y,s,v}) と SLOTS_V8[15] ({k,t,p,f,l,r,o,y})
  の重複文字により、単独ベース語の多くが偽陽性として複合語に分類される。

検証内容:
  1. words_base_only.txt (正解・単独ベース語 4,716語) の各語に
     find_v8_splits_first() を適用 → 偽陽性件数を計測
  2. 動的分類の全体複合語数 vs 正解複合語数を比較
  3. 偽陽性が先頭文字 {l,r,o,y,s,v} に集中しているかを確認

正しい処理であれば:
  words_base_only.txt の語は複合語として分類されず、
  単独ベース語として抽出されるはずである。

実行:
  cd /home/practi/work_voy
  python hypothesis/04_re-compound_hmm/source/verify_classification.py
"""

from pathlib import Path
from datetime import datetime
from collections import Counter
import re


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")


# ── パス設定 ──────────────────────────────────────────────────────────────
SINGLE_WORDS_PATH   = Path("hypothesis/00_slot_model/data/words_base_only.txt")
COMPOUND_SORT_PATH  = Path("hypothesis/00_slot_model/data/compound_words_sorted.txt")
COMPOUND_SPLIT_PATH = Path("hypothesis/00_slot_model/data/compound_words.txt")
UNIQUE_WORDS_PATH   = Path("hypothesis/00_slot_model/data/unique_word.txt")
OUT_DIR             = Path("hypothesis/04_re-compound_hmm/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 1. V8 文法定義（compound_boundary_analysis.py から移植）
# ════════════════════════════════════════════════════════════════════════
SLOTS_V8 = [
    ["l", "r", "o", "y", "s", "v"],
    ["q", "s", "d", "x", "l", "r", "h", "z"],
    ["o", "y"], ["d", "r"], ["t", "k", "p", "f"],
    ["ch", "sh"], ["cth", "ckh", "cph", "cfh"],
    ["eee", "ee", "e", "g"],
    ["k", "t", "p", "f", "ch", "sh", "l", "r", "o", "y"],
    ["s", "d", "c"], ["o", "a", "y"], ["iii", "ii", "i"],
    ["d", "l", "r", "m", "n"], ["s"], ["y"],
    ["k", "t", "p", "f", "l", "r", "o", "y"],
]


def parse_greedy(word: str) -> tuple:
    pos, matched = 0, []
    for idx, options in enumerate(SLOTS_V8):
        if pos >= len(word):
            break
        for opt in options:
            if word.startswith(opt, pos):
                matched.append((idx, opt))
                pos += len(opt)
                break
    return matched, word[pos:]


def is_base(word: str) -> bool:
    m, r = parse_greedy(word)
    return r == "" and bool(m)


def find_v8_splits_first(word: str):
    """
    最初の有効分割を返す (2基以上のみ)。
    word 自体が is_base(word)==True であっても、部分分割が見つかれば返す。
    ← これがバグの原因
    """
    for i in range(1, len(word)):
        p1 = word[:i]
        if not is_base(p1):
            continue
        rest = word[i:]
        if is_base(rest):
            return (p1, rest)
        for j in range(1, len(rest)):
            p2 = rest[:j]
            if not is_base(p2):
                continue
            rest2 = rest[j:]
            if is_base(rest2):
                return (p1, p2, rest2)
            for kk in range(1, len(rest2)):
                if is_base(rest2[:kk]) and is_base(rest2[kk:]):
                    return (p1, p2, rest2[:kk], rest2[kk:])
    return None


# ════════════════════════════════════════════════════════════════════════
# 2. データロード
# ════════════════════════════════════════════════════════════════════════
def load_word_list(path: Path) -> list:
    """テキストファイルから語リストを読み込む（空行・コメント除外）。"""
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("<"):
            words.append(line)
    return words


def load_compound_splits(path: Path) -> dict:
    """
    compound_words.txt の形式をパース。
    形式: "[N基] word  ->  base1 + base2 + ..."
    戻り値: dict[word -> tuple(base1, base2, ...)]
    """
    splits = {}
    pattern = re.compile(r"\[(\d+)基\]\s+(\S+)\s+->\s+(.+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = pattern.match(line)
        if not m:
            continue
        word = m.group(2)
        bases = tuple(b.strip() for b in m.group(3).split("+"))
        splits[word] = bases
    return splits


# ════════════════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("=== find_v8_splits_first() 分類バグ検証 開始 ===")

    # データロード
    log(f"単独ベース語（正解）ロード: {SINGLE_WORDS_PATH}")
    single_words_gt = load_word_list(SINGLE_WORDS_PATH)
    log(f"  正解・単独ベース語数: {len(single_words_gt):,} 語")

    log(f"複合語（正解・ソート済み）ロード: {COMPOUND_SORT_PATH}")
    compound_words_gt = set(load_word_list(COMPOUND_SORT_PATH))
    log(f"  正解・複合語数: {len(compound_words_gt):,} 語")

    log(f"複合語（正解・分割情報付き）ロード: {COMPOUND_SPLIT_PATH}")
    compound_splits_gt = load_compound_splits(COMPOUND_SPLIT_PATH)
    log(f"  分割情報付き複合語数: {len(compound_splits_gt):,} 語")

    log(f"全語彙ロード: {UNIQUE_WORDS_PATH}")
    all_words = load_word_list(UNIQUE_WORDS_PATH)
    all_words_set = set(all_words)
    log(f"  全語彙数: {len(all_words_set):,} 語")

    # ── Step 1: 動的分類の全体集計 ──────────────────────────────────────
    log("\n--- Step 1: 動的分類（find_v8_splits_first）の全体集計 ---")
    dynamic_compound = {}
    dynamic_single   = []
    dynamic_unmatched = []

    for w in sorted(all_words_set):
        if len(w) < 2:
            continue
        splits = find_v8_splits_first(w)
        if splits is not None:
            dynamic_compound[w] = splits
        elif is_base(w):
            dynamic_single.append(w)
        else:
            dynamic_unmatched.append(w)

    log(f"  動的分類: 複合語    = {len(dynamic_compound):,} 語")
    log(f"  動的分類: 単独ベース語 = {len(dynamic_single):,} 語")
    log(f"  動的分類: 未マッチ  = {len(dynamic_unmatched):,} 語")

    # ── Step 2: 偽陽性の検出（単独ベース語 → 誤って複合語と判定）──────
    log("\n--- Step 2: 偽陽性検出（words_base_only.txt 中で誤判定された語）---")
    false_positives = []  # (word, splits) - 単独ベース語なのに複合語判定
    for word in single_words_gt:
        splits = find_v8_splits_first(word)
        if splits is not None:
            false_positives.append((word, splits))

    fp_count = len(false_positives)
    fp_rate  = fp_count / len(single_words_gt) * 100 if single_words_gt else 0.0
    log(f"  偽陽性件数: {fp_count:,} / {len(single_words_gt):,} 語 ({fp_rate:.1f}%)")

    # 先頭文字別の偽陽性分布
    fp_first_chars = Counter(w[0] for w, _ in false_positives if w)
    slot0_chars = set(SLOTS_V8[0])  # {l, r, o, y, s, v}
    fp_slot0 = sum(c for ch, c in fp_first_chars.items() if ch in slot0_chars)
    fp_total_first = sum(fp_first_chars.values())
    log(f"  偽陽性のうち SLOTS_V8[0]({set(SLOTS_V8[0])}) で始まる語: "
        f"{fp_slot0:,} / {fp_total_first:,} ({fp_slot0/fp_total_first*100:.1f}% if fp_total_first else 0)")

    # ── Step 3: 偽陰性の確認（正解複合語が動的分類で検出されない）──────
    log("\n--- Step 3: 偽陰性確認（正解複合語が動的分類で未検出）---")
    false_negatives = [w for w in compound_words_gt if w not in dynamic_compound]
    fn_count = len(false_negatives)
    log(f"  偽陰性件数: {fn_count:,} / {len(compound_words_gt):,} 語")

    # ── Step 4: 比較サマリ ───────────────────────────────────────────────
    log("\n--- Step 4: 分類数の比較 ---")
    log(f"  正解・複合語数   : {len(compound_words_gt):,} 語")
    log(f"  動的分類・複合語数: {len(dynamic_compound):,} 語  "
        f"（約 {len(dynamic_compound)/len(compound_words_gt):.1f} 倍）")
    log(f"  差分（過剰分類）  : {len(dynamic_compound) - len(compound_words_gt):,} 語")

    # ── Step 5: レポート生成 ─────────────────────────────────────────────
    log("\n--- Step 5: レポート生成 ---")

    # 代表例（偽陽性の最初の30件）
    fp_examples = false_positives[:30]
    fp_example_lines = []
    for word, splits in fp_examples:
        fp_example_lines.append(f"| `{word}` | `{' + '.join(splits)}` | {word[0] in slot0_chars} |")

    # 先頭文字別テーブル
    first_char_lines = []
    for ch in sorted(fp_first_chars, key=lambda c: -fp_first_chars[c]):
        cnt  = fp_first_chars[ch]
        is_s0 = "✓" if ch in slot0_chars else ""
        first_char_lines.append(f"| `{ch}` | {cnt:,} | {cnt/fp_count*100:.1f}% | {is_s0} |")

    # 全体分類テーブル
    single_words_set = set(single_words_gt)
    compound_words_dynamic_set = set(dynamic_compound.keys())

    report = f"""# find_v8_splits_first() 分類バグ検証レポート

**作成日時**: {datetime.now():%Y-%m-%d %H:%M:%S}
**スクリプト**: `hypothesis/04_re-compound_hmm/source/verify_classification.py`

---

## 1. 問題の概要

`hypothesis/02_compound_hmm` および `hypothesis/03_trigram/source/trigram_role_analysis.py` で
使用されている `find_v8_splits_first()` 関数は、動的に語を「複合語 / 単独ベース語」に分類する。

この関数の問題点:
- `word` 自体が `is_base(word) == True`（単独ベース語）であっても、
  `word[:i]` と `word[i:]` の両方が valid base であれば**複合語と判定**してしまう。
- SLOTS_V8[0]（`{{l,r,o,y,s,v}}`）と SLOTS_V8[15]（`{{k,t,p,f,l,r,o,y}}`）の重複文字により、
  これらの文字で始まる単独ベース語の多くが誤って分割可能と判定される。

---

## 2. 分類数の比較

| 分類方法 | 複合語数 | 単独ベース語数 | 備考 |
|---------|---------|-------------|------|
| **正解** (`compound_words_sorted.txt` / `words_base_only.txt`) | **{len(compound_words_gt):,}** | **{len(single_words_gt):,}** | `hypothesis/00_slot_model` が生成 |
| 動的分類 (`find_v8_splits_first()`) | **{len(dynamic_compound):,}** | {len(dynamic_single):,} | hypothesis/02, 03 で使用 |
| **差分（過剰分類）** | **+{len(dynamic_compound) - len(compound_words_gt):,}** | − | 約 **{len(dynamic_compound)/len(compound_words_gt):.1f}倍** に膨張 |

動的分類による複合語数は正解の約 **{len(dynamic_compound)/len(compound_words_gt):.1f} 倍**に膨張している。

---

## 3. 偽陽性分析（単独ベース語 → 誤って複合語と判定）

`words_base_only.txt`（正解・単独ベース語 {len(single_words_gt):,} 語）に対して
`find_v8_splits_first()` を適用した結果:

| 指標 | 値 |
|-----|-----|
| 偽陽性件数 | **{fp_count:,} 語** ({fp_rate:.1f}%) |
| SLOTS_V8[0] で始まる偽陽性 | **{fp_slot0:,} 語** ({fp_slot0/fp_count*100:.1f}% of FP) |
| 偽陰性件数（正解複合語が未検出） | {fn_count:,} 語 |

**「処理が正しい場合 `words_base_only.txt` が抽出されるはずです」の検証結果**:
→ `words_base_only.txt` の {fp_rate:.1f}% ({fp_count:,} 語) が誤って複合語と判定される。
→ 正しい処理では偽陽性 = 0 件であるべき。

---

## 4. 偽陽性の先頭文字分布

SLOTS_V8[0] = `{{l, r, o, y, s, v}}` は基の先頭文字集合であり、
末尾文字集合 SLOTS_V8[15] との重複 `{{l, r, o, y}}` が偽陽性の主因となっている。

| 先頭文字 | 偽陽性件数 | 割合 | SLOTS_V8[0] |
|---------|----------|------|------------|
{chr(10).join(first_char_lines)}

---

## 5. 偽陽性の代表例（最初の {min(30, fp_count)} 件）

| 単独ベース語 | 誤った分割 | SLOTS_V8[0] |
|------------|-----------|------------|
{chr(10).join(fp_example_lines)}

---

## 6. 結論

`find_v8_splits_first()` は単独ベース語 `words_base_only.txt` の
**{fp_rate:.1f}% ({fp_count:,}語)** を誤って複合語と判定する。

この偽陽性により、hypothesis/02 および 03 の分析では:
- 分析対象の複合語数が正解 {len(compound_words_gt):,} 語から **{len(dynamic_compound):,} 語** に膨張
- B-start / B-end のメトリクスが偽陽性含有の集合を対象に計算されている
- 偽陽性の境界位置は実際には単独ベース語の語中に設定されたノイズである

**修正方針**: `hypothesis/04_re-compound_hmm` では、動的分類を廃止し、
`compound_words.txt`（分割情報付き）を正解データとして直接使用する。

---

_本レポートは `verify_classification.py` により自動生成。_
"""

    report_path = OUT_DIR / "classification_discrepancy_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"レポート保存: {report_path}")

    # テキストサマリも出力
    print("\n" + "=" * 60)
    print("【検証サマリ】")
    print(f"  正解複合語数     : {len(compound_words_gt):,}")
    print(f"  動的分類複合語数 : {len(dynamic_compound):,}  (約{len(dynamic_compound)/len(compound_words_gt):.1f}倍)")
    print(f"  偽陽性数（FP）   : {fp_count:,} / {len(single_words_gt):,} 単独ベース語  ({fp_rate:.1f}%)")
    print(f"  偽陰性数（FN）   : {fn_count:,} / {len(compound_words_gt):,} 正解複合語")
    print(f"  → words_base_only.txt の語を正しく抽出できる処理への修正が必要")
    print("=" * 60)

    log("=== 検証完了 ===")
