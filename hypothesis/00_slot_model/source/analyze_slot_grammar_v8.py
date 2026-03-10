#!/usr/bin/env python3
"""
Voynich Manuscript - Slot Grammar Analyzer v8

v8文法の定義:
  v7スロット文法に z をslot1へ追加
  (tazain, zepchy に出現するグリフ z を slot1 の語頭子音として収録)
"""

from pathlib import Path
from functools import lru_cache

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


def parse_greedy(word: str) -> tuple[list, str]:
    pos = 0
    matched = []
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


@lru_cache(maxsize=None)
def is_v8(word: str) -> bool:
    """v8文法を最大4回繰り返してマッチするか"""
    if is_base(word):
        return True
    for i in range(1, len(word)):
        p1 = word[:i]
        if not is_base(p1):
            continue
        rest = word[i:]
        if is_base(rest):
            return True
        for j in range(1, len(rest)):
            p2 = rest[:j]
            if not is_base(p2):
                continue
            rest2 = rest[j:]
            if is_base(rest2):
                return True
            for k in range(1, len(rest2)):
                if is_base(rest2[:k]) and is_base(rest2[k:]):
                    return True
    return False


def main():
    base = Path(__file__).parent
    words = [w.strip() for w in (base / "unique_word.txt").read_text(encoding="utf-8").splitlines() if w.strip()]

    matched_v8 = []
    unmatched_v8 = []

    for word in words:
        if is_v8(word):
            matched_v8.append(word)
        else:
            unmatched_v8.append(word)

    print(f"総単語数    : {len(words)}")
    print(f"v8 マッチ   : {len(matched_v8)} ({len(matched_v8)/len(words)*100:.1f}%)")
    print(f"v8 未マッチ : {len(unmatched_v8)} ({len(unmatched_v8)/len(words)*100:.1f}%)")

    print("\n--- v8 未マッチ単語 (全件) ---")
    for w in unmatched_v8:
        m, r = parse_greedy(w)
        slot_str = " ".join(f"[{i}:{s}]" for i, s in m) if m else "(none)"
        print(f"  {w:28s} {slot_str}  remaining={r!r}")

    out_path = base / "unmatched_words_v8.txt"
    lines = []
    for w in unmatched_v8:
        m, r = parse_greedy(w)
        slot_str = " ".join(f"[{i}:{s}]" for i, s in m) if m else "(none)"
        lines.append(f"{w}\t{slot_str}\tremaining={r!r}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[出力] v8 未マッチ単語 -> {out_path}")


if __name__ == "__main__":
    main()
