slots = [
    ['l', 'r', 'o', 'y', 's', 'v'],           # 0
    ['q', 's', 'd', 'x', 'l', 'r', 'h', 'z'], # 1
    ['o', 'y'],                                  # 2
    ['d', 'r'],                                  # 3
    ['t', 'k', 'p', 'f'],                        # 4
    ['ch', 'sh'],                                # 5
    ['cth', 'ckh', 'cph', 'cfh'],                # 6
    ['eee', 'ee', 'e', 'g'],                     # 7
    ['k', 't', 'p', 'f', 'ch', 'sh', 'l', 'r', 'o', 'y'],  # 8
    ['s', 'd', 'c'],                             # 9
    ['o', 'a', 'y'],                             # 10
    ['iii', 'ii', 'i'],                          # 11
    ['d', 'l', 'r', 'm', 'n'],                   # 12
    ['s'],                                       # 13
    ['y'],                                       # 14
    ['k', 't', 'p', 'f', 'l', 'r', 'o', 'y'],   # 15
]

def parse_base_greedy(word, start=0):
    """Greedily parse one V4 base from word[start:]. Returns end_pos."""
    pos = start
    next_slot = 0
    while next_slot <= 15 and pos < len(word):
        found = False
        for slot_idx in range(next_slot, 16):
            for cand in slots[slot_idx]:
                if word[pos:pos+len(cand)] == cand:
                    pos += len(cand)
                    next_slot = slot_idx + 1
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return pos  # returns start if nothing matched

def parse_word(word):
    """Parse word into 1-4 V4 bases. Returns list of (start,end) or None."""
    n = len(word)
    # Try increasing number of bases; prefer fewer
    pos = 0
    bases = []
    for _ in range(4):
        if pos >= n:
            break
        end = parse_base_greedy(word, pos)
        if end == pos:
            return None
        bases.append((pos, end))
        pos = end
        if pos == n:
            return bases
    return None

with open('v02_simple_word/unique_word.txt', encoding='utf-8') as f:
    words = [ln.strip() for ln in f if ln.strip()]

single, compound, failed = [], [], []

for w in words:
    r = parse_word(w)
    if r is None:
        failed.append(w)
    elif len(r) == 1:
        single.append(w)
    else:
        compound.append((w, r))

print(f"総語数: {len(words)}")
print(f"単体語 (1 base):  {len(single)}")
print(f"複合語 (2+ bases): {len(compound)}")
print(f"未解析:           {len(failed)}")

print("\n=== 複合語一覧 (貪欲マッチによる分離) ===")
for w, bases in compound:
    parts = [w[s:e] for s, e in bases]
    n = len(parts)
    print(f"[{n}基] {w}  ->  {' + '.join(parts)}")

# 複合語を構成するベース語の集合（単体語と重複含む）
compound_bases = sorted(set(w[s:e] for w, bases in compound for s, e in bases))
print(f"\n=== 複合語から得られる構成ベース語 ({len(compound_bases)} 語) ===")
for b in compound_bases:
    print(b)

if failed:
    print(f"\n=== 未解析語 ({len(failed)} 語) ===")
    for w in failed:
        print(w)
