slots = [
    ['l', 'r', 'o', 'y', 's', 'v'],
    ['q', 's', 'd', 'x', 'l', 'r', 'h', 'z'],
    ['o', 'y'],
    ['d', 'r'],
    ['t', 'k', 'p', 'f'],
    ['ch', 'sh'],
    ['cth', 'ckh', 'cph', 'cfh'],
    ['eee', 'ee', 'e', 'g'],
    ['k', 't', 'p', 'f', 'ch', 'sh', 'l', 'r', 'o', 'y'],
    ['s', 'd', 'c'],
    ['o', 'a', 'y'],
    ['iii', 'ii', 'i'],
    ['d', 'l', 'r', 'm', 'n'],
    ['s'],
    ['y'],
    ['k', 't', 'p', 'f', 'l', 'r', 'o', 'y'],
]

def parse_base_greedy(word, start=0):
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
    return pos

def parse_word(word):
    n = len(word)
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

# 統計
print(f"総語数: {len(words)}")
print(f"単体語 (1 base):   {len(single)}")
print(f"複合語 (2+ bases): {len(compound)}")
print(f"  うち 2基:        {sum(1 for _,b in compound if len(b)==2)}")
print(f"  うち 3基:        {sum(1 for _,b in compound if len(b)==3)}")
print(f"  うち 4基:        {sum(1 for _,b in compound if len(b)==4)}")
print(f"未解析:            {len(failed)}")

# 複合語ファイル
with open('v02_simple_word/compound_words.txt', 'w', encoding='utf-8') as f:
    f.write(f"# 複合語一覧 ({len(compound)} 語) — 貪欲マッチによる V4 ベース分離\n")
    f.write("# 形式: [N基] 元語  ->  base1 + base2 + ...\n\n")
    for w, bases in compound:
        parts = [w[s:e] for s, e in bases]
        f.write(f"[{len(parts)}基] {w}  ->  {' + '.join(parts)}\n")

# 構成ベース語ファイル (複合語から取り出したベース語のみ、単体語は含まない)
comp_bases = sorted(set(w[s:e] for w, bases in compound for s, e in bases))
with open('v02_simple_word/compound_bases.txt', 'w', encoding='utf-8') as f:
    f.write(f"# 複合語から抽出されたベース語一覧 ({len(comp_bases)} 語)\n\n")
    for b in comp_bases:
        f.write(b + '\n')

print(f"\n複合語構成ベース語のユニーク数: {len(comp_bases)}")

# 先頭30件だけ表示
print("\n--- 複合語サンプル (先頭30件) ---")
for w, bases in compound[:30]:
    parts = [w[s:e] for s, e in bases]
    print(f"  [{len(parts)}基] {w}  ->  {' + '.join(parts)}")

if failed:
    print(f"\n=== 未解析語 ({len(failed)} 語) ===")
    for w in failed: print(w)
