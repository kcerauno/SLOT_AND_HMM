"""
e2_grammar_variants.py
========================
提案 E-2: V8 文法バリアント感度分析

目的:
  SLOTS_V8 の主要スロット定義を小幅に変更し、変更後の compound_words を再生成した上で
  hypothesis/05 と同じ役割分析を実行。Focus State が安定して同定されるかを確認する。

文法バリアント:
  V0_baseline  : SLOTS_V8 オリジナル（基準）
  V1_expand_vow: slot 2 (V_a: o,y) に 'i' を追加 → [o, y, i]
  V2_revert_z  : slot 1 (C2) から 'z' を除去 → v7 文法相当
  V3_expand_va2: slot 10 (V_a2: o,a,y) に 'e' を追加 → [o, a, y, e]

注記:
  元の compound_words.txt は "V4 文法" で生成されており、SLOTS_V8 では is_base() にならない
  "ck" 等の文字列を含む。E-2 では SLOTS_V8 をベースラインとして新たに複合語リストを生成し、
  各バリアントと比較する。単独語参照 (words_base_only.txt) は固定する。

実行:
  cd /home/practi/work_voy
  PYTHONPATH=/home/practi/work_voy/.venv/lib/python3.10/site-packages \\
    python3.10 hypothesis/08_v8_sensitivity/source/e2_grammar_variants.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
from pathlib import Path
from scipy import stats
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore")

# ── hypothesis/06 の common.py を再利用 ──────────────────────────────────────
_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent.parent
SRC06 = ROOT / "hypothesis/06_state_characterization/source"
sys.path.insert(0, str(SRC06))

import common  # noqa: E402

# ── 設定 ──────────────────────────────────────────────────────────────────────
OUT_DIR = ROOT / "hypothesis/08_v8_sensitivity/results/e2_variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [7, 8]

# ── 文法バリアント定義 ────────────────────────────────────────────────────────
# SLOTS_V8 オリジナル（analyze_slot_grammar_v8.py と同一）
_SLOTS_V8_BASE = [
    ["l", "r", "o", "y", "s", "v"],
    ["q", "s", "d", "x", "l", "r", "h", "z"],
    ["o", "y"],
    ["d", "r"],
    ["t", "k", "p", "f"],
    ["ch", "sh"],
    ["cth", "ckh", "cph", "cfh"],
    ["eee", "ee", "e", "g"],
    ["k", "t", "p", "f", "ch", "sh", "l", "r", "o", "y"],
    ["s", "d", "c"],
    ["o", "a", "y"],
    ["iii", "ii", "i"],
    ["d", "l", "r", "m", "n"],
    ["s"],
    ["y"],
    ["k", "t", "p", "f", "l", "r", "o", "y"],
]

def _copy_slots(base: list) -> list:
    return [list(slot) for slot in base]

VARIANTS: dict[str, list] = {
    "V0_baseline": _copy_slots(_SLOTS_V8_BASE),
    "V1_expand_vow": (lambda s: (s.__setitem__(2, ["o", "y", "i"]) or s))(_copy_slots(_SLOTS_V8_BASE)),
    "V2_revert_z":   (lambda s: (s.__setitem__(1, ["q", "s", "d", "x", "l", "r", "h"]) or s))(_copy_slots(_SLOTS_V8_BASE)),
    "V3_expand_va2": (lambda s: (s.__setitem__(10, ["o", "a", "y", "e"]) or s))(_copy_slots(_SLOTS_V8_BASE)),
}

VARIANT_DESCRIPTIONS = {
    "V0_baseline":   "基準 (SLOTS_V8 オリジナル)",
    "V1_expand_vow": "拡張: slot 2 (V_a) に 'i' 追加 → [o, y, i]",
    "V2_revert_z":   "制限: slot 1 (C2) から 'z' 除去 → v7 相当",
    "V3_expand_va2": "拡張: slot 10 (V_a2) に 'e' 追加 → [o, a, y, e]",
}

VARIANT_DIFF = {
    "V0_baseline":   "変更なし",
    "V1_expand_vow": "slot[2]: ['o','y'] → ['o','y','i']  (+1文字)",
    "V2_revert_z":   "slot[1]: ['q','s','d','x','l','r','h','z'] → ['q','s','d','x','l','r','h']  (-1文字)",
    "V3_expand_va2": "slot[10]: ['o','a','y'] → ['o','a','y','e']  (+1文字)",
}


# ════════════════════════════════════════════════════════════════════════════
# 1. バリアント文法による 基判定・複合語生成
# ════════════════════════════════════════════════════════════════════════════
def parse_greedy_variant(word: str, slots: list) -> tuple[list, str]:
    """貪欲マッチ: 各スロットを先頭から試し、マッチしたオプションを消費する。"""
    pos = 0
    matched = []
    for idx, options in enumerate(slots):
        if pos >= len(word):
            break
        for opt in options:
            if word.startswith(opt, pos):
                matched.append((idx, opt))
                pos += len(opt)
                break
    return matched, word[pos:]


def is_base_variant(word: str, slots: list) -> bool:
    """word が slots 文法で完全にマッチするか（残余なし）。"""
    _, remaining = parse_greedy_variant(word, slots)
    return remaining == "" and bool(word)


def find_splits_variant(word: str, slots: list, max_bases: int = 4) -> list | None:
    """
    word を 2〜max_bases 個の基に分割する最初の有効な分割を返す。
    見つからない場合は None（単独基または分割不可）。
    """
    if is_base_variant(word, slots):
        return None  # 単独基（複合語ではない）

    L = len(word)
    for i in range(1, L):
        p1 = word[:i]
        if not is_base_variant(p1, slots):
            continue
        rest = word[i:]

        # 2 分割
        if is_base_variant(rest, slots):
            return [p1, rest]

        if max_bases < 3:
            continue

        # 3 分割
        for j in range(1, len(rest)):
            p2 = rest[:j]
            if not is_base_variant(p2, slots):
                continue
            rest2 = rest[j:]

            if is_base_variant(rest2, slots):
                return [p1, p2, rest2]

            if max_bases < 4:
                continue

            # 4 分割
            for kk in range(1, len(rest2)):
                if is_base_variant(rest2[:kk], slots) and is_base_variant(rest2[kk:], slots):
                    return [p1, p2, rest2[:kk], rest2[kk:]]
    return None


def generate_compound_list(all_words: list, slots: list) -> dict:
    """
    all_words から各バリアント文法で複合語リストを生成。
    {word: (base1, base2, ...)} を返す。
    """
    compounds = {}
    for word in all_words:
        splits = find_splits_variant(word, slots)
        if splits is not None:
            compounds[word] = tuple(splits)
    return compounds


def save_compound_list(compounds: dict, out_path: Path, variant_name: str, description: str):
    """compound_words.txt と同じ形式で保存。"""
    lines = [
        f"# 複合語一覧 ({len(compounds):,} 語) — {variant_name}: {description}",
        "# 形式: [N基] 元語  ->  base1 + base2 + ...",
        "",
    ]
    for word in sorted(compounds.keys()):
        bases   = compounds[word]
        n_bases = len(bases)
        bases_str = " + ".join(bases)
        lines.append(f"[{n_bases}基] {word}  ->  {bases_str}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    common.log(f"    複合語リスト保存: {out_path.name}  ({len(compounds):,} 語)")


# ════════════════════════════════════════════════════════════════════════════
# 2. グループ収集 & 役割分析（hypothesis/05 と同一ロジック）
# ════════════════════════════════════════════════════════════════════════════
def collect_groups(compound_splits: dict, decoded_compound: dict,
                   decoded_single: dict, single_words: list) -> dict:
    b_start, b_end, s_head, s_mid = [], [], [], []

    for word, splits in compound_splits.items():
        states = decoded_compound.get(word)
        if states is None:
            continue
        bd_end, bd_start = common.get_boundary_positions(splits)
        for pos, state in enumerate(states):
            if pos in bd_start:
                b_start.append(int(state))
            if pos in bd_end:
                b_end.append(int(state))

    for word in single_words:
        states = decoded_single.get(word)
        if states is None:
            continue
        L = len(states)
        s_head.append(int(states[0]))
        s_mid.append(int(states[L // 2]))

    return {"B-start": b_start, "B-end": b_end, "S-head": s_head, "S-mid": s_mid}


def fisher_2x2(group_a: list, group_b: list, focus: int, k: int) -> dict:
    ca = np.bincount(group_a, minlength=k).astype(float)
    cb = np.bincount(group_b, minlength=k).astype(float)
    a_t = int(ca[focus]); a_o = int(ca.sum()) - a_t
    b_t = int(cb[focus]); b_o = int(cb.sum()) - b_t
    if a_t + b_t == 0:
        return {"odds": float("nan"), "p": float("nan"),
                "a_rate": 0.0, "b_rate": 0.0}
    odds, p = stats.fisher_exact([[a_t, a_o], [b_t, b_o]], alternative="two-sided")
    return {
        "odds":   float(odds),
        "p":      float(p),
        "a_rate": ca[focus] / ca.sum() * 100 if ca.sum() > 0 else 0.0,
        "b_rate": cb[focus] / cb.sum() * 100 if cb.sum() > 0 else 0.0,
    }


def run_role_analysis(groups: dict, k: int, occupancy: np.ndarray) -> dict | None:
    """
    data-driven Focus State 同定（Phantom 除外）+ 4 比較 Fisher 検定。
    hypothesis/05 の run_role_analysis と同一ロジック。
    """
    if len(groups["B-start"]) == 0:
        return None

    b_start_counts = np.bincount(groups["B-start"], minlength=k).astype(float)
    rates = b_start_counts.copy()
    for s in range(k):
        if occupancy[s] == 0:
            rates[s] = -1.0
    focus = int(np.argmax(rates))

    group_rates  = {}
    group_counts = {}
    group_totals = {}
    for name, sl in groups.items():
        c = np.bincount(sl, minlength=k).astype(float)
        t = c.sum()
        group_counts[name] = c
        group_totals[name] = t
        group_rates[name]  = c / t * 100 if t > 0 else c * 0

    comparisons = [
        ("B-start", "S-head",  "①confound検定 (B-start vs S-head)"),
        ("B-start", "S-mid",   "②構造効果 (B-start vs S-mid)"),
        ("B-end",   "B-start", "③境界末尾 vs 境界先頭"),
        ("B-end",   "S-mid",   "④境界末尾 vs 語中央"),
    ]
    fisher_results = {}
    for gA, gB, label in comparisons:
        fisher_results[label] = fisher_2x2(groups[gA], groups[gB], focus, k)

    return {
        "focus_state":    focus,
        "group_rates":    group_rates,
        "group_counts":   group_counts,
        "group_totals":   group_totals,
        "fisher_results": fisher_results,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 可視化
# ════════════════════════════════════════════════════════════════════════════
def plot_group_rates(role_result: dict, k: int, variant_name: str, out_path: Path):
    fs          = role_result["focus_state"]
    group_rates = role_result["group_rates"]
    group_totals = role_result["group_totals"]

    order   = ["B-start", "B-end", "S-head", "S-mid"]
    rates   = [group_rates[g][fs] for g in order]
    totals  = [int(group_totals[g]) for g in order]
    colors  = ["#C0392B", "#E59866", "#2980B9", "#7FB3D3"]
    xlabels = [f"{g}\n(n={t:,})" for g, t in zip(order, totals)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(xlabels, rates, color=colors, alpha=0.88, edgecolor="white", linewidth=1.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    f1 = role_result["fisher_results"]["①confound検定 (B-start vs S-head)"]
    if not np.isnan(f1["p"]):
        note = (f"Fisher①: p={f1['p']:.3f} (n.s.)" if f1["p"] > 0.05
                else f"Fisher①: p={f1['p']:.2e}  → 構造効果")
        note_color = "#C0392B" if f1["p"] > 0.05 else "#1A5276"
        ax.text(0.5, 0.97, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=note_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

    ax.set_ylabel(f"S{fs} 出現率 (%)", fontsize=11)
    ax.set_title(
        f"[E-2 バリアント] {variant_name}  k={k}  4グループ S{fs} 出現率\n"
        f"複合語 {int(group_totals['B-start']):,} 境界 / 単独語 {int(group_totals['S-head']):,} 語",
        fontsize=10,
    )
    ax.set_ylim(0, max(rates) * 1.40 + 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    common.log(f"      Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# 4. Markdown レポート
# ════════════════════════════════════════════════════════════════════════════
def build_report(all_variant_results: dict, ref_compound_count: int) -> str:
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# E-2: 文法バリアント感度分析レポート",
        "",
        f"生成日時: {now}",
        "",
        "## 概要",
        "",
        "SLOTS_V8 のスロット定義を小幅に変更し、変更後の複合語リストで",
        "同じ役割分析を実行。Focus State が安定して同定されるかを確認する。",
        "",
        f"参照: 元の compound_words.txt (V4 文法) = **{ref_compound_count:,}** 語  "
        "(V4 文法は 'ck' 等を単独基として扱うため SLOTS_V8 ベースラインとは異なる)",
        "",
        "## 文法バリアント定義",
        "",
        "| バリアント | 変更内容 |",
        "|-----------|---------|",
    ]
    for vname, vdesc in VARIANT_DESCRIPTIONS.items():
        lines.append(f"| {vname} | {vdesc} |")

    lines += [
        "",
        "---",
        "",
        "## 複合語数の比較",
        "",
        "| バリアント | 複合語数 | 差分 (vs V0) |",
        "|-----------|---------|------------|",
    ]
    base_n = None
    for vname, vres in all_variant_results.items():
        n = vres.get("n_compounds", 0)
        if vname == "V0_baseline":
            base_n = n
        diff_str = "" if base_n is None or vname == "V0_baseline" else f"{n - base_n:+d}"
        lines.append(f"| {vname} | {n:,} | {diff_str} |")

    lines += [
        "",
        "---",
        "",
        "## 役割分析結果サマリー",
        "",
        "| バリアント | k | Focus State | B-start 率 | S-head 率 | Fisher① p | 判定 |",
        "|-----------|---|------------|-----------|---------|---------|------|",
    ]

    for vname, vres in all_variant_results.items():
        for k, role in vres.get("roles", {}).items():
            if role is None:
                lines.append(f"| {vname} | {k} | — | — | — | — | 分析失敗 |")
                continue
            fs     = role["focus_state"]
            gr     = role["group_rates"]
            f1     = role["fisher_results"]["①confound検定 (B-start vs S-head)"]
            bstart = gr["B-start"][fs]
            shead  = gr["S-head"][fs]
            p      = f1["p"]
            if np.isnan(p):
                verdict, p_str = "計算不能", "N/A"
            elif p > 0.05:
                verdict, p_str = "有意差なし", f"{p:.3f}"
            elif bstart > shead:
                verdict, p_str = "**構造効果あり**", f"{p:.2e}"
            else:
                verdict, p_str = "**逆転**", f"{p:.2e}"
            lines.append(
                f"| {vname} | {k} | S{fs} | {bstart:.1f}% | {shead:.1f}% "
                f"| {p_str} | {verdict} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 解釈指針",
        "",
        "- **全バリアントで同じ Focus State + 有意な Fisher p**: 結果は文法定義に対して頑健",
        "- **Focus State が変わるが Fisher p は有意**: 部分的頑健（状態割り当てが変動するが境界シグナルは保持）",
        "- **特定バリアントで Fisher p が非有意**: 該当スロット変更が境界シグナルに影響",
        "",
        "## 注記: V0_baseline と元 compound_words.txt の関係",
        "",
        "元の compound_words.txt は 'V4 文法' で生成されており、'ck' などを単独基として扱う。",
        "SLOTS_V8 の `is_base()` ではこれらは基と認識されないため、",
        "V0_baseline の複合語リストは元ファイルと異なる。",
        "E-2 では SLOTS_V8 内部での一貫した比較を目的とする。",
        "",
        "_本レポートは e2_grammar_variants.py により自動生成。_",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    common.log("=== E-2: 文法バリアント感度分析 開始 ===")

    # データロード
    ref_compound_splits = common.load_compound_splits()  # 参照用（元 V4 ファイル）
    single_words        = common.load_single_words()
    all_words           = common.load_all_words()
    char2idx            = common.ORIG_CHAR2IDX

    single_words = [
        w for w in single_words
        if len(w) >= 2 and all(c in char2idx for c in w)
    ]
    common.log(f"単独語（参照固定）: {len(single_words):,}")
    common.log(f"全語彙: {len(all_words):,}")
    common.log(f"元 compound_words.txt (V4): {len(ref_compound_splits):,} 語")

    # モデルロード（trigram のみ）
    models = {}
    for k in K_LIST:
        m = common.load_trigram_model(k)
        if m:
            models[k] = m
            common.log(f"Trigram k={k}: ロード完了 (logL={m['logL']:.2f})")

    # 単独語デコード（一度だけ）
    common.log("\n単独語デコード中（全モデル共通）...")
    decoded_single = {}
    for k, model in models.items():
        common.log(f"  Trigram k={k}...")
        decoded_single[k] = common.decode_words(single_words, model, "trigram")

    # Viterbi 占有率（一度だけ）
    common.log("\nViterbi 占有率計算中...")
    occupancy = {}
    for k, model in models.items():
        occ = common.compute_occupancy(all_words, model, "trigram")
        occupancy[k] = occ
        phantoms = [f"S{s}" for s in range(k) if occ[s] == 0.0]
        common.log(f"  Trigram k={k}  Phantom: {phantoms}")

    # ── バリアントごとに分析 ─────────────────────────────────────────────────
    all_variant_results = {}

    for vname, slots in VARIANTS.items():
        common.log(f"\n{'═' * 60}")
        common.log(f"バリアント: {vname}")
        common.log(f"  {VARIANT_DESCRIPTIONS[vname]}")
        common.log(f"  変更: {VARIANT_DIFF[vname]}")
        common.log(f"{'═' * 60}")

        # 複合語リスト生成
        common.log("  複合語リスト生成中...")
        compounds_raw = generate_compound_list(all_words, slots)
        # 語彙フィルタ
        compounds = {
            w: s for w, s in compounds_raw.items()
            if all(c in char2idx for c in w)
        }
        common.log(f"  複合語数: {len(compounds):,}  (全 DB: {len(compounds_raw):,})")

        # 複合語リスト保存
        save_compound_list(compounds, OUT_DIR / f"compounds_{vname}.txt",
                           vname, VARIANT_DESCRIPTIONS[vname])

        roles = {}
        for k, model in models.items():
            common.log(f"\n  [k={k}] 複合語デコード中 ({len(compounds):,} 語)...")
            decoded_compound = common.decode_words(compounds.keys(), model, "trigram")
            common.log(f"    デコード成功: {len(decoded_compound):,} 語")

            groups = collect_groups(compounds, decoded_compound, decoded_single[k], single_words)
            for gname, glist in groups.items():
                common.log(f"    {gname}: {len(glist):,} 観測")

            role = run_role_analysis(groups, k, occupancy[k])
            if role is None:
                common.log(f"    [k={k}] 役割分析失敗（B-start 観測数ゼロ）")
                roles[k] = None
                continue

            fs = role["focus_state"]
            gr = role["group_rates"]
            f1 = role["fisher_results"]["①confound検定 (B-start vs S-head)"]
            common.log(f"    Focus State: S{fs}  "
                       f"B-start={gr['B-start'][fs]:.1f}%  "
                       f"S-head={gr['S-head'][fs]:.1f}%  "
                       f"Fisher①p={f1['p']:.2e}")
            roles[k] = role

            # 可視化
            plot_group_rates(
                role, k, vname,
                OUT_DIR / f"role_analysis_{vname}_k{k}.png"
            )

        all_variant_results[vname] = {
            "n_compounds": len(compounds),
            "roles":       roles,
        }

    # レポート生成
    common.log("\nMarkdown レポート生成中...")
    report_path = OUT_DIR / "variant_report.md"
    report_path.write_text(
        build_report(all_variant_results, len(ref_compound_splits)),
        encoding="utf-8",
    )
    common.log(f"レポート保存: {report_path}")
    common.log(f"\n✓ E-2 完了。出力先: {OUT_DIR.resolve()}")
