"""
single_vs_compound_analysis.py
================================
Section 6.2 検証: 非複合語（単独ベース語）との比較

交絡因子の検証:
  V8文法は複合語の第2基以降の先頭文字を {o,q,l,r,y} などに制約する。
  HMMがそれらを独立にクラスタリングした結果として境界集中が生じているなら、
  単独ベース語の語頭でも同じ状態が高頻度で現れるはず（交絡）。
  逆に複合境界の方が有意に高いなら、構造そのものが効いている。

4グループ比較:
  B-start : 複合語の第2基以降の先頭文字 (V8複合境界の先頭)
  B-end   : 複合語の第n基の末尾文字    (V8複合境界の末尾)
  S-head  : 単独ベース語の pos=0       (語頭)
  S-mid   : 単独ベース語の pos=L//2    (語中央)

コア検定 (Fisher①): B-start vs S-head
  p > 0.05 → 交絡確認（文字種が状態を決定）
  p < 0.05 かつ rate(B-start) > rate(S-head) → 複合構造の独立効果
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
except ImportError:
    print("ERROR: PyTorch がインストールされていません。")
    import sys
    sys.exit(1)


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {DEVICE}")


# ── 設定 ──────────────────────────────────────────────────────────────────
DB_PATH           = "data/voynich.db"
MODEL_CACHE       = Path("hypothesis/01_bigram/results/hmm_model_cache")
OUT_DIR           = Path("hypothesis/02_compound_hmm/results")
SINGLE_WORDS_PATH = Path("hypothesis/00_slot_model/data/words_base_only.txt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST        = [7, 8]
PHANTOM_STATE = {7: 3, 8: 4}   # 真の縮退状態（Viterbiに出現しない）
MIN_WORD_LEN  = 2

BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"


# ════════════════════════════════════════════════════════════════════════
# 1. V8文法定義 (compound_boundary_analysis.py から移植)
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


def get_boundary_positions(splits: tuple) -> tuple:
    boundary_end, boundary_start = set(), set()
    cumlen = 0
    for base in splits[:-1]:
        cumlen += len(base)
        boundary_end.add(cumlen - 1)
        boundary_start.add(cumlen)
    return boundary_end, boundary_start


# ════════════════════════════════════════════════════════════════════════
# 2. HMMモデル & Viterbi (compound_boundary_analysis.py から移植)
# ════════════════════════════════════════════════════════════════════════
def load_model(k: int):
    path = MODEL_CACHE / f"full_k{k}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {
        "start": d["start"], "trans": d["trans"],
        "emiss": d["emiss"], "logL": float(d["logL"][0]),
    }


def viterbi_pt(log_start, log_trans, log_emiss, X_np):
    X = torch.tensor(X_np, dtype=torch.long, device=DEVICE)
    T, K = X.shape[0], log_start.shape[0]
    log_delta = torch.empty(T, K, device=DEVICE)
    psi       = torch.empty(T, K, dtype=torch.long, device=DEVICE)
    log_delta[0] = log_start + log_emiss[:, X[0]]
    psi[0] = 0
    for t in range(1, T):
        vals = log_delta[t - 1].unsqueeze(1) + log_trans
        max_vals, argmax_vals = torch.max(vals, dim=0)
        log_delta[t] = max_vals + log_emiss[:, X[t]]
        psi[t] = argmax_vals
    path = torch.empty(T, dtype=torch.long, device=DEVICE)
    path[T - 1] = torch.argmax(log_delta[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path.cpu().numpy()


def decode_words(words, char2idx, log_start, log_trans, log_emiss) -> dict:
    """Viterbi デコードして {word -> 状態列(BOS/EOS除外)} を返す。"""
    results = {}
    for word in words:
        seq = np.array(
            [char2idx[BOS_CHAR]] + [char2idx[c] for c in word] + [char2idx[EOS_CHAR]],
            dtype=np.int32,
        )
        try:
            path = viterbi_pt(log_start, log_trans, log_emiss, seq)
            states = path[1:-1]   # BOS/EOS を除く
            if len(states) > 0:
                results[word] = states
        except Exception:
            pass
    return results


# ════════════════════════════════════════════════════════════════════════
# 3. 4グループの状態収集
# ════════════════════════════════════════════════════════════════════════
def collect_groups(compound_splits, single_words, decoded_compound, decoded_single) -> dict:
    """
    4グループの状態リストを収集する。

    Returns
    -------
    dict with keys: "B-start", "B-end", "S-head", "S-mid"
    """
    b_start, b_end, s_head, s_mid = [], [], [], []

    # 複合語: 境界先頭 / 境界末尾
    for word, splits in compound_splits.items():
        states = decoded_compound.get(word)
        if states is None:
            continue
        bd_end, bd_start = get_boundary_positions(splits)
        for pos, state in enumerate(states):
            if pos in bd_start:
                b_start.append(int(state))
            if pos in bd_end:
                b_end.append(int(state))

    # 単独ベース語: 語頭 / 語中央
    for word in single_words:
        states = decoded_single.get(word)
        if states is None:
            continue
        L = len(states)
        s_head.append(int(states[0]))
        s_mid.append(int(states[L // 2]))

    return {"B-start": b_start, "B-end": b_end, "S-head": s_head, "S-mid": s_mid}


# ════════════════════════════════════════════════════════════════════════
# 4. 統計検定
# ════════════════════════════════════════════════════════════════════════
def fisher_2x2(group_a: list, group_b: list, target_state: int, k: int) -> dict:
    """Fisher 正確検定: グループ A と B で target_state の出現率を比較。"""
    ca = np.bincount(group_a, minlength=k).astype(float)
    cb = np.bincount(group_b, minlength=k).astype(float)
    a_t = int(ca[target_state]);  a_o = int(ca.sum()) - a_t
    b_t = int(cb[target_state]);  b_o = int(cb.sum()) - b_t
    if a_t + b_t == 0:
        return {"odds": np.nan, "p": np.nan, "a_rate": 0.0, "b_rate": 0.0,
                "a_n": int(ca.sum()), "b_n": int(cb.sum())}
    odds, p = stats.fisher_exact([[a_t, a_o], [b_t, b_o]], alternative="two-sided")
    return {
        "odds":   odds,
        "p":      p,
        "a_rate": ca[target_state] / ca.sum() * 100 if ca.sum() > 0 else 0.0,
        "b_rate": cb[target_state] / cb.sum() * 100 if cb.sum() > 0 else 0.0,
        "a_n":    int(ca.sum()),
        "b_n":    int(cb.sum()),
    }


def chi2_4groups(groups: dict, target_state: int, k: int) -> dict:
    """4グループ間で target_state 出現率のカイ二乗検定。"""
    rows = []
    for name in ["B-start", "B-end", "S-head", "S-mid"]:
        states = groups[name]
        c = np.bincount(states, minlength=k).astype(float)
        rows.append([c[target_state], c.sum() - c[target_state]])
    contingency = np.array(rows)
    ok = contingency.sum(axis=1) > 0
    if ok.sum() < 2:
        return {"chi2": np.nan, "p": np.nan, "dof": np.nan}
    chi2, p, dof, _ = stats.chi2_contingency(contingency[ok])
    return {"chi2": chi2, "p": p, "dof": int(dof)}


def run_analysis(groups: dict, k: int, phantom_s: int):
    """各グループの状態レートを計算し統計検定を実施する。"""
    b_start_counts = np.bincount(groups["B-start"], minlength=k).astype(float)
    b_start_total  = b_start_counts.sum()
    if b_start_total == 0:
        return None

    # B-start での最大状態 (Phantom 除外) を focus_state とする
    rates_excl = b_start_counts / b_start_total
    rates_excl[phantom_s] = -1.0
    focus_state = int(np.argmax(rates_excl))

    # 各グループの全状態レート
    group_rates  = {}
    group_counts = {}
    group_totals = {}
    for name, states in groups.items():
        c = np.bincount(states, minlength=k).astype(float)
        t = c.sum()
        group_counts[name] = c
        group_totals[name] = t
        group_rates[name]  = c / t * 100 if t > 0 else c * 0

    # Fisher 検定（4ペア）
    comparisons = [
        ("B-start", "S-head",  "①confound検定 (B-start vs S-head)"),
        ("B-start", "S-mid",   "②構造効果 (B-start vs S-mid)"),
        ("B-end",   "B-start", "③境界末尾 vs 境界先頭"),
        ("B-end",   "S-mid",   "④境界末尾 vs 語中央"),
    ]
    fisher_results = {}
    for gA, gB, label in comparisons:
        fisher_results[label] = fisher_2x2(groups[gA], groups[gB], focus_state, k)

    # 4グループ カイ二乗
    chi2_res = chi2_4groups(groups, focus_state, k)

    return {
        "focus_state":   focus_state,
        "group_rates":   group_rates,
        "group_counts":  group_counts,
        "group_totals":  group_totals,
        "fisher_results": fisher_results,
        "chi2":          chi2_res,
    }


# ════════════════════════════════════════════════════════════════════════
# 5. 可視化
# ════════════════════════════════════════════════════════════════════════
def plot_group_rates(result: dict, k: int, out_dir: Path) -> str:
    """4グループの focus_state 出現率棒グラフを保存して filename を返す。"""
    focus_state  = result["focus_state"]
    group_rates  = result["group_rates"]
    group_totals = result["group_totals"]

    order  = ["B-start", "B-end", "S-head", "S-mid"]
    rates  = [group_rates[g][focus_state] for g in order]
    totals = [int(group_totals[g]) for g in order]

    colors  = ["#C0392B", "#E59866", "#2980B9", "#7FB3D3"]
    xlabels = [f"{g}\n(n={t:,})" for g, t in zip(order, totals)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(xlabels, rates, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=1.5)

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Fisher① 結果をテキストボックスで注記
    f1 = result["fisher_results"]["①confound検定 (B-start vs S-head)"]
    if not np.isnan(f1["p"]):
        if f1["p"] > 0.05:
            note = f"Fisher①: p={f1['p']:.3f} (n.s.) → 交絡の可能性"
            note_color = "#C0392B"
        else:
            note = f"Fisher①: p={f1['p']:.2e} → 複合構造の独立効果"
            note_color = "#1A5276"
        ax.text(
            0.5, 0.97, note,
            transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color=note_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
        )

    ax.set_ylabel(f"S{focus_state} 出現率 (%)", fontsize=11)
    ax.set_title(
        f"4グループ S{focus_state} 出現率比較  k={k}\n"
        f"赤=複合語境界, 青=単独ベース語",
        fontsize=12,
    )
    ax.set_ylim(0, max(rates) * 1.40 + 1)
    plt.tight_layout()

    fname = f"group_s{focus_state}_rate_k{k}.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


# ════════════════════════════════════════════════════════════════════════
# 6. Markdown レポート
# ════════════════════════════════════════════════════════════════════════
def build_md_report(results_by_k: dict, compound_splits: dict,
                    single_words: list, plot_files: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Section 6.2 検証: 非複合語（単独ベース語）との比較",
        "",
        f"生成日時: {now}",
        "",
        "## 検証概要",
        "",
        "**交絡仮説**: V8文法は複合語の第2基以降の先頭文字を {o,q,l,r,y} などに制約する。",
        "HMMがこれらを独立にクラスタリングした結果として境界集中が生じているだけかもしれない。",
        "",
        "**検証設計**: 4グループ比較",
        "",
        "| グループ | 定義 | 語種 |",
        "|---------|------|------|",
        "| B-start | 複合語の第2基以降の先頭文字（境界先頭） | 複合語 |",
        "| B-end   | 複合語の第n基の末尾文字（境界末尾）   | 複合語 |",
        "| S-head  | 単独ベース語の pos=0（語頭）          | 単独ベース語 |",
        "| S-mid   | 単独ベース語の pos=L//2（語中央）     | 単独ベース語 |",
        "",
        "**判定基準 (Fisher①: B-start vs S-head)**:",
        "- `rate(B-start) ≈ rate(S-head)` (p > 0.05) → **交絡確認** — 文字種制約で説明可能",
        "- `rate(B-start) >> rate(S-head)` (p < 0.05) → **構造効果あり** — 複合構造が独立に影響",
        "",
        f"**対象語数**: V8複合語 {len(compound_splits):,} 語 / 単独ベース語 {len(single_words):,} 語",
        "",
    ]

    for k, result in results_by_k.items():
        if result is None:
            lines += [f"## k={k}", "", "⚠ 分析失敗（モデルキャッシュが見つかりません）", ""]
            continue

        phantom_s   = PHANTOM_STATE[k]
        focus_state = result["focus_state"]
        gr          = result["group_rates"]
        gt          = result["group_totals"]
        gc          = result["group_counts"]
        chi2_res    = result["chi2"]

        # ── ヘッダ & 図 ────────────────────────────────────────────────
        lines += [
            f"## k={k}  (Phantom State: S{phantom_s} / Focus State: S{focus_state})",
            "",
        ]
        if k in plot_files:
            lines += [f"![S{focus_state} group rates](../{plot_files[k]})", ""]

        # ── 4グループ出現率テーブル ────────────────────────────────────
        lines += [
            f"### S{focus_state} 出現率（4グループ）",
            "",
            f"| グループ | S{focus_state} 出現率 | 件数 | 合計 |",
            "|---------|--------------|------|------|",
        ]
        for g in ["B-start", "B-end", "S-head", "S-mid"]:
            rate  = gr[g][focus_state]
            cnt   = int(gc[g][focus_state])
            total = int(gt[g])
            lines.append(f"| {g} | {rate:.2f}% | {cnt:,} | {total:,} |")

        lines.append("")

        # ── Fisher 検定テーブル ────────────────────────────────────────
        lines += [
            f"### Fisher 正確検定（S{focus_state} 出現率の差）",
            "",
            "| 比較 | グループA | グループB | 率A | 率B | p値 | オッズ比 | 判定 |",
            "|-----|-----------|-----------|-----|-----|-----|---------|------|",
        ]
        for label, fr in result["fisher_results"].items():
            p    = fr["p"]
            odds = fr["odds"]
            if np.isnan(p):
                p_str, verdict = "N/A", "—"
            elif p < 0.001:
                p_str   = f"{p:.2e}"
                verdict = "**有意差あり**"
            elif p < 0.05:
                p_str   = f"{p:.4f}"
                verdict = "有意差あり"
            else:
                p_str   = f"{p:.4f}"
                verdict = "有意差なし"
            odds_str = f"{odds:.3f}" if not np.isnan(odds) else "N/A"
            lines.append(
                f"| {label} "
                f"| {fr['a_rate']:.2f}% (n={fr['a_n']:,}) "
                f"| {fr['b_rate']:.2f}% (n={fr['b_n']:,}) "
                f"| {fr['a_rate']:.2f}% | {fr['b_rate']:.2f}% "
                f"| {p_str} | {odds_str} | {verdict} |"
            )

        lines.append("")

        # ── 4グループ カイ二乗 ─────────────────────────────────────────
        if not np.isnan(chi2_res["chi2"]):
            lines += [
                f"### 4グループ間 カイ二乗検定（S{focus_state}）",
                "",
                f"χ² = {chi2_res['chi2']:.2f},  dof = {chi2_res['dof']},  p = {chi2_res['p']:.3e}",
                "",
            ]
        else:
            lines += [
                f"### 4グループ間 カイ二乗検定（S{focus_state}）",
                "",
                "（計算不能: グループのいずれかに観測なし）",
                "",
            ]

        # ── 解釈 ──────────────────────────────────────────────────────
        f1 = result["fisher_results"]["①confound検定 (B-start vs S-head)"]
        bstart_rate = gr["B-start"][focus_state]
        shead_rate  = gr["S-head"][focus_state]
        smid_rate   = gr["S-mid"][focus_state]
        bend_rate   = gr["B-end"][focus_state]

        lines += [f"### 解釈 (k={k})", ""]

        if not np.isnan(f1["p"]):
            if f1["p"] > 0.05:
                lines += [
                    f"**【交絡確認】** B-start ({bstart_rate:.2f}%) ≈ S-head ({shead_rate:.2f}%)",
                    f"(Fisher①: p = {f1['p']:.4f}, 有意差なし)",
                    "",
                    f"S{focus_state} の境界集中は V8 文法の文字種制約（第2基先頭 = {{o,q,l,r,y}} など）",
                    "によって説明できる可能性が高い。",
                    "HMM は複合語構造を独立に検出しているとは言えない。",
                ]
            elif bstart_rate > shead_rate:
                lines += [
                    f"**【構造効果あり】** B-start ({bstart_rate:.2f}%) >> S-head ({shead_rate:.2f}%)",
                    f"(Fisher①: p = {f1['p']:.4e})",
                    "",
                    f"S{focus_state} の境界集中は文字種制約だけでは説明できず、",
                    "複合語構造そのものが HMM 状態遷移に影響している可能性がある。",
                    "V8 文法（独立）と HMM（独立）が同じ複合構造を発見したことを示唆する。",
                ]
            else:
                # B-start << S-head: 交絡仮説を逆方向で否定
                lines += [
                    f"**【交絡否定・語頭状態仮説】** B-start ({bstart_rate:.2f}%) << S-head ({shead_rate:.2f}%)",
                    f"(Fisher①: p = {f1['p']:.4e}, odds = {f1['odds']:.3f})",
                    "",
                    "交絡仮説は「B-start ≈ S-head（どちらも基先頭文字）」を予測するが、",
                    "実際には B-start が S-head を大幅に下回る。",
                    "この **逆転** は単純な文字種クラスタリング仮説を否定する。",
                    "",
                    f"S{focus_state} は「語全体の先頭（S-head: {shead_rate:.1f}%）」に特化した",
                    "**語頭状態** として機能していると解釈できる。",
                    "複合語の内部基先頭（B-start: {:.1f}%）では、".format(bstart_rate) +
                    "語の途中にあるため語頭状態が抑制される。",
                    "",
                    f"注目: B-end ({bend_rate:.2f}%) が B-start ({bstart_rate:.2f}%) を上回る点も重要。",
                    f"S{focus_state} は複合語内で「基の末尾（次の基への遷移点）」にも集中しており、",
                    "これは複合構造を反映した状態配分だが、",
                    "その機能は「文字種クラスタリング」ではなく「基末尾遷移点への特化」を示す。",
                ]
        else:
            lines += ["Fisher①の計算が不能でした（B-start または S-head の観測数が 0）。"]

        # B-end の補足
        lines += [
            "",
            f"**B-end (境界末尾) の S{focus_state} 率: {bend_rate:.2f}%**",
            f"（B-end > B-start の場合: S{focus_state} は基末尾の遷移点に集中 → 基終端ハブ状態の可能性）",
            f"（B-end < B-start の場合: 境界位置全体の傾向は先頭側に偏る → 文字種仮説寄り）",
            "",
        ]

    # ── 総括 ─────────────────────────────────────────────────────────────
    lines += [
        "## 総括",
        "",
        "本検証は `compound_boundary_analysis.py` が報告した",
        "「HMM 状態が V8 複合語境界に有意集中する」結果の交絡因子を検証した。",
        "",
        "| k | 判定 | B-start 率 | S-head 率 | Fisher① p |",
        "|---|------|-----------|----------|----------|",
    ]
    for k, result in results_by_k.items():
        if result is None:
            lines.append(f"| {k} | 分析失敗 | — | — | — |")
            continue
        focus_state = result["focus_state"]
        f1 = result["fisher_results"]["①confound検定 (B-start vs S-head)"]
        bstart = result["group_rates"]["B-start"][focus_state]
        shead  = result["group_rates"]["S-head"][focus_state]
        p1     = f1["p"]
        if np.isnan(p1):
            verdict = "計算不能"
        elif p1 > 0.05:
            verdict = "**交絡確認**"
        elif bstart > shead:
            verdict = "**構造効果あり**"
        else:
            verdict = "**交絡否定（逆転）**"
        p_str = f"{p1:.3e}" if not np.isnan(p1) else "N/A"
        lines.append(f"| {k} | {verdict} | {bstart:.2f}% | {shead:.2f}% | {p_str} |")

    lines += [
        "",
        "---",
        "_本レポートは `single_vs_compound_analysis.py` により自動生成。_",
    ]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("Section 6.2 検証: 非複合語（単独ベース語）との比較 開始")

    # ── データロード ──────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    words_all = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
        conn,
    )["word"].tolist()
    conn.close()

    all_types = sorted(set(w for w in words_all if len(w) >= MIN_WORD_LEN))
    log(f"ユニーク単語数: {len(all_types):,}")

    # 語彙構築 (HMM と同じ char2idx)
    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR, EOS_CHAR, PAD_CHAR] + raw_chars
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    log(f"語彙サイズ: {len(all_chars)}")

    # ── V8 複合語の特定 ───────────────────────────────────────────────────
    log("V8複合語を特定中...")
    compound_splits = {}
    for w in all_types:
        splits = find_v8_splits_first(w)
        if splits is not None:
            compound_splits[w] = splits
    log(f"  V8複合語 (2基以上): {len(compound_splits):,} 語")

    # ── 単独ベース語のロード ─────────────────────────────────────────────
    log(f"単独ベース語ロード中: {SINGLE_WORDS_PATH}")
    single_words_raw = SINGLE_WORDS_PATH.read_text(encoding="utf-8").splitlines()
    # char2idx にない文字を持つ語は HMM のシーケンスに変換できないので除外
    single_words = [
        w for w in single_words_raw
        if len(w) >= MIN_WORD_LEN and all(c in char2idx for c in w)
    ]
    log(f"  単独ベース語: {len(single_words):,} 語 (元: {len(single_words_raw):,} 語)")

    # ── HMM モデルごとに分析 ─────────────────────────────────────────────
    results_by_k = {}
    plot_files   = {}

    for k in K_LIST:
        log(f"{'='*50}\n  k = {k}\n{'='*50}")
        phantom_s = PHANTOM_STATE[k]

        info = load_model(k)
        if info is None:
            log(f"  ERROR: モデルキャッシュが見つかりません: {MODEL_CACHE}/full_k{k}.npz")
            results_by_k[k] = None
            continue
        log(f"  モデルロード完了 (logL={info['logL']:.2f})")

        log_start = torch.tensor(np.log(info["start"] + 1e-35), device=DEVICE)
        log_trans  = torch.tensor(np.log(info["trans"]  + 1e-35), device=DEVICE)
        log_emiss  = torch.tensor(np.log(info["emiss"]  + 1e-35), device=DEVICE)

        # Viterbi デコード
        log(f"  Viterbiデコード: 複合語 {len(compound_splits):,} 語...")
        decoded_compound = decode_words(
            compound_splits.keys(), char2idx, log_start, log_trans, log_emiss
        )
        log(f"  完了: {len(decoded_compound):,} 語")

        log(f"  Viterbiデコード: 単独ベース語 {len(single_words):,} 語...")
        decoded_single = decode_words(
            single_words, char2idx, log_start, log_trans, log_emiss
        )
        log(f"  完了: {len(decoded_single):,} 語")

        # 4グループ収集
        groups = collect_groups(
            compound_splits, single_words, decoded_compound, decoded_single
        )
        for g, states in groups.items():
            log(f"    {g}: {len(states):,} 観測")

        # 分析
        result = run_analysis(groups, k, phantom_s)
        if result is None:
            log("  ERROR: 分析失敗（B-start の観測数が 0）")
            results_by_k[k] = None
            continue

        focus_state = result["focus_state"]
        log(f"  Focus State: S{focus_state} (Phantom=S{phantom_s})")
        for g in ["B-start", "B-end", "S-head", "S-mid"]:
            rate = result["group_rates"][g][focus_state]
            log(f"    {g}: {rate:.2f}%")

        for label, fr in result["fisher_results"].items():
            if not np.isnan(fr["p"]):
                log(f"  Fisher {label}: p={fr['p']:.3e},  odds={fr['odds']:.3f}")

        chi2_res = result["chi2"]
        if not np.isnan(chi2_res["chi2"]):
            log(f"  Chi2 (4groups): χ²={chi2_res['chi2']:.2f}, p={chi2_res['p']:.3e}")

        results_by_k[k] = result

        # プロット
        fname = plot_group_rates(result, k, OUT_DIR)
        plot_files[k] = f"results/{fname}"

    # ── Markdown レポート ─────────────────────────────────────────────────
    log("Markdown レポート生成中...")
    md = build_md_report(results_by_k, compound_splits, single_words, plot_files)
    report_path = OUT_DIR / "single_vs_compound_report.md"
    report_path.write_text(md, encoding="utf-8")
    log(f"レポート保存: {report_path}")
    log(f"完了。出力先: {OUT_DIR.resolve()}")
