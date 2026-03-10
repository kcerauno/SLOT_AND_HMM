"""
analysis_BC_boundary_transition.py
=====================================
提案 B：B-end 特化状態の同定
提案 C：境界通過時の状態遷移パターン分析

【提案 B】
  hypothesis/05 で B-start 特化状態（Focus State）を同定したのと同一手法で、
  B-end 特化状態を同定し、B-start Focus State との比較を行う。

  分析内容:
    1. 各状態の B-end / B-start / S-head / S-mid 出現率を集計
    2. B-end 最大状態（B-end Focus State）を phantom 除外で data-driven に同定
    3. B-start Focus State（hypothesis/05 の結果）との対称性を検定
    4. Fisher 正確検定：B-end Focus State が B-start より有意に多いか（逆も）

【提案 C】
  複合語境界（B-end / B-start）位置を中心とした ±2 ウィンドウで
  Viterbi 状態 5-gram を収集し、境界パターンと非境界（単独語語中）を比較する。

  分析内容:
    1. 境界 5-gram (s_{-2},s_{-1},s_0,s_{+1},s_{+2}) 収集
    2. 非境界 5-gram（単独語の語中 positions）収集
    3. カイ二乗検定：各ウィンドウ位置（-2〜+2）での状態分布
    4. 境界特有の上位 5-gram 表示
    5. Bigram vs Trigram の比較

出力:
  results/B_bend_focus_trigram_k{7,8}.png
  results/B_asymmetry_trigram_k{7,8}.png
  results/C_5gram_pos_dist_{bigram,trigram}_k{7,8}.png
  results/C_top5gram_{bigram,trigram}_k{7,8}.png
  results/analysis_B_bend_report.md
  results/analysis_C_transition_report.md

実行:
  cd /home/practi/work_voy
  PYTHONPATH=/home/practi/work_voy/.venv/lib/python3.10/site-packages \\
    python3.10 hypothesis/06_state_characterization/source/analysis_BC_boundary_transition.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
from pathlib import Path
from datetime import datetime
from collections import Counter
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    log, OUT_DIR,
    FOCUS_STATES, PHANTOM_STATES,
    load_trigram_model, load_bigram_model,
    load_compound_splits, load_single_words, load_all_words,
    decode_words, compute_occupancy,
    get_boundary_positions,
)


# ════════════════════════════════════════════════════════════════════════
# 共通ユーティリティ：4グループ状態収集
# ════════════════════════════════════════════════════════════════════════
def collect_groups(compound_splits: dict, single_words: list,
                   decoded_compound: dict, decoded_single: dict) -> dict:
    b_start, b_end, s_head, s_mid = [], [], [], []

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

    for word in single_words:
        states = decoded_single.get(word)
        if states is None or len(states) == 0:
            continue
        L = len(states)
        s_head.append(int(states[0]))
        s_mid.append(int(states[L // 2]))

    return {"B-start": b_start, "B-end": b_end, "S-head": s_head, "S-mid": s_mid}


def find_focus_state(group_key: str, groups: dict, k: int, occupancy: np.ndarray) -> int:
    """指定グループ（B-start or B-end）出現率最大の non-phantom 状態"""
    counts = np.bincount(groups[group_key], minlength=k).astype(float)
    rates  = counts / (counts.sum() + 1e-10)
    for s in range(k):
        if occupancy[s] == 0.0:
            rates[s] = -1.0
    return int(np.argmax(rates))


def fisher_2x2(ga: list, gb: list, state: int, k: int,
               alternative: str = "two-sided") -> dict:
    ca = np.bincount(ga, minlength=k).astype(float)
    cb = np.bincount(gb, minlength=k).astype(float)
    a_t = int(ca[state]); a_o = int(ca.sum()) - a_t
    b_t = int(cb[state]); b_o = int(cb.sum()) - b_t
    if a_t + b_t == 0:
        return {"odds": np.nan, "p": np.nan,
                "a_rate": 0.0, "b_rate": 0.0,
                "a_n": int(ca.sum()), "b_n": int(cb.sum())}
    odds, p = stats.fisher_exact([[a_t, a_o], [b_t, b_o]], alternative=alternative)
    return {
        "odds":   odds,
        "p":      p,
        "a_rate": ca[state] / ca.sum() * 100 if ca.sum() > 0 else 0.0,
        "b_rate": cb[state] / cb.sum() * 100 if cb.sum() > 0 else 0.0,
        "a_n":    int(ca.sum()),
        "b_n":    int(cb.sum()),
    }


# ════════════════════════════════════════════════════════════════════════
# 提案 B：B-end 特化状態の同定
# ════════════════════════════════════════════════════════════════════════
def run_proposal_B(groups: dict, k: int, occupancy: np.ndarray,
                   bstart_fs: int) -> dict:
    """
    B-end Focus State を同定し、B-start Focus State と対比する。
    """
    # 全状態の各グループ出現率
    group_rates = {}
    group_counts = {}
    for name, gl in groups.items():
        cnt = np.bincount(gl, minlength=k).astype(float)
        group_counts[name] = cnt
        tot = cnt.sum()
        group_rates[name] = cnt / tot * 100 if tot > 0 else cnt * 0

    # B-end Focus State（B-end 率最大の non-phantom 状態）
    bend_fs = find_focus_state("B-end", groups, k, occupancy)

    # Fisher 検定群（B-end Focus State について）
    comparisons_bend = {
        "B-end vs S-mid (end特化？)":   fisher_2x2(groups["B-end"],  groups["S-mid"],   bend_fs, k),
        "B-end vs S-head (頭部と比較)": fisher_2x2(groups["B-end"],  groups["S-head"],  bend_fs, k),
        "B-end vs B-start (start非特化)": fisher_2x2(groups["B-end"], groups["B-start"], bend_fs, k),
    }

    # B-start Focus State で B-end の集中度を確認（逆方向）
    comparisons_bstart_bend = {
        "B-start vs B-end (B-start FS の B-end 出現)":
            fisher_2x2(groups["B-start"], groups["B-end"], bstart_fs, k),
    }

    # 全状態の B-end / B-start 率を比較し asymmetry を定量化
    asym = []
    for s in range(k):
        if occupancy[s] == 0.0:
            continue
        be_rate = group_rates["B-end"][s]
        bs_rate = group_rates["B-start"][s]
        asym.append({
            "state":   s,
            "B-end":   be_rate,
            "B-start": bs_rate,
            "diff":    be_rate - bs_rate,  # 正 → B-end 特化, 負 → B-start 特化
        })

    return {
        "bend_fs":               bend_fs,
        "bstart_fs":             bstart_fs,
        "group_rates":           group_rates,
        "group_counts":          group_counts,
        "comparisons_bend":      comparisons_bend,
        "comparisons_bstart_bend": comparisons_bstart_bend,
        "asymmetry":             asym,
    }


def plot_B_focus(res_B: dict, k: int, model_type: str, out_path: Path):
    """B-end / B-start Focus State の各グループ出現率を並べてプロット"""
    bend_fs   = res_B["bend_fs"]
    bstart_fs = res_B["bstart_fs"]
    gr        = res_B["group_rates"]
    order     = ["B-start", "B-end", "S-head", "S-mid"]
    colors    = ["#C0392B", "#E59866", "#2980B9", "#7FB3D3"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, fs, label in [
        (axes[0], bstart_fs, f"B-start Focus: S{bstart_fs}"),
        (axes[1], bend_fs,   f"B-end Focus:   S{bend_fs}"),
    ]:
        rates  = [gr[g][fs] for g in order]
        totals = [int(res_B["group_counts"][g].sum()) for g in order]
        xlabels = [f"{g}\n(n={t:,})" for g, t in zip(order, totals)]
        bars = ax.bar(xlabels, rates, color=colors, alpha=0.88, edgecolor="white")
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2, f"{r:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel(f"S{fs} 出現率 (%)", fontsize=10)
        ax.set_title(f"[{model_type} k={k}]\n{label}", fontsize=10)
        ax.set_ylim(0, max(rates) * 1.45 + 1)

    plt.suptitle(f"[{model_type} k={k}] B-start / B-end Focus State 比較", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_B_asymmetry(res_B: dict, k: int, model_type: str, out_path: Path):
    """全状態の B-end 率 vs B-start 率のスキャッタープロット + 差分棒グラフ"""
    asym = res_B["asymmetry"]
    bend_fs   = res_B["bend_fs"]
    bstart_fs = res_B["bstart_fs"]

    states  = [a["state"]   for a in asym]
    be_rate = [a["B-end"]   for a in asym]
    bs_rate = [a["B-start"] for a in asym]
    diffs   = [a["diff"]    for a in asym]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：散布図
    ax = axes[0]
    for s, be, bs in zip(states, be_rate, bs_rate):
        color = ("#C0392B" if s == bend_fs else
                 "#2980B9" if s == bstart_fs else "#AAB7B8")
        ax.scatter(bs, be, s=90, color=color, zorder=3)
        ax.text(bs + 0.2, be + 0.2, f"S{s}", fontsize=8, color=color)
    lim = max(max(be_rate), max(bs_rate)) * 1.15
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("B-start 出現率 (%)", fontsize=10)
    ax.set_ylabel("B-end 出現率 (%)", fontsize=10)
    ax.set_title(f"[{model_type} k={k}]\nB-end 率 vs B-start 率\n"
                 f"(赤: B-end Focus S{bend_fs}, 青: B-start Focus S{bstart_fs})", fontsize=9)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    # 右：差分棒グラフ
    ax2 = axes[1]
    bar_colors = []
    for s, d in zip(states, diffs):
        if s == bend_fs:
            bar_colors.append("#C0392B")
        elif s == bstart_fs:
            bar_colors.append("#2980B9")
        else:
            bar_colors.append("#AAB7B8" if d >= 0 else "#85C1E9")
    bars = ax2.bar([f"S{s}" for s in states], diffs,
                   color=bar_colors, alpha=0.88, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("状態", fontsize=10)
    ax2.set_ylabel("B-end 率 − B-start 率 (%)", fontsize=10)
    ax2.set_title(f"[{model_type} k={k}]\n非対称性 (正 → B-end 特化, 負 → B-start 特化)", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# 提案 C：境界 5-gram 収集
# ════════════════════════════════════════════════════════════════════════
def collect_5grams(decoded_compound: dict, compound_splits: dict,
                   decoded_single: dict, single_words: list,
                   window: int = 2) -> dict:
    """
    境界位置（B-end / B-start）と非境界（単独語語中）を中心とした
    ウィンドウ状態列を収集する。

    Returns
    -------
    {
      "B-end":  list of (s_{-w}, ..., s_0, ..., s_{+w}) tuples（s_0 が B-end）
      "B-start": 同上
      "neutral": 同上（単独語の語中位置）
    }
    """
    W = window

    grams = {"B-end": [], "B-start": [], "neutral": []}

    # 複合語
    for word, splits in compound_splits.items():
        states = decoded_compound.get(word)
        if states is None:
            continue
        N = len(states)
        bd_end, bd_start = get_boundary_positions(splits)

        for pos in range(N):
            if pos < W or pos + W >= N:
                continue  # エッジを除外

            gram = tuple(states[pos - W: pos + W + 1])

            if pos in bd_end:
                grams["B-end"].append(gram)
            elif pos in bd_start:
                grams["B-start"].append(gram)

    # 単独語（語中位置を neutral として収集）
    for word in single_words:
        states = decoded_single.get(word)
        if states is None:
            continue
        N = len(states)
        for pos in range(W, N - W):
            # 語頭・語末は除外（境界的な位置を避ける）
            if pos == 0 or pos == N - 1:
                continue
            grams["neutral"].append(tuple(states[pos - W: pos + W + 1]))

    return grams


def chi_square_pos(grams_boundary: list, grams_neutral: list, k: int, pos: int) -> dict:
    """
    ウィンドウ内の特定位置 pos の状態分布について
    境界 vs 非境界の chi-square 検定
    """
    bc = np.bincount([g[pos] for g in grams_boundary], minlength=k).astype(float)
    nc = np.bincount([g[pos] for g in grams_neutral],  minlength=k).astype(float)
    valid = (bc + nc) > 0
    if valid.sum() < 2:
        return {"chi2": np.nan, "p": np.nan}
    table = np.vstack([bc[valid], nc[valid]])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"chi2": chi2, "p": p, "dof": dof,
            "n_boundary": int(bc.sum()), "n_neutral": int(nc.sum())}


def plot_C_pos_dist(grams_bend: list, grams_bstart: list, grams_neutral: list,
                    k: int, model_type: str, out_path: Path, window: int = 2):
    """ウィンドウ各位置での状態分布（B-end / B-start / neutral）"""
    W = window
    pos_labels = [f"s_{{t{i:+d}}}" if i != 0 else "s_t (境界)" for i in range(-W, W + 1)]
    n_pos = 2 * W + 1

    fig, axes = plt.subplots(1, n_pos, figsize=(n_pos * 3.5, 5), sharey=False)

    for pi, ax in enumerate(axes):
        bend_dist   = np.bincount([g[pi] for g in grams_bend],   minlength=k).astype(float)
        bstart_dist = np.bincount([g[pi] for g in grams_bstart], minlength=k).astype(float)
        neut_dist   = np.bincount([g[pi] for g in grams_neutral], minlength=k).astype(float)

        bend_dist   /= bend_dist.sum()   + 1e-10
        bstart_dist /= bstart_dist.sum() + 1e-10
        neut_dist   /= neut_dist.sum()   + 1e-10

        x = np.arange(k)
        w = 0.28
        ax.bar(x - w, bend_dist   * 100, w, label="B-end",   color="#4878CF", alpha=0.85)
        ax.bar(x,     bstart_dist * 100, w, label="B-start", color="#F0814E", alpha=0.85)
        ax.bar(x + w, neut_dist   * 100, w, label="neutral", color="#82CA9D", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in range(k)], fontsize=7, rotation=45)
        ax.set_title(pos_labels[pi], fontsize=9)
        ax.set_ylabel("出現率 (%)" if pi == 0 else "", fontsize=8)
        if pi == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"[{model_type} k={k}] 境界 ±{W} ウィンドウ内 状態分布\n"
                 f"(B-end n={len(grams_bend)}, B-start n={len(grams_bstart)}, neutral n={len(grams_neutral)})",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_C_top5grams(grams_bend: list, grams_bstart: list, grams_neutral: list,
                     model_type: str, k: int, out_path: Path, top_n: int = 15):
    """境界上位 5-gram と neutral 上位 5-gram の比較"""
    cnt_bend   = Counter(grams_bend)
    cnt_bstart = Counter(grams_bstart)
    cnt_neut   = Counter(grams_neutral)

    n_bend   = max(len(grams_bend),   1)
    n_bstart = max(len(grams_bstart), 1)
    n_neut   = max(len(grams_neutral), 1)

    # 境界 5-gram（B-end ∪ B-start の上位）
    all_boundary = Counter(grams_bend) + Counter(grams_bstart)
    top_boundary = [g for g, _ in all_boundary.most_common(top_n)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [f"B-end (n={n_bend})", f"B-start (n={n_bstart})", f"neutral (n={n_neut})"]
    cnts   = [cnt_bend, cnt_bstart, cnt_neut]
    norms  = [n_bend, n_bstart, n_neut]
    colors = ["#4878CF", "#F0814E", "#82CA9D"]

    for ax, title, cnt, norm, color in zip(axes, titles, cnts, norms, colors):
        top_grams = [g for g, _ in cnt.most_common(top_n)]
        rates = [cnt[g] / norm * 100 for g in top_grams]
        gram_labels = [str(g).replace(", ", ",") for g in top_grams]
        ax.barh(range(len(top_grams))[::-1], rates[::-1], color=color, alpha=0.85)
        ax.set_yticks(range(len(top_grams)))
        ax.set_yticklabels([f"({','.join(f'S{s}' for s in g)})"
                            for g in top_grams[::-1]], fontsize=7)
        ax.set_xlabel("出現率 (%)", fontsize=9)
        ax.set_title(f"[{model_type} k={k}]\n{title}\n上位 {top_n} 5-gram", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# レポート生成
# ════════════════════════════════════════════════════════════════════════
def build_B_report(results_B: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 提案 B：B-end 特化状態の同定レポート",
        "",
        f"生成日時: {now}",
        "",
        "## 概要",
        "",
        "hypothesis/05 で同定した B-start Focus State と同一手法で、",
        "B-end 出現率が最大の状態（B-end Focus State）を同定し、",
        "B-start/B-end の役割が同一状態か別状態かを検証する。",
        "",
        "---",
        "",
        "## サマリー",
        "",
        "| モデル | k | B-start Focus | B-end Focus | 同一? | B-end率(end_fs) | B-start率(end_fs) |",
        "|--------|---|--------------|------------|------|----------------|-----------------|",
    ]

    for key, res in results_B.items():
        mt, k    = key
        bend_fs  = res["bend_fs"]
        bstart_fs = res["bstart_fs"]
        gr = res["group_rates"]
        same = "**同一**" if bend_fs == bstart_fs else "異なる"
        lines.append(
            f"| {mt} | {k} | S{bstart_fs} | S{bend_fs} | {same} "
            f"| {gr['B-end'][bend_fs]:.2f}% | {gr['B-start'][bend_fs]:.2f}% |"
        )

    lines += ["", "---", ""]

    for key, res in results_B.items():
        mt, k = key
        bend_fs  = res["bend_fs"]
        bstart_fs = res["bstart_fs"]
        gr = res["group_rates"]
        gc = res["group_counts"]
        asym = res["asymmetry"]

        lines += [
            f"## {mt} k={k}",
            "",
            f"- B-start Focus State: **S{bstart_fs}** (B-start率={gr['B-start'][bstart_fs]:.2f}%)",
            f"- B-end Focus State:   **S{bend_fs}**   (B-end率={gr['B-end'][bend_fs]:.2f}%)",
            "",
            "### 各状態の B-end / B-start 出現率",
            "",
            "| 状態 | B-end (%) | B-start (%) | 差 (end−start) | 判定 |",
            "|-----|----------|-----------|--------------|------|",
        ]
        for a in asym:
            s = a["state"]
            mark = ("★B-end Focus" if s == bend_fs else
                    ("☆B-start Focus" if s == bstart_fs else ""))
            lines.append(
                f"| S{s} | {a['B-end']:.2f}% | {a['B-start']:.2f}% "
                f"| {a['diff']:+.2f}% | {mark} |"
            )

        lines += ["", "### B-end Focus State の Fisher 検定", ""]
        for label, fr in res["comparisons_bend"].items():
            p_str = f"{fr['p']:.2e}" if not np.isnan(fr["p"]) else "N/A"
            lines.append(f"- **{label}**: 率A={fr['a_rate']:.2f}%, 率B={fr['b_rate']:.2f}%, p={p_str}")

        lines += ["", "### B-start Focus State での B-end 集中度（逆方向確認）", ""]
        for label, fr in res["comparisons_bstart_bend"].items():
            p_str = f"{fr['p']:.2e}" if not np.isnan(fr["p"]) else "N/A"
            lines.append(f"- **{label}**: 率A={fr['a_rate']:.2f}%, 率B={fr['b_rate']:.2f}%, p={p_str}")

        lines += ["", "### 解釈", ""]
        if bend_fs == bstart_fs:
            lines.append(
                f"**【同一状態】** B-end / B-start ともに S{bend_fs} → "
                "境界前後を同一の HMM 状態が担当。境界をまたぐ連続した役割。"
            )
        else:
            be_in_bs = gr["B-end"][bstart_fs]
            bs_in_be = gr["B-start"][bend_fs]
            lines.append(
                f"**【異なる状態】** B-end=S{bend_fs}, B-start=S{bstart_fs} → "
                "境界末尾と境界先頭が別状態で表現されている（状態遷移による境界マーキング）。"
            )
            lines.append(
                f"  - B-start Focus S{bstart_fs} での B-end 率: {be_in_bs:.2f}%"
            )
            lines.append(
                f"  - B-end Focus S{bend_fs} での B-start 率: {bs_in_be:.2f}%"
            )

        lines += ["", "---", ""]

    lines += ["_本レポートは analysis_BC_boundary_transition.py により自動生成。_"]
    return "\n".join(lines)


def build_C_report(results_C: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 提案 C：境界通過時の状態遷移パターン分析レポート",
        "",
        f"生成日時: {now}",
        "",
        "## 概要",
        "",
        "複合語の境界位置（B-end / B-start）を中心とした ±2 ウィンドウで",
        "Viterbi 状態 5-gram を収集し、中立位置（単独語語中）と比較する。",
        "各ウィンドウ位置でのカイ二乗検定により、境界の前後どちらで状態分布が変化するかを定量化。",
        "",
        "---",
        "",
        "## サマリー",
        "",
        "| モデル | k | B-end n | B-start n | neutral n | 最大 chi2 位置 | p |",
        "|--------|---|--------|----------|----------|-------------|---|",
    ]

    W = 2
    pos_labels = [f"t{i:+d}" for i in range(-W, W + 1)]

    for key, res in results_C.items():
        mt, k = key
        grams = res["grams"]
        chi2_bend  = res["chi2_bend"]
        chi2_bstart = res["chi2_bstart"]
        n_bend  = len(grams["B-end"])
        n_bstart = len(grams["B-start"])
        n_neut  = len(grams["neutral"])

        # 最大 chi2 位置（B-end を基準）
        best_pi  = max(range(2 * W + 1),
                       key=lambda i: chi2_bend[i].get("chi2") or 0.0)
        best_chi2 = chi2_bend[best_pi].get("chi2", np.nan)
        best_p    = chi2_bend[best_pi].get("p", np.nan)
        lines.append(
            f"| {mt} | {k} | {n_bend} | {n_bstart} | {n_neut} "
            f"| {pos_labels[best_pi]} | {best_p:.2e} |"
        )

    lines += ["", "---", ""]

    for key, res in results_C.items():
        mt, k = key
        grams    = res["grams"]
        chi2_bend  = res["chi2_bend"]
        chi2_bstart = res["chi2_bstart"]

        lines += [
            f"## {mt} k={k}",
            "",
            "### ウィンドウ各位置の chi-square 検定（境界 vs neutral）",
            "",
            "| 位置 | B-end chi2 | B-end p | B-start chi2 | B-start p | 判定 |",
            "|-----|-----------|--------|------------|----------|------|",
        ]

        for pi, label in enumerate(pos_labels):
            cb = chi2_bend[pi]
            cs = chi2_bstart[pi]
            chi2_b = cb.get("chi2", np.nan)
            p_b    = cb.get("p",    np.nan)
            chi2_s = cs.get("chi2", np.nan)
            p_s    = cs.get("p",    np.nan)
            sig = ("**有意**" if (not np.isnan(p_b) and p_b < 0.05) or
                               (not np.isnan(p_s) and p_s < 0.05) else "")
            lines.append(
                f"| {label} | {chi2_b:.2f} | {p_b:.2e} "
                f"| {chi2_s:.2f} | {p_s:.2e} | {sig} |"
            )

        # 最も特徴的な 5-gram トップ 5
        cnt_bend  = Counter(grams["B-end"])
        cnt_neut  = Counter(grams["neutral"])
        n_b = max(len(grams["B-end"]), 1)
        n_n = max(len(grams["neutral"]), 1)

        lines += ["", "### B-end 上位 5 5-gram（B-end 率 / neutral 率）", ""]
        for gram, cnt in cnt_bend.most_common(5):
            rate_b = cnt / n_b * 100
            rate_n = cnt_neut.get(gram, 0) / n_n * 100
            gram_str = "(" + ", ".join(f"S{s}" for s in gram) + ")"
            lines.append(f"- {gram_str}: B-end={rate_b:.2f}%, neutral={rate_n:.2f}%")

        cnt_bstart = Counter(grams["B-start"])
        n_bs = max(len(grams["B-start"]), 1)
        lines += ["", "### B-start 上位 5 5-gram（B-start 率 / neutral 率）", ""]
        for gram, cnt in cnt_bstart.most_common(5):
            rate_bs = cnt / n_bs * 100
            rate_n  = cnt_neut.get(gram, 0) / n_n * 100
            gram_str = "(" + ", ".join(f"S{s}" for s in gram) + ")"
            lines.append(f"- {gram_str}: B-start={rate_bs:.2f}%, neutral={rate_n:.2f}%")

        # 解釈
        sig_bend_pos = [pos_labels[pi] for pi in range(2 * W + 1)
                        if not np.isnan(chi2_bend[pi].get("p", np.nan))
                        and chi2_bend[pi]["p"] < 0.05]
        sig_bstart_pos = [pos_labels[pi] for pi in range(2 * W + 1)
                          if not np.isnan(chi2_bstart[pi].get("p", np.nan))
                          and chi2_bstart[pi]["p"] < 0.05]
        lines += ["", "### 解釈", ""]
        lines.append(f"- B-end 有意位置: {sig_bend_pos if sig_bend_pos else 'なし'}")
        lines.append(f"- B-start 有意位置: {sig_bstart_pos if sig_bstart_pos else 'なし'}")

        both_sig = set(sig_bend_pos) & set(sig_bstart_pos)
        only_bend = set(sig_bend_pos) - set(sig_bstart_pos)
        only_bstart = set(sig_bstart_pos) - set(sig_bend_pos)

        if both_sig:
            lines.append(f"- B-end/B-start 共通有意位置: {sorted(both_sig)} → 境界全体に跨る遷移変化")
        if only_bend:
            lines.append(f"- B-end のみ有意: {sorted(only_bend)} → 基の終端直前・直後で状態が変化")
        if only_bstart:
            lines.append(f"- B-start のみ有意: {sorted(only_bstart)} → 基の開始直前・直後で状態が変化")

        lines += ["", "---", ""]

    lines += ["_本レポートは analysis_BC_boundary_transition.py により自動生成。_"]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("提案 B・C：境界状態分析 開始")

    compound_splits = load_compound_splits()
    single_words    = load_single_words()
    all_words       = load_all_words()

    log(f"複合語: {len(compound_splits)}, 単独語: {len(single_words)}, 全語: {len(all_words)}")

    results_B: dict = {}
    results_C: dict = {}

    models_to_run = [
        ("trigram", load_trigram_model, [7, 8]),
        ("bigram",  load_bigram_model,  [7, 8]),
    ]

    for model_type, load_fn, k_list in models_to_run:
        for k in k_list:
            log(f"\n{'─'*60}")
            log(f"{model_type} k={k}")
            log(f"{'─'*60}")

            model = load_fn(k)
            if model is None:
                continue
            log(f"  logL = {model['logL']:.2f}")

            # Viterbi 占有率（phantom 同定）
            log("  Viterbi 占有率を計算中...")
            occupancy = compute_occupancy(all_words, model, model_type)
            phantoms  = [s for s in range(k) if occupancy[s] == 0.0]
            log(f"  Phantom: {[f'S{s}' for s in phantoms]}")

            # Viterbi デコード
            log(f"  複合語デコード ({len(compound_splits)} 語)...")
            decoded_compound = decode_words(compound_splits.keys(), model, model_type)
            log(f"  単独語デコード ({len(single_words)} 語)...")
            decoded_single   = decode_words(single_words, model, model_type)

            # 4グループ収集
            groups = collect_groups(compound_splits, single_words,
                                    decoded_compound, decoded_single)
            for g, lst in groups.items():
                log(f"    {g}: {len(lst)} 観測")

            bstart_fs = FOCUS_STATES.get((model_type, k),
                                         find_focus_state("B-start", groups, k, occupancy))

            # ── 提案 B ──────────────────────────────────────────────
            res_B = run_proposal_B(groups, k, occupancy, bstart_fs)
            log(f"  B-end Focus State: S{res_B['bend_fs']}")
            log(f"  B-start Focus State: S{res_B['bstart_fs']}")
            log(f"  同一: {res_B['bend_fs'] == res_B['bstart_fs']}")

            results_B[(model_type, k)] = res_B

            out_b1 = OUT_DIR / f"B_bend_focus_{model_type}_k{k}.png"
            plot_B_focus(res_B, k, model_type, out_b1)
            log(f"  Saved: {out_b1.name}")

            out_b2 = OUT_DIR / f"B_asymmetry_{model_type}_k{k}.png"
            plot_B_asymmetry(res_B, k, model_type, out_b2)
            log(f"  Saved: {out_b2.name}")

            # ── 提案 C ──────────────────────────────────────────────
            log("  5-gram 収集中...")
            grams = collect_5grams(decoded_compound, compound_splits,
                                   decoded_single, single_words)
            log(f"  B-end={len(grams['B-end'])}, B-start={len(grams['B-start'])}, "
                f"neutral={len(grams['neutral'])}")

            W = 2
            chi2_bend   = [chi_square_pos(grams["B-end"],   grams["neutral"], k, pi)
                           for pi in range(2 * W + 1)]
            chi2_bstart = [chi_square_pos(grams["B-start"], grams["neutral"], k, pi)
                           for pi in range(2 * W + 1)]

            results_C[(model_type, k)] = {
                "grams":        grams,
                "chi2_bend":    chi2_bend,
                "chi2_bstart":  chi2_bstart,
            }

            out_c1 = OUT_DIR / f"C_5gram_pos_dist_{model_type}_k{k}.png"
            plot_C_pos_dist(grams["B-end"], grams["B-start"], grams["neutral"],
                            k, model_type, out_c1)
            log(f"  Saved: {out_c1.name}")

            out_c2 = OUT_DIR / f"C_top5gram_{model_type}_k{k}.png"
            plot_C_top5grams(grams["B-end"], grams["B-start"], grams["neutral"],
                             model_type, k, out_c2)
            log(f"  Saved: {out_c2.name}")

    # レポート生成
    report_B = build_B_report(results_B)
    path_B = OUT_DIR / "analysis_B_bend_report.md"
    path_B.write_text(report_B, encoding="utf-8")
    log(f"\nレポート保存: {path_B}")

    report_C = build_C_report(results_C)
    path_C = OUT_DIR / "analysis_C_transition_report.md"
    path_C.write_text(report_C, encoding="utf-8")
    log(f"レポート保存: {path_C}")

    log("✓ 提案 B・C 完了。出力先: " + str(OUT_DIR.resolve()))
