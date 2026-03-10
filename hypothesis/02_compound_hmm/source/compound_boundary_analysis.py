"""
compound_boundary_analysis.py
==============================
V8複合語仮説 × HMM Phantom State 境界検証

検証仮説:
  HMMの縮退状態（Phantom State）は BOS/EOS 位置専用のハブではなく、
  V8文法が独立に発見したベース間の境界位置に特化した遷移ハブとして機能している。

もしPhantom State が V8複合境界位置に統計的に集中するなら、
2つの独立手法が同じ構造を「発見」したことになり、複合構造の存在に対する
強力な相互検証となる。

Phantom State: k=7 → S3、k=8 → S4 (占有率0%が確認済み)
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import font_manager
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


def _setup_jp_font():
    candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic", "Noto Sans CJK JP"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_setup_jp_font()

# ── 設定 ──────────────────────────────────────────────────────────────
DB_PATH     = "data/voynich.db"
MODEL_CACHE = Path("hypothesis/01_bigram/results/hmm_model_cache")
OUT_DIR     = Path("hypothesis/02_compound_hmm/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST         = [7, 8]
PHANTOM_STATE  = {7: 3, 8: 4}   # k=7: S3、k=8: S4
N_SHUFFLE      = 1000            # Null分布生成のシャッフル回数
MIN_WORD_LEN   = 2
TOP_WORDS_PLOT = 30              # 状態パスヒートマップの表示語数

BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"


# ════════════════════════════════════════════════════════════════════════
# 1. V8文法定義 (analyze_slot_grammar_v8.py から移植)
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


# ── V8複合語分割 ──────────────────────────────────────────────────────
def find_v8_splits_first(word: str):
    """
    is_v8 と同一の探索順で最初の有効分割を返す (2基以上のみ)。
    1基マッチの場合は None を返す（単独ベース語は複合語ではない）。
    V8複合語でない場合も None。
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
            for k in range(1, len(rest2)):
                if is_base(rest2[:k]) and is_base(rest2[k:]):
                    return (p1, p2, rest2[:k], rest2[k:])
    return None


def get_boundary_positions(splits: tuple) -> tuple:
    """
    splits = (base1, base2, ..., baseN) から境界位置を計算。

    Returns
    -------
    boundary_end   : set of char positions that are the LAST char of a non-final base
    boundary_start : set of char positions that are the FIRST char of a non-first base
    """
    boundary_end   = set()
    boundary_start = set()
    cumlen = 0
    for base in splits[:-1]:
        cumlen += len(base)
        boundary_end.add(cumlen - 1)   # 最終文字 of base_i
        boundary_start.add(cumlen)     # 先頭文字 of base_{i+1}
    return boundary_end, boundary_start


# ════════════════════════════════════════════════════════════════════════
# 2. HMMモデル
# ════════════════════════════════════════════════════════════════════════
def load_model(k):
    path = MODEL_CACHE / f"full_k{k}.npz"
    if path.exists():
        d = np.load(path)
        return {
            "start": d["start"], "trans": d["trans"],
            "emiss": d["emiss"], "logL": float(d["logL"][0])
        }
    return None


# ── Viterbi (slot_hmm_state_analysis.py から移植) ──────────────────────
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


# ════════════════════════════════════════════════════════════════════════
# 3. 境界位置分析
# ════════════════════════════════════════════════════════════════════════
def analyze_boundaries(info, compound_splits, char2idx, k):
    """
    各複合語を Viterbi デコードし、境界/非境界位置の状態を収集。

    Parameters
    ----------
    compound_splits : dict[word -> tuple of bases]

    Returns
    -------
    dict with keys:
      "boundary_end"   : list of states at end-of-base positions
      "boundary_start" : list of states at start-of-base positions
      "boundary_both"  : list of states at any seam position
      "non_boundary"   : list of states at non-seam positions
      "word_paths"     : dict[word -> (states, splits, boundary_end_set, boundary_start_set)]
    """
    log_start = torch.tensor(np.log(info["start"] + 1e-35), device=DEVICE)
    log_trans  = torch.tensor(np.log(info["trans"]  + 1e-35), device=DEVICE)
    log_emiss  = torch.tensor(np.log(info["emiss"]  + 1e-35), device=DEVICE)

    bd_end_states   = []
    bd_start_states = []
    bd_both_states  = []
    non_bd_states   = []
    word_paths      = {}

    for word, splits in compound_splits.items():
        seq = np.array(
            [char2idx[BOS_CHAR]] + [char2idx[c] for c in word] + [char2idx[EOS_CHAR]],
            dtype=np.int32,
        )
        try:
            path = viterbi_pt(log_start, log_trans, log_emiss, seq)
        except Exception:
            continue

        states = path[1:-1]   # BOS/EOS を除く
        if len(states) == 0:
            continue

        bd_end, bd_start = get_boundary_positions(splits)
        bd_both = bd_end | bd_start

        for pos, state in enumerate(states):
            in_end   = pos in bd_end
            in_start = pos in bd_start
            in_both  = pos in bd_both
            if in_end:
                bd_end_states.append(state)
            if in_start:
                bd_start_states.append(state)
            if in_both:
                bd_both_states.append(state)
            else:
                non_bd_states.append(state)

        word_paths[word] = (states, splits, bd_end, bd_start)

    return {
        "boundary_end":   bd_end_states,
        "boundary_start": bd_start_states,
        "boundary_both":  bd_both_states,
        "non_boundary":   non_bd_states,
        "word_paths":     word_paths,
    }


# ════════════════════════════════════════════════════════════════════════
# 4. 統計検定
# ════════════════════════════════════════════════════════════════════════
def statistical_tests(boundary_states, non_boundary_states, k, phantom_s, rng):
    """
    境界位置 vs 非境界位置の状態分布を統計的に検定。

    Returns
    -------
    dict with test results
    """
    b_counts  = np.bincount(boundary_states,     minlength=k).astype(float)
    nb_counts = np.bincount(non_boundary_states, minlength=k).astype(float)
    b_total   = b_counts.sum()
    nb_total  = nb_counts.sum()

    # カイ二乗検定（全状態の分布比較）
    contingency = np.array([b_counts, nb_counts])
    # 期待値が0の列（Phantom Stateが全く出現しない場合）を除外
    col_ok = (b_counts + nb_counts) > 0
    chi2_val, chi2_p, chi2_dof, _ = stats.chi2_contingency(
        contingency[:, col_ok]
    )

    # Phantom State の enrichment
    b_phantom  = b_counts[phantom_s]
    nb_phantom = nb_counts[phantom_s]
    b_other    = b_total  - b_phantom
    nb_other   = nb_total - nb_phantom

    # Fisher 正確検定（境界にPhantomが多いか）
    table = [[int(b_phantom), int(b_other)],
             [int(nb_phantom), int(nb_other)]]
    if b_phantom + nb_phantom > 0:
        odds_ratio, fisher_p = stats.fisher_exact(table, alternative="greater")
    else:
        odds_ratio, fisher_p = np.nan, np.nan

    # enrichment ratio
    b_rate  = b_phantom  / b_total  if b_total  > 0 else 0.0
    nb_rate = nb_phantom / nb_total if nb_total > 0 else 0.0
    enrich  = b_rate / nb_rate if nb_rate > 0 else np.nan

    # Null分布（シャッフル検定）
    all_states = np.array(boundary_states + non_boundary_states)
    n_bd = len(boundary_states)
    null_enrichments = []
    for _ in range(N_SHUFFLE):
        shuffled = rng.permutation(all_states)
        s_bd  = shuffled[:n_bd]
        s_nb  = shuffled[n_bd:]
        s_bc  = np.bincount(s_bd, minlength=k).astype(float)
        s_nbc = np.bincount(s_nb, minlength=k).astype(float)
        s_br  = s_bc[phantom_s]  / s_bd.size  if s_bd.size  > 0 else 0.0
        s_nbr = s_nbc[phantom_s] / s_nb.size  if s_nb.size  > 0 else 0.0
        null_enrichments.append(s_br / s_nbr if s_nbr > 0 else np.nan)

    null_arr  = np.array([v for v in null_enrichments if not np.isnan(v)])
    pctile    = np.mean(null_arr <= enrich) * 100 if len(null_arr) > 0 else np.nan

    return {
        "b_counts":        b_counts,
        "nb_counts":       nb_counts,
        "b_total":         b_total,
        "nb_total":        nb_total,
        "chi2":            chi2_val,
        "chi2_p":          chi2_p,
        "chi2_dof":        chi2_dof,
        "b_phantom":       b_phantom,
        "nb_phantom":      nb_phantom,
        "b_rate":          b_rate,
        "nb_rate":         nb_rate,
        "enrichment":      enrich,
        "odds_ratio":      odds_ratio,
        "fisher_p":        fisher_p,
        "null_enrichments": null_arr,
        "pctile":          pctile,
    }


# ════════════════════════════════════════════════════════════════════════
# 5. 可視化
# ════════════════════════════════════════════════════════════════════════
def plot_state_distribution(test, k, phantom_s, out_dir):
    """図1: 境界vs非境界 状態分布バーチャート"""
    b_counts  = test["b_counts"]
    nb_counts = test["nb_counts"]
    b_total   = test["b_total"]
    nb_total  = test["nb_total"]

    b_rates  = b_counts  / b_total  * 100 if b_total  > 0 else b_counts * 0
    nb_rates = nb_counts / nb_total * 100 if nb_total > 0 else nb_counts * 0

    x = np.arange(k)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, k * 1.2), 5))
    bars_b  = ax.bar(x - width / 2, b_rates,  width, label="境界位置 (seam)", color="#4878CF", alpha=0.85)
    bars_nb = ax.bar(x + width / 2, nb_rates, width, label="非境界位置",       color="#F0814E", alpha=0.85)

    # Phantom State をハイライト
    for bar in [bars_b[phantom_s], bars_nb[phantom_s]]:
        bar.set_edgecolor("red")
        bar.set_linewidth(2.5)

    # 数値ラベル
    for bar in list(bars_b) + list(bars_nb):
        h = bar.get_height()
        if h > 0.1:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" + (" ★" if i == phantom_s else "") for i in range(k)], fontsize=9)
    ax.set_xlabel("HMM 状態", fontsize=10)
    ax.set_ylabel("出現割合 (%)", fontsize=10)
    ax.set_title(
        f"V8複合境界 vs 非境界 での状態分布  k={k}\n"
        f"★=Phantom State (S{phantom_s})  "
        f"enrichment={test['enrichment']:.2f}x  Fisher p={test['fisher_p']:.2e}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = out_dir / f"boundary_state_dist_k{k}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path.name}")


def plot_phantom_enrichment(test, k, phantom_s, out_dir):
    """図2: Phantom State Null分布 vs 実測値"""
    null_arr = test["null_enrichments"]
    enrich   = test["enrichment"]
    pctile   = test["pctile"]

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(null_arr) > 0:
        ax.hist(null_arr, bins=40, color="steelblue", alpha=0.7, label=f"Null分布 ({N_SHUFFLE}回シャッフル)")
    ax.axvline(enrich, color="red", linewidth=2.5,
               label=f"実測値 {enrich:.3f}x\n(上位{100 - pctile:.1f}%ile, p≈{test['fisher_p']:.2e})")
    ax.set_xlabel("Phantom State enrichment ratio (境界/非境界)", fontsize=10)
    ax.set_ylabel("頻度", fontsize=10)
    ax.set_title(f"Phantom State (S{phantom_s}) 境界集中度  k={k}", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = out_dir / f"phantom_enrichment_k{k}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path.name}")


def plot_word_paths(analysis, k, phantom_s, out_dir, top_n=TOP_WORDS_PLOT):
    """図3: 複合語 Viterbi 状態パス ヒートマップ（境界位置マーク付き）"""
    word_paths = analysis["word_paths"]

    # 境界の多い単語 / Phantom出現単語を優先して表示
    def sort_key(item):
        word, (states, splits, bd_end, bd_start) = item
        n_bd = len(bd_end | bd_start)
        n_phantom = np.sum(states == phantom_s)
        return -(n_phantom * 10 + n_bd)

    sorted_words = sorted(word_paths.items(), key=sort_key)[:top_n]
    if not sorted_words:
        return

    max_len = max(len(states) for _, (states, _, _, _) in sorted_words)
    mat  = np.full((len(sorted_words), max_len), -1, dtype=int)
    seam = np.zeros((len(sorted_words), max_len), dtype=bool)

    ylabels = []
    for row, (word, (states, splits, bd_end, bd_start)) in enumerate(sorted_words):
        L = len(states)
        mat[row, :L] = states
        for pos in (bd_end | bd_start):
            if pos < L:
                seam[row, pos] = True
        ylabels.append(word)

    fig, ax = plt.subplots(figsize=(max(10, max_len * 0.6), max(6, len(sorted_words) * 0.35)))
    cmap = plt.cm.get_cmap("tab10", k + 1)
    cmap.set_under("white")

    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=k - 1,
                   interpolation="nearest")

    # seam位置に白線
    for row in range(len(sorted_words)):
        for col in range(max_len):
            if seam[row, col]:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                           fill=False, edgecolor="white", linewidth=2))

    # Phantom State セルに × マーク
    for row in range(len(sorted_words)):
        for col in range(max_len):
            if mat[row, col] == phantom_s:
                ax.text(col, row, "×", ha="center", va="center",
                        fontsize=8, color="red", fontweight="bold")

    ax.set_yticks(range(len(sorted_words)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("文字位置 (BOS/EOS 除く)", fontsize=9)
    ax.set_title(
        f"複合語 Viterbi 状態パス  k={k}\n"
        f"■白枠=V8境界位置  ×=Phantom State(S{phantom_s})",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label=f"HMM状態 (0〜{k-1})", shrink=0.6)
    plt.tight_layout()
    path = out_dir / f"word_paths_k{k}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path.name}")


# ════════════════════════════════════════════════════════════════════════
# 6. テキストレポート
# ════════════════════════════════════════════════════════════════════════
def build_report(analysis, test, k, phantom_s, compound_splits):
    b_counts  = test["b_counts"]
    nb_counts = test["nb_counts"]
    b_total   = int(test["b_total"])
    nb_total  = int(test["nb_total"])
    n_compounds = len(analysis["word_paths"])
    n_bd_end   = len(analysis["boundary_end"])
    n_bd_start = len(analysis["boundary_start"])
    n_bd_both  = len(analysis["boundary_both"])
    n_non      = len(analysis["non_boundary"])

    # 境界での各状態出現率
    b_rates  = (b_counts  / b_total  * 100) if b_total  > 0 else b_counts * 0
    nb_rates = (nb_counts / nb_total * 100) if nb_total > 0 else nb_counts * 0
    diff     = b_rates - nb_rates

    # 上位差分状態
    top_idx = np.argsort(-diff)

    lines = [
        "=" * 72,
        f"V8複合語仮説 × HMM Phantom State 境界検証  k={k}",
        f"  生成日時: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 72,
        "",
        f"  対象複合語数 (V8文法 2基以上): {n_compounds:,} 語",
        f"  境界位置文字数 (end+start重複有): {n_bd_both:,}",
        f"    うち end-of-base:    {n_bd_end:,}",
        f"    うち start-of-base:  {n_bd_start:,}",
        f"  非境界位置文字数:               {n_non:,}",
        "",
        "─" * 72,
        f"  Phantom State (S{phantom_s}) 境界集中分析",
        "─" * 72,
        f"  境界位置での出現率:     {test['b_rate']*100:6.3f}%  ({int(test['b_phantom']):,} / {b_total:,})",
        f"  非境界位置での出現率:   {test['nb_rate']*100:6.3f}%  ({int(test['nb_phantom']):,} / {nb_total:,})",
        f"  enrichment ratio:       {test['enrichment']:.4f}x",
        f"  odds ratio (Fisher):    {test['odds_ratio']:.4f}",
        f"  Fisher exact p-value:   {test['fisher_p']:.3e}  (alternative: greater)",
        f"  Null分布パーセンタイル:  {test['pctile']:.1f}%ile  ({N_SHUFFLE}回シャッフル)",
        "",
        "─" * 72,
        "  全状態 カイ二乗検定（境界 vs 非境界 分布）",
        "─" * 72,
        f"  chi2 = {test['chi2']:.2f},  dof = {test['chi2_dof']},  p = {test['chi2_p']:.3e}",
        "",
        "─" * 72,
        "  各状態の境界出現率 vs 非境界出現率（差分降順）",
        "─" * 72,
        f"  {'状態':>5}  {'境界(%)':>9}  {'非境界(%)':>10}  {'差分':>8}  {'境界N':>7}  {'非境界N':>8}",
        f"  {'─'*5}  {'─'*9}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*8}",
    ]
    for s in top_idx:
        marker = " ★Phantom" if s == phantom_s else ""
        lines.append(
            f"  S{s:>4d}  {b_rates[s]:>8.3f}%  {nb_rates[s]:>9.3f}%  {diff[s]:>+7.3f}%"
            f"  {int(b_counts[s]):>7,}  {int(nb_counts[s]):>8,}{marker}"
        )

    lines += [
        "",
        "─" * 72,
        "  解釈",
        "─" * 72,
    ]
    enrich = test["enrichment"]
    fp     = test["fisher_p"]
    if not np.isnan(enrich):
        if enrich > 2.0 and fp < 0.01:
            verdict = "【強い支持】Phantom Stateはベース間境界位置に有意に集中している。"
        elif enrich > 1.2 and fp < 0.05:
            verdict = "【弱い支持】Phantom Stateはやや境界に集中しているが効果量は限定的。"
        elif enrich < 0.8 and fp < 0.05:
            verdict = "【棄却】Phantom Stateは境界を回避する傾向がある（逆の結果）。"
        elif int(test["b_phantom"]) == 0 and int(test["nb_phantom"]) == 0:
            verdict = "【データ不足】Phantom StateはViterbiパスに一切現れなかった（真の縮退状態）。"
        else:
            verdict = f"【中立】境界集中は統計的に有意でない (enrichment={enrich:.2f}x, p={fp:.3e})。"
    else:
        verdict = "【計算不能】Phantom StateがViterbiパスに出現しなかった。"

    lines += [
        f"  {verdict}",
        "",
        "─" * 72,
        "  最大境界集中状態（Phantom State 以外）",
        "─" * 72,
    ]
    # Phantom State 以外で最大の差分を持つ状態
    diff_excl_phantom = diff.copy()
    diff_excl_phantom[phantom_s] = -np.inf
    best_s = int(np.argmax(diff_excl_phantom))
    best_diff = diff[best_s]
    lines += [
        f"  S{best_s}: 境界率={b_rates[best_s]:.3f}%  非境界率={nb_rates[best_s]:.3f}%  差分={best_diff:+.3f}%",
        f"  境界N={int(b_counts[best_s]):,}  非境界N={int(nb_counts[best_s]):,}",
        "",
    ]
    if best_diff > 10.0:
        lines += [
            f"  !! S{best_s} が V8複合語境界に強く集中しています (Δ={best_diff:+.1f}%)。",
            f"  !! Phantom State S{phantom_s} が Viterbi に出現しない場合でも、",
            f"  !! S{best_s} が「境界遷移ハブ」として機能している可能性があります。",
        ]

    lines += [
        "",
        "  V8仮説との相互検証:",
    ]
    if not np.isnan(enrich) and enrich > 1.5 and fp < 0.05:
        lines += [
            "  → V8文法（独立）とHMM（独立）が同じ「複合語境界」を発見。",
            "  → ヴォイニッチ語における複合構造の存在に対する強力な証拠。",
        ]
    elif best_diff > 10.0:
        lines += [
            f"  → Phantom State S{phantom_s} は境界に出現しないが、S{best_s} が境界集中を示す。",
            "  → V8複合境界はHMMの特定状態遷移パターンと相関している可能性がある。",
            "  → 追加検証: S{best_s} の放出分布とV8境界文字の対応を調査することを推奨。".format(best_s=best_s),
        ]
    else:
        lines += [
            "  → 現時点では V8境界とHMM Phantom State の一致は確認されなかった。",
            "  → 境界の定義（end/start）や k の選択を変えて追加検証が必要。",
        ]

    lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("V8複合語仮説 × HMM Phantom State 境界検証 開始")
    rng = np.random.default_rng(42)

    # ── データロード ──────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    words_all = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
        conn,
    )["word"].tolist()
    conn.close()

    all_types = sorted(set(words_all))
    all_types = [w for w in all_types if len(w) >= MIN_WORD_LEN]
    log(f"ユニーク単語数: {len(all_types):,}")

    # 語彙構築（HMMのchar2idxと合わせる）
    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR, EOS_CHAR, PAD_CHAR] + raw_chars
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    log(f"語彙サイズ: {len(all_chars)}")

    # ── V8複合語の特定 ────────────────────────────────────────────────
    log("V8複合語を特定中...")
    compound_splits = {}   # word -> tuple of bases
    n_single = 0
    n_unmatched = 0
    for w in all_types:
        splits = find_v8_splits_first(w)
        if splits is None:
            # is_base(w) かチェックして単独ベース語か未マッチか判別
            if is_base(w):
                n_single += 1
            else:
                n_unmatched += 1
        else:
            compound_splits[w] = splits

    log(f"  V8複合語 (2基以上): {len(compound_splits):,} 語")
    log(f"  V8単独ベース語:     {n_single:,} 語")
    log(f"  V8未マッチ語:       {n_unmatched:,} 語")

    # 分割数の内訳
    from collections import Counter
    split_counts = Counter(len(v) for v in compound_splits.values())
    for nb, cnt in sorted(split_counts.items()):
        log(f"    [{nb}基]: {cnt:,} 語")

    all_reports = []

    for k in K_LIST:
        log(f"{'='*50}\n  k = {k}\n{'='*50}")
        phantom_s = PHANTOM_STATE[k]

        # ── モデルロード ──────────────────────────────────────────────
        info = load_model(k)
        if info is None:
            log(f"  ERROR: キャッシュが見つかりません: {MODEL_CACHE}/full_k{k}.npz")
            log("  先に slot_hmm_state_analysis.py を実行してください。")
            continue
        log(f"  モデルロード完了 (logL={info['logL']:.2f})")

        # ── 境界位置分析 ──────────────────────────────────────────────
        log(f"  Viterbi デコード + 境界分析中 ({len(compound_splits):,} 複合語)...")
        analysis = analyze_boundaries(info, compound_splits, char2idx, k)
        log(f"  デコード完了: {len(analysis['word_paths']):,} 語")
        log(f"  境界位置:  {len(analysis['boundary_both']):,} 件")
        log(f"  非境界位置: {len(analysis['non_boundary']):,} 件")

        # ── 統計検定 ──────────────────────────────────────────────────
        log(f"  統計検定中 (Phantom=S{phantom_s}, shuffle={N_SHUFFLE}回)...")
        test = statistical_tests(
            analysis["boundary_both"], analysis["non_boundary"],
            k, phantom_s, rng
        )
        log(f"  Phantom S{phantom_s}: enrichment={test['enrichment']:.3f}x, "
            f"Fisher p={test['fisher_p']:.3e}")

        # ── レポート ──────────────────────────────────────────────────
        report = build_report(analysis, test, k, phantom_s, compound_splits)
        all_reports.append(report)
        print(report)

        # ── 可視化 ────────────────────────────────────────────────────
        plot_state_distribution(test, k, phantom_s, OUT_DIR)
        plot_phantom_enrichment(test, k, phantom_s, OUT_DIR)
        plot_word_paths(analysis, k, phantom_s, OUT_DIR)

        # end-of-base 単独分析も実施
        if analysis["boundary_end"]:
            test_end = statistical_tests(
                analysis["boundary_end"], analysis["non_boundary"],
                k, phantom_s, rng
            )
            log(f"  [end-of-base のみ] enrichment={test_end['enrichment']:.3f}x, "
                f"p={test_end['fisher_p']:.3e}")

        if analysis["boundary_start"]:
            test_start = statistical_tests(
                analysis["boundary_start"], analysis["non_boundary"],
                k, phantom_s, rng
            )
            log(f"  [start-of-base のみ] enrichment={test_start['enrichment']:.3f}x, "
                f"p={test_start['fisher_p']:.3e}")

    # ── レポート保存 ──────────────────────────────────────────────────
    report_path = OUT_DIR / "compound_boundary_report.txt"
    report_path.write_text("\n\n".join(all_reports), encoding="utf-8")
    log(f"\nレポート保存: {report_path}")
    log(f"完了。出力先: {OUT_DIR.resolve()}")
