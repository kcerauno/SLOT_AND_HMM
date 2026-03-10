"""
correct_boundary_analysis.py
==============================
V8複合語仮説 × HMM Phantom State 境界検証（正確版）

hypothesis/02_compound_hmm/source/compound_boundary_analysis.py の修正版。

【変更点】
  旧: find_v8_splits_first() で動的に複合語を分類
      → SLOTS_V8[0] 始まりの単独ベース語を誤判定し複合語数が ~8,058 に膨張

  新: compound_words.txt（分割情報付き）を正解データとして直接ロード
      → 正解の 3,363 語のみを複合語として使用

実行:
  cd /home/practi/work_voy
  python hypothesis/04_re-compound_hmm/source/correct_boundary_analysis.py
"""

import re
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
    candidates = ["Noto Sans CJK JP", "IPAexGothic", "Yu Gothic", "Meiryo", "MS Gothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_setup_jp_font()

# ── 設定 ──────────────────────────────────────────────────────────────────
DB_PATH             = "data/voynich.db"
MODEL_CACHE         = Path("hypothesis/01_bigram/results/hmm_model_cache")
COMPOUND_SPLIT_PATH = Path("hypothesis/00_slot_model/data/compound_words.txt")
OUT_DIR             = Path("hypothesis/04_re-compound_hmm/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST         = [7, 8]
PHANTOM_STATE  = {7: 3, 8: 4}   # k=7: S3、k=8: S4
N_SHUFFLE      = 1000
MIN_WORD_LEN   = 2
TOP_WORDS_PLOT = 30

BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"

# hypothesis/02 との比較用（旧動的分類の結果）
OLD_COMPOUND_COUNT = {7: 8058, 8: 8058}   # analysis_report.md 記載値（概算）


# ════════════════════════════════════════════════════════════════════════
# 1. 正解データのロード
# ════════════════════════════════════════════════════════════════════════
def load_compound_splits(path: Path) -> dict:
    """
    compound_words.txt をパース。
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
# 2. V8 文法定義（境界位置計算用）
# ════════════════════════════════════════════════════════════════════════
def get_boundary_positions(splits: tuple) -> tuple:
    """
    splits = (base1, base2, ..., baseN) から境界位置を計算。
    boundary_end   : 各基（最終基以外）の最終文字位置
    boundary_start : 各基（第1基以外）の先頭文字位置
    """
    boundary_end   = set()
    boundary_start = set()
    cumlen = 0
    for base in splits[:-1]:
        cumlen += len(base)
        boundary_end.add(cumlen - 1)
        boundary_start.add(cumlen)
    return boundary_end, boundary_start


# ════════════════════════════════════════════════════════════════════════
# 3. HMM モデル & Viterbi（compound_boundary_analysis.py から移植）
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
# 4. 境界位置分析（compound_boundary_analysis.py から移植・変更なし）
# ════════════════════════════════════════════════════════════════════════
def analyze_boundaries(info, compound_splits, char2idx, k):
    log_start = torch.tensor(np.log(info["start"] + 1e-35), device=DEVICE)
    log_trans  = torch.tensor(np.log(info["trans"]  + 1e-35), device=DEVICE)
    log_emiss  = torch.tensor(np.log(info["emiss"]  + 1e-35), device=DEVICE)

    bd_end_states   = []
    bd_start_states = []
    bd_both_states  = []
    non_bd_states   = []
    word_paths      = {}

    skipped = 0
    for word, splits in compound_splits.items():
        # char2idx に存在しない文字を持つ語はスキップ
        if not all(c in char2idx for c in word):
            skipped += 1
            continue

        seq = np.array(
            [char2idx[BOS_CHAR]] + [char2idx[c] for c in word] + [char2idx[EOS_CHAR]],
            dtype=np.int32,
        )
        try:
            path = viterbi_pt(log_start, log_trans, log_emiss, seq)
        except Exception:
            continue

        states = path[1:-1]
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

    if skipped > 0:
        log(f"  char2idx未登録のためスキップ: {skipped:,} 語")

    return {
        "boundary_end":   bd_end_states,
        "boundary_start": bd_start_states,
        "boundary_both":  bd_both_states,
        "non_boundary":   non_bd_states,
        "word_paths":     word_paths,
    }


# ════════════════════════════════════════════════════════════════════════
# 5. 統計検定（compound_boundary_analysis.py から移植・変更なし）
# ════════════════════════════════════════════════════════════════════════
def statistical_tests(boundary_states, non_boundary_states, k, phantom_s, rng):
    b_counts  = np.bincount(boundary_states,     minlength=k).astype(float)
    nb_counts = np.bincount(non_boundary_states, minlength=k).astype(float)
    b_total   = b_counts.sum()
    nb_total  = nb_counts.sum()

    col_ok = (b_counts + nb_counts) > 0
    contingency = np.array([b_counts, nb_counts])
    chi2_val, chi2_p, chi2_dof, _ = stats.chi2_contingency(contingency[:, col_ok])

    b_phantom  = b_counts[phantom_s]
    nb_phantom = nb_counts[phantom_s]
    b_other    = b_total  - b_phantom
    nb_other   = nb_total - nb_phantom

    table = [[int(b_phantom), int(b_other)],
             [int(nb_phantom), int(nb_other)]]
    if b_phantom + nb_phantom > 0:
        odds_ratio, fisher_p = stats.fisher_exact(table, alternative="greater")
    else:
        odds_ratio, fisher_p = np.nan, np.nan

    b_rate  = b_phantom  / b_total  if b_total  > 0 else 0.0
    nb_rate = nb_phantom / nb_total if nb_total > 0 else 0.0
    enrich  = b_rate / nb_rate if nb_rate > 0 else np.nan

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

    null_arr = np.array([v for v in null_enrichments if not np.isnan(v)])
    pctile   = np.mean(null_arr <= enrich) * 100 if len(null_arr) > 0 else np.nan

    return {
        "b_counts":         b_counts,
        "nb_counts":        nb_counts,
        "b_total":          b_total,
        "nb_total":         nb_total,
        "chi2":             chi2_val,
        "chi2_p":           chi2_p,
        "chi2_dof":         chi2_dof,
        "b_phantom":        b_phantom,
        "nb_phantom":       nb_phantom,
        "b_rate":           b_rate,
        "nb_rate":          nb_rate,
        "enrichment":       enrich,
        "odds_ratio":       odds_ratio,
        "fisher_p":         fisher_p,
        "null_enrichments": null_arr,
        "pctile":           pctile,
    }


# ════════════════════════════════════════════════════════════════════════
# 6. 可視化（compound_boundary_analysis.py から移植・タイトル変更）
# ════════════════════════════════════════════════════════════════════════
def plot_state_distribution(test, k, phantom_s, out_dir, suffix=""):
    b_counts  = test["b_counts"]
    nb_counts = test["nb_counts"]
    b_total   = test["b_total"]
    nb_total  = test["nb_total"]

    b_rates  = b_counts  / b_total  * 100 if b_total  > 0 else b_counts * 0
    nb_rates = nb_counts / nb_total * 100 if nb_total > 0 else nb_counts * 0

    x = np.arange(len(b_counts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, k * 1.2), 5))
    bars_b  = ax.bar(x - width / 2, b_rates,  width, label="境界位置 (seam)", color="#4878CF", alpha=0.85)
    bars_nb = ax.bar(x + width / 2, nb_rates, width, label="非境界位置",       color="#F0814E", alpha=0.85)

    for bar in [bars_b[phantom_s], bars_nb[phantom_s]]:
        bar.set_edgecolor("red")
        bar.set_linewidth(2.5)

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
        f"[正確版] V8複合境界 vs 非境界 での状態分布  k={k}\n"
        f"★=Phantom State (S{phantom_s})  "
        f"enrichment={test['enrichment']:.2f}x  Fisher p={test['fisher_p']:.2e}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = out_dir / f"corrected_boundary_state_dist_k{k}{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path.name}")


def plot_phantom_enrichment(test, k, phantom_s, out_dir, suffix=""):
    null_arr = test["null_enrichments"]
    enrich   = test["enrichment"]
    pctile   = test["pctile"]

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(null_arr) > 0:
        ax.hist(null_arr, bins=40, color="steelblue", alpha=0.7,
                label=f"Null分布 ({N_SHUFFLE}回シャッフル)")
    ax.axvline(enrich, color="red", linewidth=2.5,
               label=f"実測値 {enrich:.3f}x\n(上位{100 - pctile:.1f}%ile, p≈{test['fisher_p']:.2e})")
    ax.set_xlabel("Phantom State enrichment ratio (境界/非境界)", fontsize=10)
    ax.set_ylabel("頻度", fontsize=10)
    ax.set_title(f"[正確版] Phantom State (S{phantom_s}) 境界集中度  k={k}", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = out_dir / f"corrected_phantom_enrichment_k{k}{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path.name}")


def plot_word_paths(analysis, k, phantom_s, out_dir, top_n=TOP_WORDS_PLOT):
    word_paths = analysis["word_paths"]

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

    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=k - 1, interpolation="nearest")

    for row in range(len(sorted_words)):
        for col in range(max_len):
            if seam[row, col]:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                           fill=False, edgecolor="white", linewidth=2))

    for row in range(len(sorted_words)):
        for col in range(max_len):
            if mat[row, col] == phantom_s:
                ax.text(col, row, "×", ha="center", va="center",
                        fontsize=8, color="red", fontweight="bold")

    ax.set_yticks(range(len(sorted_words)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("文字位置 (BOS/EOS 除く)", fontsize=9)
    ax.set_title(
        f"[正確版] 複合語 Viterbi 状態パス  k={k}\n"
        f"■白枠=V8境界位置  ×=Phantom State(S{phantom_s})",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label=f"HMM状態 (0〜{k-1})", shrink=0.6)
    plt.tight_layout()
    path = out_dir / f"corrected_word_paths_k{k}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path.name}")


# ════════════════════════════════════════════════════════════════════════
# 7. テキストレポート（hypothesis/02 との比較追加）
# ════════════════════════════════════════════════════════════════════════
def build_report(analysis, test, k, phantom_s, compound_splits,
                 test_end=None, test_start=None):
    b_counts  = test["b_counts"]
    nb_counts = test["nb_counts"]
    b_total   = int(test["b_total"])
    nb_total  = int(test["nb_total"])
    n_compounds = len(analysis["word_paths"])
    n_bd_end   = len(analysis["boundary_end"])
    n_bd_start = len(analysis["boundary_start"])
    n_bd_both  = len(analysis["boundary_both"])
    n_non      = len(analysis["non_boundary"])

    b_rates  = (b_counts  / b_total  * 100) if b_total  > 0 else b_counts * 0
    nb_rates = (nb_counts / nb_total * 100) if nb_total > 0 else nb_counts * 0
    diff     = b_rates - nb_rates
    top_idx  = np.argsort(-diff)

    lines = [
        "=" * 72,
        f"[正確版] V8複合語仮説 × HMM Phantom State 境界検証  k={k}",
        f"  生成日時: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"  正解データ: compound_words.txt ({len(compound_splits):,} 語)",
        "=" * 72,
        "",
        f"  対象複合語数 (V8正解データ): {n_compounds:,} 語",
        f"    ← 旧動的分類: 約 {OLD_COMPOUND_COUNT.get(k, '?'):,} 語（約{OLD_COMPOUND_COUNT.get(k,1)/n_compounds:.1f}倍の膨張を修正）",
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
    ]

    if test_end is not None:
        lines += [
            "",
            f"  [end-of-base のみ] enrichment={test_end['enrichment']:.3f}x, p={test_end['fisher_p']:.3e}",
        ]
    if test_start is not None:
        lines += [
            f"  [start-of-base のみ] enrichment={test_start['enrichment']:.3f}x, p={test_start['fisher_p']:.3e}",
        ]

    lines += [
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

    lines += ["", "─" * 72, "  解釈", "─" * 72]
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

    lines += [f"  {verdict}", ""]

    # hypothesis/02 との比較
    lines += [
        "─" * 72,
        "  hypothesis/02 との比較（偽陽性修正の影響）",
        "─" * 72,
        f"  分析対象複合語数: {n_compounds:,} 語  ← 旧: 約 {OLD_COMPOUND_COUNT.get(k, '?'):,} 語",
        f"  enrichment ratio: {enrich:.4f}x",
        "  ※ enrich が旧値より高い → 偽陽性除去でシグナルが純化",
        "  ※ enrich が旧値と同程度 → 偽陽性は結果に大きく影響しなかった（頑健性確認）",
        "",
    ]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("[正確版] V8複合語仮説 × HMM Phantom State 境界検証 開始")
    rng = np.random.default_rng(42)

    # ── 正解複合語データのロード ─────────────────────────────────────────
    log(f"正解複合語データロード: {COMPOUND_SPLIT_PATH}")
    compound_splits_all = load_compound_splits(COMPOUND_SPLIT_PATH)
    log(f"  正解複合語数: {len(compound_splits_all):,} 語")

    from collections import Counter
    split_dist = Counter(len(v) for v in compound_splits_all.values())
    for nb, cnt in sorted(split_dist.items()):
        log(f"    [{nb}基]: {cnt:,} 語")

    # ── DB から語彙をロード（char2idx 構築用）────────────────────────────
    log(f"DBロード: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    words_all = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
        conn,
    )["word"].tolist()
    conn.close()

    all_types = sorted(set(words_all))
    all_types = [w for w in all_types if len(w) >= MIN_WORD_LEN]
    log(f"ユニーク単語数: {len(all_types):,}")

    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR, EOS_CHAR, PAD_CHAR] + raw_chars
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    log(f"語彙サイズ: {len(all_chars)}")

    # char2idx に存在する語のみ使用
    compound_splits = {
        w: splits for w, splits in compound_splits_all.items()
        if all(c in char2idx for c in w)
    }
    skipped = len(compound_splits_all) - len(compound_splits)
    log(f"  char2idx 対応済み複合語: {len(compound_splits):,} 語  (スキップ: {skipped:,} 語)")

    all_reports = []

    for k in K_LIST:
        log(f"{'='*50}\n  k = {k}\n{'='*50}")
        phantom_s = PHANTOM_STATE[k]

        info = load_model(k)
        if info is None:
            log(f"  ERROR: キャッシュが見つかりません: {MODEL_CACHE}/full_k{k}.npz")
            continue
        log(f"  モデルロード完了 (logL={info['logL']:.2f})")

        log(f"  Viterbi デコード + 境界分析中 ({len(compound_splits):,} 複合語)...")
        analysis = analyze_boundaries(info, compound_splits, char2idx, k)
        log(f"  デコード完了: {len(analysis['word_paths']):,} 語")
        log(f"  境界位置:   {len(analysis['boundary_both']):,} 件")
        log(f"  非境界位置: {len(analysis['non_boundary']):,} 件")

        log(f"  統計検定中 (Phantom=S{phantom_s}, shuffle={N_SHUFFLE}回)...")
        test = statistical_tests(
            analysis["boundary_both"], analysis["non_boundary"],
            k, phantom_s, rng
        )
        log(f"  Phantom S{phantom_s}: enrichment={test['enrichment']:.3f}x, "
            f"Fisher p={test['fisher_p']:.3e}")

        test_end = test_start = None
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

        report = build_report(analysis, test, k, phantom_s, compound_splits,
                               test_end, test_start)
        all_reports.append(report)
        print(report)

        plot_state_distribution(test, k, phantom_s, OUT_DIR)
        plot_phantom_enrichment(test, k, phantom_s, OUT_DIR)
        plot_word_paths(analysis, k, phantom_s, OUT_DIR)

    report_path = OUT_DIR / "corrected_boundary_report.txt"
    report_path.write_text("\n\n".join(all_reports), encoding="utf-8")
    log(f"\nレポート保存: {report_path}")
    log(f"完了。出力先: {OUT_DIR.resolve()}")
