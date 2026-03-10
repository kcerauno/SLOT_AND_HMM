"""
transition_after_bend_analysis.py
===================================
S4 → ? 実証遷移分析: B-end 直後の状態分布を確認する

interpretation_notes.md Section 7.2 に対応:
  「S4 → S? の遷移確率を確認し『S4の直後にどの状態が来るか』を検証する」

理論遷移確率（trans[S4, :]）と、実際の Viterbi パス上での
B-end → B-start 遷移の実証分布を比較する。

分析設計:
  - 全複合語を Viterbi デコード
  - B-end 位置の状態 → B-start 位置の状態 の実証遷移を集計
  - 「B-end = S4」の条件付きで、B-start に現れる状態の分布を算出
  - 理論遷移行列 trans[S4, :] と並べて比較
  - 全状態の B-end → B-start 遷移ヒートマップも出力

出力:
  transition_after_bend_k{k}.png     : S4 からの実証遷移バーチャート
  bend_to_bstart_heatmap_k{k}.png    : 全状態の B-end→B-start 遷移ヒートマップ
  transition_after_bend_report.md    : 数値レポート（Markdown）
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


def _setup_jp_font():
    candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic", "Noto Sans CJK JP"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_setup_jp_font()


# ── 設定 ─────────────────────────────────────────────────────────────────
DB_PATH     = "data/voynich.db"
MODEL_CACHE = Path("hypothesis/01_bigram/results/hmm_model_cache")
OUT_DIR     = Path("hypothesis/02_compound_hmm/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST        = [7, 8]
PHANTOM_STATE = {7: 3, 8: 4}
FOCUS_STATE   = {7: 4, 8: 5}
MIN_WORD_LEN  = 2

BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"


# ════════════════════════════════════════════════════════════════════════
# 1. V8 文法
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


def parse_greedy(word):
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


def is_base(word):
    m, r = parse_greedy(word)
    return r == "" and bool(m)


def find_v8_splits_first(word):
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


def get_boundary_positions(splits):
    boundary_end, boundary_start = [], []
    cumlen = 0
    for base in splits[:-1]:
        cumlen += len(base)
        boundary_end.append(cumlen - 1)
        boundary_start.append(cumlen)
    return boundary_end, boundary_start  # 順序を保持したリスト


# ════════════════════════════════════════════════════════════════════════
# 2. HMM / Viterbi
# ════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {DEVICE}")


def load_model(k):
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


def decode_words(words, char2idx, log_start, log_trans, log_emiss):
    results = {}
    for word in words:
        seq = np.array(
            [char2idx[BOS_CHAR]] + [char2idx[c] for c in word] + [char2idx[EOS_CHAR]],
            dtype=np.int32,
        )
        try:
            path   = viterbi_pt(log_start, log_trans, log_emiss, seq)
            states = path[1:-1]
            if len(states) > 0:
                results[word] = states
        except Exception:
            pass
    return results


# ════════════════════════════════════════════════════════════════════════
# 3. B-end → B-start 実証遷移の収集
# ════════════════════════════════════════════════════════════════════════
def collect_bend_to_bstart(compound_splits, decoded, k):
    """
    B-end 位置の状態 s_end と、その直後の B-start 位置の状態 s_start のペアを収集する。

    Returns
    -------
    pairs           : list of (s_end, s_start)
    empirical_matrix: np.ndarray (k, k) - bend_state × bstart_state のカウント
    """
    pairs = []
    empirical_matrix = np.zeros((k, k), dtype=int)

    for word, splits in compound_splits.items():
        states = decoded.get(word)
        if states is None:
            continue
        bend_positions, bstart_positions = get_boundary_positions(splits)
        # 各境界ペア (B-end_i, B-start_i) を処理
        for b_end_pos, b_start_pos in zip(bend_positions, bstart_positions):
            if b_end_pos < len(states) and b_start_pos < len(states):
                s_end   = int(states[b_end_pos])
                s_start = int(states[b_start_pos])
                pairs.append((s_end, s_start))
                empirical_matrix[s_end, s_start] += 1

    return pairs, empirical_matrix


# ════════════════════════════════════════════════════════════════════════
# 4. 可視化
# ════════════════════════════════════════════════════════════════════════
def plot_focus_transition_comparison(empirical_matrix, theoretical_trans,
                                     k, focus_s, phantom_s, out_dir):
    """
    理論遷移確率 vs 実証遷移頻度のバーチャート（focus_s の行）。
    """
    emp_row   = empirical_matrix[focus_s].astype(float)
    emp_total = emp_row.sum()
    emp_prob  = emp_row / emp_total if emp_total > 0 else emp_row

    theo_row = theoretical_trans[focus_s]

    x = np.arange(k)
    width = 0.38

    labels = []
    for s in range(k):
        suffix = " [Ph]" if s == phantom_s else (" [Fo]" if s == focus_s else ""  )
        labels.append(f"S{s}{suffix}")

    fig, ax = plt.subplots(figsize=(max(8, k + 4), 5))

    bars_emp  = ax.bar(x - width / 2, emp_prob,  width, label="実証 (Viterbi B-end→B-start)",
                       color="#C0392B", alpha=0.85)
    bars_theo = ax.bar(x + width / 2, theo_row,  width, label="理論 (trans[S{}, :])".format(focus_s),
                       color="#2980B9", alpha=0.85)

    for bar in bars_emp:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#C0392B")
    for bar in bars_theo:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#2980B9")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("遷移確率", fontsize=11)
    ax.set_title(
        f"k={k}  S{focus_s} からの遷移確率: 実証 vs 理論\n"
        f"赤: B-end=S{focus_s} のとき次の B-start 状態（実証, n={int(emp_total)}）\n"
        f"青: 理論遷移行列 trans[S{focus_s}, :]",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(emp_prob.max(), theo_row.max()) * 1.30 + 0.02)
    plt.tight_layout()

    fname = f"transition_after_bend_k{k}.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


def plot_bend_to_bstart_heatmap(empirical_matrix, k, focus_s, phantom_s, out_dir):
    """
    全状態の B-end → B-start 実証遷移ヒートマップ（行正規化）。
    """
    # 行正規化（各 B-end 状態での B-start 分布）
    row_sums = empirical_matrix.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix_norm = np.where(row_sums > 0, empirical_matrix / row_sums, 0.0)

    labels = []
    for s in range(k):
        suffix = " [Ph]" if s == phantom_s else (" [Fo]" if s == focus_s else "")
        labels.append(f"S{s}{suffix}")

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, k + 1)))

    # 左: カウント
    im0 = axes[0].imshow(empirical_matrix, aspect="auto", cmap="YlOrRd")
    for r in range(k):
        for c in range(k):
            axes[0].text(c, r, str(empirical_matrix[r, c]),
                         ha="center", va="center", fontsize=8,
                         color="white" if empirical_matrix[r, c] > empirical_matrix.max() * 0.6 else "black")
    axes[0].set_xticks(range(k))
    axes[0].set_yticks(range(k))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel("B-start 状態", fontsize=10)
    axes[0].set_ylabel("B-end 状態", fontsize=10)
    axes[0].set_title(f"B-end → B-start 実証遷移（カウント）", fontsize=11)
    plt.colorbar(im0, ax=axes[0])

    # 右: 行正規化（確率）
    im1 = axes[1].imshow(matrix_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for r in range(k):
        for c in range(k):
            val = matrix_norm[r, c]
            axes[1].text(c, r, f"{val:.2f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if val > 0.6 else "black")
    axes[1].set_xticks(range(k))
    axes[1].set_yticks(range(k))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel("B-start 状態", fontsize=10)
    axes[1].set_ylabel("B-end 状態", fontsize=10)
    axes[1].set_title(f"B-end → B-start 実証遷移（行正規化）\nS{focus_s}行に注目", fontsize=11)
    plt.colorbar(im1, ax=axes[1])

    # focus_s の行を枠線で強調
    for ax in axes:
        ax.add_patch(plt.Rectangle(
            (-0.5, focus_s - 0.5), k, 1,
            fill=False, edgecolor="#C0392B", linewidth=2.5
        ))

    plt.suptitle(f"k={k}: 複合語境界における実証遷移行列", fontsize=12, y=1.01)
    plt.tight_layout()

    fname = f"bend_to_bstart_heatmap_k{k}.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


# ════════════════════════════════════════════════════════════════════════
# 5. Markdown レポート
# ════════════════════════════════════════════════════════════════════════
def build_md_report(results_by_k):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# S4 → ? 実証遷移分析: B-end 直後の状態分布",
        "",
        f"**生成日時**: {now}",
        "**スクリプト**: `hypothesis/02_compound_hmm/source/transition_after_bend_analysis.py`",
        "",
        "---",
        "",
        "## 検証目的",
        "",
        "`interpretation_notes.md` Section 5.3 のメカニズム検証:",
        "",
        "> S4 は S4 の直後には来にくい（自己遷移が低い）ため、",
        "> B-end で S4 が出た後の B-start では S4 が抑制される。",
        "",
        "理論値（遷移行列）と実証値（Viterbi パス上での B-end→B-start 実測遷移）を直接比較する。",
        "",
        "---",
        "",
    ]

    for k, res in results_by_k.items():
        if res is None:
            lines += [f"## k={k}", "", "⚠ モデルキャッシュが見つかりません", "", "---", ""]
            continue

        focus_s   = FOCUS_STATE[k]
        phantom_s = PHANTOM_STATE[k]
        emp_mat   = res["empirical_matrix"]
        theo_trans = res["trans"]
        total_pairs = int(emp_mat.sum())

        lines += [
            f"## k={k}  (Phantom: S{phantom_s}, Focus: S{focus_s})",
            "",
            f"**集計対象**: 全複合語の全境界ペア（B-end, B-start）合計 **{total_pairs:,}** ペア",
            "",
        ]

        # 図
        for fig_key in ["focus_comparison", "heatmap"]:
            fname = res.get(f"plot_{fig_key}")
            alt = {
                "focus_comparison": f"S{focus_s} transition comparison k={k}",
                "heatmap":          f"B-end to B-start heatmap k={k}",
            }[fig_key]
            if fname:
                lines.append(f"![{alt}](../results/{fname})")
        lines.append("")

        # Focus state の実証遷移分布
        emp_row   = emp_mat[focus_s].astype(float)
        emp_total = int(emp_row.sum())
        emp_prob  = emp_row / emp_total if emp_total > 0 else emp_row
        theo_row  = theo_trans[focus_s]

        lines += [
            f"### S{focus_s} を B-end 状態としたときの B-start 状態分布",
            "",
            f"（「B-end = S{focus_s}」の条件付き B-start 状態分布, n={emp_total:,}）",
            "",
            "| B-start 状態 | 実証確率 | 実証カウント | 理論確率 (trans) | 差分 | 注記 |",
            "|-------------|---------|------------|----------------|------|------|",
        ]
        for s_to in range(k):
            diff = float(emp_prob[s_to]) - float(theo_row[s_to])
            note = ""
            if s_to == focus_s:
                note = "**自己遷移**"
            elif s_to == phantom_s:
                note = "[Phantom]"
            lines.append(
                f"| S{s_to} "
                f"| {emp_prob[s_to]:.3f} ({emp_prob[s_to]*100:.1f}%) "
                f"| {int(emp_row[s_to]):,} "
                f"| {theo_row[s_to]:.3f} ({theo_row[s_to]*100:.1f}%) "
                f"| {diff:+.3f} "
                f"| {note} |"
            )

        self_emp  = float(emp_prob[focus_s])
        self_theo = float(theo_row[focus_s])
        lines += [
            "",
            f"> **S{focus_s} 自己遷移**: 実証 = {self_emp:.3f} ({self_emp*100:.1f}%) / "
            f"理論 = {self_theo:.3f} ({self_theo*100:.1f}%)",
            "",
        ]

        # 実証遷移行列の全体
        lines += [
            f"### B-end → B-start 実証遷移行列（全状態、行正規化）",
            "",
            "行: B-end 状態 / 列: B-start 状態 / 値: 実証確率（行合計=1）",
            "",
        ]
        header = "| B-end \\ B-start |" + "".join(f" S{s} |" for s in range(k))
        sep    = "|" + "---|" * (k + 1)
        lines += [header, sep]
        for r in range(k):
            row_sum = emp_mat[r].sum()
            if row_sum > 0:
                row_probs = emp_mat[r] / row_sum
            else:
                row_probs = np.zeros(k)
            note = ""
            if r == focus_s:
                note = " **[Focus]**"
            elif r == phantom_s:
                note = " [Phantom]"
            cell_strs = " | ".join(
                f"**{row_probs[c]:.2f}**" if c == r else f"{row_probs[c]:.2f}"
                for c in range(k)
            )
            lines.append(f"| S{r}{note} | {cell_strs} |")

        lines += ["", f"（カウント合計: {total_pairs:,}）", "", "---", ""]

    # 総括
    lines += [
        "## 総括",
        "",
        "本分析は理論遷移行列（Baum-Welch 学習結果）と",
        "実際の Viterbi パス上での B-end → B-start 実証遷移を直接比較した。",
        "",
        "| k | Focus | 実証自己遷移 | 理論自己遷移 | 差分 | 解釈 |",
        "|---|-------|------------|------------|------|------|",
    ]
    for k, res in results_by_k.items():
        if res is None:
            lines.append(f"| {k} | — | — | — | — | 分析失敗 |")
            continue
        focus_s   = FOCUS_STATE[k]
        emp_mat   = res["empirical_matrix"]
        theo_trans = res["trans"]
        emp_row   = emp_mat[focus_s].astype(float)
        emp_total = emp_row.sum()
        emp_prob  = emp_row / emp_total if emp_total > 0 else emp_row
        self_emp  = float(emp_prob[focus_s])
        self_theo = float(theo_trans[focus_s, focus_s])
        diff      = self_emp - self_theo
        interp = (
            "実証 < 理論 → 実際の境界では自己遷移が抑制されている"
            if diff < -0.05 else
            "実証 ≈ 理論 → 境界遷移は理論値通り"
            if abs(diff) <= 0.05 else
            "実証 > 理論 → 境界位置で自己遷移が促進されている"
        )
        lines.append(
            f"| {k} | S{focus_s} "
            f"| {self_emp:.3f} ({self_emp*100:.1f}%) "
            f"| {self_theo:.3f} ({self_theo*100:.1f}%) "
            f"| {diff:+.3f} "
            f"| {interp} |"
        )

    lines += [
        "",
        "_本レポートは `transition_after_bend_analysis.py` により自動生成。_",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("S4→? 実証遷移分析 開始")

    # ── データロード ──────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    words_all = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
        conn,
    )["word"].tolist()
    conn.close()

    all_types = sorted(set(w for w in words_all if len(w) >= MIN_WORD_LEN))
    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR, EOS_CHAR, PAD_CHAR] + raw_chars
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    log(f"ユニーク単語数: {len(all_types):,}  語彙サイズ: {len(all_chars)}")

    # ── V8 複合語 ─────────────────────────────────────────────────────
    log("V8複合語を特定中...")
    compound_splits = {}
    for w in all_types:
        splits = find_v8_splits_first(w)
        if splits is not None:
            compound_splits[w] = splits
    log(f"  V8複合語: {len(compound_splits):,} 語")

    all_results = {}

    for k in K_LIST:
        log(f"\n{'='*55}\n  k = {k}\n{'='*55}")
        phantom_s = PHANTOM_STATE[k]
        focus_s   = FOCUS_STATE[k]

        info = load_model(k)
        if info is None:
            all_results[k] = None
            continue
        log(f"  モデルロード完了 (logL={info['logL']:.2f})")

        log_start  = torch.tensor(np.log(info["start"] + 1e-35), device=DEVICE)
        log_trans  = torch.tensor(np.log(info["trans"]  + 1e-35), device=DEVICE)
        log_emiss  = torch.tensor(np.log(info["emiss"]  + 1e-35), device=DEVICE)

        log(f"  Viterbiデコード中: {len(compound_splits):,} 語...")
        decoded = decode_words(
            compound_splits.keys(), char2idx, log_start, log_trans, log_emiss
        )
        log(f"  完了: {len(decoded):,} 語")

        log("  B-end → B-start 遷移を収集中...")
        pairs, empirical_matrix = collect_bend_to_bstart(compound_splits, decoded, k)
        log(f"  収集ペア数: {len(pairs):,}")

        # S{focus_s} が B-end のケース
        focus_as_bend = int(empirical_matrix[focus_s].sum())
        log(f"  B-end = S{focus_s} のケース: {focus_as_bend:,}")

        emp_row   = empirical_matrix[focus_s].astype(float)
        emp_prob  = emp_row / emp_row.sum() if emp_row.sum() > 0 else emp_row
        theo_row  = info["trans"][focus_s]

        log(f"  S{focus_s} → ? 実証 vs 理論:")
        for s_to in range(k):
            marker = " ★" if s_to == focus_s else ""
            log(f"    → S{s_to}: 実証 {emp_prob[s_to]:.3f}  理論 {theo_row[s_to]:.3f}{marker}")

        # 全状態の B-end→B-start
        log("\n  B-end → B-start 実証遷移行列（行正規化）:")
        for r in range(k):
            row_sum = empirical_matrix[r].sum()
            if row_sum > 0:
                row_p = empirical_matrix[r] / row_sum
                row_str = "  ".join(f"→S{c}:{row_p[c]:.2f}" for c in range(k))
                note = " [Focus]" if r == focus_s else (" [Phantom]" if r == phantom_s else "")
                log(f"    S{r}{note}: {row_str}")

        # プロット
        fname_comp = plot_focus_transition_comparison(
            empirical_matrix, info["trans"], k, focus_s, phantom_s, OUT_DIR
        )
        fname_heat = plot_bend_to_bstart_heatmap(
            empirical_matrix, k, focus_s, phantom_s, OUT_DIR
        )

        all_results[k] = {
            "trans":            info["trans"],
            "empirical_matrix": empirical_matrix,
            "pairs":            pairs,
            "plot_focus_comparison": fname_comp,
            "plot_heatmap":     fname_heat,
        }

    # ── Markdown レポート ─────────────────────────────────────────────
    log("\nMarkdown レポート生成中...")
    md = build_md_report(all_results)
    report_path = OUT_DIR / "transition_after_bend_report.md"
    report_path.write_text(md, encoding="utf-8")
    log(f"レポート保存: {report_path}")
    log(f"\n完了。出力先: {OUT_DIR.resolve()}")
