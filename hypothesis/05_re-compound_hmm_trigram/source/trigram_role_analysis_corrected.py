"""
trigram_role_analysis_corrected.py
====================================
Trigram HMM 役割分析（正確版）+ s_{t-2} 文脈曖昧性解消検証

hypothesis/03_trigram/source/trigram_role_analysis.py の全バグ修正版。
trigram_hmm_fast.py のモデル定義（パラメータ名・npz 保存形式）を基準とする。

【修正点】
  Bug 1: find_v8_splits_first() 偽陽性
         → compound_words.txt を直接ロード（正解 3,363 語のみ使用）

  Bug 2: 近似 Trigram Viterbi（logsumexp 周辺化）
         → trigram_hmm_fast.py の TrigramHMM_Batched.viterbi() と同一の
            exact Viterbi（(k×k) デルタ行列）を採用

  Bug 3: バックトラッキング off-by-one（trigram_hmm_fast.py L386 と共通）
         → psi_list[t_back - 2] を psi_list[t_back - 1] に修正

  Bug 4: start_trans 未ロード
         → npz から start_trans を含めてロード

  Bug 5: Focus State ハードコード S6
         → B-start 出現率最大の状態を data-driven に同定
            （Phantom = Viterbi 占有率 0% の状態は候補から除外）

分析内容:
  A. 4グループ役割分析（hypothesis/04 と同じ枠組み）
     - B-start / B-end / S-head / S-mid の Focus State 出現率
     - Fisher 正確検定（4比較）

  B. s_{t-2} 文脈曖昧性解消（Trigram 固有の仮説）
     - Focus State 登場位置での s_{t-2} 分布を B-end vs B-start で比較
     - 各 s_{t-2} に対し Fisher 正確検定（両側）

実行:
  cd /home/practi/work_voy
  PYTHONPATH=/home/practi/work_voy/.venv/lib/python3.10/site-packages \\
    python3.10 hypothesis/05_re-compound_hmm_trigram/source/trigram_role_analysis_corrected.py
"""

import re
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
from pathlib import Path
from datetime import datetime
from scipy import stats
from collections import Counter
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
DB_PATH             = "data/voynich.db"
TRIGRAM_CACHE       = Path("hypothesis/03_trigram/results/hmm_model_cache")
COMPOUND_SPLIT_PATH = Path("hypothesis/00_slot_model/data/compound_words.txt")
SINGLE_WORDS_PATH   = Path("hypothesis/00_slot_model/data/words_base_only.txt")
OUT_DIR             = Path("hypothesis/05_re-compound_hmm_trigram/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST       = [7, 8]
MIN_WORD_LEN = 2
BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"


# ════════════════════════════════════════════════════════════════════════
# 1. データロード（Bug 1 修正）
# ════════════════════════════════════════════════════════════════════════
def load_compound_splits(path: Path) -> dict:
    """compound_words.txt をパース。形式: "[N基] word  ->  base1 + base2 + ..." """
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


def load_word_list(path: Path) -> list:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("<"):
            words.append(line)
    return words


def get_boundary_positions(splits: tuple) -> tuple:
    """B-end（基末尾位置）と B-start（基先頭位置）を返す"""
    boundary_end, boundary_start = set(), set()
    cumlen = 0
    for base in splits[:-1]:
        cumlen += len(base)
        boundary_end.add(cumlen - 1)
        boundary_start.add(cumlen)
    return boundary_end, boundary_start


# ════════════════════════════════════════════════════════════════════════
# 2. モデルロード（Bug 4 修正: start_trans 含む）
# ════════════════════════════════════════════════════════════════════════
def load_trigram_model(k: int) -> dict | None:
    """trigram_hmm_fast.py の save_model() 形式で保存された npz をロード"""
    path = TRIGRAM_CACHE / f"trigram_k{k}.npz"
    if not path.exists():
        log(f"  モデルが見つかりません: {path}")
        return None
    d = np.load(path)
    return {
        "log_start":       torch.tensor(np.log(d["start"]       + 1e-35), device=DEVICE),
        "log_start_trans": torch.tensor(np.log(d["start_trans"] + 1e-35), device=DEVICE),
        "log_transmat":    torch.tensor(np.log(d["trans"]        + 1e-35), device=DEVICE),
        "log_emiss":       torch.tensor(np.log(d["emiss"]        + 1e-35), device=DEVICE),
        "logL":            float(d["logL"][0]),
        "k":               int(d["trans"].shape[0]),
    }


# ════════════════════════════════════════════════════════════════════════
# 3. Exact Trigram Viterbi（Bug 2,3,4 修正済み）
# ════════════════════════════════════════════════════════════════════════
def viterbi_trigram(model: dict, X_np: np.ndarray) -> list:
    """
    trigram_hmm_fast.py の TrigramHMM_Batched.viterbi() と同一ロジック。
    バックトラッキングの off-by-one バグ（Bug 3）を修正済み。
    """
    log_start       = model["log_start"]
    log_start_trans = model["log_start_trans"]
    log_transmat    = model["log_transmat"]
    log_emiss       = model["log_emiss"]

    X = torch.tensor(X_np, dtype=torch.long, device=DEVICE)
    T = X.shape[0]
    k = log_start.shape[0]

    if T == 0:
        return []

    log_delta_1d = log_start + log_emiss[:, X[0]]   # (k,)

    if T == 1:
        return [torch.argmax(log_delta_1d).item()]

    # t=1
    log_delta = (log_delta_1d.unsqueeze(1)
                 + log_start_trans
                 + log_emiss[:, X[1]].unsqueeze(0))  # (k, k)

    psi_list = []  # psi_list[idx][j, l] = best s_{t-2} at time t=idx+2
    for t in range(2, T):
        vals = log_delta.unsqueeze(2) + log_transmat   # (k, k, k)
        max_vals, argmax_i = torch.max(vals, dim=0)    # (k, k)
        new_delta = max_vals + log_emiss[:, X[t]].unsqueeze(0)
        psi_list.append(argmax_i.cpu())
        log_delta = new_delta

    # バックトラック（Bug 3 修正: t_back - 2 → t_back - 1）
    flat_idx = torch.argmax(log_delta)
    j_last = (flat_idx // k).item()
    l_last = (flat_idx % k).item()

    path = [0] * T
    path[T - 1] = l_last
    path[T - 2] = j_last
    for t_back in range(T - 2, 1, -1):
        path[t_back - 1] = psi_list[t_back - 1][path[t_back], path[t_back + 1]].item()

    return path


def decode_words(words, char2idx: dict, model: dict) -> dict:
    """語リストを Viterbi デコード。{word: states(BOS/EOS 除く)} を返す"""
    results = {}
    for word in words:
        if not all(c in char2idx for c in word):
            continue
        seq = np.array(
            [char2idx[BOS_CHAR]] + [char2idx[c] for c in word] + [char2idx[EOS_CHAR]],
            dtype=np.int32,
        )
        try:
            path = viterbi_trigram(model, seq)
            states = path[1:-1]
            if len(states) > 0:
                results[word] = states
        except Exception:
            pass
    return results


# ════════════════════════════════════════════════════════════════════════
# 4. Viterbi 占有率（全語）→ Phantom State 同定
# ════════════════════════════════════════════════════════════════════════
def compute_viterbi_occupancy(model: dict, all_words: list, char2idx: dict) -> np.ndarray:
    """全語の Viterbi デコードから各状態の占有率を返す"""
    k = model["k"]
    counts = np.zeros(k, dtype=int)
    total  = 0
    for word in all_words:
        if not all(c in char2idx for c in word):
            continue
        seq = np.array(
            [char2idx[BOS_CHAR]] + [char2idx[c] for c in word] + [char2idx[EOS_CHAR]],
            dtype=np.int32,
        )
        try:
            path = viterbi_trigram(model, seq)
            for s in path[1:-1]:
                counts[s] += 1
                total += 1
        except Exception:
            pass
    return counts / total if total > 0 else np.zeros(k)


# ════════════════════════════════════════════════════════════════════════
# 5. 4グループ状態収集
# ════════════════════════════════════════════════════════════════════════
def collect_groups(compound_splits, single_words,
                   decoded_compound, decoded_single) -> dict:
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
        if states is None:
            continue
        L = len(states)
        s_head.append(int(states[0]))
        s_mid.append(int(states[L // 2]))

    return {"B-start": b_start, "B-end": b_end, "S-head": s_head, "S-mid": s_mid}


# ════════════════════════════════════════════════════════════════════════
# 6. Fisher 正確検定（4比較）
# ════════════════════════════════════════════════════════════════════════
def fisher_2x2(group_a: list, group_b: list, target_state: int, k: int) -> dict:
    ca = np.bincount(group_a, minlength=k).astype(float)
    cb = np.bincount(group_b, minlength=k).astype(float)
    a_t = int(ca[target_state]); a_o = int(ca.sum()) - a_t
    b_t = int(cb[target_state]); b_o = int(cb.sum()) - b_t
    if a_t + b_t == 0:
        return {"odds": np.nan, "p": np.nan,
                "a_rate": 0.0, "b_rate": 0.0,
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


def run_role_analysis(groups: dict, k: int, occupancy: np.ndarray) -> dict | None:
    """
    Focus State を B-start 出現率最大の状態から動的同定（Bug 5 修正）。
    Phantom（占有率 0%）の状態は候補から除外。
    """
    b_start_counts = np.bincount(groups["B-start"], minlength=k).astype(float)
    b_start_total  = b_start_counts.sum()
    if b_start_total == 0:
        return None

    # Phantom 状態を除外して Focus State を選択
    non_phantom = [s for s in range(k) if occupancy[s] > 0]
    if not non_phantom:
        return None

    rates_for_focus = b_start_counts.copy()
    for s in range(k):
        if occupancy[s] == 0:
            rates_for_focus[s] = -1.0  # phantom を候補から除外
    focus_state = int(np.argmax(rates_for_focus))

    group_rates  = {}
    group_counts = {}
    group_totals = {}
    for name, state_list in groups.items():
        c = np.bincount(state_list, minlength=k).astype(float)
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
        fisher_results[label] = fisher_2x2(groups[gA], groups[gB], focus_state, k)

    return {
        "focus_state":    focus_state,
        "group_rates":    group_rates,
        "group_counts":   group_counts,
        "group_totals":   group_totals,
        "fisher_results": fisher_results,
    }


# ════════════════════════════════════════════════════════════════════════
# 7. s_{t-2} 文脈曖昧性解消分析（Trigram 固有）
# ════════════════════════════════════════════════════════════════════════
def collect_context_data(model: dict, compound_splits: dict, char2idx: dict) -> pd.DataFrame:
    """
    複合語の各位置について (s_{t-1}, s_{t-2}, label) を収集。
    label ∈ {'B-end', 'B-start', 'inner'}
    """
    records = []
    for word, splits in compound_splits.items():
        if not all(c in char2idx for c in word):
            continue
        seq = np.array(
            [char2idx[BOS_CHAR]] + [char2idx[c] for c in word] + [char2idx[EOS_CHAR]],
            dtype=np.int32,
        )
        try:
            path = viterbi_trigram(model, seq)
        except Exception:
            continue
        states = path[1:-1]
        if len(states) < 2:
            continue
        bd_end, bd_start = get_boundary_positions(splits)
        for pos in range(len(states)):
            s_t  = int(states[pos])
            s_t1 = int(states[pos - 1]) if pos >= 1 else -1
            s_t2 = int(states[pos - 2]) if pos >= 2 else -1
            if pos in bd_end:
                label = "B-end"
            elif pos in bd_start:
                label = "B-start"
            else:
                label = "inner"
            records.append({"word": word, "pos": pos,
                            "s_t": s_t, "s_t1": s_t1, "s_t2": s_t2,
                            "label": label})
    return pd.DataFrame(records)


def analyze_context_disambiguation(ctx_df: pd.DataFrame, focus_state: int, k: int) -> dict:
    """
    Focus State が登場する位置での s_{t-2} 分布を B-end vs B-start で比較。
    """
    fs_df     = ctx_df[ctx_df["s_t"] == focus_state]
    bend_fs   = fs_df[fs_df["label"] == "B-end"]
    bstart_fs = fs_df[fs_df["label"] == "B-start"]

    fisher_results = []
    for ctx in range(k):
        n_bend_ctx   = int((bend_fs["s_t2"]   == ctx).sum())
        n_bstart_ctx = int((bstart_fs["s_t2"] == ctx).sum())
        n_bend_other   = len(bend_fs)   - n_bend_ctx
        n_bstart_other = len(bstart_fs) - n_bstart_ctx

        if n_bend_ctx + n_bstart_ctx > 0:
            odds, p = stats.fisher_exact(
                [[n_bend_ctx, n_bend_other], [n_bstart_ctx, n_bstart_other]],
                alternative="two-sided")
        else:
            odds, p = np.nan, np.nan

        fisher_results.append({
            "ctx_state": ctx,
            "n_bend":    n_bend_ctx,
            "n_bstart":  n_bstart_ctx,
            "rate_bend":   n_bend_ctx   / len(bend_fs)   * 100 if len(bend_fs)   > 0 else 0.0,
            "rate_bstart": n_bstart_ctx / len(bstart_fs) * 100 if len(bstart_fs) > 0 else 0.0,
            "odds":  odds,
            "p":     p,
        })

    return {
        "n_bend_total":   len(bend_fs),
        "n_bstart_total": len(bstart_fs),
        "fisher":         fisher_results,
        "ctx_dist_bend":   dict(Counter(bend_fs["s_t2"].tolist())),
        "ctx_dist_bstart": dict(Counter(bstart_fs["s_t2"].tolist())),
    }


# ════════════════════════════════════════════════════════════════════════
# 8. 可視化
# ════════════════════════════════════════════════════════════════════════
def plot_group_rates(result: dict, k: int, out_path: Path):
    focus_state  = result["focus_state"]
    group_rates  = result["group_rates"]
    group_totals = result["group_totals"]

    order  = ["B-start", "B-end", "S-head", "S-mid"]
    rates  = [group_rates[g][focus_state] for g in order]
    totals = [int(group_totals[g]) for g in order]
    colors = ["#C0392B", "#E59866", "#2980B9", "#7FB3D3"]
    xlabels = [f"{g}\n(n={t:,})" for g, t in zip(order, totals)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(xlabels, rates, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=1.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    f1 = result["fisher_results"]["①confound検定 (B-start vs S-head)"]
    if not np.isnan(f1["p"]):
        if f1["p"] > 0.05:
            note = f"Fisher①: p={f1['p']:.3f} (n.s.) → 交絡の可能性"
            note_color = "#C0392B"
        else:
            note = f"Fisher①: p={f1['p']:.2e} → 複合構造の独立効果"
            note_color = "#1A5276"
        ax.text(0.5, 0.97, note, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=note_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

    ax.set_ylabel(f"S{focus_state} 出現率 (%)", fontsize=11)
    ax.set_title(
        f"[正確版 Trigram]  k={k}  4グループ S{focus_state} 出現率\n"
        f"正解複合語 3,363語 / 正解単独ベース語 {int(group_totals['S-head']):,}語",
        fontsize=11,
    )
    ax.set_ylim(0, max(rates) * 1.40 + 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_context_distribution(ctx_result: dict, focus_state: int, k: int, out_path: Path):
    """Focus State 登場位置での s_{t-2} 分布（B-end vs B-start）"""
    state_labels = [f"S{s}" for s in range(k)]
    dist_bend   = ctx_result["ctx_dist_bend"]
    dist_bstart = ctx_result["ctx_dist_bstart"]
    total_bend   = max(sum(dist_bend.values()), 1)
    total_bstart = max(sum(dist_bstart.values()), 1)

    rates_bend   = [dist_bend.get(s, 0)   / total_bend   * 100 for s in range(k)]
    rates_bstart = [dist_bstart.get(s, 0) / total_bstart * 100 for s in range(k)]

    x = np.arange(k)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, k * 1.3), 5))
    ax.bar(x - width / 2, rates_bend,   width, label="B-end",   color="#4878CF", alpha=0.85)
    ax.bar(x + width / 2, rates_bstart, width, label="B-start", color="#F0814E", alpha=0.85)

    for r in ctx_result["fisher"]:
        s = r["ctx_state"]
        if not np.isnan(r["p"]) and r["p"] < 0.05:
            y_max = max(rates_bend[s], rates_bstart[s])
            ax.annotate("*", xy=(s, y_max + 1), ha="center", fontsize=14, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(state_labels)
    ax.set_xlabel("前文脈状態 (s_{t-2})", fontsize=10)
    ax.set_ylabel("出現率 (%)", fontsize=10)
    ax.set_title(
        f"[正確版 Trigram]  k={k}  Focus State S{focus_state}\n"
        f"Focus State 登場位置での s_{{t-2}} 分布（B-end vs B-start）",
        fontsize=11,
    )
    ax.legend()
    ax.text(0.5, 0.97,
            f"B-end: {ctx_result['n_bend_total']} 件  /  B-start: {ctx_result['n_bstart_total']} 件",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# 9. Markdown レポート
# ════════════════════════════════════════════════════════════════════════
def build_report(results_by_k: dict, compound_splits: dict, single_words: list) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# [正確版] Trigram HMM 役割分析レポート",
        "",
        f"生成日時: {now}",
        "",
        "## 修正内容",
        "",
        "- Bug 1: find_v8_splits_first() 偽陽性 → compound_words.txt を直接ロード",
        "- Bug 2: 近似 Viterbi (logsumexp 周辺化) → trigram_hmm_fast.py と同一の exact Viterbi",
        "- Bug 3: バックトラッキング off-by-one → psi_list[t_back-1] に修正",
        "- Bug 4: start_trans 未ロード → npz から正しくロード",
        "- Bug 5: Focus State ハードコード S6 → data-driven に動的同定（phantom 除外）",
        "",
        f"**対象語数**: 正解複合語 **{len(compound_splits):,}** 語 / 正解単独ベース語 **{len(single_words):,}** 語",
        "",
        "---",
        "",
        "## サマリー",
        "",
        "| k | Focus State | B-start 率 | B-end 率 | S-head 率 | S-mid 率 | Fisher① p | 判定 |",
        "|---|------------|-----------|---------|---------|---------|----------|------|",
    ]

    for k, res in results_by_k.items():
        if res is None:
            lines.append(f"| {k} | — | — | — | — | — | — | 分析失敗 |")
            continue
        role = res["role"]
        fs   = role["focus_state"]
        gr   = role["group_rates"]
        f1   = role["fisher_results"]["①confound検定 (B-start vs S-head)"]
        bstart = gr["B-start"][fs]
        bend   = gr["B-end"][fs]
        shead  = gr["S-head"][fs]
        smid   = gr["S-mid"][fs]
        p = f1["p"]
        if np.isnan(p):
            verdict, p_str = "計算不能", "N/A"
        elif p > 0.05:
            verdict, p_str = "**交絡確認**", f"{p:.3f}"
        elif bstart > shead:
            verdict, p_str = "**構造効果あり**", f"{p:.2e}"
        else:
            verdict, p_str = "**交絡否定（逆転）**", f"{p:.2e}"
        lines.append(
            f"| {k} | S{fs} | {bstart:.2f}% | {bend:.2f}% | {shead:.2f}% | {smid:.2f}% "
            f"| {p_str} | {verdict} |"
        )

    lines += ["", "---", ""]

    for k, res in results_by_k.items():
        if res is None:
            lines += [f"## k={k}", "", "分析失敗。", "", "---", ""]
            continue

        role  = res["role"]
        ctx   = res["context"]
        occ   = res["occupancy"]
        fs    = role["focus_state"]
        gr    = role["group_rates"]
        gt    = role["group_totals"]
        gc    = role["group_counts"]

        phantoms = [f"S{s}" for s in range(k) if occ[s] == 0.0]

        lines += [
            f"## k={k}  (Focus State: S{fs})",
            "",
            f"**Phantom State**: {', '.join(phantoms) if phantoms else 'なし'}",
            "",
            f"### A. 4グループ役割分析",
            "",
            f"| グループ | S{fs} 出現率 | 件数 | 合計 |",
            "|---------|-------------|------|------|",
        ]
        for g in ["B-start", "B-end", "S-head", "S-mid"]:
            rate  = gr[g][fs]
            cnt   = int(gc[g][fs])
            total = int(gt[g])
            lines.append(f"| {g} | {rate:.2f}% | {cnt:,} | {total:,} |")

        lines += [
            "",
            f"#### Fisher 正確検定（S{fs}）",
            "",
            "| 比較 | 率A | 率B | p値 | 判定 |",
            "|-----|-----|-----|-----|------|",
        ]
        for label, fr in role["fisher_results"].items():
            p = fr["p"]
            p_str = (f"{p:.2e}" if not np.isnan(p) and p < 0.001
                     else (f"{p:.4f}" if not np.isnan(p) else "N/A"))
            verdict_f = "有意差なし" if (np.isnan(p) or p > 0.05) else "**有意差あり**"
            lines.append(f"| {label} | {fr['a_rate']:.2f}% | {fr['b_rate']:.2f}% | {p_str} | {verdict_f} |")

        # 解釈
        f1     = role["fisher_results"]["①confound検定 (B-start vs S-head)"]
        bstart = gr["B-start"][fs]
        shead  = gr["S-head"][fs]
        lines += ["", "#### 解釈", ""]
        if np.isnan(f1["p"]):
            lines.append("Fisher①の計算が不能（観測数 0）。")
        elif f1["p"] > 0.05:
            lines.append(f"**【交絡確認】** B-start ({bstart:.2f}%) ≈ S-head ({shead:.2f}%) (p={f1['p']:.4f})")
        elif bstart > shead:
            lines.append(f"**【構造効果あり】** B-start ({bstart:.2f}%) >> S-head ({shead:.2f}%) (p={f1['p']:.2e})")
        else:
            lines.append(f"**【交絡否定・語頭専用】** B-start ({bstart:.2f}%) << S-head ({shead:.2f}%) (p={f1['p']:.2e})")

        # s_{t-2} 文脈分析
        lines += [
            "",
            f"### B. s_{{t-2}} 文脈曖昧性解消（Focus State S{fs}）",
            "",
            f"Focus State S{fs} が登場する位置での s_{{t-2}} 分布：",
            f"B-end {ctx['n_bend_total']} 件 / B-start {ctx['n_bstart_total']} 件",
            "",
            "| s_{t-2} | B-end 率 | B-start 率 | Fisher p | 判定 |",
            "|---------|---------|-----------|---------|------|",
        ]
        any_sig = False
        for r in ctx["fisher"]:
            p = r["p"]
            p_str = f"{p:.2e}" if not np.isnan(p) else "N/A"
            sig = (not np.isnan(p) and p < 0.05)
            if sig:
                any_sig = True
            verdict_c = "**有意**" if sig else ""
            lines.append(
                f"| S{r['ctx_state']} | {r['rate_bend']:.1f}% | {r['rate_bstart']:.1f}% "
                f"| {p_str} | {verdict_c} |"
            )

        ctx_conclusion = (
            "**文脈分離あり**: Trigram の s_{t-2} が B-end/B-start 役割の一部を解消している。"
            if any_sig else
            "**文脈分離なし**: s_{t-2} では B-end/B-start の役割曖昧性を解消できなかった。"
        )
        lines += ["", f"**結論**: {ctx_conclusion}", "", "---", ""]

    lines += ["_本レポートは trigram_role_analysis_corrected.py により自動生成。_"]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("[正確版] Trigram HMM 役割分析 開始")

    # 正解データロード（Bug 1 修正）
    log(f"正解複合語ロード: {COMPOUND_SPLIT_PATH}")
    compound_splits_all = load_compound_splits(COMPOUND_SPLIT_PATH)
    log(f"  正解複合語数: {len(compound_splits_all):,} 語")

    log(f"正解単独ベース語ロード: {SINGLE_WORDS_PATH}")
    single_words_raw = load_word_list(SINGLE_WORDS_PATH)
    log(f"  正解単独ベース語数: {len(single_words_raw):,} 語")

    # DB から char2idx を構築
    log(f"DBロード: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    all_words_raw = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
        conn,
    )["word"].tolist()
    conn.close()

    all_types = sorted(set(w for w in all_words_raw if len(w) >= MIN_WORD_LEN))
    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR, EOS_CHAR, PAD_CHAR] + raw_chars
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    log(f"語彙サイズ: {len(all_chars)}")

    # フィルタ
    compound_splits = {
        w: splits for w, splits in compound_splits_all.items()
        if all(c in char2idx for c in w)
    }
    single_words = [
        w for w in single_words_raw
        if len(w) >= MIN_WORD_LEN and all(c in char2idx for c in w)
    ]
    log(f"有効複合語: {len(compound_splits):,} / 有効単独ベース語: {len(single_words):,}")

    results_by_k = {}

    for k in K_LIST:
        log(f"\n{'─'*60}")
        log(f"Trigram k={k}")
        log(f"{'─'*60}")

        model = load_trigram_model(k)
        if model is None:
            results_by_k[k] = None
            continue
        log(f"  モデルロード完了 (logL={model['logL']:.2f})")

        # Viterbi 占有率（全語）→ Phantom State 同定
        log("  Viterbi 占有率を計算中...")
        occupancy = compute_viterbi_occupancy(model, all_types, char2idx)
        for s in range(k):
            tag = "  ← Phantom" if occupancy[s] == 0.0 else ""
            log(f"    S{s}: {occupancy[s]:.3%}{tag}")

        # 複合語・単独語デコード
        log(f"  複合語デコード ({len(compound_splits):,} 語)...")
        decoded_compound = decode_words(compound_splits.keys(), char2idx, model)
        log(f"  完了: {len(decoded_compound):,} 語")

        log(f"  単独ベース語デコード ({len(single_words):,} 語)...")
        decoded_single = decode_words(single_words, char2idx, model)
        log(f"  完了: {len(decoded_single):,} 語")

        # 4グループ収集
        groups = collect_groups(compound_splits, single_words, decoded_compound, decoded_single)
        for g, states in groups.items():
            log(f"    {g}: {len(states):,} 観測")

        # 役割分析（Bug 5 修正: data-driven Focus State）
        role_result = run_role_analysis(groups, k, occupancy)
        if role_result is None:
            log("  ERROR: B-start の観測数が 0 または全状態が Phantom")
            results_by_k[k] = None
            continue

        fs = role_result["focus_state"]
        log(f"  Focus State: S{fs}  (data-driven、Phantom 除外)")
        for g in ["B-start", "B-end", "S-head", "S-mid"]:
            log(f"    {g}: {role_result['group_rates'][g][fs]:.2f}%")
        for label, fr in role_result["fisher_results"].items():
            if not np.isnan(fr["p"]):
                log(f"  Fisher {label}: p={fr['p']:.3e}")

        # s_{t-2} 文脈分析
        log("  s_{t-2} 文脈データ収集中...")
        ctx_df = collect_context_data(model, compound_splits, char2idx)
        log(f"  収集レコード数: {len(ctx_df):,}")

        ctx_result = analyze_context_disambiguation(ctx_df, fs, k)
        log(f"  Focus State S{fs} 登場: B-end={ctx_result['n_bend_total']}, "
            f"B-start={ctx_result['n_bstart_total']}")

        sig_ctx = sum(1 for r in ctx_result["fisher"]
                      if not np.isnan(r["p"]) and r["p"] < 0.05)
        log(f"  有意な s_{{t-2}} 差 (p<0.05): {sig_ctx}/{k} 状態")

        results_by_k[k] = {
            "role":      role_result,
            "context":   ctx_result,
            "occupancy": occupancy,
        }

        # 可視化
        plot_group_rates(role_result, k, OUT_DIR / f"role_group_k{k}.png")
        log(f"  Saved: role_group_k{k}.png")

        plot_context_distribution(ctx_result, fs, k, OUT_DIR / f"context_dist_k{k}.png")
        log(f"  Saved: context_dist_k{k}.png")

    # レポート生成
    log("\nMarkdown レポート生成中...")
    report = build_report(results_by_k, compound_splits, single_words)
    report_path = OUT_DIR / "role_analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"レポート保存: {report_path}")
    log("✓ 完了。出力先: " + str(OUT_DIR.resolve()))
