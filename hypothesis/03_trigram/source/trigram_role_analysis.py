"""
trigram_role_analysis.py
========================
Trigram HMM による役割曖昧性解消の検証

研究仮説:
  Bigram HMM では Focus State (k=7: S4, k=8: 等価状態) が
  「基末尾 (B-end)」と「基先頭 (B-start)」の両役割を混在して担っていた。
  Trigram HMM は 2 ステップ前の文脈 s_{t-2} を参照できるため、
  この役割曖昧性を局所文脈から区別できるはず。

検証方法:
  1. Focus State を同定（B-end 集中率が最大の状態）
  2. Focus State の登場位置での (s_{t-2}) 分布を B-end vs B-start で比較
  3. Fisher 正確検定で統計的有意性を評価
  4. Bigram HMM と比較して役割分離の改善を定量化

参照: idea/03_trigram_implementation_plan.md
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from collections import Counter, defaultdict
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

from matplotlib import font_manager
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
DB_PATH         = "data/voynich.db"
TRIGRAM_CACHE   = Path("hypothesis/03_trigram/results/hmm_model_cache")
BIGRAM_CACHE    = Path("hypothesis/01_bigram/results/hmm_model_cache")
OUT_DIR         = Path("hypothesis/03_trigram/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

#K_LIST          = [7, 8]   # Bigram との比較対象
K_LIST          = [9, 10]   # Bigram との比較対象
MIN_WORD_LEN    = 2
BOS_CHAR, EOS_CHAR = "^", "$"


# ════════════════════════════════════════════════════════════════════════
# 1. V8 文法定義 (02_compound_hmm/source/compound_boundary_analysis.py から移植)
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


def find_v8_splits_first(word: str):
    """最初の有効な V8 複合語分割を返す（2基以上のみ）"""
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
    """境界位置を計算: B-end (基の最終文字) と B-start (基の先頭文字)"""
    boundary_end   = set()
    boundary_start = set()
    cumlen = 0
    for base in splits[:-1]:
        cumlen += len(base)
        boundary_end.add(cumlen - 1)
        boundary_start.add(cumlen)
    return boundary_end, boundary_start


# ════════════════════════════════════════════════════════════════════════
# 2. モデルロード・Viterbi
# ════════════════════════════════════════════════════════════════════════
def load_trigram_model(k):
    path = TRIGRAM_CACHE / f"trigram_k{k}.npz"
    if not path.exists():
        log(f"Trigram k={k} モデルが見つかりません: {path}")
        return None
    d = np.load(path)
    return {
        "start":       d["start"],
        "start_trans": d["start_trans"],
        "trans":       d["trans"],
        "emiss":       d["emiss"],
        "logL":        float(d["logL"][0]),
        "type":        "trigram",
    }


def load_bigram_model(k):
    path = BIGRAM_CACHE / f"full_k{k}.npz"
    if not path.exists():
        log(f"Bigram k={k} モデルが見つかりません: {path}")
        return None
    d = np.load(path)
    return {
        "start": d["start"],
        "trans": d["trans"],
        "emiss": d["emiss"],
        "logL":  float(d["logL"][0]),
        "type":  "bigram",
    }


def viterbi_bigram(model, X_np):
    """Bigram HMM の Viterbi デコード"""
    log_start = torch.tensor(np.log(model["start"] + 1e-35), device=DEVICE)
    log_trans  = torch.tensor(np.log(model["trans"]  + 1e-35), device=DEVICE)
    log_emiss  = torch.tensor(np.log(model["emiss"]  + 1e-35), device=DEVICE)
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


def viterbi_trigram(model, X_np):
    """Trigram HMM の Viterbi デコード"""
    log_start      = torch.tensor(np.log(model["start"]       + 1e-35), device=DEVICE)
    log_start_trans = torch.tensor(np.log(model["start_trans"] + 1e-35), device=DEVICE)
    log_transmat   = torch.tensor(np.log(model["trans"]        + 1e-35), device=DEVICE)
    log_emiss      = torch.tensor(np.log(model["emiss"]        + 1e-35), device=DEVICE)

    X = torch.tensor(X_np, dtype=torch.long, device=DEVICE)
    T = X.shape[0]
    k = log_start.shape[0]
    d = DEVICE

    if T == 0:
        return []

    # t=0
    log_delta_1d = log_start + log_emiss[:, X[0]]  # (k,)

    if T == 1:
        return [torch.argmax(log_delta_1d).item()]

    # t=1: (k, k)
    log_delta = log_delta_1d.unsqueeze(1) + log_start_trans  # (k, k)
    log_delta = log_delta + log_emiss[:, X[1]].unsqueeze(0)  # (k, k)

    psi_list = []  # psi_list[t-2][j, l] = argmax_i at t

    for t in range(2, T):
        vals = log_delta.unsqueeze(2) + log_transmat  # (k, k, k)
        max_vals, argmax_i = torch.max(vals, dim=0)   # (k, k), (k, k) = (j, l)
        new_delta = max_vals + log_emiss[:, X[t]].unsqueeze(0)  # (k, k)
        psi_list.append(argmax_i.cpu())
        log_delta = new_delta

    # バックトラック
    flat_idx = torch.argmax(log_delta)
    j_last = (flat_idx // k).item()
    l_last = (flat_idx % k).item()

    path = [0] * T
    path[T - 1] = l_last
    path[T - 2] = j_last

    for t_back in range(T - 2, 1, -1):
        i_prev = psi_list[t_back - 2][path[t_back], path[t_back + 1]].item()
        path[t_back - 1] = i_prev

    return path


# ════════════════════════════════════════════════════════════════════════
# 3. 境界位置の状態列・前文脈の収集
# ════════════════════════════════════════════════════════════════════════
def collect_context_data(model, compound_splits, char2idx, k):
    """
    全複合語について Viterbi デコードを行い、
    各位置の (s_{t-2}, s_{t-1}, s_t) と境界ラベルを収集。

    Returns
    -------
    records: list of dict with keys:
      word, pos, s_t, s_t1 (s_{t-1}), s_t2 (s_{t-2}), label
      label ∈ {'B-end', 'B-start', 'inner'}
    """
    is_trigram = (model["type"] == "trigram")
    records = []

    for word, splits in compound_splits.items():
        seq = np.array(
            [char2idx[BOS_CHAR]] + [char2idx[c] for c in word] + [char2idx[EOS_CHAR]],
            dtype=np.int32,
        )
        try:
            if is_trigram:
                path = viterbi_trigram(model, seq)
            else:
                path = viterbi_bigram(model, seq)
        except Exception:
            continue

        states = path[1:-1]   # BOS/EOS を除く（長さ = len(word)）
        if len(states) < 2:
            continue

        bd_end, bd_start = get_boundary_positions(splits)

        for pos in range(len(states)):
            s_t = int(states[pos])
            s_t1 = int(states[pos - 1]) if pos >= 1 else -1   # s_{t-1}
            s_t2 = int(states[pos - 2]) if pos >= 2 else -1   # s_{t-2}

            if pos in bd_end:
                label = "B-end"
            elif pos in bd_start:
                label = "B-start"
            else:
                label = "inner"

            records.append({
                "word": word,
                "pos": pos,
                "s_t": s_t,
                "s_t1": s_t1,
                "s_t2": s_t2,
                "label": label,
            })

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════
# 4. Focus State 同定
# ════════════════════════════════════════════════════════════════════════
def identify_focus_state(df, k):
    """B-end 出現率が最高の状態を Focus State として同定"""
    bend_df  = df[df["label"] == "B-end"]
    inner_df = df[df["label"] == "inner"]

    state_stats = []
    for s in range(k):
        n_bend  = (bend_df["s_t"]  == s).sum()
        n_inner = (inner_df["s_t"] == s).sum()
        total_bend  = len(bend_df)
        total_inner = len(inner_df)
        rate_bend  = n_bend  / total_bend  if total_bend  > 0 else 0.0
        rate_inner = n_inner / total_inner if total_inner > 0 else 0.0
        state_stats.append({
            "state": s,
            "n_bend": n_bend, "n_inner": n_inner,
            "rate_bend": rate_bend, "rate_inner": rate_inner,
            "enrichment": rate_bend / rate_inner if rate_inner > 0 else np.nan,
        })

    stats_df = pd.DataFrame(state_stats)
    focus = int(stats_df.sort_values("rate_bend", ascending=False).iloc[0]["state"])
    return focus, stats_df


# ════════════════════════════════════════════════════════════════════════
# 5. 役割曖昧性の検定
# ════════════════════════════════════════════════════════════════════════
def analyze_role_ambiguity(df, focus_state, k, model_type):
    """
    Focus State が登場する位置での s_{t-2} 分布を B-end vs B-start で比較。

    Returns
    -------
    dict with Fisher test results and context distributions
    """
    fs_df = df[df["s_t"] == focus_state].copy()
    bend_fs  = fs_df[fs_df["label"] == "B-end"]
    bstart_fs = fs_df[fs_df["label"] == "B-start"]

    result = {
        "n_bend": len(bend_fs),
        "n_bstart": len(bstart_fs),
        "fisher_results": [],
        "context_dist_bend":   {},
        "context_dist_bstart": {},
    }

    if model_type == "trigram":
        context_col = "s_t2"  # s_{t-2} が Trigram の核心的な文脈
    else:
        context_col = "s_t1"  # Bigram では s_{t-1} が最大の文脈

    for ctx_state in range(k):
        if ctx_state == -1:
            continue
        n_bend_ctx   = (bend_fs[context_col]  == ctx_state).sum()
        n_bstart_ctx = (bstart_fs[context_col] == ctx_state).sum()
        n_bend_other   = len(bend_fs)   - n_bend_ctx
        n_bstart_other = len(bstart_fs) - n_bstart_ctx

        table = [[n_bend_ctx, n_bend_other],
                 [n_bstart_ctx, n_bstart_other]]

        if n_bend_ctx + n_bstart_ctx > 0:
            odds, p = stats.fisher_exact(table, alternative="two-sided")
        else:
            odds, p = np.nan, np.nan

        result["fisher_results"].append({
            "context_state": ctx_state,
            "n_bend_ctx": n_bend_ctx,
            "n_bstart_ctx": n_bstart_ctx,
            "rate_bend": n_bend_ctx / len(bend_fs) if len(bend_fs) > 0 else 0.0,
            "rate_bstart": n_bstart_ctx / len(bstart_fs) if len(bstart_fs) > 0 else 0.0,
            "odds_ratio": odds,
            "p_value": p,
        })

    # 分布
    if model_type == "trigram":
        result["context_dist_bend"]   = dict(Counter(bend_fs[context_col].tolist()))
        result["context_dist_bstart"] = dict(Counter(bstart_fs[context_col].tolist()))
    else:
        result["context_dist_bend"]   = dict(Counter(bend_fs[context_col].tolist()))
        result["context_dist_bstart"] = dict(Counter(bstart_fs[context_col].tolist()))

    return result


# ════════════════════════════════════════════════════════════════════════
# 6. 可視化
# ════════════════════════════════════════════════════════════════════════
def plot_role_context(ambig_result, focus_state, k, model_type, model_label, out_path):
    """
    Focus State での前文脈 (s_{t-2} for trigram, s_{t-1} for bigram) 分布の
    B-end vs B-start 比較棒グラフ
    """
    state_labels = [f"S{s}" for s in range(k)]
    bend_dist   = ambig_result["context_dist_bend"]
    bstart_dist = ambig_result["context_dist_bstart"]

    total_bend   = sum(bend_dist.values()) or 1
    total_bstart = sum(bstart_dist.values()) or 1

    bend_rates   = [bend_dist.get(s, 0) / total_bend   * 100 for s in range(k)]
    bstart_rates = [bstart_dist.get(s, 0) / total_bstart * 100 for s in range(k)]

    x = np.arange(k)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, k * 1.3), 5))
    ax.bar(x - width / 2, bend_rates,   width, label="B-end",   color="#4878CF", alpha=0.85)
    ax.bar(x + width / 2, bstart_rates, width, label="B-start", color="#F0814E", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(state_labels)
    ctx_label = "s_{t-2}" if model_type == "trigram" else "s_{t-1}"
    ax.set_xlabel(f"前文脈状態 ({ctx_label})", fontsize=10)
    ax.set_ylabel("出現率 (%)", fontsize=10)
    ax.set_title(
        f"{model_label}  k={k}  Focus State = S{focus_state}\n"
        f"Focus State 登場位置での {ctx_label} 分布 (B-end vs B-start)",
        fontsize=11,
    )
    ax.legend()

    # Fisher 有意な状態をアスタリスクでマーク
    for r in ambig_result["fisher_results"]:
        s = r["context_state"]
        p = r["p_value"]
        if not np.isnan(p) and p < 0.05:
            y_max = max(bend_rates[s], bstart_rates[s])
            ax.annotate("*", xy=(s, y_max + 1), ha="center", fontsize=14, color="red")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_focus_state_transition(model, focus_state, k, model_label, out_path):
    """
    Trigram: A[:, :, focus_state] のヒートマップ (s_{t-2} × s_{t-1} → Focus State の確率)
    Bigram:  A[:, focus_state] のバーチャート (s_{t-1} → Focus State の確率)
    """
    if model["type"] == "trigram":
        trans = model["trans"]  # (k, k, k)
        mat = trans[:, :, focus_state]  # (k, k) = (s_{t-2}, s_{t-1})
        state_labels = [f"S{s}" for s in range(k)]
        df_mat = pd.DataFrame(mat, index=state_labels, columns=state_labels)

        fig, ax = plt.subplots(figsize=(max(5, k * 1.2), max(4, k * 1.0)))
        sns.heatmap(df_mat, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1,
                    ax=ax, linewidths=0.3, linecolor="gray",
                    cbar_kws={"shrink": 0.7}, annot_kws={"size": 8})
        ax.set_title(
            f"{model_label}  k={k}\n"
            f"A[s_{{t-2}}, s_{{t-1}}, S{focus_state}]: Focus State への遷移確率",
            fontsize=11,
        )
        ax.set_xlabel("s_{t-1}", fontsize=9)
        ax.set_ylabel("s_{t-2}", fontsize=9)
    else:
        trans = model["trans"]  # (k, k)
        vals = trans[:, focus_state]  # (k,) = (s_{t-1},)
        state_labels = [f"S{s}" for s in range(k)]

        fig, ax = plt.subplots(figsize=(max(6, k * 1.0), 4))
        ax.bar(np.arange(k), vals, color="steelblue", alpha=0.85)
        ax.set_xticks(np.arange(k))
        ax.set_xticklabels(state_labels)
        ax.set_xlabel("s_{t-1}", fontsize=9)
        ax.set_ylabel(f"P(S{focus_state} | s_{{t-1}})", fontsize=9)
        ax.set_title(
            f"{model_label}  k={k}\n"
            f"Bigram: Focus State S{focus_state} への遷移確率",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_state_boundary_rates(stats_df, k, focus_state, model_label, out_path):
    """全状態の B-end 出現率を棒グラフ表示（Focus State を強調）"""
    state_labels = [f"S{s}" for s in range(k)]
    colors = ["#E84040" if s == focus_state else "#5588BB" for s in range(k)]

    fig, ax = plt.subplots(figsize=(max(7, k * 1.2), 4))
    ax.bar(np.arange(k), stats_df.sort_values("state")["rate_bend"] * 100,
           color=colors, alpha=0.85)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(state_labels)
    ax.set_xlabel("隠れ状態", fontsize=9)
    ax.set_ylabel("B-end 出現率 (%)", fontsize=9)
    ax.set_title(
        f"{model_label}  k={k}  各状態の B-end 出現率\n"
        f"（赤 = Focus State S{focus_state}）",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# 7. レポート生成
# ════════════════════════════════════════════════════════════════════════
def build_role_analysis_report(all_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Trigram HMM 役割曖昧性解消 分析レポート",
        f"",
        f"**生成日時**: {now}",
        f"**分析対象**: Trigram HMM vs Bigram HMM の Focus State 役割分離",
        f"",
        f"---",
        f"",
        f"## 研究仮説",
        f"",
        f"Bigram HMM では Focus State が B-end (基末尾) と B-start (基先頭) の",
        f"両役割を混在して担っていた（役割曖昧性）。",
        f"Trigram HMM は 2 ステップ前の文脈 s_{{t-2}} を参照できるため、",
        f"この曖昧性を局所文脈で区別できるはず。",
        f"",
        f"**判定基準**: B-end と B-start で s_{{t-2}} の分布が有意に異なるか (Fisher p < 0.05)",
        f"",
        f"---",
        f"",
    ]

    for res in all_results:
        model_label = res["model_label"]
        k = res["k"]
        focus = res["focus_state"]
        ambig = res["ambiguity"]
        stats_df = res["stats_df"]

        lines += [
            f"## {model_label}  k={k}",
            f"",
            f"### Focus State の同定",
            f"",
            f"| 状態 | B-end 出現率 | inner 出現率 | Enrichment |",
            f"|------|------------|------------|-----------|",
        ]
        for _, row in stats_df.sort_values("state").iterrows():
            marker = " ← **Focus State**" if int(row["state"]) == focus else ""
            lines.append(
                f"| S{int(row['state'])} | {row['rate_bend']:.1%} | {row['rate_inner']:.1%} "
                f"| {row['enrichment']:.2f} |{marker}"
            )

        ctx_label = "s_{t-2}" if "Trigram" in model_label else "s_{t-1}"
        lines += [
            f"",
            f"**同定された Focus State**: S{focus}",
            f"**分析対象文脈**: {ctx_label}",
            f"",
            f"### 役割曖昧性検定: Focus State S{focus} の前文脈分布",
            f"",
            f"B-end での件数: {ambig['n_bend']}  /  B-start での件数: {ambig['n_bstart']}",
            f"",
            f"| {ctx_label} | B-end 率 | B-start 率 | Odds Ratio | Fisher p |",
            f"|------------|---------|-----------|-----------|---------|",
        ]
        any_sig = False
        for fr in ambig["fisher_results"]:
            sig = "**" if (not np.isnan(fr["p_value"]) and fr["p_value"] < 0.05) else ""
            if sig:
                any_sig = True
            lines.append(
                f"| S{fr['context_state']} | {fr['rate_bend']:.1%} | {fr['rate_bstart']:.1%} "
                f"| {fr['odds_ratio']:.2f} | {sig}{fr['p_value']:.4f}{sig} |"
            )

        conclusion = (
            "**有意な文脈差あり**: Trigram HMM は局所文脈で役割曖昧性を部分的に解消。"
            if any_sig else
            "**有意な文脈差なし**: Trigram の局所文脈では役割曖昧性を解消できなかった。"
            " → Semi-Markov HMM (Step 2) への移行を推奨。"
        )
        lines += [
            f"",
            f"**結論**: {conclusion}",
            f"",
            f"---",
            f"",
        ]

    lines += [
        f"## 総合評価",
        f"",
        f"| モデル | k | Focus State | 有意な文脈分離 | 推奨 |",
        f"|--------|---|------------|-------------|------|",
    ]
    for res in all_results:
        ambig = res["ambiguity"]
        any_sig = any(
            not np.isnan(fr["p_value"]) and fr["p_value"] < 0.05
            for fr in ambig["fisher_results"]
        )
        rec = "Trigram で継続" if (any_sig and "Trigram" in res["model_label"]) else "HSMM へ"
        lines.append(
            f"| {res['model_label']} | {res['k']} | S{res['focus_state']} "
            f"| {'あり' if any_sig else 'なし'} | {rec} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"_参照: `idea/next_architecture_candidates.md` §4 推奨ロードマップ_",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("Loading data from DB...")
    conn = sqlite3.connect(DB_PATH)
    df_words = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
        conn,
    )
    conn.close()

    all_types = sorted(set(df_words["word"].tolist()))
    all_types = [w for w in all_types if len(w) >= MIN_WORD_LEN]
    log(f"ユニーク単語数: {len(all_types):,}")

    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR] + raw_chars + [EOS_CHAR]
    char2idx  = {c: i for i, c in enumerate(all_chars)}

    # V8 複合語分割
    log("V8 複合語分割を計算中...")
    compound_splits = {}
    for w in all_types:
        splits = find_v8_splits_first(w)
        if splits is not None:
            compound_splits[w] = splits

    log(f"複合語数: {len(compound_splits):,} / {len(all_types):,}")

    all_results = []

    for k in K_LIST:
        for model_type in ["trigram", "bigram"]:
            model_label = f"{'Trigram' if model_type == 'trigram' else 'Bigram'} HMM"

            log(f"\n{'─'*60}")
            log(f"{model_label}  k={k}")
            log(f"{'─'*60}")

            if model_type == "trigram":
                model = load_trigram_model(k)
            else:
                model = load_bigram_model(k)

            if model is None:
                continue

            k_actual = model["trans"].shape[-1]
            log(f"モデルロード完了  k={k_actual}  logL={model['logL']:.1f}")

            # 前文脈データ収集
            log("Viterbi デコード & 前文脈データ収集中...")
            df_ctx = collect_context_data(model, compound_splits, char2idx, k_actual)
            log(f"収集レコード数: {len(df_ctx):,}")

            # Focus State 同定
            focus_state, stats_df = identify_focus_state(df_ctx, k_actual)
            log(f"Focus State: S{focus_state}  B-end 率={stats_df[stats_df['state']==focus_state]['rate_bend'].values[0]:.1%}")

            # 役割曖昧性分析
            ambig = analyze_role_ambiguity(df_ctx, focus_state, k_actual, model_type)
            log(f"B-end件数={ambig['n_bend']}  B-start件数={ambig['n_bstart']}")

            # 有意な文脈差の件数を表示
            sig_count = sum(
                1 for fr in ambig["fisher_results"]
                if not np.isnan(fr["p_value"]) and fr["p_value"] < 0.05
            )
            log(f"有意な前文脈差 (p<0.05): {sig_count}/{k_actual} 状態")

            # 可視化
            suffix = f"{model_type}_k{k}"
            plot_role_context(
                ambig, focus_state, k_actual, model_type, model_label,
                OUT_DIR / f"role_context_{suffix}.png",
            )
            plot_focus_state_transition(
                model, focus_state, k_actual, model_label,
                OUT_DIR / f"focus_state_{suffix}.png",
            )
            plot_state_boundary_rates(
                stats_df, k_actual, focus_state, model_label,
                OUT_DIR / f"boundary_rates_{suffix}.png",
            )

            all_results.append({
                "model_label": model_label,
                "model_type": model_type,
                "k": k_actual,
                "focus_state": focus_state,
                "ambiguity": ambig,
                "stats_df": stats_df,
            })

    # レポート生成
    report = build_role_analysis_report(all_results)
    report_path = OUT_DIR / "role_analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"\nレポート保存: {report_path}")
    log("✓ 完了。出力先: " + str(OUT_DIR.resolve()))
