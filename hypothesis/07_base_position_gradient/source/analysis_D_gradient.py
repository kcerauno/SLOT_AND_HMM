"""
analysis_D_gradient.py
======================
提案 D: 基内位置勾配分析

【目的】
複合語の各基（base）内部で、語頭から語末に向けて状態分布がどう変化するかを分析し、
HMM 状態が「基内の位置的役割」を符号化しているかを検証する。

【方法】
1. 各基について正規化位置 pos = i / max(len(base)-1, 1) を計算
2. 3-bin 離散化: head (pos<1/3), middle (1/3<=pos<2/3), tail (pos>=2/3)
3. 絶対位置 (from_start: 0,1,2,3+) および (from_end: 0,1,2,3+) も集計
4. 線形回帰 (norm_pos → 各状態の二値出現) の傾きで勾配強度を定量化
5. 複合語の基 vs 単独語（語全体を1基として扱う）を比較

【正規化の設計（短い基への対応）】
- 長さ1基: pos=0.0 → head のみ
- 長さ2基: pos=0.0(head), 1.0(tail) → middle への寄与なし
- 長さ3+基: head/middle/tail すべて
- 絶対位置ではこの問題を回避（0文字目, 1文字目, ... / 末尾-0, 末尾-1, ...）

【事前仮説（提案 A 放射分析からの予測）】
- c/f 系状態 → 語頭集中（負の傾き）
- y/h 系状態 → 語末集中（正の傾き）
- Focus State (B-start) → head に集中（負傾き）
- B-end Focus State → tail に集中（正傾き）

【出力】
results/D_norm_heatmap_{model}_k{k}.png      — 3-bin 位置 × 状態 ヒートマップ（複合語 vs 単独語）
results/D_abs_heatmap_{model}_k{k}.png       — 絶対位置 × 状態 ヒートマップ（from_start / from_end）
results/D_focus_gradient_{model}_k{k}.png   — Focus / B-end Focus 位置分布折れ線
results/D_all_states_{model}_k{k}.png       — 全状態の位置分布グリッド + 回帰情報
results/D_regression_summary.png            — 全モデルの回帰傾き棒グラフまとめ
results/analysis_D_gradient_report.md       — 定量的結果レポート

【実行】
cd /home/practi/work_voy
PYTHONPATH=/home/practi/work_voy/.venv/lib/python3.10/site-packages \\
  python3.10 hypothesis/07_base_position_gradient/source/analysis_D_gradient.py
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
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ─── パス設定 ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent  # /home/practi/work_voy

# hypothesis/06 の共通ユーティリティを再利用
sys.path.insert(0, str(_ROOT / "hypothesis/06_state_characterization/source"))
from common import (
    log,
    load_bigram_model, load_trigram_model,
    load_compound_splits, load_single_words,
    decode_words,
    FOCUS_STATES, PHANTOM_STATES,
)

OUT_DIR = _ROOT / "hypothesis/07_base_position_gradient/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# B-end Focus States（hypothesis/06 提案 B の結果）
BEND_FOCUS_STATES = {
    ("bigram",  7): 0,   # B-end Focus: S0
    ("bigram",  8): 5,   # B-end Focus: S5
    ("trigram", 7): 5,   # B-end Focus: S5
    ("trigram", 8): 2,   # B-end Focus: S2
}

# ─── 定数 ──────────────────────────────────────────────────────────────────
N_BINS = 3
BIN_NAMES = ["head\n(0–1/3)", "middle\n(1/3–2/3)", "tail\n(2/3–1)"]
ABS_MAX = 4     # 絶対位置 0,1,2,3+ のバケット数


def pos_to_bin(pos: float) -> int:
    if pos < 1 / 3:
        return 0
    elif pos < 2 / 3:
        return 1
    else:
        return 2


# ════════════════════════════════════════════════════════════════════════════
# データ収集
# ════════════════════════════════════════════════════════════════════════════

def collect_base_records_compound(compound_splits: dict, decoded: dict) -> list:
    """
    複合語の各基について各文字位置の record を返す。
    record = (state: int, norm_pos: float, abs_from_start: int, abs_from_end: int, base_len: int)
    """
    records = []
    for word, splits in compound_splits.items():
        states = decoded.get(word)
        if states is None:
            continue
        char_offset = 0
        for base in splits:
            blen = len(base)
            base_states = states[char_offset: char_offset + blen]
            if len(base_states) != blen:
                char_offset += blen
                continue
            for i, state in enumerate(base_states):
                norm_pos = i / max(blen - 1, 1)
                records.append((
                    int(state),
                    float(norm_pos),
                    min(i, ABS_MAX - 1),
                    min(blen - 1 - i, ABS_MAX - 1),
                    blen,
                ))
            char_offset += blen
    return records


def collect_base_records_single(single_words: list, decoded: dict) -> list:
    """単独語を 1 基として扱い同じ形式で records を返す。"""
    records = []
    for word in single_words:
        states = decoded.get(word)
        if states is None:
            continue
        wlen = len(states)
        for i, state in enumerate(states):
            norm_pos = i / max(wlen - 1, 1)
            records.append((
                int(state),
                float(norm_pos),
                min(i, ABS_MAX - 1),
                min(wlen - 1 - i, ABS_MAX - 1),
                wlen,
            ))
    return records


def build_norm_matrix(records: list, k: int) -> np.ndarray:
    """records → (k, N_BINS) 各ビン内での状態分布（列ごと正規化）"""
    mat = np.zeros((k, N_BINS), dtype=float)
    for state, norm_pos, *_ in records:
        mat[int(state), pos_to_bin(norm_pos)] += 1
    col_sums = mat.sum(axis=0)
    col_sums[col_sums == 0] = 1
    return mat / col_sums


def build_abs_matrix(records: list, k: int, from_end: bool = False) -> np.ndarray:
    """records → (k, ABS_MAX) 絶対位置ごとの状態分布（列ごと正規化）"""
    mat = np.zeros((k, ABS_MAX), dtype=float)
    for rec in records:
        state, _, abs_start, abs_end, _ = rec
        pos = abs_end if from_end else abs_start
        mat[int(state), pos] += 1
    col_sums = mat.sum(axis=0)
    col_sums[col_sums == 0] = 1
    return mat / col_sums


def compute_regressions(records: list, k: int) -> dict:
    """
    各状態について norm_pos → 二値出現 (0/1) の線形回帰。
    returns: {state: (slope, intercept, r2, p_value)}
    """
    if not records:
        return {s: (0.0, 0.0, 0.0, 1.0) for s in range(k)}
    norm_pos_arr = np.array([r[1] for r in records])
    state_arr    = np.array([r[0] for r in records])
    results = {}
    for s in range(k):
        y = (state_arr == s).astype(float)
        if y.sum() < 5:
            results[s] = (0.0, 0.0, 0.0, 1.0)
            continue
        sl, ic, r_val, p_val, _ = stats.linregress(norm_pos_arr, y)
        results[s] = (float(sl), float(ic), float(r_val ** 2), float(p_val))
    return results


# ════════════════════════════════════════════════════════════════════════════
# 可視化
# ════════════════════════════════════════════════════════════════════════════

def _highlight_rows(ax, k, focus_s, bend_s, phantom_list, n_cols):
    for s in phantom_list:
        ax.add_patch(plt.Rectangle((-0.5, s - 0.5), n_cols, 1,
                                   fill=True, facecolor="gray", alpha=0.25))
    for s, color in [(focus_s, "red"), (bend_s, "orange")]:
        if s is not None:
            ax.add_patch(plt.Rectangle((-0.5, s - 0.5), n_cols, 1,
                                       fill=False, edgecolor=color, linewidth=2.5))


def plot_norm_heatmap(mat_c, mat_s, model_type, k, focus_s, bend_s, phantom_list):
    """3-bin 正規化位置 × 状態 ヒートマップ（複合語 vs 単独語）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, k * 0.7 + 2)))
    vmax = max(mat_c.max(), mat_s.max())

    for ax, mat, title in zip(axes, [mat_c, mat_s],
                               [f"Compound Bases", f"Single Words"]):
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(N_BINS))
        ax.set_xticklabels(BIN_NAMES, fontsize=9)
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"S{s}" for s in range(k)], fontsize=9)
        ax.set_xlabel("Normalized position bin")
        ax.set_ylabel("State")
        ax.set_title(f"{model_type.capitalize()} k={k}: {title}")
        plt.colorbar(im, ax=ax, label="Proportion within bin")
        _highlight_rows(ax, k, focus_s, bend_s, phantom_list, N_BINS)
        for i in range(k):
            for j in range(N_BINS):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.28 else "black")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor="none", edgecolor="red", linewidth=2, label="B-start Focus"),
              Patch(facecolor="none", edgecolor="orange", linewidth=2, label="B-end Focus"),
              Patch(facecolor="gray", alpha=0.4, label="Phantom")]
    axes[0].legend(handles=legend, loc="upper right", fontsize=7)
    plt.tight_layout()
    path = OUT_DIR / f"D_norm_heatmap_{model_type}_k{k}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {path.name}")


def plot_abs_heatmap(mat_start_c, mat_end_c, mat_start_s, mat_end_s,
                     model_type, k, focus_s, bend_s, phantom_list):
    """絶対位置（from_start / from_end）× 状態 ヒートマップ 2×2"""
    fig, axes = plt.subplots(2, 2, figsize=(14, max(8, k * 0.9 + 2)))
    labels_start = [f"pos+{i}" if i < ABS_MAX - 1 else f"pos+{i}+" for i in range(ABS_MAX)]
    labels_end   = [f"end-{i}" if i < ABS_MAX - 1 else f"end-{i}+" for i in range(ABS_MAX)]

    configs = [
        (axes[0, 0], mat_start_c, "Compound: from start", labels_start, "Purples"),
        (axes[0, 1], mat_end_c,   "Compound: from end",   labels_end,   "Oranges"),
        (axes[1, 0], mat_start_s, "Single:   from start", labels_start, "Purples"),
        (axes[1, 1], mat_end_s,   "Single:   from end",   labels_end,   "Oranges"),
    ]
    for ax, mat, title, xlabels, cmap in configs:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=mat.max())
        ax.set_xticks(range(ABS_MAX))
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"S{s}" for s in range(k)], fontsize=9)
        ax.set_title(f"{model_type.capitalize()} k={k}: {title}", fontsize=10)
        plt.colorbar(im, ax=ax, label="Proportion")
        _highlight_rows(ax, k, focus_s, bend_s, phantom_list, ABS_MAX)

    plt.suptitle(f"{model_type.capitalize()} k={k}: Absolute Position Gradient", fontsize=12)
    plt.tight_layout()
    path = OUT_DIR / f"D_abs_heatmap_{model_type}_k{k}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {path.name}")


def plot_focus_gradient(records_c, records_s, model_type, k, focus_s, bend_s, phantom_list):
    """Focus State / B-end Focus の位置分布をビン別折れ線グラフ（複合語 vs 単独語）"""
    def bin_occupancy(records, state):
        bin_cnt = np.zeros(N_BINS)
        bin_tot = np.zeros(N_BINS)
        for rec_state, norm_pos, *_ in records:
            b = pos_to_bin(norm_pos)
            bin_tot[b] += 1
            if rec_state == state:
                bin_cnt[b] += 1
        return bin_cnt / np.maximum(bin_tot, 1)

    targets = [(s, label, color)
               for s, label, color in [(focus_s, "B-start Focus", "red"),
                                       (bend_s,  "B-end Focus",   "orange")]
               if s is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(N_BINS)
    for s, label, color in targets:
        rates_c = bin_occupancy(records_c, s)
        rates_s = bin_occupancy(records_s, s)
        ax.plot(x, rates_c, "o-", color=color, linewidth=2.5, markersize=9,
                label=f"S{s} ({label}) – Compound")
        ax.plot(x, rates_s, "s--", color=color, linewidth=1.5, markersize=7, alpha=0.5,
                label=f"S{s} ({label}) – Single")

    ax.set_xticks(x)
    ax.set_xticklabels(["head\n(0–1/3)", "middle\n(1/3–2/3)", "tail\n(2/3–1)"], fontsize=10)
    ax.set_xlabel("Normalized position bin (within base)")
    ax.set_ylabel("State occupancy rate")
    ax.set_title(f"{model_type.capitalize()} k={k}: Focus State Position Gradient\n"
                 "(solid=compound bases, dashed=single words)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = OUT_DIR / f"D_focus_gradient_{model_type}_k{k}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {path.name}")


def plot_all_states(records_c, records_s, model_type, k,
                    focus_s, bend_s, phantom_list, reg_c, reg_s):
    """全状態のビン別位置分布グリッド（複合語 vs 単独語、回帰情報付き）"""
    def get_bin_rates(records):
        bin_tot = np.zeros(N_BINS)
        state_bin = np.zeros((k, N_BINS))
        for state, norm_pos, *_ in records:
            b = pos_to_bin(norm_pos)
            bin_tot[b] += 1
            state_bin[int(state), b] += 1
        return state_bin / np.maximum(bin_tot, 1)

    rates_c = get_bin_rates(records_c)
    rates_s = get_bin_rates(records_s)

    ncols = 4
    nrows = (k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()
    x = np.arange(N_BINS)

    for s in range(k):
        ax = axes[s]
        is_focus   = (s == focus_s)
        is_bend    = (s == bend_s)
        is_phantom = (s in phantom_list)
        bg = ("#ffe0e0" if is_focus else "#fff0d0" if is_bend else
              "lightgray" if is_phantom else "white")
        ax.set_facecolor(bg)

        ax.plot(x, rates_c[s], "o-", color="steelblue", linewidth=2,
                markersize=7, label="Compound")
        ax.plot(x, rates_s[s], "s--", color="forestgreen", linewidth=1.5,
                markersize=6, alpha=0.7, label="Single")

        sl_c, ic_c, r2_c, p_c = reg_c[s]
        sl_s, ic_s, r2_s, p_s = reg_s[s]
        # 回帰直線を head-tail の範囲で描画（x軸は bin index 0→2）
        x_fit = np.array([0.0, N_BINS - 1])
        ax.plot(x_fit, [ic_c, ic_c + sl_c], ":", color="steelblue", alpha=0.6, linewidth=1.2)
        ax.plot(x_fit, [ic_s, ic_s + sl_s], ":", color="forestgreen", alpha=0.6, linewidth=1.2)

        sig = "***" if p_c < 0.001 else "**" if p_c < 0.01 else "*" if p_c < 0.05 else ""
        extra = ""
        if is_focus:
            extra = " ★B-start"
        elif is_bend:
            extra = " ◆B-end"
        elif is_phantom:
            extra = " (phantom)"
        ax.set_title(f"S{s}{extra}\nslope={sl_c:+.4f} {sig}", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["H", "M", "T"], fontsize=8)
        ax.set_ylim(0, None)
        if s == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for i in range(k, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"{model_type.capitalize()} k={k}: All States Position Gradient\n"
                 "(H=head, M=middle, T=tail | blue solid=compound, green dashed=single | "
                 "dotted=regression)", fontsize=10)
    plt.tight_layout()
    path = OUT_DIR / f"D_all_states_{model_type}_k{k}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {path.name}")


def plot_regression_summary(all_results):
    """全 4 モデルの回帰傾き棒グラフまとめ図"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (model_type, k, focus_s, bend_s, phantom_list, reg_c) in enumerate(all_results):
        ax = axes[idx]
        states = list(range(k))
        slopes = [reg_c[s][0] for s in states]
        p_vals = [reg_c[s][3] for s in states]
        colors = []
        for s in states:
            if s == focus_s:      colors.append("red")
            elif s == bend_s:     colors.append("orange")
            elif s in phantom_list: colors.append("gray")
            else:                 colors.append("steelblue")

        ax.bar(states, slopes, color=colors, alpha=0.85, edgecolor="white")
        for s, p, sl in zip(states, p_vals, slopes):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if sig:
                y = sl + (0.0005 if sl >= 0 else -0.0015)
                ax.text(s, y, sig, ha="center",
                        va="bottom" if sl >= 0 else "top", fontsize=9, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(states)
        ax.set_xticklabels([f"S{s}" for s in states], fontsize=9)
        ax.set_xlabel("State")
        ax.set_ylabel("Slope (pos → state occ.)")
        ax.set_title(f"{model_type.capitalize()} k={k}\n"
                     f"(red=B-start Focus, orange=B-end Focus, gray=phantom)", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor="red",      alpha=0.85, label="B-start Focus"),
              Patch(facecolor="orange",   alpha=0.85, label="B-end Focus"),
              Patch(facecolor="steelblue",alpha=0.85, label="Other"),
              Patch(facecolor="gray",     alpha=0.6,  label="Phantom")]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=True)
    plt.suptitle("Proposal D: Linear Regression Slopes (compound bases)\n"
                 "Positive = state concentrates at tail; Negative = concentrates at head",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = OUT_DIR / "D_regression_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {path.name}")


# ════════════════════════════════════════════════════════════════════════════
# レポート生成
# ════════════════════════════════════════════════════════════════════════════

def generate_report(all_results, all_records):
    def bin_rates_for(records, s):
        bc = np.zeros(N_BINS)
        bt = np.zeros(N_BINS)
        for rec_s, norm_pos, *_ in records:
            b = pos_to_bin(norm_pos)
            bt[b] += 1
            if rec_s == s:
                bc[b] += 1
        return bc / np.maximum(bt, 1)

    lines = [
        "# 提案 D: 基内位置勾配分析 結果レポート",
        "",
        f"作成日時: {datetime.now():%Y-%m-%d}",
        "根拠: [next_analysis_proposals.md](../../05_re-compound_hmm_trigram/results/next_analysis_proposals.md)",
        "実施: [hypothesis/07_base_position_gradient/](../)",
        "",
        "---",
        "",
        "## 分析設定",
        "",
        "- 正規化位置: `pos = i / max(len(base)-1, 1)`（0=語頭, 1=語末）",
        "- 3-bin 離散化: **head** (pos<1/3) / **middle** (1/3≤pos<2/3) / **tail** (pos≥2/3)",
        "- 統計: 線形回帰（連続 norm_pos → 各状態の二値出現）",
        "  - 正の傾き → 語末に向かって状態出現率が増加",
        "  - 負の傾き → 語頭に集中",
        "- 事前仮説: c/f 系状態は head 集中（負傾き）、y/h 系状態は tail 集中（正傾き）",
        "",
        "---",
        "",
    ]

    for (model_type, k, focus_s, bend_s, phantom_list, reg_c), (_, _, records_c, records_s) \
            in zip(all_results, all_records):

        lines += [
            f"## {model_type.capitalize()} k={k}",
            "",
            f"- B-start Focus State: **S{focus_s}**",
            f"- B-end Focus State: **S{bend_s}**",
            f"- Phantom States: {phantom_list}",
            f"- 観測トークン: 複合語基 = {len(records_c):,}, 単独語 = {len(records_s):,}",
            "",
            "### 状態別位置分布と回帰（複合語基）",
            "",
            "| State | Role | head | middle | tail | slope | p-value | sig |",
            "|-------|------|-----:|------:|-----:|------:|---------|-----|",
        ]

        for s in range(k):
            rates = bin_rates_for(records_c, s)
            sl, ic, r2, p = reg_c[s]
            role = ""
            if s == focus_s:      role = "**B-start Focus★**"
            elif s == bend_s:     role = "**B-end Focus◆**"
            elif s in phantom_list: role = "(phantom)"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            lines.append(
                f"| S{s} | {role} | {rates[0]:.3f} | {rates[1]:.3f} | {rates[2]:.3f}"
                f" | {sl:+.4f} | {p:.2e} | {sig} |"
            )

        lines += [
            "",
            "### 事前仮説との照合",
            "",
            "| 仮説 | 確認結果 |",
            "|-----|---------|",
            f"| Focus State (S{focus_s}) が head に集中（負傾き） | {'✓' if reg_c[focus_s][0] < 0 else '✗'} slope={reg_c[focus_s][0]:+.4f} |",
            f"| B-end Focus (S{bend_s}) が tail に集中（正傾き） | {'✓' if reg_c[bend_s][0] > 0 else '✗'} slope={reg_c[bend_s][0]:+.4f} |",
            "",
            "---",
            "",
        ]

    lines += [
        "## 総合評価",
        "",
        "（各モデル結果を踏まえた統合評価は以下を参照）",
        "",
        "### 確立されたこと / 未解決の問い",
        "",
        "（分析実施後に手動追記）",
        "",
        "---",
        "",
        "_本レポートは analysis_D_gradient.py により自動生成。_",
        "_使用スクリプト: [source/analysis_D_gradient.py](../source/analysis_D_gradient.py)_",
    ]

    path = OUT_DIR / "analysis_D_gradient_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"  Saved: {path.name}")


# ════════════════════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════════════════════

def run_one_model(model_type: str, k: int, compound_splits: dict, single_words: list):
    log(f"=== {model_type.upper()} k={k} ===")
    focus_s  = FOCUS_STATES[(model_type, k)]
    bend_s   = BEND_FOCUS_STATES[(model_type, k)]
    phantom  = PHANTOM_STATES[(model_type, k)]

    log("  Loading model...")
    model = load_trigram_model(k) if model_type == "trigram" else load_bigram_model(k)
    if model is None:
        log("  SKIP: model not found")
        return None, None

    log("  Decoding compound words...")
    decoded_c = decode_words(list(compound_splits.keys()), model, model_type)
    log(f"  Decoded: {len(decoded_c)}/{len(compound_splits)}")

    log("  Decoding single words...")
    decoded_s = decode_words(single_words, model, model_type)
    log(f"  Decoded: {len(decoded_s)}/{len(single_words)}")

    log("  Collecting position records...")
    records_c = collect_base_records_compound(compound_splits, decoded_c)
    records_s = collect_base_records_single(single_words, decoded_s)
    log(f"  Tokens — compound: {len(records_c):,}, single: {len(records_s):,}")

    log("  Building matrices...")
    mat_norm_c    = build_norm_matrix(records_c, k)
    mat_norm_s    = build_norm_matrix(records_s, k)
    mat_start_c   = build_abs_matrix(records_c, k, from_end=False)
    mat_end_c     = build_abs_matrix(records_c, k, from_end=True)
    mat_start_s   = build_abs_matrix(records_s, k, from_end=False)
    mat_end_s     = build_abs_matrix(records_s, k, from_end=True)

    log("  Computing regressions...")
    reg_c = compute_regressions(records_c, k)
    reg_s = compute_regressions(records_s, k)

    log("  Plotting...")
    plot_norm_heatmap(mat_norm_c, mat_norm_s, model_type, k, focus_s, bend_s, phantom)
    plot_abs_heatmap(mat_start_c, mat_end_c, mat_start_s, mat_end_s,
                     model_type, k, focus_s, bend_s, phantom)
    plot_focus_gradient(records_c, records_s, model_type, k, focus_s, bend_s, phantom)
    plot_all_states(records_c, records_s, model_type, k,
                    focus_s, bend_s, phantom, reg_c, reg_s)

    return (model_type, k, focus_s, bend_s, phantom, reg_c), \
           (model_type, k, records_c, records_s)


def main():
    log("=== Proposal D: Base-Internal Position Gradient Analysis ===")

    log("Loading data...")
    compound_splits = load_compound_splits()
    single_words    = load_single_words()
    log(f"  Compounds: {len(compound_splits)}, Singles: {len(single_words)}")

    # base length distribution（参考情報）
    base_lens = []
    for splits in compound_splits.values():
        base_lens.extend(len(b) for b in splits)
    from collections import Counter
    lc = Counter(base_lens)
    log(f"  Base length distribution: {dict(sorted(lc.items()))}")

    configs = [("trigram", 7), ("trigram", 8), ("bigram", 7), ("bigram", 8)]
    all_results, all_records = [], []

    for model_type, k in configs:
        result, rec_pair = run_one_model(model_type, k, compound_splits, single_words)
        if result is not None:
            all_results.append(result)
            all_records.append(rec_pair)

    if all_results:
        log("Plotting regression summary...")
        plot_regression_summary(all_results)

        log("Generating report...")
        generate_report(all_results, all_records)

    log("=== Done ===")


if __name__ == "__main__":
    main()
