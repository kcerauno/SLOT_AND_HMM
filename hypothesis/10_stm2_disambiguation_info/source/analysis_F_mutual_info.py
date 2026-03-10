#!/usr/bin/env python3
"""
Proposal F: s_{t-2} 曖昧性解消の情報量定量化

Focus State が現れる位置（s_t = Focus）において:
1. 同時分布 P(s_{t-2}, label) を推定 (label ∈ {B-end, B-start})
2. 条件付き相互情報量 I(label; s_{t-2} | s_t = Focus) を計算
3. s_{t-2} あり/なしでの B-end/B-start 予測精度を比較
4. Bigram vs Trigram, k=7 vs k=8 の4条件で比較表を作成

注意: 境界位置は compound_words.txt の分割情報を直接使用する
      (first-fit ではなく greedy アルゴリズム準拠の正しい分割)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

# common.py を hypothesis/06 から再利用
H06_SOURCE = (
    Path(__file__).resolve().parent.parent.parent
    / "06_state_characterization" / "source"
)
sys.path.insert(0, str(H06_SOURCE))

from common import (
    log,
    load_compound_splits,
    load_bigram_model,
    load_trigram_model,
    decode_words,
    get_boundary_positions,
    FOCUS_STATES,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# コア集計
# ══════════════════════════════════════════════════════════════════════════════

def collect_focus_pairs(splits: dict, decoded: dict, focus_state: int) -> dict:
    """
    Focus State（s_t = focus_state）が現れる境界位置について、
    (s_{t-2}, label) ペアのカウント辞書を返す。

    label ∈ {"B-start", "B-end"}
    s_{t-2} は int（位置 i>=2 のとき）または None（語の先頭付近）。

    境界位置は splits（compound_words.txt 由来）のみ使用する。
    """
    counts = {}  # (s_prev2_or_None, label) -> int

    for word, bases in splits.items():
        if word not in decoded:
            continue
        states = decoded[word]
        bd_end, bd_start = get_boundary_positions(bases)

        for i, s in enumerate(states):
            if s != focus_state:
                continue

            if i in bd_start:
                label = "B-start"
            elif i in bd_end:
                label = "B-end"
            else:
                continue  # Focus State だが境界位置でない → スキップ

            s_prev2 = states[i - 2] if i >= 2 else None
            key = (s_prev2, label)
            counts[key] = counts.get(key, 0) + 1

    return counts


# ══════════════════════════════════════════════════════════════════════════════
# 情報量計算
# ══════════════════════════════════════════════════════════════════════════════

def _entropy_bits(probs: np.ndarray) -> float:
    """Shannon エントロピー（ビット単位）。ゼロは無視。"""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_information_measures(counts: dict) -> dict:
    """
    counts: {(s_prev2_or_None, label): int}

    返り値（辞書）:
      total          - Focus 境界出現総数
      n_bstart       - B-start 件数
      n_bend         - B-end 件数
      h_label        - H(label | s_t=Focus) [bits]
      h_label_given_s - H(label | s_{t-2}, s_t=Focus) [bits]
      mi             - I(label; s_{t-2} | s_t=Focus) [bits]
      acc_without    - 多数決予測精度（s_{t-2} なし）
      acc_with       - per-s_{t-2} 予測精度（s_{t-2} あり）
      n_with_prev2   - s_{t-2} が存在するサンプル数
      by_s           - {s_prev2: {"B-start": n, "B-end": n}} （詳細）
    """
    result = {}

    # ─ ラベル周辺分布 ────────────────────────────────────────────────────────
    total_label = {"B-start": 0, "B-end": 0}
    for (s, lbl), c in counts.items():
        total_label[lbl] = total_label.get(lbl, 0) + c

    total = sum(total_label.values())
    result["total"]    = total
    result["n_bstart"] = total_label.get("B-start", 0)
    result["n_bend"]   = total_label.get("B-end", 0)

    if total == 0:
        result.update({"h_label": 0.0, "h_label_given_s": 0.0, "mi": 0.0,
                        "acc_without": 0.0, "acc_with": 0.0,
                        "n_with_prev2": 0, "by_s": {}})
        return result

    # ─ H(label | s_t=Focus) ──────────────────────────────────────────────────
    p_label = np.array([total_label.get(l, 0) for l in ["B-start", "B-end"]],
                       dtype=float) / total
    h_label = _entropy_bits(p_label)
    result["h_label"] = h_label

    # ─ s_{t-2} が既知のサンプルだけ使って H(label | s_{t-2}) を計算 ──────────
    known = {(s, lbl): c for (s, lbl), c in counts.items() if s is not None}
    total_known = sum(known.values())
    result["n_with_prev2"] = total_known

    # by_s: {s_prev2: {"B-start": n, "B-end": n}}
    by_s: dict[int, dict[str, int]] = {}
    for (s, lbl), c in known.items():
        by_s.setdefault(s, {"B-start": 0, "B-end": 0})[lbl] += c
    result["by_s"] = by_s

    if total_known == 0:
        result.update({"h_label_given_s": h_label, "mi": 0.0,
                        "acc_without": max(p_label), "acc_with": max(p_label)})
        return result

    # H(label | s_{t-2}, s_t=Focus) = Σ_s P(s) H(label|s)
    h_label_given_s = 0.0
    for s, lbl_cnt in by_s.items():
        n_s = sum(lbl_cnt.values())
        p_s = n_s / total_known
        p_l_given_s = np.array([lbl_cnt.get(l, 0) for l in ["B-start", "B-end"]],
                                dtype=float) / n_s
        h_label_given_s += p_s * _entropy_bits(p_l_given_s)

    mi = max(h_label - h_label_given_s, 0.0)  # 数値誤差で負にならないよう
    result["h_label_given_s"] = h_label_given_s
    result["mi"] = mi

    # ─ 予測精度 ──────────────────────────────────────────────────────────────
    # s_{t-2} なし: 多数決（全体の多数クラスを常に予測）
    majority_n = max(total_label.values())
    result["acc_without"] = majority_n / total

    # s_{t-2} あり: s ごとに多数クラスで予測（known サンプルのみ）
    correct_with = sum(max(lbl_cnt.values()) for lbl_cnt in by_s.values())
    result["acc_with"] = correct_with / total_known

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════════════════════════════════════

def plot_joint_heatmap(by_s: dict, k: int, focus_state: int,
                       model_type: str, k_val: int, out_path: Path) -> None:
    """by_s を s_{t-2} × label のヒートマップとして表示。"""
    if not by_s:
        return

    states = list(range(k))
    labels = ["B-start", "B-end"]

    mat = np.zeros((k, 2), dtype=float)
    for s, lbl_cnt in by_s.items():
        for j, lbl in enumerate(labels):
            mat[s, j] = lbl_cnt.get(lbl, 0)

    # 行和で正規化 → P(label | s_{t-2}, s_t=Focus)
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = np.where(row_sums > 0, mat / row_sums, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.6 * k + 2))

    # カウント
    im0 = axes[0].imshow(mat, aspect="auto", cmap="Blues")
    axes[0].set_title(f"Count: P(label, s_{{t-2}} | s_t=S{focus_state})")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_yticks(states); axes[0].set_yticklabels([f"S{s}" for s in states])
    axes[0].set_ylabel("s_{t-2}")
    for s in states:
        for j in range(2):
            if mat[s, j] > 0:
                axes[0].text(j, s, f"{int(mat[s,j])}", ha="center", va="center",
                             fontsize=8, color="black")
    plt.colorbar(im0, ax=axes[0])

    # 条件付き確率
    im1 = axes[1].imshow(mat_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title(f"P(label | s_{{t-2}}, s_t=S{focus_state})")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_yticks(states); axes[1].set_yticklabels([f"S{s}" for s in states])
    axes[1].set_ylabel("s_{t-2}")
    for s in states:
        for j in range(2):
            if row_sums[s, 0] > 0:
                axes[1].text(j, s, f"{mat_norm[s,j]:.2f}", ha="center", va="center",
                             fontsize=8, color="black")
    plt.colorbar(im1, ax=axes[1])

    plt.suptitle(f"{model_type} k={k_val}  Focus State S{focus_state} — Joint Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {out_path.name}")


def plot_accuracy_comparison(rows: list, out_path: Path) -> None:
    """4条件の予測精度比較棒グラフ。rows: list of dict."""
    labels_x = [f"{r['model_type']}\nk={r['k']}" for r in rows]
    acc_wo = [r["acc_without"] * 100 for r in rows]
    acc_wi = [r["acc_with"]    * 100 for r in rows]

    x = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, acc_wo, width, label="Without s_{t-2} (majority)")
    bars2 = ax.bar(x + width / 2, acc_wi, width, label="With s_{t-2} (per-state)")

    ax.set_ylabel("Prediction Accuracy (%)")
    ax.set_title("B-end / B-start Prediction Accuracy at Focus State Positions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.axhline(50, linestyle="--", color="gray", linewidth=0.8, label="Chance")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {out_path.name}")


def plot_mi_comparison(rows: list, out_path: Path) -> None:
    """4条件の相互情報量比較。"""
    labels_x = [f"{r['model_type']}\nk={r['k']}" for r in rows]
    mi_vals    = [r["mi"]    for r in rows]
    h_vals     = [r["h_label"] for r in rows]

    x = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, h_vals, width, label="H(label | s_t=Focus) [bits]", color="steelblue")
    ax.bar(x + width / 2, mi_vals, width,
           label="MI = I(label; s_{t-2} | s_t=Focus) [bits]", color="tomato")

    ax.set_ylabel("Information [bits]")
    ax.set_title("Entropy vs Mutual Information at Focus State Positions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.legend()

    for xi, (hv, mv) in enumerate(zip(h_vals, mi_vals)):
        ax.text(xi - width / 2, hv + 0.01, f"{hv:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + width / 2, mv + 0.01, f"{mv:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# レポート生成
# ══════════════════════════════════════════════════════════════════════════════

def write_report(rows: list, detail_rows: list) -> None:
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines += [
        "# Proposal F: s_{t-2} 曖昧性解消の情報量定量化",
        "",
        f"生成日時: {now}",
        "",
        "---",
        "",
        "## 概要",
        "",
        "Focus State が現れる**境界位置**（B-start または B-end）において、",
        "2ステップ前の状態 s_{t-2} がラベル識別にどれだけ情報をもたらすかを定量化する。",
        "",
        "> **境界位置**: compound_words.txt（貪欲アルゴリズム準拠）の分割情報を直接使用。",
        "",
        "---",
        "",
        "## サマリー比較表",
        "",
        "| モデル | k | Focus State | 境界出現数 | B-start | B-end |"
        " H(label) [bits] | H(label\\|s_{t-2}) [bits] | MI [bits]"
        " | acc_without | acc_with |",
        "|--------|---|------------|---------|---------|-------|"
        "----------------|------------------------|----------|"
        "------------|---------|",
    ]

    for r in rows:
        lines.append(
            f"| {r['model_type']} | {r['k']} | S{r['focus']} "
            f"| {r['total']} | {r['n_bstart']} | {r['n_bend']} "
            f"| {r['h_label']:.4f} | {r['h_label_given_s']:.4f} "
            f"| **{r['mi']:.4f}** "
            f"| {r['acc_without']*100:.1f}% | {r['acc_with']*100:.1f}% |"
        )

    lines += [
        "",
        "- `acc_without`: s_{t-2} を使わず多数クラスで予測した場合の正解率",
        "- `acc_with`: s_{t-2} ごとに多数クラスで予測した場合の正解率",
        "",
        "---",
        "",
        "## 情報量の解釈",
        "",
        "- **H(label | s_t=Focus)**: Focus State 出現時点での境界ラベルの不確実性（ビット）",
        "  - 1.0 bit = B-start/B-end が 50/50 で完全に不確定",
        "  - 0.0 bit = 一方のみ（完全に確定済み）",
        "- **MI = I(label; s_{t-2} | s_t=Focus)**: s_{t-2} が解消できる不確実性の量",
        "  - 値が大きいほど s_{t-2} による曖昧性解消が有効",
        "- **acc_with - acc_without**: s_{t-2} による予測精度の向上幅",
        "",
        "---",
        "",
        "## s_{t-2} ごとの詳細分布",
        "",
    ]

    for d in detail_rows:
        lines += [
            f"### {d['model_type']} k={d['k']}  Focus State S{d['focus']}",
            "",
            f"全境界出現: {d['total']} 件（B-start={d['n_bstart']}, B-end={d['n_bend']}）",
            f"s_{{t-2}} が取得可能: {d['n_with_prev2']} 件",
            "",
            "| s_{t-2} | B-start | B-end | 合計 | P(B-start) | P(B-end) | 予測 |",
            "|---------|---------|-------|------|----------|--------|------|",
        ]
        if d["by_s"]:
            for s in sorted(d["by_s"].keys()):
                lbl_cnt = d["by_s"][s]
                ns = lbl_cnt.get("B-start", 0)
                ne = lbl_cnt.get("B-end",   0)
                tot_s = ns + ne
                p_start = ns / tot_s if tot_s > 0 else 0.0
                p_end   = ne / tot_s if tot_s > 0 else 0.0
                pred = "B-start" if ns >= ne else "B-end"
                lines.append(
                    f"| S{s} | {ns} | {ne} | {tot_s} "
                    f"| {p_start:.3f} | {p_end:.3f} | {pred} |"
                )
        lines.append("")

    lines += [
        "---",
        "",
        "## 参照",
        "",
        "- 境界定義: `hypothesis/00_slot_model/data/compound_words.txt`（貪欲アルゴリズム準拠）",
        "- モデル: `hypothesis/01_bigram/results/hmm_model_cache/full_k{7,8}.npz`",
        "- モデル: `hypothesis/03_trigram/results/hmm_model_cache/trigram_k{7,8}.npz`",
        "- common.py: `hypothesis/06_state_characterization/source/common.py`",
        "",
        "_本レポートは analysis_F_mutual_info.py により自動生成。_",
    ]

    report_path = OUT_DIR / "analysis_F_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"  Report saved: {report_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log("=== Proposal F: s_{t-2} 曖昧性解消の情報量定量化 ===")

    log("Loading compound splits (from compound_words.txt / greedy algorithm)...")
    splits = load_compound_splits()
    log(f"  Compound words: {len(splits)}")

    compound_list = list(splits.keys())

    configs = [
        ("bigram",  7),
        ("bigram",  8),
        ("trigram", 7),
        ("trigram", 8),
    ]

    summary_rows = []
    detail_rows  = []

    for model_type, k_val in configs:
        log(f"\n--- {model_type} k={k_val} ---")

        # モデルロード
        if model_type == "bigram":
            model = load_bigram_model(k_val)
        else:
            model = load_trigram_model(k_val)
        if model is None:
            log(f"  SKIP: model not found")
            continue

        focus_state = FOCUS_STATES[(model_type, k_val)]
        log(f"  Focus State: S{focus_state}")

        # Viterbi デコード（複合語のみ）
        log("  Decoding compound words...")
        decoded = decode_words(compound_list, model, model_type)
        log(f"  Decoded: {len(decoded)} / {len(compound_list)} words")

        # (s_{t-2}, label) ペアの集計
        counts = collect_focus_pairs(splits, decoded, focus_state)
        total_pairs = sum(counts.values())
        log(f"  Focus-boundary pairs: {total_pairs}")

        # 情報量計算
        measures = compute_information_measures(counts)
        measures["model_type"] = model_type
        measures["k"]          = k_val
        measures["focus"]      = focus_state

        log(f"  H(label)             = {measures['h_label']:.4f} bits")
        log(f"  H(label|s_{{t-2}})    = {measures['h_label_given_s']:.4f} bits")
        log(f"  MI                   = {measures['mi']:.4f} bits")
        log(f"  acc_without s_{{t-2}} = {measures['acc_without']*100:.1f}%")
        log(f"  acc_with    s_{{t-2}} = {measures['acc_with']*100:.1f}%")

        # ヒートマップ描画
        plot_joint_heatmap(
            measures["by_s"], k_val, focus_state,
            model_type, k_val,
            OUT_DIR / f"F_joint_heatmap_{model_type}_k{k_val}.png",
        )

        summary_rows.append(measures)
        detail_rows.append(measures)

    # 比較グラフ
    if summary_rows:
        log("\nPlotting summary figures...")
        plot_accuracy_comparison(summary_rows, OUT_DIR / "F_accuracy_comparison.png")
        plot_mi_comparison(summary_rows, OUT_DIR / "F_mi_comparison.png")

    # レポート
    log("\nWriting report...")
    write_report(summary_rows, detail_rows)

    log("\n=== Done ===")


if __name__ == "__main__":
    main()
