"""
analysis_A_emission.py
========================
提案 A：放射確率行列の文字レベル分析

目的:
  Bigram/Trigram モデルの各状態が「どの文字を好んで放射するか」を明らかにし、
  HMM 状態の実体（音節・字種・位置的役割）を文字分布から解釈する。

分析内容:
  1. 放射確率行列ヒートマップ（Voynich 文字 22 文字 × 状態数）
  2. 各状態の上位 5 文字と確率
  3. Focus State（B-start 特化状態）の文字分布の詳細分析
  4. Voynich 文字をグループ（gallows / bench / vowel / sibilant 等）に分類し
     各状態がどのグループを好むかを可視化
  5. V8 スロット対応表：各状態の上位文字が担う V8 スロット名を列挙
  6. Focus State vs 全状態平均: 分布の KL ダイバージェンス

出力:
  results/emission_heatmap_{bigram,trigram}_k{7,8}.png
  results/emission_top_chars_{bigram,trigram}_k{7,8}.png
  results/emission_focus_state_{bigram,trigram}_k{7,8}.png
  results/analysis_A_emission_report.md

実行:
  cd /home/practi/work_voy
  PYTHONPATH=/home/practi/work_voy/.venv/lib/python3.10/site-packages \\
    python3.10 hypothesis/06_state_characterization/source/analysis_A_emission.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    log, OUT_DIR,
    VOY_CHARS, VOY_INDICES,
    CHAR_TO_SLOTS, CHAR_GROUP_MAP, CHAR_GROUPS,
    FOCUS_STATES, PHANTOM_STATES,
    load_bigram_model, load_trigram_model,
    load_compound_splits,
)


# ════════════════════════════════════════════════════════════════════════
# 1. ヒートマップ
# ════════════════════════════════════════════════════════════════════════
def plot_emission_heatmap(emiss: np.ndarray, k: int, model_type: str,
                          focus_state: int, phantoms: list, out_path: Path):
    """放射確率行列ヒートマップ（Voynich 文字のみ）"""
    data = emiss[:, VOY_INDICES]   # (k, 22)

    fig, ax = plt.subplots(figsize=(14, max(4, k * 0.8)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="P(char | state)")

    ax.set_xticks(range(len(VOY_CHARS)))
    ax.set_xticklabels(VOY_CHARS, fontsize=9)
    ax.set_yticks(range(k))

    state_labels = []
    for s in range(k):
        if s in phantoms:
            state_labels.append(f"S{s} [phantom]")
        elif s == focus_state:
            state_labels.append(f"S{s} ★Focus")
        else:
            state_labels.append(f"S{s}")
    ax.set_yticklabels(state_labels, fontsize=9)

    # グループ区切り線を x 軸に追加
    group_order = ["gallows", "bench", "vowel", "sibilant", "liquid", "nasal", "other"]
    group_colors = {
        "gallows": "#C0392B", "bench": "#8E44AD", "vowel": "#2980B9",
        "sibilant": "#27AE60", "liquid": "#E67E22", "nasal": "#16A085", "other": "#95A5A6",
    }
    for xi, char in enumerate(VOY_CHARS):
        grp = CHAR_GROUP_MAP.get(char, "other")
        color = group_colors[grp]
        ax.axvline(xi - 0.5, color="white", linewidth=0.4, alpha=0.5)
        ax.text(xi, k - 0.1, char, ha="center", va="bottom", fontsize=7.5,
                color=color, fontweight="bold", transform=ax.get_xaxis_transform())

    ax.set_xlabel("Voynich 文字", fontsize=10)
    ax.set_ylabel("隠れ状態", fontsize=10)
    ax.set_title(f"[{model_type.capitalize()} k={k}] 放射確率ヒートマップ\n"
                 f"(★: Focus State S{focus_state},  [phantom]: 縮退状態)", fontsize=11)

    # 凡例
    patches = [mpatches.Patch(color=group_colors[g], label=g) for g in group_order]
    ax.legend(handles=patches, loc="upper right", fontsize=7.5,
              bbox_to_anchor=(1.22, 1.0), title="文字グループ")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# 2. 上位文字バーチャート
# ════════════════════════════════════════════════════════════════════════
def plot_top_chars(emiss: np.ndarray, k: int, model_type: str,
                   focus_state: int, phantoms: list, out_path: Path, top_n: int = 8):
    """各状態の上位 N 文字 + 確率を横並びで表示"""
    n_cols = 3
    n_rows = (k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    axes = np.array(axes).flatten()

    group_colors = {
        "gallows": "#C0392B", "bench": "#8E44AD", "vowel": "#2980B9",
        "sibilant": "#27AE60", "liquid": "#E67E22", "nasal": "#16A085", "other": "#95A5A6",
    }

    for s in range(k):
        ax = axes[s]
        # Voynich 文字の確率のみ表示
        voy_probs = emiss[s, VOY_INDICES]
        top_idx = np.argsort(voy_probs)[::-1][:top_n]
        top_chars = [VOY_CHARS[i] for i in top_idx]
        top_probs = voy_probs[top_idx]
        bar_colors = [group_colors.get(CHAR_GROUP_MAP.get(c, "other"), "#95A5A6")
                      for c in top_chars]

        bars = ax.bar(range(top_n), top_probs, color=bar_colors, alpha=0.88,
                      edgecolor="white", linewidth=1.2)
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(top_chars, fontsize=10, fontweight="bold")
        ax.set_ylabel("P(char|state)", fontsize=8)

        title_suffix = " [★ Focus]" if s == focus_state else (" [phantom]" if s in phantoms else "")
        ax.set_title(f"S{s}{title_suffix}", fontsize=10,
                     color="#C0392B" if s == focus_state else "#555555")
        ax.tick_params(axis="y", labelsize=7)

    for i in range(k, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"[{model_type.capitalize()} k={k}] 各状態 上位 {top_n} 文字",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# 3. Focus State 詳細（複合語先頭文字分布との比較）
# ════════════════════════════════════════════════════════════════════════
def compute_base_initial_dist(compound_splits: dict) -> np.ndarray:
    """複合語の全基の先頭文字の経験分布 → 長さ 22 の numpy 配列（VOY_CHARS 順）"""
    counts = np.zeros(len(VOY_CHARS), dtype=float)
    for splits in compound_splits.values():
        for base in splits:
            fc = base[0] if base else ""
            if fc in VOY_CHARS:
                counts[VOY_CHARS.index(fc)] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p = np.maximum(p, eps);  p = p / p.sum()
    q = np.maximum(q, eps);  q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def plot_focus_state_detail(emiss: np.ndarray, focus_state: int, base_init_dist: np.ndarray,
                             model_type: str, k: int, out_path: Path):
    """Focus State の放射分布 vs 複合語先頭文字分布"""
    fs_emiss = emiss[focus_state, VOY_INDICES]
    fs_emiss = fs_emiss / fs_emiss.sum()

    # 全状態平均（phantom 除く）
    phantoms = PHANTOM_STATES.get((model_type, k), [])
    non_phantom = [s for s in range(k) if s not in phantoms]
    mean_emiss = emiss[non_phantom, :][:, VOY_INDICES].mean(axis=0)
    mean_emiss = mean_emiss / mean_emiss.sum()

    x = np.arange(len(VOY_CHARS))
    width = 0.27

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 上段: 3 分布の並べ比較
    ax = axes[0]
    ax.bar(x - width, fs_emiss,       width, label=f"Focus State S{focus_state}", color="#C0392B", alpha=0.85)
    ax.bar(x,         mean_emiss,     width, label="全状態平均 (non-phantom)",    color="#7FB3D3", alpha=0.85)
    ax.bar(x + width, base_init_dist, width, label="複合語基・先頭文字 (実測)",   color="#82E0AA", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(VOY_CHARS, fontsize=9)
    ax.set_ylabel("確率", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_title(f"[{model_type.capitalize()} k={k}] Focus State S{focus_state} 放射分布 vs 基先頭文字分布", fontsize=11)

    kl_fs_base = kl_divergence(fs_emiss, base_init_dist)
    kl_avg_base = kl_divergence(mean_emiss, base_init_dist)
    ax.text(0.01, 0.97,
            f"KL(Focus ∥ base_init) = {kl_fs_base:.3f}\n"
            f"KL(avg   ∥ base_init) = {kl_avg_base:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

    # 下段: Focus / mean の比（対数スケール）
    ax2 = axes[1]
    ratio = np.log2((fs_emiss + 1e-8) / (mean_emiss + 1e-8))
    colors = ["#C0392B" if r > 0 else "#2980B9" for r in ratio]
    ax2.bar(x, ratio, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(VOY_CHARS, fontsize=9)
    ax2.set_ylabel("log₂( Focus / 全状態平均 )", fontsize=10)
    ax2.set_title(f"Focus State S{focus_state} の各文字への偏り（正 = Focus が多く放射）", fontsize=10)

    # V8 スロット注釈
    for xi, char in enumerate(VOY_CHARS):
        slots = CHAR_TO_SLOTS.get(char, set())
        slot_str = "/".join(sorted(slots)[:2]) if slots else ""
        if slot_str:
            ax2.text(xi, ax2.get_ylim()[0] - 0.03, slot_str, ha="center", va="top",
                     fontsize=5.5, color="#555", rotation=30,
                     transform=ax2.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# 4. テキスト集計（各状態の上位文字表）
# ════════════════════════════════════════════════════════════════════════
def summarize_emission(emiss: np.ndarray, k: int, focus_state: int,
                       phantoms: list, top_n: int = 5) -> list[dict]:
    rows = []
    for s in range(k):
        voy_probs = emiss[s, VOY_INDICES]
        top_idx = np.argsort(voy_probs)[::-1][:top_n]
        top_chars = [(VOY_CHARS[i], voy_probs[i]) for i in top_idx]

        # 上位文字の V8 スロット一覧
        slot_set = set()
        for c, _ in top_chars:
            slot_set.update(CHAR_TO_SLOTS.get(c, set()))

        # グループ分布（VOY_CHARS 全体の sum by group）
        group_dist = defaultdict(float)
        for xi, c in enumerate(VOY_CHARS):
            g = CHAR_GROUP_MAP.get(c, "other")
            group_dist[g] += voy_probs[xi]

        dominant_group = max(group_dist, key=group_dist.get)

        rows.append({
            "state":        s,
            "is_focus":     (s == focus_state),
            "is_phantom":   (s in phantoms),
            "top_chars":    top_chars,
            "v8_slots":     sorted(slot_set),
            "dominant_group": dominant_group,
            "group_dist":   dict(group_dist),
        })
    return rows


# ════════════════════════════════════════════════════════════════════════
# 5. レポート生成
# ════════════════════════════════════════════════════════════════════════
def build_report(all_results: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 提案 A：放射確率行列の文字レベル分析レポート",
        "",
        f"生成日時: {now}",
        "",
        "## 概要",
        "",
        "Bigram/Trigram HMM の放射確率行列 `emiss` (k×32) を分析し、",
        "各状態がどの Voynich 文字を好んで放射するかを明らかにする。",
        "Focus State（B-start 特化状態）の放射分布を複合語基の先頭文字分布と比較する。",
        "",
        "---",
        "",
    ]

    for res in all_results:
        model_type = res["model_type"]
        k          = res["k"]
        fs         = res["focus_state"]
        phantoms   = res["phantoms"]
        summaries  = res["summaries"]
        kl_fs      = res["kl_fs_base"]
        kl_avg     = res["kl_avg_base"]

        lines += [
            f"## {model_type.capitalize()} k={k}  (Focus State: S{fs})",
            "",
            f"- Phantom State: {', '.join(f'S{s}' for s in phantoms) if phantoms else 'なし'}",
            f"- KL(Focus S{fs} ∥ 基先頭文字分布) = **{kl_fs:.3f}**",
            f"- KL(全状態平均   ∥ 基先頭文字分布) = {kl_avg:.3f}",
            ("- → Focus State は基先頭文字分布に **より近い** "
             if kl_fs < kl_avg else
             "- → Focus State は基先頭文字分布から全状態平均より **遠い**"),
            "",
            "### 各状態の上位 5 文字と V8 スロット対応",
            "",
            "| 状態 | 上位文字 (prob) | 支配的グループ | V8 スロット候補 | 判定 |",
            "|-----|---------------|-------------|----------------|------|",
        ]

        for row in summaries:
            s    = row["state"]
            flag = ("**★Focus**" if row["is_focus"] else
                    ("Phantom"   if row["is_phantom"] else ""))
            top_str  = ", ".join(f"{c}({p:.3f})" for c, p in row["top_chars"])
            slot_str = ", ".join(row["v8_slots"][:4]) if row["v8_slots"] else "—"
            lines.append(
                f"| S{s} | {top_str} | {row['dominant_group']} | {slot_str} | {flag} |"
            )

        lines += ["", "### Focus State 解釈", ""]
        fs_row = next(r for r in summaries if r["state"] == fs)
        top_char_list = [c for c, _ in fs_row["top_chars"]]
        lines.append(
            f"Focus State S{fs} の上位文字: **{', '.join(top_char_list)}**"
        )
        lines.append(f"支配的グループ: **{fs_row['dominant_group']}**")
        lines.append(f"関連 V8 スロット: {', '.join(fs_row['v8_slots'])}")

        # 解釈コメント
        dominant = fs_row["dominant_group"]
        interp_map = {
            "gallows": "gallows 文字 (t/k/p/f) を好む → 複合語先頭の典型的パターン（gallows で基が始まる）",
            "bench":   "bench 文字 (q/d) を好む → 特定の語頭接頭辞パターン",
            "vowel":   "母音的文字 (a/e/o/y/i) を好む → 複合語境界での母音接続",
            "sibilant": "摩擦音 (s/h) を好む → sibilant 語頭",
            "liquid":  "流音 (l/r) を好む → C1 スロット相当 (l/r で始まる基)",
        }
        interp = interp_map.get(dominant, f"{dominant} 文字群を好む")
        lines += [f"**解釈**: {interp}", "", "---", ""]

    lines += [
        "## 全モデル比較サマリー",
        "",
        "| モデル | k | Focus State | 上位文字 (top3) | 支配グループ | KL(Focus∥base) | KL(avg∥base) |",
        "|--------|---|------------|--------------|------------|---------------|-------------|",
    ]
    for res in all_results:
        mt = res["model_type"]
        k  = res["k"]
        fs = res["focus_state"]
        fs_row = next(r for r in res["summaries"] if r["state"] == fs)
        top3 = ", ".join(c for c, _ in fs_row["top_chars"][:3])
        lines.append(
            f"| {mt} | {k} | S{fs} | {top3} | {fs_row['dominant_group']} "
            f"| {res['kl_fs_base']:.3f} | {res['kl_avg_base']:.3f} |"
        )

    lines += ["", "_本レポートは analysis_A_emission.py により自動生成。_"]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("提案 A：放射確率行列の文字レベル分析 開始")

    # 複合語基の先頭文字分布（B-start の実測参照分布として使用）
    compound_splits = load_compound_splits()
    base_init_dist  = compute_base_initial_dist(compound_splits)
    log(f"複合語ロード: {len(compound_splits)} 語 → 基先頭文字分布を算出")

    all_results = []

    for model_type, load_fn in [("bigram", load_bigram_model), ("trigram", load_trigram_model)]:
        for k in [7, 8]:
            log(f"\n{'─'*60}")
            log(f"{model_type} k={k}")

            model = load_fn(k)
            if model is None:
                continue
            log(f"  logL = {model['logL']:.2f}")

            emiss    = model["emiss"]          # (k, 32)
            fs       = FOCUS_STATES[(model_type, k)]
            phantoms = PHANTOM_STATES.get((model_type, k), [])

            # 確認
            log(f"  emiss shape: {emiss.shape}")
            log(f"  Focus State: S{fs},  Phantoms: {phantoms}")

            # 上位文字表
            summaries = summarize_emission(emiss, k, fs, phantoms)
            for row in summaries:
                top5 = ", ".join(f"{c}({p:.3f})" for c, p in row["top_chars"])
                tag = " ★Focus" if row["is_focus"] else (" [phantom]" if row["is_phantom"] else "")
                log(f"    S{row['state']}{tag}: {top5}  [{row['dominant_group']}]")

            # KL ダイバージェンス
            fs_emiss  = emiss[fs, VOY_INDICES]
            fs_emiss  = fs_emiss / (fs_emiss.sum() + 1e-10)
            non_ph    = [s for s in range(k) if s not in phantoms]
            mean_em   = emiss[non_ph, :][:, VOY_INDICES].mean(axis=0)
            mean_em   = mean_em / (mean_em.sum() + 1e-10)
            kl_fs  = kl_divergence(fs_emiss, base_init_dist)
            kl_avg = kl_divergence(mean_em,  base_init_dist)
            log(f"  KL(Focus S{fs} ∥ base_init) = {kl_fs:.3f}")
            log(f"  KL(all_avg    ∥ base_init) = {kl_avg:.3f}")

            tag2 = OUT_DIR / f"emission_heatmap_{model_type}_k{k}.png"
            plot_emission_heatmap(emiss, k, model_type, fs, phantoms, tag2)
            log(f"  Saved: {tag2.name}")

            tag3 = OUT_DIR / f"emission_top_chars_{model_type}_k{k}.png"
            plot_top_chars(emiss, k, model_type, fs, phantoms, tag3)
            log(f"  Saved: {tag3.name}")

            tag4 = OUT_DIR / f"emission_focus_detail_{model_type}_k{k}.png"
            plot_focus_state_detail(emiss, fs, base_init_dist, model_type, k, tag4)
            log(f"  Saved: {tag4.name}")

            all_results.append({
                "model_type": model_type,
                "k":          k,
                "focus_state": fs,
                "phantoms":   phantoms,
                "summaries":  summaries,
                "kl_fs_base": kl_fs,
                "kl_avg_base": kl_avg,
            })

    # Markdown レポート
    report = build_report(all_results)
    report_path = OUT_DIR / "analysis_A_emission_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"\nレポート保存: {report_path}")
    log("✓ 提案 A 完了。出力先: " + str(OUT_DIR.resolve()))
