"""
emission_analysis.py
================================
S4 (k=7) / S5 (k=8) 放出確率の直接確認

interpretation_notes.md Section 7 に対応する検証:

  7.1  S4 の emission probability 上位文字が {l, r, o, y, s} と一致するか直接確認
  7.2  S4 → S? の遷移確率を確認し「S4の直後にどの状態が来るか」を検証
  7.3  k=8 の S0 の役割の再検討（全状態の放出分布を確認）

出力:
  emission_heatmap_k{k}.png      : 全状態 × 上位文字 ヒートマップ
  emission_focus_k{k}.png        : focus state の文字別放出確率バーチャート
                                   （V8 boundary-compatible chars をハイライト）
  transition_heatmap_k{k}.png    : 遷移行列ヒートマップ
  emission_analysis_report.md    : 数値レポート（Markdown）
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
import warnings

warnings.filterwarnings("ignore")


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")


# ── フォント設定 ──────────────────────────────────────────────────────────
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
PHANTOM_STATE = {7: 3, 8: 4}   # Viterbi に出現しない縮退状態
FOCUS_STATE   = {7: 4, 8: 5}   # single_vs_compound_analysis.py で特定された境界集中状態

BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"
SPECIAL_CHARS = {BOS_CHAR, EOS_CHAR, PAD_CHAR}
MIN_WORD_LEN  = 2

# V8 文法で定義された境界互換文字集合
#   SLOTS_V8[0]   : 基の先頭文字 = {l, r, o, y, s, v}
#   SLOTS_V8[13]  : 基の末尾候補 = {s}
#   SLOTS_V8[14]  : 基の末尾候補 = {y}
#   SLOTS_V8[15]  : 基の末尾候補 = {k, t, p, f, l, r, o, y}
# → 先頭 ∩ 末尾 の共通文字 = {l, r, o, y, s}
V8_BASE_START  = frozenset(["l", "r", "o", "y", "s", "v"])
V8_BASE_END    = frozenset(["s", "y", "k", "t", "p", "f", "l", "r", "o", "y"])
V8_BOUNDARY_COMPAT = V8_BASE_START & V8_BASE_END   # {l, r, o, y, s}


# ════════════════════════════════════════════════════════════════════════
# 1. モデルロード
# ════════════════════════════════════════════════════════════════════════
def load_model(k: int) -> dict | None:
    path = MODEL_CACHE / f"full_k{k}.npz"
    if not path.exists():
        log(f"  ERROR: キャッシュが見つかりません: {path}")
        return None
    d = np.load(path)
    return {
        "start": d["start"],   # (k,)
        "trans": d["trans"],   # (k, k)
        "emiss": d["emiss"],   # (k, vocab_size)
        "logL":  float(d["logL"][0]),
    }


# ════════════════════════════════════════════════════════════════════════
# 2. Emission probability の分析
# ════════════════════════════════════════════════════════════════════════
def analyze_emission(emiss: np.ndarray, char2idx: dict, k: int) -> dict:
    """
    各状態の放出確率を分析して辞書で返す。

    Returns
    -------
    {
      "state_topchars"     : {state_idx: [(char, prob), ...]} 上位10文字,
      "boundary_compat_mass": {state_idx: float}  V8境界互換文字への確率質量 (%),
      "base_start_mass"     : {state_idx: float}  V8基先頭文字への確率質量 (%),
      "base_end_mass"       : {state_idx: float}  V8基末尾文字への確率質量 (%),
      "idx2char"            : {int: str},          インデックス→文字 逆引き
      "voynich_chars"       : list[str],           特殊文字を除いた文字リスト
      "voynich_indices"     : list[int],           上記のインデックス
    }
    """
    idx2char = {v: k for k, v in char2idx.items()}

    # 特殊文字（BOS/EOS/PAD）を除いたヴォイニッチ文字のみ
    voynich_indices = [
        i for i, c in sorted(idx2char.items())
        if c not in SPECIAL_CHARS
    ]
    voynich_chars = [idx2char[i] for i in voynich_indices]

    state_topchars       = {}
    boundary_compat_mass = {}
    base_start_mass      = {}
    base_end_mass        = {}

    for s in range(k):
        probs = emiss[s]  # shape: (vocab_size,)

        # ヴォイニッチ文字のみの確率（特殊文字を除外し正規化）
        voy_probs_raw = np.array([probs[i] for i in voynich_indices])
        total_voy     = voy_probs_raw.sum()
        if total_voy > 0:
            voy_probs = voy_probs_raw / total_voy  # ヴォイニッチ文字内で正規化
        else:
            voy_probs = voy_probs_raw

        # 上位10文字
        top_indices = np.argsort(voy_probs)[::-1][:10]
        state_topchars[s] = [(voynich_chars[i], float(voy_probs[i])) for i in top_indices]

        # V8 文字集合への確率質量
        compat_mass = sum(
            voy_probs[voynich_chars.index(c)]
            for c in V8_BOUNDARY_COMPAT
            if c in voynich_chars
        )
        start_mass = sum(
            voy_probs[voynich_chars.index(c)]
            for c in V8_BASE_START
            if c in voynich_chars
        )
        end_mass = sum(
            voy_probs[voynich_chars.index(c)]
            for c in V8_BASE_END
            if c in voynich_chars
        )

        boundary_compat_mass[s] = float(compat_mass) * 100
        base_start_mass[s]      = float(start_mass) * 100
        base_end_mass[s]        = float(end_mass) * 100

    return {
        "state_topchars":      state_topchars,
        "boundary_compat_mass": boundary_compat_mass,
        "base_start_mass":     base_start_mass,
        "base_end_mass":       base_end_mass,
        "idx2char":            idx2char,
        "voynich_chars":       voynich_chars,
        "voynich_indices":     voynich_indices,
    }


# ════════════════════════════════════════════════════════════════════════
# 3. 可視化
# ════════════════════════════════════════════════════════════════════════
def plot_emission_heatmap(emiss: np.ndarray, ea: dict, k: int,
                          phantom_s: int, focus_s: int, out_dir: Path) -> str:
    """全状態 × 上位文字のヒートマップ（top-30文字を使用）。"""
    voynich_chars   = ea["voynich_chars"]
    voynich_indices = ea["voynich_indices"]

    # 全状態合計で上位30文字を選択
    emiss_voy = emiss[:, voynich_indices]  # (k, n_voy)
    total_probs = emiss_voy.sum(axis=0)
    top30_local = np.argsort(total_probs)[::-1][:30]
    top30_chars  = [voynich_chars[i] for i in top30_local]

    # 正規化（ヴォイニッチ文字内で各状態を正規化）
    emiss_norm = emiss_voy / (emiss_voy.sum(axis=1, keepdims=True) + 1e-35)
    heatmap_data = emiss_norm[:, top30_local]   # (k, 30)

    state_labels = []
    for s in range(k):
        suffix = ""
        if s == phantom_s:
            suffix = " [Phantom]"
        elif s == focus_s:
            suffix = " [Focus]"
        state_labels.append(f"S{s}{suffix}")

    fig, ax = plt.subplots(figsize=(14, max(4, k * 0.9)))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=0)

    ax.set_xticks(range(len(top30_chars)))
    ax.set_xticklabels(top30_chars, rotation=45, ha="right", fontsize=9)

    ax.set_yticks(range(k))
    ax.set_yticklabels(state_labels, fontsize=10)

    # V8 boundary-compatible chars を枠で強調
    for xi, char in enumerate(top30_chars):
        if char in V8_BOUNDARY_COMPAT:
            rect = plt.Rectangle(
                (xi - 0.5, -0.5), 1, k, fill=False, edgecolor="#1A5276", linewidth=2.5
            )
            ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="放出確率（ヴォイニッチ文字内正規化）")
    ax.set_title(
        f"HMM k={k}: 全状態 放出確率ヒートマップ（上位30文字）\n"
        f"青枠: V8境界互換文字 {{{', '.join(sorted(V8_BOUNDARY_COMPAT))}}}",
        fontsize=11,
    )
    plt.tight_layout()

    fname = f"emission_heatmap_k{k}.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


def plot_emission_focus(emiss: np.ndarray, ea: dict, k: int,
                        focus_s: int, phantom_s: int, out_dir: Path) -> str:
    """focus state の全ヴォイニッチ文字に対する放出確率バーチャート。"""
    voynich_chars   = ea["voynich_chars"]
    voynich_indices = ea["voynich_indices"]

    emiss_voy = emiss[focus_s, voynich_indices]
    total     = emiss_voy.sum()
    if total > 0:
        emiss_norm = emiss_voy / total
    else:
        emiss_norm = emiss_voy

    # 確率順にソート
    order      = np.argsort(emiss_norm)[::-1]
    sorted_chars = [voynich_chars[i] for i in order]
    sorted_probs = [emiss_norm[i] for i in order]

    # 色設定
    bar_colors = []
    for char in sorted_chars:
        if char in V8_BOUNDARY_COMPAT:
            bar_colors.append("#C0392B")   # 赤: boundary-compatible (∩)
        elif char in V8_BASE_START:
            bar_colors.append("#E59866")   # オレンジ: 基先頭のみ
        elif char in V8_BASE_END:
            bar_colors.append("#5DADE2")   # 青: 基末尾のみ
        else:
            bar_colors.append("#BDC3C7")   # グレー: 非境界文字

    fig, ax = plt.subplots(figsize=(max(12, len(sorted_chars) * 0.45), 5))
    bars = ax.bar(range(len(sorted_chars)), sorted_probs, color=bar_colors, edgecolor="white")

    ax.set_xticks(range(len(sorted_chars)))
    ax.set_xticklabels(sorted_chars, fontsize=9)
    ax.set_ylabel("放出確率（ヴォイニッチ文字内正規化）", fontsize=11)

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C0392B", label=f"V8境界互換 (∩) = {{{', '.join(sorted(V8_BOUNDARY_COMPAT))}}}"),
        Patch(facecolor="#E59866", label=f"V8基先頭のみ = {{{', '.join(sorted(V8_BASE_START - V8_BOUNDARY_COMPAT))}}}"),
        Patch(facecolor="#5DADE2", label=f"V8基末尾のみ = {{{', '.join(sorted(V8_BASE_END - V8_BOUNDARY_COMPAT))}}}"),
        Patch(facecolor="#BDC3C7", label="非境界文字"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # 確率質量の注記
    compat_mass = ea["boundary_compat_mass"][focus_s]
    start_mass  = ea["base_start_mass"][focus_s]
    end_mass    = ea["base_end_mass"][focus_s]
    note = (
        f"S{focus_s} 確率質量: 境界互換={compat_mass:.1f}%  "
        f"基先頭集合={start_mass:.1f}%  基末尾集合={end_mass:.1f}%"
    )
    ax.text(
        0.5, 0.97, note,
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
    )

    ax.set_title(
        f"HMM k={k}  S{focus_s} [Focus State] 放出確率分布\n"
        f"赤: V8境界互換文字 {{l,r,o,y,s}}  オレンジ: 基先頭のみ  青: 基末尾のみ",
        fontsize=11,
    )
    plt.tight_layout()

    fname = f"emission_focus_k{k}.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


def plot_transition_heatmap(trans: np.ndarray, k: int,
                             phantom_s: int, focus_s: int, out_dir: Path) -> str:
    """遷移行列ヒートマップ（focus state の行を強調）。"""
    state_labels = []
    for s in range(k):
        suffix = " [Ph]" if s == phantom_s else (" [Fo]" if s == focus_s else "")
        state_labels.append(f"S{s}{suffix}")

    fig, ax = plt.subplots(figsize=(k + 2, k + 1))
    im = ax.imshow(trans, aspect="auto", cmap="Blues", vmin=0, vmax=trans.max())

    # セル内に確率値を表示
    for row in range(k):
        for col in range(k):
            val = trans[row, col]
            ax.text(
                col, row, f"{val:.2f}",
                ha="center", va="center",
                fontsize=9 if k <= 8 else 7,
                color="white" if val > trans.max() * 0.6 else "black",
            )

    ax.set_xticks(range(k))
    ax.set_xticklabels(state_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(k))
    ax.set_yticklabels(state_labels, fontsize=9)
    ax.set_xlabel("遷移先 (to)", fontsize=10)
    ax.set_ylabel("遷移元 (from)", fontsize=10)

    # focus state の行・列を枠線で強調
    for s_idx in [focus_s]:
        ax.add_patch(plt.Rectangle(
            (-0.5, s_idx - 0.5), k, 1,
            fill=False, edgecolor="#C0392B", linewidth=2.5
        ))
        ax.add_patch(plt.Rectangle(
            (s_idx - 0.5, -0.5), 1, k,
            fill=False, edgecolor="#C0392B", linewidth=2.5, linestyle="--"
        ))

    plt.colorbar(im, ax=ax, label="遷移確率")
    ax.set_title(
        f"HMM k={k}: 遷移行列\n"
        f"赤実線: S{focus_s}（Focus）の遷移元行、赤破線: 遷移先列",
        fontsize=11,
    )
    plt.tight_layout()

    fname = f"transition_heatmap_k{k}.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


# ════════════════════════════════════════════════════════════════════════
# 4. Markdown レポート生成
# ════════════════════════════════════════════════════════════════════════
def build_md_report(results: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Emission Probability 直接確認レポート",
        "",
        f"**生成日時**: {now}",
        "**スクリプト**: `hypothesis/02_compound_hmm/source/emission_analysis.py`",
        "**HMMモデル**: `hypothesis/01_bigram/results/hmm_model_cache/full_k7.npz`, `full_k8.npz`",
        "",
        "---",
        "",
        "## 検証目的",
        "",
        "`interpretation_notes.md` Section 7 の未解決問い:",
        "",
        "1. **S4 放出分布**: S4 (k=7) の emission probability 上位文字が V8 境界互換文字 `{l, r, o, y, s}` と一致するか",
        "2. **S4 遷移行列**: S4 → S? の遷移確率（S4→S4 自己遷移が低いか）",
        "3. **k=8 全状態の放出分布**: S0 の役割の再検討",
        "",
        "### V8 文字集合の定義",
        "",
        "| 集合 | 文字 | 根拠 |",
        "|------|------|------|",
        f"| 基先頭 (SLOTS_V8[0]) | `{{{', '.join(sorted(V8_BASE_START))}}}` | 各基の先頭文字 |",
        f"| 基末尾 (SLOTS_V8[13-15]) | `{{{', '.join(sorted(V8_BASE_END))}}}` | 各基の末尾文字 |",
        f"| **境界互換 (∩)** | **`{{{', '.join(sorted(V8_BOUNDARY_COMPAT))}}}`** | 先頭 ∩ 末尾 |",
        "",
        "---",
        "",
    ]

    for k, res in results.items():
        if res is None:
            lines += [f"## k={k}", "", "⚠ モデルキャッシュが見つかりません", "", "---", ""]
            continue

        phantom_s = PHANTOM_STATE[k]
        focus_s   = FOCUS_STATE[k]
        ea        = res["emission_analysis"]
        trans     = res["trans"]

        lines += [
            f"## k={k}  (Phantom: S{phantom_s}, Focus: S{focus_s})",
            "",
            f"**logL**: {res['logL']:.2f}",
            "",
        ]

        # ── 図 ────────────────────────────────────────────────────────
        for fig_key in ["heatmap", "focus", "transition"]:
            fname = res.get(f"plot_{fig_key}")
            if fname:
                alt = {
                    "heatmap":    f"emission heatmap k={k}",
                    "focus":      f"focus state S{focus_s} emission k={k}",
                    "transition": f"transition heatmap k={k}",
                }[fig_key]
                lines.append(f"![{alt}](../results/{fname})")
        lines.append("")

        # ── 全状態の V8 文字集合への確率質量 ──────────────────────────
        lines += [
            f"### 全状態の V8 境界文字への確率質量",
            "",
            "| 状態 | 境界互換 (∩) | 基先頭集合 | 基末尾集合 | 注記 |",
            "|------|-------------|-----------|-----------|------|",
        ]
        for s in range(k):
            compat = ea["boundary_compat_mass"][s]
            start  = ea["base_start_mass"][s]
            end_m  = ea["base_end_mass"][s]
            note   = ""
            if s == phantom_s:
                note = "**[Phantom]**"
            elif s == focus_s:
                note = "**[Focus]** ← 境界集中状態"
            lines.append(
                f"| S{s} | {compat:.1f}% | {start:.1f}% | {end_m:.1f}% | {note} |"
            )
        lines.append("")

        # ── Focus state の上位10文字 ───────────────────────────────────
        topchars = ea["state_topchars"][focus_s]
        lines += [
            f"### S{focus_s} [Focus State] 上位10文字",
            "",
            "| 順位 | 文字 | 放出確率 | V8境界文字? |",
            "|------|------|---------|------------|",
        ]
        for rank, (char, prob) in enumerate(topchars, 1):
            if char in V8_BOUNDARY_COMPAT:
                v8_flag = "✓ 境界互換 (∩)"
            elif char in V8_BASE_START:
                v8_flag = "△ 基先頭のみ"
            elif char in V8_BASE_END:
                v8_flag = "△ 基末尾のみ"
            else:
                v8_flag = "—"
            lines.append(f"| {rank} | `{char}` | {prob:.4f} | {v8_flag} |")
        lines.append("")

        # ── S{focus_s} 確率質量サマリー ───────────────────────────────
        compat_mass = ea["boundary_compat_mass"][focus_s]
        start_mass  = ea["base_start_mass"][focus_s]
        end_mass    = ea["base_end_mass"][focus_s]
        lines += [
            f"### S{focus_s} 確率質量サマリー",
            "",
            f"- **V8 境界互換文字 {{l,r,o,y,s}} への確率質量**: {compat_mass:.1f}%",
            f"- **V8 基先頭集合 全体 への確率質量**: {start_mass:.1f}%",
            f"- **V8 基末尾集合 全体 への確率質量**: {end_mass:.1f}%",
            "",
        ]

        # ── 遷移行列: focus state の行 ──────────────────────────────────
        focus_trans_row = trans[focus_s]
        lines += [
            f"### S{focus_s} → ? 遷移確率（行方向）",
            "",
            "| 遷移先 | 確率 | 注記 |",
            "|--------|------|------|",
        ]
        for s_to in range(k):
            prob = focus_trans_row[s_to]
            note = ""
            if s_to == focus_s:
                note = "**自己遷移**"
            elif s_to == phantom_s:
                note = "[Phantom]"
            lines.append(f"| S{s_to} | {prob:.4f} | {note} |")
        self_trans = focus_trans_row[focus_s]
        lines += [
            "",
            f"> S{focus_s} 自己遷移確率: **{self_trans:.4f}**",
            f"> （自己遷移が低いほど、B-end の直後 B-start で S{focus_s} が抑制されるメカニズムが強く働く）",
            "",
        ]

        # ── 全状態の上位5文字（全体像把握用） ─────────────────────────
        lines += [
            f"### 全状態の上位5文字（全体像）",
            "",
            "| 状態 | 1位 | 2位 | 3位 | 4位 | 5位 | 注記 |",
            "|------|-----|-----|-----|-----|-----|------|",
        ]
        for s in range(k):
            top5 = ea["state_topchars"][s][:5]
            def fmt(char, prob):
                if char in V8_BOUNDARY_COMPAT:
                    return f"**{char}**({prob:.3f})"
                return f"{char}({prob:.3f})"
            top5_str = " | ".join(fmt(c, p) for c, p in top5)
            note = ""
            if s == phantom_s:
                note = "[Phantom]"
            elif s == focus_s:
                note = "[Focus]"
            lines.append(f"| S{s} | {top5_str} | {note} |")
        lines.append("")

        lines += ["---", ""]

    # ── 総括 ──────────────────────────────────────────────────────────
    lines += [
        "## 総括: 交絡 vs 構造効果の最終評価",
        "",
        "本分析は emission probability の直接観察により、以下を検証する:",
        "",
        "| 検証項目 | 判定基準 |",
        "|---------|---------|",
        "| S4 が境界互換文字 {l,r,o,y,s} に特化しているか | 確率質量 > 50% → 文字種クラスタリングが主因 |",
        "| S4 の自己遷移が低いか | < 0.3 → B-end後B-startでの抑制メカニズムが成立 |",
        "| 全状態で V8 境界文字集合への質量が S4 で最大か | S4 が突出 → 境界特化状態 |",
        "",
        "_このレポートは `emission_analysis.py` により自動生成。_",
    ]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("Emission Probability 直接確認 開始")
    log(f"V8境界互換文字: {sorted(V8_BOUNDARY_COMPAT)}")

    # ── データロード (char2idx 構築) ─────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    words_all = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
        conn,
    )["word"].tolist()
    conn.close()

    all_types = sorted(set(w for w in words_all if len(w) >= MIN_WORD_LEN))
    log(f"ユニーク単語数: {len(all_types):,}")

    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR, EOS_CHAR, PAD_CHAR] + raw_chars
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    log(f"語彙サイズ: {len(all_chars)}  (特殊3 + ヴォイニッチ{len(raw_chars)})")
    log(f"ヴォイニッチ文字: {raw_chars}")

    # ── モデルごとに分析 ─────────────────────────────────────────────
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

        emiss = info["emiss"]   # (k, vocab_size)
        trans = info["trans"]   # (k, k)

        log(f"  emiss shape: {emiss.shape},  trans shape: {trans.shape}")

        # ── Emission 分析 ─────────────────────────────────────────
        log("  Emission probability 分析中...")
        ea = analyze_emission(emiss, char2idx, k)
        log(f"  ヴォイニッチ文字数: {len(ea['voynich_chars'])}")

        # Focus state の上位文字をログ出力
        log(f"  S{focus_s} [Focus] 上位10文字:")
        for char, prob in ea["state_topchars"][focus_s]:
            marker = "★" if char in V8_BOUNDARY_COMPAT else " "
            log(f"    {marker} '{char}': {prob:.4f}")
        log(f"  S{focus_s} 境界互換確率質量: {ea['boundary_compat_mass'][focus_s]:.1f}%")
        log(f"  S{focus_s} 自己遷移: {trans[focus_s, focus_s]:.4f}")

        # Phantom state の確認
        log(f"  S{phantom_s} [Phantom] 上位5文字:")
        for char, prob in ea["state_topchars"][phantom_s][:5]:
            log(f"    '{char}': {prob:.4f}")

        # ── プロット ─────────────────────────────────────────────
        log("  ヒートマップ生成中...")
        fname_heatmap = plot_emission_heatmap(
            emiss, ea, k, phantom_s, focus_s, OUT_DIR
        )
        log("  Focus state バーチャート生成中...")
        fname_focus = plot_emission_focus(
            emiss, ea, k, focus_s, phantom_s, OUT_DIR
        )
        log("  遷移行列ヒートマップ生成中...")
        fname_trans = plot_transition_heatmap(
            trans, k, phantom_s, focus_s, OUT_DIR
        )

        all_results[k] = {
            "logL":            info["logL"],
            "emiss":           emiss,
            "trans":           trans,
            "emission_analysis": ea,
            "plot_heatmap":    fname_heatmap,
            "plot_focus":      fname_focus,
            "plot_transition": fname_trans,
        }

    # ── Markdown レポート ─────────────────────────────────────────────
    log("\nMarkdown レポート生成中...")
    md = build_md_report(all_results)
    report_path = OUT_DIR / "emission_analysis_report.md"
    report_path.write_text(md, encoding="utf-8")
    log(f"レポート保存: {report_path}")
    log(f"\n完了。出力先: {OUT_DIR.resolve()}")
