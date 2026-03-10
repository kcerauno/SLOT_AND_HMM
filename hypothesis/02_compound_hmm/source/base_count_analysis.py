"""
base_count_analysis.py
========================
基数別パターン分析: 2基 vs 3基 vs 4基複合語で境界集中が変わるか

interpretation_notes.md Section 7.4 に対応:
  「基数別の分析: 2基複合語 vs 3基複合語で境界集中パターンが変わるか」

分析設計:
  - 複合語を基数（2基/3基/4基）でグループ分け
  - 各グループで B-end / B-start の Focus State (S4/S5) 出現率を算出
  - グループ間の差異を Fisher 検定で検証
  - 3基以上の複合語では「内部境界」vs「外部境界」の比較も実施

出力:
  base_count_rates_k{k}.png          : 基数別 × 位置別 (B-end/B-start) Focus State 出現率
  base_count_inner_outer_k{k}.png    : 3基複合語の内部境界 vs 外部境界比較
  base_count_report.md               : 数値レポート（Markdown）
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    """各境界の (B-end位置, B-start位置, 境界インデックス) を返す。"""
    boundaries = []
    cumlen = 0
    for i, base in enumerate(splits[:-1]):
        cumlen += len(base)
        boundaries.append({
            "bend_pos":    cumlen - 1,
            "bstart_pos":  cumlen,
            "boundary_idx": i,         # 0-indexed: 0=第1境界, 1=第2境界, ...
            "n_bases":     len(splits),
        })
    return boundaries


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
# 3. 基数別集計
# ════════════════════════════════════════════════════════════════════════
def collect_by_base_count(compound_splits, decoded, focus_s, n_bases_list=(2, 3, 4)):
    """
    基数別に B-end / B-start の Focus State 出現データを収集する。

    Returns
    -------
    data : dict
      {
        n_bases: {
          "words":          int,         語数
          "bend_states":    list[int],   全 B-end 位置の状態
          "bstart_states":  list[int],   全 B-start 位置の状態
          "by_boundary_idx": {           境界インデックス別
            boundary_idx: {
              "bend":   list[int],
              "bstart": list[int],
            }
          },
        }
      }
    """
    data = {n: {
        "words": 0,
        "bend_states":   [],
        "bstart_states": [],
        "by_boundary_idx": {},
    } for n in n_bases_list}

    for word, splits in compound_splits.items():
        n = len(splits)
        if n not in data:
            continue
        states = decoded.get(word)
        if states is None:
            continue

        data[n]["words"] += 1
        boundaries = get_boundary_positions(splits)

        for binfo in boundaries:
            b_end_pos   = binfo["bend_pos"]
            b_start_pos = binfo["bstart_pos"]
            bidx        = binfo["boundary_idx"]

            if b_end_pos < len(states) and b_start_pos < len(states):
                s_end   = int(states[b_end_pos])
                s_start = int(states[b_start_pos])
                data[n]["bend_states"].append(s_end)
                data[n]["bstart_states"].append(s_start)

                if bidx not in data[n]["by_boundary_idx"]:
                    data[n]["by_boundary_idx"][bidx] = {"bend": [], "bstart": []}
                data[n]["by_boundary_idx"][bidx]["bend"].append(s_end)
                data[n]["by_boundary_idx"][bidx]["bstart"].append(s_start)

    return data


def compute_focus_rate(states_list, focus_s):
    """focus_s の出現率 (%) とカウントを返す。"""
    if not states_list:
        return 0.0, 0, 0
    arr   = np.array(states_list)
    total = len(arr)
    count = int((arr == focus_s).sum())
    return count / total * 100, count, total


def fisher_2x2_focus(states_a, states_b, focus_s):
    """2グループ間で focus_s 出現率を Fisher 検定。"""
    def _ct(states):
        n_focus = sum(1 for s in states if s == focus_s)
        n_other = len(states) - n_focus
        return n_focus, n_other

    fa, oa = _ct(states_a)
    fb, ob = _ct(states_b)
    if fa + fb == 0:
        return np.nan, np.nan
    odds, p = stats.fisher_exact([[fa, oa], [fb, ob]], alternative="two-sided")
    return float(odds), float(p)


# ════════════════════════════════════════════════════════════════════════
# 4. 可視化
# ════════════════════════════════════════════════════════════════════════
def plot_base_count_rates(base_data, focus_s, k, out_dir):
    """
    基数別 × 位置種別 (B-end / B-start) の Focus State 出現率を並べたバーチャート。
    """
    n_bases_list = sorted(base_data.keys())
    pos_types = [("bend_states", "B-end"), ("bstart_states", "B-start")]

    x       = np.arange(len(n_bases_list))
    width   = 0.38
    colors  = {"B-end": "#C0392B", "B-start": "#E59866"}

    fig, ax = plt.subplots(figsize=(max(8, len(n_bases_list) * 3), 5))

    for offset, (state_key, label) in zip([-0.5, 0.5], pos_types):
        rates  = []
        totals = []
        for n in n_bases_list:
            rate, cnt, total = compute_focus_rate(base_data[n][state_key], focus_s)
            rates.append(rate)
            totals.append(total)

        bars = ax.bar(
            x + offset * width, rates, width,
            label=label, color=colors[label], alpha=0.88, edgecolor="white"
        )
        for bar, rate, total in zip(bars, rates, totals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{rate:.1f}%\n(n={total:,})",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    xlabels = [f"{n}基複合語\n({base_data[n]['words']:,}語)" for n in n_bases_list]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel(f"S{focus_s} 出現率 (%)", fontsize=11)
    ax.set_title(
        f"k={k}: 基数別 Focus State S{focus_s} 出現率\n"
        f"赤=B-end（基末尾）、オレンジ=B-start（基先頭）",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    all_rates = [
        compute_focus_rate(base_data[n][sk], focus_s)[0]
        for n in n_bases_list for sk, _ in pos_types
    ]
    ax.set_ylim(0, max(all_rates) * 1.35 + 2)
    plt.tight_layout()

    fname = f"base_count_rates_k{k}.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


def plot_inner_outer_boundaries(base_data, focus_s, k, out_dir):
    """
    3基複合語における境界インデックス別 (第1境界 vs 第2境界) 比較。
    4基複合語も同様に境界インデックス別で比較する。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, n_bases in enumerate([3, 4]):
        ax = axes[ax_idx]
        bdata = base_data.get(n_bases)
        if bdata is None or not bdata["by_boundary_idx"]:
            ax.text(0.5, 0.5, f"{n_bases}基複合語データなし",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{n_bases}基複合語（データなし）")
            continue

        boundary_indices = sorted(bdata["by_boundary_idx"].keys())
        n_boundaries     = len(boundary_indices)

        x      = np.arange(n_boundaries)
        width  = 0.38
        colors = {"bend": "#C0392B", "bstart": "#E59866"}
        labels = {"bend": "B-end（基末尾）", "bstart": "B-start（基先頭）"}

        for offset, pos_type in zip([-0.5, 0.5], ["bend", "bstart"]):
            rates  = []
            totals = []
            for bidx in boundary_indices:
                bd_data = bdata["by_boundary_idx"].get(bidx, {"bend": [], "bstart": []})
                states  = bd_data[pos_type]
                rate, cnt, total = compute_focus_rate(states, focus_s)
                rates.append(rate)
                totals.append(total)

            bars = ax.bar(
                x + offset * width, rates, width,
                label=labels[pos_type], color=colors[pos_type], alpha=0.88, edgecolor="white"
            )
            for bar, rate, total in zip(bars, rates, totals):
                if total > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.4,
                        f"{rate:.1f}%\n(n={total:,})",
                        ha="center", va="bottom", fontsize=7, fontweight="bold",
                    )

        xlabels = [f"境界{bidx+1}" for bidx in boundary_indices]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=11)
        ax.set_ylabel(f"S{focus_s} 出現率 (%)", fontsize=10)
        ax.set_title(
            f"k={k}: {n_bases}基複合語 境界位置別 S{focus_s} 出現率\n"
            f"（語数: {bdata['words']:,}）",
            fontsize=11,
        )
        ax.legend(fontsize=9)

    plt.suptitle(
        f"k={k}: 複合語内境界位置別の Focus State S{focus_s} 出現率",
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    fname = f"base_count_inner_outer_k{k}.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


# ════════════════════════════════════════════════════════════════════════
# 5. Markdown レポート
# ════════════════════════════════════════════════════════════════════════
def build_md_report(results_by_k, n_bases_list=(2, 3, 4)):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 基数別パターン分析: 2基 vs 3基 vs 4基複合語",
        "",
        f"**生成日時**: {now}",
        "**スクリプト**: `hypothesis/02_compound_hmm/source/base_count_analysis.py`",
        "",
        "---",
        "",
        "## 検証目的",
        "",
        "`interpretation_notes.md` Section 7.4 に対応:",
        "",
        "> 基数別の分析: 2基複合語 vs 3基複合語で境界集中パターンが変わるか",
        "",
        "複合語構造が真に存在する場合、境界での Focus State 集中は基数によらず安定するはず。",
        "逆に偶然や文字制約の産物であれば、基数による系統的変化が予想される。",
        "",
        "また3基・4基複合語では「内部境界 vs 外部境界」の比較により、",
        "語全体の端に近い境界と中間の境界でパターンが異なるかも確認する。",
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
        base_data = res["base_data"]

        lines += [
            f"## k={k}  (Phantom: S{phantom_s}, Focus: S{focus_s})",
            "",
        ]

        for fig_key in ["rates", "inner_outer"]:
            fname = res.get(f"plot_{fig_key}")
            alt = {
                "rates":       f"base count rates k={k}",
                "inner_outer": f"inner outer boundaries k={k}",
            }[fig_key]
            if fname:
                lines.append(f"![{alt}](../results/{fname})")
        lines.append("")

        # ── 語数テーブル ───────────────────────────────────────────
        lines += [
            "### 語数・境界数",
            "",
            "| 基数 | 語数 | 境界数/語 | 総境界数 |",
            "|------|------|---------|---------|",
        ]
        for n in n_bases_list:
            bd = base_data.get(n, {})
            words   = bd.get("words", 0)
            n_bound = n - 1
            total_b = len(bd.get("bend_states", []))
            lines.append(f"| {n}基 | {words:,} | {n_bound} | {total_b:,} |")
        lines.append("")

        # ── B-end / B-start 基数別出現率テーブル ──────────────────
        lines += [
            f"### S{focus_s} 出現率（基数別 × 位置別）",
            "",
            "| 基数 | B-end 率 | B-end n | B-start 率 | B-start n | B-end > B-start? |",
            "|------|---------|--------|-----------|----------|----------------|",
        ]
        for n in n_bases_list:
            bd = base_data.get(n, {})
            bend_rate, bend_cnt, bend_tot   = compute_focus_rate(bd.get("bend_states", []),   focus_s)
            bstart_rate, bstart_cnt, bstart_tot = compute_focus_rate(bd.get("bstart_states", []), focus_s)
            flag = "✓" if bend_rate > bstart_rate else ("=" if abs(bend_rate - bstart_rate) < 0.5 else "✗")
            lines.append(
                f"| {n}基 "
                f"| {bend_rate:.1f}% "
                f"| {bend_tot:,} "
                f"| {bstart_rate:.1f}% "
                f"| {bstart_tot:,} "
                f"| {flag} |"
            )
        lines.append("")

        # ── Fisher 検定（基数間） ──────────────────────────────────
        lines += [
            "### Fisher 検定: 基数間の B-end 出現率比較",
            "",
            "| 比較 | 率A | 率B | p値 | オッズ比 | 判定 |",
            "|------|-----|-----|-----|---------|------|",
        ]
        pairs = []
        ns = [n for n in n_bases_list if base_data.get(n, {}).get("bend_states")]
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                na, nb = ns[i], ns[j]
                sa = base_data[na]["bend_states"]
                sb = base_data[nb]["bend_states"]
                odds, p = fisher_2x2_focus(sa, sb, focus_s)
                rate_a, _, _ = compute_focus_rate(sa, focus_s)
                rate_b, _, _ = compute_focus_rate(sb, focus_s)
                if np.isnan(p):
                    p_str, verdict = "N/A", "—"
                elif p < 0.001:
                    p_str, verdict = f"{p:.2e}", "**有意差あり**"
                elif p < 0.05:
                    p_str, verdict = f"{p:.4f}", "有意差あり"
                else:
                    p_str, verdict = f"{p:.4f}", "有意差なし"
                lines.append(
                    f"| {na}基 vs {nb}基 (B-end) "
                    f"| {rate_a:.1f}% "
                    f"| {rate_b:.1f}% "
                    f"| {p_str} "
                    f"| {odds:.3f} if not np.isnan(odds) else 'N/A'"
                    f"| {verdict} |"
                )
        lines.append("")

        # ── 3基複合語の境界インデックス別詳細 ─────────────────────
        for n_bases in [3, 4]:
            bd_detail = base_data.get(n_bases, {}).get("by_boundary_idx", {})
            if not bd_detail:
                continue
            lines += [
                f"### {n_bases}基複合語: 境界インデックス別 S{focus_s} 出現率",
                "",
                f"| 境界 | B-end 率 | B-end n | B-start 率 | B-start n |",
                "|------|---------|--------|-----------|----------|",
            ]
            for bidx in sorted(bd_detail.keys()):
                bd = bd_detail[bidx]
                r_e, c_e, t_e = compute_focus_rate(bd["bend"],   focus_s)
                r_s, c_s, t_s = compute_focus_rate(bd["bstart"], focus_s)
                lines.append(
                    f"| 境界{bidx+1} (base{bidx+1}/base{bidx+2}間) "
                    f"| {r_e:.1f}% | {t_e:,} "
                    f"| {r_s:.1f}% | {t_s:,} |"
                )
            lines.append("")

        lines += ["---", ""]

    # 総括
    lines += [
        "## 総括",
        "",
        "| 問い | 期待 (複合語構造が実在) | 期待 (文字制約の偶然) |",
        "|------|--------------------|--------------------|",
        "| 基数間で B-end 率が安定するか | 安定（構造は基数によらない） | 変動あり |",
        "| B-end > B-start が全基数で成立するか | 全基数で成立 | 特定基数のみ |",
        "| 3基: 第1境界 ≈ 第2境界か | 両境界で同程度集中 | 非対称になる可能性 |",
        "",
        "_本レポートは `base_count_analysis.py` により自動生成。_",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════════════════
N_BASES_LIST = (2, 3, 4)

if __name__ == "__main__":
    log("基数別パターン分析 開始")

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

    # ── V8 複合語（基数別にカウント） ────────────────────────────────
    log("V8複合語を特定中...")
    compound_splits = {}
    for w in all_types:
        splits = find_v8_splits_first(w)
        if splits is not None:
            compound_splits[w] = splits

    count_by_n = {}
    for w, splits in compound_splits.items():
        n = len(splits)
        count_by_n[n] = count_by_n.get(n, 0) + 1
    log(f"  総複合語数: {len(compound_splits):,}")
    for n in sorted(count_by_n):
        log(f"    {n}基: {count_by_n[n]:,} 語")

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

        # ── 基数別集計 ────────────────────────────────────────────
        log("  基数別集計中...")
        base_data = collect_by_base_count(compound_splits, decoded, focus_s, N_BASES_LIST)

        for n in N_BASES_LIST:
            bd = base_data[n]
            bend_rate,   _, bend_tot   = compute_focus_rate(bd["bend_states"],   focus_s)
            bstart_rate, _, bstart_tot = compute_focus_rate(bd["bstart_states"], focus_s)
            log(f"    {n}基 ({bd['words']:,}語): "
                f"B-end={bend_rate:.1f}%(n={bend_tot}) "
                f"B-start={bstart_rate:.1f}%(n={bstart_tot})")

        # 3基複合語の境界インデックス別
        for n_bases in [3, 4]:
            bd3 = base_data.get(n_bases, {}).get("by_boundary_idx", {})
            if bd3:
                log(f"    {n_bases}基複合語 境界インデックス別:")
                for bidx in sorted(bd3.keys()):
                    bd = bd3[bidx]
                    r_e, _, t_e = compute_focus_rate(bd["bend"],   focus_s)
                    r_s, _, t_s = compute_focus_rate(bd["bstart"], focus_s)
                    log(f"      境界{bidx+1}: B-end={r_e:.1f}%(n={t_e})  B-start={r_s:.1f}%(n={t_s})")

        # Fisher 検定（基数間）
        log("  Fisher 検定（基数間）:")
        ns = [n for n in N_BASES_LIST if base_data.get(n, {}).get("bend_states")]
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                na, nb = ns[i], ns[j]
                sa = base_data[na]["bend_states"]
                sb = base_data[nb]["bend_states"]
                odds, p = fisher_2x2_focus(sa, sb, focus_s)
                ra, _, _ = compute_focus_rate(sa, focus_s)
                rb, _, _ = compute_focus_rate(sb, focus_s)
                log(f"    {na}基 vs {nb}基 (B-end): {ra:.1f}% vs {rb:.1f}%  p={p:.3e} odds={odds:.3f}")

        # ── プロット ──────────────────────────────────────────────
        fname_rates      = plot_base_count_rates(base_data, focus_s, k, OUT_DIR)
        fname_inner_outer = plot_inner_outer_boundaries(base_data, focus_s, k, OUT_DIR)

        all_results[k] = {
            "base_data":         base_data,
            "plot_rates":        fname_rates,
            "plot_inner_outer":  fname_inner_outer,
        }

    # ── Markdown レポート ─────────────────────────────────────────────
    log("\nMarkdown レポート生成中...")
    md = build_md_report(all_results, N_BASES_LIST)
    report_path = OUT_DIR / "base_count_report.md"
    report_path.write_text(md, encoding="utf-8")
    log(f"レポート保存: {report_path}")
    log(f"\n完了。出力先: {OUT_DIR.resolve()}")
