"""
trigram_boundary_analysis.py
=============================
Trigram HMM 境界集中分析（正確版）

hypothesis/04_re-compound_hmm/source/correct_boundary_analysis.py の Trigram 対応版。

【修正点】
  Bug 1: find_v8_splits_first() 偽陽性
         → compound_words.txt を直接ロード（正解 3,363 語のみ使用）

  Bug 2: 近似 Trigram Viterbi（correct_role_analysis.py の logsumexp 周辺化）
         → trigram_hmm_fast.py の TrigramHMM_Batched.viterbi() と同一ロジックを採用
            （(k×k) デルタ行列で exact Trigram Viterbi）

  Bug 3: バックトラッキング off-by-one（trigram_hmm_fast.py L386 と共通）
         → psi_list[t_back - 2] を psi_list[t_back - 1] に修正

  Bug 4: start_trans 未ロード
         → npz から start_trans を含めてロード

分析内容:
  1. Viterbi 占有率（全 corpus 対象）→ Phantom State 同定
  2. 境界集中分析（複合語 3,363 語対象）
     - 境界位置 vs 非境界位置の状態分布 chi-square 検定
     - 各状態の境界集中率 Fisher exact 検定

実行:
  cd /home/practi/work_voy
  PYTHONPATH=/home/practi/work_voy/.venv/lib/python3.10/site-packages \\
    python3.10 hypothesis/05_re-compound_hmm_trigram/source/trigram_boundary_analysis.py
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
OUT_DIR             = Path("hypothesis/05_re-compound_hmm_trigram/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST       = [7, 8]
MIN_WORD_LEN = 2
BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"


# ════════════════════════════════════════════════════════════════════════
# 1. データロード
# ════════════════════════════════════════════════════════════════════════
def load_compound_splits(path: Path) -> dict:
    """
    compound_words.txt をパース。
    形式: "[N基] word  ->  base1 + base2 + ..."
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


def get_boundary_positions(splits: tuple) -> tuple:
    """B-end（基末尾位置）と B-start（基先頭位置）を返す"""
    boundary_end, boundary_start = set(), set()
    cumlen = 0
    for base in splits[:-1]:
        cumlen += len(base)
        boundary_end.add(cumlen - 1)   # 基の最終文字（0-indexed）
        boundary_start.add(cumlen)      # 次の基の先頭文字
    return boundary_end, boundary_start


# ════════════════════════════════════════════════════════════════════════
# 2. モデルロード
# ════════════════════════════════════════════════════════════════════════
def load_trigram_model(k: int) -> dict | None:
    """
    trigram_k{k}.npz をロード。
    trigram_hmm_fast.py の save_model() と同一形式:
      start, start_trans, trans (k×k×k), emiss, logL
    """
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
# 3. Exact Trigram Viterbi（trigram_hmm_fast.py の viterbi() を修正）
# ════════════════════════════════════════════════════════════════════════
def viterbi_trigram(model: dict, X_np: np.ndarray) -> list:
    """
    Exact Trigram Viterbi デコード。
    trigram_hmm_fast.py TrigramHMM_Batched.viterbi() と同一ロジックだが
    バックトラッキングの off-by-one バグ（t_back-2 → t_back-1）を修正済み。

    Parameters
    ----------
    model  : load_trigram_model() の返り値
    X_np   : int32 配列（BOS + 観測 + EOS）

    Returns
    -------
    path : list[int]  長さ T の Viterbi 最適状態列
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

    # t=0
    log_delta_1d = log_start + log_emiss[:, X[0]]   # (k,)

    if T == 1:
        return [torch.argmax(log_delta_1d).item()]

    # t=1  log_delta[i, j] = joint(s_0=i, s_1=j) の最大対数確率
    log_delta = (log_delta_1d.unsqueeze(1)
                 + log_start_trans
                 + log_emiss[:, X[1]].unsqueeze(0))  # (k, k)

    psi_list = []  # psi_list[idx][j, l] = best s_{t-2} at time t=idx+2
    for t in range(2, T):
        vals = log_delta.unsqueeze(2) + log_transmat   # (k, k, k): (s_{t-2}, s_{t-1}, s_t)
        max_vals, argmax_i = torch.max(vals, dim=0)    # (k, k): max over s_{t-2}
        new_delta = max_vals + log_emiss[:, X[t]].unsqueeze(0)  # (k, k)
        psi_list.append(argmax_i.cpu())
        log_delta = new_delta

    # バックトラック
    flat_idx = torch.argmax(log_delta)
    j_last = (flat_idx // k).item()   # s_{T-2}
    l_last = (flat_idx % k).item()    # s_{T-1}

    path = [0] * T
    path[T - 1] = l_last
    path[T - 2] = j_last

    # Bug 3 修正: psi_list[t_back - 2] → psi_list[t_back - 1]
    for t_back in range(T - 2, 1, -1):
        path[t_back - 1] = psi_list[t_back - 1][path[t_back], path[t_back + 1]].item()

    return path


# ════════════════════════════════════════════════════════════════════════
# 4. Viterbi 占有率分析（全 corpus 対象）
# ════════════════════════════════════════════════════════════════════════
def compute_viterbi_occupancy(model: dict, all_words: list, char2idx: dict) -> dict:
    """全語のデコードから各状態の Viterbi 出現カウントを集計"""
    k = model["k"]
    counts = np.zeros(k, dtype=int)
    total  = 0
    skip   = 0

    for word in all_words:
        if not all(c in char2idx for c in word):
            skip += 1
            continue
        seq = np.array(
            [char2idx[BOS_CHAR]] + [char2idx[c] for c in word] + [char2idx[EOS_CHAR]],
            dtype=np.int32,
        )
        try:
            path = viterbi_trigram(model, seq)
            states = path[1:-1]   # BOS/EOS を除く
            for s in states:
                counts[s] += 1
                total += 1
        except Exception:
            skip += 1

    log(f"  Viterbi 占有率: {total:,} 観測, スキップ {skip}")
    occupancy = counts / total if total > 0 else np.zeros(k)
    return {"counts": counts, "total": total, "occupancy": occupancy}


# ════════════════════════════════════════════════════════════════════════
# 5. 境界集中分析（複合語対象）
# ════════════════════════════════════════════════════════════════════════
def collect_boundary_states(model: dict, compound_splits: dict, char2idx: dict) -> dict:
    """
    複合語の各位置を「境界位置（B-end or B-start）」と「非境界位置」に分類し
    それぞれの状態リストを返す。
    """
    k = model["k"]
    boundary_states     = []
    non_boundary_states = []

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

        states = path[1:-1]   # BOS/EOS 除く、長さ = len(word)
        if len(states) < 2:
            continue

        bd_end, bd_start = get_boundary_positions(splits)
        boundary_pos = bd_end | bd_start

        for pos, state in enumerate(states):
            if pos in boundary_pos:
                boundary_states.append(int(state))
            else:
                non_boundary_states.append(int(state))

    return {
        "boundary":     boundary_states,
        "non_boundary": non_boundary_states,
    }


def chi_square_boundary(boundary_states: list, non_boundary_states: list, k: int) -> dict:
    """境界 vs 非境界の状態分布に対する chi-square 独立性検定"""
    bc = np.bincount(boundary_states,     minlength=k).astype(float)
    nc = np.bincount(non_boundary_states, minlength=k).astype(float)

    # 期待頻度が 0 の行を除外
    valid = (bc + nc) > 0
    table = np.vstack([bc[valid], nc[valid]])

    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"chi2": chi2, "p": p, "dof": dof,
            "n_boundary": int(bc.sum()), "n_non_boundary": int(nc.sum())}


def fisher_per_state(boundary_states: list, non_boundary_states: list, k: int) -> list:
    """各状態について境界 vs 非境界の Fisher 正確検定（一側: greater）"""
    bc = np.bincount(boundary_states,     minlength=k).astype(float)
    nc = np.bincount(non_boundary_states, minlength=k).astype(float)
    total_b = bc.sum()
    total_n = nc.sum()

    results = []
    for s in range(k):
        a, b_ = int(bc[s]), int(total_b - bc[s])
        c, d  = int(nc[s]), int(total_n - nc[s])
        if a + c == 0:
            odds, p = np.nan, np.nan
        else:
            odds, p = stats.fisher_exact([[a, b_], [c, d]], alternative="greater")
        rate_b = bc[s] / total_b * 100 if total_b > 0 else 0.0
        rate_n = nc[s] / total_n * 100 if total_n > 0 else 0.0
        results.append({
            "state":    s,
            "n_boundary":     int(bc[s]),
            "n_non_boundary": int(nc[s]),
            "rate_boundary":  rate_b,
            "rate_non_boundary": rate_n,
            "enrichment": (rate_b / rate_n) if rate_n > 0 else np.nan,
            "odds":  odds,
            "p":     p,
        })
    return results


# ════════════════════════════════════════════════════════════════════════
# 6. 可視化
# ════════════════════════════════════════════════════════════════════════
def plot_viterbi_occupancy(occupancy: np.ndarray, k: int, out_path: Path):
    state_labels = [f"S{s}" for s in range(k)]
    phantom_mask = occupancy == 0.0
    colors = ["#E84040" if phantom_mask[s] else "#5588BB" for s in range(k)]

    fig, ax = plt.subplots(figsize=(max(7, k * 1.2), 4))
    ax.bar(np.arange(k), occupancy * 100, color=colors, alpha=0.88,
           edgecolor="white", linewidth=1.2)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(state_labels)
    ax.set_xlabel("隠れ状態", fontsize=10)
    ax.set_ylabel("Viterbi 占有率 (%)", fontsize=10)
    phantoms = [f"S{s}" for s in range(k) if phantom_mask[s]]
    title_note = f"  Phantom: {', '.join(phantoms)}" if phantoms else "  Phantom なし"
    ax.set_title(f"[正確版 Trigram]  k={k}  Viterbi 占有率\n{title_note}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_distribution(fisher_results: list, k: int, out_path: Path):
    state_labels = [f"S{s}" for s in range(k)]
    rates_b = [r["rate_boundary"]     for r in fisher_results]
    rates_n = [r["rate_non_boundary"] for r in fisher_results]

    x = np.arange(k)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, k * 1.3), 5))
    ax.bar(x - width / 2, rates_b, width, label="境界位置",    color="#E84040", alpha=0.85)
    ax.bar(x + width / 2, rates_n, width, label="非境界位置",  color="#5588BB", alpha=0.85)

    for r in fisher_results:
        s = r["state"]
        if not np.isnan(r["p"]) and r["p"] < 0.05:
            y_max = max(rates_b[s], rates_n[s])
            ax.annotate("*", xy=(s, y_max + 0.5), ha="center", fontsize=13, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(state_labels)
    ax.set_xlabel("隠れ状態", fontsize=10)
    ax.set_ylabel("出現率 (%)", fontsize=10)
    ax.set_title(f"[正確版 Trigram]  k={k}  境界 vs 非境界 状態分布\n(*: Fisher p<0.05)", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# 7. Markdown レポート
# ════════════════════════════════════════════════════════════════════════
def build_report(results_by_k: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# [正確版] Trigram HMM 境界集中分析レポート",
        "",
        f"生成日時: {now}",
        "",
        "## 修正内容",
        "",
        "- Bug 1: find_v8_splits_first() 偽陽性 → compound_words.txt を直接ロード",
        "- Bug 2: 近似 Trigram Viterbi (logsumexp 周辺化) → trigram_hmm_fast.py と同一の exact Viterbi",
        "- Bug 3: バックトラッキング off-by-one → psi_list[t_back-1] に修正",
        "- Bug 4: start_trans 未ロード → npz から正しくロード",
        "",
        "---",
        "",
        "## Viterbi 占有率サマリー",
        "",
        "| k | Phantom State (0% 占有) | 非ゼロ状態数 |",
        "|---|------------------------|------------|",
    ]
    for k, res in results_by_k.items():
        occ = res["occupancy"]["occupancy"]
        phantoms = [f"S{s}" for s in range(k) if occ[s] == 0.0]
        non_zero = k - len(phantoms)
        lines.append(f"| {k} | {', '.join(phantoms) if phantoms else 'なし'} | {non_zero} |")

    lines += ["", "---", ""]

    for k, res in results_by_k.items():
        occ      = res["occupancy"]
        chi_res  = res["chi_square"]
        fisher   = res["fisher"]

        lines += [
            f"## k={k}",
            "",
            "### Viterbi 占有率",
            "",
            "| 状態 | 出現カウント | 占有率 | 判定 |",
            "|------|-----------|-------|------|",
        ]
        for s in range(k):
            rate = occ["occupancy"][s] * 100
            cnt  = occ["counts"][s]
            tag  = "**Phantom**" if cnt == 0 else ""
            lines.append(f"| S{s} | {cnt:,} | {rate:.2f}% | {tag} |")

        lines += [
            "",
            "### 境界集中 chi-square 検定（境界 vs 非境界）",
            "",
            f"- chi2 = {chi_res['chi2']:.2f},  p = {chi_res['p']:.3e},  dof = {chi_res['dof']}",
            f"- 境界位置数: {chi_res['n_boundary']:,}  /  非境界位置数: {chi_res['n_non_boundary']:,}",
            "",
            "### 各状態 Fisher 正確検定（境界集中、一側）",
            "",
            "| 状態 | 境界率 | 非境界率 | Enrichment | Fisher p | 判定 |",
            "|------|--------|---------|-----------|---------|------|",
        ]
        for r in fisher:
            p_str = f"{r['p']:.2e}" if not np.isnan(r["p"]) else "N/A"
            enr   = f"{r['enrichment']:.2f}" if not np.isnan(r["enrichment"]) else "N/A"
            occ_tag = "**Phantom**" if occ["counts"][r["state"]] == 0 else ""
            sig_tag = "**有意**" if (not np.isnan(r["p"]) and r["p"] < 0.05) else ""
            verdict = f"{occ_tag} {sig_tag}".strip()
            lines.append(
                f"| S{r['state']} | {r['rate_boundary']:.2f}% | {r['rate_non_boundary']:.2f}% "
                f"| {enr} | {p_str} | {verdict} |"
            )
        lines += ["", "---", ""]

    lines += ["_本レポートは trigram_boundary_analysis.py により自動生成。_"]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("[正確版] Trigram HMM 境界集中分析 開始")

    # 正解複合語ロード（Bug 1 修正）
    log(f"正解複合語ロード: {COMPOUND_SPLIT_PATH}")
    compound_splits_all = load_compound_splits(COMPOUND_SPLIT_PATH)
    log(f"  正解複合語数: {len(compound_splits_all):,} 語")

    # DB から全単語をロード（char2idx 構築 + Viterbi 占有率用）
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
    log(f"ユニーク単語数: {len(all_types):,}  語彙サイズ: {len(all_chars)}")

    # char2idx でフィルタした複合語
    compound_splits = {
        w: splits for w, splits in compound_splits_all.items()
        if all(c in char2idx for c in w)
    }
    log(f"有効複合語数: {len(compound_splits):,}")

    results_by_k = {}

    for k in K_LIST:
        log(f"\n{'─'*60}")
        log(f"Trigram k={k}")
        log(f"{'─'*60}")

        model = load_trigram_model(k)
        if model is None:
            continue
        log(f"  モデルロード完了 (logL={model['logL']:.2f})")

        # Viterbi 占有率（全 corpus）
        log("  Viterbi 占有率を計算中（全語）...")
        occ = compute_viterbi_occupancy(model, all_types, char2idx)
        for s in range(k):
            tag = "  ← Phantom" if occ["counts"][s] == 0 else ""
            log(f"    S{s}: {occ['occupancy'][s]:.3%}{tag}")

        # 境界集中分析（複合語のみ）
        log(f"  境界集中分析（複合語 {len(compound_splits):,} 語）...")
        bnd = collect_boundary_states(model, compound_splits, char2idx)
        log(f"    境界位置: {len(bnd['boundary']):,}  /  非境界: {len(bnd['non_boundary']):,}")

        chi_res = chi_square_boundary(bnd["boundary"], bnd["non_boundary"], k)
        log(f"  chi2={chi_res['chi2']:.2f}  p={chi_res['p']:.3e}")

        fisher = fisher_per_state(bnd["boundary"], bnd["non_boundary"], k)
        for r in fisher:
            if not np.isnan(r["p"]) and r["p"] < 0.05:
                log(f"    S{r['state']}: 境界率={r['rate_boundary']:.2f}%  "
                    f"非境界率={r['rate_non_boundary']:.2f}%  p={r['p']:.2e}")

        results_by_k[k] = {
            "occupancy":  occ,
            "chi_square": chi_res,
            "fisher":     fisher,
        }

        # 可視化
        plot_viterbi_occupancy(
            occ["occupancy"], k,
            OUT_DIR / f"viterbi_occupancy_k{k}.png",
        )
        log(f"  Saved: viterbi_occupancy_k{k}.png")

        plot_boundary_distribution(
            fisher, k,
            OUT_DIR / f"boundary_dist_k{k}.png",
        )
        log(f"  Saved: boundary_dist_k{k}.png")

    # レポート生成
    report = build_report(results_by_k)
    report_path = OUT_DIR / "boundary_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"\nレポート保存: {report_path}")
    log("✓ 完了。出力先: " + str(OUT_DIR.resolve()))
