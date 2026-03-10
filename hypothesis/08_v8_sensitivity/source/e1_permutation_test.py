"""
e1_permutation_test.py
========================
提案 E-1: 置換検定（境界ラベルのシャッフル）

目的:
  compound_words.txt の境界位置をランダムにシャッフルした帰無分布を構築し、
  実際の Fisher p 値が帰無分布の上位 5% に入るかを確認する。

仮説:
  H₀: 境界ラベルが語内でランダムに配置されても同程度の B-start 特化シグナルが得られる
  H₁: 実際の V8 文法境界位置にのみ有意なシグナルが存在する

効率化:
  Viterbi デコードは一度だけ実行し、境界ラベルのみをシャッフルする。
  → 1000 回の置換が数秒〜数分で完了。

統計量:
  -log10(Fisher p) for B-start vs S-head（比較①）
  Focus State は各置換でも data-driven に選択（hypothesis/05 と同一手順）

実行:
  cd /home/practi/work_voy
  PYTHONPATH=/home/practi/work_voy/.venv/lib/python3.10/site-packages \\
    python3.10 hypothesis/08_v8_sensitivity/source/e1_permutation_test.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ── hypothesis/06 の common.py を再利用 ──────────────────────────────────────
_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent.parent
SRC06 = ROOT / "hypothesis/06_state_characterization/source"
sys.path.insert(0, str(SRC06))

import common  # noqa: E402

# ── 設定 ──────────────────────────────────────────────────────────────────────
OUT_DIR    = ROOT / "hypothesis/08_v8_sensitivity/results/e1_permutation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM     = 1000
BASE_SEED  = 42
K_LIST     = [7, 8]
MODEL_TYPES = ["trigram", "bigram"]


# ════════════════════════════════════════════════════════════════════════════
# 1. グループ収集（decoded を受け取り、境界スプリット情報を使う）
# ════════════════════════════════════════════════════════════════════════════
def collect_groups(compound_splits: dict, decoded_compound: dict,
                   decoded_single: dict, single_words: list) -> dict:
    """4 グループ（B-start, B-end, S-head, S-mid）の状態リストを収集。"""
    b_start, b_end, s_head, s_mid = [], [], [], []

    for word, splits in compound_splits.items():
        states = decoded_compound.get(word)
        if states is None:
            continue
        bd_end, bd_start = common.get_boundary_positions(splits)
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


# ════════════════════════════════════════════════════════════════════════════
# 2. 統計量の計算
# ════════════════════════════════════════════════════════════════════════════
def compute_statistic(groups: dict, k: int, occupancy: np.ndarray) -> float | None:
    """
    data-driven Focus State（B-start 出現率最大、Phantom 除外）を選択し、
    B-start vs S-head の 2x2 分割表から chi2 統計量を返す（上限なし）。

    注記: -log10(Fisher p) はキャップ (300) で飽和するため chi2 を採用。
    chi2 = N*(a*d - b*c)^2 / (r1*r2*c1*c2) を直接計算。
    Focus State が特定できない or 観測数 0 の場合は None。
    """
    if len(groups["B-start"]) == 0 or len(groups["S-head"]) == 0:
        return None

    b_start_counts = np.bincount(groups["B-start"], minlength=k).astype(float)
    rates = b_start_counts.copy()
    for s in range(k):
        if occupancy[s] == 0:
            rates[s] = -1.0  # Phantom を候補から除外
    focus = int(np.argmax(rates))

    ca = np.bincount(groups["B-start"], minlength=k)
    cb = np.bincount(groups["S-head"],  minlength=k)
    a = float(ca[focus]);  b = float(ca.sum()) - a   # B-start: focus / other
    c = float(cb[focus]);  d = float(cb.sum()) - c   # S-head:  focus / other

    if a + c == 0:
        return None

    N  = a + b + c + d
    r1 = a + b; r2 = c + d  # row sums
    c1 = a + c; c2 = b + d  # col sums

    if r1 == 0 or r2 == 0 or c1 == 0 or c2 == 0:
        return None

    chi2 = N * (a * d - b * c) ** 2 / (r1 * r2 * c1 * c2)
    return float(chi2)


# ════════════════════════════════════════════════════════════════════════════
# 3. 境界シャッフル
# ════════════════════════════════════════════════════════════════════════════
def shuffle_splits(compound_splits: dict, rng: np.random.Generator) -> dict:
    """
    各複合語について基数（n_bases）を固定し、境界位置をランダム化。
    語長・基数分布は保持するが、境界の場所は帰無仮説下でランダム。
    """
    shuffled = {}
    for word, bases in compound_splits.items():
        n_bases = len(bases)
        L = len(word)
        if n_bases <= 1 or L <= n_bases:
            # シャッフル不可能（すべての位置が境界になってしまう）→ そのまま
            shuffled[word] = bases
            continue
        # 1..L-1 からランダムに n_bases-1 個を選択（重複なし）
        split_pos = sorted(rng.choice(np.arange(1, L), size=n_bases - 1, replace=False).tolist())
        new_bases = []
        prev = 0
        for pos in split_pos:
            new_bases.append(word[prev:pos])
            prev = pos
        new_bases.append(word[prev:])
        shuffled[word] = tuple(new_bases)
    return shuffled


# ════════════════════════════════════════════════════════════════════════════
# 4. 置換検定メインループ
# ════════════════════════════════════════════════════════════════════════════
def run_permutation_test(model_type: str, k: int, model: dict,
                         compound_splits: dict, decoded_compound: dict,
                         decoded_single: dict, single_words: list,
                         occupancy: np.ndarray) -> dict:
    """
    置換検定を実行して帰無分布と置換 p 値を返す。
    """
    common.log(f"  [{model_type} k={k}] 実観測統計量計算中...")
    real_groups   = collect_groups(compound_splits, decoded_compound, decoded_single, single_words)
    observed_stat = compute_statistic(real_groups, k, occupancy)
    fs_real       = _get_focus_state(real_groups, k, occupancy)
    common.log(f"    Focus State: S{fs_real}  観測統計量: {observed_stat:.4f}" if observed_stat else
               f"    観測統計量: None")

    rng = np.random.default_rng(BASE_SEED + k + (10 if model_type == "bigram" else 0))
    null_stats  = []
    null_focus  = []

    common.log(f"  [{model_type} k={k}] {N_PERM} 回の置換検定実行中...")
    for i in range(N_PERM):
        shuffled  = shuffle_splits(compound_splits, rng)
        groups    = collect_groups(shuffled, decoded_compound, decoded_single, single_words)
        stat      = compute_statistic(groups, k, occupancy)
        fs_null   = _get_focus_state(groups, k, occupancy)
        if stat is not None:
            null_stats.append(stat)
            null_focus.append(fs_null)
        if (i + 1) % 200 == 0:
            common.log(f"    {i + 1}/{N_PERM} 完了  (有効: {len(null_stats)})")

    null_stats = np.array(null_stats)
    if observed_stat is not None and len(null_stats) > 0:
        p_perm = float((null_stats >= observed_stat).mean())
    else:
        p_perm = float("nan")

    common.log(f"  [{model_type} k={k}] 置換 p 値: {p_perm:.4f}  "
               f"(observed={observed_stat:.3f}, null median={np.median(null_stats):.3f})")

    return {
        "model_type":    model_type,
        "k":             k,
        "observed":      observed_stat,
        "focus_real":    fs_real,
        "null_stats":    null_stats,
        "null_focus":    null_focus,
        "p_perm":        p_perm,
        "n_valid":       len(null_stats),
    }


def _get_focus_state(groups: dict, k: int, occupancy: np.ndarray) -> int | None:
    """data-driven Focus State（Phantom 除外）の同定。"""
    if len(groups["B-start"]) == 0:
        return None
    b_start_counts = np.bincount(groups["B-start"], minlength=k).astype(float)
    rates = b_start_counts.copy()
    for s in range(k):
        if occupancy[s] == 0:
            rates[s] = -1.0
    return int(np.argmax(rates))


# ════════════════════════════════════════════════════════════════════════════
# 5. 可視化
# ════════════════════════════════════════════════════════════════════════════
def plot_null_distribution(result: dict, out_path: Path):
    null      = result["null_stats"]
    obs       = result["observed"]
    p_perm    = result["p_perm"]
    mtype     = result["model_type"]
    k         = result["k"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(null, bins=40, color="#4878CF", alpha=0.75, edgecolor="white",
            label=f"帰無分布 (N={result['n_valid']})")

    if obs is not None:
        ax.axvline(obs, color="#C0392B", linewidth=2.5, linestyle="--",
                   label=f"実観測値 ({obs:.2f})")
        # p_perm が極端に小さい場合は注釈
        if not np.isnan(p_perm):
            verdict = "帰無仮説棄却 ***" if p_perm < 0.001 else (
                      "帰無仮説棄却 *"  if p_perm < 0.05  else "棄却失敗")
            ax.text(0.97, 0.95,
                    f"置換 p = {p_perm:.4f}\n{verdict}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    ax.set_xlabel("χ² 統計量  [B-start vs S-head 2×2 分割表]", fontsize=11)
    ax.set_ylabel("頻度", fontsize=11)
    ax.set_title(
        f"置換検定 帰無分布:  {mtype}  k={k}\n"
        f"Focus State (実) = S{result['focus_real']}  |  {N_PERM} 回置換",
        fontsize=11,
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    common.log(f"    Saved: {out_path.name}")


def plot_focus_state_stability(result: dict, out_path: Path):
    """置換時に選ばれた Focus State の分布（帰無下での多様性を確認）。"""
    null_focus = result["null_focus"]
    if not null_focus:
        return
    k          = result["k"]
    mtype      = result["model_type"]
    fs_real    = result["focus_real"]
    counts     = np.bincount([f for f in null_focus if f is not None], minlength=k)

    fig, ax = plt.subplots(figsize=(max(6, k * 1.2), 4))
    colors = ["#C0392B" if s == fs_real else "#4878CF" for s in range(k)]
    ax.bar([f"S{s}" for s in range(k)], counts, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("帰無下での Focus State", fontsize=10)
    ax.set_ylabel("選択回数", fontsize=10)
    ax.set_title(
        f"帰無置換での Focus State 分布: {mtype} k={k}\n"
        f"赤 = 実観測 Focus State (S{fs_real})",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    common.log(f"    Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# 6. Markdown レポート
# ════════════════════════════════════════════════════════════════════════════
def build_report(all_results: dict) -> str:
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# E-1: 置換検定レポート",
        "",
        f"生成日時: {now}",
        "",
        "## 概要",
        "",
        "compound_words.txt の境界位置をランダムシャッフルした帰無分布を構築し、",
        "実際の境界シグナルが偶然では説明できないことを確認する。",
        "",
        f"- 置換回数: **{N_PERM}**",
        "- 統計量: `χ²` [B-start vs S-head の 2×2 分割表]（上限なし、キャップ問題を回避）",
        "- Focus State: 各置換で data-driven に選択（Phantom 除外、hypothesis/05 と同一手順）",
        "- 境界シャッフル: 各語の基数を固定し、境界位置のみランダム化",
        "",
        "## 結果サマリー",
        "",
        "| モデル | k | Focus State | 観測統計量 | 帰無中央値 | 置換 p 値 | 判定 |",
        "|--------|---|------------|-----------|---------|---------|------|",
    ]

    for key, res in all_results.items():
        obs    = res["observed"]
        p_perm = res["p_perm"]
        null   = res["null_stats"]
        obs_str    = f"{obs:.3f}" if obs is not None else "N/A"
        median_str = f"{np.median(null):.3f}" if len(null) > 0 else "N/A"
        p_str      = f"{p_perm:.4f}" if not np.isnan(p_perm) else "N/A"
        if np.isnan(p_perm):
            verdict = "計算不能"
        elif p_perm < 0.001:
            verdict = "**帰無棄却 *** (p<0.001)"
        elif p_perm < 0.05:
            verdict = "**帰無棄却 * (p<0.05)**"
        else:
            verdict = "棄却失敗"
        lines.append(
            f"| {res['model_type']} | {res['k']} | S{res['focus_real']} "
            f"| {obs_str} | {median_str} | {p_str} | {verdict} |"
        )

    lines += [
        "",
        "## 解釈",
        "",
        "- **置換 p < 0.05**: H₀ 棄却。境界シグナルは V8 文法が定義する実際の境界位置に依存しており、",
        "  ランダムな境界では再現できない。",
        "- **置換 p ≥ 0.05**: H₀ 棄却失敗。ランダムな境界でも同程度のシグナルが生まれる可能性あり。",
        "",
        "## 境界シャッフル方法の詳細",
        "",
        "各複合語 `word`（基数 `n`）に対し、`len(word)-1` 個の位置から",
        "`n-1` 個をランダムに選択（重複なし）して新しい境界とした。",
        "語長・基数分布は保持されるが、境界位置は帰無仮説下でランダム。",
        "シャッフル後の「基」は V8 文法的に有効でなくてもよい（純粋なラベル置換）。",
        "",
        "_本レポートは e1_permutation_test.py により自動生成。_",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    common.log("=== E-1: 置換検定 開始 ===")

    # データロード
    compound_splits = common.load_compound_splits()
    single_words    = common.load_single_words()
    all_words       = common.load_all_words()
    char2idx        = common.ORIG_CHAR2IDX

    # フィルタ
    compound_splits = {
        w: s for w, s in compound_splits.items()
        if all(c in char2idx for c in w)
    }
    single_words = [
        w for w in single_words
        if len(w) >= 2 and all(c in char2idx for c in w)
    ]
    common.log(f"有効複合語: {len(compound_splits):,}  有効単独語: {len(single_words):,}")

    all_results = {}

    for model_type in MODEL_TYPES:
        for k in K_LIST:
            common.log(f"\n{'─' * 60}")
            common.log(f"モデル: {model_type}  k={k}")
            common.log(f"{'─' * 60}")

            if model_type == "trigram":
                model = common.load_trigram_model(k)
            else:
                model = common.load_bigram_model(k)

            if model is None:
                common.log(f"  モデルが見つかりません。スキップ。")
                continue

            # Viterbi 占有率（Phantom State 同定）
            common.log("  Viterbi 占有率計算中（全語）...")
            occupancy = common.compute_occupancy(all_words, model, model_type)
            phantoms  = [s for s in range(k) if occupancy[s] == 0.0]
            common.log(f"  Phantom States: {['S' + str(s) for s in phantoms]}")

            # Viterbi デコード（一度だけ）
            common.log(f"  複合語デコード中 ({len(compound_splits):,} 語)...")
            decoded_compound = common.decode_words(compound_splits.keys(), model, model_type)
            common.log(f"  単独語デコード中 ({len(single_words):,} 語)...")
            decoded_single = common.decode_words(single_words, model, model_type)

            # 置換検定
            result = run_permutation_test(
                model_type, k, model,
                compound_splits, decoded_compound, decoded_single, single_words,
                occupancy,
            )
            key = f"{model_type}_k{k}"
            all_results[key] = result

            # 可視化
            plot_null_distribution(
                result, OUT_DIR / f"null_dist_{model_type}_k{k}.png"
            )
            plot_focus_state_stability(
                result, OUT_DIR / f"focus_stability_{model_type}_k{k}.png"
            )

    # レポート
    common.log("\nMarkdown レポート生成中...")
    report_path = OUT_DIR / "permutation_report.md"
    report_path.write_text(build_report(all_results), encoding="utf-8")
    common.log(f"レポート保存: {report_path}")
    common.log(f"\n✓ E-1 完了。出力先: {OUT_DIR.resolve()}")
