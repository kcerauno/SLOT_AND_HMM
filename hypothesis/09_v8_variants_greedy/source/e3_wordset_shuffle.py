"""
e3_wordset_shuffle.py
========================
提案 E-3: 語集合シャッフル検定

目的:
  compound_words.txt の B-start シグナルが「複合語という語集合の選択」に依存するかを検定する。
  単独語集合から同数（3,363）の語をランダムサンプリングし、
  compound_words.txt の基数分布に従ってランダムな境界を割り当てた帰無分布を構築する。

仮説:
  H₀: 単独語プールからランダムに選んだ語に境界を付けても、
      実際の複合語境界と同程度の B-start 特化シグナルが得られる
  H₁: B-start 特化シグナルは複合語という語集合の選択に依存しており、
      単独語の任意選択では再現できない

E-1 との違い:
  E-1: 同じ複合語集合 × ランダム境界位置  → 「境界位置」の寄与を検定
  E-3: ランダム語集合（単独語）× ランダム境界位置 → 「語集合選択」の寄与を検定

効率:
  Viterbi デコードは全単独語に対して一度だけ実行（decoded をキャッシュ）。
  各反復はラベル収集のみのため高速。

実行:
  cd /home/practi/work_voy
  PYTHONPATH=/home/practi/work_voy/.venv/lib/python3.10/site-packages \\
    python3.10 hypothesis/09_v8_variants_greedy/source/e3_wordset_shuffle.py
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
OUT_DIR   = ROOT / "hypothesis/09_v8_variants_greedy/results/e3_wordset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM     = 1000
BASE_SEED  = 99
K_LIST     = [7, 8]
MODEL_TYPES = ["trigram", "bigram"]


# ════════════════════════════════════════════════════════════════════════════
# 1. compound_words.txt から基数分布を取得
# ════════════════════════════════════════════════════════════════════════════
def get_base_count_distribution(compound_splits: dict) -> list[int]:
    """複合語ごとの基数リスト（サンプリング用）を返す。"""
    return [len(bases) for bases in compound_splits.values()]


# ════════════════════════════════════════════════════════════════════════════
# 2. 擬似複合語のランダム生成（語集合シャッフル）
# ════════════════════════════════════════════════════════════════════════════
def sample_pseudo_splits(
    word_pool: list[str],
    decoded_pool: dict[str, list],
    n_words: int,
    base_count_dist: list[int],
    rng: np.random.Generator,
) -> dict:
    """
    単独語プールから n_words 語をランダム（復元抽出）し、
    compound_words.txt の基数分布に従ってランダム境界を割り当てる。

    Returns:
        pseudo_splits: {word: tuple(bases)} — 語集合シャッフルによる擬似複合語
        pseudo_decoded: {word: states}       — 対応する Viterbi 状態列
    """
    pseudo_splits  = {}
    pseudo_decoded = {}

    # 有効語（decoded が存在する語）だけを候補にする
    valid_pool = [w for w in word_pool if decoded_pool.get(w) is not None]
    if len(valid_pool) == 0:
        return pseudo_splits, pseudo_decoded

    # 復元抽出（同じ語が複数回選ばれることがある）
    sampled_idx = rng.integers(0, len(valid_pool), size=n_words)
    base_idx    = rng.integers(0, len(base_count_dist), size=n_words)

    for i, (wi, bi) in enumerate(zip(sampled_idx, base_idx)):
        word     = valid_pool[wi]
        n_bases  = base_count_dist[bi]
        L        = len(word)

        # 衝突回避のためにユニークなキーを作成（同一語が複数回選ばれる場合）
        key = f"{word}#{i}"

        if n_bases <= 1 or L < n_bases:
            # 境界を付けられない → 2基で単純2分割
            mid = max(1, L // 2)
            bases = (word[:mid], word[mid:]) if L >= 2 else (word,)
        else:
            split_pos = sorted(
                rng.choice(np.arange(1, L), size=n_bases - 1, replace=False).tolist()
            )
            bases = []
            prev  = 0
            for pos in split_pos:
                bases.append(word[prev:pos])
                prev = pos
            bases.append(word[prev:])
            bases = tuple(bases)

        pseudo_splits[key]  = bases
        # decoded states は元の word のデコード結果を使用（文字位置は語全体）
        pseudo_decoded[key] = decoded_pool[word]

    return pseudo_splits, pseudo_decoded


# ════════════════════════════════════════════════════════════════════════════
# 3. グループ収集
# ════════════════════════════════════════════════════════════════════════════
def collect_groups(
    compound_splits: dict,
    decoded_compound: dict,
    decoded_single: dict,
    single_words: list[str],
) -> dict:
    """4 グループ（B-start, B-end, S-head, S-mid）の状態リストを収集。"""
    b_start, b_end, s_head, s_mid = [], [], [], []

    for key, splits in compound_splits.items():
        # キーに '#' が含まれる場合は擬似複合語
        word = key.split("#")[0] if "#" in key else key
        states = decoded_compound.get(key)
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
# 4. 統計量（E-1 と同一実装）
# ════════════════════════════════════════════════════════════════════════════
def compute_statistic(groups: dict, k: int, occupancy: np.ndarray) -> float | None:
    """
    data-driven Focus State を選択し、B-start vs S-head の chi2 を返す。
    """
    if len(groups["B-start"]) == 0 or len(groups["S-head"]) == 0:
        return None

    b_start_counts = np.bincount(groups["B-start"], minlength=k).astype(float)
    rates = b_start_counts.copy()
    for s in range(k):
        if occupancy[s] == 0:
            rates[s] = -1.0
    focus = int(np.argmax(rates))

    ca = np.bincount(groups["B-start"], minlength=k)
    cb = np.bincount(groups["S-head"],  minlength=k)
    a = float(ca[focus]); b = float(ca.sum()) - a
    c = float(cb[focus]); d = float(cb.sum()) - c

    if a + c == 0:
        return None

    N  = a + b + c + d
    r1 = a + b; r2 = c + d
    c1 = a + c; c2 = b + d

    if r1 == 0 or r2 == 0 or c1 == 0 or c2 == 0:
        return None

    chi2 = N * (a * d - b * c) ** 2 / (r1 * r2 * c1 * c2)
    return float(chi2)


def _get_focus_state(groups: dict, k: int, occupancy: np.ndarray) -> int | None:
    if len(groups["B-start"]) == 0:
        return None
    b_start_counts = np.bincount(groups["B-start"], minlength=k).astype(float)
    rates = b_start_counts.copy()
    for s in range(k):
        if occupancy[s] == 0:
            rates[s] = -1.0
    return int(np.argmax(rates))


# ════════════════════════════════════════════════════════════════════════════
# 5. 語集合シャッフル検定メインループ
# ════════════════════════════════════════════════════════════════════════════
def run_wordset_test(
    model_type: str,
    k: int,
    compound_splits: dict,
    decoded_compound: dict,
    decoded_single: dict,
    single_words: list[str],
    occupancy: np.ndarray,
    base_count_dist: list[int],
) -> dict:
    n_compound = len(compound_splits)

    # 実観測統計量（本物の複合語 × 本物の境界）
    common.log(f"  [{model_type} k={k}] 実観測統計量計算中...")
    real_groups   = collect_groups(compound_splits, decoded_compound, decoded_single, single_words)
    observed_stat = compute_statistic(real_groups, k, occupancy)
    fs_real       = _get_focus_state(real_groups, k, occupancy)
    common.log(f"    Focus State: S{fs_real}  観測統計量: {observed_stat:.4f}" if observed_stat else
               f"    観測統計量: None")

    rng = np.random.default_rng(BASE_SEED + k + (10 if model_type == "bigram" else 0))
    null_stats  = []
    null_focus  = []

    common.log(f"  [{model_type} k={k}] {N_PERM} 回の語集合シャッフル検定実行中...")
    for i in range(N_PERM):
        pseudo_splits, pseudo_decoded = sample_pseudo_splits(
            single_words, decoded_single, n_compound, base_count_dist, rng
        )
        groups  = collect_groups(pseudo_splits, pseudo_decoded, decoded_single, single_words)
        stat    = compute_statistic(groups, k, occupancy)
        fs_null = _get_focus_state(groups, k, occupancy)
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

    null_median = float(np.median(null_stats)) if len(null_stats) > 0 else float("nan")
    ratio = observed_stat / null_median if (observed_stat and null_median) else float("nan")

    common.log(f"  [{model_type} k={k}] 語集合 p 値: {p_perm:.4f}  "
               f"(observed={observed_stat:.3f}, null median={null_median:.3f}, ratio={ratio:.2f}x)")

    return {
        "model_type":    model_type,
        "k":             k,
        "observed":      observed_stat,
        "focus_real":    fs_real,
        "null_stats":    null_stats,
        "null_focus":    null_focus,
        "p_perm":        p_perm,
        "n_valid":       len(null_stats),
        "null_median":   null_median,
        "ratio":         ratio,
    }


# ════════════════════════════════════════════════════════════════════════════
# 6. 可視化
# ════════════════════════════════════════════════════════════════════════════
def plot_null_distribution(result: dict, out_path: Path):
    null   = result["null_stats"]
    obs    = result["observed"]
    p_perm = result["p_perm"]
    mtype  = result["model_type"]
    k      = result["k"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(null, bins=40, color="#2ECC71", alpha=0.75, edgecolor="white",
            label=f"帰無分布 (N={result['n_valid']})\n[単独語ランダムサンプル + ランダム境界]")

    if obs is not None:
        ax.axvline(obs, color="#C0392B", linewidth=2.5, linestyle="--",
                   label=f"実観測値（実複合語）= {obs:.2f}")
        if not np.isnan(p_perm):
            verdict = "語集合効果あり ***" if p_perm < 0.001 else (
                      "語集合効果あり *"  if p_perm < 0.05  else "語集合効果なし")
            ax.text(0.97, 0.95,
                    f"語集合 p = {p_perm:.4f}\n{verdict}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    ax.set_xlabel("χ² 統計量  [B-start vs S-head 2×2 分割表]", fontsize=11)
    ax.set_ylabel("頻度", fontsize=11)
    ax.set_title(
        f"語集合シャッフル検定 帰無分布:  {mtype}  k={k}\n"
        f"Focus State (実) = S{result['focus_real']}  |  {N_PERM} 回シャッフル",
        fontsize=11,
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    common.log(f"    Saved: {out_path.name}")


def plot_focus_state_stability(result: dict, out_path: Path):
    null_focus = result["null_focus"]
    if not null_focus:
        return
    k       = result["k"]
    mtype   = result["model_type"]
    fs_real = result["focus_real"]
    counts  = np.bincount([f for f in null_focus if f is not None], minlength=k)

    fig, ax = plt.subplots(figsize=(max(6, k * 1.2), 4))
    colors = ["#C0392B" if s == fs_real else "#2ECC71" for s in range(k)]
    ax.bar([f"S{s}" for s in range(k)], counts, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("帰無下での Focus State", fontsize=10)
    ax.set_ylabel("選択回数", fontsize=10)
    ax.set_title(
        f"語集合シャッフル時の Focus State 分布: {mtype} k={k}\n"
        f"赤 = 実観測 Focus State (S{fs_real})",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    common.log(f"    Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# 7. Markdown レポート
# ════════════════════════════════════════════════════════════════════════════
def build_report(all_results: dict) -> str:
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# E-3: 語集合シャッフル検定レポート",
        "",
        f"生成日時: {now}",
        "",
        "## 概要",
        "",
        "compound_words.txt の B-start シグナルが「複合語という語集合の選択」に依存するかを検定する。",
        "単独語集合（words_base_only.txt）から同数（3,363）の語をランダムサンプリングし、",
        "compound_words.txt の基数分布に従ってランダム境界を割り当てた帰無分布を 1,000 回構築。",
        "",
        f"- 検定回数: **{N_PERM}**",
        "- 統計量: `χ²` [B-start vs S-head の 2×2 分割表]（E-1 と同一）",
        "- S-head: 実際の単独語先頭の状態（E-1 と同一参照）",
        "- B-start 帰無: 単独語プールからランダムに選んだ語内のランダム位置",
        "- 語の復元抽出: 同一語が複数回選ばれることを許容（プール 4,716 語）",
        "",
        "## 結果サマリー",
        "",
        "| モデル | k | Focus State | 観測χ² | 帰無中央値 | 観測/帰無比 | 語集合 p 値 | 判定 |",
        "|--------|---|------------|--------|---------|----------|----------|------|",
    ]

    for key, res in all_results.items():
        obs     = res["observed"]
        p_perm  = res["p_perm"]
        null    = res["null_stats"]
        ratio   = res["ratio"]
        obs_str    = f"{obs:.3f}"    if obs is not None    else "N/A"
        median_str = f"{np.median(null):.3f}" if len(null) > 0 else "N/A"
        ratio_str  = f"{ratio:.2f}×" if not np.isnan(ratio) else "N/A"
        p_str      = f"{p_perm:.4f}" if not np.isnan(p_perm) else "N/A"
        if np.isnan(p_perm):
            verdict = "計算不能"
        elif p_perm < 0.001:
            verdict = "**語集合効果あり *** (p<0.001)**"
        elif p_perm < 0.05:
            verdict = "**語集合効果あり * (p<0.05)**"
        else:
            verdict = "語集合効果なし"
        lines.append(
            f"| {res['model_type']} | {res['k']} | S{res['focus_real']} "
            f"| {obs_str} | {median_str} | {ratio_str} | {p_str} | {verdict} |"
        )

    lines += [
        "",
        "## 解釈",
        "",
        "- **語集合 p < 0.05（語集合効果あり）**: 実際の複合語集合は単独語プールからの任意選択より",
        "  有意に大きな B-start 特化シグナルを持つ。",
        "  → B-start シグナルは「複合語という語集合の選択」にも依存する。",
        "",
        "- **語集合 p ≥ 0.05（語集合効果なし）**: 単独語をランダムに選んでも同程度のシグナルが得られる。",
        "  → B-start シグナルは語集合の選択ではなく、HMM の状態空間構造に由来する。",
        "",
        "## E-1 と E-3 の組み合わせ解釈",
        "",
        "| E-1（境界位置効果） | E-3（語集合効果） | 解釈 |",
        "|-------------------|-----------------|------|",
        "| 有意 | 有意 | 境界位置と語集合の両方がシグナルに寄与 |",
        "| 有意 | 非有意 | 境界位置の正確さが主因；語集合は副次的 |",
        "| 非有意 | 有意 | 語集合の選択が主因；境界位置の精度は副次的 |",
        "| 非有意 | 非有意 | シグナルは HMM 状態空間構造のみに由来 |",
        "",
        "## 語集合シャッフル方法の詳細",
        "",
        "1. 有効単独語（words_base_only.txt かつ全文字が ORIG_ALL_CHARS に含まれる）をプールとして使用",
        "2. プールから 3,363 語を復元抽出（同一語の重複を許容）",
        "3. compound_words.txt の基数分布（2基: 93.2%, 3基: 6.6%, 4基: 0.1%）から基数を決定",
        "4. 各語内の `len(word)-1` 位置から `n_bases-1` 個をランダム選択（重複なし）",
        "5. 選択した語の Viterbi デコード結果（事前計算済み）から状態を読み取る",
        "",
        "_本レポートは e3_wordset_shuffle.py により自動生成。_",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    common.log("=== E-3: 語集合シャッフル検定 開始 ===")

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

    # 基数分布の取得
    base_count_dist = get_base_count_distribution(compound_splits)
    from collections import Counter
    bc = Counter(base_count_dist)
    common.log(f"基数分布: {dict(sorted(bc.items()))}")

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

            # 語集合シャッフル検定
            result = run_wordset_test(
                model_type, k,
                compound_splits, decoded_compound, decoded_single, single_words,
                occupancy, base_count_dist,
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
    report_path = OUT_DIR / "wordset_report.md"
    report_path.write_text(build_report(all_results), encoding="utf-8")
    common.log(f"レポート保存: {report_path}")

    common.log("\n=== E-3: 語集合シャッフル検定 完了 ===")
