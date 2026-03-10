"""
slot_hmm_pytorch_max_speed.py
============================
Voynich Manuscript スロット文法仮説の数理検証
【PyTorch GPU 極限高速化版: Word Batching】

- 全単語(8,059語)をパディングして行列(バッチ)化し、GPUのコアを数万個規模で同時稼働させます。
- Pythonのループ回数を「全文字数(67,546)」から「最大単語長(約15)」へ劇的に削減。
- N_RESTARTS(20試行)も同時に並列処理。
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

try:
    import torch
except ImportError:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ERROR: PyTorchがインストールされていません。")
    import sys
    sys.exit(1)

def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"[PyTorch] Using device: {DEVICE}")

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
DB_PATH      = "data/voynich.db"
OUT_DIR      = Path("hypothesis/01_bigram/results/hmm_pt_max")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE      = [2, 3, 4, 5, 6, 7, 8]
N_RESTARTS   = 20
N_ITER       = 200
LTR_ITER     = 50
TOL          = 1e-4
MIN_WORD_LEN = 2

# ── データ準備 ────────────────────────────────────────────────────────
BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"

def prepare_batched_data(words, char2idx):
    """単語リストをパディングして (N_words, Max_Len) のテンソルにする"""
    processed = []
    for w in words:
        # BOS + word + EOS
        processed.append([char2idx[BOS_CHAR]] + [char2idx[c] for c in w] + [char2idx[EOS_CHAR]])

    max_len = max(len(p) for p in processed)
    X_matrix = []
    mask = []
    for p in processed:
        pad_len = max_len - len(p)
        X_matrix.append(p + [char2idx[PAD_CHAR]] * pad_len)
        # 実際の文字がある場所以外をマスクする (1.0: 有効, 0.0: 無効)
        mask.append([1.0] * len(p) + [0.0] * pad_len)

    return np.array(X_matrix, dtype=np.int16), np.array(mask, dtype=np.float32), max_len

# ── PyTorchによる Word-Batched HMM 実装 ─────────────────────────────
class MaxSpeedHMM_PT:
    def __init__(self, n_restarts, n_components, n_vocab, device=DEVICE):
        self.b = n_restarts  # 試行バッチ (20)
        self.k = n_components
        self.v = n_vocab
        self.device = device

        # log_startprob: (B, K)
        # log_transmat:  (B, K, K)
        # log_emission:  (B, K, V)
        self.log_startprob = None
        self.log_transmat  = None
        self.log_emission  = None

    def _init_params(self, seed=42):
        torch.manual_seed(seed)
        s = torch.rand(self.b, self.k, device=self.device) + 0.5
        self.log_startprob = torch.log(s / s.sum(dim=1, keepdim=True))

        t = torch.rand(self.b, self.k, self.k, device=self.device) + 0.5
        self.log_transmat = torch.log(t / t.sum(dim=2, keepdim=True))

        e = torch.rand(self.b, self.k, self.v, device=self.device) + 0.5
        self.log_emission = torch.log(e / e.sum(dim=2, keepdim=True))

    def fit(self, X_pt, mask_pt, n_iter=N_ITER, tol=TOL, left_to_right=False):
        """
        X_pt: (N, T)  - N:単語数, T:最大長
        mask_pt: (N, T)
        """
        N, T = X_pt.shape
        B = self.b
        K = self.k

        # 系列ごとの真の終端インデックス (EOS 位置)
        last_idx = (mask_pt.sum(dim=1) - 1).long()  # (N,)
        n_idx    = torch.arange(N, device=self.device)

        # 収束監視用 (prev_logL: 直前イテレーションの対数尤度)
        prev_logL  = torch.full((B,), -float('inf'), device=self.device)
        total_logL = prev_logL.clone()

        for it in range(n_iter):
            # 1. Forward Pass (Log-domain)
            # log_alpha: (T, N, B, K) -> 大文字 T (時間軸) は Python ループ
            # ここが高速化の肝: N(8000) と B(20) を次元に組み込む
            log_alpha = torch.empty(T, N, B, K, device=self.device)

            # t=0: 全単語の開始
            # self.log_emission[:, :, X_pt[:, 0]] -> (B, K, N)
            # 転置が必要: (N, B, K)
            emiss_0 = self.log_emission[:, :, X_pt[:, 0]].permute(2, 0, 1)
            log_alpha[0] = self.log_startprob.unsqueeze(0) + emiss_0

            for t in range(1, T):
                # (N, B, K, 1) + (1, B, K, K) -> (N, B, K, K)
                prev = log_alpha[t-1].unsqueeze(3) + self.log_transmat.unsqueeze(0)
                # dim=2 は遷移元 i
                sum_prev = torch.logsumexp(prev, dim=2)  # (N, B, K)

                emiss_t = self.log_emission[:, :, X_pt[:, t]].permute(2, 0, 1)
                log_alpha[t] = sum_prev + emiss_t

            # 2. Backward Pass (T, N, B, K)
            # 修正: 各系列の真の終端位置 (EOS = last_idx[n]) のみを log(1)=0 に初期化し、
            # それ以降の PAD 区間は -inf のままにする。
            # これにより PAD 放射確率による後退確率の汚染を防ぐ。
            log_beta = torch.full((T, N, B, K), -float('inf'), device=self.device)
            log_beta[last_idx, n_idx] = 0.0

            for t in range(T-2, -1, -1):
                # (1, B, K, K) + (N, B, 1, K) + (N, B, 1, K) -> (N, B, K, K)
                emiss_tp1 = self.log_emission[:, :, X_pt[:, t+1]].permute(2, 0, 1).unsqueeze(2)
                nxt = self.log_transmat.unsqueeze(0) + emiss_tp1 + log_beta[t+1].unsqueeze(2)
                new_beta = torch.logsumexp(nxt, dim=3)  # (N, B, K)

                # 終端位置 (last_idx[n] == t) はすでに 0.0 で初期化済み → 上書きしない
                is_term = (last_idx == t).view(N, 1, 1)  # (N, 1, 1)
                log_beta[t] = torch.where(is_term, log_beta[t], new_beta)

            # 3. 対数尤度の計算
            # alpha のみから求める（beta の初期化と独立なため常に正確）
            alpha_swapped = log_alpha.permute(1, 0, 2, 3)  # (N, T, B, K)
            end_alpha = torch.gather(
                alpha_swapped, 1,
                last_idx.view(-1, 1, 1, 1).expand(-1, 1, B, K)
            ).squeeze(1)  # (N, B, K)

            logL_per_word = torch.logsumexp(end_alpha, dim=2)  # (N, B)
            total_logL    = logL_per_word.sum(dim=0)            # (B,)

            if it > 0 and (total_logL - prev_logL).abs().max() < tol:
                break
            prev_logL = total_logL.clone()

            # --- M-step ---
            log_gamma = log_alpha + log_beta  # (T, N, B, K)
            # 正規化 (T, N, B, 1)
            norm_factor    = torch.logsumexp(log_gamma, dim=3, keepdim=True)
            # PAD位置: log_gamma=-inf, norm_factor=-inf → (-inf)-(-inf)=NaN を防ぐ
            log_gamma_norm = torch.nan_to_num(log_gamma - norm_factor, nan=-float('inf'))
            # 有効な文字のみ集計するためマスク適用
            log_gamma_norm = log_gamma_norm + torch.log(
                mask_pt.permute(1, 0).unsqueeze(2).unsqueeze(3)
            )

            # log_xi: (T-1, N, B, K, K)
            # alpha(t) + A + emiss(t+1) + beta(t+1)
            emiss_tp1_all = (
                self.log_emission[:, :, X_pt[:, 1:]]
                .permute(3, 2, 0, 1)
                .unsqueeze(3)
            )  # (T-1, N, B, 1, K)
            log_xi = (
                log_alpha[:-1].unsqueeze(4)
                + self.log_transmat.view(1, 1, B, K, K)
                + emiss_tp1_all
                + log_beta[1:].unsqueeze(3)
            )

            # 境界マスク: 有効→有効 の遷移のみ残す (PAD 区間への遷移を除外)
            xi_mask = mask_pt[:, :-1] * mask_pt[:, 1:]  # (N, T-1)
            log_xi  = log_xi + torch.log(xi_mask.permute(1, 0).view(T-1, N, 1, 1, 1))
            # 正規化 (T-1, N, B, K, K) - (T-1, N, B, 1, 1)
            # PAD位置の norm_factor=-inf による NaN を防ぐ
            log_xi  = torch.nan_to_num(log_xi - norm_factor[:-1].unsqueeze(4), nan=-float('inf'))

            # 更新 1: Start Prob (B, K) - 各系列の t=0
            new_log_start = torch.logsumexp(log_gamma_norm[0], dim=0)  # sum over N -> (B, K)
            self.log_startprob = new_log_start - torch.logsumexp(new_log_start, dim=1, keepdim=True)

            # 更新 2: Transmat (B, K, K)
            new_log_trans = torch.logsumexp(log_xi, dim=(0, 1))  # sum over T-1, N
            if left_to_right:
                m = torch.triu(torch.ones(K, K, device=self.device)).bool()
                new_log_trans[:, ~m] = -float('inf')
            self.log_transmat = new_log_trans - torch.logsumexp(new_log_trans, dim=2, keepdim=True)

            # 更新 3: Emission (B, K, V)
            # exp_gamma: (T, N, B, K) -> (B, K, T*N)
            gamma_flat = torch.exp(log_gamma_norm).permute(2, 3, 0, 1).reshape(B, K, -1)
            # X_onehot: (T*N, V) - 共通
            X_flat   = X_pt.permute(1, 0).reshape(-1)  # (T*N,)
            X_onehot = torch.zeros(T * N, self.v, device=self.device)
            X_onehot.scatter_(1, X_flat.view(-1, 1), 1.0)

            new_emiss     = torch.matmul(gamma_flat, X_onehot)  # (B, K, V)
            new_log_emiss = torch.log(new_emiss + 1e-35)
            self.log_emission = new_log_emiss - torch.logsumexp(new_log_emiss, dim=2, keepdim=True)

        return total_logL

    def get_best(self, logL_pt):
        idx = torch.argmax(logL_pt).item()
        return {
            "start": torch.exp(self.log_startprob[idx]).cpu().numpy(),
            "trans": torch.exp(self.log_transmat[idx]).cpu().numpy(),
            "emiss": torch.exp(self.log_emission[idx]).cpu().numpy(),
            "logL":  logL_pt[idx].item(),
            "best_log_start": self.log_startprob[idx],
            "best_log_trans": self.log_transmat[idx],
            "best_log_emiss": self.log_emission[idx],
        }

# ── BIC / AIC ─────────────────────────────────────────────────────────
def compute_bic(log_likelihood, X_len, k, V):
    N = X_len
    # PAD は収束後に放射確率 ≈ 0 になる疑似トークンのため自由パラメータから除外 (V_eff = V - 1)
    V_eff    = V - 1
    n_params = k * (k - 1) + k * (V_eff - 1) + (k - 1)
    bic = -2 * log_likelihood + n_params * np.log(N)
    aic = -2 * log_likelihood + 2 * n_params
    return bic, aic

# ── Viterbi (単一系列・単一モデル) ───────────────────────────────────
def viterbi_pt(log_start, log_trans, log_emiss, X_np, device):
    X = torch.tensor(X_np, dtype=torch.long, device=device)
    T = X.shape[0]
    K = log_start.shape[0]
    log_delta = torch.empty(T, K, device=device)
    psi       = torch.empty(T, K, dtype=torch.long, device=device)
    log_delta[0] = log_start + log_emiss[:, X[0]]
    psi[0] = 0
    for t in range(1, T):
        vals = log_delta[t-1].unsqueeze(1) + log_trans
        max_vals, argmax_vals = torch.max(vals, dim=0)
        log_delta[t] = max_vals + log_emiss[:, X[t]]
        psi[t] = argmax_vals
    path = torch.empty(T, dtype=torch.long, device=device)
    path[T-1] = torch.argmax(log_delta[T-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return log_delta[T-1].max().item(), path.cpu().numpy()

# ── ユーティリティ ────────────────────────────────────────────────────
def _state_label(i, k):
    labels = {
        2: ["Pref/Core", "Suff"],
        3: ["Pref", "Core", "Suff"],
        4: ["Pref", "C1",   "C2",   "Suff"],
        5: ["Pref", "C1",   "C2",   "C3",   "Suff"],
    }
    return labels.get(k, [f"S{j}" for j in range(k)])[i]

def plot_res(info, k, topo, all_chars, out_dir):
    slbls = [_state_label(i, k) for i in range(k)]
    clbls = [
        ("BOS" if c == "^" else ("EOS" if c == "$" else ("PAD" if c == "_" else c)))
        for c in all_chars
    ]

    # 遷移確率ヒートマップ
    A = pd.DataFrame(info["trans"], index=slbls, columns=slbls)
    fig, ax = plt.subplots(figsize=(max(5, k * 1.5), max(4, k * 1.3)))
    sns.heatmap(
        A, annot=True, fmt=".3f", cmap="Blues",
        vmin=0, vmax=1, ax=ax,
        linewidths=0.5, linecolor="gray",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"遷移確率行列（{topo}, k={k}）", fontsize=13)
    ax.set_xlabel("遷移先状態", fontsize=10)
    ax.set_ylabel("遷移元状態", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"transition_{topo}_k{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved: transition_{topo}_k{k}.png")

    # 放射確率ヒートマップ
    B_df = pd.DataFrame(info["emiss"], index=slbls, columns=clbls)
    fig, ax = plt.subplots(figsize=(max(14, len(all_chars) * 0.55), max(4, k * 1.5)))
    sns.heatmap(
        B_df, annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, ax=ax,
        linewidths=0.3, linecolor="gray",
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title(f"放射確率行列（{topo}, k={k}）", fontsize=13)
    ax.set_xlabel("観測文字", fontsize=10)
    ax.set_ylabel("隠れ状態", fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / f"emission_{topo}_k{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved: emission_{topo}_k{k}.png")

def plot_bic(results, out_path):
    ks     = [r["k"]     for r in results]
    bics_f = [r["bic_f"] for r in results]
    bics_l = [r["bic_l"] for r in results]

    x     = np.arange(len(ks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, bics_f, width, label="Full HMM",          color="steelblue", alpha=0.85)
    ax.bar(x + width/2, bics_l, width, label="Left-to-Right HMM", color="coral",     alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_xlabel("状態数 k")
    ax.set_ylabel("BIC（小さいほど良い）")
    ax.set_title("HMM モデル選択: BIC 比較")
    ax.legend()

    valid_f = [(b, i) for i, b in enumerate(bics_f) if not np.isnan(b)]
    valid_l = [(b, i) for i, b in enumerate(bics_l) if not np.isnan(b)]
    if valid_f:
        min_b, min_i = min(valid_f, key=lambda t: t[0])
        ax.annotate("▼", xy=(min_i - width/2, min_b),
                    fontsize=14, ha="center", va="bottom", color="steelblue")
    if valid_l:
        min_b, min_i = min(valid_l, key=lambda t: t[0])
        ax.annotate("▼", xy=(min_i + width/2, min_b),
                    fontsize=14, ha="center", va="bottom", color="coral")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {Path(out_path).name}")

def decode_examples_pt(info, k, words, char2idx, device):
    """Viterbi アルゴリズムで各単語の隠れ状態列を出力"""
    state_labels = [_state_label(i, k) for i in range(k)]
    log_start = torch.tensor(np.log(info["start"] + 1e-35), device=device)
    log_trans  = torch.tensor(np.log(info["trans"]  + 1e-35), device=device)
    log_emiss  = torch.tensor(np.log(info["emiss"]  + 1e-35), device=device)

    lines = [
        f"{'='*70}",
        f"  Viterbi 状態割り当て例（k={k}, Full HMM）",
        f"{'='*70}",
    ]

    sample       = sorted(words, key=len)
    sample_words = (
        sample[:10]
        + sample[len(sample)//2 - 5 : len(sample)//2 + 5]
        + sample[-10:]
    )
    seen = set()

    for w in sample_words:
        if w in seen:
            continue
        seen.add(w)
        seq = np.array(
            [char2idx[BOS_CHAR]] + [char2idx[c] for c in w] + [char2idx[EOS_CHAR]],
            dtype=np.int32,
        )
        try:
            _, state_seq = viterbi_pt(log_start, log_trans, log_emiss, seq, device)
            full_chars  = [BOS_CHAR] + list(w) + [EOS_CHAR]
            full_labels = [state_labels[s] for s in state_seq]
            char_disp   = " ".join(f"{c:>3}" for c in full_chars)
            state_disp  = " ".join(f"{l[:3]:>3}" for l in full_labels)
            lines.append(f"  {w:<15}  chars: {char_disp}")
            lines.append(f"  {'':15}  state: {state_disp}")
            lines.append("")
        except Exception:
            continue

    return "\n".join(lines)

# ════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("Starting MAX SPEED GPU HMM (Word Batching)...")

    conn  = sqlite3.connect(DB_PATH)
    words = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
        conn,
    )["word"].tolist()
    conn.close()

    all_types = sorted(set(words))
    all_types = [w for w in all_types if len(w) >= MIN_WORD_LEN]
    log(f"ユニーク単語数 (types): {len(all_types)}")

    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR, EOS_CHAR, PAD_CHAR] + raw_chars
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    V         = len(all_chars)

    X_np, mask_np, MaxLen = prepare_batched_data(all_types, char2idx)
    log(f"Max Word Len (T): {MaxLen} (Python loops will be this count)")
    log(f"語彙サイズ: {V} (PAD 含む)")

    X_pt    = torch.tensor(X_np,    dtype=torch.long, device=DEVICE)
    mask_pt = torch.tensor(mask_np, device=DEVICE)

    # BIC 計算用: 有効文字数 (PAD を除く)
    num_symbols = int(mask_pt.sum().item())

    results      = []
    report_lines = []
    best_k3_full = None

    for k in K_RANGE:
        log(f"{'='*50}\n  k = {k}\n{'='*50}")

        # ── Full HMM ──────────────────────────────────────────────────
        log(f"  [Full] Training {N_RESTARTS} models in parallel...")
        model = MaxSpeedHMM_PT(N_RESTARTS, k, V, DEVICE)
        model._init_params(seed=42)
        ll   = model.fit(X_pt, mask_pt, left_to_right=False)
        info = model.get_best(ll)
        log(f"  Best LogL (Full): {info['logL']:.2f}")
        plot_res(info, k, "full", all_chars, OUT_DIR)
        if k == 3:
            best_k3_full = info

        bic_f, aic_f = compute_bic(info["logL"], num_symbols, k, V)

        # ── Left-to-Right HMM ─────────────────────────────────────────
        log(f"  [LTR ] Training {N_RESTARTS} models in parallel...")
        model_ltr = MaxSpeedHMM_PT(N_RESTARTS, k, V, DEVICE)
        model_ltr._init_params(seed=77)
        ll_l   = model_ltr.fit(X_pt, mask_pt, n_iter=LTR_ITER, left_to_right=True)
        info_l = model_ltr.get_best(ll_l)
        log(f"  Best LogL (LTR):  {info_l['logL']:.2f}")
        plot_res(info_l, k, "ltr", all_chars, OUT_DIR)

        bic_l, aic_l = compute_bic(info_l["logL"], num_symbols, k, V)

        results.append({
            "k":     k,
            "bic_f": bic_f, "aic_f": aic_f,
            "bic_l": bic_l, "aic_l": aic_l,
        })
        log(f"  -> BIC Full={bic_f:.1f}  L-to-R={bic_l:.1f}")

        # レポートセクション生成
        state_labels = [_state_label(i, k) for i in range(k)]
        section = [
            f"{'='*70}",
            f"  k = {k}",
            f"{'='*70}",
            f"  [Full HMM]",
            f"    対数尤度 : {info['logL']:.2f}",
            f"    BIC      : {bic_f:.2f}",
            f"    AIC      : {aic_f:.2f}",
            f"",
            f"  遷移行列（Full）:",
        ]
        for i, row in enumerate(info["trans"]):
            vals = "  ".join(f"{v:.3f}" for v in row)
            section.append(f"    {state_labels[i]:<12} → [{vals}]")

        section += [
            f"",
            f"  [Left-to-Right HMM]",
            f"    対数尤度 : {info_l['logL']:.2f}",
            f"    BIC      : {bic_l:.2f}",
            f"    AIC      : {aic_l:.2f}",
            f"",
            f"  遷移行列（L-to-R）:",
        ]
        for i, row in enumerate(info_l["trans"]):
            vals = "  ".join(f"{v:.3f}" for v in row)
            section.append(f"    {state_labels[i]:<12} → [{vals}]")

        report_lines.append("\n".join(section))

    # ── まとめの出力 ───────────────────────────────────────────────────
    log("\n[生成ファイル出力]")
    plot_bic(results, OUT_DIR / "bic_comparison.png")

    if best_k3_full is not None:
        example_text = decode_examples_pt(
            best_k3_full, k=3, words=all_types, char2idx=char2idx, device=DEVICE
        )
        ex_path = OUT_DIR / "word_examples.txt"
        with open(ex_path, "w", encoding="utf-8") as f:
            f.write(example_text)
        log(f"  Saved: {ex_path.name}")

    report_path = OUT_DIR / "hmm_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report_lines))
    log(f"  Saved: {report_path.name}")

    log("\n" + "="*60)
    log("  BIC サマリー")
    log("="*60)
    log(f"  {'k':>4}  {'BIC(Full)':>14}  {'BIC(L-to-R)':>14}")
    log("  " + "-"*40)
    for r in results:
        log(f"  {r['k']:>4}  {r['bic_f']:>14.1f}  {r['bic_l']:>14.1f}")

    if results:
        best_k_full = min(results, key=lambda r: r["bic_f"])
        log(f"\n  最適モデル（Full）: k={best_k_full['k']}  BIC={best_k_full['bic_f']:.1f}")

    log(f"\n完了。出力先: {OUT_DIR.resolve()}")
