"""
trigram_hmm_fast.py
===================
Trigram HMM 高速学習版 — バッチ並列実装

【高速化の仕組み】
  原実装: for seq in 8060_seqs: for t in ~8_steps: GPU演算
          → Python ループ約64,000回（GPU カーネル起動コストが支配的）

  本実装: for t in ~15_steps: GPU演算(B=8060 シーケンスを一括処理)
          → Python ループ約15回（約4000倍の削減）

モデルパラメータ・保存形式・分析インタフェースは trigram_hmm_pytorch.py と同じ。
既存の trigram_hmm_pytorch.py は変更しない。
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
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch がインストールされていません。")
    import sys
    sys.exit(1)


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")


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
OUT_DIR      = Path("hypothesis/03_trigram/results")
CACHE_DIR    = OUT_DIR / "hmm_model_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE      = [9, 10]
N_RESTARTS   = 20
N_ITER       = 200
TOL          = 1e-4
MIN_WORD_LEN = 2

BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"

BIGRAM_CACHE = Path("hypothesis/01_bigram/results/hmm_model_cache")


# ── データ準備 ────────────────────────────────────────────────────────
def prepare_batch_data(words, char2idx):
    """
    全単語を (B, T_max) のパディング済みテンソルに変換する。

    Returns
    -------
    X_batch : np.ndarray (B, T_max)  int32
    lengths : np.ndarray (B,)        各系列の実際の長さ
    T_max   : int
    """
    pad_idx = char2idx[PAD_CHAR]
    processed = []
    lengths = []
    for w in words:
        seq = ([char2idx[BOS_CHAR]]
               + [char2idx[c] for c in w]
               + [char2idx[EOS_CHAR]])
        processed.append(seq)
        lengths.append(len(seq))

    T_max = max(lengths)
    X_matrix = [p + [pad_idx] * (T_max - len(p)) for p in processed]
    return (np.array(X_matrix, dtype=np.int32),
            np.array(lengths,  dtype=np.int32),
            T_max)


# ── Trigram HMM (バッチ並列版) ────────────────────────────────────────
class TrigramHMM_Batched:
    """
    全 B 系列を同時に処理する Baum-Welch 実装。

    内部ループは時間方向の T_max 回のみ。
    シーケンス方向 (B 次元) は GPU テンソル演算で並列化。

    Forward 変数:
      log_alpha[b, t, i, j] = log P(O^b_0..O^b_t, S_{t-1}=i, S_t=j)
      ただし t=0 は log_alpha[b, 0, 0, j] = log P(O^b_0, S_0=j) のみ有効。
    """

    def __init__(self, n_components, n_vocab, device=DEVICE):
        self.k = n_components
        self.v = n_vocab
        self.device = device

        self.log_start       = None  # (k,)
        self.log_start_trans = None  # (k, k)
        self.log_transmat    = None  # (k, k, k)
        self.log_emission    = None  # (k, v)

    def _init_params(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        k, v, d = self.k, self.v, self.device

        s = torch.rand(k, device=d) + 0.1
        self.log_start = torch.log(s / s.sum())

        st = torch.rand(k, k, device=d) + 0.1
        self.log_start_trans = torch.log(st / st.sum(dim=1, keepdim=True))

        tr = torch.rand(k, k, k, device=d) + 0.1
        self.log_transmat = torch.log(tr / tr.sum(dim=2, keepdim=True))

        em = torch.rand(k, v, device=d) + 0.1
        self.log_emission = torch.log(em / em.sum(dim=1, keepdim=True))

    def fit(self, X_batch_t, lengths_t, valid_mask_t, n_iter=N_ITER, tol=TOL, seed=None):
        """
        Parameters
        ----------
        X_batch_t   : torch.LongTensor (B, T_max)  パディング済み観測列
        lengths_t   : torch.LongTensor (B,)         実際の系列長
        valid_mask_t: torch.BoolTensor (B, T_max)   有効位置マスク
        """
        self._init_params(seed)

        B, T_max = X_batch_t.shape
        k, v = self.k, self.v
        d = self.device

        # 各系列の最終有効位置 (0-indexed)
        t_last = lengths_t - 1  # (B,)

        best_logL = -float('inf')
        INF = float('inf')

        for it in range(n_iter):

            # ════════════════════════════════════════════════
            # Forward pass: O(T_max) GPU 演算
            # ════════════════════════════════════════════════
            log_alpha = torch.full((B, T_max, k, k), -INF, device=d)

            # t=0: la_1d[b, j] = log_start[j] + log_emiss[j, X[b,0]]
            # log_emiss[:, X[:,0]] : (k, B) → .T → (B, k)
            la_1d = (self.log_start.unsqueeze(0)
                     + self.log_emission[:, X_batch_t[:, 0]].T)   # (B, k)
            log_alpha[:, 0, 0, :] = la_1d  # 行 0 のみ有効

            # t=1: la_2d[b, i, j] = la_1d[b,i] + start_trans[i,j] + emiss[j, X[b,1]]
            la_2d = (la_1d.unsqueeze(2)
                     + self.log_start_trans.unsqueeze(0))          # (B, k, k)
            la_2d = la_2d + self.log_emission[:, X_batch_t[:, 1]].T.unsqueeze(1)
            log_alpha[:, 1, :, :] = la_2d

            # t >= 2: Trigram 遷移
            for t in range(2, T_max):
                prev = log_alpha[:, t - 1, :, :]       # (B, k, k) = (B, i, j)
                # (B, k, k, 1) + (1, k, k, k) → (B, k, k, k) → logsumexp_i → (B, k, k)
                new_la = torch.logsumexp(
                    prev.unsqueeze(3) + self.log_transmat.unsqueeze(0),
                    dim=1
                )                                                    # (B, j, l)
                new_la = new_la + self.log_emission[:, X_batch_t[:, t]].T.unsqueeze(1)

                # 有効な系列のみ更新 (パディング位置には書かない)
                active = valid_mask_t[:, t]             # (B,)
                log_alpha[:, t, :, :] = torch.where(
                    active.view(B, 1, 1), new_la, log_alpha[:, t, :, :])

            # ── 対数尤度 ──────────────────────────────────
            # 各系列の最終位置の alpha を集めて logsumexp
            last_alpha = log_alpha[torch.arange(B, device=d), t_last]  # (B, k, k)
            seq_logL = torch.logsumexp(last_alpha.reshape(B, -1), dim=1)  # (B,)
            logL = seq_logL.sum().item()

            if np.isnan(logL):
                return -INF

            if it > 0 and (logL - best_logL) < tol:
                break
            best_logL = logL

            # ════════════════════════════════════════════════
            # Backward pass: O(T_max) GPU 演算
            # ════════════════════════════════════════════════
            log_beta = torch.full((B, T_max, k, k), -INF, device=d)

            # 各系列の最終位置で β = 0 (log(1))
            log_beta[torch.arange(B, device=d), t_last, :, :] = 0.0

            # t = T_max-2 down to 1: Trigram 後向き
            for t in range(T_max - 2, 0, -1):
                lb_next = log_beta[:, t + 1, :, :]     # (B, k, k) = (B, j, l)
                e_next = self.log_emission[:, X_batch_t[:, t + 1]].T  # (B, l)
                # (1, k, k, k) + (B, 1, 1, k) + (B, 1, k, k)
                val = (self.log_transmat.unsqueeze(0)
                       + e_next.view(B, 1, 1, k)
                       + lb_next.unsqueeze(1))          # (B, k, k, k)
                new_lb = torch.logsumexp(val, dim=3)    # (B, k, k) = (B, i, j)

                # t が系列の最終位置より前の場合のみ更新
                active = (t < t_last)                   # (B,)
                log_beta[:, t, :, :] = torch.where(
                    active.view(B, 1, 1), new_lb, log_beta[:, t, :, :])

            # t=0: start_trans 後向き
            # beta_0[b, j] = logsumexp_l(start_trans[j,l] + emiss[l, X[b,1]] + beta[b,1,j,l])
            lb_1 = log_beta[:, 1, :, :]                # (B, k, k) = (B, j, l)
            e_1  = self.log_emission[:, X_batch_t[:, 1]].T  # (B, l)
            val_0 = (self.log_start_trans.unsqueeze(0)
                     + e_1.view(B, 1, k)
                     + lb_1)                             # (B, k, k) = (B, j, l)
            beta_0_1d = torch.logsumexp(val_0, dim=2)  # (B, k) = (B, j)
            log_beta[:, 0, 0, :] = beta_0_1d           # 行 0 のみ有効

            # ════════════════════════════════════════════════
            # M-step
            # ════════════════════════════════════════════════
            # γ_2d[b, t, i, j] = log_alpha[b,t,i,j] + log_beta[b,t,i,j]
            log_gamma_2d = log_alpha + log_beta         # (B, T_max, k, k)

            # Z[b, t] = logsumexp_{i,j}(γ_2d[b,t,i,j])
            Z = torch.logsumexp(
                log_gamma_2d.reshape(B, T_max, -1), dim=2)   # (B, T_max)

            # 正規化 γ_2d
            log_gamma_norm = log_gamma_2d - Z.unsqueeze(2).unsqueeze(3)  # (B, T_max, k, k)

            # γ_1d[b, t, j] = logsumexp_i(γ_2d_norm[b,t,i,j])
            log_gamma_1d = torch.logsumexp(log_gamma_norm, dim=2)  # (B, T_max, k)

            # パディング位置を -inf にマスク (emission 更新の汚染を防ぐ)
            log_gamma_1d = torch.where(
                valid_mask_t.unsqueeze(2), log_gamma_1d,
                torch.full_like(log_gamma_1d, -INF))

            # 1. start: t=0 の γ_1d を全系列で集計
            new_log_start = torch.logsumexp(log_gamma_1d[:, 0, :], dim=0)  # (k,)
            self.log_start = new_log_start - torch.logsumexp(new_log_start, dim=0)

            # 2. start_trans: t=1 の γ_2d_norm を全系列で集計
            new_log_st = torch.logsumexp(log_gamma_norm[:, 1, :, :], dim=0)  # (k, k)
            self.log_start_trans = new_log_st - torch.logsumexp(
                new_log_st, dim=1, keepdim=True)

            # 3. transmat: ξ を t=1..T_max-2 で逐次集計
            # - t=0 は start_trans が担うため除外
            # - t >= t_last[b] は系列境界のため除外
            trans_accum = torch.full((k, k, k), -INF, device=d)

            for t in range(1, T_max - 1):
                la_t  = log_alpha[:, t, :, :]           # (B, k, k) = (B, i, j)
                lb_t1 = log_beta[:, t + 1, :, :]       # (B, k, k) = (B, j, l)
                e_t1  = self.log_emission[:, X_batch_t[:, t + 1]].T  # (B, l)

                # log_xi[b, i, j, l] = la[b,i,j] + A[i,j,l] + e[b,l] + lb[b,j,l] - Z[b,t]
                log_xi_t = (la_t.unsqueeze(3)
                            + self.log_transmat.unsqueeze(0)
                            + e_t1.view(B, 1, 1, k)
                            + lb_t1.unsqueeze(1)
                            - Z[:, t].view(B, 1, 1, 1))  # (B, k, k, k)

                # 系列境界は除外: t >= t_last[b] の系列は invalid
                active_b = (t < t_last)                  # (B,)
                log_xi_t = torch.where(
                    active_b.view(B, 1, 1, 1), log_xi_t,
                    torch.full_like(log_xi_t, -INF))

                # 全系列で集計: logsumexp_b → (k, k, k)
                step = torch.logsumexp(log_xi_t, dim=0)
                trans_accum = torch.logaddexp(trans_accum, step)

            # 行 (i, j) ごとに正規化
            self.log_transmat = (trans_accum
                                 - torch.logsumexp(trans_accum, dim=2, keepdim=True))

            # 4. emission: γ_1d を使って E[j, v] = Σ_{b,t} [X[b,t]=v] * γ[b,t,j]
            # (B*T_max, k) の gamma と (B*T_max, V) の one-hot の行列積
            gamma_flat = log_gamma_1d.reshape(B * T_max, k)  # (B*T_max, k)
            X_flat     = X_batch_t.reshape(-1)               # (B*T_max,)

            X_onehot = torch.zeros(B * T_max, v, device=d)
            X_onehot.scatter_(1, X_flat.unsqueeze(1), 1.0)

            max_lg  = gamma_flat.max(dim=0, keepdim=True).values
            exp_g   = torch.exp(gamma_flat - max_lg)         # (B*T_max, k)
            new_emiss = torch.matmul(exp_g.T, X_onehot)      # (k, v)
            log_new_emiss = torch.log(new_emiss + 1e-30) + max_lg.T
            self.log_emission = (log_new_emiss
                                 - torch.logsumexp(log_new_emiss, dim=1, keepdim=True))

        return best_logL

    def score(self, X_batch_t, lengths_t, valid_mask_t):
        """全系列の対数尤度合計を計算"""
        B, T_max = X_batch_t.shape
        k, d = self.k, self.device
        INF = float('inf')
        t_last = lengths_t - 1

        log_alpha = torch.full((B, T_max, k, k), -INF, device=d)
        la_1d = self.log_start.unsqueeze(0) + self.log_emission[:, X_batch_t[:, 0]].T
        log_alpha[:, 0, 0, :] = la_1d
        la_2d = la_1d.unsqueeze(2) + self.log_start_trans.unsqueeze(0)
        la_2d = la_2d + self.log_emission[:, X_batch_t[:, 1]].T.unsqueeze(1)
        log_alpha[:, 1, :, :] = la_2d
        for t in range(2, T_max):
            prev = log_alpha[:, t - 1, :, :]
            new_la = torch.logsumexp(
                prev.unsqueeze(3) + self.log_transmat.unsqueeze(0), dim=1)
            new_la = new_la + self.log_emission[:, X_batch_t[:, t]].T.unsqueeze(1)
            active = valid_mask_t[:, t]
            log_alpha[:, t, :, :] = torch.where(
                active.view(B, 1, 1), new_la, log_alpha[:, t, :, :])

        last_alpha = log_alpha[torch.arange(B, device=d), t_last]
        return torch.logsumexp(last_alpha.reshape(B, -1), dim=1).sum().item()

    def viterbi(self, X_np):
        """単一系列の Viterbi デコード (trigram_hmm_pytorch.py と同一ロジック)"""
        X = torch.tensor(X_np, dtype=torch.long, device=self.device)
        T = X.shape[0]
        k, d = self.k, self.device
        INF = float('inf')

        if T == 0:
            return -INF, []

        log_delta_1d = self.log_start + self.log_emission[:, X[0]]

        if T == 1:
            best = torch.argmax(log_delta_1d).item()
            return log_delta_1d[best].item(), [best]

        log_delta = (log_delta_1d.unsqueeze(1) + self.log_start_trans
                     + self.log_emission[:, X[1]].unsqueeze(0))  # (k, k)

        psi_list = []
        for t in range(2, T):
            vals = log_delta.unsqueeze(2) + self.log_transmat  # (k, k, k)
            max_vals, argmax_i = torch.max(vals, dim=0)
            new_delta = max_vals + self.log_emission[:, X[t]].unsqueeze(0)
            psi_list.append(argmax_i.cpu())
            log_delta = new_delta

        best_score = torch.max(log_delta).item()
        flat_idx   = torch.argmax(log_delta)
        j_last, l_last = (flat_idx // k).item(), (flat_idx % k).item()

        path = [0] * T
        path[T - 1] = l_last
        path[T - 2] = j_last
        for t_back in range(T - 2, 1, -1):
            path[t_back - 1] = psi_list[t_back - 2][path[t_back], path[t_back + 1]].item()

        return best_score, path


# ── 学習ループ ────────────────────────────────────────────────────────
def run_training_fast(k, X_batch_t, lengths_t, valid_mask_t):
    best_model = None
    best_logL  = -float('inf')
    for seed in range(N_RESTARTS):
        model = TrigramHMM_Batched(n_components=k, n_vocab=V, device=DEVICE)
        logL = model.fit(X_batch_t, lengths_t, valid_mask_t,
                         n_iter=N_ITER, tol=TOL, seed=seed)
        if logL > best_logL:
            best_logL = logL
            best_model = model
        log(f"  seed={seed:02d}  logL={logL:.2f}  best={best_logL:.2f}")
    return best_model, best_logL


def get_transmat(model):
    return torch.exp(model.log_transmat).cpu().numpy()     # (k, k, k)


def get_emission(model):
    return torch.exp(model.log_emission).cpu().numpy()     # (k, v)


def compute_bic(log_likelihood, X_len, k, V):
    n_params = (k - 1) + k * (k - 1) + k * k * (k - 1) + k * (V - 1)
    bic = -2 * log_likelihood + n_params * np.log(X_len)
    aic = -2 * log_likelihood + 2 * n_params
    return bic, aic, n_params


def save_model(model, logL, path):
    np.savez(
        path,
        start       = torch.exp(model.log_start).cpu().numpy(),
        start_trans = torch.exp(model.log_start_trans).cpu().numpy(),
        trans       = torch.exp(model.log_transmat).cpu().numpy(),
        emiss       = torch.exp(model.log_emission).cpu().numpy(),
        logL        = np.array([logL]),
    )


# ── 可視化 ────────────────────────────────────────────────────────────
def plot_transition_heatmap(transmat, k, out_path):
    ncols = min(k, 4)
    nrows = (k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.array(axes).reshape(-1)
    state_labels = [f"S{j}" for j in range(k)]
    for i in range(k):
        ax = axes[i]
        mat = pd.DataFrame(transmat[i], index=state_labels, columns=state_labels)
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    ax=ax, linewidths=0.3, linecolor="gray",
                    cbar_kws={"shrink": 0.7}, annot_kws={"size": 7})
        ax.set_title(f"s_{{t-2}} = S{i}", fontsize=10)
        ax.set_xlabel("s_t", fontsize=8)
        ax.set_ylabel("s_{{t-1}}", fontsize=8)
        ax.tick_params(labelsize=7)
    for i in range(k, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(f"Trigram 遷移テンソル P(s_t|s_{{t-2}},s_{{t-1}})  k={k}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_emission_heatmap(emiss, k, all_chars, out_path):
    char_labels  = [("BOS" if c == "^" else ("EOS" if c == "$" else c))
                    for c in all_chars]
    state_labels = [f"S{j}" for j in range(k)]
    B = pd.DataFrame(emiss, index=state_labels, columns=char_labels)
    fig, ax = plt.subplots(figsize=(max(14, len(all_chars) * 0.55), max(4, k * 1.5)))
    sns.heatmap(B, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, ax=ax,
                linewidths=0.3, linecolor="gray", cbar_kws={"shrink": 0.6},
                annot_kws={"size": 7})
    ax.set_title(f"放出確率行列 (Trigram HMM fast, k={k})", fontsize=12)
    ax.set_xlabel("観測文字", fontsize=9)
    ax.set_ylabel("隠れ状態", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bic_comparison(trigram_results, bigram_bics, out_path):
    ks      = [r["k"] for r in trigram_results]
    tri_bics = [r["bic"] for r in trigram_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ks))
    ax.bar(x, tri_bics, width=0.4, label="Trigram HMM", color="steelblue", alpha=0.85)
    for k_val, bic_val in bigram_bics.items():
        if k_val in ks:
            xi = ks.index(k_val)
            ax.hlines(bic_val, xi - 0.25, xi + 0.25, colors="coral",
                      linestyles="--", linewidths=2)
            ax.annotate(f"Bigram\n{bic_val:.0f}", xy=(xi + 0.28, bic_val),
                        fontsize=7, color="coral", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_xlabel("状態数 k")
    ax.set_ylabel("BIC（小さいほど良い）")
    ax.set_title("BIC 比較: Trigram HMM vs Bigram HMM")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def decode_examples(model, k, words, char2idx):
    state_labels = [f"S{j}" for j in range(k)]
    lines = [f"{'='*70}", f"  Viterbi 状態割り当て例 (Trigram HMM fast, k={k})", f"{'='*70}"]
    sample = sorted(words, key=len)
    sample_words = sample[:5] + sample[len(sample)//2-3:len(sample)//2+3] + sample[-5:]
    seen = set()
    for w in sample_words:
        if w in seen:
            continue
        seen.add(w)
        seq = ([char2idx[BOS_CHAR]] + [char2idx[c] for c in w]
               + [char2idx[EOS_CHAR]])
        X_w = np.array(seq, dtype=np.int32)
        try:
            _, state_seq = model.viterbi(X_w)
            full_chars  = [BOS_CHAR] + list(w) + [EOS_CHAR]
            char_disp   = " ".join(f"{c:>3}" for c in full_chars)
            state_disp  = " ".join(f"S{s:>1}" for s in state_seq)
            lines.append(f"  {w:<15}  chars: {char_disp}")
            lines.append(f"  {'':15}  state: {state_disp}")
            lines.append("")
        except Exception:
            continue
    return "\n".join(lines)


def build_training_report(results, bigram_bics, all_chars):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Trigram HMM 学習レポート (高速版)",
        f"",
        f"**生成日時**: {now}",
        f"**設定**: K_RANGE={K_RANGE}, N_RESTARTS={N_RESTARTS}, N_ITER={N_ITER}",
        f"**語彙サイズ V**: {len(all_chars)}",
        f"",
        f"## BIC / AIC",
        f"",
        f"| k | logL | n_params | BIC (Trigram) | AIC | BIC (Bigram 参照) |",
        f"|---|------|----------|--------------|-----|-----------------|",
    ]
    for r in results:
        ref = bigram_bics.get(r["k"], float("nan"))
        lines.append(
            f"| {r['k']} | {r['logL']:.1f} | {r['n_params']} "
            f"| **{r['bic']:.1f}** | {r['aic']:.1f} | {ref:.1f} |"
        )
    lines += ["", "---", "",
              "_次のステップ: `trigram_role_analysis.py` で役割曖昧性を検証_"]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("Loading data from DB...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
        conn,
    )
    conn.close()

    all_types = sorted(set(df["word"].tolist()))
    all_types = [w for w in all_types if len(w) >= MIN_WORD_LEN]
    log(f"ユニーク単語数: {len(all_types):,}")

    # 語彙構築 (PAD を追加)
    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR] + raw_chars + [EOS_CHAR, PAD_CHAR]
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    V         = len(all_chars)
    log(f"語彙サイズ: {V}  (PAD 込み)")

    # バッチデータ準備
    X_np, lengths_np, T_max = prepare_batch_data(all_types, char2idx)
    log(f"X_batch: {X_np.shape}  T_max={T_max}")

    # GPU テンソルに変換
    X_batch_t    = torch.tensor(X_np,     dtype=torch.long, device=DEVICE)
    lengths_t    = torch.tensor(lengths_np, dtype=torch.long, device=DEVICE)
    t_range      = torch.arange(T_max, device=DEVICE).unsqueeze(0)
    valid_mask_t = t_range < lengths_t.unsqueeze(1)  # (B, T_max)

    # 観測総数 (BIC 計算用)
    total_obs = int(lengths_np.sum())
    log(f"観測総数: {total_obs:,}")

    # Bigram BIC 参照
    bigram_bics = {}
    for k_bi in [7, 8]:
        p = BIGRAM_CACHE / f"full_k{k_bi}.npz"
        if p.exists():
            d = np.load(p)
            logL_bi = float(d["logL"][0])
            k_val   = d["trans"].shape[0]
            n_bi    = k_val * (k_val - 1) + k_val * (V - 1) + (k_val - 1)
            bic_bi  = -2 * logL_bi + n_bi * np.log(total_obs)
            bigram_bics[k_bi] = bic_bi
            log(f"Bigram k={k_bi}: logL={logL_bi:.1f}  BIC={bic_bi:.1f}")

    results = []

    for k in K_RANGE:
        log(f"\n{'─'*60}")
        log(f"Trigram HMM (fast)  k = {k}  (N_RESTARTS={N_RESTARTS})")
        log(f"{'─'*60}")

        model, logL = run_training_fast(k, X_batch_t, lengths_t, valid_mask_t)

        if model is None or logL == -float('inf'):
            log(f"k={k}: 学習失敗")
            continue

        bic, aic, n_params = compute_bic(logL, total_obs, k, V)
        log(f"k={k}  logL={logL:.2f}  BIC={bic:.1f}  AIC={aic:.1f}  n_params={n_params}")

        cache_path = CACHE_DIR / f"trigram_k{k}.npz"
        save_model(model, logL, cache_path)
        log(f"モデル保存: {cache_path}")

        transmat     = get_transmat(model)
        emiss        = get_emission(model)
        start_trans  = torch.exp(model.log_start_trans).cpu().numpy()

        plot_transition_heatmap(transmat, k, OUT_DIR / f"transition_heatmap_k{k}.png")
        plot_emission_heatmap(emiss, k, all_chars, OUT_DIR / f"emission_heatmap_k{k}.png")

        results.append({"k": k, "logL": logL, "bic": bic, "aic": aic,
                         "n_params": n_params, "start_trans": start_trans})

        ex_text = decode_examples(model, k, all_types, char2idx)
        (OUT_DIR / f"word_examples_k{k}.txt").write_text(ex_text, encoding="utf-8")

    if results:
        plot_bic_comparison(results, bigram_bics, OUT_DIR / "bic_comparison.png")

    report = build_training_report(results, bigram_bics, all_chars)
    report_path = OUT_DIR / "training_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"レポート: {report_path}")
    log("✓ 完了。出力先: " + str(OUT_DIR.resolve()))
