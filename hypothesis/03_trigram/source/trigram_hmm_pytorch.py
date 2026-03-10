"""
trigram_hmm_pytorch.py
======================
Voynich Manuscript 複合語仮説の数理検証
【Trigram HMM: P(s_t | s_{t-1}, s_{t-2}) PyTorch GPU 版】

Bigram HMM では S4 (Focus State) に「基末尾」と「基先頭」の役割が混在していた。
Trigram HMM は 2 ステップ前の文脈を参照することで、この役割曖昧性を
局所文脈から解消できるか検証するための学習スクリプト。

パラメータ構成:
  log_start      : (k,)      — P(S_0 = j)
  log_start_trans: (k, k)    — P(S_1 = j | S_0 = i)  ← 第 1 ステップ専用
  log_transmat   : (k, k, k) — P(S_t = l | S_{t-2}=i, S_{t-1}=j)  ← t >= 2 で使用
  log_emission   : (k, v)    — 同 Bigram

参照: idea/03_trigram_implementation_plan.md
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

K_RANGE      = [5, 6, 7, 8]
N_RESTARTS   = 20
N_ITER       = 200
TOL          = 1e-4
MIN_WORD_LEN = 2

BOS_CHAR, EOS_CHAR = "^", "$"

# Bigram モデルとの BIC 比較用
BIGRAM_CACHE = Path("hypothesis/01_bigram/results/hmm_model_cache")


# ── データ準備 ────────────────────────────────────────────────────────
def encode_words(words, char2idx):
    """単語リストを連結した観測列に変換（Bigram と同形式）"""
    seqs = []
    lengths = []
    for w in words:
        seq = [char2idx[BOS_CHAR]] + [char2idx[c] for c in w] + [char2idx[EOS_CHAR]]
        seqs.extend(seq)
        lengths.append(len(seq))
    X = np.array(seqs, dtype=np.int32)
    return X, lengths


# ── Trigram HMM 実装 (Log-Domain) ────────────────────────────────────
class TrigramHMM_PT:
    """
    Trigram HMM: P(s_t | s_{t-1}, s_{t-2})

    パラメータ:
      log_start      : (k,)       — 初期状態分布
      log_start_trans: (k, k)     — 第1遷移 (t=0→1 専用)
      log_transmat   : (k, k, k)  — Trigram 遷移 (t>=2 で使用)
                                    dim0=s_{t-2}, dim1=s_{t-1}, dim2=s_t
      log_emission   : (k, v)     — 放出分布

    Forward 変数: log_alpha[i, j] = log P(O_1..O_t, S_{t-1}=i, S_t=j)  shape (k, k)
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

        # Trigram 遷移: (k, k, k) — 行 (i, j) ごとに正規化
        tr = torch.rand(k, k, k, device=d) + 0.1
        self.log_transmat = torch.log(tr / tr.sum(dim=2, keepdim=True))

        em = torch.rand(k, v, device=d) + 0.1
        self.log_emission = torch.log(em / em.sum(dim=1, keepdim=True))

    def _forward(self, X, starts, ends):
        """
        Trigram Forward アルゴリズム (全系列連結版)

        Returns:
          log_alpha_all : (T, k, k) — t 時点の (s_{t-1}, s_t) 結合前向き確率
                          ただし t=0 は (k,) のみ意味があるため padding あり
          log_alpha_1d  : (T,) を除いた第0時点用 (k,) テンソル (各系列開始時)
          per_seq_logL  : 各系列の対数尤度 (1D Tensor)
        """
        T = X.shape[0]
        k = self.k
        d = self.device

        # (T, k, k) を確保。t=0 は (k,) なので、log_alpha[0] は使わない
        log_alpha = torch.full((T, k, k), -float('inf'), device=d)

        # 各系列の開始 t での初期化、続くステップの伝播
        for seq_idx in range(len(starts)):
            t0 = starts[seq_idx].item()
            t_end = ends[seq_idx].item()  # exclusive

            # t=t0: log_alpha_1d[j] = log_start[j] + log_emiss[j, O_{t0}]
            la_1d = self.log_start + self.log_emission[:, X[t0]]  # (k,)

            # 行0に格納。他の行は -inf のまま。
            # これにより Z[t0] = logsumexp_j(la_1d + beta_0) が正しく計算される。
            log_alpha[t0, 0, :] = la_1d

            if t_end - t0 == 1:
                continue

            # t=t0+1: log_alpha[t0+1, i, j]
            #   = la_1d[i] + log_start_trans[i, j] + log_emiss[j, O_{t0+1}]
            t1 = t0 + 1
            # la_1d: (k,) → (k, 1) broadcast
            la_2d = la_1d.unsqueeze(1) + self.log_start_trans  # (k, k)
            la_2d = la_2d + self.log_emission[:, X[t1]].unsqueeze(0)  # (k, k)
            log_alpha[t1] = la_2d

            # t >= t0+2: Trigram 遷移
            for t in range(t0 + 2, t_end):
                # log_alpha[t, j, l] = logsumexp_i(log_alpha[t-1, i, j] + log_transmat[i, j, l])
                #                     + log_emiss[l, O_t]
                # log_alpha[t-1]: (k, k) = (i, j)
                # log_transmat  : (k, k, k) = (i, j, l)
                prev = log_alpha[t - 1]  # (k, k) = (i, j)
                # prev.unsqueeze(2): (k, k, 1) + log_transmat: (k, k, k)
                new_la = torch.logsumexp(prev.unsqueeze(2) + self.log_transmat, dim=0)  # (k, k)
                new_la = new_la + self.log_emission[:, X[t]].unsqueeze(0)  # (k, k)
                log_alpha[t] = new_la

        return log_alpha

    def _backward(self, X, starts, ends):
        """
        Trigram Backward アルゴリズム

        Returns:
          log_beta: (T, k, k) — t 時点の (s_{t-1}, s_t) 結合後向き確率
        """
        T = X.shape[0]
        k = self.k
        d = self.device

        log_beta = torch.full((T, k, k), -float('inf'), device=d)

        for seq_idx in range(len(starts)):
            t0 = starts[seq_idx].item()
            t_end = ends[seq_idx].item()  # exclusive
            T_seq = t_end - t0

            if T_seq == 1:
                log_beta[t0, 0, :] = 0.0
                continue

            # t = t_end - 1: β = 0 (log(1) = 0)
            log_beta[t_end - 1] = 0.0

            # t_end-2 の場合: start_trans の逆方向
            # β_{t0+1}(i, j) = Σ_l start_trans[i,j]? いや、それは log_start_trans
            # 実際には t0+1 から t0 への逆伝播は:
            # β_{t0}(j) = Σ_l [start_trans[j, l] + emiss[l, O_{t0+1}] + β_{t0+1}(j, l)]
            # ← この t0 での β は (k,) だが、実装上は (k, k) に格納
            # (後の集計で使いやすいよう、log_beta[t0, i, j] に β_{t0}(j) を i に依存しない形で入れる)

            # t >= t0+2: Trigram 後向き
            for t in range(t_end - 2, t0, -1):
                # log_beta[t, i, j] = logsumexp_l(log_transmat[i,j,l] + log_emiss[l, O_{t+1}] + log_beta[t+1, j, l])
                lb_next = log_beta[t + 1]  # (k, k) = (j, l)
                emiss_next = self.log_emission[:, X[t + 1]]  # (k,) = (l,)
                # log_transmat: (k, k, k) = (i, j, l)
                # lb_next: (k, k) = (j, l) → unsqueeze(0): (1, k, k)
                # emiss_next: (k,) → view(1, 1, k): (1, 1, k)
                val = self.log_transmat + emiss_next.view(1, 1, -1) + lb_next.unsqueeze(0)  # (k, k, k)
                log_beta[t] = torch.logsumexp(val, dim=2)  # (k, k) = (i, j)

            # t = t0: β を (k,) として集計するため、start_trans 経由の逆伝播
            # log_beta[t0, *, j]:
            # β_{t0}(j) = Σ_l [log_start_trans[j, l] + log_emiss[l, O_{t0+1}] + log_beta[t0+1, j, l]]
            t1 = t0 + 1
            lb_1 = log_beta[t1]       # (k, k) = (j, l)
            emiss_1 = self.log_emission[:, X[t1]]  # (k,) = (l,)
            # log_start_trans: (k, k) = (j, l)
            val_0 = self.log_start_trans + emiss_1.unsqueeze(0) + lb_1  # (k, k)
            beta_0_1d = torch.logsumexp(val_0, dim=1)  # (k,) = (j,)
            # log_beta[t0] に (k,) を broadcast して格納 (i 次元に関わらず同値)
            log_beta[t0] = beta_0_1d.unsqueeze(0).expand(k, k)

        return log_beta

    def _compute_logL(self, log_alpha, starts, ends):
        """各系列の対数尤度を計算"""
        logL_total = 0.0
        for seq_idx in range(len(starts)):
            t_last = ends[seq_idx].item() - 1
            logL_total += torch.logsumexp(log_alpha[t_last].reshape(-1), dim=0).item()
        return logL_total

    def fit(self, X_np, lengths, n_iter=N_ITER, tol=TOL, seed=None):
        """Baum-Welch アルゴリズム (Trigram HMM 版)"""
        self._init_params(seed)
        X = torch.tensor(X_np, dtype=torch.long, device=self.device)

        # 系列の開始・終了位置
        lengths_arr = np.array(lengths)
        ends_np = np.cumsum(lengths_arr)
        starts_np = np.concatenate([[0], ends_np[:-1]])
        starts = torch.tensor(starts_np, device=self.device)
        ends   = torch.tensor(ends_np,   device=self.device)

        best_logL = -float('inf')

        for it in range(n_iter):
            # ── E-step ──────────────────────────────────────────────
            log_alpha = self._forward(X, starts, ends)   # (T, k, k)
            log_beta  = self._backward(X, starts, ends)  # (T, k, k)

            logL = self._compute_logL(log_alpha, starts, ends)
            if np.isnan(logL):
                return -float('inf')

            if it > 0 and (logL - best_logL) < tol:
                break
            best_logL = logL

            # ── M-step ──────────────────────────────────────────────
            k, v = self.k, self.v
            d = self.device
            T = X.shape[0]

            # γ_2d[t, i, j] = P(S_{t-1}=i, S_t=j | O)
            log_gamma_2d = log_alpha + log_beta  # (T, k, k)

            # 正規化定数 Z_t = logsumexp_{i,j}(log_alpha[t] + log_beta[t])
            Z = torch.logsumexp(log_gamma_2d.reshape(T, -1), dim=1)  # (T,)

            # 正規化された γ_2d
            log_gamma_norm = log_gamma_2d - Z.view(T, 1, 1)  # (T, k, k)

            # γ_1d[t, j] = P(S_t=j | O) = logsumexp_i γ_2d[t, i, j]
            log_gamma_1d = torch.logsumexp(log_gamma_norm, dim=1)  # (T, k)

            # 1. start の更新: start[j] ∝ Σ_seqs γ_1d[t0, j]
            start_accum = torch.stack([log_gamma_1d[s] for s in starts_np])  # (N_seqs, k)
            new_log_start = torch.logsumexp(start_accum, dim=0)
            self.log_start = new_log_start - torch.logsumexp(new_log_start, dim=0)

            # 2. start_trans の更新: start_trans[i, j] ∝ Σ_seqs γ_2d[t1, i, j]
            t1_indices = starts_np + 1  # t0+1 が有効な系列のみ
            valid_t1 = [i for i in t1_indices if i < T]
            if valid_t1:
                st_accum = torch.stack([log_gamma_norm[t] for t in valid_t1])  # (N, k, k)
                new_log_st = torch.logsumexp(st_accum, dim=0)  # (k, k)
                self.log_start_trans = new_log_st - torch.logsumexp(new_log_st, dim=1, keepdim=True)

            # 3. transmat の更新
            # ξ_t(i, j, l) = log_alpha[t, i, j] + log_transmat[i, j, l]
            #               + log_emiss[l, O_{t+1}] + log_beta[t+1, j, l] - Z[t]
            # 有効な t: t0+2 ≤ t ≤ t_end-2 (境界マスク)
            trans_accum = torch.full((k, k, k), -float('inf'), device=d)

            # 全時点の ξ を一括計算 (境界は後でマスク)
            # log_alpha[:-1]: (T-1, k, k), log_beta[1:]: (T-1, k, k)
            # log_transmat: (k, k, k) → unsqueeze(0): (1, k, k, k)
            # log_emiss[:, X[1:]]: (k, T-1) → permute → (T-1, 1, 1, k)
            emiss_next = self.log_emission[:, X[1:]].T.reshape(T - 1, 1, 1, k)  # (T-1, 1, 1, k)
            la = log_alpha[:-1].unsqueeze(3)   # (T-1, k, k, 1)
            lb = log_beta[1:].unsqueeze(1)     # (T-1, 1, k, k)
            tr = self.log_transmat.unsqueeze(0)  # (1, k, k, k)

            log_xi = la + tr + emiss_next + lb  # (T-1, k, k, k)
            log_xi = log_xi - Z[:-1].view(T - 1, 1, 1, 1)

            # 無効な t をマスク:
            # - 各系列の最終位置 (t = t_end - 1): 境界越え
            # - 各系列の t0 (start): この遷移は start_trans が担う → transmat 更新から除外
            # - t0+1 は除外しない: A[S_{t0}, S_{t0+1}, S_{t0+2}] は有効な Trigram 遷移
            invalid_mask = torch.zeros(T - 1, dtype=torch.bool, device=d)
            for seq_idx in range(len(starts_np)):
                t0 = starts_np[seq_idx]
                t_end_val = ends_np[seq_idx]
                # 境界: t_end-1 の位置（X[t_end] は次系列の観測）
                if t_end_val - 1 < T - 1:
                    invalid_mask[t_end_val - 1] = True
                # t0: start_trans の更新用 (transmat からは除外)
                if t0 < T - 1:
                    invalid_mask[t0] = True
                # ※ t0+1 は有効な Trigram 遷移 A[S_t0, S_{t0+1}, S_{t0+2}] なので除外しない

            log_xi[invalid_mask] = -float('inf')

            trans_accum = torch.logsumexp(log_xi, dim=0)  # (k, k, k)
            # 行 (i, j) ごとに正規化
            trans_accum_norm = trans_accum - torch.logsumexp(trans_accum, dim=2, keepdim=True)
            self.log_transmat = trans_accum_norm

            # 4. emission の更新
            # E[j, v] = Σ_t [O_t=v] × γ_1d[t, j]
            X_onehot = torch.zeros(T, v, device=d)
            X_onehot.scatter_(1, X.unsqueeze(1), 1.0)

            max_lg = log_gamma_1d.max(dim=0, keepdim=True).values
            exp_g = torch.exp(log_gamma_1d - max_lg)         # (T, k)
            new_emiss = torch.matmul(exp_g.T, X_onehot)      # (k, v)
            log_new_emiss = torch.log(new_emiss + 1e-30) + max_lg.T
            log_new_emiss = log_new_emiss - torch.logsumexp(log_new_emiss, dim=1, keepdim=True)
            self.log_emission = log_new_emiss

        return best_logL

    def score(self, X_np, lengths):
        """対数尤度を計算"""
        X = torch.tensor(X_np, dtype=torch.long, device=self.device)
        ends_np = np.cumsum(lengths)
        starts_np = np.concatenate([[0], ends_np[:-1]])
        starts = torch.tensor(starts_np, device=self.device)
        ends   = torch.tensor(ends_np,   device=self.device)
        log_alpha = self._forward(X, starts, ends)
        return self._compute_logL(log_alpha, starts_np, ends_np)

    def viterbi(self, X_np):
        """単一系列の Viterbi デコーディング"""
        X = torch.tensor(X_np, dtype=torch.long, device=self.device)
        T = X.shape[0]
        k = self.k
        d = self.device

        if T == 0:
            return -float('inf'), []

        # t=0: log_delta_1d[j]
        log_delta_1d = self.log_start + self.log_emission[:, X[0]]  # (k,)

        if T == 1:
            best_s = torch.argmax(log_delta_1d).item()
            return log_delta_1d[best_s].item(), [best_s]

        # log_delta[i, j] = max_{S_1..S_{t-1}} P(...)  shape (k, k)
        log_delta = torch.full((k, k), -float('inf'), device=d)
        psi_2d = []  # psi_2d[t-2][i, j] = argmax_h at time t (for t >= 2)

        # t=1: (i, j) 初期化
        log_delta = log_delta_1d.unsqueeze(1) + self.log_start_trans  # (k, k)
        log_delta = log_delta + self.log_emission[:, X[1]].unsqueeze(0)  # (k, k)

        # t >= 2
        for t in range(2, T):
            # new_delta[j, l] = max_i(log_delta[i, j] + log_transmat[i, j, l]) + emiss[l, O_t]
            # log_delta: (k, k) = (i, j) → unsqueeze(2): (k, k, 1)
            # log_transmat: (k, k, k) = (i, j, l)
            vals = log_delta.unsqueeze(2) + self.log_transmat  # (k, k, k)
            max_vals, argmax_i = torch.max(vals, dim=0)  # (k, k), (k, k) = (j, l)
            new_delta = max_vals + self.log_emission[:, X[t]].unsqueeze(0)  # (k, k)
            psi_2d.append(argmax_i.cpu())  # (k, k) = (j, l)
            log_delta = new_delta

        # バックトラック
        best_score = torch.max(log_delta).item()
        j_last, l_last = (log_delta == log_delta.max()).nonzero(as_tuple=False)[0]
        j_last, l_last = j_last.item(), l_last.item()

        path = [0] * T
        path[T - 1] = l_last
        path[T - 2] = j_last

        for t_back in range(T - 2, 1, -1):
            # psi_2d[t_back - 2][j, l] = argmax_i at time t_back + 1
            # 現在 path[t_back] = j, path[t_back+1] = l
            i_prev = psi_2d[t_back - 2][path[t_back], path[t_back + 1]].item()
            path[t_back - 1] = i_prev

        # t=1 から t=0 の復元は log_delta_1d から
        # (path[1] = j は既に確定)

        return best_score, path


# ── 学習ループ ────────────────────────────────────────────────────────
def run_training(k, X_all, L_all):
    best_model = None
    best_logL = -float('inf')
    for seed in range(N_RESTARTS):
        model = TrigramHMM_PT(n_components=k, n_vocab=V, device=DEVICE)
        logL = model.fit(X_all, L_all, n_iter=N_ITER, tol=TOL, seed=seed)
        if logL > best_logL:
            best_logL = logL
            best_model = model
        log(f"  seed={seed:02d}  logL={logL:.2f}  best={best_logL:.2f}")
    return best_model, best_logL


def get_transmat(model):
    return torch.exp(model.log_transmat).cpu().numpy()  # (k, k, k)


def get_emission(model):
    return torch.exp(model.log_emission).cpu().numpy()  # (k, v)


def compute_bic(log_likelihood, X_len, k, V):
    """Trigram HMM のパラメータ数に基づく BIC"""
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
    """
    Trigram 遷移テンソル (k, k, k) を k 枚のサブプロットで可視化。
    各サブプロット = P(S_t=l | S_{t-2}=i, S_{t-1}=j)  (fixed i)
    """
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
        ax.set_ylabel("s_{t-1}", fontsize=8)
        ax.tick_params(labelsize=7)

    # 余分な axes を非表示
    for i in range(k, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Trigram 遷移テンソル P(s_t | s_{{t-2}}, s_{{t-1}})  k={k}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_emission_heatmap(emiss, k, all_chars, out_path):
    char_labels = [("BOS" if c == "^" else ("EOS" if c == "$" else c)) for c in all_chars]
    state_labels = [f"S{j}" for j in range(k)]
    B = pd.DataFrame(emiss, index=state_labels, columns=char_labels)

    fig, ax = plt.subplots(figsize=(max(14, len(all_chars) * 0.55), max(4, k * 1.5)))
    sns.heatmap(B, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, ax=ax,
                linewidths=0.3, linecolor="gray", cbar_kws={"shrink": 0.6},
                annot_kws={"size": 7})
    ax.set_title(f"放出確率行列 (Trigram HMM, k={k})", fontsize=12)
    ax.set_xlabel("観測文字", fontsize=9)
    ax.set_ylabel("隠れ状態", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bic_comparison(trigram_results, bigram_bics, out_path):
    """Trigram BIC と Bigram BIC 参照値の比較グラフ"""
    ks = [r["k"] for r in trigram_results]
    tri_bics = [r["bic"] for r in trigram_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ks))
    ax.bar(x, tri_bics, width=0.4, label="Trigram HMM", color="steelblue", alpha=0.85)

    # Bigram 参照値（破線）
    for k_val, bic_val in bigram_bics.items():
        if k_val in ks:
            xi = ks.index(k_val)
            ax.hlines(bic_val, xi - 0.3, xi + 0.3, colors="coral", linestyles="--",
                      linewidths=2, label=f"Bigram k={k_val}" if k_val == list(bigram_bics.keys())[0] else "")
            ax.annotate(f"Bigram\n{bic_val:.0f}", xy=(xi, bic_val),
                        xytext=(xi + 0.35, bic_val), fontsize=7, color="coral",
                        va="center")

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
    """Viterbi による状態割り当ての例"""
    state_labels = [f"S{j}" for j in range(k)]
    lines = [f"{'='*70}", f"  Viterbi 状態割り当て例（Trigram HMM, k={k}）", f"{'='*70}"]

    sample = sorted(words, key=len)
    sample_words = sample[:5] + sample[len(sample)//2 - 3:len(sample)//2 + 3] + sample[-5:]
    seen = set()

    for w in sample_words:
        if w in seen:
            continue
        seen.add(w)
        seq = [char2idx[BOS_CHAR]] + [char2idx[c] for c in w] + [char2idx[EOS_CHAR]]
        X_w = np.array(seq, dtype=np.int32)
        try:
            _, state_seq = model.viterbi(X_w)
            full_chars  = [BOS_CHAR] + list(w) + [EOS_CHAR]
            full_labels = [state_labels[s] for s in state_seq]
            char_disp  = " ".join(f"{c:>3}" for c in full_chars)
            state_disp = " ".join(f"{l:>3}" for l in full_labels)
            lines.append(f"  {w:<15}  chars: {char_disp}")
            lines.append(f"  {'':15}  state: {state_disp}")
            lines.append("")
        except Exception:
            continue
    return "\n".join(lines)


def build_training_report(results, bigram_bics, all_chars):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Trigram HMM 学習レポート",
        f"",
        f"**生成日時**: {now}",
        f"**データ**: Voynich manuscript words ({DB_PATH})",
        f"**設定**: K_RANGE={K_RANGE}, N_RESTARTS={N_RESTARTS}, N_ITER={N_ITER}",
        f"**語彙サイズ V**: {len(all_chars)}",
        f"",
        f"---",
        f"",
        f"## BIC / AIC 比較",
        f"",
        f"| k | logL | n_params | BIC (Trigram) | AIC (Trigram) | BIC (Bigram 参照) |",
        f"|---|------|----------|--------------|--------------|-----------------|",
    ]
    for r in results:
        bigram_ref = bigram_bics.get(r["k"], float("nan"))
        lines.append(
            f"| {r['k']} | {r['logL']:.1f} | {r['n_params']} "
            f"| {r['bic']:.1f} | {r['aic']:.1f} | {bigram_ref:.1f} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## 各 k の遷移行列（start_trans: S_1|S_0 の Bigram 部分）",
        f"",
    ]
    for r in results:
        k = r["k"]
        lines.append(f"### k={k}")
        lines.append(f"")
        lines.append(f"```")
        for i in range(k):
            row = "  ".join(f"{v:.3f}" for v in r["start_trans"][i])
            lines.append(f"  S{i} → [{row}]")
        lines.append(f"```")
        lines.append(f"")

    lines += [
        f"---",
        f"",
        f"## 出力ファイル",
        f"",
        f"- `hmm_model_cache/trigram_k{{k}}.npz` — 学習済みモデル",
        f"- `transition_heatmap_k{{k}}.png` — 遷移テンソル可視化",
        f"- `emission_heatmap_k{{k}}.png` — 放出確率行列",
        f"- `bic_comparison.png` — BIC 比較グラフ",
        f"",
        f"---",
        f"",
        f"_次のステップ: `trigram_role_analysis.py` で役割曖昧性の解消を検証_",
    ]
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

    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR] + raw_chars + [EOS_CHAR]
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    V         = len(all_chars)

    X_all, L_all = encode_words(all_types, char2idx)
    log(f"観測総数: {len(X_all):,}  語彙: {V}")

    # Bigram BIC 参照値を読み込む
    bigram_bics = {}
    for k in [7, 8]:
        cache_path = BIGRAM_CACHE / f"full_k{k}.npz"
        if cache_path.exists():
            d = np.load(cache_path)
            logL_bi = float(d["logL"][0])
            k_val = d["trans"].shape[0]
            n_params_bi = k_val * (k_val - 1) + k_val * (V - 1) + (k_val - 1)
            bic_bi = -2 * logL_bi + n_params_bi * np.log(len(X_all))
            bigram_bics[k] = bic_bi
            log(f"Bigram k={k}: logL={logL_bi:.1f}  BIC={bic_bi:.1f}")

    results = []

    for k in K_RANGE:
        log(f"\n{'─'*60}")
        log(f"Trigram HMM  k = {k}  (N_RESTARTS={N_RESTARTS})")
        log(f"{'─'*60}")

        model, logL = run_training(k, X_all, L_all)

        if model is None or logL == -float('inf'):
            log(f"k={k}: 学習失敗")
            continue

        bic, aic, n_params = compute_bic(logL, len(X_all), k, V)
        log(f"k={k}  logL={logL:.2f}  BIC={bic:.1f}  AIC={aic:.1f}  n_params={n_params}")

        # モデル保存
        cache_path = CACHE_DIR / f"trigram_k{k}.npz"
        save_model(model, logL, cache_path)
        log(f"モデル保存: {cache_path}")

        # 可視化
        transmat = get_transmat(model)
        emiss    = get_emission(model)
        start_trans = torch.exp(model.log_start_trans).cpu().numpy()

        plot_transition_heatmap(transmat, k, OUT_DIR / f"transition_heatmap_k{k}.png")
        plot_emission_heatmap(emiss, k, all_chars, OUT_DIR / f"emission_heatmap_k{k}.png")

        results.append({
            "k": k, "logL": logL, "bic": bic, "aic": aic,
            "n_params": n_params, "start_trans": start_trans,
        })

        # Viterbi デコード例
        ex_text = decode_examples(model, k, all_types, char2idx)
        ex_path = OUT_DIR / f"word_examples_k{k}.txt"
        ex_path.write_text(ex_text, encoding="utf-8")

    # BIC 比較グラフ
    if results:
        plot_bic_comparison(results, bigram_bics, OUT_DIR / "bic_comparison.png")

    # レポート生成
    report = build_training_report(results, bigram_bics, all_chars)
    report_path = OUT_DIR / "training_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"レポート保存: {report_path}")

    log("\n✓ 完了。出力先: " + str(OUT_DIR.resolve()))
