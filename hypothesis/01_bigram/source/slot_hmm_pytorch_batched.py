"""
slot_hmm_pytorch_batched.py
===========================
Voynich Manuscript スロット文法仮説の数理検証
【PyTorch GPU バッチ並列化版】

- N_RESTARTS 回の独立した学習試行を、GPUのバッチ次元として同時に実行します。
- これにより、GPUの演算器（コア）を最大限に活用し、実行時間を劇的に短縮します。
- ログには実行時刻を付与します。
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
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ERROR: PyTorchがインストールされていません。")
    import sys
    sys.exit(1)

# ── ログ出力ユーティリティ ──────────────────────────────────────────────
def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

# ── デバイス設定 ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"[PyTorch] Using device: {DEVICE}")
if DEVICE.type == "cuda":
    log(f"          GPU Name: {torch.cuda.get_device_name(0)}")

# ── 日本語フォント設定 ──────────────────────────────────────────────────
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
OUT_DIR      = Path("hypothesis/01_bigram/results/hmm_pt_batched")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE      = [2, 3, 4, 5]
N_RESTARTS   = 20       # 同時に実行する並列試行数（バッチサイズ）
N_ITER       = 200      # 1回の学習の最大反復数
LTR_ITER     = 50       # Left-to-Rightの後処理ループ回数
TOL          = 1e-4     # 収束判定の相対許容誤差
MIN_WORD_LEN = 2

# ── データ準備 ────────────────────────────────────────────────────────
BOS, EOS = "^", "$"

def encode_words(words, char2idx):
    seqs = []
    lengths = []
    for w in words:
        seq = [char2idx[BOS]] + [char2idx[c] for c in w] + [char2idx[EOS]]
        seqs.extend(seq)
        lengths.append(len(seq))
    X = np.array(seqs, dtype=np.int16)
    return X, lengths

# ── PyTorchによるバッチHMM実装 (Log-Domain) ───────────────────────────
class BatchedCategoricalHMM_PT:
    def __init__(self, n_restarts, n_components, n_vocab, device=DEVICE):
        self.b = n_restarts   # バッチサイズ (並列試行数)
        self.k = n_components # 状態数
        self.v = n_vocab      # 語彙数
        self.device = device
        
        # モデルパラメータ (B, K, ...)
        self.log_startprob = None # (B, K)
        self.log_transmat  = None # (B, K, K)
        self.log_emission  = None # (B, K, V)

    def _init_params(self, seed=0):
        torch.manual_seed(seed)
        
        # (B, K)
        start = torch.rand(self.b, self.k, device=self.device) + 0.1
        self.log_startprob = torch.log(start / start.sum(dim=1, keepdim=True))

        # (B, K, K)
        trans = torch.rand(self.b, self.k, self.k, device=self.device) + 0.1
        self.log_transmat = torch.log(trans / trans.sum(dim=2, keepdim=True))

        # (B, K, V)
        emiss = torch.rand(self.b, self.k, self.v, device=self.device) + 0.1
        self.log_emission = torch.log(emiss / emiss.sum(dim=2, keepdim=True))

    def _forward(self, X):
        """バッチ対応前向きアルゴリズム: log alpha (T, B, K)"""
        T = X.shape[0]
        log_alpha = torch.empty(T, self.b, self.k, device=self.device)
        
        # t=0: (B, K)
        # self.log_emission[:, :, X[0]] -> (B, K)
        log_alpha[0] = self.log_startprob + self.log_emission[:, :, X[0]]
        
        for t in range(1, T):
            # log_alpha[t-1] is (B, K)
            # unsqueeze(2) -> (B, K, 1) + (B, K, K) -> (B, K, K)
            prev = log_alpha[t-1].unsqueeze(2) + self.log_transmat
            # logsumexp over dim=1 (遷移元 i) -> (B, K)
            log_alpha[t] = torch.logsumexp(prev, dim=1) + self.log_emission[:, :, X[t]]
            
        return log_alpha

    def _backward(self, X):
        """バッチ対応後ろ向きアルゴリズム: log beta (T, B, K)"""
        T = X.shape[0]
        log_beta = torch.empty(T, self.b, self.k, device=self.device)
        
        # t=T-1
        log_beta[T-1] = 0.0
        
        for t in range(T-2, -1, -1):
            # (B, K, K) + (B, 1, K) + (B, 1, K) -> (B, K, K)
            nxt = self.log_transmat + self.log_emission[:, :, X[t+1]].unsqueeze(1) + log_beta[t+1].unsqueeze(1)
            # logsumexp over dim=2 (遷移先 j) -> (B, K)
            log_beta[t] = torch.logsumexp(nxt, dim=2)
            
        return log_beta

    def fit(self, X_pt, starts_pt, ends_pt, n_iter=N_ITER, tol=TOL, left_to_right=False):
        """Baum-Welch (バッチ並列更新)"""
        best_logL_per_batch = torch.full((self.b,), -float('inf'), device=self.device)
        T_total = X_pt.shape[0]

        for it in range(n_iter):
            log_alpha = self._forward(X_pt)
            log_beta = self._backward(X_pt)
            
            # log_gamma: (T, B, K)
            log_gamma = log_alpha + log_beta
            
            # 対数尤度: (B,)
            # 各系列の終端 ends-1 における logsumexp (Rabiner)
            # log_alpha[ends_pt - 1] -> (N_words, B, K) -> logsumexp(dim=2) -> (N_words, B) -> sum(dim=0) -> (B,)
            seq_logprobs = torch.logsumexp(log_alpha[ends_pt - 1], dim=2)
            current_logL = seq_logprobs.sum(dim=0)

            # 収束判定 (バッチ全体で改善が見られなくなったら終了、あるいは単純化のため固定ループ)
            if it > 0:
                diff = (current_logL - best_logL_per_batch).abs().max()
                if diff < tol:
                    break
            best_logL_per_batch = current_logL

            # M-step 更新のための集計 -----------------------------------------
            # 正規化項: (T, B, 1)
            norm_factor = torch.logsumexp(log_gamma, dim=2, keepdim=True)
            log_gamma_norm = log_gamma - norm_factor

            # log_xi: (T-1, B, K, K)
            # alpha(t) + A + emiss(t+1) + beta(t+1)
            log_xi = log_alpha[:-1].unsqueeze(3) + \
                     self.log_transmat.unsqueeze(0) + \
                     self.log_emission[:, :, X_pt[1:]].permute(2, 0, 1).unsqueeze(2) + \
                     log_beta[1:].unsqueeze(2)
            
            # 正規化 (T-1, B, K, K)
            log_xi = log_xi - norm_factor[:-1].unsqueeze(3)

            # 系列境界のマスク (Ends-1)
            invalid_idx = ends_pt[:-1] - 1
            log_xi[invalid_idx] = -float('inf')

            # 1. Start probability 更新 (B, K)
            # 各単語の開始時点 starts_pt における log_gamma_norm の平均
            new_log_start = torch.logsumexp(log_gamma_norm[starts_pt], dim=0) - torch.log(torch.tensor(float(len(starts_pt)), device=self.device))
            self.log_startprob = new_log_start - torch.logsumexp(new_log_start, dim=1, keepdim=True)

            # 2. Transition matrix 更新 (B, K, K)
            new_log_trans = torch.logsumexp(log_xi, dim=0) # sum over T-1
            if left_to_right:
                mask = torch.triu(torch.ones(self.k, self.k, device=self.device)).bool()
                new_log_trans[:, ~mask] = -float('inf')
            
            self.log_transmat = new_log_trans - torch.logsumexp(new_log_trans, dim=2, keepdim=True)

            # 3. Emission matrix 更新 (B, K, V)
            # (T, B, K) -> (B, K, T)
            exp_gamma = torch.exp(log_gamma_norm).permute(1, 2, 0)
            
            # X_onehot: (T, V) -> 各Bに共通
            X_onehot = torch.zeros(T_total, self.v, device=self.device)
            X_onehot.scatter_(1, X_pt.unsqueeze(1), 1.0)

            # (B, K, T) @ (T, V) -> (B, K, V)
            new_emiss_linear = torch.matmul(exp_gamma, X_onehot)
            # かなり小さい値になるので対数に戻して正規化
            # 安定化のため極小値を足す
            new_log_emiss = torch.log(new_emiss_linear + 1e-30)
            self.log_emission = new_log_emiss - torch.logsumexp(new_log_emiss, dim=2, keepdim=True)

        return best_logL_per_batch

    def get_best_model_params(self, log_likelihoods):
        """尤度が最大のインデックスを取得し、そのパラメータを返す"""
        best_idx = torch.argmax(log_likelihoods).item()
        return {
            "startprob": torch.exp(self.log_startprob[best_idx]).cpu().numpy(),
            "transmat": torch.exp(self.log_transmat[best_idx]).cpu().numpy(),
            "emission": torch.exp(self.log_emission[best_idx]).cpu().numpy(),
            "logL": log_likelihoods[best_idx].item(),
            "best_idx": best_idx,
            "best_log_startprob": self.log_startprob[best_idx],
            "best_log_transmat": self.log_transmat[best_idx],
            "best_log_emission": self.log_emission[best_idx]
        }

# ── Viterbi (単体モデル用) ──────────────────────────────────────────
def viterbi_pt(log_start, log_trans, log_emiss, X_np, device):
    X = torch.tensor(X_np, dtype=torch.long, device=device)
    T = X.shape[0]
    K = log_start.shape[0]
    
    log_delta = torch.empty(T, K, device=device)
    psi = torch.empty(T, K, dtype=torch.long, device=device)
    
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

# ── 評価ユーティリティ ────────────────────────────────────────────────
def compute_bic(log_likelihood, X_len, k, V):
    N = X_len
    n_params = k * (k - 1) + k * (V - 1) + (k - 1)
    bic = -2 * log_likelihood + n_params * np.log(N)
    aic = -2 * log_likelihood + 2 * n_params
    return bic, aic

def _state_label(i, k):
    labels = {
        2: ["Prefix/Core", "Suffix"],
        3: ["Prefix", "Core", "Suffix"],
        4: ["Prefix", "Core-1", "Core-2", "Suffix"],
        5: ["Prefix", "Core-1", "Core-2", "Core-3", "Suffix"],
    }
    return labels.get(k, [f"S{j}" for j in range(k)])[i]

def plot_transition(transmat, k, topology, out_path):
    state_labels = [_state_label(i, k) for i in range(k)]
    A = pd.DataFrame(transmat, index=state_labels, columns=state_labels)
    fig, ax = plt.subplots(figsize=(max(5, k * 1.5), max(4, k * 1.3)))
    sns.heatmap(A, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor="gray", cbar_kws={"shrink": 0.8})
    ax.set_title(f"遷移確率行列（{topology}, k={k}）", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_emission(emissionprob, k, topology, all_chars, out_path):
    state_labels = [_state_label(i, k) for i in range(k)]
    char_labels  = [("BOS" if c == "^" else ("EOS" if c == "$" else c)) for c in all_chars]
    B = pd.DataFrame(emissionprob, index=state_labels, columns=char_labels)
    fig, ax = plt.subplots(figsize=(max(14, len(all_chars) * 0.55), max(4, k * 1.5)))
    sns.heatmap(B, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, ax=ax,
                linewidths=0.3, linecolor="gray", cbar_kws={"shrink": 0.6})
    ax.set_title(f"放射確率行列（{topology}, k={k}）", fontsize=13)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# メイン処理
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("Starting slot_hmm_pytorch_batched.py...")
    
    log(f"Loading data from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''", conn)
    conn.close()

    all_types = sorted(set(df["word"].tolist()))
    all_types = [w for w in all_types if len(w) >= MIN_WORD_LEN]
    log(f"  Unique words (types): {len(all_types):,}")

    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS] + raw_chars + [EOS]
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    V         = len(all_chars)

    X_all, L_all = encode_words(all_types, char2idx)
    log(f"  Total observation symbols: {len(X_all):,}")
    log(f"  Vocab size: {V}")

    # PyTorchテンソル化
    X_pt = torch.tensor(X_all, dtype=torch.long, device=DEVICE)
    lengths_pt = torch.tensor(L_all, device=DEVICE)
    ends_pt = torch.cumsum(lengths_pt, dim=0)
    starts_pt = torch.cat([torch.tensor([0], device=DEVICE), ends_pt[:-1]])

    report_lines = []
    bic_results  = []
    best_results_k3 = None

    for k in K_RANGE:
        log(f"{'─'*60}")
        log(f"  k = {k}")
        log(f"{'─'*60}")

        # ── Full HMM (バッチ並列)
        log(f"  [Full] GPU Batch学習中... (BATCH_SIZE={N_RESTARTS}, ITER={N_ITER})")
        model_full = BatchedCategoricalHMM_PT(N_RESTARTS, k, V, DEVICE)
        model_full._init_params(seed=42)
        log_likelihoods = model_full.fit(X_pt, starts_pt, ends_pt, n_iter=N_ITER, left_to_right=False)
        
        info = model_full.get_best_model_params(log_likelihoods)
        bic_full, aic_full = compute_bic(info["logL"], len(X_all), k, V)
        
        plot_transition(info["transmat"], k, "Full_Batch", OUT_DIR / f"transition_full_k{k}.png")
        plot_emission(info["emission"], k, "Full_Batch", all_chars, OUT_DIR / f"emission_full_k{k}.png")

        if k == 3:
            best_results_k3 = info

        # ── Left-to-Right HMM (バッチ並列)
        log(f"  [L-to-R] GPU Batch学習中... (BATCH_SIZE={N_RESTARTS}, ITER={LTR_ITER})")
        model_ltr = BatchedCategoricalHMM_PT(N_RESTARTS, k, V, DEVICE)
        model_ltr._init_params(seed=77)
        log_likelihoods_ltr = model_ltr.fit(X_pt, starts_pt, ends_pt, n_iter=LTR_ITER, left_to_right=True)
        
        info_ltr = model_ltr.get_best_model_params(log_likelihoods_ltr)
        bic_ltr, aic_ltr = compute_bic(info_ltr["logL"], len(X_all), k, V)
        
        plot_transition(info_ltr["transmat"], k, "LTR_Batch", OUT_DIR / f"transition_ltr_k{k}.png")
        plot_emission(info_ltr["emission"], k, "LTR_Batch", all_chars, OUT_DIR / f"emission_ltr_k{k}.png")

        bic_results.append({
            "k": k,
            "bic_full": bic_full, "ll_full": info["logL"],
            "bic_ltr":  bic_ltr,  "ll_ltr":  info_ltr["logL"]
        })

        # レポート記述
        section = [
            f"{'='*70}", f"  k = {k} [GPU Batched Parallel]", f"{'='*70}",
            f"  [Full HMM]  logL: {info['logL']:.2f}, BIC: {bic_full:.2f}",
            f"  [L-to-R HMM] logL: {info_ltr['logL']:.2f}, BIC: {bic_ltr:.2f}",
            f"", f"  Best Transition Matrix (Full):"
        ]
        labels = [_state_label(i, k) for i in range(k)]
        for i, row in enumerate(info["transmat"]):
            vals = "  ".join(f"{v:.3f}" for v in row)
            section.append(f"    {labels[i]:<12} -> [{vals}]")
        report_lines.append("\n".join(section))
        log(f"  -> Best logL: {info['logL']:.2f}")

    # ── まとめ
    log("Finalizing results...")
    
    # BICグラフ
    ks = [r["k"] for r in bic_results]
    bics_f = [r["bic_full"] for r in bic_results]
    bics_l = [r["bic_ltr"] for r in bic_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ks))
    ax.bar(x - 0.2, bics_f, 0.4, label="Full")
    ax.bar(x + 0.2, bics_l, 0.4, label="LTR")
    ax.set_xticks(x); ax.set_xticklabels(ks); ax.set_ylabel("BIC"); ax.legend()
    plt.savefig(OUT_DIR / "bic_comparison.png"); plt.close()

    # Viterbi例
    if best_results_k3:
        log("Generating Viterbi examples for k=3...")
        text = decode_examples(best_results_k3["best_log_startprob"], 
                               best_results_k3["best_log_transmat"], 
                               best_results_k3["best_log_emission"], 
                               all_types, char2idx)
        # 再定義(単体Viterbi用)
        def decode_local(l_start, l_trans, l_emiss, words, c2i):
            lines = [f"{'='*70}", f"  Viterbi Example (k=3, Full)", f"{'='*70}"]
            sample = sorted(words, key=len)
            sample_words = sample[:5] + sample[len(sample)//2-5:len(sample)//2+5] + sample[-5:]
            for w in sample_words:
                seq = [c2i[BOS]] + [c2i[c] for c in w] + [c2i[EOS]]
                _, path = viterbi_pt(l_start, l_trans, l_emiss, np.array(seq), DEVICE)
                lbls = [_state_label(p, 3) for p in path]
                lines.append(f"  {w:<15} : {'-'.join(l[:2] for l in lbls)}")
            return "\n".join(lines)
        
        with open(OUT_DIR / "word_examples.txt", "w", encoding="utf-8") as f:
            f.write(decode_local(best_results_k3["best_log_startprob"], 
                                 best_results_k3["best_log_transmat"], 
                                 best_results_k3["best_log_emission"], 
                                 all_types, char2idx))

    with open(OUT_DIR / "hmm_report.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(report_lines))

    log(f"All processes completed. Output: {OUT_DIR.resolve()}")
