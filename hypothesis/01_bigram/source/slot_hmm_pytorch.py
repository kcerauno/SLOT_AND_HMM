"""
slot_hmm_pytorch.py
===================
Voynich Manuscript スロット文法仮説の数理検証
【PyTorch GPU 高速化版】

- PyTorchのテンソル演算を用いてBaum-WelchアルゴリズムとViterbiアルゴリズムを実装します。
- 対数領域(Log-domain)で計算を行うため、長い系列でもアンダーフローしません。
- 観測系列全体をGPUテンソルに載せて一括計算するため、hmmlearnより劇的に高速です。
- 実行環境（GTX 1650等）にPyTorchがインストールされている必要があります。
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
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("WARNING: PyTorchがインストールされていません。このスクリプトはPyTorch環境で実行してください。")
    import sys
    sys.exit(1)

# ── デバイス設定 ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PyTorch] Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"          GPU Name: {torch.cuda.get_device_name(0)}")

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
OUT_DIR      = Path("hypothesis/01_bigram/results/hmm_pt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE      = [2, 3, 4, 5]
N_RESTARTS   = 20       # ランダムシードを変えての多重スタート回数
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

# ── PyTorchによるHMM実装 (Log-Domain) ─────────────────────────────────
# 参考: Rabiner (1989) 前向き・後ろ向きアルゴリズムの対数領域拡張
class CategoricalHMM_PT:
    def __init__(self, n_components, n_vocab, device=DEVICE):
        self.k = n_components
        self.v = n_vocab
        self.device = device
        
        # モデルパラメータ (Log空間)
        self.log_startprob = None
        self.log_transmat  = None
        self.log_emission  = None

    def _init_params(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        # Dirichlet分布のような偏りを持たせるため、一様乱数を取り込んで正規化
        start = torch.rand(self.k, device=self.device) + 0.1
        self.log_startprob = torch.log(start / start.sum())

        trans = torch.rand(self.k, self.k, device=self.device) + 0.1
        self.log_transmat = torch.log(trans / trans.sum(dim=1, keepdim=True))

        emiss = torch.rand(self.k, self.v, device=self.device) + 0.1
        self.log_emission = torch.log(emiss / emiss.sum(dim=1, keepdim=True))

    def _forward(self, X):
        """前向きアルゴリズム: log alpha計算"""
        T = X.shape[0]
        log_alpha = torch.empty(T, self.k, device=self.device)
        
        # t=0
        log_alpha[0] = self.log_startprob + self.log_emission[:, X[0]]
        
        for t in range(1, T):
            # log \alpha_{t}(j) = log \sum_i \exp(log \alpha_{t-1}(i) + log A_{ij}) + log B_j(O_t)
            prev = log_alpha[t-1].unsqueeze(1) + self.log_transmat
            log_alpha[t] = torch.logsumexp(prev, dim=0) + self.log_emission[:, X[t]]
            
        return log_alpha

    def _backward(self, X):
        """後ろ向きアルゴリズム: log beta計算"""
        T = X.shape[0]
        log_beta = torch.empty(T, self.k, device=self.device)
        
        # t=T-1
        log_beta[T-1] = 0.0 # log(1) = 0
        
        for t in range(T-2, -1, -1):
            # log \beta_t(i) = log \sum_j \exp(log A_{ij} + log B_j(O_{t+1}) + log \beta_{t+1}(j))
            nxt = self.log_transmat + self.log_emission[:, X[t+1]] + log_beta[t+1]
            log_beta[t] = torch.logsumexp(nxt, dim=1)
            
        return log_beta

    def fit(self, X_np, lengths, n_iter=N_ITER, tol=TOL, seed=None, left_to_right=False):
        """Baum-Welch Algorithm（複数系列をつなげたXを入力）"""
        self._init_params(seed)
        X = torch.tensor(X_np, dtype=torch.long, device=self.device)
        
        T_total = X.shape[0]
        
        # 系列の開始・終了位置のインデックスを作成
        lengths_t = torch.tensor(lengths, device=self.device)
        ends = torch.cumsum(lengths_t, dim=0)
        starts = torch.cat([torch.tensor([0], device=self.device), ends[:-1]])
        
        best_logL = -float('inf')
        
        for it in range(n_iter):
            # E-step
            log_alpha = self._forward(X)
            log_beta = self._backward(X)
            
            # 系列間の境界をまたぐ遷移確率をマスクするため、
            # \xi_t(i, j) = P(q_t=i, q_{t+1}=j | O) を計算
            # ただし、系列の終端 (t = end - 1) では \xi は計算せず集計もしない
            
            # log_gamma: P(q_t=i | O)
            log_gamma = log_alpha + log_beta
            # 尤度計算: 各系列の開始時点の log_alpha の logsumexp の合計
            seq_logprobs = torch.logsumexp(log_alpha[starts] + log_beta[starts], dim=1)
            logL = seq_logprobs.sum().item()
            
            if np.isnan(logL):
                return -float('inf')

            # 収束判定
            if it > 0 and (logL - best_logL) < tol:
                break
            best_logL = logL
            
            # M-step 更新のための集計
            valid_gamma = log_gamma.clone()
            
            # log \xi 計算: サイズ (T-1, K, K)
            # log \xi_t = log \alpha_t(i) + log A_{ij} + log B_j(O_{t+1}) + log \beta_{t+1}(j) - log P(O)
            norm_factor = torch.logsumexp(log_gamma, dim=1)
            log_gamma_norm = log_gamma - norm_factor.unsqueeze(1)
            
            log_xi = log_alpha[:-1].unsqueeze(2) + \
                     self.log_transmat.unsqueeze(0) + \
                     self.log_emission[:, X[1:]].T.unsqueeze(1) + \
                     log_beta[1:].unsqueeze(1)
            
            # 各時点ごとの正規化定数 (t時点での P(O_t..))
            log_xi_norm = log_xi - norm_factor[:-1].unsqueeze(1).unsqueeze(2)

            # バッチ間でまたがる遷移（t = seq_end - 1）を無効化（-inf）
            invalid_transitions = ends[:-1] - 1
            log_xi_norm[invalid_transitions] = -float('inf')
            
            # 1. Start probability 更新
            new_log_start = torch.logsumexp(log_gamma_norm[starts], dim=0) - torch.log(torch.tensor(len(starts), dtype=torch.float, device=self.device))
            self.log_startprob = new_log_start
            
            # 2. Transition matrix 更新
            new_log_trans = torch.logsumexp(log_xi_norm, dim=0)
            new_log_trans = new_log_trans - torch.logsumexp(new_log_trans, dim=1, keepdim=True)
            
            if left_to_right:
                # 下三角を 0 に制約（log領域では -inf）
                mask = torch.triu(torch.ones(self.k, self.k, device=self.device)).bool()
                new_log_trans[~mask] = -float('inf')
                # 再正規化
                new_log_trans = new_log_trans - torch.logsumexp(new_log_trans, dim=1, keepdim=True)
                
            self.log_transmat = new_log_trans
            
            # 3. Emission matrix 更新
            # M_jk = \sum_{t \\text{s.t.} O_t=k} \gamma_t(j)
            # 高速化のため、X_t のワンホットテンソルと行列積をとる
            # X_onehot: (T, V)
            X_onehot = torch.zeros(T_total, self.v, device=self.device)
            X_onehot.scatter_(1, X.unsqueeze(1), 1.0)
            
            # 確率領域に戻して乗算（ここはメモリと精度に注意）
            # log ( \sum_t exp(log_gamma_norm_t) * X_onehot_t )
            # 効率的な実装:
            max_log_gamma = log_gamma_norm.max(dim=0, keepdim=True).values
            exp_gamma = torch.exp(log_gamma_norm - max_log_gamma)
            
            new_emiss = torch.matmul(exp_gamma.T, X_onehot)  # (K, T) x (T, V) -> (K, V)
            #log_new_emiss = torch.log(new_emiss) + max_log_gamma.T.squeeze()
            log_new_emiss = torch.log(new_emiss) + max_log_gamma.T
            
            log_new_emiss = log_new_emiss - torch.logsumexp(log_new_emiss, dim=1, keepdim=True)
            self.log_emission = log_new_emiss

        return best_logL

    def score(self, X_np, lengths):
        X = torch.tensor(X_np, dtype=torch.long, device=self.device)
        log_alpha = self._forward(X)
        
        lengths_t = torch.tensor(lengths, device=self.device)
        ends = torch.cumsum(lengths_t, dim=0)
        starts = torch.cat([torch.tensor([0], device=self.device), ends[:-1]])
        
        # log_beta(t=starts) を足す必要はない。尤度は純粋にアルファパスの和。
        # Rabiner 式: P(O) = \sum_i \alpha_T(i) => ただし各系列の終わり ends-1 で取る
        seq_logprobs = torch.logsumexp(log_alpha[ends - 1], dim=1)
        return seq_logprobs.sum().item()

    def viterbi(self, X_np):
        """Viterbiアルゴリズム (単一系列用)"""
        X = torch.tensor(X_np, dtype=torch.long, device=self.device)
        T = X.shape[0]
        
        log_delta = torch.empty(T, self.k, device=self.device)
        psi = torch.empty(T, self.k, dtype=torch.long, device=self.device)
        
        log_delta[0] = self.log_startprob + self.log_emission[:, X[0]]
        psi[0] = 0
        
        for t in range(1, T):
            # log_delta_{t}(j) = max_i (log_delta_{t-1}(i) + log A_{ij}) + log B_j(O_t)
            vals = log_delta[t-1].unsqueeze(1) + self.log_transmat
            max_vals, argmax_vals = torch.max(vals, dim=0)
            
            log_delta[t] = max_vals + self.log_emission[:, X[t]]
            psi[t] = argmax_vals

        path = torch.empty(T, dtype=torch.long, device=self.device)
        path[T-1] = torch.argmax(log_delta[T-1])
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
            
        return log_delta[T-1].max().item(), path.cpu().numpy()

# ── 学習・評価ループ（GPU版） ──────────────────────────────────────────────
def run_gpu_training(k, X_all, L_all, topology="Full"):
    best_model = None
    best_logL = -float('inf')
    is_ltr = (topology == "L-to-R")
    
    for seed in range(N_RESTARTS):
        model = CategoricalHMM_PT(n_components=k, n_vocab=V, device=DEVICE)
        # L-to-Rの場合は N_ITER の回数分制約付きで回し、さらに LTR_ITER 分追加で回す
        n_iters = LTR_ITER if is_ltr else N_ITER
        
        logL = model.fit(X_all, L_all, n_iter=n_iters, seed=seed, left_to_right=is_ltr)
        
        # モデルの尤度を再計算して一番良いものを保存
        if logL > best_logL:
            best_logL = logL
            best_model = model
            
    return best_model, best_logL

# ── その他のユーティリティ（numpy への変換） ──────────────────────────────
def get_transmat(model):
    return torch.exp(model.log_transmat).cpu().numpy()

def get_emission(model):
    return torch.exp(model.log_emission).cpu().numpy()

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
    ax.set_xlabel("遷移先状態", fontsize=10)
    ax.set_ylabel("遷移元状態", fontsize=10)
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
    ax.set_xlabel("観測文字", fontsize=10)
    ax.set_ylabel("隠れ状態", fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_bic(results, out_path):
    ks = [r["k"] for r in results]
    bics_full = [r["bic_full"] for r in results]
    bics_ltr  = [r["bic_ltr"]  for r in results]

    x = np.arange(len(ks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, bics_full, width, label="Full HMM", color="steelblue", alpha=0.85)
    ax.bar(x + width/2, bics_ltr,  width, label="Left-to-Right HMM", color="coral", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_xlabel("状態数 k")
    ax.set_ylabel("BIC（小さいほど良い）")
    ax.set_title("HMM モデル選択: BIC 比較 [PyTorch]")
    ax.legend()

    min_full = min(zip(bics_full, ks), key=lambda t: t[0])
    min_ltr  = min(zip(bics_ltr, ks),  key=lambda t: t[0])
    ax.annotate("▼", xy=(min_full[1] - K_RANGE[0] - width/2, min_full[0]),
                fontsize=14, ha="center", va="bottom", color="steelblue")
    ax.annotate("▼", xy=(min_ltr[1] - K_RANGE[0] + width/2, min_ltr[0]),
                fontsize=14, ha="center", va="bottom", color="coral")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def decode_examples(model, k, words, char2idx):
    state_labels = [_state_label(i, k) for i in range(k)]
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  Viterbi 状態割り当て例（k={k}, Full HMM）")
    lines.append(f"{'='*70}")

    sample = sorted(words, key=len)
    sample_words = sample[:10] + sample[len(sample)//2-5:len(sample)//2+5] + sample[-10:]
    seen = set()

    for w in sample_words:
        if w in seen:
            continue
        seen.add(w)
        seq = [char2idx[BOS]] + [char2idx[c] for c in w] + [char2idx[EOS]]
        X_w = np.array(seq, dtype=np.int16)
        try:
            _, state_seq = model.viterbi(X_w)
            full_chars  = [BOS] + list(w) + [EOS]
            full_labels = [state_labels[s] for s in state_seq]
            char_disp  = " ".join(f"{c:>3}" for c in full_chars)
            state_disp = " ".join(f"{l[:3]:>3}" for l in full_labels)
            lines.append(f"  {w:<15}  chars: {char_disp}")
            lines.append(f"  {'':15}  state: {state_disp}")
            lines.append("")
        except Exception as e:
            continue
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# メインプロセス
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading data from DB...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
        conn
    )
    conn.close()

    all_types = sorted(set(df["word"].tolist()))
    all_types = [w for w in all_types if len(w) >= MIN_WORD_LEN]
    print(f"  ユニーク単語数（types）: {len(all_types):,}")

    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS] + raw_chars + [EOS]
    char2idx  = {c: i for i, c in enumerate(all_chars)}
    V         = len(all_chars)

    X_all, L_all = encode_words(all_types, char2idx)
    print(f"  観測総数: {len(X_all):,}")
    print(f"  語彙サイズ: {V}")

    report_lines = []
    bic_results  = []
    best_k3_full = None

    for k in K_RANGE:
        print(f"\n{'─'*60}")
        print(f"  k = {k}")
        print(f"{'─'*60}")

        # ── Full HMM
        print(f"  [Full] GPU学習中... (N_RESTARTS={N_RESTARTS})")
        model_full, ll_full = run_gpu_training(k, X_all, L_all, topology="Full")

        if model_full is None:
            print(f"  [Full] k={k}: 学習失敗")
            continue

        bic_full, aic_full = compute_bic(ll_full, len(X_all), k, V)
        trans_full = get_transmat(model_full)
        emiss_full = get_emission(model_full)
        
        plot_transition(trans_full, k, "Full_PT", OUT_DIR / f"transition_full_k{k}.png")
        plot_emission(emiss_full, k, "Full_PT", all_chars, OUT_DIR / f"emission_full_k{k}.png")

        if k == 3:
            best_k3_full = model_full

        # ── Left-to-Right HMM
        print(f"  [L-to-R] GPU学習中... (N_RESTARTS={N_RESTARTS})")
        model_ltr, ll_ltr = run_gpu_training(k, X_all, L_all, topology="L-to-R")

        if model_ltr is None:
            print(f"  [L-to-R] k={k}: 学習失敗")
            bic_ltr, aic_ltr, ll_ltr = np.nan, np.nan, np.nan
        else:
            bic_ltr, aic_ltr = compute_bic(ll_ltr, len(X_all), k, V)
            trans_ltr = get_transmat(model_ltr)
            emiss_ltr = get_emission(model_ltr)
            
            plot_transition(trans_ltr, k, "L-to-R_PT", OUT_DIR / f"transition_ltr_k{k}.png")
            plot_emission(emiss_ltr, k, "L-to-R_PT", all_chars, OUT_DIR / f"emission_ltr_k{k}.png")

        bic_results.append({
            "k": k,
            "bic_full": bic_full, "aic_full": aic_full, "ll_full": ll_full,
            "bic_ltr":  bic_ltr,  "aic_ltr":  aic_ltr,  "ll_ltr":  ll_ltr,
        })

        # レポート生成
        section = [
            f"{'='*70}",
            f"  k = {k} [PyTorch GPU]",
            f"{'='*70}",
            f"  [Full HMM]",
            f"    対数尤度 : {ll_full:.2f}",
            f"    BIC      : {bic_full:.2f}",
            f"    AIC      : {aic_full:.2f}",
            f"",
            f"  遷移行列（Full）:",
        ]
        state_labels = [_state_label(i, k) for i in range(k)]
        for i, row in enumerate(trans_full):
            vals = "  ".join(f"{v:.3f}" for v in row)
            section.append(f"    {state_labels[i]:<12} → [{vals}]")

        if model_ltr:
            section += [
                f"",
                f"  [Left-to-Right HMM]",
                f"    対数尤度 : {ll_ltr:.2f}",
                f"    BIC      : {bic_ltr:.2f}",
                f"    AIC      : {aic_ltr:.2f}",
                f"",
                f"  遷移行列（L-to-R）:",
            ]
            for i, row in enumerate(trans_ltr):
                vals = "  ".join(f"{v:.3f}" for v in row)
                section.append(f"    {state_labels[i]:<12} → [{vals}]")

        report_lines.append("\n".join(section))
        print(f"  -> BIC Full={bic_full:.1f}  L-to-R={bic_ltr:.1f}")

    # ── まとめの出力
    print("\n[生成ファイル出力]")
    plot_bic(bic_results, OUT_DIR / "bic_comparison.png")

    if best_k3_full:
        example_text = decode_examples(best_k3_full, k=3, words=all_types, char2idx=char2idx)
        ex_path = OUT_DIR / "word_examples.txt"
        with open(ex_path, "w", encoding="utf-8") as f:
            f.write(example_text)
        print(f"  Saved: {ex_path.name}")

    report_path = OUT_DIR / "hmm_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report_lines))
    print(f"  Saved: {report_path.name}")

    print("\n" + "="*60)
    print("  BIC サマリー [PyTorch GPU]")
    print("="*60)
    print(f"  {'k':>4}  {'BIC(Full)':>14}  {'BIC(L-to-R)':>14}")
    print("  " + "-"*40)
    for r in bic_results:
        print(f"  {r['k']:>4}  {r['bic_full']:>14.1f}  {r['bic_ltr']:>14.1f}")
    
    if bic_results:
        best_k_full = min(bic_results, key=lambda r: r["bic_full"])
        print(f"\n  最適モデル（Full）: k={best_k_full['k']}  BIC={best_k_full['bic_full']:.1f}")

    print("\n✓ 完了。出力先:", OUT_DIR.resolve())
