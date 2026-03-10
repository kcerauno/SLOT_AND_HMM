# 実装計画: Trigram HMM (hypothesis/03_trigram)

**作成日**: 2026-03-04
**背景**: `idea/next_architecture_candidates.md` §2.1 Trigram HMM

---

## Context

`idea/next_architecture_candidates.md` §2.1 の提案に基づき、Bigram HMM で S4 (Focus State) に
「基末尾」と「基先頭」の役割が混在している問題（役割曖昧性）を Trigram HMM で解消できるか検証する。

**研究仮説**: 遷移確率を P(s_t | s_{t-1}) → P(s_t | s_{t-1}, s_{t-2}) に拡張することで、
2ステップ前の文脈の違いから「基末尾の S4」と「基先頭の S4」を区別して学習できる。

**判定基準**: Viterbi デコード結果で、Focus State が B-end 位置と B-start 位置で
異なる s_{t-2} 分布を持つか（= 文脈が有意に分かれるか）。

---

## ディレクトリ構造

```
hypothesis/03_trigram/
├── source/
│   ├── trigram_hmm_pytorch.py      # Trigram HMM 学習スクリプト
│   └── trigram_role_analysis.py    # 役割曖昧性分析スクリプト
└── results/
    ├── hmm_model_cache/
    │   ├── trigram_k5.npz
    │   ├── trigram_k6.npz
    │   ├── trigram_k7.npz
    │   └── trigram_k8.npz
    ├── transition_heatmap_k{k}.png  # 遷移テンソルの可視化 (k 枚のサブプロット)
    ├── emission_heatmap_k{k}.png
    ├── bic_comparison.png           # Trigram BIC + Bigram BIC 参照線
    ├── role_context_k{k}.png        # Focus State の s_{t-2} 分布比較 (B-end vs B-start)
    ├── focus_state_trigram_k{k}.png # Focus State 前文脈ヒートマップ
    ├── training_report.md           # 学習メトリクスレポート
    └── role_analysis_report.md      # 役割曖昧性解消の解釈レポート
```

---

## 重要ファイル・参照

| 参照先 | パス |
|--------|------|
| Voynich DB | `data/voynich.db` |
| Bigram モデルキャッシュ | `hypothesis/01_bigram/results/hmm_model_cache/full_k{k}.npz` |
| Bigram モデル形式 | `start:(k,)`, `trans:(k,k)`, `emiss:(k,32)`, `logL:(1,)` |
| Bigram HMM 実装 | `hypothesis/01_bigram/source/slot_hmm_pytorch.py` |
| V8 文法・分割関数 | `hypothesis/02_compound_hmm/source/compound_boundary_analysis.py` |
| 分析スクリプトパターン | `hypothesis/02_compound_hmm/source/base_count_analysis.py` |

---

## ファイル 1: `trigram_hmm_pytorch.py`

### クラス設計: `TrigramHMM_PT`

Bigram の `CategoricalHMM_PT` を Trigram に拡張。

**パラメータ構成:**
```
log_start      : (k,)      — P(S_0=j)
log_start_trans: (k, k)    — P(S_1=j | S_0=i)  ← 第1ステップのみ使用
log_transmat   : (k, k, k) — P(S_t=l | S_{t-2}=i, S_{t-1}=j)  ← t≥2 で使用
log_emission   : (k, v)    — 同 Bigram
```

**BIC パラメータ数:**
```
n_params = (k-1)           # start
         + k*(k-1)         # start_trans (k行、各行 k-1 自由度)
         + k*k*(k-1)       # transmat (k² 行、各行 k-1 自由度)
         + k*(V-1)         # emission
```

**Forward アルゴリズム:**
```
t=0: log_alpha_1d[j] = log_start[j] + log_emiss[j, O_0]         # shape (k,)
t=1: log_alpha[i, j] = log_alpha_1d[i] + log_start_trans[i,j]
                      + log_emiss[j, O_1]                         # shape (k, k)
t≥2: log_alpha_new[j, l] = logsumexp_i(log_alpha[i, j] + log_transmat[i, j, l])
                           + log_emiss[l, O_t]                    # shape (k, k)
```

**Backward アルゴリズム:**
```
t=T-1: log_beta[i, j] = 0                                        # shape (k, k)
t<T-1: log_beta[i, j] = logsumexp_l(log_transmat[i, j, l]
                                    + log_emiss[l, O_{t+1}]
                                    + log_beta_next[j, l])        # shape (k, k)
```

**Viterbi アルゴリズム:**
```
t=0: log_delta_1d[j] = log_start[j] + log_emiss[j, O_0]
t=1: log_delta[i, j] = log_delta_1d[i] + log_start_trans[i,j] + log_emiss[j, O_1]
t≥2: log_delta[j, l] = max_i(log_delta[i, j] + log_transmat[i, j, l]) + log_emiss[l, O_t]
     psi[t][j, l]    = argmax_i
バックトラック: (j, l) ペアから状態列を復元
```

**Baum-Welch M-step (シーケンス境界マスク):**
- ξ_t(i,j,l) = α_t(i,j) + A(i,j,l) + B(l, O_{t+1}) + β_{t+1}(j,l) - Z
- 系列の終端位置（ends[:-1]-1）はシーケンス境界のため ξ をマスク
- t=0 は log_start の更新のみ（log_transmat の更新に使わない）
- t=1 は log_start_trans の更新のみ（log_transmat の更新に使わない）

**モデル保存形式 (.npz):**
```
start       : (k,)
start_trans : (k, k)
trans       : (k, k, k)
emiss       : (k, v)
logL        : (1,)
```

**学習設定:**
```python
K_RANGE    = [5, 6, 7, 8]  # k=7,8 は Bigram との直接比較、k=5,6 は BIC 景観把握
N_RESTARTS = 20
N_ITER     = 200
TOL        = 1e-4
```

**出力ファイル:**
- `results/hmm_model_cache/trigram_k{k}.npz`
- `results/transition_heatmap_k{k}.png` — log_transmat の (k,k,k) テンソルを k 枚の (k×k) サブプロットとして表示。各サブプロットのタイトル = "s_{t-2} = S{i}"
- `results/emission_heatmap_k{k}.png` — Bigram と同フォーマット
- `results/bic_comparison.png` — Trigram BIC (棒グラフ) + Bigram BIC 参照値 (破線)
- `results/training_report.md` — BIC表、対数尤度、パラメータ数

---

## ファイル 2: `trigram_role_analysis.py`

### 目的

Trigram HMM が役割曖昧性（Focus State の B-end/B-start 混在）を解消できているか定量的に検証。

### 主要な分析ステップ

**Step 1: Focus State 同定**
- 全単語に Viterbi デコードを適用
- V8 の `find_v8_splits_first()` で複合語境界ラベルを付与 (B-end / B-start / S-head / S-mid)
- 各 Trigram 状態の B-end 占有率を計算 → 最高の状態を Focus State とする

**Step 2: 前文脈分布の比較**
- Focus State が登場した全時点について、その (s_{t-2}, s_{t-1}) を記録
- B-end 位置グループ vs B-start 位置グループで (s_{t-2}) の分布を比較
- Fisher 正確検定: 各 s_{t-2} ∈ {S0..S_{k-1}} について 2×2 分割表

**Step 3: Bigram との比較**
- Bigram k=7,8 の同じ Viterbi パスで同様の集計を行い、前文脈分布の区別能を比較
- 定量指標: 「B-end での支配的 s_{t-2}」と「B-start での支配的 s_{t-2}」の重複度

**Step 4: 遷移構造の解釈**
- log_transmat[:, :, Focus_State] の可視化: "どの文脈 (s_{t-2}, s_{t-1}) から Focus State に遷移するか"
- B-end パターン（基内部状態 → Focus State）と B-start パターン（Focus State → Focus State）の確率差

### 出力ファイル

- `results/role_context_k{k}.png` — (s_{t-2}) 分布の棒グラフ (B-end vs B-start 比較)
- `results/focus_state_trigram_k{k}.png` — Focus State への遷移確率 A[:, :, Focus] のヒートマップ (s_{t-2} × s_{t-1} の 2D)
- `results/role_analysis_report.md` — Fisher 検定結果、解釈、Bigram との比較、結論

---

## 実行手順

```bash
# 作業ディレクトリ: /home/practi/work_voy
source .venv/bin/activate

# Step 1: 学習 (約20〜40分、GPU依存)
python hypothesis/03_trigram/source/trigram_hmm_pytorch.py

# Step 2: 役割分析
python hypothesis/03_trigram/source/trigram_role_analysis.py
```

---

## 検証方法

1. **モデルキャッシュ**: `results/hmm_model_cache/trigram_k7.npz` が生成されること
2. **BIC 表**: `training_report.md` に k=5,6,7,8 のBIC値が揃っていること
3. **役割曖昧性テスト**: `role_analysis_report.md` に Fisher 検定 p値が記載されていること
   - 有意 (p < 0.05): Trigram が役割を区別 → 「局所文脈で十分」結論
   - 非有意: HSMM (Step 2) へ進む根拠
4. **グラフ確認**: transition_heatmap でスパースな行（Phantom State 候補）の有無を確認

---

## 注意点・リスク

| リスク | 対策 |
|--------|------|
| GPU メモリ: ξ テンソル (T×K×K×K) が ~138MB | K=8, T≈67546 で推定。GTX 1650 (4GB VRAM) で許容範囲。問題時は mini-batch 化 |
| パラメータ過多による Phantom State 増加 | BIC が Bigram を上回る場合も記録し、分析の材料として扱う |
| Focus State が k により変わる | Step 1 の自動同定ロジックで対応 |
| 実行時間 (Trigram は Bigram の ~7倍遅い) | N_RESTARTS=20 は維持。GPU があれば許容範囲 |
