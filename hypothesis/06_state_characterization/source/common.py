"""
hypothesis/06_state_characterization の共通ユーティリティ

語彙定義:
  モデルは修正前 DB (32 文字語彙) で学習済み。
  修正後 DB では <@H=2>fshdar → fshdar、<@H=3>tchedy → tchedy に変更されたため、
  アノテーション文字 2,3,<,=,>,@,H (indices 3-9) は現 DB に存在しないが、
  モデルとの整合性のため ORIG_ALL_CHARS (32 文字) を引き続き使用する。
"""

import re
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    DEVICE = None


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")


# ─── パス定義 ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent.parent          # /home/practi/work_voy

DB_PATH             = ROOT / "data/voynich.db"
BIGRAM_CACHE        = ROOT / "hypothesis/01_bigram/results/hmm_model_cache"
TRIGRAM_CACHE       = ROOT / "hypothesis/03_trigram/results/hmm_model_cache"
COMPOUND_SPLIT_PATH = ROOT / "hypothesis/00_slot_model/data/compound_words.txt"
SINGLE_WORDS_PATH   = ROOT / "hypothesis/00_slot_model/data/words_base_only.txt"
OUT_DIR             = ROOT / "hypothesis/06_state_characterization/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 語彙（モデル学習時 32 文字） ──────────────────────────────────────────
BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"

# sorted(DB 修正前の全文字) = 特殊 3 + アノテーション 7 + Voynich 22 = 32
ORIG_RAW_CHARS = [
    "2", "3", "<", "=", ">", "@", "H",              # indices 3-9  (アノテーション)
    "a", "c", "d", "e", "f", "g", "h", "i", "k",   # indices 10-18
    "l", "m", "n", "o", "p", "q", "r", "s", "t",   # indices 19-27
    "v", "x", "y", "z",                              # indices 28-31
]
ORIG_ALL_CHARS = [BOS_CHAR, EOS_CHAR, PAD_CHAR] + ORIG_RAW_CHARS
ORIG_CHAR2IDX  = {c: i for i, c in enumerate(ORIG_ALL_CHARS)}

# Voynich 文字のみ (表示・分析用)：indices 10-31
VOY_CHARS   = ORIG_ALL_CHARS[10:]
VOY_INDICES = list(range(10, 32))

# アノテーション文字 indices（emission 可視化で除外）
ANNOT_INDICES = list(range(3, 10))

# ─── V8 スロット定義（character-level） ────────────────────────────────────
SLOTS_V8_ENTRIES = [
    ["l", "r", "o", "y", "s", "v"],
    ["q", "s", "d", "x", "l", "r", "h", "z"],
    ["o", "y"],
    ["d", "r"],
    ["t", "k", "p", "f"],
    ["ch", "sh"],
    ["cth", "ckh", "cph", "cfh"],
    ["eee", "ee", "e", "g"],
    ["k", "t", "p", "f", "ch", "sh", "l", "r", "o", "y"],
    ["s", "d", "c"],
    ["o", "a", "y"],
    ["iii", "ii", "i"],
    ["d", "l", "r", "m", "n"],
    ["s"],
    ["y"],
    ["k", "t", "p", "f", "l", "r", "o", "y"],
]
SLOT_NAMES = [
    "C1", "C2", "V_a", "C_simple", "gallows",
    "C_digraph", "C_trigraph", "V_multi",
    "C4_or_V", "C_bench", "V_a2", "V_conson",
    "C5", "C6", "V_final", "C4b",
]

# char → set of slot names（先頭文字が一致するスロット）
CHAR_TO_SLOTS: dict[str, set] = {}
for _entries, _name in zip(SLOTS_V8_ENTRIES, SLOT_NAMES):
    for _entry in _entries:
        _fc = _entry[0]
        CHAR_TO_SLOTS.setdefault(_fc, set()).add(_name)

# 意味的グループ（可視化用）
CHAR_GROUPS = {
    "gallows":   ["t", "k", "p", "f"],
    "bench":     ["q", "d"],
    "vowel":     ["a", "e", "o", "y", "i"],
    "sibilant":  ["s", "h"],
    "liquid":    ["l", "r"],
    "nasal":     ["m", "n"],
    "other":     ["c", "g", "v", "x", "z"],
}
CHAR_GROUP_MAP = {c: g for g, cs in CHAR_GROUPS.items() for c in cs}

# ─── known Focus States ──────────────────────────────────────────────────
FOCUS_STATES = {
    ("bigram",   7): 2,
    ("bigram",   8): 1,
    ("trigram",  7): 2,
    ("trigram",  8): 5,
}
PHANTOM_STATES = {
    ("bigram",  7): [3],
    ("bigram",  8): [4],
    ("trigram", 7): [6],
    ("trigram", 8): [6],
}


# ════════════════════════════════════════════════════════════════════════
# データロード
# ════════════════════════════════════════════════════════════════════════
def load_compound_splits() -> dict:
    splits = {}
    pattern = re.compile(r"\[(\d+)基\]\s+(\S+)\s+->\s+(.+)")
    for line in COMPOUND_SPLIT_PATH.read_text("utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            splits[m.group(2)] = tuple(b.strip() for b in m.group(3).split("+"))
    return splits


def load_single_words() -> list:
    words = []
    for line in SINGLE_WORDS_PATH.read_text("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("<"):
            words.append(line)
    return words


def load_all_words() -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = pd.read_sql_query(
        "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''", conn
    )["word"].tolist()
    conn.close()
    return sorted(set(w for w in rows if len(w) >= 2))


def get_boundary_positions(splits: tuple) -> tuple[set, set]:
    bd_end, bd_start = set(), set()
    cumlen = 0
    for base in splits[:-1]:
        cumlen += len(base)
        bd_end.add(cumlen - 1)
        bd_start.add(cumlen)
    return bd_end, bd_start


# ════════════════════════════════════════════════════════════════════════
# モデルロード
# ════════════════════════════════════════════════════════════════════════
def load_bigram_model(k: int) -> dict | None:
    path = BIGRAM_CACHE / f"full_k{k}.npz"
    if not path.exists():
        log(f"  [bigram k={k}] not found: {path}")
        return None
    d = np.load(path)
    return {
        "start":        d["start"],
        "transmat":     d["trans"],
        "emiss":        d["emiss"],
        "log_start":    np.log(d["start"] + 1e-35),
        "log_transmat": np.log(d["trans"]  + 1e-35),
        "log_emiss":    np.log(d["emiss"]  + 1e-35),
        "logL": float(d["logL"][0]),
        "k":    int(d["emiss"].shape[0]),
    }


def load_trigram_model(k: int) -> dict | None:
    path = TRIGRAM_CACHE / f"trigram_k{k}.npz"
    if not path.exists():
        log(f"  [trigram k={k}] not found: {path}")
        return None
    d = np.load(path)
    m = {
        "emiss":        d["emiss"],
        "log_start":       torch.tensor(np.log(d["start"]       + 1e-35), device=DEVICE),
        "log_start_trans": torch.tensor(np.log(d["start_trans"] + 1e-35), device=DEVICE),
        "log_transmat":    torch.tensor(np.log(d["trans"]        + 1e-35), device=DEVICE),
        "log_emiss":       torch.tensor(np.log(d["emiss"]        + 1e-35), device=DEVICE),
        "logL": float(d["logL"][0]),
        "k":    int(d["trans"].shape[0]),
    }
    return m


# ════════════════════════════════════════════════════════════════════════
# Viterbi デコーダ（trigram_hmm_fast.py バグ修正済み版と同一ロジック）
# ════════════════════════════════════════════════════════════════════════
def word_to_seq(word: str) -> np.ndarray:
    return np.array(
        [ORIG_CHAR2IDX[BOS_CHAR]] + [ORIG_CHAR2IDX[c] for c in word] + [ORIG_CHAR2IDX[EOS_CHAR]],
        dtype=np.int32,
    )


def viterbi_trigram(model: dict, X_np: np.ndarray) -> list:
    log_start       = model["log_start"]
    log_start_trans = model["log_start_trans"]
    log_transmat    = model["log_transmat"]
    log_emiss       = model["log_emiss"]

    X = torch.tensor(X_np, dtype=torch.long, device=DEVICE)
    T = X.shape[0]
    k = log_start.shape[0]

    if T == 0:
        return []
    log_delta_1d = log_start + log_emiss[:, X[0]]
    if T == 1:
        return [torch.argmax(log_delta_1d).item()]

    log_delta = (log_delta_1d.unsqueeze(1)
                 + log_start_trans
                 + log_emiss[:, X[1]].unsqueeze(0))
    psi_list = []
    for t in range(2, T):
        vals = log_delta.unsqueeze(2) + log_transmat
        max_vals, argmax_i = torch.max(vals, dim=0)
        log_delta = max_vals + log_emiss[:, X[t]].unsqueeze(0)
        psi_list.append(argmax_i.cpu())

    flat_idx = torch.argmax(log_delta)
    j_last = (flat_idx // k).item()
    l_last = (flat_idx % k).item()

    path = [0] * T
    path[T - 1] = l_last
    path[T - 2] = j_last
    for t_back in range(T - 2, 1, -1):
        path[t_back - 1] = psi_list[t_back - 1][path[t_back], path[t_back + 1]].item()
    return path


def viterbi_bigram(model: dict, X_np: np.ndarray) -> list:
    log_start    = model["log_start"]
    log_transmat = model["log_transmat"]
    log_emiss    = model["log_emiss"]
    T = len(X_np)
    k = len(log_start)
    if T == 0:
        return []

    delta = log_start + log_emiss[:, X_np[0]]
    psi = [np.zeros(k, dtype=int)]
    for t in range(1, T):
        trans_vals = delta[:, None] + log_transmat  # (k, k)
        best = np.argmax(trans_vals, axis=0)
        delta = trans_vals[best, np.arange(k)] + log_emiss[:, X_np[t]]
        psi.append(best)

    path = [0] * T
    path[T - 1] = int(np.argmax(delta))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]
    return path


def decode_words(words, model: dict, model_type: str = "trigram") -> dict:
    """返り値: {word: states_list (BOS/EOS 除く)}"""
    viterbi = viterbi_trigram if model_type == "trigram" else viterbi_bigram
    results = {}
    for word in words:
        if not all(c in ORIG_CHAR2IDX for c in word):
            continue
        try:
            path = viterbi(model, word_to_seq(word))
            states = path[1:-1]
            if states:
                results[word] = states
        except Exception:
            pass
    return results


def compute_occupancy(all_words: list, model: dict, model_type: str = "trigram") -> np.ndarray:
    k = model["k"]
    counts = np.zeros(k, dtype=int)
    total = 0
    decoded = decode_words(all_words, model, model_type)
    for states in decoded.values():
        for s in states:
            counts[s] += 1
            total += 1
    return counts / total if total > 0 else np.zeros(k)
