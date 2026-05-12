import json
import os
from datetime import datetime

try:
    import gdown
except ImportError:
    gdown = None

import numpy as np
from scipy.io import loadmat

# We import the internal subset, lags, and shift functions to build a "White-Box" solver
from task_and_baseline import (
    baseline, 
    build_task_helpers, 
    MODEL_SUBSET, 
    MODEL_LAGS, 
    shift_signal, 
    shifted_window
)

def ensure_dataset(path="challenge.mat"):
    if os.path.exists(path):
        return
    if gdown is None:
        raise RuntimeError(
            "challenge.mat is missing and gdown is not installed. "
            "Install gdown or place the dataset in the repo root."
        )
    url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
    gdown.download(url, path, quiet=False, fuzzy=True)


ensure_dataset("challenge.mat")

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def run_diagnostics(tx_n, rx, helpers):
    print("\n--- Running Diagnostics ---")
    score_filter = helpers["score_filter"]
    
    # 1. Baseline residual analysis
    tx_pred = helpers["fit_tx_prediction"](rx)
    residual = rx - tx_pred
    
    band_residual = np.zeros_like(residual)
    for ch in range(4):
        band_residual[:, ch] = score_filter(residual[:, ch])
        
    cov = band_residual.conj().T @ band_residual
    evals, _ = np.linalg.eigh(cov)
    print(f"1. Global Eigenvalues of Band-Filtered Residual:")
    print(f"   {evals}")
    print(f"   Global Rank-1 Dominance: {evals[-1] / np.sum(evals):.4f}")
    
    # 2. Block-based eigenvalue analysis (Time-variance check)
    print("\n2. Rank-1 Dominance over time blocks (Is the source moving?):")
    num_blocks = 8
    block_size = len(band_residual) // num_blocks
    for i in range(num_blocks):
        block = band_residual[i*block_size : (i+1)*block_size]
        c = block.conj().T @ block
        ev, _ = np.linalg.eigh(c)
        print(f"   Block {i+1}: {ev[-1] / np.sum(ev):.4f}")
        
    # 3. Check for out-of-band energy ratio
    p_unfiltered = np.mean(np.abs(residual)**2)
    p_filtered = np.mean(np.abs(band_residual)**2)
    print(f"\n3. Power Analysis:")
    print(f"   Unfiltered Residual Power: {p_unfiltered:.2e}")
    print(f"   Filtered Residual Power:   {p_filtered:.2e}")
    print(f"   Ratio (Band / Total):      {p_filtered / p_unfiltered:.4f}")
    
    print("---------------------------\n")


def run_diagnostics_2(tx_n, rx, helpers):
    print("\n--- Diagnostics 2.0: The Raw Subspace ---")
    score_filter = helpers["score_filter"]
    
    # 1. Analyze the RAW received signal in the frequency band
    band_rx = np.zeros_like(rx)
    for ch in range(4):
        band_rx[:, ch] = score_filter(rx[:, ch])
        
    cov_raw = band_rx.conj().T @ band_rx
    evals_raw, vecs_raw = np.linalg.eigh(cov_raw)
    
    print("1. Eigenvalues of RAW band_rx:")
    print(f"   {evals_raw}")
    print(f"   Raw Rank-1 Dominance: {evals_raw[-1]/np.sum(evals_raw):.4f}")
    
    # 2. Test a "Rank-1 First" subtraction
    v_dom = vecs_raw[:, -1]
    shared = band_rx @ v_dom
    denom = np.vdot(v_dom, v_dom) + 1e-30
    
    rank1_raw = np.zeros_like(band_rx)
    for ch in range(4):
        rank1_raw[:, ch] = (shared * np.conj(v_dom[ch])) / denom
        
    # See how the baseline TX fitter performs on a signal where 
    # the RAW external interference is already removed
    cleaner_rx = rx - rank1_raw
    tx_pred_test = helpers["fit_tx_prediction"](cleaner_rx)
    
    # Calculate a rough pseudo-score
    test_residual = cleaner_rx - tx_pred_test
    p_orig = np.mean(np.abs(rx)**2)
    p_res = np.mean(np.abs(test_residual)**2)
    print(f"\n2. Pseudo-score if we extract Rank-1 BEFORE TX fitting:")
    print(f"   ~ {10 * np.log10(p_orig / p_res):.2f} dB")
    print("---------------------------\n")


def your_canceller(tx_n, rx):
    """
    TX + rank-1 canceller with joint band-limited scaling.
    """
    del tx_n
    fit_tx = helpers["fit_tx_prediction"]
    score_filter = helpers["score_filter"]

    def extract_rank1_from_band(band_matrix, quiet_mask=None):
        if quiet_mask is not None and np.any(quiet_mask):
            cov_src = band_matrix[quiet_mask]
        else:
            cov_src = band_matrix

        cov = cov_src.conj().T @ cov_src / max(cov_src.shape[0], 1)
        _, vecs = np.linalg.eigh(cov)
        shared = band_matrix @ vecs[:, -1]
        denom = np.vdot(shared, shared) + 1e-30
        return np.column_stack(
            [
                (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
                for ch in range(band_matrix.shape[1])
            ]
        )

    def band_limit(x):
        out = np.zeros_like(x)
        for ch in range(x.shape[1]):
            out[:, ch] = score_filter(x[:, ch])
        return out

    # Stage 1: TX prediction from the raw received signal.
    tx_part = fit_tx(rx)
    residual = rx - tx_part

    # Stage 2: rank-1 estimate on the band-limited residual.
    rank1_part = extract_rank1_from_band(band_limit(residual))

    # Stage 3: jointly scale TX and rank-1 terms to minimize band power.
    band_rx = band_limit(rx)
    band_tx = band_limit(tx_part)
    band_rank1 = rank1_part

    y = band_rx.reshape(-1)
    x = np.column_stack([band_tx.reshape(-1), band_rank1.reshape(-1)])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    alpha = coef[0]
    beta = coef[1]

    return rx - (alpha * tx_part + beta * rank1_part)


def log_trial(avg_db, per_channel_db, path="trial_history.jsonl"):
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "average_db": float(avg_db),
        "per_channel_db": [float(v) for v in per_channel_db],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# run_diagnostics_2(tx_n, rx, helpers)


# print("\n=== Baseline ===")
# baseline_reds, baseline_avg = helpers["score"](
#     rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
# )

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    # "baseline": {
    #     "per_channel_db": baseline_reds,
    #     "average_db": baseline_avg,
    # },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

log_trial(yours_avg, yours_reds)
