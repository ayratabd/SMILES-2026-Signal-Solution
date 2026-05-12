# SMILES-2026 Signal Interference Cancellation Solution

## 1. Reproducibility Instructions
- **Environment:** Python 3.x with `numpy`, `scipy`, and `gdown`.
- **Commands:** From the repository root, run:

```bash
python applicant_solution.py
```

- This loads `challenge.mat`, runs the baseline helper functions, executes `your_canceller`, and writes `results.json`.

## 1.1 Results Summary

| Run | ch0 (dB) | ch1 (dB) | ch2 (dB) | ch3 (dB) | Avg (dB) |
| --- | --- | --- | --- | --- | --- |
| Baseline | 3.98 | 4.86 | 3.49 | 3.74 | 4.02 |
| Yours | 11.58 | 9.97 | 10.61 | 9.37 | 10.38 |

## 2. Final Solution Description
I modified only `your_canceller` in `applicant_solution.py` to implement a TX + rank-1 canceller with a final joint scaling step that matches the evaluator's explainability checks.

**Stage 1 - TX-dependent interference removal.** I use the provided helper `fit_tx_prediction` to estimate the nonlinear TX-driven component from the raw received signal, then subtract it:

$$\text{tx\_part} = \text{fit\_tx}(rx), \quad res = rx - \text{tx\_part}$$

**Stage 2 - Rank-1 spatial interference removal in the scored band.** I band-limit the residual with `score_filter`, form a 4x4 spatial covariance matrix, and take its dominant eigenvector as the spatial signature of the external interference. Projecting the band-limited residual onto this vector yields the rank-1 component:

$$\text{cov} = R^H R, \quad v = \arg\max\_v v^H \text{cov} v$$

$$\text{rank1\_part} = (R v) v^H$$

**Stage 3 - Joint scaling in the scored band.** I solve a 2-parameter least squares fit in the scored band to scale the TX and rank-1 components jointly, minimizing the residual band power while keeping the removed signal explainable as TX + rank-1:

$$\min\_{\alpha,\beta} \lVert R - \alpha T - \beta E \rVert^2$$

This improves the cancellation metric without breaking the evaluator's validity checks because the removed component remains a TX-model term plus a rank-1 spatial term in the scored band.

This explicitly removes both interference sources required by the validity checks: a TX-driven component and a spatially coherent rank-1 component within the scored band.

## 3. Experiments and Failed Attempts
- **TX-only baseline:** Using only `fit_tx_prediction` gives a modest improvement but leaves a strong spatially coherent residual, which is insufficient for the >8 dB target.
- **Rank-1 before TX or full-band covariance:** I considered extracting the spatial component from the raw or full-band residual. This risks mixing TX leakage or out-of-band energy into the spatial signature, which can reduce cancellation and threaten the explainability guard. Restricting the covariance to the scored band avoids that issue.