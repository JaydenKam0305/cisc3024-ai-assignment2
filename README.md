# Bayesian Handwritten Digit Recognition

CISC3024 AI Assignment 2, October 2026. All implementation code and report preparation were performed by the Codex AI assistant.

This is an independent educational implementation of the **diagonal discriminative Variational Bayesian Last Layer** (VBLL) formulation, based on Theorem 2 / Eq. 14 of Harrison, Willes and Snoek, ICLR 2024: https://arxiv.org/abs/2404.11599

The implementation uses a standard normal weight prior, a diagonal variational covariance, and zero additional logit noise. It is not a reproduction of the paper's large-scale benchmarks, and it does not depend on the authors' `vbll` package. The authors' separate implementation is https://github.com/VectorInstitute/vbll .

## Run

Use Python 3.12 and a CPU. From this folder:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python experiment.py --output results
python plot_results.py --results results
```

The 8 x 8 digit image dataset is bundled with scikit-learn; a separate dataset download is unnecessary. The stratified train/validation/test split is fixed at random seed 42. Runs use training seeds 0 through 4. Exact timing and minor numerical differences depend on the machine and software versions.

## Files

- `experiment.py`: model, deterministic variational loss, training and evaluation.
- `test_experiment.py`: Gaussian KL, sampled weight/logit moments, Jensen bound and gradient, probability and metric checks.
- `plot_results.py`: figures derived from the result files.
- `AI_WORKFLOW.md`: actual user request, search queries and a truthful summary of AI work.
- `results/metrics.csv`: one row per model, seed and test condition.
- `results/summary.json`: means and sample standard deviations across five runs.
- `results/predictions.npz`: probabilities, labels and indices used for verification.
- `results/history.json`: training loss and validation negative log likelihood.
- `results/split_indices.json`: exact dataset partition.
- `results/environment.json`: software versions and execution environment.
- `results/*_seed0.pt`: selected seed 0 model weights.

## Method and interpretation

For features h augmented by a constant bias feature, each class row has a Gaussian approximate posterior q(w_k) = N(m_k, diag(v_k)). A logit has mean m_k^T h and variance v_k^T h^2. The batch objective is the mean of `logsumexp(mean + variance / 2) - mean_of_true_class`, plus Gaussian KL divided by the training set size. Training uses no posterior sampling. Prediction averages 1,024 sampled Gaussian-logit softmax vectors; checkpoint selection uses 128 samples and validation labels only.

Both methods share the feature architecture, feature initialization and mini-batch sequence for each seed. Baseline loss and head regularization differ from VBLL as documented in the report. The comparison therefore evaluates the full training methods and does not isolate the effect of posterior averaging alone.

Clean data and Gaussian corruption of standard deviations 0.2 and 0.4 are evaluated on the same held-out test images. A fixed pixel permutation is used only as a synthetic distribution-shift probe. Its entropy AUROC does not establish performance on genuine external out-of-distribution datasets. Reported entropy is predictive uncertainty, not a pure estimate of epistemic uncertainty.

## Publishing the source webpage

Create a public GitHub repository named `cisc3024-ai-assignment2`, then upload this folder's contents. Open the repository while signed out to verify public access. Use that repository's actual URL in section 6 of the report. The authors' upstream repository is a reference and must not be presented as the student's own source code.
