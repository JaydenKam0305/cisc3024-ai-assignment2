# Actual AI workflow for the 2026 revision

## User requests

The student supplied AIassign2.pdf and asked in Chinese for help completing AI Assignment 2, with the six required report elements: how AI was asked to find the method, method description, AI implementation, settings/results, learning, and source webpage. The student requested English and supplied GitHub username JaydenKam0305. After the initial 2024 VBLL report, the student asked “有2026年的嗎 24年的好像有點舊了” (Is there a 2026 one? The 2024 one seems a little old), and approved the AI's recommendation of the ICML 2026 Rich-BLL paper.

## Search and verification

The initial search included `2024 2025 evidential deep learning classification Dirichlet image classification paper loss` and `2024 Laplace approximation last layer image classification paper Bayesian`. For the 2026 revision, AI inspected the official PMLR ICML 2026 record, arXiv 2602.01279 version 2, the authors' OATML publication page, and the image-classification section and appendix. It verified the official Fashion-MNIST source and checksum. The proceedings establish the 2026 date; the 2024 model is retained as a baseline.

Sources:
- https://proceedings.mlr.press/v306/calvo-ordonez26a.html
- https://arxiv.org/html/2602.01279v2
- https://oatml.cs.ox.ac.uk/publications/202602_Calvo-Ordonez_BayesianLastLayers.html
- https://github.com/zalandoresearch/fashion-mnist

## Actions carried out by Codex

1. Read projection/posterior formulas and classification details, then selected the class-specific variant for an independent compact implementation. The paper's main classification uses a shared global transform; this difference is disclosed.
2. Implemented exact hidden-layer scalar-logit Jacobians, ridge projection, per-class compact Gaussian covariance, uniform nested subsets, N/k posterior scaling, and Monte Carlo softmax integration.
3. Reused AI-written baseline training utilities, retrained softmax and diagonal VBLL for five seeds, and evaluated seven configurations on clean/noisy digits and two OOD probes.
4. Downloaded official Fashion-MNIST images with checksum verification; fixed and saved the 1000 sampled indices; resized to 8×8.
5. Ran ten numerical tests. A new identity test initially used a float32 identity matrix inside a float64 calculation; AI made the identity explicitly float64 and reran all tests successfully. No experimental model code changed in that repair.
6. Generated figures from saved measurements and prepared the English report, explaining negligible clean predictive differences, mixed OOD results, approximation scope, finite integration error, and timing limits.
7. Prepared current source and results for the student's existing public assignment repository. Browser upload was interrupted when the Mac locked and resumed after the student unlocked it.

This is an action log, not a fabricated transcript of extra student prompts. The student did not write the implementation code. Report reflections are AI-drafted lessons supported by the work and should be reviewed by the student.
