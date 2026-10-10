"""AI-written diagonal discriminative VBLL experiment (Harrison et al., ICLR 2024).
Training implements Eq. 14 with zero additional logit noise and a Gaussian prior.
Prediction integrates the Gaussian last-layer logits by Monte Carlo.
Run: python experiment.py --output results
"""
import argparse, copy, csv, json, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import sklearn

torch.set_num_threads(1)

class Classifier(nn.Module):
    def __init__(self, bayesian=False):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Linear(32, 10)
        self.bayesian = bayesian
        if bayesian:
            self.log_var = nn.Parameter(torch.full((10, 33), -7.0))

    def moments(self, x):
        h = self.features(x)
        mu = self.head(h)
        if not self.bayesian:
            return mu, torch.zeros_like(mu)
        augmented = torch.cat([h, torch.ones((len(h), 1), device=h.device)], dim=1)
        var = augmented.square() @ self.log_var.exp().T
        return mu, var

    def kl(self):
        # KL(q || N(0,I)), including bias as a weight on a constant feature.
        mean = torch.cat([self.head.weight, self.head.bias[:, None]], dim=1)
        return 0.5 * (self.log_var.exp() + mean.square() - 1 - self.log_var).sum()

    def objective(self, x, y, n_train):
        mu, var = self.moments(x)
        if self.bayesian:
            return (torch.logsumexp(mu + 0.5 * var, dim=1) - mu.gather(1, y[:, None]).squeeze(1)).mean() + self.kl() / n_train
        return nn.functional.cross_entropy(mu, y)

    @torch.no_grad()
    def probabilities(self, x, samples=1024, seed=9000):
        mu, var = self.moments(x)
        if not self.bayesian:
            return mu.softmax(-1).numpy()
        # Independent class rows imply independent Gaussian logits for each image.
        # Common standard normal draws across inputs reduce memory; marginal
        # per-image estimates remain valid. No test labels enter this step.
        generator = torch.Generator().manual_seed(seed)
        eps = torch.randn((samples, 1, 10), generator=generator)
        probs = torch.zeros_like(mu)
        for chunk in eps.split(64):
            probs += (mu[None] + var.sqrt()[None] * chunk).softmax(-1).sum(0)
        return (probs / samples).numpy()


def metrics(p, y):
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=2e-6)
    pred, conf = p.argmax(1), p.max(1)
    correct = pred == y
    ece = 0.0
    for k in range(10):
        mask = (conf >= k / 10) & (conf < (k + 1) / 10 if k < 9 else conf <= 1)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return dict(accuracy=float(correct.mean()), nll=float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
                brier=float(np.square(p - np.eye(10)[y]).sum(1).mean()), ece=float(ece),
                confidence=float(conf.mean()), entropy=float(entropy(p).mean()))


def entropy(p):
    return -(p * np.log(p.clip(1e-12))).sum(1)


def train(x, y, xv, yv, seed, bayesian):
    torch.manual_seed(seed)
    model = Classifier(bayesian)
    groups = [dict(params=model.features.parameters(), weight_decay=1e-4),
              dict(params=model.head.parameters(), weight_decay=0 if bayesian else 1e-4)]
    if bayesian:
        groups.append(dict(params=[model.log_var], weight_decay=0))
    optim = torch.optim.Adam(groups, lr=0.001)
    gen = torch.Generator().manual_seed(seed + 100)
    best_nll, best_epoch, state = float('inf'), None, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, 151):
        model.train()
        losses = []
        for idx in torch.randperm(len(x), generator=gen).split(64):
            optim.zero_grad()
            loss = model.objective(x[idx], y[idx], len(x))
            assert torch.isfinite(loss)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        if epoch % 5 == 0:
            model.eval()
            nll = metrics(model.probabilities(xv, samples=128), yv)['nll']
            history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation_nll=nll))
            if nll < best_nll:
                best_nll, best_epoch, state = nll, epoch, copy.deepcopy(model.state_dict())
    elapsed = time.perf_counter() - start
    model.load_state_dict(state)
    model.eval()
    return model, best_epoch, best_nll, elapsed, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ds = load_digits()
    x = (ds.data / 16.0).astype('float32'); y = ds.target
    trainval, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=42, stratify=y)
    train_idx, val = train_test_split(trainval, test_size=.25, random_state=42, stratify=y[trainval])
    assert not (set(train_idx) & set(val) or set(train_idx) & set(test) or set(val) & set(test))
    xtr, ytr = torch.tensor(x[train_idx]), torch.tensor(y[train_idx])
    xv, yv = torch.tensor(x[val]), y[val]
    xt, yt = torch.tensor(x[test]), y[test]
    rng = np.random.default_rng(2026)
    noise = rng.normal(size=xt.shape).astype('float32')
    conditions = {'clean': xt, 'noise_0.2': torch.tensor(np.clip(x[test] + .2 * noise, 0, 1)),
                  'noise_0.4': torch.tensor(np.clip(x[test] + .4 * noise, 0, 1))}
    perm = rng.permutation(64)
    shuffled = xt[:, perm]
    rows, histories, preds = [], {}, {}
    for seed in [0, 1, 2, 3, 4]:
        for bayesian, name in [(False, 'Softmax MLP'), (True, 'Diagonal VBLL')]:
            model, epoch, val_nll, elapsed, hist = train(xtr, ytr, xv, yv, seed, bayesian)
            histories[f'{name}_{seed}'] = hist
            clean = model.probabilities(xt)
            shift = model.probabilities(shuffled)
            auroc = roc_auc_score(np.r_[np.zeros(len(xt)), np.ones(len(xt))], np.r_[entropy(clean), entropy(shift)])
            for condition, xx in conditions.items():
                p = clean if condition == 'clean' else model.probabilities(xx)
                row = dict(model=name, seed=seed, condition=condition, best_epoch=epoch,
                           validation_nll=val_nll, train_seconds=elapsed, shift_auroc=float(auroc), **metrics(p, yt))
                rows.append(row)
                preds[f'{name}_{seed}_{condition}'] = p
            if seed == 0:
                torch.save(model.state_dict(), out / ('vbll_seed0.pt' if bayesian else 'softmax_seed0.pt'))
            print(name, seed, epoch, metrics(clean, yt), flush=True)
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for name in ['Softmax MLP', 'Diagonal VBLL']:
        summary[name] = {}
        for c in conditions:
            rr = [r for r in rows if r['model'] == name and r['condition'] == c]
            summary[name][c] = {k: dict(mean=float(np.mean([r[k] for r in rr])), sd=float(np.std([r[k] for r in rr], ddof=1)))
                                for k in ['accuracy','nll','brier','ece','confidence','entropy','train_seconds','shift_auroc']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'history.json').write_text(json.dumps(histories, indent=2))
    np.savez_compressed(out / 'predictions.npz', targets=yt, test_indices=test, **preds)
    (out / 'split_indices.json').write_text(json.dumps(dict(train=train_idx.tolist(),validation=val.tolist(),test=test.tolist())))
    import importlib.metadata as md
    env = dict(python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
               processor=platform.processor(), device='CPU', threads=1,
               versions={p:md.version(p) for p in ['torch','numpy','scikit-learn','scipy','matplotlib']},
               split_sizes=dict(train=len(train_idx),validation=len(val),test=len(test)), seeds=list(range(5)),
               prediction_samples=1024, validation_samples=128, noise_seed=2026, split_seed=42)
    (out / 'environment.json').write_text(json.dumps(env, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
