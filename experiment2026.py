"""AI implementation of Rich-BLL (ICML 2026), class-specific logit variant.
Equations 9, 13 and 21 of Calvo Ordonez et al. with ridge=1, noise variance=.1.
This educational study uses digits/MLP, not the paper's CIFAR/CNN/muP benchmark.
The class-specific option is described in paper Section 4.4; its main experiments
use a shared global transform instead. Logit posteriors are treated independently.
"""
import argparse, csv, gzip, hashlib, importlib.metadata as md, json, platform, struct, time, urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from PIL import Image
from experiment import Classifier, train, metrics, entropy

torch.set_num_threads(1)
NOISE_VAR = .1
RIDGE = 1.
METHODS = ['Softmax MLP','VBLL 2024','BLL','Rich-BLL full','Rich-BLL k256','Rich-BLL k128','Rich-BLL k64']


def features(model, x):
    with torch.no_grad():
        h = model.features(x)
        return torch.cat([h, torch.ones(len(h), 1)], 1).double()


def hidden_jacobian(model, x, cls):
    """Exact gradients of one scalar logit with respect to hidden parameters."""
    params = {k: v.detach() for k,v in model.features.named_parameters()}
    w, b = model.head.weight.detach()[cls], model.head.bias.detach()[cls]
    def scalar(pp, xx):
        return functional_call(model.features, pp, (xx,)) @ w + b
    pieces = []
    for xb in x.split(64):
        grads = vmap(jacrev(scalar, argnums=0), in_dims=(None, 0))(params, xb)
        pieces.append(torch.cat([v.reshape(len(xb), -1) for v in grads.values()], 1).double())
    return torch.cat(pieces)


def correction(h, j, ridge=RIDGE):
    # A^T is the ridge solution; ridge matches Appendix D.3.
    c = torch.linalg.solve(h.T @ h + ridge * torch.eye(h.shape[1],dtype=h.dtype), h.T @ j)
    return torch.eye(h.shape[1],dtype=h.dtype) + c @ c.T


def posterior(h, g, population_size=None, noise_var=NOISE_VAR):
    n = len(h) if population_size is None else population_size
    eye = torch.eye(h.shape[1], dtype=h.dtype)
    g_inv = torch.cholesky_inverse(torch.linalg.cholesky(g))
    precision = (n / len(h)) * (h.T @ h) / noise_var + g_inv
    return torch.cholesky_inverse(torch.linalg.cholesky(precision))


def variances(h, post):
    # post has shape (K,r,r); each row is a separate scalar-logit posterior.
    return torch.einsum('nr,krs,ns->nk',h,post,h).clamp_min(0)


def predictive(mu, var, samples=1024, seed=9000):
    gen = torch.Generator().manual_seed(seed)
    eps = torch.randn(samples,1,mu.shape[1],generator=gen,dtype=mu.dtype)
    p = torch.zeros_like(mu)
    for chunk in eps.split(64):
        p += (mu[None] + var.sqrt()[None]*chunk).softmax(-1).sum(0)
    return (p / samples).numpy()


def fashion_data(cache):
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/'t10k-images-idx3-ubyte.gz'
    url='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz'
    if not path.exists(): urllib.request.urlretrieve(url,path)
    digest=hashlib.md5(path.read_bytes()).hexdigest()
    assert digest=='bef4ecab320f06d8554ea6380940ec79', 'Dataset checksum mismatch'
    raw=gzip.decompress(path.read_bytes()); magic,n,rows,cols=struct.unpack('>IIII',raw[:16])
    assert (magic,n,rows,cols)==(2051,10000,28,28)
    a=np.frombuffer(raw,dtype=np.uint8,offset=16).reshape(n,rows,cols)
    indices=np.random.default_rng(2026).choice(n,1000,replace=False)
    resized=np.stack([np.asarray(Image.fromarray(a[i]).resize((8,8),Image.Resampling.BILINEAR)) for i in indices])
    return torch.tensor(resized.reshape(1000,64).astype('float32')/255), indices, dict(url=url,md5=digest)


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results2026'); ap.add_argument('--cache',default='data_cache'); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    ds=load_digits(); x=(ds.data/16).astype('float32'); y=ds.target
    tv,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
    tr,va=train_test_split(tv,test_size=.25,random_state=42,stratify=y[tv])
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    xtr,ytr=torch.tensor(x[tr]),torch.tensor(y[tr]); xv,yv=torch.tensor(x[va]),y[va]; xt,yt=torch.tensor(x[te]),y[te]
    rng=np.random.default_rng(2026); noise=rng.normal(size=xt.shape).astype('float32')
    inputs={'clean':xt,'noise_0.2':torch.tensor(np.clip(x[te]+.2*noise,0,1)),
            'noise_0.4':torch.tensor(np.clip(x[te]+.4*noise,0,1)),'pixel_permutation':xt[:,rng.permutation(64)]}
    xf,fidx,fmeta=fashion_data(Path(args.cache)); inputs['fashion']=xf
    rows=[]; ood=[]; setup=[]; convergence=[]; saved={}; history={}; subsets={}; numeric=[]
    for seed in range(5):
        model,epoch,vnll,secs,hist=train(xtr,ytr,xv,yv,seed,False)
        vmodel,vepoch,vvnll,vsecs,vhist=train(xtr,ytr,xv,yv,seed,True)
        history[str(seed)]=dict(softmax=hist,vbll=vhist,softmax_epoch=epoch,vbll_epoch=vepoch)
        h=features(model,xtr); r=h.shape[1]; eye=torch.eye(r,dtype=torch.float64)
        order=np.random.default_rng(4000+seed).permutation(len(tr))
        subset_map={f'Rich-BLL k{k}':order[:k] for k in [256,128,64]}; subset_map['Rich-BLL full']=np.arange(len(tr))
        subsets[str(seed)]={name:idx.tolist() for name,idx in subset_map.items()}
        allg={name:[] for name in subset_map}; fitsec={name:0. for name in subset_map}
        extract_time=0.
        for c in range(10):
            t=time.perf_counter(); j=hidden_jacobian(model,xtr,c); extract_time+=time.perf_counter()-t
            for name,idx in subset_map.items():
                t=time.perf_counter(); g=correction(h[idx],j[idx]); allg[name].append(g); fitsec[name]+=time.perf_counter()-t
            del j
        posts={}; gs={name:torch.stack(v) for name,v in allg.items()}
        t=time.perf_counter(); basepost=posterior(h,eye); bll_fit=time.perf_counter()-t
        posts['BLL']=basepost.repeat(10,1,1)
        for name,idx in subset_map.items():
            t=time.perf_counter(); posts[name]=torch.stack([posterior(h[idx],g,population_size=len(tr)) for g in gs[name]])
            fitsec[name]+=time.perf_counter()-t
        setups=[dict(seed=seed,method='BLL',n_cov=len(tr),gradient_seconds=0.,fit_seconds=bll_fit)]
        setups += [dict(seed=seed,method=name,n_cov=len(idx),gradient_seconds=extract_time,fit_seconds=fitsec[name]) for name,idx in subset_map.items()]
        # Shared all-training Jacobian extraction amortizes the ablation. Timings
        # do not claim the cost of an independently optimized subsampled pipeline.
        setup.extend(setups)
        vals={}; probs={}; means={}
        for condition,xx in inputs.items():
            hh=features(model,xx)
            with torch.no_grad(): mu=model.moments(xx)[0].double(); vmu,vvar=vmodel.moments(xx)
            means[condition]=mu
            probs[('Softmax MLP',condition)]=mu.softmax(-1).numpy()
            probs[('VBLL 2024',condition)]=vmodel.probabilities(xx)
            vals[('VBLL 2024',condition)]=vvar.detach().double()
            for name,post in posts.items():
                vv=variances(hh,post); vals[(name,condition)]=vv
                probs[(name,condition)]=predictive(mu,vv)
            if condition=='clean':
                minimum=(vals[('Rich-BLL full',condition)]-vals[('BLL',condition)]).min().item()
                assert minimum > -1e-9
                numeric.append(dict(seed=seed,min_rich_minus_bll_variance=minimum))
            for name in METHODS:
                saved[f'p__{name}__{seed}__{condition}']=probs[(name,condition)]
                if name!='Softmax MLP': saved[f'v__{name}__{seed}__{condition}']=vals[(name,condition)].numpy()
                if condition in ['clean','noise_0.2','noise_0.4']:
                    rows.append(dict(model=name,seed=seed,condition=condition,**metrics(probs[(name,condition)],yt)))
        for name in METHODS:
            for condition in ['fashion','pixel_permutation']:
                p0,p1=probs[(name,'clean')],probs[(name,condition)]
                labs=np.r_[np.zeros(len(p0)),np.ones(len(p1))]
                row=dict(model=name,seed=seed,ood=condition,entropy_auroc=float(roc_auc_score(labs,np.r_[entropy(p0),entropy(p1)])),variance_auroc='')
                if name!='Softmax MLP':
                    # Select the class with highest mean logit; never use an OOD label.
                    ii=means['clean'].argmax(1); jj=means[condition].argmax(1)
                    vv0=vals[(name,'clean')][torch.arange(len(ii)),ii].numpy()
                    vv1=vals[(name,condition)][torch.arange(len(jj)),jj].numpy()
                    row['variance_auroc']=float(roc_auc_score(labs,np.r_[vv0,vv1]))
                ood.append(row)
        if seed==0:
            reference=probs[('Rich-BLL full','clean')]
            for count in [64,256,1024,4096]:
                pp=predictive(means['clean'],vals[('Rich-BLL full','clean')],samples=count)
                convergence.append(dict(samples=count,nll=metrics(pp,yt)['nll'],max_probability_change_vs_1024=float(abs(pp-reference).max())))
            torch.save(dict(backbone=model.state_dict(),vbll=vmodel.state_dict(),posteriors=posts,corrections=gs),out/'models_seed0.pt')
        print('seed',seed,'completed',[(n,round(metrics(probs[(n,'clean')],yt)['nll'],5)) for n in METHODS],flush=True)
    write_csv(out/'metrics.csv',rows); write_csv(out/'ood_metrics.csv',ood); write_csv(out/'setup_times.csv',setup); write_csv(out/'mc_convergence.csv',convergence)
    summary={}
    for name in METHODS:
        summary[name]={}
        for cond in ['clean','noise_0.2','noise_0.4']:
            rr=[r for r in rows if r['model']==name and r['condition']==cond]
            summary[name][cond]={k:dict(mean=float(np.mean([r[k] for r in rr])),sd=float(np.std([r[k] for r in rr],ddof=1))) for k in ['accuracy','nll','brier','ece','confidence','entropy']}
        for cond in ['fashion','pixel_permutation']:
            rr=[r for r in ood if r['model']==name and r['ood']==cond]
            summary[name][cond]={}
            for k in ['entropy_auroc','variance_auroc']:
                values=[r[k] for r in rr if r[k]!='']
                if values: summary[name][cond][k]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'split_indices.json').write_text(json.dumps(dict(train=tr.tolist(),validation=va.tolist(),test=te.tolist(),fashion=fidx.tolist(),subsets_relative_to_train=subsets)))
    (out/'numeric_checks.json').write_text(json.dumps(numeric,indent=2))
    np.savez_compressed(out/'predictions.npz',targets=yt,**saved)
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),device='CPU',threads=1,versions={k:md.version(k) for k in ['torch','numpy','scikit-learn','scipy','matplotlib','Pillow']},fashion=fmeta,ridge=RIDGE,noise_variance=NOISE_VAR,seeds=list(range(5)),train=len(tr),validation=len(va),test=len(te),fashion_ood=1000,prediction_samples=1024),indent=2))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
