"""Generate report figures from saved measurements; no model retraining."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',default='results2026'); a=ap.parse_args()
    r=Path(a.results); s=json.loads((r/'summary.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180})
    names=['Softmax MLP','VBLL 2024','BLL','Rich-BLL full','Rich-BLL k128']
    fig,ax=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    for name in names:
        for aa,metric,factor in [(ax[0],'accuracy',100),(ax[1],'nll',1)]:
            aa.errorbar([0,.2,.4],[s[name][c][metric]['mean']*factor for c in ['clean','noise_0.2','noise_0.4']],
                yerr=[s[name][c][metric]['sd']*factor for c in ['clean','noise_0.2','noise_0.4']],marker='o',label=name,alpha=.8,capsize=3)
    ax[0].set_ylabel('Accuracy (%)');ax[1].set_ylabel('Negative log likelihood')
    for aa in ax:aa.set_xlabel('Pixel noise standard deviation');aa.set_xticks([0,.2,.4]);aa.grid(alpha=.2)
    ax[1].legend(fontsize=8);fig.savefig(r/'robustness2026.png');plt.close(fig)
    names=['BLL','Rich-BLL full','Rich-BLL k256','Rich-BLL k128','Rich-BLL k64']
    fig,ax=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    for j,cond in enumerate(['fashion','pixel_permutation']):
        ax[j].bar(np.arange(5),[s[n][cond]['variance_auroc']['mean'] for n in names],yerr=[s[n][cond]['variance_auroc']['sd'] for n in names],capsize=3,color=['#8797a2','#245d80','#3e7c9b','#649ab3','#93b9c9'])
        ax[j].axhline(.5,ls='--',color='gray',lw=1);ax[j].set_ylim(0,1);ax[j].set_xticks(np.arange(5),['BLL','Full','k256','k128','k64']);ax[j].set_title(['Fashion-MNIST','Permuted digit pixels'][j]);ax[j].set_ylabel('Logit variance AUROC')
    fig.savefig(r/'ood2026.png');plt.close(fig)
    z=np.load(r/'predictions.npz');a=z['v__BLL__0__clean'].ravel();b=z['v__Rich-BLL full__0__clean'].ravel()
    fig,ax=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    ax[0].scatter(a,b,s=5,alpha=.25); lim=max(a.max(),b.max());ax[0].plot([0,lim],[0,lim],'k--',lw=1);ax[0].set_xlabel('BLL logit variance');ax[0].set_ylabel('Rich-BLL full logit variance');ax[0].set_title('Seed 0 clean images and all classes')
    ratios=[]
    for name in names[1:]:ratios.append(np.median(z[f'v__{name}__0__clean']/z['v__BLL__0__clean']))
    ax[1].bar(range(4),ratios,color='#3e7c9b');ax[1].axhline(1,ls='--',color='gray');ax[1].set_xticks(range(4),['Full','k256','k128','k64']);ax[1].set_ylabel('Median variance ratio to BLL');ax[1].set_title('Seed 0 subsampling sensitivity')
    fig.savefig(r/'variance2026.png');plt.close(fig)

if __name__=='__main__':main()
