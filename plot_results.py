"""Generate report figures from measured experiment outputs."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--results', default='results')
    args = parser.parse_args(); root = Path(args.results)
    summary = json.loads((root/'summary.json').read_text())
    models = ['Softmax MLP','Diagonal VBLL']; colors=['#475569','#087f8c']
    conditions=['clean','noise_0.2','noise_0.4']; labels=['Clean','Noise 0.2','Noise 0.4']
    fig, axes = plt.subplots(1,3,figsize=(10.4,3.2),layout='constrained')
    for ax, metric, title in zip(axes,['accuracy','nll','ece'],['Accuracy ↑','Negative log likelihood ↓','Calibration error ↓']):
        for model,color in zip(models,colors):
            v=[summary[model][c][metric]['mean'] for c in conditions]
            e=[summary[model][c][metric]['sd'] for c in conditions]
            ax.errorbar(range(3),v,yerr=e,marker='o',capsize=3,color=color,label=model)
        ax.set_xticks(range(3),labels); ax.set_title(title,fontsize=11); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.savefig(root/'robustness.png',dpi=220); plt.close(fig)
    data=np.load(root/'predictions.npz'); y=data['targets']
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.4),layout='constrained')
    for model,color in zip(models,colors):
        # Reliability uses separate run predictions pooled, not ensemble averaging.
        p=np.concatenate([data[f'{model}_{s}_clean'] for s in range(5)])
        yy=np.tile(y,5); conf=p.max(1); correct=p.argmax(1)==yy
        xx=[]; acc=[]
        for k in range(10):
            mask=(conf>=k/10)&((conf<(k+1)/10) if k<9 else (conf<=1))
            if mask.sum(): xx.append(conf[mask].mean()); acc.append(correct[mask].mean())
        axes[0].plot(xx,acc,'o-',color=color,label=model)
    axes[0].plot([0,1],[0,1],'--',color='gray'); axes[0].set(xlabel='Mean confidence',ylabel='Observed accuracy',title='Clean reliability pooled across five runs',xlim=(0,1),ylim=(0,1)); axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    matrix=confusion_matrix(y,data['Diagonal VBLL_0_clean'].argmax(1))
    axes[1].imshow(matrix,cmap='Blues'); axes[1].set(xlabel='Predicted digit',ylabel='True digit',title='VBLL confusion matrix for seed 0',xticks=range(10),yticks=range(10))
    for i in range(10):
        for j in range(10):
            if matrix[i,j]: axes[1].text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>20 else 'black')
    fig.savefig(root/'diagnostics.png',dpi=220); plt.close(fig)

if __name__=='__main__': main()
