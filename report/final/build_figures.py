"""Report figures from frozen plot_data_final.npz (built by prepare_update.py); --prepare is the original bundle step.
No fitting. Each output is a new report artifact, separate from teammates' report files.
"""
from pathlib import Path
import argparse, importlib.util, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
COL={'A_ref':'#777777','B':'#D55E00','C':'#0072B2','B+':'#A65A93','C+':'#009E73','E':'#444444','B_nmse':'#E69F00','C_nmse':'#CC79A7','C_ens':'#8B5A00'}

def load_npz(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def prepare():
    data=PROJECT/'work/wake/data';art=PROJECT/'outputs';co=art/'coordination'
    val=load_npz(data/'validation.npz')
    with np.load(data/'train.npz',allow_pickle=False) as f:train_k=f['k'];train_th=f['theta0_deg']
    raw={n:load_npz(co/f'{prefix}_predictions.npz') for n,prefix in [('A_ref','A_official'),('B','B'),('C','C')]}
    keys=lambda d:list(zip(d['case_id'].tolist(),d['win_start'].tolist()))
    for d in raw.values():assert keys(d)==keys(raw['B'])
    idx={str(k):i for i,k in enumerate(val['case_id'])}
    leads={}
    for name,d in raw.items():
        rows=[]
        for w,(case,start) in enumerate(keys(d)):
            ci=idx[case];truth=val['vort'][ci,start:start+20].astype(float)
            denom=max(float(((truth-val['mean_vort'][ci].astype(float))**2).mean()),1e-12)
            rows.append(((d['forecast'][w].astype(float)-truth)**2).mean((1,2))/denom)
        leads[name]=np.array(rows)
    plus=load_npz(art/'C_from_Bplus_s0/diag.npz');pi={k:i for i,k in enumerate(zip(plus['case_id'].tolist(),plus['start'].tolist()))}
    take=[pi[k] for k in keys(raw['B'])];leads['C+']=plus['lead_nmse'][take]
    ensemble=load_npz(co/'reference-ensemble/predictions.npz')
    e=[]
    for w,(case,start) in enumerate(keys(ensemble)):
        ci=idx[case];truth=val['vort'][ci,start:start+20].astype(float)
        denom=max(float(((truth-val['mean_vort'][ci].astype(float))**2).mean()),1e-12)
        e.append(((ensemble['forecast'][w].astype(float)-truth)**2).mean((1,2))/denom)
    leads['E']=np.array(e)
    ordered_cases=[str(k) for k in val['case_id']]
    cases=np.array(raw['B']['case_id'])
    case_a=np.array([np.minimum(leads['A_ref'][cases==c],4).mean() for c in ordered_cases])
    dist=np.abs(case_a-np.median(case_a));ci=int(np.flatnonzero(np.isclose(dist,dist.min(),atol=1e-12,rtol=0))[0])
    case=ordered_cases[ci];wi=int(np.flatnonzero(cases==case)[0]);start=int(raw['B']['win_start'][wi])
    # Qualitative C+ fields are computed on CPU; verify their lead errors against GPU diagnostics.
    import torch
    torch.set_num_threads(1)
    source=art/'Bplus_s0/run_baseline.py'
    spec=importlib.util.spec_from_file_location('report_model',source);rb=importlib.util.module_from_spec(spec);spec.loader.exec_module(rb)
    model=rb.WakeMirrorAvg();model.load_state_dict(torch.load(art/'Bplus_s0/final_model.pt',map_location='cpu',weights_only=True));model.eval()
    k=float(val['k'][ci]);th=float(val['theta0_deg'][ci]);phase=torch.tensor(k,dtype=torch.float32)*torch.tensor(float(val['t_frames'][start]),dtype=torch.float32)
    hist=torch.from_numpy(val['vort'][ci,start-10:start].astype(np.float32))[None]
    const=torch.tensor([val['k'][ci]/(2*math.pi),math.radians(th),torch.sin(phase).item(),torch.cos(phase).item()],dtype=torch.float32)[None,:,None,None]
    x=torch.cat([hist,const.expand(1,4,48,96)],1)
    with torch.no_grad():cp=model(x)[0,:20*48*96].reshape(20,48,96).numpy()
    truth=val['vort'][ci,start:start+20].astype(float);denom=max(float(((truth-val['mean_vort'][ci].astype(float))**2).mean()),1e-12)
    check=((cp.astype(float)-truth)**2).mean((1,2))/denom
    difference=float(np.abs(check-leads['C+'][wi]).max());assert difference<1e-5,difference
    fields=np.stack([truth[[0,19]],raw['B']['forecast'][wi,[0,19]],raw['C']['forecast'][wi,[0,19]],cp[[0,19]]],axis=1)
    # Arrays: two leads x four methods x y x x.
    frozen={'train_k':train_k,'train_theta':train_th,'val_k':val['k'],'val_theta':val['theta0_deg'],'case_id':cases,'case_order':np.array(ordered_cases),'fields':fields}
    frozen.update({f'lead_{n}':v for n,v in leads.items()})
    np.savez_compressed(ROOT/'plot_data.npz',**frozen)
    meta={'selected_case':case,'case_index':ci,'window_index':wi,'window_start':start,'selection_rule':'Nearest median A_ref case error; first case in source order on ties within1e-12; first declared window','k':k,'theta0_deg':th,'Cplus_qualitative_inference':'CPU only, no fitting; compared with saved GPU lead errors','Cplus_max_abs_lead_nmse_difference':difference,'n_cases':12,'n_windows':48,'uncertainty':'Case bootstrap2000resamples, seed0; not seed uncertainty or selection correction','data_source':'https://doi.org/10.5281/zenodo.24386760','input_snapshot':'Verified symmetry results; final submission selection remains pending'}
    (ROOT/'figure_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(color='#DDDDDD',lw=.5,zorder=0)
    ax.tick_params(labelsize=7.5)
    ax.set_axisbelow(True)

def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.04)
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=220,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def paired(a,b):
    """Paired per-case change; 2000 case resamples, default_rng(0), 2.5/97.5% quantiles (unchanged rule)."""
    delta=(a-b).mean(1);r=np.random.default_rng(0);boots=delta[r.integers(len(delta),size=(2000,len(delta)))].mean(1)
    return delta,np.quantile(boots,[.025,.975])

def make():
    d=load_npz(ROOT/'plot_data_final.npz');meta=json.loads((ROOT/'figure_provenance.json').read_text())
    nums=json.loads((ROOT/'report_numbers.json').read_text());val=nums['validation']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'axes.linewidth':.6})
    # Figure 1: (a) where the data and the hidden tiers lie; (b) validation vs portal for each submission.
    portal=json.loads((ROOT/'portal_scores.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.35),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
    ax=axs[0]
    # Hidden tiers (TASK_CARD): only one coordinate of T3a/T3f is published; shading spans the training range of the other.
    kk=np.linspace(1.8,9.4,300)
    ax.fill_between(kk,np.degrees(2.3/kk),np.degrees(3.3/kk),color='#CC79A7',alpha=.18,lw=0,zorder=1)
    ax.add_patch(Rectangle((2.0,38),5.0,7,facecolor='#E69F00',alpha=.22,lw=0,zorder=1))
    ax.add_patch(Rectangle((7.5,5),1.5,30,facecolor='#D55E00',alpha=.18,lw=0,zorder=1))
    ax.scatter(d['train_k'],d['train_theta'],s=3,c='#999999',alpha=.45,label='Train (810)',rasterized=True,zorder=2)
    ax.scatter(d['val_k'],d['val_theta'],s=12,c=COL['C'],edgecolors='white',lw=.3,label='Validation (12)',zorder=3)
    ax.text(4.5,41.5,'T3a: '+r'$\theta_0$ 38–45°',ha='center',va='center',fontsize=6.5,color='#8B5A00')
    ax.text(8.25,20,'T3f:\n$k$ 7.5–9',ha='center',va='center',fontsize=6.5,color='#A0400A')
    ax.text(2.35,33.5,r'T2: $k\theta_0$'+'\n2.3–3.3 rad',ha='left',va='center',fontsize=6.5,color='#8E3E72')
    ax.set(xlabel=r'Reduced frequency $k$',ylabel=r'Amplitude $\theta_0$ [deg]',xlim=(1.8,9.4),ylim=(3,47),title='(a) Data and hidden tiers')
    ax.legend(loc='lower left',frameon=True,facecolor='white',edgecolor='none',framealpha=.9,handletextpad=.2,borderpad=.25,fontsize=6.5,markerscale=1.2);style(ax)
    ax=axs[1];subs=portal['submissions'];b=portal['baseline']
    ax.scatter(b['validation'],b['portal'],marker='*',s=70,color='#444444',zorder=4)
    ax.annotate('organizer baseline',(b['validation'],b['portal']),xytext=(0,-9),textcoords='offset points',fontsize=6,color='#444444',ha='right',va='top')
    conf=[s for s in subs if s['confirmed']]
    for p,q in zip(subs[:-1],subs[1:]):
        ax.annotate('',xy=(q['validation'],q['portal']),xytext=(p['validation'],p['portal']),
                    arrowprops={'arrowstyle':'->','color':'#777777','lw':.8,'shrinkA':4,'shrinkB':4,'ls':'--' if not q['confirmed'] else '-'})
    for s in subs:
        ax.scatter(s['validation'],s['portal'],s=26,zorder=5,facecolor=COL['C'] if s['confirmed'] else 'white',edgecolor=COL['C'],lw=1.1)
    short={8:'averaging only',19:'+ mirror training',52:'+ all windows, shrink',95:'+ period copy',110:'+ soft copy blend',
           120:'+ var.-norm. loss',129:'+ ensemble'}
    off={8:(-6,0,'right','center'),19:(-6,0,'right','center'),52:(5,0,'left','center'),95:(5,2,'left','bottom'),
         110:(5,0,'left','center'),120:(3,-7,'left','top'),129:(-2,-7,'right','top')}
    for s in subs:
        lab=f"#{s['id']} {short[s['id']]}"+('' if s['confirmed'] else '\n(under review)')
        dx,dy,ha,va=off[s['id']]
        ax.annotate(lab,(s['validation'],s['portal']),xytext=(dx,dy),textcoords='offset points',fontsize=6,color='#222222',ha=ha,va=va)
    ax.annotate('',xy=(.205,.215),xytext=(.265,.215),arrowprops={'arrowstyle':'->','color':COL['C'],'lw':1.1})
    ax.text(.235,.222,'training changes\nmove left',fontsize=6.3,color=COL['C'],ha='center',va='bottom')
    ax.annotate('',xy=(.126,.195),xytext=(.126,.265),arrowprops={'arrowstyle':'->','color':COL['C+'],'lw':1.1})
    ax.text(.122,.23,'period copy\nmoves down',fontsize=6.3,color=COL['C+'],ha='right',va='center')
    best=min(conf,key=lambda s:s['portal'])
    gp=(b['portal']-best['portal'])/b['portal'];gv=(b['validation']-val['C_ens']['forecast_nmse'])/b['validation']
    ax.text(.079,.477,f"vs organizer baseline: portal −{gp:.0%} ({b['portal']:.4f}→{best['portal']:.4f})\nvalidation −{gv:.0%} ({b['validation']:.4f}→{val['C_ens']['forecast_nmse']:.4f})",
            ha='left',va='top',fontsize=6,color='#222222',bbox=dict(facecolor='white',edgecolor='#BBBBBB',lw=.5,pad=2))
    ax.set(xlabel='Validation NMSE (public, in-range)',ylabel='Portal score (hidden tiers)',xlim=(.075,.275),ylim=(.09,.485),
           xticks=[.1,.15,.2,.25],yticks=[.2,.3,.4],title='(b) Submissions: validation vs portal')
    style(ax);save(fig,'fig1_setting')
    meta['fig1']={'overall_gain':{'portal':[b['portal'],best['portal'],gp],'validation':[b['validation'],val['C_ens']['forecast_nmse'],gv]}}
    # Figure 2: horizon curves and paired per-case changes.
    cases=d['case_id'];order=d['case_order']
    names=['A_ref','B','C','C+','C_nmse','C_ens','E','A_avg']
    S={n:np.array([np.minimum(d['lead_'+n],4)[cases==c].mean(0) for c in order]) for n in names}
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.15),layout='constrained',gridspec_kw={'width_ratios':[1.15,1]})
    ax=axs[0];t=np.arange(1,21)/10
    endlab={'A_ref':'A ref','C_nmse':r'C$_{\rm nmse}$','C_ens':r'C$_{\rm ens}$'}
    ends=[]
    for name in ['A_ref','B','C','C+','C_nmse','C_ens']:
        m=S[name].mean(0);ends.append((m[-1],name))
        ax.plot(t,m,color=COL[name],lw=2.3 if name=='C_ens' else 1.5,ls='--' if name=='A_ref' else '-',label=endlab.get(name,name))
    ends.sort();ys=[];
    for v,name in ends:                         # direct end labels, nudged apart to avoid overlap
        y=max(v,ys[-1]+.032) if ys else v;ys.append(y)
        ax.text(2.03,y,endlab.get(name,name),color=COL[name],fontsize=6.5,va='center')
    ax.set(xlabel=r'Forecast lead [$c/U$]',ylabel='Clipped forecast NMSE',xlim=(.1,2),ylim=(0,.5),title='(a) Forecast horizon')
    ax.legend(frameon=False,ncol=2,loc='upper left',handlelength=1.7,fontsize=6.5);style(ax)
    ax=axs[1];audits={}
    rows=[('C','B'),('C+','C'),('C_nmse','C+'),('C_ens','C_nmse'),('C','E')]
    lab=lambda n:endlab.get(n,n)
    for y,(first,second) in enumerate(rows):
        delta,ci=paired(S[first],S[second]);color=COL[first] if second!='E' else COL['E']
        jitter=np.linspace(-.13,.13,len(delta));ax.scatter(delta,y+jitter,s=10,color=color,alpha=.6,zorder=3)
        ax.plot(ci,[y,y],color=color,lw=2.4,zorder=4);ax.scatter(delta.mean(),y,marker='D',s=22,facecolor='white',edgecolor=color,zorder=5)
        key=f'{first} - {second}';audits[key]={'mean':float(delta.mean()),'case_bootstrap95':ci.tolist(),'improved_cases':int((delta<0).sum())}
        ref=nums['paired'][key]['forecast_nmse'];assert abs(ref['mean']-delta.mean())<1e-12 and np.allclose(ref['ci95'],ci),key
    ax.axvline(0,color='#555555',lw=.8,ls='--')
    ax.set(yticks=range(len(rows)),yticklabels=[f'{lab(a)} $-$ {lab(b)}' for a,b in rows],xlabel=r'Paired $\Delta$NMSE',xlim=(-.105,.055),ylim=(len(rows)-.6,-.4),title='(b) Paired case changes');style(ax)
    save(fig,'fig2_accuracy');(ROOT/'paired_figure_values.json').write_text(json.dumps(audits,indent=2)+'\n')
    # Figure 3: recorded period-copy evidence (per-window recomputation was not possible: server offline, train.npz not local).
    pb=portal['proxy_blend_frequency'];dp=portal['deployed_k_ge_6.98_training_windows']
    fig,axs=plt.subplots(1,2,figsize=(5.5,1.2),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
    ax=axs[0];labs=['CNN + shrink','copy, hard gate $r<0.3$','copy, soft blend'];v=[pb['CNN + shrink'],pb['hard gate r<0.3'],pb['soft blend (r 0.5-1.2)']]
    ax.barh(range(3),v,color=['#999999',COL['C+'],COL['C+']],alpha=1,height=.6)
    for i,(x,txt) in enumerate(zip(v,['0.689','0.270','0.1905'])):ax.text(x+.01,i,txt,va='center',fontsize=6.5)  # recorded precision
    ax.set(yticks=range(3),yticklabels=labs,xlim=(0,.82),ylim=(2.5,-.5),xlabel='Forecast NMSE',title='(a) Frequency proxy ($k$ 6–7, CNN saw $k\\leq5.5$)')
    ax.tick_params(axis='y',labelsize=6.5);style(ax)
    ax=axs[1];labs=['CNN (C+)','hybrid (copy %d/%d)'%(dp['copy_used'],dp['windows'])];v=[dp['cnn'],dp['hybrid']]
    ax.barh(range(2),v,color=['#999999',COL['C+']],height=.55)
    for i,x in enumerate(v):ax.text(x+.004,i,f'{x:.3f}',va='center',fontsize=6.5)
    ax.set(yticks=range(2),yticklabels=labs,xlim=(0,.2),ylim=(1.5,-.5),xlabel='Forecast NMSE',title='(b) Training windows, $k\\geq6.98$')
    ax.tick_params(axis='y',labelsize=6.5);style(ax)
    save(fig,'fig3_period_copy')
    (ROOT/'figure_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    (ROOT/'figures').mkdir(exist_ok=True)
    if args.prepare:prepare()
    make()
