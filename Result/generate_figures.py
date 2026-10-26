#!/usr/bin/env python3
"""Rebuild figures/tables from archived evidence. Run from any directory."""
from pathlib import Path
import csv, json, hashlib, os, textwrap
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/dllm-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/dllm-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'Result'
R=ROOT/'s-flm/reports'; B=ROOT/'s-flm/experiment_results/rl-evidence-20261025/outputs'
for d in ['figures','data','tables']: (OUT/d).mkdir(exist_ok=True)
SOURCES=set(); BLUE='#2878B5'; ORANGE='#E58B36'; GREEN='#369984'; GRAY='#757C88'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def load(p):
 SOURCES.add(p); return json.loads(p.read_text())
def lines(p):
 SOURCES.add(p);return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def readcsv(p):
 SOURCES.add(p);return list(csv.DictReader(p.open()))
def export(name,rows):
 rows=list(rows)
 with (OUT/'data'/f'{name}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,name,note):
 wrapped='\n'.join('\n'.join(textwrap.wrap(line,width=int(fig.get_figwidth()*17))) for line in note.splitlines())
 fig.text(.015,.018,wrapped,fontsize=8,color='#555555',va='bottom')
 footer=(len(wrapped.splitlines())*10+12)/(fig.get_figheight()*72)
 fig.tight_layout(rect=(0,footer,1,1))
 for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=300)
 plt.close(fig)
def ci(x):
 x=np.asarray(x);rng=np.random.default_rng(20261026)
 return np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1),[.025,.975])
# Fig1: actual per-batch training objective; sparse epoch NLL is NOT interpolated.
f=R.parent/'checkpoints/sft/20261022-base-aligned-01/metadata/metrics/version_0/metrics.csv'
m=readcsv(f);loss=[dict(update=int(x['step'])+1,loss=float(x['trainer/loss'])) for x in m if x.get('trainer/loss')]
acc=load(R.parent/'eval_runs/gsm8k-sft-validation/20261022-base-aligned-validation/milestone_summary.json')
export('fig01_training_loss',loss);export('fig01_accuracy',[dict(step=x['step'],correct=x['num_correct'],total=x['num_total'],accuracy_pct=x['accuracy']*100) for x in acc])
export('fig01_sparse_nll',[dict(update=int(x['step'])+1,metric=k,value=float(x[k])) for x in m for k in ['train/nll','val/nll','val/code_eos_nll'] if x.get(k)])
fig,ax=plt.subplots(1,2,figsize=(10,3.7));ax[0].plot([x['update'] for x in loss],[x['loss'] for x in loss],color=BLUE,lw=1.2)
ax[0].set(title='(a) Full SFT training objective',xlabel='Optimizer updates',ylabel='Logged batch loss')
ax[1].plot([x['step'] for x in acc],[x['accuracy']*100 for x in acc],'o-',color=ORANGE)
for x in acc:ax[1].annotate(f"{x['num_correct']}/632",(x['step'],x['accuracy']*100),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
ax[1].set(title='(b) Validation generation',xlabel='Optimizer updates',ylabel='Execution accuracy (%)',ylim=(0,10));ax[1].set_xticks([0,100,200,711])
save(fig,'fig01_sft_loss_accuracy','Same Full SFT run; EMA, T=1, 32 NFE, seed 1, 632 validation questions. Batch loss is not epoch NLL.')
# Fig2: same BF16 random-mask overfit run, no FP32 code-only mixing.
s=load(R/'overfit16-20261023-01/summary.json');d=load(R/'overfit16-20261023-01/mask-diagnostic/results.json')
export('fig02_overfit',[dict(step=x['step'],code_eos_nll=x['code_eos_nll'],correct=x['correct'],questions=16) for x in s])
maskrows=[]
for step,v in d['models'].items():
 for x in v['mask_tests']:
  z=x['aggregate']['non_docstring_code_eos'];maskrows.append(dict(step=int(step),mask_ratio=x['rate'],ce=z['ce'],accuracy=z['accuracy'],tokens=z['tokens']))
export('fig02_mask_diagnosis',maskrows)
fig,ax=plt.subplots(1,2,figsize=(10,3.9));t=ax[0].twinx();t.spines['right'].set_visible(True)
ax[0].plot([x['step'] for x in s],[x['code_eos_nll'] for x in s],'o-',color=BLUE,label='Fixed-noise code/EOS NLL')
t.plot([x['step'] for x in s],[x['correct'] for x in s],'s--',color=ORANGE,label='Execution correct')
ax[0].set(title='(a) Fixed 16-question overfitting',xlabel='Optimizer updates',ylabel='Code/EOS NLL');t.set(ylabel='Correct questions / 16',ylim=(0,16));t.set_yticks([0,4,8,12,16]);t.grid(False)
a,b=ax[0].get_legend_handles_labels();c,e=t.get_legend_handles_labels();ax[0].legend(a+c,b+e,fontsize=8,loc='upper right')
for step,color in zip(sorted({r['step'] for r in maskrows}),[GRAY,BLUE,ORANGE]):
 z=[r for r in maskrows if r['step']==step];ax[1].plot([r['mask_ratio']*100 for r in z],[r['ce'] for r in z],'o-',color=color,label='Base' if step==0 else f'Step {step}')
ax[1].set(title='(b) Fixed-MASK reconstruction',xlabel='Answer MASK ratio (%)',ylabel='Non-docstring code/EOS token CE');ax[1].legend()
save(fig,'fig02_sft_overfit_mask','Same 16 training questions. Generation: raw SFT, seed 1, 128 NFE. Right: fixed masks; PAD excluded.')
# Fig3/Table2: legacy full-output protocol throughout, independent 500-update run.
g=[]
for step in [100,250,500]:
 z=load(B/f'grpo-500-20261024-01/validation-{step}/comparison.json');v=[x for x in z['comparisons'] if x['protocol']=='legacy-full'];v.sort(key=lambda x:x['seed'])
 if step==100:g.append(dict(step=0,seed1=v[0]['base_correct'],seed2=v[1]['base_correct']))
 g.append(dict(step=step,seed1=v[0]['rl_correct'],seed2=v[1]['rl_correct']))
for x in g:x['accuracy_pct']=(x['seed1']+x['seed2'])/1264*100;x['delta_pp']=x['accuracy_pct']-g[0]['accuracy_pct']
export('fig03_grpo_validation',g)
fig,ax=plt.subplots(figsize=(6.3,3.7));ax.axhline(g[0]['accuracy_pct'],color=GRAY,ls='--',label='Base EMA')
ax.plot([x['step'] for x in g],[x['accuracy_pct'] for x in g],'o-',color=BLUE,label='Mean of generation seeds 1 / 2')
for x in g:ax.annotate(f"{x['accuracy_pct']:.2f}%",(x['step'],x['accuracy_pct']),xytext=(0,9),textcoords='offset points',ha='center')
ax.set(xlabel='Optimizer updates',ylabel='Validation execution accuracy (%)',title='Online full-MASK GRPO',ylim=(26.3,27.35));ax.set_xticks([0,100,250,500]);ax.legend(fontsize=8,loc='lower left')
save(fig,'fig03_grpo_training_performance','632 validation questions; T=0.1, 32 NFE, two generation seeds. Historical full-output scoring; one training seed.')
# Fig4: all-answer success only, matching stated values; question bootstrap CIs archived.
d=load(B/'grpo-partial-mask-20261024-01/comparison.json');rr=[x for x in d['comparisons'] if x['reward']==1 and x['region']=='all_answer'];rr.sort(key=lambda x:-x['rate'])
f4=[dict(mask_ratio=x['rate'],delta=x['mean_logp_delta'],ci_low=x['mean_logp_ci95'][0],ci_high=x['mean_logp_ci95'][1],questions=x['questions']) for x in rr];export('fig04_proxy_transfer',f4)
fig,ax=plt.subplots(figsize=(6.5,3.9));y=np.array([x['delta'] for x in f4]);lo=np.array([x['ci_low'] for x in f4]);hi=np.array([x['ci_high'] for x in f4])
ax.bar(range(3),y,color=[BLUE,ORANGE,GREEN],width=.55);ax.errorbar(range(3),y,yerr=[y-lo,hi-y],fmt='none',color='#333333',capsize=5);ax.axhline(0,color=GRAY,lw=1)
ax.set_xticks(range(3),[f"{x['mask_ratio']*100:.0f}%" for x in f4]);ax.set(xlabel='Answer MASK ratio',ylabel='Change in mean token logP (RL - Base)',title='Full-MASK proxy transfer to partial MASK')
for i,x in enumerate(f4):ax.annotate(f"{x['delta']:+.6f}",(i,x['ci_high']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
ax.margins(y=.25);save(fig,'fig04_mask_proxy_transfer','32 fixed successful programs; FP32. All-answer region. Error bars: archived question-bootstrap 95% CI; teacher-forced masks.')
# Fig5: heldout means across four shared mask plans.
f5=[]
for name,label in [('grpo-partial75-overfit-20261024-01','Signed'),('grpo-partial75-positive-20261024-01','Positive-only')]:
 d=load(B/name/'summary.json')
 for reward,key in [(1,'success_mean_logp'),(0,'failure_mean_logp')]:
  delta=np.mean([x[key] for x in d['final']['heldout']])-np.mean([x[key] for x in d['initial']['heldout']]);f5.append(dict(objective=label,program='Success' if reward else 'Failure',delta=float(delta),updates=200))
export('fig05_signed_positive',f5)
fig,ax=plt.subplots(figsize=(6.5,3.8));xs=np.arange(2)
for j,(label,color) in enumerate([('Success',BLUE),('Failure',ORANGE)]):
 vals=[x['delta'] for x in f5 if x['program']==label];bars=ax.bar(xs+(j-.5)*.3,vals,width=.3,label=f'{label} program',color=color);ax.bar_label(bars,fmt='%+.4f',padding=4,fontsize=9)
ax.set_xticks(xs,['Signed','Positive-only']);ax.axhline(0,color=GRAY,lw=1);ax.set(title='Learning on independent 75% MASK states',ylabel='Change in mean token logP',ylim=(-.29,.065));ax.legend(loc='lower right',fontsize=9)
save(fig,'fig05_signed_vs_positive','Same 8 questions, program pairs, initialization and masks; 200 updates. Four independent MASK plans; no KL / clipping.')
# Fig6: cluster questions, preserving all 32 draws within question.
d=load(B/'grpo-positive-coverage-20261024-01/all_batches.json')['combined'];comp=load(B/'grpo-positive-coverage-20261024-01/comparison.json')['combined'];f6=[];pq=[]
for temp in ['0.1','0.3']:
 for model in ['base','positive8','positive32']:
  s=d[model][temp];v=[x['success_probability'] for x in s['per_question']];bounds=ci(v)
  f6.append(dict(model=model,temperature=float(temp),accuracy_pct=np.mean(v)*100,ci_low_pct=bounds[0]*100,ci_high_pct=bounds[1]*100,questions=len(v),draws_per_question=32))
  pq.extend(dict(model=model,temperature=temp,**x) for x in s['per_question'])
export('fig06_positive_accuracy',f6);export('fig06_per_question',pq)
export('fig06_paired_differences',[dict(comparison=k,temperature=t,delta_pp=v['mean_delta']*100,ci_low_pp=v['question_bootstrap_ci95'][0]*100,ci_high_pp=v['question_bootstrap_ci95'][1]*100) for k,ts in comp.items() for t,v in ts.items()])
fig,ax=plt.subplots(figsize=(7.3,4.2));xs=np.arange(3)
for j,(temp,col) in enumerate([(0.1,BLUE),(0.3,ORANGE)]):
 z=[x for x in f6 if x['temperature']==temp];y=np.array([x['accuracy_pct'] for x in z]);lo=np.array([x['ci_low_pct'] for x in z]);hi=np.array([x['ci_high_pct'] for x in z]);pos=xs+(j-.5)*.34
 ax.bar(pos,y,width=.32,color=col,label=f'T={temp}');ax.errorbar(pos,y,yerr=[y-lo,hi-y],fmt='none',capsize=3,color='#333333')
 for p,v in zip(pos,y):ax.text(p,v+.4,f'{v:.2f}',ha='center',fontsize=8)
ax.set_xticks(xs,['Base','Positive 8','Positive 32']);ax.set(ylabel='Mean single-draw success (%)',title='Positive learning: 128-question development set',ylim=(0,40));ax.legend(loc='upper left',fontsize=9)
text='; '.join(f"T={t}: {comp['base-to-positive32'][t]['mean_delta']*100:+.2f} pp [{comp['base-to-positive32'][t]['question_bootstrap_ci95'][0]*100:+.2f}, {comp['base-to-positive32'][t]['question_bootstrap_ci95'][1]*100:+.2f}]" for t in ['0.1','0.3'])
save(fig,'fig06_positive_learning_performance','Two sampling batches, 32 draws/question; question-bootstrap 95% CIs.\nPaired Positive 32 - Base: '+text)
# Fig7: same machine/same 632 questions, 16 draws/question.
d=load(B/'grpo-positive128-20261025-01/validation/comparison.json');f7=[dict(temperature=float(t),delta_pp=d[t]['mean_delta']*100,ci_low_pp=d[t]['question_bootstrap_ci95'][0]*100,ci_high_pp=d[t]['question_bootstrap_ci95'][1]*100) for t in ['0.1','0.3']];export('fig07_scaling',f7)
fig,ax=plt.subplots(figsize=(6.5,3.8));y=np.array([x['delta_pp'] for x in f7]);lo=np.array([x['ci_low_pp'] for x in f7]);hi=np.array([x['ci_high_pp'] for x in f7]);ax.errorbar(range(2),y,yerr=[y-lo,hi-y],fmt='o',ms=8,color=BLUE,capsize=6);ax.axhline(0,color=GRAY,ls='--');ax.set_xticks(range(2),['T=0.1','T=0.3']);ax.set(xlim=(-.5,1.5),ylabel='Accuracy difference (percentage points)',title='Scaling: Positive 128 - Positive 32')
for i,x in enumerate(f7):ax.annotate(f"{x['delta_pp']:+.3f} pp",(i,x['delta_pp']),xytext=(12,0),textcoords='offset points',va='center')
save(fig,'fig07_positive_learning_scaling','632 development questions; 16 draws/question/temperature, FP32, A800, 32 NFE. Paired question-bootstrap 95% CI.')
# Appendix A1: already archived learning curves, no additional experiment.
fig,ax=plt.subplots(1,2,figsize=(10,3.8));appendix=[]
for name,label,color in [('grpo-partial75-positive-20261024-01','Positive 8',GRAY),('grpo-positive32-20261024-01/train','Positive 32',BLUE),('grpo-positive128-20261025-01/train','Positive 128',ORANGE)]:
 scores=lines(B/name/'scores.jsonl');updates=lines(B/name/'updates.jsonl')
 for x in updates:appendix.append(dict(model=label,step=x['step'],metric='loss_before_update',value=x['loss_before_update']))
 ax[0].plot([x['step'] for x in updates],[x['loss_before_update'] for x in updates],label=label,color=color)
 held=[]
 for x in scores:
  value=np.mean([h['success_mean_logp'] for h in x['heldout']]) if 'heldout' in x else np.mean([h['mean_logp'] for h in x['mask_scores'][1:]]);held.append(value);appendix.append(dict(model=label,step=x['step'],metric='heldout_success_logp',value=value))
 ax[1].plot([x['step'] for x in scores],held,'o-',label=label,color=color)
ax[0].set(title='(a) Positive-only training loss',xlabel='Optimizer updates',ylabel='Loss before update');ax[1].set(title='(b) Independent-MASK success score',xlabel='Optimizer updates',ylabel='Mean token logP');ax[0].legend(fontsize=8);ax[1].legend(fontsize=8);export('appendix_a1_positive_training',appendix)
save(fig,'appendix_a1_positive_training','75% fixed MASK, FP32; different training program sets and compute budgets. Independent MASKs are not unseen questions.')
# Markdown result tables: all values extracted from machine-readable evidence.
def mdtable(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)
parts=['# 实验结果表\n\n由 `generate_figures.py` 从本地证据生成。百分比保留两位；pp为百分点。不同协议分组展示，不做跨组排名。\n']
parts.append('## Table 1 — SFT Methods Comparison\n\n### A. 历史T=1：632题、EMA、32 NFE、生成seed 1\n')
rows=[]
for dirname,label in [('20261022-base-aligned-validation','Full SFT'),('20261022-lora-control-validation','LoRA')]:
 p=R.parent/'eval_runs/gsm8k-sft-validation'/dirname/'milestone_summary.json'
 if not p.exists() and label=='LoRA':
  matches=list((R.parent/'eval_runs/gsm8k-sft-validation').glob('*lora*control*/milestone_summary.json'));p=matches[0] if matches else p
 if p.exists():
  for x in load(p):rows.append([label if x['step'] else 'Base ('+label+'对照)',x['step'],f"{x['num_correct']}/632",f"{x['accuracy']*100:.2f}%"])
parts.append(mdtable(['模型','更新数','正确数','Accuracy'],rows))
parts.append('\n### B. Blackwell低温复评：632题、T=0.1、EMA、32 NFE、生成seed 1\n')
d=load(R/'lowtemp-sft-comparison-20261023-01/comparison.json');names={'base':'Base','uniform200':'LoRA 200','weighted200':'Semantic Weighted 200','weighted711':'Semantic Weighted 711'}
parts.append(mdtable(['模型','正确数','Accuracy'],[[names[x['model']],f"{x['correct']}/632",f"{x['accuracy']*100:.2f}%"] for x in d['rows']]))
parts.append('\n### C. 同题190题SFT：632题、T=0.1、EMA、32 NFE、生成seeds 1/2\n')
d=load(R/'selftrain-pilot-20261023-01/comparison.json');parts.append(mdtable(['模型','Seed 1','Seed 2','平均Accuracy'],[[{'base':'Base','original':'同题原PoT SFT','generated':'Self-generated SFT'}.get(k,k),*v['correct_by_seed'],f"{v['mean_accuracy']*100:.2f}%"] for k,v in d['means'].items()]))
parts.append('\n### D. A800 LIFT对照：632题、T=0.1、EMA、32 NFE、生成seeds 1/2\n')
d=load(R/'lift-pilot-20261023-01/comparison.json');parts.append(mdtable(['模型','Seed 1','Seed 2','平均Accuracy'],[[k,*v['correct_by_seed'],f"{v['mean_accuracy']*100:.2f}%"] for k,v in d['means'].items()]))
d=load(R/'lift-pilot-20261023-01/c711_summary.json');parts.append(mdtable(['模型','Seed 1','Seed 2','平均Accuracy'],[['LIFT C711',*[x['correct'] for x in d['rows']],f"{d['mean_accuracy']*100:.2f}%"]]))
parts.append('\na=原随机MASK，b=随机恢复，c=LIFT-3。各组Base是各次同机对照。历史Full SFT没有此表所需的低温完整节点比较，不补填。\n')
parts.append('\n## Table 2 — Online GRPO Results\n\n同一次500步独立运行；632题，T=0.1、32 NFE，历史完整输出评分。两个生成seed重复同一题集，只有一个训练seed。\n')
parts.append(mdtable(['模型','Seed 1 /632','Seed 2 /632','平均Accuracy','相对Base (pp)'],[['Base' if x['step']==0 else f"RL{x['step']}",x['seed1'],x['seed2'],f"{x['accuracy_pct']:.2f}%",f"{x['delta_pp']:+.2f}"] for x in g]))
# Final Test: validate complete evidence before populating the benchmark table.
F=OUT/'final-test-20261026-01'
assert (F/'COMPLETE').exists() and not (F/'FAILED').exists(), 'Final Test is not complete'
protocol=load(F/'protocol.json')
assert protocol['questions']==1319 and protocol['seeds']==[1,2] and protocol['temperature']==.1
assert protocol['nfe']==32 and protocol['batch_size']==8 and protocol['precision']=='float32'
labels={'base_ema':'Base EMA','sft711_ema':'SFT711 EMA','positive32_raw':'Positive 32 raw'}
final_rows=[];final_values={};per_question=[]
from collections import Counter
for model,label in labels.items():
 draws=[];counts=[]
 for seed in [1,2]:
  rs=lines(F/f'{model}-seed{seed}.jsonl')
  assert len(rs)==1319 and [x['sample_id'] for x in rs]==[f'gsm8k_test_{i:04d}' for i in range(1319)]
  assert all(x['seed']==seed and x['reward'] in [0,1] and not x['execution'].get('infrastructure_failure') for x in rs)
  counts.append(sum(x['reward'] for x in rs));draws.append([x['reward'] for x in rs])
  per_question.extend(dict(model=model,seed=seed,sample_id=x['sample_id'],reward=x['reward'],raw_correct=x['execution']['raw']['correct'],retries=x['infrastructure_retries']) for x in rs)
 values=np.mean(draws,axis=0);final_values[model]=values;bounds=ci(values)*100
 final_rows.append(dict(model=label,seed1_correct=int(counts[0]),seed2_correct=int(counts[1]),questions=1319,accuracy_pct=float(values.mean()*100),ci_low_pct=float(bounds[0]),ci_high_pct=float(bounds[1])))
manifest=load(F/'pre-resume-results-manifest.json')
for name,m in manifest.items():assert hashlib.sha256((F/name).read_bytes()[:m['bytes']]).hexdigest()==m['sha256']
audits=[load(F/f'{model}-sampler-audit.json') for model in labels];assert all(x==audits[0] for x in audits)
assert all(load(F/f'{model}-precision.json')['backbone_output_dtype']=='torch.float32' for model in labels)
comparison=load(F/'comparison.json');paired=[]
for key,v in comparison.items():
 z=v['combined'];paired.append(dict(comparison=key,delta_pp=z['mean_delta']*100,ci_low_pp=z['question_bootstrap_ci95'][0]*100,ci_high_pp=z['question_bootstrap_ci95'][1]*100))
export('final_test_accuracy',final_rows);export('final_test_paired_differences',paired);export('final_test_per_question',per_question)
(OUT/'data/final_test_verification.json').write_text(json.dumps(dict(records=len(per_question),unique_model_seed_question_keys=len({(x['model'],x['seed'],x['sample_id']) for x in per_question}),completed=True,previous_results_hashes_preserved=True,sampler_audits_equal=True,actual_backbone_dtype='torch.float32',infrastructure_failures=0,absolute_ci_bootstrap_replicates=10000,absolute_ci_seed=20261026,paired_ci_source='archived comparison.json',resume_note='original infinite-loop result handled with unchanged 2-second per-execution timeout'),indent=2))
parts.append('\n## Table 3 — Final GSM8K Test Comparison\n\n统一协议：同一A800、1319题、T=0.1、32 NFE、FP32、seeds 1/2、batch 8。共7914条有效逐题记录。Base和SFT使用EMA，Positive32使用raw；Test有历史探索性使用，并非完全未触碰测试集。\n')
parts.append(mdtable(['模型','Seed 1 /1319','Seed 2 /1319','平均Accuracy','按题Bootstrap 95% CI'],[[x['model'],x['seed1_correct'],x['seed2_correct'],f"{x['accuracy_pct']:.2f}%",f"[{x['ci_low_pct']:.2f}%, {x['ci_high_pct']:.2f}%]"] for x in final_rows]))
parts.append('\n两个生成seed在每道题内取均值，再按1319道题聚类Bootstrap；不把2638次生成视为2638道独立题。单模型区间：10000次重采样，NumPy seed=20261026；配对差值区间直接使用归档结果。\n')
parts.append(mdtable(['比较（后者−前者）','差值 (pp)','按题配对95% CI (pp)'],[[x['comparison'],f"{x['delta_pp']:+.3f}",f"[{x['ci_low_pp']:+.3f}, {x['ci_high_pp']:+.3f}]"] for x in paired]))
parts.append('\nPositive32平均正确率最高，但相对Base的配对区间包含零；两个seed分别相对Base少6题、多19题，尚不能认定稳定优于Base。SFT也未显示收益。续评前记录哈希保持不变；最终结果无基础设施失败。第259题死循环经修复后正常计为模型执行失败，并非删除该题或放宽单次程序限时。\n')
# Extra appendix visualization; seven core figures remain unchanged.
fig,ax=plt.subplots(1,2,figsize=(10.6,4.2))
y=np.array([x['accuracy_pct'] for x in final_rows]);lo=np.array([x['ci_low_pct'] for x in final_rows]);hi=np.array([x['ci_high_pct'] for x in final_rows])
ax[0].bar(range(3),y,color=[GRAY,BLUE,GREEN],width=.6);ax[0].errorbar(range(3),y,yerr=[y-lo,hi-y],fmt='none',color='#333333',capsize=4)
for i,v in enumerate(y):ax[0].text(i,v+.35,f'{v:.2f}%',ha='center',fontsize=9)
ax[0].set_xticks(range(3),['Base EMA','SFT711 EMA','Positive 32']);ax[0].set(title='(a) Full GSM8K Test',ylabel='Mean execution accuracy (%)',ylim=(0,30))
z=paired[:2];y=np.array([x['delta_pp'] for x in z]);lo=np.array([x['ci_low_pp'] for x in z]);hi=np.array([x['ci_high_pp'] for x in z]);ax[1].errorbar(range(2),y,yerr=[y-lo,hi-y],fmt='o',capsize=5,color=BLUE);ax[1].axhline(0,color=GRAY,ls='--');ax[1].set_xticks(range(2),['SFT711 - Base','Positive 32 - Base']);ax[1].set(title='(b) Paired difference',ylabel='Accuracy difference (percentage points)',xlim=(-.5,1.5))
for i,v in enumerate(y):ax[1].annotate(f'{v:+.3f} pp',(i,v),xytext=(8,0),textcoords='offset points',va='center',fontsize=9)
save(fig,'appendix_a2_final_test_comparison','1319 questions; T=0.1, FP32, 32 NFE, A800, generation seeds 1/2. Question-bootstrap 95% CIs. Test has prior exploratory use.')
parts.append('\n## Table 4 — Computational Cost\n\n训练batch计时、训练含验证wall计时均为日志定义，不含模型下载和全部生成评估。显存为PyTorch peak allocated，非nvidia-smi总占用。未知项保留“未记录”。\n')
cost=[]
configs=[('Full SFT','A800',R.parent/'checkpoints/sft/20261022-base-aligned-01/metadata/metrics/version_0'),('Semantic Weighted','RTX PRO 6000 Blackwell',R/'semantic-weighted-20261023-01/train/metrics/version_0'),('Self-generated SFT','RTX PRO 6000 Blackwell',R/'selftrain-pilot-20261023-01/train-generated/metrics/version_0'),('同题原PoT','RTX PRO 6000 Blackwell',R/'selftrain-pilot-20261023-01/train-original/metrics/version_0'),('LIFT A200','A800',R/'lift-pilot-20261023-01/train-a/metrics/version_0'),('LIFT B200','A800',R/'lift-pilot-20261023-01/train-b/metrics/version_0'),('LIFT C200','A800',R/'lift-pilot-20261023-01/train-c/metrics/version_0'),('LIFT C200→711续训','A800',R/'lift-pilot-20261023-01/train-c-711/metrics/version_0')]
for label,gpu,p in configs:
 s=load(p/'summary.json');m=readcsv(p/'metrics.csv');peaks=[float(x['monitor/peak_allocated_mib']) for x in m if x.get('monitor/peak_allocated_mib')]
 cost.append([label,gpu,s['successful_updates_session'],f"{s['train_batch_seconds']:.2f}",f"{s['wall_seconds_including_validation']:.2f}",f'{max(peaks):.1f}' if peaks else '未记录','未统一记录'])
for label,gpu,updates in [('Online GRPO','RTX PRO 6000 Blackwell',500),('Positive 8','RTX PRO 6000 Blackwell',200),('Positive 32','RTX PRO 6000 Blackwell',200),('Positive 128','A800',200)]:cost.append([label,gpu,updates,'未记录于所选结构化摘要','未记录于所选结构化摘要','未记录于所选结构化摘要','另见历史流水线日志，未统一计时'])
parts.append(mdtable(['运行','GPU','本session有效更新','训练batch累计(s)','训练含验证wall(s)','峰值allocated(MiB)','生成/评估时间'],cost))
parts.append('\n### 最终Test生成及执行评估开销\n')
evalcost=[]
for model,filename,scope in [('base_ema','base_ema-cost.json','完整双seed，2638条'),('sft711_ema','sft711_ema-cost.json','完整双seed，2638条'),('positive32_raw','positive32_raw-resume-cost.json','仅续评session，新增1061条；含恢复批次重生成')]:
 c=load(F/filename);evalcost.append([labels[model],f"{c['evaluation_seconds']:.2f}",f"{c['evaluation_seconds']/60:.2f}",f"{c['peak_allocated_bytes']/2**20:.1f}",f"{c['peak_reserved_bytes']/2**20:.1f}",scope])
parts.append(mdtable(['模型','评估耗时(s)','评估耗时(min)','峰值allocated(MiB)','峰值reserved(MiB)','统计范围'],evalcost))
parts.append('\n评估耗时含该计时区间内的生成及执行评分，不含模型加载。Positive32仅记录恢复session开销，不能与另外两行作完整评估速度对比，也不能据其重建完整运行峰值。原Positive32中断前精确总耗时未记录。RL流水线其他耗时不以文件mtime估算；硬件与题池不同，不据此作算法加速排名。\n')
table_text='\n'.join(parts)
(OUT/'tables/results_tables.md').write_text(table_text)
for piece in table_text.split('\n## Table ')[1:]:
 number=piece.split(' ',1)[0];(OUT/'tables'/f'table_{number}.md').write_text('## Table '+piece)
# Hash exact inputs for reproducibility.
(OUT/'data/source_manifest.json').write_text(json.dumps([dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(SOURCES)],indent=2))
# Contact sheet is regenerated together with figures and tables.
from PIL import Image, ImageDraw
preview_files=sorted((OUT/'figures').glob('*.png'))
canvas=Image.new('RGB',(2100,900),'#eeeeee');draw=ImageDraw.Draw(canvas)
for i,path in enumerate(preview_files):
 im=Image.open(path);im.thumbnail((690,275));x=i%3*700;y=i//3*300
 canvas.paste(im,(x,y+20));draw.text((x+8,y+3),path.stem,fill='black')
canvas.save(OUT/'preview.png')
print('Generated 7 core figures + 2 appendix figures, CSV data and 4 Markdown tables (final Test complete).')
