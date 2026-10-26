import json, sys, warnings
from pathlib import Path
run=Path('/root/autodl-tmp/s-flm/lift-pilot-20261023-01')
sys.path.insert(0,str(run/'source/scripts'))
from analyze_lift_pilot import audit, compare, normalized_ast
warnings.filterwarnings('ignore', category=SyntaxWarning)
rows=[]
for seed in (1,2):
    models={}
    for label in ('base','c200','c711'):
        directory=run/'evaluation'/f'{label}-seed{seed}'
        data=json.loads((directory/'results.json').read_text())
        records=data['records']
        assert len(records)==632
        audits=[audit(r,i) for i,r in enumerate(records)]
        flags=[a['status']=='correct' for a in audits]
        assert sum(flags)==data['num_correct']
        models[label]={'records':records,'flags':flags,'asts':[normalized_ast(r['response']) for r in records]}
        if label=='c711':
            (directory/'execution_audit.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in audits))
        assert [(r['prompt'],r['response_ground_truth']) for r in records]==[(r['prompt'],r['response_ground_truth']) for r in models['base']['records']]
    rows.append({'seed':seed,'correct':sum(models['c711']['flags']),'total':632,'accuracy':sum(models['c711']['flags'])/632,'vs_base':compare(models['base'],models['c711']),'vs_c200':compare(models['c200'],models['c711'])})
summary={'protocol':{'split':'validation','questions':632,'seeds':[1,2],'weights':'EMA','sampler':'ancestral','steps':32,'temperature':0.1},'training':json.loads((run/'train-c-711/metrics/version_0/summary.json').read_text()),'rows':rows,'mean_accuracy':sum(r['correct'] for r in rows)/1264,'limitations':['One training seed; generation seeds repeat the same validation questions.','No A/B control continued to 711 steps.','Resume restores Lightning states; exact sample replay has not been verified.']}
(run/'c711_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
