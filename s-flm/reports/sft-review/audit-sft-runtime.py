import hashlib,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'/root/s-flm')
import torch,pyarrow.parquet as pq
from hydra import initialize_config_dir,compose
import dataloader
with initialize_config_dir(version_base=None,config_dir='/root/s-flm/configs'):
 c=compose(config_name='sft',overrides=['wandb=null','data.train_file=/root/s-flm/data/gsm8k_pot/sft_train.jsonl','data.validation_file=/root/s-flm/data/gsm8k_pot/sft_validation.jsonl','data.cache_dir=/root/s-flm/data_cache/gsm8k_pot'])
tok=dataloader.get_tokenizer(c)
sources=json.loads(Path('/root/s-flm/data/gsm8k_pot/split_manifest.json').read_text())['upstream_sources']['openai/gsm8k']['files']
original={}
for split in ['train','test']:
 f=sources['main/'+split+'-00000-of-00001.parquet']['path']
 original[split]=pq.read_table(f).to_pylist()
test_questions={' '.join(r['question'].split()).casefold() for r in original['test']}
result={'tokenizer_size':len(tok),'separator':repr(c.data.separator),'splits':{}}
all_questions=[]
for split,mode in [('train','train'),('validation','valid')]:
 rows=[json.loads(line) for line in Path('/root/s-flm/data/gsm8k_pot/sft_'+split+'.jsonl').read_text().splitlines()]
 ds=dataloader.get_gsm8k_pot_dataset(c,tok,mode)
 lengths=[];pad=targets=0;ids=set();questions=set()
 for r,x in zip(rows,ds):
  q=' '.join(r['question'].split()).casefold();questions.add(q);ids.add(r['gsm8k_train_id'])
  assert q not in test_questions
  gold=original['train'][r['gsm8k_train_id']]
  assert r['question']==gold['question'] and r['reference_answer']==gold['answer']
  prefix=[tok.bos_token_id]+tok(r['question'].strip(),add_special_tokens=False).input_ids+tok(c.data.separator,add_special_tokens=False).input_ids
  seq=prefix+tok(r['code'].strip(),add_special_tokens=False).input_ids+[tok.eos_token_id]
  n=len(seq);assert n==r['unpadded_total_tokens'] and n<=512
  assert x['input_ids'].tolist()==seq+[tok.pad_token_id]*(512-n)
  assert x['attention_mask'].tolist()==[0]*len(prefix)+[1]*(512-len(prefix))
  lengths.append(n);pad+=512-n;targets+=512-len(prefix)
 result['splits'][split]={'rows':len(rows),'tokens_min':min(lengths),'tokens_max':max(lengths),'tokens_mean':statistics.mean(lengths),'pad_fraction_of_supervision':pad/targets,'supervised_tokens':targets,'padding_tokens':pad}
 all_questions.append((ids,questions))
assert not all_questions[0][0]&all_questions[1][0]
assert not all_questions[0][1]&all_questions[1][1]
p=Path('/root/s-flm/checkpoints/tinygsm/mdlm.ckpt')
ckpt=torch.load(p,map_location='cpu',weights_only=False)
b=ckpt['hyper_parameters']['config']
result['base']={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'global_step':ckpt['global_step'],'ema_updates':ckpt['ema']['num_updates'],'separator':repr(b.data.separator),'train_on_prompt':b.data.train_on_prompt,'train_on_pad':b.data.train_on_pad}
assert b.data.separator==c.data.separator and b.data.train_on_prompt==c.data.train_on_prompt and b.data.train_on_pad==c.data.train_on_pad
for key in ['hidden_size','n_blocks','n_heads','length']:
 assert b.model[key]==c.model[key],key
result['cross_split_overlap']=0;result['gsm8k_test_question_overlap']=0
Path('/root/setup-logs/sft-data-review.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
