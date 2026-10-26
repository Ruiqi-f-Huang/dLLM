import json,hashlib
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
out=Path('/root/autodl-tmp/s-flm/outputs/mathinstruct-audit'); api=HfApi(); manifest={}
for repo,files in [('TIGER-Lab/MathInstruct',['MathInstruct.json']),('openai/gsm8k',['main/train-00000-of-00001.parquet','main/test-00000-of-00001.parquet']),('TinyGSM/TinyGSM',['data/train-00000-of-00017.parquet'])]:
 info=api.dataset_info(repo,files_metadata=True)
 manifest[repo]={'revision':info.sha,'files':{}}
 for f in files:
  path=hf_hub_download(repo,f,repo_type='dataset',revision=info.sha)
  meta=next(x for x in info.siblings if x.rfilename==f)
  h=hashlib.sha256(Path(path).read_bytes()).hexdigest()
  expected=meta.lfs.sha256 if meta.lfs else None
  assert expected is None or h==expected
  manifest[repo]['files'][f]={'path':path,'sha256':h,'size':Path(path).stat().st_size}
  (out/'sources.json').write_text(json.dumps(manifest,indent=2))
  print('Downloaded and verified',repo,f,flush=True)
print('DOWNLOAD COMPLETE',flush=True)
