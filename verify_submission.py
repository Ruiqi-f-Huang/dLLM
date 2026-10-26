"""Verify submission file hashes without GPU or third-party packages."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'SUBMISSION_MANIFEST.json').read_text())
    failures=[]
    for rel,item in manifest['files'].items():
        p=ROOT/rel
        if not p.is_file():failures.append((rel,'missing'));continue
        if p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            failures.append((rel,'size or SHA256 mismatch'))
    print(json.dumps(dict(verified=len(manifest['files'])-len(failures),total=len(manifest['files']),failures=failures),ensure_ascii=False))
    if failures:sys.exit(1)

if __name__=='__main__':main()
