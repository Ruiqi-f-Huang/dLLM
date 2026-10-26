"""Verify the extracted evidence snapshot without GPU or external dependencies."""
import hashlib
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / '_archive/manifest.json').read_text())
    failures = []
    for name, record in manifest.items():
        path = root / name
        if not path.is_file():
            failures.append({'file': name, 'reason': 'missing'})
        elif path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            failures.append({'file': name, 'reason': 'size or SHA256 mismatch'})
    result = {'verified_files': len(manifest) - len(failures), 'total_files': len(manifest), 'failures': failures}
    (root / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
