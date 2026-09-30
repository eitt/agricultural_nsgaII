"""Verify the published package inventory; do not alter any file."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
expected = json.loads((root / 'SHA256SUMS.json').read_text())
failures = []
for name, digest in expected.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        failures.append(name)
if failures:
    raise SystemExit('Missing or changed files: ' + ', '.join(failures))
print(f'All {len(expected)} packaged file hashes match.')
