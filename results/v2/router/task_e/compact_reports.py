"""Lossless JSON whitespace compaction; summaries remain human-readable."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'results/v2/router'
records = []
for directory in [BASE / 'p4-v1adapter', BASE / 'synthetic16/p4']:
    paths = [directory / 'report.json'] + sorted(p for p in directory.rglob('report.json') if p != directory / 'report.json')
    for path in paths:
        original = path.read_bytes()
        value = json.loads(original)
        # A pretty representation can reconstruct the original byte stream.
        newline = '\r\n' if b'\r\n' in original else '\n'
        assert (json.dumps(value, indent=2, allow_nan=False) + '\n').replace('\n', newline).encode() == original
        if 'packaged_p4_report_sha256' in value:
            value['packaged_p4_report_sha256'] = hashlib.sha256((directory / 'report.json').read_bytes()).hexdigest()
        packed = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
        assert json.loads(packed) == value
        path.write_bytes(packed)
        records.append(dict(path=str(path.relative_to(ROOT)), original_sha256=hashlib.sha256(original).hexdigest(),
                            compact_sha256=hashlib.sha256(packed).hexdigest(), original_newline=repr(newline),
                            saved_bytes=len(original) - len(packed)))
(BASE / 'task_e/json_compaction.json').write_text(json.dumps(records, indent=2) + '\n')
print('Bytes saved:', sum(r['saved_bytes'] for r in records))
