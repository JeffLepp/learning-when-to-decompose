"""Create and independently verify a portable snapshot without runtime/model weights."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
EXCLUDED = {'.venv', '__pycache__', '.git'}


def main():
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and not any(part in EXCLUDED for part in p.relative_to(ROOT).parts)
                   and p.name != 'HANDOFF_MANIFEST.json')
    manifest = {
        'description': 'Portable project snapshot; source, data, records, fitted controller, docs. No Python environment or Ollama model weights.',
        'sha256': {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files},
    }
    manifest_path = ROOT / 'HANDOFF_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    files.append(manifest_path)
    # Never silently replace a previously delivered snapshot.
    archive = ROOT.parent / 'FinalProject-Handoff.zip'
    suffix = 2
    while archive.exists():
        archive = ROOT.parent / f'FinalProject-Handoff-{suffix}.zip'
        suffix += 1
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, 'FinalProject/' + p.relative_to(ROOT).as_posix())
    with ZipFile(archive) as z:
        assert z.testzip() is None, 'ZIP integrity failure'
        saved = json.loads(z.read('FinalProject/HANDOFF_MANIFEST.json'))
        for name, expected in saved['sha256'].items():
            assert sha256(z.read('FinalProject/' + name)).hexdigest() == expected, name
        expected_names = {'FinalProject/' + name for name in saved['sha256']}
        expected_names.add('FinalProject/HANDOFF_MANIFEST.json')
        assert set(z.namelist()) == expected_names
    print(f'Created and readback-verified {archive}')
    print(f'{len(files)} files, {archive.stat().st_size:,} bytes; every archived file hash matched.')


if __name__ == '__main__':
    main()
