"""Prepare reviewed source changes for the connected GitHub Git-data API.

Only the explicit project directories are included. Workspace research notes,
virtual environments, build output and .git metadata are never part of this
manifest. This helper makes no network calls and does not write Git metadata.
"""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    listing = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'ls-tree', '-rz', 'HEAD'], cwd=ROOT)
    originals = {}
    for line in listing.split(b'\0'):
        if not line:
            continue
        header, path = line.split(b'\t', 1)
        mode, kind, sha = header.decode().split()
        originals[path.decode()] = {'mode': mode, 'type': kind, 'sha': sha}
    original_shas = {entry['sha'] for entry in originals.values()}
    candidates = {ROOT / path for path in originals if (ROOT / path).is_file()}
    for dirname in ['docs', 'config', 'tools', 'overrides', 'archive', '.github']:
        for path in (ROOT / dirname).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name != '.DS_Store':
                candidates.add(path)
    candidates.update(ROOT / path for path in ['.gitignore', 'README.md', 'mkdocs.yml', 'requirements.txt', 'publish.ps1'])
    metadata, content = [], []
    for path in sorted(candidates):
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        sha = blob_sha(data)
        if relative in originals and sha == originals[relative]['sha']:
            continue
        entry = {'path': relative, 'mode': '100644', 'type': 'blob'}
        if sha in original_shas:
            entry['sha'] = sha
            metadata.append(entry)
        else:
            entry['content'] = data.decode('utf-8')
            content.append(entry)
    for path in originals:
        if not (ROOT / path).is_file():
            metadata.append({'path': path, 'mode': '100644', 'type': 'blob', 'sha': None})
    # Bounded request sizes keep both tool output and GitHub requests reviewable.
    batches = [metadata[index:index + 150] for index in range(0, len(metadata), 150)]
    batch, size = [], 0
    for entry in content:
        entry_size = len(json.dumps(entry, ensure_ascii=False))
        if size + entry_size > 48000 and batch:
            batches.append(batch)
            batch, size = [], 0
        batch.append(entry)
        size += entry_size
    if batch:
        batches.append(batch)
    destination = ROOT / '.local/publish'
    destination.mkdir(parents=True, exist_ok=True)
    for index, entries in enumerate(batches):
        (destination / f'{index:03}.json').write_text(json.dumps(entries, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'batches': len(batches), 'reused_or_deleted_entries': len(metadata), 'text_entries': len(content), 'source_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}))


if __name__ == '__main__':
    main()
