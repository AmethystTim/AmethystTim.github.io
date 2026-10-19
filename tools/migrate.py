"""One-time, reproducible migration from the original main branch snapshot.

The source archive and baseline are retained in .local while working. Existing
content is never reconstructed from the rendered site.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import yaml
from links import destinations, resolve_local, rewrite

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
SNAPSHOT = ROOT / '.local' / 'original'
REPORT = ROOT / 'config' / 'migration.json'


def main():
    if REPORT.exists():
        raise SystemExit('Migration already completed; inspect config/migration.json.')
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / '.local' / 'original.tar'
    with archive.open('wb') as stream:
        subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'archive', 'HEAD'], cwd=ROOT, stdout=stream, check=True)
    with tarfile.open(archive) as stream:
        stream.extractall(SNAPSHOT, filter='data')
    original_docs = SNAPSHOT / 'docs'
    markdown = sorted(original_docs.rglob('*.md'))
    mapping = {}
    for source in markdown:
        rel = source.relative_to(original_docs)
        if rel.parts[:2] == ('blog', 'posts'):
            destination = rel.parent / rel.stem / rel.name
        else:
            destination = rel if rel.name == 'index.md' else rel.parent / rel.stem / 'index.md'
        mapping[source.resolve()] = DOCS / destination

    used = set()
    copied = []
    missing = []
    documents = []
    for source in markdown:
        destination = mapping[source.resolve()]
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding='utf-8-sig')

        def update(url):
            parts = urlsplit(url)
            target = None
            # These used to depend on GitHub's blob/raw endpoints, even though
            # all the images were already in the repository.
            if (parts.netloc.lower() in {'github.com', 'raw.githubusercontent.com'}
                    and re.match(r'^/amethysttim/amethysttim.github.io/(?:blob/|raw/)?(?:main|master)/docs/', unquote(parts.path), re.I)):
                path = re.split(r'/(?:main|master)/docs/', unquote(parts.path), flags=re.I)[1]
                target = (original_docs / path).resolve()
            elif not parts.scheme and not parts.netloc:
                target = resolve_local(url, source, original_docs)
            else:
                return url
            if target is None:
                return url
            if not target.exists():
                # One historical path contained redundant, non-existent
                # intermediate directories; resolve() normalizes it.
                missing.append({'document': source.relative_to(original_docs).as_posix(), 'url': url})
                return url
            suffix = ('#' + parts.fragment) if parts.fragment else ''
            if target in mapping:
                new_target = mapping[target]
                return quote(os.path.relpath(new_target, destination.parent).replace(os.sep, '/'), safe='/,:&()') + suffix
            if target.is_file():
                new_target = destination.parent / target.name
                if new_target.exists() and new_target.read_bytes() != target.read_bytes():
                    digest = hashlib.sha256(target.read_bytes()).hexdigest()[:8]
                    new_target = destination.parent / f'{target.stem}-{digest}{target.suffix}'
                if new_target.resolve() != target.resolve():
                    shutil.copy2(target, new_target)
                used.add(target)
                copied.append({'source': target.relative_to(SNAPSHOT).as_posix(), 'destination': new_target.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
                return quote(new_target.name, safe='-_.()') + suffix
            return url

        updated = rewrite(text, update)
        destination.write_text(updated, encoding='utf-8', newline='\n')
        old_working_path = DOCS / source.relative_to(original_docs)
        if old_working_path != destination and old_working_path.exists():
            old_working_path.unlink()
        documents.append({'source': source.relative_to(SNAPSHOT).as_posix(), 'destination': destination.relative_to(ROOT).as_posix()})

    # Remove the old central asset copies only after every document is written.
    # Keep every unreferenced original asset outside the published docs tree.
    unassigned = []
    new_assets = {entry['destination'] for entry in copied}
    for source in original_docs.rglob('*'):
        if not source.is_file() or source.suffix == '.md' or source.name == '.DS_Store':
            continue
        rel = source.relative_to(original_docs)
        if rel.parts[0] in {'javascripts', 'stylesheets'} or source.name.startswith('.'):
            continue
        original_working = DOCS / rel
        if source.resolve() not in used:
            backup = ROOT / 'archive' / 'unassigned-assets' / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup)
            unassigned.append(backup.relative_to(ROOT).as_posix())
        if original_working.exists() and original_working.relative_to(ROOT).as_posix() not in new_assets:
            original_working.unlink()
    for folder in sorted(DOCS.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

    # Retain the hand-curated order and labels while allowing automatic additions.
    config_text = (ROOT / 'mkdocs.yml').read_text(encoding='utf-8')
    nav = yaml.safe_load('nav:\n' + config_text.split('\nnav:\n', 1)[1])['nav']

    def remap(value):
        if isinstance(value, str):
            target = mapping.get((original_docs / value).resolve())
            return target.relative_to(DOCS).as_posix() if target else value
        if isinstance(value, list):
            return [remap(item) for item in value]
        return {key: remap(item) for key, item in value.items()}
    (ROOT / 'config').mkdir(exist_ok=True)
    (ROOT / 'config' / 'navigation.yml').write_text(yaml.safe_dump({'nav': remap(nav)}, allow_unicode=True, sort_keys=False), encoding='utf-8')
    report = {'source_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'documents': documents, 'assets': copied, 'unassigned_assets': unassigned, 'missing_references': missing}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'documents': len(documents), 'asset_references': len(copied), 'unique_original_assets': len(used), 'unassigned_assets': len(unassigned), 'missing_references': missing}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
