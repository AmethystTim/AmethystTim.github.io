"""Audit this migration against the retained original snapshot and built site."""
import hashlib
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from links import rewrite

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value and key in {'src', 'href'}:
                self.urls.append(value)


def main():
    report = json.loads((ROOT / 'config/migration.json').read_text(encoding='utf-8'))
    errors = []
    editable_landings = {'docs/index.md', 'docs/blog/index.md', 'docs/课程笔记/index.md', 'docs/Paper/index.md', 'docs/比赛/index.md'}
    for record in report['documents']:
        source = ROOT / '.local/original' / record['source']
        target = ROOT / record['destination']
        if not target.exists():
            errors.append(f'Missing migrated document: {target}')
        elif record['source'] not in editable_landings:
            before = rewrite(source.read_text(encoding='utf-8-sig'), lambda url: '@DESTINATION')
            after = rewrite(target.read_text(encoding='utf-8-sig'), lambda url: '@DESTINATION')
            if before != after:
                errors.append(f'Article text changed beyond link destinations: {record["source"]}')
    for record in report['assets']:
        target = ROOT / record['destination']
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
            errors.append(f'Asset hash mismatch: {record["destination"]}')
    for path in report['unassigned_assets']:
        original = ROOT / '.local/original/docs' / Path(path).relative_to('archive/unassigned-assets')
        if (ROOT / path).read_bytes() != original.read_bytes():
            errors.append(f'Unassigned asset mismatch: {path}')

    baseline = ROOT / '.local/baseline'
    site = ROOT / 'site'
    old_pages = {path.relative_to(baseline).as_posix() for path in baseline.rglob('*.html')}
    new_pages = {path.relative_to(site).as_posix() for path in site.rglob('*.html')}
    missing_pages = old_pages - new_pages
    errors.extend(f'Old public URL missing: {page}' for page in sorted(missing_pages))
    local_links = 0
    exists = {}
    for path in site.rglob('*.html'):
        parser = Links()
        parser.feed(path.read_text(encoding='utf-8'))
        for url in parser.urls:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            raw = unquote(parts.path)
            target = Path(os.path.abspath(site / raw.lstrip('/') if raw.startswith('/') else path.parent / raw))
            if target not in exists:
                exists[target] = target.exists()
            if not exists[target]:
                errors.append(f'Broken rendered link: {path.relative_to(site)} -> {url}')
            local_links += 1
    search = (site / 'search/search_index.json').read_text(encoding='utf-8')
    if 'blog/2025/01/23/' in search:
        errors.append('Draft blog article leaked into search.')
    home = (site / 'index.html').read_text(encoding='utf-8')
    for token in ['hero-title', 'data-md-color-scheme="default"', 'data-md-color-scheme="slate"']:
        if token not in home:
            errors.append(f'Homepage missing {token}')
    result = {'original_documents': len(report['documents']), 'asset_hash_checks': len(report['assets']), 'unassigned_assets_preserved': len(report['unassigned_assets']), 'original_public_pages_preserved': len(old_pages - missing_pages), 'rendered_local_links_checked': local_links, 'errors': errors}
    output = ROOT / '.local/verification.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
