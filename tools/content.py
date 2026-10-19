"""Manage one-document-one-directory content. Run with --help for examples."""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit
from uuid import uuid4

import yaml

from links import IMAGE_SUFFIXES, destinations, resolve_local, rewrite

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
SECTIONS = {'notes': '课程笔记', 'papers': 'Paper', 'contests': '比赛', 'posts': 'blog/posts'}


def inside(path, root=None):
    root = root or DOCS
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError(f'Path must stay inside {root}')
    return resolved


def document(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path if path.parts[:1] == ('docs',) else DOCS / path
    path = inside(path)
    if path.is_dir():
        markdown = list(path.glob('*.md'))
        if len(markdown) != 1:
            raise ValueError('Specify a directory containing exactly one Markdown document.')
        path = markdown[0]
    if not path.is_file() or path.suffix != '.md':
        raise ValueError(f'Document not found: {path}')
    return path


def bundle(path):
    if path.relative_to(DOCS).as_posix() in {'index.md', 'library/index.md', 'about/index.md', 'blog/index.md', 'Paper/index.md', '课程笔记/index.md', '比赛/index.md'}:
        raise ValueError('This is a site landing page. Edit it directly instead of moving/deleting it as an article.')
    folder = path.parent
    # Collection and root landing pages own subdocuments, so they cannot be
    # moved/deleted as if they were a single article.
    markdown = list(folder.rglob('*.md'))
    if folder == DOCS or len(markdown) != 1 or markdown[0] != path:
        raise ValueError('This directory contains other documents. Select an individual article.')
    return inside(folder)


def create(args):
    raw = Path(args.path)
    if raw.is_absolute() or '..' in raw.parts or not raw.parts or raw.suffix:
        raise ValueError('--path must be a relative directory path without an extension.')
    folder = inside(DOCS / SECTIONS[args.section] / raw)
    if folder.exists():
        raise ValueError(f'Destination already exists: {folder}')
    folder.mkdir(parents=True)
    post = args.section == 'posts'
    meta = {'title': args.title, 'date': date.today(), 'tags': args.tag or []}
    if post:
        meta.update({'draft': not args.publish, 'authors': ['dinorex'], 'categories': [args.category or '随笔']})
    path = folder / (raw.name + '.md' if post else 'index.md')
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    body = f'---\n{front}\n---\n\n# {args.title}\n\n在这里写下正文。\n'
    if post:
        body += '\n<!-- more -->\n\n## 正文\n\n'
    path.write_text(body, encoding='utf-8')
    print(path.relative_to(ROOT).as_posix())


def search(args):
    query = args.query.casefold()
    found = 0
    for path in sorted(DOCS.rglob('*.md')):
        relative = path.relative_to(DOCS).as_posix()
        text = path.read_text(encoding='utf-8')
        if not query or query in (relative + '\n' + text).casefold():
            print(relative)
            if query:
                for number, line in enumerate(text.splitlines(), 1):
                    if query in line.casefold():
                        print(f'  {number}: {line.strip()[:140]}')
                        break
            found += 1
    print(f'{found} document(s)')


def edit(args):
    path = document(args.document)
    editor = os.environ.get('EDITOR')
    if editor:
        subprocess.run(shlex.split(editor, posix=os.name != 'nt') + [str(path)], check=True)
    elif shutil.which('code'):
        subprocess.run([shutil.which('code'), '--wait', str(path)], check=True)
    elif os.name == 'nt':
        os.startfile(path)
    else:
        raise ValueError(f'Set EDITOR to your text editor, then retry. File: {path}')


def incoming(folder):
    refs = []
    for path in DOCS.rglob('*.md'):
        if path.is_relative_to(folder):
            continue
        for url in destinations(path.read_text(encoding='utf-8')):
            target = resolve_local(url, path, DOCS)
            if target and target.is_relative_to(folder):
                refs.append(path.relative_to(DOCS).as_posix())
                break
    return refs


def remove(args):
    path = document(args.document)
    folder = bundle(path)
    refs = incoming(folder)
    if refs and not args.force:
        raise ValueError('Other documents link here; update their links first or use --force:\n' + '\n'.join(refs))
    ident = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:6]
    trash = ROOT / '.trash' / ident
    trash.mkdir(parents=True)
    manifest = {'path': folder.relative_to(DOCS).as_posix(), 'document': path.name, 'deleted_at': datetime.now().isoformat()}
    (trash / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.move(str(folder), str(trash / 'content'))
    print(f'Moved document and assets to .trash/{ident}. Restore: python tools/content.py restore {ident}')


def restore(args):
    trash_root = (ROOT / '.trash').resolve()
    if not args.id:
        for manifest in sorted(trash_root.glob('*/manifest.json')):
            data = json.loads(manifest.read_text(encoding='utf-8'))
            print(f'{manifest.parent.name}  {data["path"]}')
        return
    trash = inside(trash_root / args.id, trash_root)
    manifest = json.loads((trash / 'manifest.json').read_text(encoding='utf-8'))
    destination = inside(DOCS / manifest['path'])
    if destination.exists():
        raise ValueError(f'Destination already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(trash / 'content'), str(destination))
    (trash / 'manifest.json').unlink()
    trash.rmdir()
    print(destination.relative_to(ROOT).as_posix())


def move(args):
    path = document(args.document)
    old_folder = bundle(path)
    new_folder = inside(DOCS / args.destination)
    if new_folder.exists() or new_folder.is_relative_to(old_folder):
        raise ValueError('Destination already exists or is inside the source directory.')
    posts = DOCS / 'blog/posts'
    if path.is_relative_to(posts) != new_folder.is_relative_to(posts):
        raise ValueError('Moving between blog posts and notes requires converting metadata; create the new type first.')
    mapping = {p.resolve(): new_folder / p.relative_to(old_folder) for p in old_folder.rglob('*') if p.is_file()}
    changes = {}
    for source in DOCS.rglob('*.md'):
        destination = mapping.get(source.resolve(), source)

        def update(url):
            target = resolve_local(url, source, DOCS)
            if target is None:
                return url
            relocated = mapping.get(target, new_folder / target.relative_to(old_folder) if target.is_relative_to(old_folder) else target)
            if source == destination and target == relocated:
                return url
            parts = urlsplit(url)
            relative = os.path.relpath(relocated, destination.parent).replace(os.sep, '/')
            return quote(relative, safe='/,:&()') + ('?' + parts.query if parts.query else '') + ('#' + parts.fragment if parts.fragment else '')

        changes[destination] = rewrite(source.read_text(encoding='utf-8'), update)
    new_folder.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_folder), str(new_folder))
    for destination, content in changes.items():
        if destination.read_text(encoding='utf-8') != content:
            destination.write_text(content, encoding='utf-8')
    print(f'{path.relative_to(DOCS)} -> {mapping[path.resolve()].relative_to(DOCS)}; relative links updated.')


def attach(args):
    path = document(args.document)
    bundle(path)
    source = Path(args.source).resolve()
    if not source.is_file():
        raise ValueError('Attachment does not exist.')
    target = path.parent / source.name
    if target.exists():
        raise ValueError('A file with this name already exists in the article directory.')
    shutil.copy2(source, target)
    prefix = '!' if source.suffix.lower() in IMAGE_SUFFIXES - {'.pdf'} else ''
    print(f'{prefix}[{source.stem}](<{source.name}>)')


def check(args):
    failures = []
    count = 0
    for path in DOCS.rglob('*.md'):
        count += 1
        text = path.read_text(encoding='utf-8')
        for url in destinations(text):
            target = resolve_local(url, path, DOCS)
            if target is None:
                if '/amethysttim.github.io/' in url.lower() and '/docs/images/' in url:
                    failures.append(f'{path.relative_to(DOCS)}: legacy GitHub image: {url}')
                continue
            if not target.exists():
                failures.append(f'{path.relative_to(DOCS)}: missing {url}')
            elif target.suffix.lower() in IMAGE_SUFFIXES and target.parent != path.parent:
                failures.append(f'{path.relative_to(DOCS)}: attachment must live with document: {url}')
    if (DOCS / 'images').exists():
        failures.append('The old central docs/images directory still exists.')
    if failures:
        raise ValueError('\n'.join(failures))
    print(f'Checked {count} documents: local links and attachment locations are valid.')


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='command', required=True)
    new = commands.add_parser('new', help='Create a document bundle; navigation updates automatically.')
    new.add_argument('--section', choices=SECTIONS, required=True)
    new.add_argument('--path', required=True, help='Relative directory within the section, no extension.')
    new.add_argument('--title', required=True)
    new.add_argument('--tag', action='append')
    new.add_argument('--category')
    new.add_argument('--publish', action='store_true', help='For posts: start published rather than draft.')
    new.set_defaults(action=create)
    for name in ['list', 'search']:
        command = commands.add_parser(name, help='Find documents by path or full text, including drafts.')
        command.add_argument('query', nargs='?', default='')
        command.set_defaults(action=search)
    command = commands.add_parser('edit', help='Open a document in EDITOR or the local Markdown editor.')
    command.add_argument('document')
    command.set_defaults(action=edit)
    command = commands.add_parser('delete', help='Move an article and its assets to the local recycle bin.')
    command.add_argument('document')
    command.add_argument('--force', action='store_true', help='Allow deletion when other articles link here.')
    command.set_defaults(action=remove)
    command = commands.add_parser('restore', help='List deleted bundles, or restore one by ID.')
    command.add_argument('id', nargs='?')
    command.set_defaults(action=restore)
    command = commands.add_parser('move', help='Move/rename a bundle and update relative document links.')
    command.add_argument('document')
    command.add_argument('destination', help='Destination directory relative to docs/.')
    command.set_defaults(action=move)
    command = commands.add_parser('attach', help='Copy an image/attachment into the article directory.')
    command.add_argument('document')
    command.add_argument('source')
    command.set_defaults(action=attach)
    commands.add_parser('check', help='Validate local links and attachment locations.').set_defaults(action=check)
    return cli


def main():
    args = parser().parse_args()
    try:
        args.action(args)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
