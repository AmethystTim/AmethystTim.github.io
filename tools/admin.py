"""Local admin console: upload, organise, preview and publish blog documents.

Run from the project root:
    .\\.venv\\Scripts\\python.exe tools/admin.py

It listens on 127.0.0.1 only. Every API call needs the per-launch token that
is embedded in the page, so other websites cannot drive it from the browser.
"""
import argparse
import atexit
import base64
import binascii
import contextlib
import io
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import traceback
import urllib.error
import urllib.request
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, unquote, urlsplit

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import content  # noqa: E402  (shares DOCS/ROOT so tests can redirect both)
from links import destinations, resolve_local, rewrite  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = {'notes': '课程笔记', 'papers': '论文阅读', 'contests': '竞赛手记', 'posts': '生活随笔'}
LANDING = {'index.md', 'library/index.md', 'about/index.md', 'blog/index.md',
           'Paper/index.md', '课程笔记/index.md', '比赛/index.md'}
META_ORDER = ['title', 'date', 'tags', 'description', 'draft', 'authors', 'categories']
INVALID = re.compile(r'[<>:"/\\|?*#%\x00-\x1f]')
RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
FRONT = re.compile(r'\A---[ \t]*\n(.*?)^---[ \t]*(?:\n|\Z)', re.S | re.M)
PUBLISHED_PATHS = ['docs', 'config', 'tools', 'overrides', 'archive', '.github', 'README.md',
                   'mkdocs.yml', 'requirements.txt', '.gitignore', 'publish.ps1', 'admin.cmd']
WRITE_LOCK = threading.Lock()


# ---------------------------------------------------------------- helpers

def docs():
    return content.DOCS


def root():
    return content.ROOT


def section_root(key):
    if key not in content.SECTIONS:
        raise ValueError(f'未知板块：{key}')
    return docs() / content.SECTIONS[key]


def section_of(rel):
    for key, folder in sorted(content.SECTIONS.items(), key=lambda item: -len(item[1])):
        if rel.startswith(folder + '/'):
            return key
    return None


def segment(name, what='名称'):
    name = str(name or '').strip()
    if (not name or name in {'.', '..'} or INVALID.search(name) or name[0] == '.' or name[-1] == '.'
            or name.split('.')[0].upper() in RESERVED):
        raise ValueError(f'{what}不合法：「{name}」。不能为空，不能以点开头或结尾，也不能包含 < > : " / \\ | ? * # %')
    return name


def category_path(value):
    parts = [part.strip() for part in str(value or '').replace('\\', '/').split('/')]
    return '/'.join(segment(part, '分类名') for part in parts if part)


def split_front(text):
    text = text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    match = FRONT.match(text)
    if not match:
        return {}, text
    meta = yaml.safe_load(match[1]) or {}
    if not isinstance(meta, dict):
        raise ValueError('文首元数据（front matter）必须是「键: 值」形式。')
    return meta, text[match.end():].lstrip('\n')


def ordered(meta):
    keys = [key for key in META_ORDER if key in meta] + [key for key in meta if key not in META_ORDER]
    return {key: meta[key] for key in keys}


def compose(meta, body):
    body = body.replace('\r\n', '\n').lstrip('\n')
    if not body.endswith('\n'):
        body += '\n'
    if not meta:
        return body
    front = yaml.safe_dump(ordered(meta), allow_unicode=True, sort_keys=False, default_flow_style=False).strip()
    return f'---\n{front}\n---\n\n{body}'


def h1_span(body):
    """Locate the first level-one ATX heading outside fenced code."""
    fence, offset = False, 0
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(('```', '~~~')):
            fence = not fence
        elif not fence:
            match = re.match(r'#[ \t]+(.+?)[ \t#]*$', line.rstrip('\n'))
            if match:
                return offset + match.start(1), offset + match.end(1), match[1]
        offset += len(line)
    return None


def fallback_title(path):
    folder = path.parent
    if path.stem == 'index' and folder.name == 'i':
        folder = folder.parent
    return folder.name if path.stem in {'index', 'i'} else path.stem


def effective_title(meta, body, path):
    heading = h1_span(body)
    return str(meta.get('title') or (heading[2] if heading else fallback_title(path)))


def parse_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, dict):
        return parse_date(value.get('created'))
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            raise ValueError(f'日期格式应为 YYYY-MM-DD：{value}') from None
    return None


def site_url():
    try:
        match = re.search(r'^site_url:\s*(\S+)', (root() / 'mkdocs.yml').read_text(encoding='utf-8'), re.M)
    except OSError:
        match = None
    return (match[1].strip('\'"') if match else '/').rstrip('/') + '/'


def page_url(rel, meta, title):
    """Mirror MkDocs (directory URLs) and the Material blog post URL rules."""
    if rel.startswith('blog/posts/'):
        created = parse_date(meta.get('date'))
        if not created:
            return None
        from pymdownx.slugs import slugify
        slug = meta.get('slug') or slugify(case='lower')(title, '-')
        url = f'blog/{created:%Y/%m/%d}/{slug}/'
    else:
        path = PurePosixPath(rel)
        url = (path.parent.as_posix() if path.name == 'index.md' else path.with_suffix('').as_posix()) + '/'
    return quote(url, safe='/')


def read_doc(rel):
    path = content.document(rel)
    rel = path.relative_to(docs()).as_posix()
    if rel in LANDING or not section_of(rel):
        raise ValueError('只能管理四个板块中的文章；首页、文档库等入口页请直接编辑文件。')
    return path, rel


def record(path):
    rel = path.relative_to(docs()).as_posix()
    key = section_of(rel)
    meta, body = split_front(path.read_text(encoding='utf-8'))
    title = effective_title(meta, body, path)
    folder = path.parent
    if key == 'posts':
        category = ''
    else:
        parent = folder.parent
        category = parent.relative_to(section_root(key)).as_posix() if parent != section_root(key) else ''
        category = '' if category == '.' else category
    created = parse_date(meta.get('date'))
    tags = meta.get('tags') or []
    categories = meta.get('categories') or []
    url = page_url(rel, meta, title)
    draft = bool(meta.get('draft'))
    return {
        'path': rel, 'section': key, 'category': category, 'slug': folder.name, 'title': title,
        'date': created.isoformat() if created else '', 'tags': [str(tag) for tag in tags] if isinstance(tags, list) else [str(tags)],
        'categories': [str(item) for item in categories] if isinstance(categories, list) else [str(categories)],
        'draft': draft, 'description': str(meta.get('description') or ''), 'url': url,
        'online': None if draft or not url else site_url() + url,
        'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds'),
    }


def all_documents():
    output = []
    for path in sorted(docs().rglob('*.md')):
        rel = path.relative_to(docs()).as_posix()
        if rel in LANDING or not section_of(rel):
            continue
        if any(part.startswith('.') for part in path.relative_to(docs()).parts):
            continue
        try:
            output.append(record(path))
        except (ValueError, yaml.YAMLError) as exc:
            output.append({'path': rel, 'section': section_of(rel), 'title': rel, 'error': str(exc), 'category': '',
                           'slug': path.parent.name, 'date': '', 'tags': [], 'categories': [], 'draft': False,
                           'description': '', 'url': None, 'online': None, 'modified': ''})
    return output


# ------------------------------------------------------------- categories

def post_category_file():
    return root() / 'config' / 'post-categories.yml'


def stored_post_categories():
    file = post_category_file()
    if not file.exists():
        return []
    data = yaml.safe_load(file.read_text(encoding='utf-8')) or {}
    return [str(item) for item in data.get('categories', [])]


def store_post_categories(names):
    file = post_category_file()
    file.parent.mkdir(parents=True, exist_ok=True)
    header = '# 生活随笔可选分类（由本地管理台维护）。文章实际分类写在各自元数据的 categories 中。\n'
    file.write_text(header + yaml.safe_dump({'categories': names}, allow_unicode=True, sort_keys=False), encoding='utf-8')


def article_folders(key):
    base = section_root(key)
    folders = set()
    for path in base.rglob('*.md') if base.exists() else []:
        if path.relative_to(docs()).as_posix() not in LANDING and path.parent != base:
            folders.add(path.parent)
    return folders


def categories(key):
    if key == 'posts':
        counts = {}
        for item in all_documents():
            if item['section'] == 'posts':
                for name in item['categories']:
                    counts[name] = counts.get(name, 0) + 1
        names = list(dict.fromkeys(stored_post_categories() + sorted(counts)))
        return [{'path': name, 'count': counts.get(name, 0)} for name in names]
    base = section_root(key)
    if not base.exists():
        return []
    articles = article_folders(key)
    output = []
    for folder in sorted(path for path in base.rglob('*') if path.is_dir()):
        if folder in articles or any(article in folder.parents for article in articles):
            continue
        if any(part.startswith('.') for part in folder.relative_to(base).parts):
            continue
        count = sum(1 for article in articles if folder in article.parents)
        output.append({'path': folder.relative_to(base).as_posix(), 'count': count})
    return output


def create_category(key, name):
    if key == 'posts':
        name = segment(name, '分类名')
        names = stored_post_categories()
        if name in names or any(item['path'] == name for item in categories('posts')):
            raise ValueError(f'分类「{name}」已存在。')
        store_post_categories(names + [name])
        return name
    path = category_path(name)
    if not path:
        raise ValueError('请填写分类名。')
    folder = content.inside(section_root(key) / path)
    if folder.exists():
        raise ValueError(f'分类「{path}」已存在。')
    if any(article == folder or article in folder.parents for article in article_folders(key)):
        raise ValueError('分类不能建在某篇文章的目录里。')
    folder.mkdir(parents=True)
    (folder / '.gitkeep').write_text('', encoding='utf-8')
    return path


def delete_category(key, name):
    if key == 'posts':
        used = [item['path'] for item in all_documents() if item['section'] == 'posts' and name in item['categories']]
        if used:
            raise ValueError('仍有随笔使用该分类：\n' + '\n'.join(used))
        store_post_categories([item for item in stored_post_categories() if item != name])
        return
    path = category_path(name)
    folder = content.inside(section_root(key) / path)
    if not folder.is_dir():
        raise ValueError('分类不存在。')
    leftovers = [p.relative_to(docs()).as_posix() for p in folder.rglob('*')
                 if p.is_file() and p.name not in {'.gitkeep', '.DS_Store'}]
    if leftovers:
        raise ValueError('分类中仍有文件，请先移走或删除：\n' + '\n'.join(leftovers[:20]))
    shutil.rmtree(folder)


def rename_category(key, old, new):
    if key == 'posts':
        new = segment(new, '分类名')
        if any(item['path'] == new for item in categories('posts')):
            raise ValueError(f'分类「{new}」已存在。')
        for item in all_documents():
            if item['section'] == 'posts' and old in item['categories']:
                path, _ = read_doc(item['path'])
                meta, body = split_front(path.read_text(encoding='utf-8'))
                meta['categories'] = [new if str(name) == old else name for name in meta.get('categories', [])]
                path.write_text(compose(meta, body), encoding='utf-8')
        names = stored_post_categories()
        store_post_categories([new if name == old else name for name in names] if old in names else names + [new])
        return new
    base = section_root(key)
    old, new = category_path(old), category_path(new)
    source, target = content.inside(base / old), content.inside(base / new)
    if not source.is_dir():
        raise ValueError('原分类不存在。')
    if target.exists() or target.is_relative_to(source):
        raise ValueError('新分类已存在，或位于原分类内部。')
    for folder in sorted(article for article in article_folders(key) if source in article.parents):
        markdown = next(folder.glob('*.md'))
        destination = target / folder.relative_to(source)
        move_bundle(markdown, destination)
    for leftover in sorted(source.rglob('*')):
        destination = target / leftover.relative_to(source)
        if leftover.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif leftover.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(leftover), str(destination))
    target.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(source)
    return new


# -------------------------------------------------------------- navigation

def navigation_file():
    return root() / 'config' / 'navigation.yml'


def nav_scalar(text):
    if text != text.strip() or re.search(r'[:#\[\]{},&*!|>\'"%@`]', text) or text[:1] in '-?':
        return "'" + text.replace("'", "''") + "'"
    return text


def rewrite_nav_prefix(old, new):
    """Keep curated navigation entries pointing at a moved article."""
    file = navigation_file()
    if not file.exists():
        return
    text = file.read_text(encoding='utf-8')
    updated = re.sub(r'(?<=[:-] )' + re.escape(old + '/'), lambda _: new + '/', text)
    if updated != text:
        file.write_text(updated, encoding='utf-8')


def rewrite_nav_label(rel, title):
    file = navigation_file()
    if not file.exists():
        return
    text = file.read_text(encoding='utf-8')
    pattern = re.compile(r'^(\s*- )(.+?)(: ' + re.escape(rel) + r'\s*)$', re.M)
    updated = pattern.sub(lambda m: m[1] + nav_scalar(title) + m[3], text)
    if updated != text:
        file.write_text(updated, encoding='utf-8')


# ------------------------------------------------------------- documents

def quiet(function, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        function(SimpleNamespace(**kwargs))
    return output.getvalue().strip()


def redirect_file():
    return root() / 'config' / 'redirects.yml'


def add_redirect(old, new):
    """Record old URL path -> new URL path; tools/hooks.py turns these into redirect pages."""
    if old == new:
        return
    file = redirect_file()
    data = (yaml.safe_load(file.read_text(encoding='utf-8')) or {}).get('redirects') or {} if file.exists() else {}
    data = {str(key): (new if str(value) == old else str(value)) for key, value in data.items()}
    data[old] = new
    data.pop(new, None)  # a page lives at `new` now
    data = {key: value for key, value in data.items() if key != value}
    file.parent.mkdir(parents=True, exist_ok=True)
    header = '# 旧网址 -> 新网址（不含首尾斜杠）。构建时为旧网址生成跳转页；移动文章时由本地管理台自动维护。\n'
    file.write_text(header + yaml.safe_dump({'redirects': data}, allow_unicode=True, sort_keys=True), encoding='utf-8')


def move_bundle(markdown, destination, redirect=True):
    old = markdown.parent.relative_to(docs()).as_posix()
    new = destination.relative_to(docs()).as_posix()
    quiet(content.move, document=markdown.relative_to(docs()).as_posix(), destination=new)
    rewrite_nav_prefix(old, new)
    if redirect and markdown.name == 'index.md' and not old.startswith('blog/'):
        add_redirect(old, new)
    return destination / markdown.name


def attachment_name(name):
    name = re.split(r'[\\/]', str(name or ''))[-1]
    name = segment(name, '附件名')
    if name.lower().endswith(('.md', '.markdown')):
        raise ValueError('附件不能是 Markdown 文件；每篇文章只能有一个正文。')
    return name


def decode_attachments(items):
    files = {}
    for item in items or []:
        name = attachment_name(item.get('name'))
        if name in files:
            raise ValueError(f'附件重名：{name}')
        try:
            files[name] = base64.b64decode(item.get('data') or '', validate=True)
        except (binascii.Error, ValueError):
            raise ValueError(f'附件读取失败：{name}') from None
    return files


def local_basename(url):
    if re.match(r'^(?:[a-z][a-z0-9+.-]*:)?//', url, re.I) and not url.lower().startswith('file:'):
        return None
    if url.startswith(('#', 'data:', 'mailto:')):
        return None
    path = unquote(urlsplit(url).path if not re.match(r'^[a-z]:[\\/]', url, re.I) else url)
    return re.split(r'[\\/]', path)[-1] or None


def relink(text, path, names):
    """Point references like `assets/a.png` or `C:\\...\\a.png` at uploaded files beside the article."""
    def update(url):
        base = local_basename(url)
        if base not in names:
            return url
        target = resolve_local(url, path, docs())
        if target is not None and target == path.parent / base:
            return url
        fragment = urlsplit(url).fragment if not re.match(r'^[a-z]:[\\/]', url, re.I) else ''
        return quote(base, safe=',:&()') + ('#' + fragment if fragment else '')
    return rewrite(text, update)


def warnings_for(path, text):
    notes = []
    for url in destinations(text):
        if re.match(r'^(?:[a-z]:[\\/]|file:)', url, re.I):
            notes.append(f'引用了本机路径，网站上无法显示：{url}')
            continue
        target = resolve_local(url, path, docs())
        if target is not None and not target.exists():
            notes.append(f'引用的文件不存在：{url}')
    if path.relative_to(docs()).as_posix().startswith('blog/posts/') and '<!-- more -->' not in text:
        notes.append('随笔正文没有 <!-- more --> 摘要分隔符，随笔列表会显示全文。')
    return notes


def clean_meta(update, section):
    """Normalise values sent by the form; empty values mean 'remove this key'."""
    output = {}
    for key, value in (update or {}).items():
        if key not in {'title', 'date', 'tags', 'description', 'draft', 'categories', 'authors'}:
            continue
        if key == 'date':
            value = parse_date(value)
        elif key in {'tags', 'categories', 'authors'}:
            if isinstance(value, str):
                value = [part.strip() for part in re.split(r'[,，、]', value)]
            value = [str(item).strip() for item in value or [] if str(item).strip()]
            value = list(dict.fromkeys(value))
        elif key == 'draft':
            if section != 'posts':
                continue
            value = bool(value)
        else:
            value = str(value or '').strip()
        output[key] = value
    return output


def apply_meta(meta, update):
    meta = dict(meta)
    for key, value in update.items():
        if value is None or value == '' or value == []:
            meta.pop(key, None)
        elif key == 'date' and isinstance(meta.get('date'), dict):
            meta['date'] = {**meta['date'], 'created': value}
        else:
            meta[key] = value
    return meta


def upload(section, category, slug, meta, markdown, attachments=(), filename=''):
    base = section_root(section)
    if markdown is None or not str(markdown).strip():
        raise ValueError('请选择一个有正文的 .md 文件。')
    uploaded, body = split_front(markdown)
    slug = segment(slug or Path(str(filename or '')).stem, '文章目录名')
    if section == 'posts':
        folder = content.inside(base / slug)
        path = folder / f'{slug}.md'
    else:
        folder = content.inside(base / category_path(category) / slug)
        path = folder / 'index.md'
    if folder.exists():
        raise ValueError(f'已存在同名文章目录：{folder.relative_to(docs()).as_posix()}')
    if any(article == folder or article in folder.parents for article in article_folders(section)):
        raise ValueError('不能把文章放进另一篇文章的目录里。')
    files = decode_attachments(attachments)
    merged = apply_meta(uploaded, {k: v for k, v in clean_meta(meta, section).items() if v not in ('', [], None)})
    merged.setdefault('date', date.today())
    if section == 'posts':
        merged.setdefault('draft', False)
        merged.setdefault('authors', ['dinorex'])
        merged.setdefault('categories', ['随笔'])
    body = relink(body, path, set(files))
    folder.mkdir(parents=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    text = compose(merged, body)
    path.write_text(text, encoding='utf-8')
    result = record(path)
    result['warnings'] = warnings_for(path, text)
    return result


def load(rel):
    path, rel = read_doc(rel)
    meta, body = split_front(path.read_text(encoding='utf-8'))
    result = record(path)
    result['body'] = body
    result['extra'] = sorted(key for key in meta if key not in {'title', 'date', 'tags', 'description', 'draft', 'categories'})
    result['attachments'] = sorted(p.name for p in path.parent.iterdir() if p.is_file() and p != path and not p.name.startswith('.'))
    return result


def save(rel, meta=None, body=None):
    path, rel = read_doc(rel)
    section = section_of(rel)
    original, old_body = split_front(path.read_text(encoding='utf-8'))
    old_title = effective_title(original, old_body, path)
    before = record(path)
    update = clean_meta(meta, section)
    if body is None:
        body = old_body
    else:
        imported, body = split_front(body)
        update = {**update, **{k: v for k, v in clean_meta(imported, section).items() if v not in ('', [], None)}}
    if 'title' in update and update['title'] == old_title and 'title' not in original:
        update.pop('title')  # unchanged derived title: keep the file free of a redundant key
    new_meta = apply_meta(original, update)
    new_title = effective_title(new_meta, body, path)
    heading = h1_span(body)
    if new_title != old_title and heading and heading[2] == old_title:
        body = body[:heading[0]] + new_title + body[heading[1]:]
    text = compose(new_meta, body)
    path.write_text(text, encoding='utf-8')
    if new_title != old_title:
        rewrite_nav_label(rel, new_title)
    result = record(path)
    result['warnings'] = warnings_for(path, text)
    if section == 'posts' and before['url'] and result['url'] != before['url']:
        result['warnings'].append('标题或日期变化后，这篇随笔的网址也变了；旧链接会失效。')
    return result


def move(rel, category, slug):
    path, rel = read_doc(rel)
    section = section_of(rel)
    if section == 'posts':
        raise ValueError('随笔通过「随笔分类」归类，不需要移动目录；请在编辑页修改分类。')
    destination = content.inside(section_root(section) / category_path(category) / segment(slug, '文章目录名'))
    if any(article == destination or article in destination.parents for article in article_folders(section)):
        raise ValueError('不能把文章移进另一篇文章的目录里。')
    return record(move_bundle(path, destination))


def attach(rel, attachments, overwrite=False):
    path, rel = read_doc(rel)
    files = decode_attachments(attachments)
    clashes = [name for name in files if (path.parent / name).exists()]
    if clashes and not overwrite:
        raise ValueError('文章目录中已有同名文件：' + '、'.join(clashes))
    snippets = []
    for name, data in files.items():
        (path.parent / name).write_bytes(data)
        image = Path(name).suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.avif', '.bmp'}
        target = quote(name, safe=',:&()')
        snippets.append(f'{"!" if image else ""}[{Path(name).stem}]({target})')
    return {'snippets': snippets}


def delete(rel, force=False):
    _, rel = read_doc(rel)
    return quiet(content.remove, document=rel, force=bool(force))


def trash():
    output = []
    for manifest in sorted((root() / '.trash').glob('*/manifest.json'), reverse=True):
        data = json.loads(manifest.read_text(encoding='utf-8'))
        output.append({'id': manifest.parent.name, 'path': data.get('path', ''), 'deleted_at': data.get('deleted_at', '')})
    return output


def restore(ident):
    return quiet(content.restore, id=segment(ident, '回收站编号'))


def check():
    try:
        return {'ok': True, 'output': quiet(content.check)}
    except ValueError as exc:
        return {'ok': False, 'output': str(exc)}


def state():
    return {
        'sections': [{'key': key, 'label': LABELS[key], 'folder': folder} for key, folder in content.SECTIONS.items()],
        'documents': all_documents(),
        'categories': {key: categories(key) for key in content.SECTIONS},
        'trash': trash(),
        'site': site_url(),
    }


# ------------------------------------------------------ preview & publish

def port_open(port):
    # Windows lets another program hold [::]:port while we bind 127.0.0.1:port,
    # and browsers may then reach either one through "localhost"; check both.
    for host in ('127.0.0.1', '::1'):
        with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=0.3):
            return True
    return False


def free_port(start, attempts=50):
    for port in range(start, start + attempts):
        if not port_open(port):
            return port
    raise OSError(f'{start}–{start + attempts - 1} 端口都被占用了。')


def tail(path, limit=40):
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return ''
    return '\n'.join(lines[-limit:])


class Preview:
    """A managed `mkdocs serve`: every saved file rebuilds and reloads open pages."""

    def __init__(self, port):
        self.base_port = self.port = port
        self.process = None
        self.log = root() / '.local' / 'admin' / 'preview.log'

    def managed(self):
        return self.process is not None and self.process.poll() is None

    def online(self):
        # Only our own mkdocs process counts: a busy port may belong to anything.
        return self.managed() and port_open(self.port)

    def status(self):
        return {'managed': self.managed(), 'online': self.online(),
                'url': f'http://127.0.0.1:{self.port}/', 'log': tail(self.log, 12)}

    def ready(self, url):
        """MkDocs' 404 page has no live-reload script, so only frame pages that exist."""
        if not re.fullmatch(r'[A-Za-z0-9%/_.\-~]*', url or ''):
            raise ValueError('无效的预览地址。')
        request = urllib.request.Request(f'http://127.0.0.1:{self.port}/{url}', method='HEAD')
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return {'ready': response.status == 200}
        except (OSError, urllib.error.HTTPError):
            return {'ready': False}

    def start(self):
        if self.managed():
            return self.status()
        self.port = free_port(self.base_port)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        output = open(self.log, 'wb')
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        self.process = subprocess.Popen([sys.executable, '-m', 'mkdocs', 'serve', '-a', f'127.0.0.1:{self.port}', '--livereload'],
                                        cwd=root(), stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        env=env, creationflags=flags)
        output.close()
        return self.status()

    def stop(self):
        if self.managed():
            self.process.terminate()
            try:
                self.process.wait(5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        return self.status()


def git(*args):
    command = ['git', '-c', f'safe.directory={root().as_posix()}', '-c', 'core.quotePath=false', *args]
    return subprocess.run(command, cwd=root(), capture_output=True, text=True, encoding='utf-8', errors='replace')


def pending():
    result = git('status', '--porcelain', '--untracked-files=all', '--', *PUBLISHED_PATHS)
    if result.returncode != 0:
        return {'error': (result.stderr or result.stdout).strip(), 'changes': []}
    changes = [{'status': line[:2].strip(), 'path': line[3:].strip().strip('"')} for line in result.stdout.splitlines() if line.strip()]
    return {'error': '', 'changes': changes}


class Publisher:
    """Runs publish.ps1 (checks, tests, strict build, commit, push) in the background."""

    def __init__(self):
        self.state, self.lines, self.code, self.started = 'idle', [], None, ''
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            return {'state': self.state, 'code': self.code, 'started': self.started, 'log': '\n'.join(self.lines[-400:])}

    def start(self, message):
        message = str(message or '').strip() or 'Update blog from local admin'
        shell = 'powershell' if os.name == 'nt' else shutil.which('pwsh')
        if not shell:
            raise ValueError('需要 PowerShell（pwsh）才能运行 publish.ps1。')
        with self.lock:
            if self.state == 'running':
                raise ValueError('发布正在进行中。')
            self.state, self.lines, self.code = 'running', [], None
            self.started = datetime.now().isoformat(timespec='seconds')
        script = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                  "& (Join-Path $PWD 'publish.ps1') -Message $env:BLOG_PUBLISH_MESSAGE; exit $LASTEXITCODE")
        env = {**os.environ, 'BLOG_PUBLISH_MESSAGE': message, 'PYTHONIOENCODING': 'utf-8'}
        threading.Thread(target=self._run, args=([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', script], env),
                         daemon=True).start()
        return self.status()

    def _run(self, command, env):
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.Popen(command, cwd=root(), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, creationflags=flags)
            for raw in process.stdout:
                line = raw.decode('utf-8', errors='replace').rstrip()
                with self.lock:
                    self.lines.append(line)
            code = process.wait()
        except OSError as exc:
            with self.lock:
                self.lines.append(f'无法启动发布脚本：{exc}')
            code = -1
        with self.lock:
            self.code, self.state = code, 'success' if code == 0 else 'failed'


# ------------------------------------------------------------------ server

class Admin(BaseHTTPRequestHandler):
    token = ''
    port = 0
    preview = None
    publisher = None
    max_body = 200 * 1024 * 1024

    def log_message(self, format, *args):
        if self.path.startswith('/api/') and not self.path.startswith(('/api/status',)):
            sys.stderr.write('%s %s\n' % (self.command, self.path))

    def send(self, status, payload, kind='application/json; charset=utf-8'):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, default=str).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.end_headers()
        self.wfile.write(data)

    def trusted(self):
        # Rejects DNS-rebinding hosts and requests from other origins.
        host = self.headers.get('Host', '')
        return host in {f'127.0.0.1:{self.port}', f'localhost:{self.port}'}

    def authorised(self):
        return self.trusted() and secrets.compare_digest(self.headers.get('X-Admin-Token', ''), self.token)

    def do_GET(self):
        if not self.trusted():
            return self.send(403, {'error': 'forbidden host'})
        parts = urlsplit(self.path)
        if parts.path in {'/', '/index.html'}:
            page = (HERE / 'admin' / 'index.html').read_text(encoding='utf-8').replace('__ADMIN_TOKEN__', self.token)
            return self.send(200, page.encode('utf-8'), 'text/html; charset=utf-8')
        if not parts.path.startswith('/api/'):
            return self.not_admin_page(parts)
        if not self.authorised():
            return self.send(403, {'error': '令牌无效，请从启动时打开的页面访问。'})
        query = {key: values[-1] for key, values in parse_qs(parts.query).items()}
        routes = {
            '/api/state': state,
            '/api/doc': lambda: load(query.get('path', '')),
            '/api/preview/ready': lambda: self.preview.ready(query.get('url', '')),
            '/api/status': lambda: {'preview': self.preview.status(), 'publish': self.publisher.status(), 'pending': pending()},
        }
        self.dispatch(routes.get(parts.path))

    def do_POST(self):
        if not self.authorised():
            return self.send(403, {'error': '令牌无效，请从启动时打开的页面访问。'})
        length = int(self.headers.get('Content-Length') or 0)
        if length > self.max_body:
            return self.send(413, {'error': '上传内容过大（上限 200MB）。'})
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except json.JSONDecodeError:
            return self.send(400, {'error': '请求格式错误。'})
        b = body.get
        writes = {
            '/api/upload': lambda: upload(b('section'), b('category'), b('slug'), b('meta'), b('markdown'), b('attachments'), b('filename')),
            '/api/doc/save': lambda: save(b('path'), b('meta'), b('body')),
            '/api/doc/move': lambda: move(b('path'), b('category'), b('slug')),
            '/api/doc/delete': lambda: {'message': delete(b('path'), b('force'))},
            '/api/doc/attach': lambda: attach(b('path'), b('attachments'), b('overwrite')),
            '/api/category/create': lambda: {'path': create_category(b('section'), b('name'))},
            '/api/category/rename': lambda: {'path': rename_category(b('section'), b('old'), b('new'))},
            '/api/category/delete': lambda: delete_category(b('section'), b('name')) or {},
            '/api/trash/restore': lambda: {'message': restore(b('id'))},
        }
        others = {
            '/api/check': check,
            '/api/preview/start': self.preview.start,
            '/api/preview/stop': self.preview.stop,
            '/api/publish': lambda: self.publisher.start(b('message')),
        }
        route = urlsplit(self.path).path
        if route in writes:
            with WRITE_LOCK:
                return self.dispatch(writes[route])
        self.dispatch(others.get(route))

    def not_admin_page(self, parts):
        """Blog URLs opened on the admin port go to the same page in the live preview."""
        preview = self.preview
        if preview is not None and preview.online():
            target = f'http://127.0.0.1:{preview.port}/{parts.path.lstrip("/")}' + (f'?{parts.query}' if parts.query else '')
            self.send_response(302)
            self.send_header('Location', target)
            self.send_header('Content-Length', '0')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            return
        page = ('<!doctype html><meta charset="utf-8"><title>博客管理台</title>'
                '<body style="font:15px/1.7 system-ui,sans-serif;max-width:36rem;margin:12vh auto;padding:0 16px">'
                '<h1 style="font-weight:500">这里是博客管理台</h1>'
                '<p>这个地址只提供管理页面，不提供博客文章。本地实时预览还没有运行，请在管理台顶部点「启动预览」。</p>'
                '<p><a href="/">打开管理台</a></p></body>')
        self.send(404, page.encode('utf-8'), 'text/html; charset=utf-8')

    def dispatch(self, handler):
        if handler is None:
            return self.send(404, {'error': 'not found'})
        try:
            self.send(200, handler())
        except (ValueError, OSError, yaml.YAMLError, StopIteration) as exc:
            self.send(400, {'error': str(exc) or exc.__class__.__name__})
        except Exception as exc:  # surface unexpected failures in the UI, keep the server alive
            traceback.print_exc()
            self.send(500, {'error': f'内部错误：{exc}'})


def serve(port=8765, preview_port=8000, open_browser=True, with_preview=True):
    Admin.token = secrets.token_urlsafe(24)
    Admin.preview = Preview(preview_port)
    Admin.publisher = Publisher()
    chosen = free_port(port)
    if chosen != port:
        print(f'端口 {port} 已被其他程序占用，管理台改用 {chosen}。', flush=True)
    server = ThreadingHTTPServer(('127.0.0.1', chosen), Admin)
    Admin.port = server.server_address[1]
    atexit.register(Admin.preview.stop)
    if with_preview:
        Admin.preview.start()
        if Admin.preview.port != preview_port:
            print(f'端口 {preview_port} 已被其他程序占用，本地预览改用 {Admin.preview.port}。', flush=True)
    url = f'http://127.0.0.1:{Admin.port}/'
    print(f'博客管理台：{url}\n本地预览：http://127.0.0.1:{Admin.preview.port}/（保存后自动刷新）\n按 Ctrl+C 退出。', flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        Admin.preview.stop()


def main():
    cli = argparse.ArgumentParser(description='Local blog admin console.')
    cli.add_argument('--port', type=int, default=8765)
    cli.add_argument('--preview-port', type=int, default=8000)
    cli.add_argument('--no-browser', action='store_true')
    cli.add_argument('--no-preview', action='store_true', help='Do not start mkdocs serve automatically.')
    args = cli.parse_args()
    serve(args.port, args.preview_port, not args.no_browser, not args.no_preview)


if __name__ == '__main__':
    main()
