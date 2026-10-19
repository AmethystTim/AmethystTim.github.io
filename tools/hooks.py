"""Generate navigation, library and home collections directly from the files."""
import html
import posixpath
import re
from datetime import date
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import yaml
from mkdocs.plugins import event_priority
from mkdocs.utils import get_relative_url

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = [('课程笔记', '课程笔记', 'NOTES', '{ }', '从数学到计算机，给知识搭一个框架。'),
            ('Paper', '论文阅读', 'RESEARCH', '∑', '走进论文里的方法，也追问背后的为什么。'),
            ('比赛', '竞赛手记', 'PROJECTS', '↗', '从一道问题出发，在实践里寻找答案。'),
            ('blog', '生活随笔', 'JOURNAL', '✳', '技术之外，留一点空间给生活和自己。')]
RECORDS = []


def metadata(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    meta = {}
    body = text
    if text.startswith('---\n'):
        _, front, body = text.split('---', 2)
        meta = yaml.safe_load(front) or {}
    heading = re.search(r'^#\s+(.+)$', body, re.M)
    filename = Path(path).stem
    fallback = Path(path).parent.name if filename in {'index', 'i'} else filename
    return meta, str(meta.get('title') or (heading[1].strip() if heading else fallback)), body


def leaves(items):
    for item in items:
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for value in item.values():
                yield from leaves(value if isinstance(value, list) else [value])


def tree(paths, prefix=''):
    result = []
    groups = {}
    for path in sorted(paths):
        remainder = path[len(prefix):]
        if '/' not in remainder:
            result.append(path)
        else:
            folder = remainder.split('/')[0]
            groups.setdefault(folder, []).append(path)
    for folder, children in groups.items():
        if len(children) == 1 and children[0].endswith('/index.md'):
            result.append({folder: children[0]})
        else:
            result.append({folder: tree(children, prefix + folder + '/')})
    return result


def on_config(config):
    docs = Path(config.docs_dir)
    existing = {path.relative_to(docs).as_posix() for path in docs.rglob('*.md')}
    curated = yaml.safe_load((ROOT / 'config/navigation.yml').read_text(encoding='utf-8'))['nav']

    def prune(items):
        output = []
        for item in items:
            if isinstance(item, str):
                if item in existing:
                    output.append(item)
            else:
                for label, value in item.items():
                    children = prune(value if isinstance(value, list) else [value])
                    if children:
                        output.append({label: children if isinstance(value, list) else children[0]})
        return output

    # The original sequence and article titles are preserved within each section.
    nav = [{'首页': 'index.md'}, {'文档库': 'library/index.md'}, {'随笔': ['blog/index.md']}]
    legacy = dict(pair for item in curated if isinstance(item, dict) for pair in item.items())
    for key, label, old_label in [('课程笔记', '课程笔记', 'Notes'), ('Paper', '论文阅读', 'Thesis'), ('比赛', '竞赛手记', 'Contests')]:
        items = prune(legacy.get(old_label, []))
        covered = set(leaves(items))
        missing = [path for path in existing if path.startswith(key + '/') and path not in covered]
        items.extend(tree(missing, key + '/'))
        nav.append({label: items})
    reserved = set(leaves(nav)) | {'about/index.md'}
    other = [path for path in existing if path not in reserved and not path.startswith(('blog/',))]
    if other:
        nav.append({'更多记录': tree(other)})
    nav.append({'关于': 'about/index.md'})
    config.nav = nav
    return config


@event_priority(-100)
def on_files(files, config):
    global RECORDS
    RECORDS = []
    for file in files.documentation_pages():
        path = file.src_uri
        section = next((key for key, *_ in SECTIONS if path.startswith(key + '/')), None)
        if not section or (section == 'blog' and not path.startswith('blog/posts/')):
            continue
        # Synthetic blog archives/categories aren't source documents.
        if not (Path(config.docs_dir) / path).is_file():
            continue
        meta, title, body = metadata(file.abs_src_path)
        if meta.get('draft') or path in {f'{key}/index.md' for key, *_ in SECTIONS}:
            continue
        stamp = meta.get('date', '')
        if isinstance(stamp, dict):
            stamp = stamp.get('created', '')
        categories = meta.get('categories') or [dict((row[0], row[1]) for row in SECTIONS)[section]]
        RECORDS.append({'path': path, 'title': title, 'url': file.url, 'section': section, 'date': str(stamp), 'category': str(categories[0]), 'search': (title + ' ' + path + ' ' + ' '.join(map(str, meta.get('tags', [])))).lower()})
    collections = []
    for key, name, en, symbol, description in SECTIONS:
        collections.append({'name': name, 'en': en, 'symbol': symbol, 'description': description, 'url': key + '/', 'count': sum(record['section'] == key for record in RECORDS)})
    config.extra['collections'] = collections
    config.extra['recent_posts'] = sorted((record for record in RECORDS if record['section'] == 'blog'), key=lambda r: r['date'], reverse=True)[:3]
    return files


def on_page_markdown(markdown, page, config, files):
    path = page.file.src_uri
    if path.startswith('blog/posts/'):
        # MkDocs rewrites Markdown image URLs, but raw HTML image URLs need
        # their own rebasing after the blog plugin assigns the dated post URL.
        pattern = re.compile(r'(?P<prefix>\b(?:src|href)\s*=\s*["\'])(?P<url>[^"\']+)(?P<suffix>["\'])', re.I)
        def rebase_html(match):
            parts = urlsplit(match['url'])
            if parts.scheme or parts.netloc or not parts.path:
                return match[0]
            target_path = posixpath.normpath(posixpath.join(posixpath.dirname(path), unquote(parts.path)))
            target = files.get_file_from_path(target_path)
            if not target:
                return match[0]
            url = get_relative_url(target.url, page.url)
            if parts.fragment:
                url += '#' + parts.fragment
            return match['prefix'] + url + match['suffix']
        markdown = pattern.sub(rebase_html, markdown)
    if path == 'library/index.md':
        escape = html.escape
        output = ['<div data-library>', '<label for="library-search">按标题、路径或标签查找</label>', '<input id="library-search" class="library-search" data-library-search type="search" placeholder="试试：操作系统、Diffusion、数学建模…" autocomplete="off">', '<div class="library-filters" role="group" aria-label="文档分类">', '<button class="library-filter" type="button" data-category="all" aria-pressed="true">全部</button>']
        for key, name, *_ in SECTIONS:
            output.append(f'<button class="library-filter" type="button" data-category="{key}" aria-pressed="false">{name}</button>')
        output.extend(['</div>', f'<p class="library-count" data-library-count role="status" aria-live="polite">{len(RECORDS)} 篇文档</p>', '<ul class="library-list">'])
        for record in sorted(RECORDS, key=lambda r: (r['section'], r['path'])):
            url = get_relative_url(record['url'], page.url)
            edit = config.repo_url.rstrip('/') + '/edit/main/docs/' + quote(record['path'], safe='/')
            output.append(f'<li class="library-item" data-library-item data-section="{record["section"]}" data-search="{escape(record["search"], quote=True)}"><div><a class="library-item-title" href="{escape(url, quote=True)}">{escape(record["title"])}</a><div class="library-item-path">{escape(record["path"])}</div></div><a class="library-item-edit" href="{edit}" target="_blank" rel="noopener">编辑 ↗</a></li>')
        output.extend(['</ul>', '<p class="library-empty" data-library-empty hidden>没有匹配的文档。试试其他关键词或切换分类。</p>', '</div>'])
        return markdown + '\n\n' + '\n'.join(output)
    if path in {'课程笔记/index.md', 'Paper/index.md', '比赛/index.md'}:
        key = path.split('/')[0]
        records = [record for record in RECORDS if record['section'] == key]
        markdown += '\n\n## 全部记录\n\n'
        for record in sorted(records, key=lambda r: r['path']):
            relative = Path(record['path']).relative_to(key).as_posix()
            markdown += f'- [{record["title"]}](<{relative}>)\n'
        return markdown
    return markdown
