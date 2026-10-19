"""Shared Markdown/HTML destination handling for migration and document moves."""
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

MARKDOWN = re.compile(r'(?P<prefix>!?\[[^\]\n]*\]\()(?P<url><[^>\n]+>|(?:[^()\s]|\([^()\n]*\))+)(?P<suffix>(?:\s+["\'][^\n]*?["\'])?\))')
HTML = re.compile(r'(?P<prefix>\b(?:src|href)\s*=\s*["\'])(?P<url>[^"\']+)(?P<suffix>["\'])', re.I)
REFERENCE = re.compile(r'(?P<prefix>^\s{0,3}\[[^\]\n]+\]:\s*)(?P<url><[^>\n]+>|\S+)(?P<suffix>)', re.M)
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp', '.avif', '.bmp', '.pdf'}
PROTECTED = re.compile(r'(```.*?```|~~~.*?~~~|\$\$.*?\$\$|(?<!\\)\$[^\n$]+\$|`[^`\n]*`)', re.S)


def rewrite(text, callback):
    def replace(match):
        url = match['url']
        angle = url.startswith('<') and url.endswith('>')
        value = callback(url[1:-1] if angle else url)
        if angle or ' ' in value:
            value = '<' + value + '>'
        return match['prefix'] + value + match['suffix']
    chunks = PROTECTED.split(text)
    for index in range(0, len(chunks), 2):
        for pattern in (MARKDOWN, HTML, REFERENCE):
            chunks[index] = pattern.sub(replace, chunks[index])
    return ''.join(chunks)


def destinations(text):
    for chunk in PROTECTED.split(text)[::2]:
        for pattern in (MARKDOWN, HTML, REFERENCE):
            for match in pattern.finditer(chunk):
                yield match['url'].strip('<>')


def resolve_local(url, source, docs):
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path:
        return None
    raw = unquote(parts.path)
    target = (docs / raw.lstrip('/') if raw.startswith('/') else source.parent / raw).resolve()
    return target if target.is_relative_to(docs.resolve()) else None
