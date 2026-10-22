"""Checks for the local admin console, isolated from the real docs."""
import base64
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import admin
import content

PNG = base64.b64encode(b'\x89PNG fake').decode()


class AdminOperations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original_root, self.original_docs = content.ROOT, content.DOCS
        content.ROOT = Path(self.temp.name)
        content.DOCS = content.ROOT / 'docs'
        for folder in ['Paper', '课程笔记', '比赛', 'blog/posts', 'library']:
            (content.DOCS / folder).mkdir(parents=True)
        (content.ROOT / 'config').mkdir()
        (content.ROOT / 'mkdocs.yml').write_text('site_url: https://example.test/\n', encoding='utf-8')

    def tearDown(self):
        content.ROOT, content.DOCS = self.original_root, self.original_docs
        self.temp.cleanup()

    def read(self, rel):
        return (content.DOCS / rel).read_text(encoding='utf-8')

    def test_upload_merges_metadata_and_relinks_attachments(self):
        markdown = ('---\r\ntitle: Old\r\ntags: [a]\r\n---\r\n# Heading\r\n\r\n'
                    '![fig](assets/fig.png)\n![abs](C:\\Users\\me\\Pictures\\shot.png)\n![gone](missing.png)\n')
        result = admin.upload('papers', '图像恢复', 'demo', {'title': 'New', 'tags': [], 'date': '2026-01-02'},
                              markdown, [{'name': 'fig.png', 'data': PNG}, {'name': 'shot.png', 'data': PNG}], 'demo.md')
        self.assertEqual(result['path'], 'Paper/图像恢复/demo/index.md')
        self.assertEqual(result['url'], 'Paper/%E5%9B%BE%E5%83%8F%E6%81%A2%E5%A4%8D/demo/')
        text = self.read(result['path'])
        meta, body = admin.split_front(text)
        self.assertEqual(meta['title'], 'New')
        self.assertEqual(meta['tags'], ['a'])
        self.assertEqual(str(meta['date']), '2026-01-02')
        self.assertIn('![fig](fig.png)', body)
        self.assertIn('![abs](shot.png)', body)
        self.assertNotIn('\r', text)
        self.assertTrue((content.DOCS / 'Paper/图像恢复/demo/fig.png').is_file())
        self.assertEqual(result['warnings'], ['引用的文件不存在：missing.png'])
        with self.assertRaises(ValueError):
            admin.upload('papers', '图像恢复', 'demo', {}, '# again', [], 'demo.md')
        with self.assertRaises(ValueError):
            admin.upload('papers', '../escape', 'x', {}, '# x', [], 'x.md')

    def test_upload_post_uses_blog_conventions(self):
        result = admin.upload('posts', '', 'hello', {'title': 'Hello World', 'date': '2026-10-22', 'categories': ['生活'], 'draft': False},
                              'Intro\n', [], 'hello.md')
        self.assertEqual(result['path'], 'blog/posts/hello/hello.md')
        self.assertEqual(result['url'], 'blog/2026/10/22/hello-world/')
        self.assertEqual(result['online'], 'https://example.test/blog/2026/10/22/hello-world/')
        meta, _ = admin.split_front(self.read(result['path']))
        self.assertEqual(meta['authors'], ['dinorex'])
        self.assertIs(meta['draft'], False)
        self.assertTrue(any('more' in note for note in result['warnings']))

    def test_save_syncs_heading_and_curated_navigation_label(self):
        folder = content.DOCS / 'Paper/topic'
        folder.mkdir()
        (folder / 'index.md').write_text('# Topic\n\nBody\n', encoding='utf-8')
        nav = content.ROOT / 'config/navigation.yml'
        nav.write_text('nav:\n- Thesis:\n  - Topic: Paper/topic/index.md\n', encoding='utf-8')
        admin.save('Paper/topic/index.md', {'title': 'Topic', 'tags': ['x'], 'description': ''})
        self.assertEqual(self.read('Paper/topic/index.md'), '---\ntags:\n- x\n---\n\n# Topic\n\nBody\n')
        admin.save('Paper/topic/index.md', {'title': 'Renamed: v2', 'tags': []})
        meta, body = admin.split_front(self.read('Paper/topic/index.md'))
        self.assertEqual(meta, {'title': 'Renamed: v2'})
        self.assertTrue(body.startswith('# Renamed: v2\n'))
        self.assertIn("- 'Renamed: v2': Paper/topic/index.md", nav.read_text(encoding='utf-8'))

    def test_legacy_i_folder_is_titled_after_its_parent(self):
        folder = content.DOCS / 'Paper/VAE/i'
        folder.mkdir(parents=True)
        (folder / 'index.md').write_text('text\n', encoding='utf-8')
        record = admin.record(folder / 'index.md')
        self.assertEqual((record['title'], record['category'], record['slug']), ('VAE', 'VAE', 'i'))

    def test_folder_category_lifecycle_moves_articles_and_links(self):
        self.assertEqual(admin.create_category('papers', '旧分类/子类'), '旧分类/子类')
        admin.upload('papers', '旧分类/子类', 'a', {'title': 'A'}, '# A\n![x](x.png)\n', [{'name': 'x.png', 'data': PNG}], 'a.md')
        admin.upload('papers', '', 'b', {'title': 'B'}, '# B\n[A](../旧分类/子类/a/index.md)\n', [], 'b.md')
        nav = content.ROOT / 'config/navigation.yml'
        nav.write_text('nav:\n- Thesis:\n  - A: Paper/旧分类/子类/a/index.md\n', encoding='utf-8')
        listed = {item['path']: item['count'] for item in admin.categories('papers')}
        self.assertEqual(listed, {'旧分类': 1, '旧分类/子类': 1})
        with self.assertRaises(ValueError):
            admin.delete_category('papers', '旧分类')
        admin.rename_category('papers', '旧分类', '新分类')
        self.assertFalse((content.DOCS / 'Paper/旧分类').exists())
        self.assertTrue((content.DOCS / 'Paper/新分类/子类/a/x.png').is_file())
        self.assertTrue((content.DOCS / 'Paper/新分类/子类/.gitkeep').is_file())
        self.assertIn('../新分类/子类/a/index.md', unquote(self.read('Paper/b/index.md')))
        self.assertIn('Paper/新分类/子类/a/index.md', nav.read_text(encoding='utf-8'))
        admin.move('Paper/新分类/子类/a/index.md', '', 'a2')
        admin.delete_category('papers', '新分类')
        self.assertFalse((content.DOCS / 'Paper/新分类').exists())

    def test_moves_record_redirects_and_build_writes_them(self):
        import hooks
        admin.upload('papers', '', 'a', {'title': 'A'}, '# A\n', [], 'a.md')
        admin.move('Paper/a/index.md', '图像', 'a')
        admin.move('Paper/图像/a/index.md', '', 'b')
        admin.upload('papers', '', 'a', {'title': 'New A'}, '# A again\n', [], 'a.md')
        admin.add_redirect('Paper/b', 'Paper/a')  # the old page is back at its URL
        redirects = (content.ROOT / 'config/redirects.yml')
        original = hooks.ROOT
        hooks.ROOT = content.ROOT
        try:
            self.assertEqual(hooks.redirects(), {'Paper/图像/a': 'Paper/a', 'Paper/b': 'Paper/a'})
            site = content.ROOT / 'site'
            (site / 'Paper/a').mkdir(parents=True)
            (site / 'Paper/a/index.html').write_text('real', encoding='utf-8')
            hooks.on_post_build(type('Config', (), {'site_dir': str(site), 'site_url': 'https://example.test/'})())
        finally:
            hooks.ROOT = original
        stub = (site / 'Paper/图像/a/index.html').read_text(encoding='utf-8')
        self.assertIn('url=../../a/', stub)
        self.assertIn('href="https://example.test/Paper/a/"', stub)
        self.assertEqual((site / 'Paper/a/index.html').read_text(encoding='utf-8'), 'real')
        self.assertTrue(redirects.read_text(encoding='utf-8').startswith('#'))

    def test_library_groups_follow_the_docs_folders(self):
        import hooks
        self.assertEqual(hooks.group_of('课程笔记/大三上/操作系统/进程/index.md', '课程笔记', '课程笔记'), '大三上')
        self.assertEqual(hooks.group_of('Paper/数学基础/VAE/index.md', 'Paper', '论文阅读'), '数学基础')
        self.assertEqual(hooks.group_of('Paper/VAE/index.md', 'Paper', '论文阅读'), hooks.UNGROUPED)
        self.assertEqual(hooks.group_of('blog/posts/p/p.md', 'blog', '生活'), '生活')
        records = [{'group': g} for g in ['大三上', hooks.UNGROUPED, '大二下', '大一', '大二上', '大三上']]
        self.assertEqual(hooks.groups(records), [('大一', 1), ('大二上', 1), ('大二下', 1), ('大三上', 2), (hooks.UNGROUPED, 1)])

    def test_post_categories(self):
        admin.create_category('posts', '空分类')
        admin.upload('posts', '', 'p', {'date': '2026-01-01', 'categories': ['旧']}, '# P\n<!-- more -->\n', [], 'p.md')
        self.assertEqual({c['path']: c['count'] for c in admin.categories('posts')}, {'空分类': 0, '旧': 1})
        with self.assertRaises(ValueError):
            admin.delete_category('posts', '旧')
        admin.rename_category('posts', '旧', '新')
        meta, _ = admin.split_front(self.read('blog/posts/p/p.md'))
        self.assertEqual(meta['categories'], ['新'])
        admin.delete_category('posts', '空分类')
        self.assertEqual([c['path'] for c in admin.categories('posts')], ['新'])

    def test_delete_and_restore_round_trip(self):
        admin.upload('contests', '', 'c', {'title': 'C'}, '# C\n', [], 'c.md')
        admin.delete('比赛/c/index.md')
        self.assertFalse((content.DOCS / '比赛/c').exists())
        [entry] = admin.trash()
        admin.restore(entry['id'])
        self.assertTrue((content.DOCS / '比赛/c/index.md').is_file())


class AdminServer(unittest.TestCase):
    def setUp(self):
        admin.Admin.token = 'secret-token'
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), admin.Admin)
        admin.Admin.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f'http://127.0.0.1:{admin.Admin.port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def status(self, path, headers=None, data=None):
        request = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read()

    def test_api_requires_token_and_local_host(self):
        self.assertEqual(self.status('/api/state')[0], 403)
        self.assertEqual(self.status('/api/check', data=b'{}')[0], 403)
        self.assertEqual(self.status('/', {'Host': 'evil.test'})[0], 403)
        code, body = self.status('/')
        self.assertEqual(code, 200)
        self.assertIn(b'secret-token', body)
        code, body = self.status('/api/nope', {'X-Admin-Token': 'secret-token'})
        self.assertEqual((code, json.loads(body)), (404, {'error': 'not found'}))
        code, body = self.status('/Paper/')  # a blog URL opened on the admin port
        self.assertEqual(code, 404)
        self.assertIn('博客管理台'.encode(), body)

    def test_ports_held_on_ipv6_only_count_as_busy(self):
        import socket
        try:
            holder = socket.socket(socket.AF_INET6)
            holder.bind(('::', 0))
        except OSError:
            self.skipTest('IPv6 unavailable')
        with holder:
            holder.listen()
            port = holder.getsockname()[1]
            self.assertTrue(admin.port_open(port))
            self.assertNotEqual(admin.free_port(port), port)


if __name__ == '__main__':
    unittest.main()
