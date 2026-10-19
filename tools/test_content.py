"""Integration checks for reversible content operations, isolated from real docs."""
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import content


class ContentOperations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original_root, self.original_docs = content.ROOT, content.DOCS
        content.ROOT = Path(self.temp.name)
        content.DOCS = content.ROOT / 'docs'
        content.DOCS.mkdir()

    def tearDown(self):
        content.ROOT, content.DOCS = self.original_root, self.original_docs
        self.temp.cleanup()

    def run_command(self, *args):
        parsed = content.parser().parse_args(args)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            parsed.action(parsed)
        return output.getvalue()

    def test_move_updates_incoming_links_and_keeps_local_image(self):
        self.run_command('new', '--section', 'papers', '--path', 'original', '--title', 'Original')
        folder = content.DOCS / 'Paper/original'
        (folder / 'image.png').write_bytes(b'test-image')
        (folder / 'index.md').write_text('# Original\n![Figure](image.png)\n[Other](../other/index.md)\n$$[f](x)$$\n', encoding='utf-8')
        other = content.DOCS / 'Paper/other'
        other.mkdir()
        (other / 'index.md').write_text('# Other\n[Original](../original/index.md#figure)\n', encoding='utf-8')
        self.run_command('move', 'Paper/original', 'Paper/nested/renamed')
        moved = content.DOCS / 'Paper/nested/renamed'
        self.assertEqual((moved / 'image.png').read_bytes(), b'test-image')
        self.assertIn('![Figure](image.png)', (moved / 'index.md').read_text())
        self.assertIn('../../other/index.md', (moved / 'index.md').read_text())
        self.assertIn('$$[f](x)$$', (moved / 'index.md').read_text())
        self.assertIn('../nested/renamed/index.md#figure', (other / 'index.md').read_text())
        self.run_command('check')

    def test_delete_and_restore_include_images_without_data_loss(self):
        self.run_command('new', '--section', 'notes', '--path', 'sample', '--title', 'Sample')
        folder = content.DOCS / '课程笔记/sample'
        (folder / 'image.png').write_bytes(b'test-image')
        original = (folder / 'index.md').read_bytes()
        self.run_command('delete', str(folder))
        self.assertFalse(folder.exists())
        trash = next((content.ROOT / '.trash').iterdir())
        self.assertEqual((trash / 'content/image.png').read_bytes(), b'test-image')
        self.run_command('restore', trash.name)
        self.assertEqual((folder / 'index.md').read_bytes(), original)
        self.assertEqual((folder / 'image.png').read_bytes(), b'test-image')

    def test_guards_reject_traversal_collection_deletion_and_linked_deletion(self):
        with self.assertRaises(ValueError):
            self.run_command('new', '--section', 'papers', '--path', '../../outside', '--title', 'No')
        self.run_command('new', '--section', 'papers', '--path', 'sample', '--title', 'Sample')
        (content.DOCS / 'Paper/index.md').write_text('# Collection\n[Sample](sample/index.md)', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.run_command('delete', 'Paper/index.md')
        with self.assertRaises(ValueError):
            self.run_command('delete', 'Paper/sample')
        self.assertTrue((content.DOCS / 'Paper/sample/index.md').exists())

    def test_posts_start_as_drafts_and_live_in_their_own_directory(self):
        self.run_command('new', '--section', 'posts', '--path', 'my-post', '--title', 'My post')
        text = (content.DOCS / 'blog/posts/my-post/my-post.md').read_text(encoding='utf-8')
        self.assertIn('draft: true', text)
        self.assertIn('<!-- more -->', text)


if __name__ == '__main__':
    unittest.main()
