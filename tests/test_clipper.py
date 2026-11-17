import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clip import build_video_filter, source_identity, select_clip_start, DEFAULT_MAX_SECONDS


class ClipperReleaseTests(unittest.TestCase):
    def test_landscape_gameplay_stays_centered_without_cropping(self):
        graph = build_video_filter('captions.ass')
        self.assertIn('[b]scale=1080:-2[fg]', graph)
        self.assertIn('overlay=(W-w)/2:(H-h)/2', graph)
        self.assertIn('boxblur=30:5', graph)

    def test_unknown_creator_does_not_require_a_repository_allowlist(self):
        source = source_identity({'uploader': 'Creator', 'channel_id': 'UC-new'}, test=False)
        self.assertEqual(source['name'], 'Creator')
        self.assertFalse(source['test'])

    def test_live_capture_defaults_to_running_until_stopped(self):
        self.assertEqual(DEFAULT_MAX_SECONDS, 0)

    def test_highlight_starts_before_the_peak_and_stays_inside_the_chunk(self):
        self.assertAlmostEqual(select_clip_start(21.3, 6, 30, 90), 15.3)
        self.assertEqual(select_clip_start(2, 6, 30, 90), 0)
        self.assertEqual(select_clip_start(89, 6, 30, 90), 60)


if __name__ == '__main__':
    unittest.main()
