import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clip import (
    build_video_filter, choose_crop_offset, choose_vertical_game_roi, reaction_score,
    should_apply_smart_crop,
    source_identity, select_clip_start,
    DEFAULT_MAX_SECONDS, pick_moments,
)


class ClipperReleaseTests(unittest.TestCase):
    def test_landscape_gameplay_stays_centered_without_cropping(self):
        graph = build_video_filter('captions.ass')
        self.assertIn('[b]scale=1080:-2[fg]', graph)
        self.assertIn('overlay=(W-w)/2:(H-h)/2', graph)
        self.assertIn('boxblur=30:5', graph)

    def test_smart_crop_moves_away_from_a_persistent_left_facecam(self):
        samples = [[(180, 90, 70, 70)] for _ in range(4)]
        offset = choose_crop_offset(1920, 1080, samples)
        self.assertGreater(offset, 0.5)

    def test_smart_crop_moves_away_from_a_persistent_right_facecam(self):
        samples = [[(1670, 90, 70, 70)] for _ in range(4)]
        offset = choose_crop_offset(1920, 1080, samples)
        self.assertLess(offset, 0.5)

    def test_smart_crop_uses_safe_center_without_repeated_side_face_detections(self):
        self.assertEqual(choose_crop_offset(1920, 1080, [[(180, 90, 70, 70)]]), 0.5)
        self.assertEqual(choose_crop_offset(1920, 1080, [[], [], []]), 0.5)

    def test_portrait_stack_uses_gameplay_below_a_persistent_top_facecam(self):
        samples = [[(140, 30, 70, 60)] for _ in range(5)]
        roi = choose_vertical_game_roi(360, 640, samples, active_bottom=540)
        self.assertIsNotNone(roi)
        x, y, width, height = roi
        self.assertEqual((x, width), (0, 360))
        self.assertGreaterEqual(y, 110)
        self.assertLessEqual(y + height, 540)

    def test_portrait_without_a_persistent_facecam_keeps_the_full_source(self):
        self.assertIsNone(choose_vertical_game_roi(360, 640, [[], [], []], 540))

    def test_uncertain_face_detection_falls_back_to_full_source_framing(self):
        self.assertFalse(should_apply_smart_crop(True, 0.5, None))
        self.assertTrue(should_apply_smart_crop(True, 1.0, None))
        self.assertTrue(should_apply_smart_crop(True, 0.5, (0, 120, 360, 420)))
        self.assertFalse(should_apply_smart_crop(False, 1.0, None))

    def test_generic_filler_words_do_not_count_as_a_strong_reaction(self):
        filler = [(0, 0.2, word) for word in 'oh no no okay here we go'.split()]
        self.assertEqual(reaction_score(filler), 0)

    def test_clear_reaction_phrases_score_above_filler(self):
        strong = [(0, 0.2, word) for word in 'holy shit that was insane'.split()]
        self.assertGreater(reaction_score(strong), 2)

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

    def test_each_selected_clip_uses_the_peak_inside_its_own_window(self):
        audio = np.full(90 * 16000, 0.01, dtype=np.float32)
        audio[20 * 16000:22 * 16000] = 0.8
        audio[60 * 16000:62 * 16000] = 0.8
        with patch('clip.load_audio', return_value=audio), patch('clip.whisper_words', return_value=[]):
            chosen, _ = pick_moments('unused.ts', clip_len=30, top=2, min_score=0, lead_in=6)
        starts = [start for _, start in chosen]
        self.assertEqual(len(starts), 2)
        self.assertEqual(len(set(starts)), 2)
        self.assertTrue(any(14 <= start <= 16 for start in starts), starts)
        self.assertTrue(any(54 <= start <= 56 for start in starts), starts)


if __name__ == '__main__':
    unittest.main()
