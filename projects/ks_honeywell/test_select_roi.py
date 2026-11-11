"""ROI persistence and OpenCV selector tests without a display dependency."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

from inference_manual_pickup import inside_roi, load_roi, save_roi
from select_roi import select_from_video, selection_to_source


class RoiTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.mp4"
        self.source.touch()
        self.output = self.root / "roi.json"
        self.cv2 = MagicMock()
        self.cap = self.cv2.VideoCapture.return_value
        self.cap.isOpened.return_value = True
        frame = MagicMock()
        frame.shape = (1080, 1920, 3)
        self.cap.read.return_value = (True, frame)

    def select(self, frame_index=0):
        with redirect_stdout(io.StringIO()):
            return select_from_video(self.source, self.output, frame_index,
                                     cv2_module=self.cv2)

    def test_selector_scales_coordinates_and_saves_original_dimensions(self):
        self.cv2.selectROI.return_value = (100, 50, 400, 300)
        self.assertTrue(self.select(frame_index=10))
        self.cap.set.assert_called_once_with(self.cv2.CAP_PROP_POS_FRAMES, 10)
        self.assertEqual(load_roi(self.output, 1920, 1080), (150, 75, 750, 525))
        self.cv2.resize.assert_called_once()
        self.assertEqual(self.cv2.resize.call_args.args[1], (1280, 720))
        self.cap.release.assert_called_once()
        self.cv2.destroyWindow.assert_called_once()

    def test_cancel_keeps_existing_saved_roi(self):
        save_roi(self.output, (10, 20, 30, 40), 1920, 1080)
        before = self.output.read_bytes()
        self.cv2.selectROI.return_value = (0, 0, 0, 0)
        self.assertFalse(self.select())
        self.assertEqual(self.output.read_bytes(), before)
        self.cv2.destroyWindow.assert_called_once()

    def test_load_rejects_missing_malformed_mismatched_and_invalid_roi(self):
        with self.assertRaisesRegex(ValueError, "Cannot read ROI"):
            load_roi(self.output, 640, 480)
        for content in ['{', '[]',
                        '{"frame_width":640,"frame_height":480,"roi_xyxy":[0,0,0,20]}',
                        '{"frame_width":640,"frame_height":480,"roi_xyxy":[0,0,641,20]}',
                        '{"frame_width":640,"frame_height":480,"roi_xyxy":[0,0,5.5,20]}']:
            self.output.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_roi(self.output, 640, 480)
        save_roi(self.output, (0, 0, 640, 480), 640, 480)
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            load_roi(self.output, 1280, 720)

    def test_containment_includes_boundary_but_not_partial_boxes(self):
        roi = (10, 20, 100, 200)
        self.assertTrue(inside_roi(roi, roi))
        self.assertTrue(inside_roi((20, 30, 90, 190), roi))
        self.assertFalse(inside_roi((9, 30, 90, 190), roi))
        self.assertFalse(inside_roi((20, 30, 90, 201), roi))
        self.assertFalse(inside_roi((20, 30, 20, 190), roi))
        self.assertTrue(inside_roi((20, 30, 90, 190), None))

    def test_selector_read_failure_releases_capture(self):
        self.cap.read.return_value = (False, None)
        with self.assertRaisesRegex(ValueError, "Cannot read frame"):
            self.select()
        self.cap.release.assert_called_once()
        self.cv2.selectROI.assert_not_called()

    def test_selector_gui_failure_closes_window(self):
        self.cv2.selectROI.side_effect = RuntimeError("GUI unavailable")
        with self.assertRaisesRegex(RuntimeError, "GUI unavailable"):
            self.select()
        self.cv2.destroyWindow.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_full_preview_maps_to_full_frame_and_cancel_is_empty(self):
        self.assertEqual(selection_to_source((0, 0, 1280, 720), (1280, 720), (1920, 1080)),
                         (0, 0, 1920, 1080))
        self.assertIsNone(selection_to_source((0, 0, 0, 100), (640, 480), (640, 480)))


if __name__ == "__main__":
    unittest.main()
