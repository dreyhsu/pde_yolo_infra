"""Behavioral tests; no GPU, OpenCV, or model weights required."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

import inference_manual_pickup as pickup


class PickupLogicTests(unittest.TestCase):
    def test_overlap_requires_positive_area(self):
        box = (0, 0, 10, 10)
        self.assertFalse(pickup.intersects(box, (20, 0, 30, 10)))
        self.assertFalse(pickup.intersects(box, (10, 0, 20, 10)))
        self.assertFalse(pickup.intersects(box, (10, 10, 20, 20)))
        self.assertTrue(pickup.intersects(box, (9, 9, 20, 20)))
        self.assertTrue(pickup.intersects(box, (1, 1, 2, 2)))

    def test_short_contact_resets_timer(self):
        state = pickup.PickupState()
        for timestamp, contact in [(0, True), (.1, True), (.15, False), (.2, True), (.3, True)]:
            self.assertFalse(state.update(contact, timestamp, .2, .3))
        self.assertTrue(state.update(True, .4, .2, .3))
        self.assertEqual(state.count, 1)

    def test_sustained_contact_counts_once(self):
        state = pickup.PickupState()
        events = [state.update(True, index / 30, .2, .3) for index in range(90)]
        self.assertEqual(sum(events), 1)
        self.assertTrue(events[6])

    def test_release_and_repeated_pickup(self):
        state = pickup.PickupState()
        for timestamp in [0, .2]:
            state.update(True, timestamp, .2, .3)
        state.update(False, .3, .2, .3)
        state.update(False, .5, .2, .3)
        self.assertFalse(state.update(True, .55, .2, .3))
        self.assertFalse(state.update(True, .8, .2, .3))
        self.assertEqual(state.count, 1)
        state.update(False, .9, .2, .3)
        state.update(False, 1.2, .2, .3)
        self.assertTrue(state.armed)
        self.assertFalse(state.update(True, 1.3, .2, .3))
        self.assertTrue(state.update(True, 1.5, .2, .3))
        self.assertEqual(state.count, 2)

    def test_no_contact_never_fires_and_zero_duration_fires_immediately(self):
        state = pickup.PickupState()
        for timestamp in [0, 1, 2]:
            self.assertFalse(state.update(False, timestamp, .2, .3))
        self.assertEqual(state.count, 0)
        self.assertTrue(state.update(True, 3, 0, .3))

    def test_simultaneous_classes_and_duplicate_instances(self):
        manuals = [("manual1", .9, (0, 0, 10, 10)),
                   ("manual1", .8, (1, 1, 8, 8)),
                   ("manual2", .9, (0, 0, 10, 10)),
                   ("manual3", .9, (50, 50, 60, 60))]
        hands = [("hand", .9, (2, 2, 6, 6))]
        contacts = pickup.contact_classes(manuals, hands)
        self.assertEqual(contacts, {"manual1", "manual2"})
        states = {name: pickup.PickupState() for name in pickup.MANUALS}
        for timestamp in [0, .2, .5]:
            for name, state in states.items():
                state.update(name in contacts, timestamp, .2, .3)
        self.assertEqual([s.count for s in states.values()], [1, 1, 0, 0])
        self.assertEqual(pickup.contact_classes(manuals, []), set())
        self.assertEqual(pickup.contact_classes([], hands), set())

    def test_names_are_resolved_independent_of_order(self):
        names = {0: "other", 1: "manual4", 2: "manual2", 3: "manual1", 4: "manual3"}
        self.assertEqual(pickup.manual_class_ids(names),
                         {1: "manual4", 2: "manual2", 3: "manual1", 4: "manual3"})
        self.assertEqual(pickup.manual_class_ids(list(pickup.MANUALS))[0], "manual1")
        with self.assertRaisesRegex(ValueError, "manual3, manual4"):
            pickup.manual_class_ids({0: "manual1", 1: "manual2"})


class VideoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ["input.mp4", "manual.pt", "hand.pt"]:
            (self.root / name).touch()
        self.args = pickup.build_parser().parse_args([
            "--source", str(self.root / "input.mp4"),
            "--manual-weights", str(self.root / "manual.pt"),
            "--hand-weights", str(self.root / "hand.pt"),
            "--output", str(self.root / "out.mp4"), "--device", "cpu"])
        self.cv2 = MagicMock()
        self.cap = self.cv2.VideoCapture.return_value
        self.cap.isOpened.return_value = True
        self.cap.get.return_value = 30.0
        self.frame = MagicMock()
        self.frame.shape = (480, 640, 3)
        self.cap.read.side_effect = [(True, self.frame)] * 12 + [(False, None)]
        self.writer = self.cv2.VideoWriter.return_value
        self.writer.isOpened.return_value = True
        self.manual_model = MagicMock(task="detect")
        self.manual_model.names = dict(enumerate(pickup.MANUALS))
        self.hand_model = MagicMock(task="detect")
        for model, row in [(self.manual_model, [0, 0, 10, 10, .9, 0]),
                           (self.hand_model, [2, 2, 8, 8, .8, 0])]:
            result = MagicMock()
            result.boxes.data.cpu.return_value.tolist.return_value = [row]
            model.predict.return_value = [result]
        self.factory = MagicMock(side_effect=[self.manual_model, self.hand_model])

    def run_video(self):
        with redirect_stdout(io.StringIO()):
            return pickup.run(self.args, cv2_module=self.cv2, model_factory=self.factory)

    def test_pipeline_preserves_fps_dimensions_and_runs_both_models(self):
        states = self.run_video()
        self.assertEqual(states["manual1"].count, 1)
        self.assertEqual(self.manual_model.predict.call_count, 12)
        self.assertEqual(self.hand_model.predict.call_count, 12)
        self.assertEqual(self.writer.write.call_count, 12)
        self.assertEqual(self.cv2.VideoWriter.call_args.args[2:], (30.0, (640, 480)))
        self.assertEqual(self.manual_model.predict.call_args.kwargs["device"], "cpu")
        self.assertEqual(self.manual_model.predict.call_args.kwargs["classes"], [0, 1, 2, 3])
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()
        self.assertTrue(self.cv2.rectangle.called)
        texts = [call.args[1] for call in self.cv2.putText.call_args_list]
        self.assertIn("PICKUP: manual1", texts)
        self.assertTrue(any("manual1: 1 pickups" in text for text in texts))

    def test_missing_file_and_output_errors(self):
        (self.root / "manual.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Manual weights not found"):
            self.run_video()
        (self.root / "manual.pt").touch()
        self.args.output = self.args.source
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            self.run_video()
        self.args.output = str(self.root / "out.avi")
        with self.assertRaisesRegex(ValueError, ".mp4 extension"):
            self.run_video()
        self.args.output = str(self.root / "existing.mp4")
        Path(self.args.output).touch()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_video()

    def test_invalid_fps(self):
        for fps in [0, -1, float("nan"), float("inf")]:
            with self.subTest(fps=fps):
                self.cap.get.return_value = fps
                with self.assertRaisesRegex(ValueError, "Invalid source FPS"):
                    self.run_video()
        self.assertEqual(self.cap.release.call_count, 4)
        self.factory.assert_not_called()

    def test_unreadable_source(self):
        self.cap.isOpened.return_value = False
        with self.assertRaisesRegex(ValueError, "Cannot open source"):
            self.run_video()
        self.cap.release.assert_called_once()

    def test_empty_video(self):
        self.cap.read.side_effect = [(False, None)]
        with self.assertRaisesRegex(ValueError, "no readable frames"):
            self.run_video()
        self.cap.release.assert_called_once()

    def test_writer_failure_releases_resources(self):
        self.writer.isOpened.return_value = False
        with self.assertRaisesRegex(OSError, "Cannot create MP4 writer"):
            self.run_video()
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()

    def test_inference_failure_releases_resources(self):
        self.hand_model.predict.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_video()
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
