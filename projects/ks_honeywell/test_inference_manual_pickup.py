"""Behavioral tests; no GPU, OpenCV, or model weights required."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import inference_manual_pickup as pickup


class PickupLogicTests(unittest.TestCase):
    def test_hand_memory_assigns_pair_and_matches_single_even_past_image_midpoint(self):
        hands = [("hand", .8, (10, 20, 110, 220)),
                 ("hand", .9, (410, 20, 510, 220))]
        memory = pickup.HandSideMemory()
        self.assertEqual(memory.update(hands[:1]), [])
        self.assertEqual(memory.update(list(reversed(hands))),
                         [("right", .8, hands[0][2]), ("left", .9, hands[1][2])])
        self.assertEqual(memory.update(hands[1:]), [("left", .9, hands[1][2])])
        self.assertEqual(memory.update([]), [])
        self.assertEqual(memory.centers, {"right": (60, 120), "left": (460, 120)})
        # A lone box around x=300 stays nearest the remembered left hand,
        # irrespective of the full-frame midpoint and without moving references.
        lone = [("hand", .7, (280, 20, 320, 220))]
        self.assertEqual(memory.update(lone), [("left", .7, lone[0][2])])
        self.assertEqual(memory.centers["left"], (460, 120))
        # A new two-hand frame refreshes both references.
        memory.update([("hand", .9, (80, 0, 120, 40)),
                       ("hand", .8, (180, 0, 220, 40))])
        self.assertEqual(memory.centers, {"right": (100, 20), "left": (200, 20)})
        invalid = [("hand", .9, (20, 20, 10, 30)),
                   ("hand", .9, (10, 20, float("nan"), 30))]
        self.assertEqual(memory.update(invalid), [])

    def test_hand_contact_quadrants_are_equal_quarters(self):
        box = (10, 20, 110, 220)
        self.assertEqual(pickup.hand_contact_quadrant(box, "right"), (60, 120, 110, 220))
        self.assertEqual(pickup.hand_contact_quadrant(box, "left"), (10, 120, 60, 220))
        # Contact outside the chosen quarter must not qualify even when the
        # full hand box overlaps the manual.
        right_zone = pickup.hand_contact_quadrant(box, "right")
        self.assertFalse(pickup.intersects(right_zone, (10, 20, 50, 100)))
        self.assertFalse(pickup.intersects(right_zone, (10, 130, 50, 200)))
        self.assertFalse(pickup.intersects(right_zone, (110, 130, 120, 200)))
        self.assertTrue(pickup.intersects(right_zone, (100, 200, 120, 230)))
        with self.assertRaisesRegex(ValueError, "Unknown handedness"):
            pickup.hand_contact_quadrant(box, "unknown")

    def test_landmark_adapter_converts_rgb_and_uses_monotonic_milliseconds(self):
        # Exercise the adapter without importing the optional ML libraries.
        adapter = pickup.IndexFingerLandmarker.__new__(pickup.IndexFingerLandmarker)
        adapter.mp = MagicMock()
        adapter.cv2 = MagicMock()
        adapter.model = MagicMock()
        adapter.model.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[])
        adapter.last_timestamp_ms = -1
        frame = MagicMock()
        frame.shape = (480, 640, 3)
        for timestamp in [0, .0001, .0002, .0333]:
            self.assertEqual(adapter.detect(frame, timestamp), [])
        self.assertEqual([call.args[1] for call in adapter.model.detect_for_video.call_args_list],
                         [0, 1, 2, 33])
        adapter.cv2.cvtColor.assert_called_with(frame, adapter.cv2.COLOR_BGR2RGB)
        adapter.mp.Image.assert_called_with(image_format=adapter.mp.ImageFormat.SRGB,
                                            data=adapter.cv2.cvtColor.return_value)
        adapter.close()
        adapter.model.close.assert_called_once()

    def test_index_tip_inside_boundary_and_outside(self):
        objects = [(name, .9, (0, 0, 10, 10)) for name in pickup.MANUALS]
        objects.append(("scanner_in_box", .9, (20, 20, 30, 30)))
        for tip in [(5, 5), (0, 0), (10, 10)]:
            self.assertEqual(pickup.fingertip_contacts(objects, [tip]), set(pickup.MANUALS))
        for tips in [[], [(11, 5)], [(25, 25)], [(float("nan"), 5)], [(5, float("inf"))]]:
            self.assertEqual(pickup.fingertip_contacts(objects, tips), set())

    def test_landmarks_map_to_original_pixels_and_skip_invalid_tips(self):
        hand = [SimpleNamespace(x=.5, y=.25) for _ in range(21)]
        result = SimpleNamespace(hand_landmarks=[hand, hand[:8]])
        self.assertEqual(pickup.pixel_landmarks(result, 640, 480)[0][8], (320, 120))
        hand[8] = SimpleNamespace(x=1.1, y=.25)
        self.assertEqual(pickup.pixel_landmarks(result, 640, 480), [])
        hand[8] = SimpleNamespace(x=float("nan"), y=.25)
        self.assertEqual(pickup.pixel_landmarks(result, 640, 480), [])

    def test_remembers_four_manual_classes_and_refreshes_without_old_trails(self):
        memory = pickup.ManualBoxMemory()
        self.assertEqual(memory.update([]), ([], set()))
        objects = [(name, .9, (0, 0, 10, 10)) for name in pickup.MANUALS]
        memory.update(objects + [("scanner_in_box", .9, (20, 20, 30, 30))])
        cached, remembered = memory.update([])
        self.assertEqual(cached, objects)
        self.assertEqual(remembered, set(pickup.MANUALS))
        moved = ("manual1", .95, (50, 50, 60, 60))
        current, remembered = memory.update([moved])
        self.assertEqual(len(current), 4)
        self.assertEqual(remembered, {"manual2", "manual3", "manual4"})
        cached, _ = memory.update([])
        self.assertIn(moved, cached)
        self.assertNotIn(objects[0], cached)

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
        shuffled = list(reversed(pickup.WORKFLOW_STEPS))
        names = {str(index + 1): name for index, name in enumerate(shuffled)}
        names[0] = "other"
        resolved = pickup.workflow_class_ids(names)
        self.assertEqual(set(resolved.values()), set(pickup.WORKFLOW_STEPS))
        self.assertEqual(resolved[1], pickup.FINAL_STEP)
        with self.assertRaisesRegex(ValueError, "label_on_box"):
            pickup.workflow_class_ids(list(pickup.WORKFLOW_STEPS[:-1]))

    def test_workflow_enforces_order_and_does_not_cache_early_detections(self):
        workflow = pickup.WorkflowState()
        second = pickup.ORDERED_STEPS[1]
        self.assertIsNone(workflow.update_detection({second}, 0.0, 0.2))
        self.assertIsNone(workflow.update_detection({second}, 1.0, 0.2))
        self.assertEqual(workflow.status(pickup.ORDERED_STEPS[0]), "CURRENT")
        self.assertEqual(workflow.status(second), "PENDING")

        first = pickup.ORDERED_STEPS[0]
        self.assertIsNone(workflow.update_detection({first, second}, 2.0, 0.2))
        self.assertEqual(workflow.update_detection({first, second}, 2.2, 0.2), first)
        self.assertIsNone(workflow.update_detection({second}, 2.21, 0.2))
        self.assertEqual(workflow.update_detection({second}, 2.41, 0.2), second)

    def test_workflow_manuals_are_unordered_and_final_is_last(self):
        workflow = pickup.WorkflowState()
        workflow.completed.update(pickup.ORDERED_STEPS)
        self.assertEqual(workflow.phase, "Pick manuals - any order")
        self.assertTrue(all(workflow.status(name) == "CURRENT" for name in pickup.MANUALS))
        self.assertIsNone(workflow.update_detection({pickup.FINAL_STEP}, 0.0, 0.0))
        self.assertEqual(workflow.accept_manual_events({"manual3", "manual1"}),
                         ["manual1", "manual3"])
        self.assertEqual(workflow.accept_manual_events({"manual4", "manual2"}),
                         ["manual2", "manual4"])
        self.assertEqual(workflow.phase, "Apply label to box")
        self.assertEqual(workflow.update_detection({pickup.FINAL_STEP}, 1.0, 0.0),
                         pickup.FINAL_STEP)
        self.assertTrue(workflow.complete)
        self.assertEqual(workflow.completed_count, 11)


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
            "--output", str(self.root / "out.mp4"), "--device", "cpu",
            "--contact-mode", "hand-box", "--contact-duration", "0"])
        self.args.landmark_model = str(self.root / "hand_landmarker.task")
        Path(self.args.landmark_model).touch()
        self.landmarker = MagicMock()
        self.landmarker.detect.return_value = [[(5, 5)] * 21]
        self.landmark_factory = MagicMock(return_value=self.landmarker)
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
        self.manual_model.names = dict(enumerate(pickup.WORKFLOW_STEPS))
        self.hand_model = MagicMock(task="detect")
        manual_rows = [[0, 0, 10, 10, .9, index]
                       for index in range(len(pickup.ORDERED_STEPS) + 1)]
        for model, rows in [(self.manual_model, manual_rows),
                            (self.hand_model, [[2, 2, 8, 8, .8, 0]])]:
            result = MagicMock()
            result.boxes.data.cpu.return_value.tolist.return_value = rows
            model.predict.return_value = [result]
        self.factory = MagicMock(side_effect=[self.manual_model, self.hand_model])

    def run_video(self):
        with redirect_stdout(io.StringIO()):
            return pickup.run(self.args, cv2_module=self.cv2, model_factory=self.factory,
                              landmark_factory=self.landmark_factory)

    def rows_with_manual1(self, box):
        rows = [[0, 0, 10, 10, .9, index]
                for index in range(len(pickup.ORDERED_STEPS))]
        manual_id = pickup.WORKFLOW_STEPS.index("manual1")
        rows.append([*box, .9, manual_id])
        return rows

    def test_quadrant_touch_uses_memory_without_landmark_model(self):
        self.args.contact_mode = "hand-quadrant"
        Path(self.args.landmark_model).unlink()
        visible = self.manual_model.predict.return_value[0]
        hidden = SimpleNamespace(boxes=None)
        self.manual_model.predict.side_effect = [[visible]] * 6 + [[hidden]] * 6
        # Initialize with two hands; the rightmost box is the left hand, even
        # though both are near the image-left edge. Then only that hand remains.
        single = self.hand_model.predict.return_value[0]
        pair = MagicMock()
        pair.boxes.data.cpu.return_value.tolist.return_value = [
            [0, 0, 1, 1, .9, 0], [2, 2, 8, 8, .8, 0]]
        self.hand_model.predict.side_effect = [[pair]] + [[single]] * 11
        states = self.run_video()
        self.assertEqual(states["manual1"].count, 1)
        self.landmark_factory.assert_not_called()
        texts = [call.args[1] for call in self.cv2.putText.call_args_list]
        self.assertIn("LEFT HAND ZONE", texts)
        self.assertIn("TOUCH: manual1", texts)
        self.assertIn("manual1 0.90 CONTACT REMEMBERED", texts)
        self.assertNotIn("hand 0.80", texts)
        self.assertEqual(self.cv2.line.call_count, 0)
        # Full hand (2,2,8,8) is divided into quarters; only lower-left is active.
        self.assertTrue(any(call.args[1:3] == ((2, 5), (5, 8))
                            for call in self.cv2.rectangle.call_args_list))

    def test_whole_hand_overlap_outside_selected_quarter_is_not_contact(self):
        self.args.contact_mode = "hand-quadrant"
        # Image-left hand is right hand: lower-right counts, not upper-right.
        self.manual_model.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = \
            self.rows_with_manual1((12, 2, 18, 8))
        self.hand_model.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = [
            [0, 0, 20, 20, .8, 0], [100, 100, 120, 120, .9, 0]]
        states = self.run_video()
        self.assertEqual(states["manual1"].count, 0)
        self.landmark_factory.assert_not_called()

    def test_right_hand_uses_lower_right_quarter(self):
        self.args.contact_mode = "hand-quadrant"
        self.args.image_left_hand = "right"
        self.manual_model.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = \
            self.rows_with_manual1((12, 12, 18, 18))
        self.hand_model.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = [
            [0, 0, 20, 20, .8, 0], [100, 100, 120, 120, .9, 0]]
        states = self.run_video()
        self.assertEqual(states["manual1"].count, 1)
        texts = [call.args[1] for call in self.cv2.putText.call_args_list]
        self.assertIn("RIGHT HAND ZONE", texts)

    def test_roi_excludes_boxes_but_keeps_them_visible(self):
        path = self.root / "roi.json"
        pickup.save_roi(path, (20, 20, 100, 100), 640, 480)
        self.args.roi = str(path)
        states = self.run_video()
        self.assertTrue(all(state.count == 0 for state in states.values()))
        texts = [call.args[1] for call in self.cv2.putText.call_args_list]
        self.assertIn("COUNTING ROI", texts)
        self.assertIn("manual1 0.90", texts)
        self.assertNotIn("hand 0.80", texts)

    def test_roi_rejects_partially_outside_hand_even_when_quarter_is_inside(self):
        self.args.contact_mode = "hand-quadrant"
        path = self.root / "roi.json"
        pickup.save_roi(path, (0, 0, 10, 10), 640, 480)
        self.args.roi = str(path)
        self.hand_model.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = [
            [-5, -5, 8, 8, .8, 0], [100, 100, 120, 120, .9, 0]]
        states = self.run_video()
        self.assertTrue(all(state.count == 0 for state in states.values()))

    def test_roi_allows_fully_contained_boxes(self):
        path = self.root / "roi.json"
        pickup.save_roi(path, (0, 0, 10, 10), 640, 480)
        self.args.roi = str(path)
        self.assertEqual(self.run_video()["manual1"].count, 1)

    def test_roi_dimension_mismatch_stops_before_models_load(self):
        path = self.root / "roi.json"
        pickup.save_roi(path, (0, 0, 10, 10), 1280, 720)
        self.args.roi = str(path)
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            self.run_video()
        self.factory.assert_not_called()
        self.cap.release.assert_called_once()

    def test_index_tip_touch_continues_through_manual_occlusion(self):
        self.args.contact_mode = "index-finger"
        visible = self.manual_model.predict.return_value[0]
        hidden = SimpleNamespace(boxes=None)
        self.manual_model.predict.side_effect = [[visible]] * 6 + [[hidden]] * 6
        states = self.run_video()
        self.assertEqual(states["manual1"].count, 1)
        self.assertEqual(self.landmarker.detect.call_count, 12)
        self.assertAlmostEqual(self.landmarker.detect.call_args.args[1], 11 / 30)
        self.landmarker.close.assert_called_once()
        texts = [call.args[1] for call in self.cv2.putText.call_args_list]
        self.assertIn("INDEX TIP", texts)
        self.assertIn("TOUCH: manual1", texts)
        self.assertIn("manual1 0.90 CONTACT REMEMBERED", texts)
        self.assertTrue(self.cv2.circle.called)

    def test_hand_overlap_alone_does_not_trigger_index_touch(self):
        self.args.contact_mode = "index-finger"
        # First six frames: index outside manual despite hand-box overlap;
        # remaining frames: no landmarks. Neither case can produce a touch.
        self.landmarker.detect.side_effect = [[[(25, 25)] * 21]] * 6 + [[]] * 6
        states = self.run_video()
        self.assertTrue(all(state.count == 0 for state in states.values()))

    def test_missing_landmark_model_fails_before_inference(self):
        self.args.contact_mode = "index-finger"
        Path(self.args.landmark_model).unlink()
        with self.assertRaisesRegex(ValueError, "Hand landmark model not found"):
            self.run_video()
        self.factory.assert_not_called()

    def test_landmark_failure_releases_all_resources(self):
        self.args.contact_mode = "index-finger"
        self.landmarker.detect.side_effect = RuntimeError("landmark failed")
        with self.assertRaisesRegex(RuntimeError, "landmark failed"):
            self.run_video()
        self.landmarker.close.assert_called_once()
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()

    def test_pipeline_preserves_fps_dimensions_and_runs_both_models(self):
        states = self.run_video()
        self.assertEqual(states["manual1"].count, 1)
        self.assertEqual(self.manual_model.predict.call_count, 12)
        self.assertEqual(self.hand_model.predict.call_count, 12)
        self.assertEqual(self.writer.write.call_count, 12)
        self.assertEqual(self.cv2.VideoWriter.call_args.args[2:], (30.0, (640, 480)))
        self.assertEqual(self.manual_model.predict.call_args.kwargs["device"], "cpu")
        self.assertNotIn("classes", self.manual_model.predict.call_args.kwargs)
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()
        self.assertTrue(self.cv2.rectangle.called)
        texts = [call.args[1] for call in self.cv2.putText.call_args_list]
        self.assertIn("PICKUP: manual1", texts)
        self.assertIn("WORKFLOW", texts)
        self.assertNotIn("hand 0.80", texts)

    def test_all_eleven_classes_are_drawn_but_only_manuals_trigger_pickups(self):
        names = list(pickup.MANUALS) + [
            "scanner_with_label", "scanner_without_label", "scanner_in_bag",
            "box_folded", "scanner_in_box", "battery_in_bag", "label_on_box"]
        # String IDs match the supplied YAML mapping. Only non-manual boxes
        # overlap the hand; they must be visible without triggering pickup.
        self.manual_model.names = {str(index): name for index, name in enumerate(names)}
        rows = [[50, 50, 60, 60, .9, index] if index < 4
                else [0, 0, 10, 10, .9, index] for index in range(len(names))]
        result = self.manual_model.predict.return_value[0]
        result.boxes.data.cpu.return_value.tolist.return_value = rows
        states = self.run_video()
        texts = [call.args[1] for call in self.cv2.putText.call_args_list]
        for name in names:
            self.assertIn(f"{name} 0.90", texts)
        self.assertNotIn("hand 0.80", texts)
        self.assertTrue(all(state.count == 0 for state in states.values()))
        self.assertNotIn("classes", self.manual_model.predict.call_args.kwargs)
        self.assertFalse(any("PICKUP:" in text for text in texts))

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
