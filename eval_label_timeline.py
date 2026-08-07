"""
Interactive ground-truth labeller for video timelines.

Draw one or more ROIs, scrub the video, and mark the frame ranges in which each
target class is visible inside an ROI. Several classes may be active at once --
ground truth is multi-label, not a single state machine.

    python eval_label_timeline.py --project projects/hph_hw --video test1 --scale 0.6

Everything is typed INSIDE the video window -- the terminal is only used for
startup logs. Press 'h' in the window for the key list.
"""

import argparse
import copy
import os

import cv2
import numpy as np

from eval_common import (
    build_color_map,
    frames_to_timecode,
    load_classes,
    load_gt,
    load_project,
    new_gt,
    resolve_class,
    save_gt,
)

WINDOW = "eval_label_timeline"
TRACKBAR = "frame"
STRIP_ROW_H = 16
STRIP_PAD = 24

KEY_ENTER = (13, 10)
KEY_BACKSPACE = (8, 127)
KEY_ESC = 27
KEY_NONE = 255

HELP = """
--- Ground truth labelling (all input happens in the video window) ---
  trackbar / click strip : seek
  a / d                  : -1 / +1 frame
  , / .                  : -10 / +10 frames
  [ / ]                  : -100 / +100 frames
  space                  : play / pause
  g                      : goto frame
  i                      : mark IN at playhead
  o                      : mark OUT -> pick class + count
  c                      : cancel a pending IN
  r                      : ROI draw mode (2 clicks, then 's' to name it)
  R                      : delete an ROI
  t                      : cycle the active ROI
  n / p                  : select next / previous segment
  j                      : jump playhead to the selected segment
  e                      : edit the selected segment
  x                      : delete the selected segment
  u                      : undo the last change
  f                      : toggle the ROI overlay
  h                      : print this help
  w                      : write the ground truth file now
  q / ESC                : quit (asks to save if there are changes)
"""


def hex_to_bgr(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return (b, g, r)


class FrameSource:
    """
    Random access to video frames by decode-order index.

    By default the whole video is decoded once, sequentially, and held as JPEG
    blobs. This matters for correctness, not just speed: cv2's CAP_PROP_POS_FRAMES
    seek can land on a different frame than the Nth decoded frame on B-frame MP4s,
    and ultralytics decodes sequentially. If the labeller and the detection cache
    disagree on what "frame 431" means, every metric silently shifts.
    """

    def __init__(self, path, preload=True, jpeg_quality=85):
        self.path = path
        self.preload = preload
        self._blobs = []
        self._cap = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"could not open video: {path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not preload:
            self._cap = cap
            self.total_frames = reported
            print("⚠ --no-preload: seeking instead of decoding sequentially. Frame indices may")
            print("  drift from what eval_run_detect.py sees on VFR / B-frame sources.")
            return

        print(f"⏳ Decoding {path} ...")
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ok, buf = cv2.imencode(".jpg", frame, params)
            if not ok:
                break
            self._blobs.append(buf)
            if len(self._blobs) % 500 == 0:
                print(f"   ... {len(self._blobs)} frames")
        cap.release()

        self.total_frames = len(self._blobs)
        if self.total_frames == 0:
            raise IOError(f"decoded 0 frames from {path}")
        mb = sum(b.nbytes for b in self._blobs) / 1e6
        print(f"✅ {self.total_frames} frames cached in RAM ({mb:.0f} MB)")
        if reported != self.total_frames:
            print(f"   ⚠ container reported {reported} frames; using the decoded count.")

    def get(self, idx):
        idx = max(0, min(self.total_frames - 1, idx))
        if self.preload:
            return cv2.imdecode(self._blobs[idx], cv2.IMREAD_COLOR)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        return frame if ok else np.zeros((self.height, self.width, 3), np.uint8)

    def release(self):
        if self._cap is not None:
            self._cap.release()


class Labeller:
    def __init__(self, project_dir, cfg, classes, video_entry, scale, preload):
        self.project_dir = project_dir
        self.cfg = cfg
        self.classes = classes
        self.alias = cfg.get("class_alias", {})
        self.color_map = {c: hex_to_bgr(h) for c, h in build_color_map(classes).items()}
        self.video_name = video_entry["name"]
        self.src = FrameSource(video_entry["path"], preload=preload)
        self.scale = scale

        self.gt = load_gt(project_dir, self.video_name)
        if self.gt is None:
            self.gt = new_gt(
                self.video_name, video_entry["path"], self.src.fps,
                self.src.total_frames, self.src.width, self.src.height,
            )
            print(f"🆕 New ground truth for {self.video_name}")
        else:
            print(f"📂 Loaded {len(self.gt['segments'])} segments, {len(self.gt['rois'])} ROI(s)")
            if self.gt["total_frames"] != self.src.total_frames:
                print(f"   ⚠ stored total_frames={self.gt['total_frames']} but this video decodes "
                      f"to {self.src.total_frames}. Existing segments may be misaligned.")

        self.frame_idx = 0
        self.playing = False
        self.pending_in = None
        self.sel_idx = -1
        self.active_roi = self.gt["rois"][0]["name"] if self.gt["rois"] else None
        self.roi_mode = False
        self.roi_points = []
        self.show_rois = True
        self.dirty = False
        self.undo_stack = []
        self.status = ""
        self.should_quit = False
        self.prompt = None
        self._suppress_trackbar = False

        self.disp_w = int(self.src.width * scale)
        self.disp_h = int(self.src.height * scale)
        self.strip_h = STRIP_PAD + STRIP_ROW_H * len(classes)

    # -- modal prompt ------------------------------------------------------
    #
    # Everything is typed into the OpenCV window. An earlier version used
    # input(), which blocks the cv2 event loop: navigation keys pressed while
    # the terminal happened to have focus were swallowed as answers, and a
    # stray keystroke could confirm a segment deletion. Keeping input in one
    # window makes that impossible.

    def ask(self, label, on_submit, hint_lines=None, initial="", numeric=False):
        self.prompt = {
            "label": label, "cb": on_submit, "hint": hint_lines or [],
            "buf": str(initial), "numeric": numeric,
        }
        self.playing = False

    def _prompt_key(self, key):
        p = self.prompt
        if key == KEY_ESC:
            self.prompt = None
            self.status = "cancelled"
        elif key in KEY_ENTER:
            self.prompt = None
            p["cb"](p["buf"].strip())
        elif key in KEY_BACKSPACE:
            p["buf"] = p["buf"][:-1]
        elif 32 <= key <= 126:
            ch = chr(key)
            if not p["numeric"] or ch.isdigit() or ch == "-":
                p["buf"] += ch

    def _draw_prompt(self, canvas):
        p = self.prompt
        lines = list(p["hint"]) + ["", f"{p['label']} {p['buf']}_", "",
                                   "Enter = confirm    ESC = cancel"]
        pad, line_h = 12, 20
        box_h = min(self.disp_h, len(lines) * line_h + pad * 2)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.disp_w, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (0, 0), (self.disp_w - 1, box_h - 1), (0, 220, 255), 2)

        for i, text in enumerate(lines):
            y = pad + line_h * (i + 1) - 6
            if y > box_h - 4:
                break
            is_input = text.startswith(p["label"])
            cv2.putText(canvas, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 if is_input else 0.45,
                        (0, 220, 255) if is_input else (235, 235, 235),
                        2 if is_input else 1, cv2.LINE_AA)

    # -- state -------------------------------------------------------------

    def push_undo(self):
        self.undo_stack.append(
            (copy.deepcopy(self.gt["segments"]), copy.deepcopy(self.gt["rois"]))
        )
        self.undo_stack = self.undo_stack[-50:]
        self.dirty = True

    def undo(self):
        if not self.undo_stack:
            self.status = "nothing to undo"
            return
        self.gt["segments"], self.gt["rois"] = self.undo_stack.pop()
        self.sel_idx = min(self.sel_idx, len(self.gt["segments"]) - 1)
        roi_names = {r["name"] for r in self.gt["rois"]}
        if self.active_roi not in roi_names:
            self.active_roi = next(iter(roi_names), None)
        self.status = "undone"

    def seek(self, idx):
        self.frame_idx = max(0, min(self.src.total_frames - 1, int(idx)))
        self._suppress_trackbar = True
        cv2.setTrackbarPos(TRACKBAR, WINDOW, self.frame_idx)
        self._suppress_trackbar = False

    def segments_of(self, roi_name):
        return [s for s in self.gt["segments"] if s["roi"] == roi_name]

    def next_segment_id(self):
        return max((s["id"] for s in self.gt["segments"]), default=0) + 1

    def class_hint(self):
        return ["Pick a class (type the index or the name):"] + [
            f"  [{i}] {c}" for i, c in enumerate(self.classes)
        ]

    # -- rendering ---------------------------------------------------------

    def render(self):
        frame = self.src.get(self.frame_idx)
        disp = cv2.resize(frame, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)

        if self.show_rois:
            for roi in self.gt["rois"]:
                p1 = (int(roi["x1"] * self.scale), int(roi["y1"] * self.scale))
                p2 = (int(roi["x2"] * self.scale), int(roi["y2"] * self.scale))
                is_active = roi["name"] == self.active_roi
                cv2.rectangle(disp, p1, p2, (0, 255, 0) if is_active else (140, 140, 140),
                              2 if is_active else 1)
                cv2.putText(disp, roi["name"], (p1[0] + 4, p1[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        for pt in self.roi_points:
            cv2.circle(disp, (int(pt[0] * self.scale), int(pt[1] * self.scale)), 5, (0, 0, 255), -1)
        if len(self.roi_points) == 2:
            cv2.rectangle(
                disp,
                (int(self.roi_points[0][0] * self.scale), int(self.roi_points[0][1] * self.scale)),
                (int(self.roi_points[1][0] * self.scale), int(self.roi_points[1][1] * self.scale)),
                (0, 0, 255), 2,
            )

        if self.prompt is not None:
            self._draw_prompt(disp)
        else:
            self._draw_hud(disp)
        return np.vstack([disp, self._draw_strip()])

    def _draw_hud(self, disp):
        lines = [
            f"{'*' if self.dirty else ' '} frame {self.frame_idx}/{self.src.total_frames - 1}"
            f"  {frames_to_timecode(self.frame_idx, self.src.fps)}"
            f"  roi={self.active_roi or '-'}"
            + ("  [PLAY]" if self.playing else "")
        ]
        if self.pending_in is not None:
            lines.append(f"IN @ {self.pending_in} -- press 'o' to close the segment")
        if self.roi_mode:
            lines.append("ROI MODE: click 2 points, 's' to name it, ESC to exit")

        active = [
            f"{s['class']}x{s['count']}"
            for s in self.segments_of(self.active_roi)
            if s["start_frame"] <= self.frame_idx <= s["end_frame"]
        ]
        lines.append("active: " + (", ".join(active) if active else "-"))
        if 0 <= self.sel_idx < len(self.gt["segments"]):
            s = self.gt["segments"][self.sel_idx]
            lines.append(
                f"selected #{s['id']} {s['class']}x{s['count']} "
                f"[{s['start_frame']}-{s['end_frame']}] @{s['roi']}"
            )
        if self.status:
            lines.append(self.status)

        for i, text in enumerate(lines):
            y = 22 + i * 20
            cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_strip(self):
        """One row per class for the active ROI, with a playhead cursor."""
        strip = np.full((self.strip_h, self.disp_w, 3), 32, np.uint8)
        total = max(1, self.src.total_frames)
        selected_id = (
            self.gt["segments"][self.sel_idx]["id"]
            if 0 <= self.sel_idx < len(self.gt["segments"]) else None
        )

        for k, cls in enumerate(self.classes):
            y0 = STRIP_PAD + k * STRIP_ROW_H
            y1 = y0 + STRIP_ROW_H - 3
            cv2.rectangle(strip, (0, y0), (self.disp_w, y1), (48, 48, 48), -1)
            for seg in self.segments_of(self.active_roi):
                if seg["class"] != cls:
                    continue
                x0 = int(seg["start_frame"] / total * self.disp_w)
                x1 = max(x0 + 1, int((seg["end_frame"] + 1) / total * self.disp_w))
                cv2.rectangle(strip, (x0, y0), (x1, y1), self.color_map[cls], -1)
                if seg["id"] == selected_id:
                    cv2.rectangle(strip, (x0, y0), (x1, y1), (255, 255, 255), 1)
            cv2.putText(strip, cls[:14], (4, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (220, 220, 220), 1, cv2.LINE_AA)

        if self.pending_in is not None:
            xp = int(self.pending_in / total * self.disp_w)
            cv2.line(strip, (xp, STRIP_PAD), (xp, self.strip_h), (0, 200, 255), 1)

        xc = int(self.frame_idx / total * self.disp_w)
        cv2.line(strip, (xc, 0), (xc, self.strip_h), (0, 0, 255), 1)
        cv2.putText(strip, f"ROI: {self.active_roi or '(none - press r)'}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return strip

    # -- mouse / trackbar --------------------------------------------------

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.prompt is not None:
            return
        if y >= self.disp_h:
            self.seek(x / max(1, self.disp_w) * self.src.total_frames)
            return
        if self.roi_mode:
            if len(self.roi_points) >= 2:
                self.roi_points = []
            self.roi_points.append((int(x / self.scale), int(y / self.scale)))

    def on_trackbar(self, pos):
        if not self._suppress_trackbar and self.prompt is None:
            self.frame_idx = max(0, min(self.src.total_frames - 1, pos))

    # -- actions -----------------------------------------------------------

    def mark_out(self):
        if self.pending_in is None:
            self.status = "press 'i' to mark IN first"
            return
        if not self.gt["rois"]:
            self.status = "draw an ROI first (press 'r')"
            return

        start, end = sorted((self.pending_in, self.frame_idx))

        def commit(roi, cls, count):
            self.push_undo()
            self.gt["segments"].append({
                "id": self.next_segment_id(), "roi": roi, "class": cls, "count": count,
                "start_frame": start, "end_frame": end, "note": "",
            })
            self.gt["segments"].sort(key=lambda s: (s["start_frame"], s["class"]))
            self.sel_idx = next(
                i for i, s in enumerate(self.gt["segments"])
                if s["start_frame"] == start and s["class"] == cls and s["end_frame"] == end
            )
            self.pending_in = None
            self.status = f"+ {cls} x{count} [{start}-{end}] @{roi} ({end - start + 1} frames)"
            print(f"✅ {self.status[2:]}")

        def ask_count(roi, cls):
            def done(raw):
                count = int(raw) if raw.isdigit() and int(raw) > 0 else 1
                commit(roi, cls, count)
            self.ask(f"count for {cls} [1]:", done,
                     [f"Segment [{start}-{end}] -- {end - start + 1} frames",
                      f"class = {cls}", "",
                      "How many instances are visible? Enter = 1"], numeric=True)

        def ask_class(roi):
            def done(raw):
                if not raw:
                    self.status = "cancelled"
                    return
                cls = resolve_class(raw, self.classes, self.alias)
                if cls is None:
                    self.status = f"'{raw}' is not a project class"
                    ask_class(roi)
                    return
                ask_count(roi, cls)
            self.ask("class:", done,
                     [f"Segment [{start}-{end}] @{roi}"] + self.class_hint())

        if len(self.gt["rois"]) <= 1:
            ask_class(self.active_roi)
            return

        def done_roi(raw):
            rois = self.gt["rois"]
            if not raw:
                ask_class(self.active_roi)
            elif raw.isdigit() and 0 <= int(raw) < len(rois):
                ask_class(rois[int(raw)]["name"])
            elif any(r["name"] == raw for r in rois):
                ask_class(raw)
            else:
                ask_class(self.active_roi)
        self.ask(f"roi [{self.active_roi}]:", done_roi,
                 ["Which ROI does this segment belong to? Enter = active"]
                 + [f"  [{i}] {r['name']}" for i, r in enumerate(self.gt["rois"])])

    def save_roi(self):
        if len(self.roi_points) != 2:
            self.status = "click two points first"
            return

        def done(name):
            if not name:
                self.status = "cancelled"
                return
            if any(r["name"] == name for r in self.gt["rois"]):
                self.status = f"ROI '{name}' already exists"
                return
            (x1, y1), (x2, y2) = self.roi_points
            self.push_undo()
            self.gt["rois"].append({
                "name": name,
                "x1": min(x1, x2), "y1": min(y1, y2),
                "x2": max(x1, x2), "y2": max(y1, y2),
            })
            self.active_roi = name
            self.roi_points = []
            self.roi_mode = False
            self.status = f"ROI '{name}' added"
            print(f"✅ {self.status}")

        self.ask("ROI name:", done, ["Name this region, e.g. tray / bagging"])

    def delete_roi(self):
        if not self.gt["rois"]:
            return

        def done(raw):
            rois = self.gt["rois"]
            if not raw.isdigit() or not (0 <= int(raw) < len(rois)):
                self.status = "cancelled"
                return
            roi = rois[int(raw)]
            used = len(self.segments_of(roi["name"]))
            if used:
                self.status = f"'{roi['name']}' still has {used} segment(s)"
                return
            self.push_undo()
            rois.pop(int(raw))
            if self.active_roi == roi["name"]:
                self.active_roi = rois[0]["name"] if rois else None
            self.status = f"ROI '{roi['name']}' deleted"

        self.ask("delete roi index:", done,
                 ["Delete which ROI?"]
                 + [f"  [{i}] {r['name']}" for i, r in enumerate(self.gt["rois"])],
                 numeric=True)

    def edit_segment(self):
        if not (0 <= self.sel_idx < len(self.gt["segments"])):
            self.status = "select a segment first ('n'/'p')"
            return
        seg = self.gt["segments"][self.sel_idx]
        header = (f"#{seg['id']}  {seg['class']} x{seg['count']}  "
                  f"[{seg['start_frame']}-{seg['end_frame']}] @{seg['roi']}")

        def apply(field, raw):
            self.push_undo()
            if field == "class":
                cls = resolve_class(raw, self.classes, self.alias)
                if cls is None:
                    self.undo_stack.pop()
                    self.status = f"'{raw}' is not a project class"
                    return
                seg["class"] = cls
            elif field == "count":
                seg["count"] = int(raw) if raw.isdigit() and int(raw) > 0 else seg["count"]
            elif field in ("start", "end"):
                value = int(raw) if raw.lstrip("-").isdigit() else self.frame_idx
                seg[f"{field}_frame"] = max(0, min(self.src.total_frames - 1, value))
                if seg["start_frame"] > seg["end_frame"]:
                    seg["start_frame"], seg["end_frame"] = seg["end_frame"], seg["start_frame"]
            elif field == "roi":
                rois = self.gt["rois"]
                if raw.isdigit() and 0 <= int(raw) < len(rois):
                    seg["roi"] = rois[int(raw)]["name"]
                elif any(r["name"] == raw for r in rois):
                    seg["roi"] = raw
            elif field == "note":
                seg["note"] = raw
            self.gt["segments"].sort(key=lambda s: (s["start_frame"], s["class"]))
            self.sel_idx = next(i for i, s in enumerate(self.gt["segments"])
                                if s["id"] == seg["id"])
            self.status = (f"#{seg['id']} -> {seg['class']} x{seg['count']} "
                           f"[{seg['start_frame']}-{seg['end_frame']}] @{seg['roi']}")

        def pick_field(field):
            field = field.lower()
            if field == "class":
                self.ask("new class:", lambda v: apply("class", v),
                         [header] + self.class_hint())
            elif field == "count":
                self.ask("new count:", lambda v: apply("count", v),
                         [header, "", "How many instances?"],
                         initial=str(seg["count"]), numeric=True)
            elif field in ("start", "end"):
                self.ask(f"new {field} frame:", lambda v: apply(field, v),
                         [header, "", f"Enter accepts the playhead ({self.frame_idx})"],
                         initial=str(self.frame_idx), numeric=True)
            elif field == "roi":
                self.ask("new roi:", lambda v: apply("roi", v),
                         [header] + [f"  [{i}] {r['name']}"
                                     for i, r in enumerate(self.gt["rois"])])
            elif field == "note":
                self.ask("note:", lambda v: apply("note", v), [header],
                         initial=seg.get("note", ""))
            else:
                self.status = f"unknown field '{field}'"

        self.ask("field:", pick_field,
                 [f"Editing {header}", "",
                  "Which field? class / count / start / end / roi / note"])

    def delete_segment(self):
        if not (0 <= self.sel_idx < len(self.gt["segments"])):
            self.status = "select a segment first ('n'/'p')"
            return
        seg = self.gt["segments"][self.sel_idx]

        def done(raw):
            # Requires the full word, not a bare 'y': a single stray keystroke
            # should never be able to destroy hand-labelled work.
            if raw.lower() != "delete":
                self.status = "delete cancelled"
                return
            self.push_undo()
            self.gt["segments"].pop(self.sel_idx)
            self.sel_idx = min(self.sel_idx, len(self.gt["segments"]) - 1)
            self.status = f"segment #{seg['id']} deleted (press 'u' to undo)"
            print(f"🗑 {self.status}")

        self.ask("type 'delete' to confirm:", done,
                 [f"Delete #{seg['id']}  {seg['class']} x{seg['count']}",
                  f"  [{seg['start_frame']}-{seg['end_frame']}] @{seg['roi']}",
                  "", "Anything else cancels."])

    def select_step(self, delta):
        if not self.gt["segments"]:
            self.status = "no segments yet"
            return
        self.sel_idx = (self.sel_idx + delta) % len(self.gt["segments"])
        seg = self.gt["segments"][self.sel_idx]
        self.status = (f"selected #{seg['id']} {seg['class']} x{seg['count']} "
                       f"[{seg['start_frame']}-{seg['end_frame']}]")

    def goto(self):
        self.ask("goto frame:", lambda raw: self.seek(int(raw)) if raw.isdigit() else None,
                 [f"Jump to a frame (0 - {self.src.total_frames - 1})"], numeric=True)

    def save(self):
        self.gt["total_frames"] = self.src.total_frames
        self.gt["fps"] = self.src.fps
        path = save_gt(self.project_dir, self.gt)
        self.dirty = False
        self.status = f"saved {len(self.gt['segments'])} segments"
        print(f"💾 {len(self.gt['segments'])} segments -> {path}")

    def request_quit(self):
        if not self.dirty:
            self.should_quit = True
            return

        def done(raw):
            answer = raw.lower()
            if answer.startswith("y"):
                self.save()
                self.should_quit = True
            elif answer.startswith("n"):
                self.should_quit = True
            else:
                self.status = "quit cancelled"

        self.ask("save before quit? (y/n):", done,
                 [f"{len(self.gt['segments'])} segments, unsaved changes.",
                  "", "y = save and quit    n = discard and quit    ESC = stay"])

    # -- main loop ---------------------------------------------------------

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, self.disp_w, self.disp_h + self.strip_h)
        cv2.createTrackbar(TRACKBAR, WINDOW, 0, max(1, self.src.total_frames - 1),
                           self.on_trackbar)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        print(HELP)

        while not self.should_quit:
            cv2.imshow(WINDOW, self.render())
            delay = max(1, int(1000 / self.src.fps)) if self.playing else 20
            key = cv2.waitKey(delay) & 0xFF

            if self.prompt is not None:
                if key != KEY_NONE:
                    self._prompt_key(key)
                continue

            if self.playing and key == KEY_NONE:
                if self.frame_idx >= self.src.total_frames - 1:
                    self.playing = False
                else:
                    self.seek(self.frame_idx + 1)
                continue
            if key == KEY_NONE:
                continue

            self.status = ""
            if key == ord(" "):
                self.playing = not self.playing
            elif key == ord("a"):
                self.seek(self.frame_idx - 1)
            elif key == ord("d"):
                self.seek(self.frame_idx + 1)
            elif key == ord(","):
                self.seek(self.frame_idx - 10)
            elif key == ord("."):
                self.seek(self.frame_idx + 10)
            elif key == ord("["):
                self.seek(self.frame_idx - 100)
            elif key == ord("]"):
                self.seek(self.frame_idx + 100)
            elif key == ord("g"):
                self.goto()
            elif key == ord("i"):
                self.pending_in = self.frame_idx
                self.status = f"IN @ {self.frame_idx}"
            elif key == ord("o"):
                self.mark_out()
            elif key == ord("c"):
                self.pending_in = None
                self.status = "pending IN cancelled"
            elif key == ord("r"):
                self.roi_mode = not self.roi_mode
                self.roi_points = []
                self.status = "ROI mode " + ("ON: click 2 points, 's' to name" if self.roi_mode
                                             else "off")
            elif key == ord("R"):
                self.delete_roi()
            elif key == ord("s") and self.roi_mode:
                self.save_roi()
            elif key == ord("t"):
                if self.gt["rois"]:
                    names = [r["name"] for r in self.gt["rois"]]
                    idx = names.index(self.active_roi) if self.active_roi in names else -1
                    self.active_roi = names[(idx + 1) % len(names)]
                    self.status = f"active ROI: {self.active_roi}"
            elif key == ord("n"):
                self.select_step(1)
            elif key == ord("p"):
                self.select_step(-1)
            elif key == ord("j"):
                if 0 <= self.sel_idx < len(self.gt["segments"]):
                    self.seek(self.gt["segments"][self.sel_idx]["start_frame"])
            elif key == ord("e"):
                self.edit_segment()
            elif key == ord("x"):
                self.delete_segment()
            elif key == ord("u"):
                self.undo()
            elif key == ord("f"):
                self.show_rois = not self.show_rois
            elif key == ord("h"):
                print(HELP)
            elif key == ord("w"):
                self.save()
            elif key in (ord("q"), KEY_ESC):
                if self.roi_mode and key == KEY_ESC:
                    self.roi_mode = False
                    self.roi_points = []
                    self.status = "ROI mode off"
                    continue
                self.request_quit()

        cv2.destroyAllWindows()
        self.src.release()


def main():
    parser = argparse.ArgumentParser(description="Label temporal ground truth in an ROI.")
    parser.add_argument("--project", required=True, help="Path to projects/<name>")
    parser.add_argument("--video", required=True, help="Video name from config.json videos[]")
    parser.add_argument("--scale", type=float, default=0.6, help="Display scale (default 0.6)")
    parser.add_argument("--no-preload", action="store_true",
                        help="Seek instead of caching frames in RAM (risks frame-index drift)")
    args = parser.parse_args()

    cfg = load_project(args.project)
    classes = load_classes(args.project, cfg)

    entry = next((v for v in cfg["videos"] if v["name"] == args.video), None)
    if entry is None:
        print(f"❌ video {args.video!r} not in config.json. Available: "
              f"{[v['name'] for v in cfg['videos']]}")
        return
    if not os.path.exists(entry["path"]):
        print(f"❌ video file not found: {entry['path']}")
        return

    Labeller(args.project, cfg, classes, entry, args.scale, not args.no_preload).run()


if __name__ == "__main__":
    main()
