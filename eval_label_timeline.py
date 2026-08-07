"""
Interactive ground-truth labeller for video timelines.

Draw one or more ROIs, scrub the video, and mark the frame ranges in which each
target class is visible inside an ROI. Several classes may be active at once --
ground truth is multi-label, not a single state machine.

    python eval_label_timeline.py --project projects/hph_hw --video test1 --scale 0.6

Press 'h' inside the window for the key list.
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

HELP = """
--- Ground truth labelling ---
  trackbar / click strip : seek
  a / d                  : -1 / +1 frame
  , / .                  : -10 / +10 frames
  [ / ]                  : -100 / +100 frames
  space                  : play / pause
  g                      : goto frame (terminal)
  i                      : mark IN at playhead
  o                      : mark OUT -> prompts class + count (terminal)
  c                      : cancel a pending IN
  r                      : ROI draw mode (2 clicks, then 's' to save, ESC to exit)
  R                      : delete an ROI
  t                      : cycle the active ROI
  n / p                  : select next / previous segment
  j                      : jump playhead to the selected segment
  e                      : edit the selected segment (terminal)
  x                      : delete the selected segment
  u                      : undo the last change
  f                      : toggle the ROI overlay
  h                      : print this help
  w                      : write the ground truth file now
  q / ESC                : quit (prompts to save if there are changes)
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
        self._suppress_trackbar = False

        self.disp_w = int(self.src.width * scale)
        self.disp_h = int(self.src.height * scale)
        self.strip_h = STRIP_PAD + STRIP_ROW_H * len(classes)

    # -- state -------------------------------------------------------------

    def push_undo(self):
        self.undo_stack.append(
            (copy.deepcopy(self.gt["segments"]), copy.deepcopy(self.gt["rois"]))
        )
        self.undo_stack = self.undo_stack[-50:]
        self.dirty = True

    def undo(self):
        if not self.undo_stack:
            print("nothing to undo")
            return
        self.gt["segments"], self.gt["rois"] = self.undo_stack.pop()
        self.sel_idx = min(self.sel_idx, len(self.gt["segments"]) - 1)
        roi_names = {r["name"] for r in self.gt["rois"]}
        if self.active_roi not in roi_names:
            self.active_roi = next(iter(roi_names), None)
        print("↩ undone")

    def seek(self, idx):
        self.frame_idx = max(0, min(self.src.total_frames - 1, int(idx)))
        self._suppress_trackbar = True
        cv2.setTrackbarPos(TRACKBAR, WINDOW, self.frame_idx)
        self._suppress_trackbar = False

    def segments_of(self, roi_name):
        return [s for s in self.gt["segments"] if s["roi"] == roi_name]

    def next_segment_id(self):
        return max((s["id"] for s in self.gt["segments"]), default=0) + 1

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
            lines.append("ROI MODE: click 2 points, 's' to save, ESC to exit")

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
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y >= self.disp_h:
            self.seek(x / max(1, self.disp_w) * self.src.total_frames)
            return
        if self.roi_mode:
            if len(self.roi_points) >= 2:
                self.roi_points = []
            self.roi_points.append((int(x / self.scale), int(y / self.scale)))

    def on_trackbar(self, pos):
        if not self._suppress_trackbar:
            self.frame_idx = max(0, min(self.src.total_frames - 1, pos))

    # -- actions -----------------------------------------------------------

    def prompt_class(self):
        print("\nClasses:")
        for i, c in enumerate(self.classes):
            print(f"  [{i}] {c}")
        while True:
            raw = input("class (name or index, blank to cancel): ").strip()
            if not raw:
                return None
            resolved = resolve_class(raw, self.classes, self.alias)
            if resolved:
                return resolved
            print(f"❌ {raw!r} is not a project class -- pick from the list above.")

    def prompt_count(self, default=1):
        raw = input(f"count [{default}]: ").strip()
        if not raw:
            return default
        try:
            n = int(raw)
            return n if n > 0 else default
        except ValueError:
            print(f"not a number, using {default}")
            return default

    def prompt_roi(self):
        rois = self.gt["rois"]
        if len(rois) <= 1:
            return self.active_roi
        print("ROIs: " + ", ".join(f"[{i}] {r['name']}" for i, r in enumerate(rois)))
        raw = input(f"roi [{self.active_roi}]: ").strip()
        if not raw:
            return self.active_roi
        if raw.isdigit() and 0 <= int(raw) < len(rois):
            return rois[int(raw)]["name"]
        return raw if any(r["name"] == raw for r in rois) else self.active_roi

    def mark_out(self):
        if self.pending_in is None:
            print("press 'i' to mark IN first")
            return
        if not self.gt["rois"]:
            print("draw an ROI first (press 'r')")
            return

        start, end = sorted((self.pending_in, self.frame_idx))
        roi = self.prompt_roi()
        cls = self.prompt_class()
        if cls is None:
            print("cancelled")
            return
        count = self.prompt_count()

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
        print(f"✅ {cls} x{count} [{start}-{end}] @{roi} ({end - start + 1} frames)")

    def save_roi(self):
        if len(self.roi_points) != 2:
            print("click two points first")
            return
        name = input("ROI name: ").strip()
        if not name:
            print("cancelled")
            return
        if any(r["name"] == name for r in self.gt["rois"]):
            print(f"❌ ROI {name!r} already exists")
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
        print(f"✅ ROI {name!r} added")

    def delete_roi(self):
        if not self.gt["rois"]:
            return
        print("ROIs: " + ", ".join(f"[{i}] {r['name']}" for i, r in enumerate(self.gt["rois"])))
        raw = input("delete which? (index, blank to cancel): ").strip()
        if not raw.isdigit() or not (0 <= int(raw) < len(self.gt["rois"])):
            print("cancelled")
            return
        roi = self.gt["rois"][int(raw)]
        used = len(self.segments_of(roi["name"]))
        if used:
            print(f"❌ {roi['name']!r} still has {used} segment(s); delete those first")
            return
        self.push_undo()
        self.gt["rois"].pop(int(raw))
        if self.active_roi == roi["name"]:
            self.active_roi = self.gt["rois"][0]["name"] if self.gt["rois"] else None
        print(f"🗑 ROI {roi['name']!r} deleted")

    def edit_segment(self):
        if not (0 <= self.sel_idx < len(self.gt["segments"])):
            print("select a segment first ('n'/'p')")
            return
        seg = self.gt["segments"][self.sel_idx]
        print(f"\nediting #{seg['id']}: {seg['class']}x{seg['count']} "
              f"[{seg['start_frame']}-{seg['end_frame']}] @{seg['roi']}")
        field = input("field (class/count/start/end/roi/note, blank to cancel): ").strip().lower()
        if not field:
            return

        self.push_undo()
        if field == "class":
            cls = self.prompt_class()
            if cls:
                seg["class"] = cls
        elif field == "count":
            seg["count"] = self.prompt_count(seg["count"])
        elif field in ("start", "end"):
            raw = input(f"{field} frame [{self.frame_idx}]: ").strip()
            value = int(raw) if raw.lstrip("-").isdigit() else self.frame_idx
            seg[f"{field}_frame"] = max(0, min(self.src.total_frames - 1, value))
            if seg["start_frame"] > seg["end_frame"]:
                seg["start_frame"], seg["end_frame"] = seg["end_frame"], seg["start_frame"]
        elif field == "roi":
            seg["roi"] = self.prompt_roi()
        elif field == "note":
            seg["note"] = input("note: ").strip()
        else:
            self.undo_stack.pop()
            print(f"unknown field {field!r}")
            return
        print(f"✅ #{seg['id']} -> {seg['class']}x{seg['count']} "
              f"[{seg['start_frame']}-{seg['end_frame']}] @{seg['roi']}")

    def delete_segment(self):
        if not (0 <= self.sel_idx < len(self.gt["segments"])):
            print("select a segment first ('n'/'p')")
            return
        seg = self.gt["segments"][self.sel_idx]
        if input(f"delete #{seg['id']} {seg['class']} "
                 f"[{seg['start_frame']}-{seg['end_frame']}]? (y/n): ").strip().lower() != "y":
            return
        self.push_undo()
        self.gt["segments"].pop(self.sel_idx)
        self.sel_idx = min(self.sel_idx, len(self.gt["segments"]) - 1)
        print(f"🗑 segment #{seg['id']} deleted")

    def select_step(self, delta):
        if not self.gt["segments"]:
            return
        self.sel_idx = (self.sel_idx + delta) % len(self.gt["segments"])
        seg = self.gt["segments"][self.sel_idx]
        print(f"→ #{seg['id']} {seg['class']}x{seg['count']} "
              f"[{seg['start_frame']}-{seg['end_frame']}] @{seg['roi']}")

    def save(self):
        self.gt["total_frames"] = self.src.total_frames
        self.gt["fps"] = self.src.fps
        path = save_gt(self.project_dir, self.gt)
        self.dirty = False
        print(f"💾 {len(self.gt['segments'])} segments -> {path}")

    # -- main loop ---------------------------------------------------------

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, self.disp_w, self.disp_h + self.strip_h)
        cv2.createTrackbar(TRACKBAR, WINDOW, 0, max(1, self.src.total_frames - 1),
                           self.on_trackbar)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        print(HELP)

        while True:
            cv2.imshow(WINDOW, self.render())
            delay = max(1, int(1000 / self.src.fps)) if self.playing else 20
            key = cv2.waitKey(delay) & 0xFF

            if self.playing and key == 255:
                if self.frame_idx >= self.src.total_frames - 1:
                    self.playing = False
                else:
                    self.seek(self.frame_idx + 1)
                continue

            if key == 255:
                continue
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
                raw = input("goto frame: ").strip()
                if raw.isdigit():
                    self.seek(int(raw))
            elif key == ord("i"):
                self.pending_in = self.frame_idx
                print(f"IN @ {self.frame_idx}")
            elif key == ord("o"):
                self.mark_out()
            elif key == ord("c"):
                self.pending_in = None
                print("pending IN cancelled")
            elif key == ord("r"):
                self.roi_mode = not self.roi_mode
                self.roi_points = []
                print("ROI mode " + ("ON: click 2 points, 's' to save" if self.roi_mode else "off"))
            elif key == ord("R"):
                self.delete_roi()
            elif key == ord("s") and self.roi_mode:
                self.save_roi()
            elif key == ord("t"):
                if self.gt["rois"]:
                    names = [r["name"] for r in self.gt["rois"]]
                    idx = names.index(self.active_roi) if self.active_roi in names else -1
                    self.active_roi = names[(idx + 1) % len(names)]
                    print(f"active ROI: {self.active_roi}")
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
            elif key in (ord("q"), 27):
                if self.roi_mode and key == 27:
                    self.roi_mode = False
                    self.roi_points = []
                    print("ROI mode off")
                    continue
                if self.dirty:
                    choice = input("Save before quit? (y/n/c): ").strip().lower()
                    if choice == "c":
                        continue
                    if choice == "y":
                        self.save()
                break

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
