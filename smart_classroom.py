#!/usr/bin/env python3
"""
AI Smart Classroom Monitoring System

Real-time classroom monitoring using YOLOv8.
"""

import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'ultralytics' package is required.\n"
        "Install it with: pip install ultralytics\n"
        f"(original error: {exc})"
    )
# Configuration
# COCO classes used in the project
TRACKED_CLASSES = {
    "person":      (0,  (60, 200, 60)),     # green
    "chair":       (56, (0, 165, 255)),     # orange
    "laptop":      (63, (255, 120, 0)),     # blue-ish
    "backpack":    (24, (200, 0, 200)),     # magenta
    "bottle":      (39, (255, 255, 0)),     # cyan
    "cell phone":  (67, (0, 0, 255)),       # red
    "tv":          (62, (180, 105, 255)),   # pink (monitor / TV)
}
COCO_ID_TO_NAME = {v[0]: k for k, v in TRACKED_CLASSES.items()}
CLASS_IDS = [v[0] for v in TRACKED_CLASSES.values()]
# Display labels
DISPLAY_NAMES = {
    "person": "Student",
    "chair": "Chair",
    "laptop": "Laptop",
    "backpack": "Backpack",
    "bottle": "Bottle",
    "cell phone": "Phone",
    "tv": "TV/Monitor",
}

def display_name(internal_name: str) -> str:
    return DISPLAY_NAMES.get(internal_name, internal_name.title())
# Presence timeout
PRESENCE_GRACE_SECONDS = 2.0
# Empty classroom alert delay
EMPTY_ROOM_ALERT_SECONDS = 8.0

def build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line arguments."""
    parser = argparse.ArgumentParser(description="AI Smart Classroom Monitoring System")
    parser.add_argument("--source", default="0",
                         help="Camera index (e.g. 0) or path to a video file. Default: 0")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Path/name of the YOLOv8 weights file. Default: yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.40,
                         help="Detection confidence threshold. Default: 0.40")
    parser.add_argument("--iou", type=float, default=0.45,
                         help="NMS IoU threshold used to remove overlapping boxes "
                              "of the same object. Default: 0.45")
    parser.add_argument("--capacity", type=int, default=20,
                         help="Total number of registered students / seats "
                              "in the classroom. Used for absent-count and "
                              "attendance alerts. Default: 20")
    parser.add_argument("--low-attendance", type=float, default=0.5,
                         help="Fraction of capacity below which a 'Low "
                              "Attendance' warning is shown. Default: 0.5")
    parser.add_argument("--screenshot-interval", type=float, default=30.0,
                         help="Seconds between automatic screenshots. Default: 30")
    parser.add_argument("--log-interval", type=float, default=5.0,
                         help="Seconds between CSV attendance log entries. Default: 5")
    parser.add_argument("--no-display", action="store_true",
                         help="Run headless (no on-screen window); useful on servers.")
    return parser

#2. SMALL HELPERS
def iou(box_a, box_b) -> float:
    """Calculate IoU."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0

def deduplicate_boxes(boxes, class_ids, scores, track_ids, iou_thresh=0.6):
    """
    Extra safety net on top of the model's own NMS: if two boxes of the SAME
    class still overlap heavily (e.g. a tracker glitch produced two ids for
    one physical student), keep only the higher-confidence one. This is what
    guarantees a student standing up / leaning is never double counted.
    """
    keep = [True] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    for i_idx in range(len(order)):
        i = order[i_idx]
        if not keep[i]:
            continue
        for j_idx in range(i_idx + 1, len(order)):
            j = order[j_idx]
            if not keep[j]:
                continue
            if class_ids[i] != class_ids[j]:
                continue
            if iou(boxes[i], boxes[j]) > iou_thresh:
                keep[j] = False
    return keep

def draw_transparent_rect(img, pt1, pt2, color, alpha=0.35):
    """Draw a semi-transparent filled rectangle (used for banners/panels)."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# 3. THE MONITOR
class SmartClassroomMonitor:
    """Owns the camera, the model, all tracked state, and dashboard drawing."""

    def __init__(self, args):
        self.args = args
        self.model = YOLO(args.model)

        source = int(args.source) if args.source.isdigit() else args.source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open video source: {args.source}")
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        margin_x, margin_y = int(self.frame_w * 0.03), int(self.frame_h * 0.05)
        self.roi = (margin_x, margin_y, self.frame_w - margin_x, self.frame_h - margin_y)

        self.person_last_seen = {}
        self.all_time_student_ids = set()

        self.empty_since = None
        self.last_screenshot_time = 0.0
        self.last_log_time = 0.0
        self.session_start = time.time()

        self.screenshot_dir = "screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self.csv_path = "attendance_log.csv"
        self._init_csv()

        # FPS smoothing
        self.fps_deque = deque(maxlen=30)
        self.prev_time = time.time()

        self.show_legend = True

    # CSV
    def _init_csv(self):
        is_new = not os.path.exists(self.csv_path)
        if is_new:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "present", "absent", "capacity",
                    "chairs", "laptops", "backpacks", "bottles",
                    "cell_phones", "tvs", "total_objects", "status",
                ])

    def _log_csv(self, counts, present, absent, status):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                present, absent, self.args.capacity,
                counts.get("chair", 0), counts.get("laptop", 0),
                counts.get("backpack", 0), counts.get("bottle", 0),
                counts.get("cell phone", 0), counts.get("tv", 0),
                sum(counts.values()), status,
            ])

    # detection
    def detect(self, frame):
        """
        Runs YOLOv8 tracking on the frame. Returns:
            boxes:      list of (x1,y1,x2,y2)
            class_ids:  list of coco ids
            scores:     list of confidences
            track_ids:  list of persistent tracker ids (or None)
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=CLASS_IDS,
            conf=self.args.conf,
            iou=self.args.iou,          
            verbose=False,
            tracker="bytetrack.yaml",
        )[0]

        boxes, class_ids, scores, track_ids = [], [], [], []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy().astype(int)
            conf = results.boxes.conf.cpu().numpy()
            ids = (results.boxes.id.cpu().numpy().astype(int)
                   if results.boxes.id is not None else [None] * len(xyxy))
            for box, c, s, tid in zip(xyxy, cls, conf, ids):
                boxes.append(tuple(box.astype(int)))
                class_ids.append(int(c))
                scores.append(float(s))
                track_ids.append(tid)

        keep = deduplicate_boxes(boxes, class_ids, scores, track_ids)
        boxes = [b for b, k in zip(boxes, keep) if k]
        class_ids = [c for c, k in zip(class_ids, keep) if k]
        scores = [s for s, k in zip(scores, keep) if k]
        track_ids = [t for t, k in zip(track_ids, keep) if k]
        return boxes, class_ids, scores, track_ids

    #book keeping
    def update_presence(self, class_ids, track_ids):
        """Update which student track ids are currently / recently present."""
        now = time.time()
        for cid, tid in zip(class_ids, track_ids):
            if COCO_ID_TO_NAME.get(cid) == "person" and tid is not None:
                self.person_last_seen[tid] = now
                self.all_time_student_ids.add(tid)

        stale = [tid for tid, t in self.person_last_seen.items()
                 if now - t > PRESENCE_GRACE_SECONDS]
        for tid in stale:
            del self.person_last_seen[tid]

        present = len(self.person_last_seen)
        absent = max(0, self.args.capacity - present)
        return present, absent

    def attendance_status(self, present, absent):
        pct = present / self.args.capacity if self.args.capacity else 0
        if present == 0:
            return "CLASS EMPTY", (0, 0, 255)
        if present > self.args.capacity:
            return "OVERCROWDED", (0, 128, 255)
        if pct < self.args.low_attendance:
            return "LOW ATTENDANCE", (0, 165, 255)
        return "NORMAL", (60, 200, 60)

    #alerts
    def check_alerts(self, present):
        now = time.time()
        alerts = []
        if present == 0:
            if self.empty_since is None:
                self.empty_since = now
            elif now - self.empty_since > EMPTY_ROOM_ALERT_SECONDS:
                alerts.append(("CLASS EMPTY - No students detected!", (0, 0, 255)))
        else:
            self.empty_since = None

        if present > self.args.capacity:
            alerts.append(("OVERCROWDING WARNING - Room over capacity!", (0, 128, 255)))

        if present > 0 and self.args.capacity and present / self.args.capacity < self.args.low_attendance:
            alerts.append(("LOW ATTENDANCE WARNING", (0, 165, 255)))

        return alerts

    def maybe_screenshot(self, frame, reason=""):
        now = time.time()
        if now - self.last_screenshot_time >= self.args.screenshot_interval:
            self.last_screenshot_time = now
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(self.screenshot_dir, f"snapshot_{ts}.jpg")
            cv2.imwrite(fname, frame)
            return fname
        return None

    def maybe_log(self, counts, present, absent, status):
        now = time.time()
        if now - self.last_log_time >= self.args.log_interval:
            self.last_log_time = now
            self._log_csv(counts, present, absent, status)

    # ------------------------------------------------------------ drawing -
    def draw_boxes(self, frame, boxes, class_ids, scores, track_ids):
        for box, cid, score, tid in zip(boxes, class_ids, scores, track_ids):
            name = COCO_ID_TO_NAME.get(cid, str(cid))
            label = display_name(name)
            color = TRACKED_CLASSES.get(name, (255, 255, 255))[1]
            x1, y1, x2, y2 = box

            thickness = 3 if name == "person" else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if name == "person":
                tag = f"{label} #{tid}" if tid is not None else label
                tag += f"  {score * 100:.0f}%"
            else:
                tag = f"{label} {score * 100:.0f}%"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            if name == "person":
                chip = "PRESENT"
                (cw, ch), _ = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cy1 = min(y2 + 4, frame.shape[0] - ch - 10)
                cv2.rectangle(frame, (x1, cy1), (x1 + cw + 10, cy1 + ch + 8), (60, 200, 60), -1)
                cv2.putText(frame, chip, (x1 + 5, cy1 + ch + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    def draw_roi(self, frame):
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "CLASSROOM ROI", (x1 + 6, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_live_indicator(self, frame):
        blink_on = int(time.time() * 2) % 2 == 0
        color = (0, 0, 255) if blink_on else (0, 0, 120)
        cv2.circle(frame, (30, 30), 8, color, -1)
        cv2.putText(frame, "LIVE", (44, 37), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_datetime(self, frame):
        text = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, text, (self.frame_w - tw - 20, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_legend(self, frame):
        """A slim, single-line legend strip along the very top of the frame.
        It never sits over the middle of the video, so it can't hide a
        student's face the way a large panel would. Press 'l' to hide it
        entirely if you want a completely clean view."""
        if not self.show_legend:
            return

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        pad = 8
        chip_w = 14

        items = [(display_name(name), color) for name, (_, color) in TRACKED_CLASSES.items()]
        widths = []
        for label, _ in items:
            (tw, _), _ = cv2.getTextSize(label, font, scale, thick)
            widths.append(chip_w + 4 + tw + 18)

        strip_y0 = 48
        strip_h = 22
        draw_transparent_rect(frame, (0, strip_y0), (self.frame_w, strip_y0 + strip_h), (20, 20, 20), 0.45)

        x = 12
        y = strip_y0 + 16
        for (label, color), w in zip(items, widths):
            if x + w > self.frame_w - 12:
                break  # never overflow off-screen; simply stop drawing more chips
            cv2.rectangle(frame, (x, y - 10), (x + chip_w, y + 2), color, -1)
            cv2.putText(frame, label, (x + chip_w + 4, y),
                        font, scale, (255, 255, 255), thick, cv2.LINE_AA)
            x += w

    def draw_alert_banner(self, frame, alerts):
        """Display alert banner."""
        if not alerts:
            return
        text, color = alerts[0]
        banner_y0 = 48 + (22 if self.show_legend else 0) + 4
        banner_h = 34
        draw_transparent_rect(frame, (0, banner_y0), (self.frame_w, banner_y0 + banner_h), color, 0.6)

        scale = 0.75
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        while tw > self.frame_w - 40 and scale > 0.35:
            scale -= 0.05
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)

        cv2.putText(frame, text, ((self.frame_w - tw) // 2, banner_y0 + banner_h // 2 + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_status_bar(self, frame, counts, present, absent, status, status_color, fps):
        """Draw the status bar."""
        bar_h = 54
        draw_transparent_rect(frame, (0, self.frame_h - bar_h), (self.frame_w, self.frame_h), (20, 20, 20), 0.65)
        total_objects = sum(counts.values())
        segments = [
            (f"STUDENTS: {present}", (60, 200, 60)),
            (f"ABSENT: {absent}", (0, 0, 255)),
            (f"CAPACITY: {self.args.capacity}", (255, 255, 255)),
            (f"OBJECTS: {total_objects}", (255, 255, 255)),
            (f"STATUS: {status}", status_color),
            (f"FPS: {fps:.1f}", (200, 200, 200)),
        ]

        thickness = 2
        gap = 30
        left_margin = 20

        def total_width(scale):
            w = left_margin
            for text, _ in segments:
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                w += tw + gap
            return w

        scale = 0.55
        while total_width(scale) > self.frame_w and scale > 0.3:
            scale -= 0.03

        x = left_margin
        y = self.frame_h - bar_h // 2 + 6
        for text, color in segments:
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            x += tw + gap

    # main
    def run(self):
        print("Smart Classroom Monitor started. "
              "Press 'q' to quit, 's' for a manual screenshot, 'l' to toggle the legend.")
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("End of stream or camera error.")
                break

            boxes, class_ids, scores, track_ids = self.detect(frame)
            present, absent = self.update_presence(class_ids, track_ids)

            counts = {}
            for cid in class_ids:
                name = COCO_ID_TO_NAME.get(cid, str(cid))
                if name != "person":
                    counts[name] = counts.get(name, 0) + 1

            status, status_color = self.attendance_status(present, absent)
            alerts = self.check_alerts(present)

            self.draw_boxes(frame, boxes, class_ids, scores, track_ids)
            self.draw_roi(frame)
            self.draw_live_indicator(frame)
            self.draw_datetime(frame)
            self.draw_legend(frame)

            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now
            self.fps_deque.append(1.0 / dt if dt > 0 else 0.0)
            fps = sum(self.fps_deque) / len(self.fps_deque)

            self.draw_status_bar(frame, counts, present, absent, status, status_color, fps)
            self.draw_alert_banner(frame, alerts)

            self.maybe_log(counts, present, absent, status)
            if alerts:
                self.maybe_screenshot(frame, reason=alerts[0][0])

            if not self.args.no_display:
                cv2.imshow("AI Smart Classroom Monitoring System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(os.path.join(self.screenshot_dir, f"manual_{ts}.jpg"), frame)
                    print(f"Manual screenshot saved ({ts}).")
                if key == ord("l"):
                    self.show_legend = not self.show_legend

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Session ended. Unique students seen this session: "
              f"{len(self.all_time_student_ids)}. Log saved to {self.csv_path}")

# 4. ENTRY POINT
def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    monitor = SmartClassroomMonitor(args)
    monitor.run()

if __name__ == "__main__":
    main()
