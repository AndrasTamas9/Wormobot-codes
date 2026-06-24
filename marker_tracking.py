"""Track colored video markers, calibrate pixel distances, and export results.

The script detects green, blue, red, and yellow markers in each frame, converts
pixel coordinates to millimetres using a calibration pattern, preserves true
frame timing through ffprobe, and can generate an annotated output video.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Dict, List

import cv2
import numpy as np


Point = Tuple[float, float]
HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


@dataclass
class MarkerConfig:
    """Color-segmentation and visualization settings for one marker."""

    name: str
    hsv_ranges: Sequence[HSVRange]
    min_area: float = 20.0
    max_area: Optional[float] = None
    draw_color_bgr: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class CalibrationResult:
    """Spatial calibration values derived from the white-square pattern."""

    px_per_mm: float
    mm_per_px: float
    median_gap_px: float
    gap_count: int
    boxes: List[Tuple[int, int, int, int]]


GREEN = MarkerConfig(
    name="green",
    hsv_ranges=(
        ((35, 40, 40), (90, 255, 255)),
    ),
    min_area=20,
    max_area=2000,
    draw_color_bgr=(0, 255, 0),
)

BLUE = MarkerConfig(
    name="blue",
    hsv_ranges=(
        ((90, 30, 70), (140, 255, 255)),
    ),
    min_area=20,
    max_area=500,
    draw_color_bgr=(255, 0, 0),
)

RED = MarkerConfig(
    name="red",
    hsv_ranges=(
        ((0, 100, 60), (8, 255, 255)),
        ((172, 100, 60), (179, 255, 255)),
    ),
    min_area=70,
    max_area=500,
    draw_color_bgr=(0, 0, 255),
)

YELLOW = MarkerConfig(
    name="yellow",
    hsv_ranges=(
        ((23, 120, 90), (35, 255, 255)),
    ),
    min_area=15,
    max_area=400,
    draw_color_bgr=(0, 255, 255),
)

MARKERS = (GREEN, BLUE, RED, YELLOW)


def make_mask(frame_bgr: np.ndarray, cfg: MarkerConfig) -> np.ndarray:
    """Create a cleaned binary HSV mask for the configured marker color."""

    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower_hsv, upper_hsv in cfg.hsv_ranges:
        lower = np.array(lower_hsv, dtype=np.uint8)
        upper = np.array(upper_hsv, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask_total = cv2.bitwise_or(mask_total, mask)

    kernel = np.ones((3, 3), np.uint8)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask_total


def contour_center_and_area(contour) -> Optional[Tuple[float, float, float]]:
    """Return a contour's centroid and area, or ``None`` for zero moments."""

    area = cv2.contourArea(contour)
    M = cv2.moments(contour)

    if abs(M["m00"]) < 1e-12:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy, area


def get_candidates(mask: np.ndarray, cfg: MarkerConfig) -> List[Tuple[float, float, float]]:
    """Extract contour centroids whose areas satisfy the marker constraints."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        result = contour_center_and_area(c)
        if result is None:
            continue

        cx, cy, area = result

        if area < cfg.min_area:
            continue
        if cfg.max_area is not None and area > cfg.max_area:
            continue

        candidates.append((cx, cy, area))

    return candidates


def distance(p: Point, q: Point) -> float:
    """Compute the Euclidean distance between two image points."""

    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def choose_largest(candidates: List[Tuple[float, float, float]]):
    """Select the candidate with the largest contour area."""

    if not candidates:
        return None, None, 0.0

    cx, cy, area = max(candidates, key=lambda item: item[2])
    return cx, cy, area


def choose_near_anchors(
    candidates: List[Tuple[float, float, float]],
    anchors: Sequence[Point],
    max_anchor_distance: float,
):
    """Select a candidate close to any anchor, with a small area-based bonus."""

    if not candidates:
        return None, None, 0.0

    if not anchors:
        return choose_largest(candidates)

    valid = []
    for cx, cy, area in candidates:
        p = (cx, cy)
        d = min(distance(p, a) for a in anchors)

        if d <= max_anchor_distance:
            score = d - 0.02 * area
            valid.append((score, cx, cy, area))

    if not valid:
        return None, None, 0.0

    _, cx, cy, area = min(valid, key=lambda item: item[0])
    return cx, cy, area


def detect_marker_largest(frame_bgr: np.ndarray, cfg: MarkerConfig):
    """Detect the largest valid component for a marker color."""

    mask = make_mask(frame_bgr, cfg)
    candidates = get_candidates(mask, cfg)
    cx, cy, area = choose_largest(candidates)
    return cx, cy, area, mask


def detect_marker_near_anchors(
    frame_bgr: np.ndarray,
    cfg: MarkerConfig,
    anchors: Sequence[Point],
    max_anchor_distance: float,
):
    """Detect a marker candidate constrained by proximity to anchor points."""

    mask = make_mask(frame_bgr, cfg)
    candidates = get_candidates(mask, cfg)
    cx, cy, area = choose_near_anchors(candidates, anchors, max_anchor_distance)
    return cx, cy, area, mask


def choose_near_point(
    candidates: List[Tuple[float, float, float]],
    target: Point,
    max_distance_px: float,
):
    """Select a candidate within a maximum pixel distance of a target point."""

    if not candidates:
        return None, None, 0.0

    valid = []
    for cx, cy, area in candidates:
        d = distance((cx, cy), target)

        if d <= max_distance_px:
            score = d - 0.02 * area
            valid.append((score, cx, cy, area))

    if not valid:
        return None, None, 0.0

    _, cx, cy, area = min(valid, key=lambda item: item[0])
    return cx, cy, area


def detect_marker_near_point(
    frame_bgr: np.ndarray,
    cfg: MarkerConfig,
    target: Point,
    max_distance_px: float,
):
    """Detect a marker candidate near a specified target point."""

    mask = make_mask(frame_bgr, cfg)
    candidates = get_candidates(mask, cfg)
    cx, cy, area = choose_near_point(candidates, target, max_distance_px)
    return cx, cy, area, mask


def valid_point(cx, cy) -> bool:
    """Return whether both coordinates of a detected point are available."""

    return cx is not None and cy is not None


def detect_calibration_from_frame(
    frame_bgr: np.ndarray,
    calib_gap_mm: float = 67.0,
    bottom_fraction: float = 0.22,
    white_threshold: int = 150,
    min_square_area: float = 300.0,
    max_square_area: float = 10000.0,
    debug_path: Optional[str] = None,
) -> CalibrationResult:
    """Estimate the pixel-to-millimetre scale from white calibration squares."""

    h, w = frame_bgr.shape[:2]
    y0 = int(h * (1.0 - bottom_fraction))
    roi = frame_bgr[y0:h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_square_area or area > max_square_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 0 or bh <= 0:
            continue

        aspect = bw / bh

        if not (0.55 <= aspect <= 1.8):
            continue

        # Convert the ROI-relative coordinate to a full-frame coordinate.
        boxes.append((x, y + y0, bw, bh))

    # Sort the detected calibration boxes from left to right.
    boxes.sort(key=lambda b: b[0])

    if len(boxes) < 2:
        raise RuntimeError(
            "Not enough white calibration squares were detected. "
            "Try lowering --calib-white-threshold, "
            "or verify that the calibration scale is visible at the bottom of the video."
        )

    gaps = []
    for b1, b2 in zip(boxes[:-1], boxes[1:]):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        right_edge_1 = x1 + w1
        left_edge_2 = x2
        gap_px = left_edge_2 - right_edge_1

        # Reject clearly invalid gaps.
        if gap_px <= 5:
            continue

        # An excessively large distance may indicate a missed calibration square.
        # Therefore, exclude overly large gaps from the calibration.
        if gap_px > 250:
            continue

        gaps.append(float(gap_px))

    if len(gaps) < 1:
        raise RuntimeError(
            "No valid black gap could be measured between the white calibration squares."
        )

    median_gap_px = float(np.median(gaps))
    px_per_mm = median_gap_px / calib_gap_mm
    mm_per_px = calib_gap_mm / median_gap_px

    if debug_path is not None:
        debug = frame_bgr.copy()

        cv2.rectangle(debug, (0, y0), (w - 1, h - 1), (255, 255, 0), 2)

        for i, (x, y, bw, bh) in enumerate(boxes):
            cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                debug,
                str(i),
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            debug,
            f"median gap = {median_gap_px:.2f} px = {calib_gap_mm:.2f} mm",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            f"scale = {px_per_mm:.5f} px/mm, {mm_per_px:.5f} mm/px",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(debug_path, debug)

    return CalibrationResult(
        px_per_mm=px_per_mm,
        mm_per_px=mm_per_px,
        median_gap_px=median_gap_px,
        gap_count=len(gaps),
        boxes=boxes,
    )


def draw_detection(
    frame: np.ndarray,
    name: str,
    cx,
    cy,
    area,
    color_bgr,
    text_y: int,
    mm_per_px: Optional[float] = None,
):
    """
    Draw a detected marker and its coordinates on the annotated video frame.
    """
    if cx is None or cy is None:
        cv2.putText(
            frame,
            f"{name}: not found",
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_bgr,
            2,
            cv2.LINE_AA,
        )
        return

    p = (int(round(cx)), int(round(cy)))

    cv2.circle(frame, p, 10, color_bgr, 2)
    cv2.drawMarker(
        frame,
        p,
        color_bgr,
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )

    if mm_per_px is None:
        label = f"{name}: ({cx:.1f}, {cy:.1f}) px, A={area:.0f}"
    else:
        label = f"{name}: ({cx * mm_per_px:.1f}, {cy * mm_per_px:.1f}) mm"

    cv2.putText(
        frame,
        label,
        (p[0] + 15, p[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color_bgr,
        2,
        cv2.LINE_AA,
    )



def read_frame_timestamps_ffprobe(video_path: str) -> List[float]:
    """
    Read the true frame timestamps from the first video stream with ffprobe.

    OpenCV's ``CAP_PROP_FPS`` often reports only the nominal frame rate stored
    in the container header. For variable-frame-rate videos, phone recordings,
    and some camera-generated MKV/MP4 files, that value may differ from the
    actual frame timing.

    The function extracts ``best_effort_timestamp_time`` values in seconds.
    The ``ffprobe`` executable, distributed with FFmpeg, must be available on
    the system PATH.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0",
        video_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe was not found. Install FFmpeg "
            "or add ffprobe to the PATH environment variable."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "ffprobe could not read the frame timestamps.\n"
            f"ffprobe stderr:\n{exc.stderr}"
        ) from exc

    timestamps: List[float] = []

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        # Some containers or ffprobe versions may emit additional fields.
        # Only the first numeric value is needed.
        value = line.split(",")[0].strip()

        try:
            timestamps.append(float(value))
        except ValueError:
            continue

    if not timestamps:
        raise RuntimeError(
            "No frame timestamps could be parsed from the ffprobe output."
        )

    # Shift the timeline so that the first frame starts at t = 0.
    t0 = timestamps[0]
    timestamps = [t - t0 for t in timestamps]

    # Verify monotonicity; malformed or duplicated timestamps may occasionally occur.
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            raise RuntimeError(
                f"Non-monotonic frame timestamp sequence: "
                f"frame {i - 1}: {timestamps[i - 1]}, frame {i}: {timestamps[i]}"
            )

    return timestamps


def estimate_fps_from_timestamps(timestamps: Sequence[float], fallback_fps: float) -> float:
    """
    Estimate an effective FPS from the true frame timestamps.

    This value is used only for writing the fixed-FPS annotated video. The CSV
    time columns continue to use the original frame timestamps.
    """
    if len(timestamps) >= 2:
        duration = timestamps[-1] - timestamps[0]
        if duration > 0:
            # N frames contain N - 1 time intervals between the first and last frame.
            return (len(timestamps) - 1) / duration

    if fallback_fps > 0:
        return fallback_fps

    return 30.0

def process_video(
    video_path: str,
    csv_path: str,
    annotated_path: Optional[str] = None,
    max_anchor_distance: float = 260.0,
    red_yellow_max_distance_mm: float = 50.0,
    calib_gap_mm: float = 67.0,
    use_mm: bool = True,
    debug_calibration_path: Optional[str] = None,
    calib_white_threshold: int = 150,
    use_true_timestamps: bool = True,
):
    """Process a video, export marker trajectories, and optionally annotate it."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open the video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_timestamps: Optional[List[float]] = None
    writer_fps = fps if fps > 0 else 30.0

    if use_true_timestamps:
        frame_timestamps = read_frame_timestamps_ffprobe(video_path)
        writer_fps = estimate_fps_from_timestamps(frame_timestamps, fallback_fps=writer_fps)

        print("True frame timing:")
        print(f"  number of timestamps:        {len(frame_timestamps)}")
        print(f"  first timestamp:             {frame_timestamps[0]:.6f} s")
        print(f"  last timestamp:              {frame_timestamps[-1]:.6f} s")
        print(f"  effective FPS for MP4 output:{writer_fps:.6f}")

        if len(frame_timestamps) >= 2:
            dts = np.diff(np.array(frame_timestamps, dtype=float))
            print(f"  median frame interval:       {np.median(dts):.6f} s")
            print(f"  mean frame interval:         {np.mean(dts):.6f} s")
            print(f"  min/max frame interval:      {np.min(dts):.6f} / {np.max(dts):.6f} s")

    # Use the first frame for spatial calibration.
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read the first video frame.")

    calibration: Optional[CalibrationResult] = None

    if use_mm:
        calibration = detect_calibration_from_frame(
            first_frame,
            calib_gap_mm=calib_gap_mm,
            white_threshold=calib_white_threshold,
            debug_path=debug_calibration_path,
        )

        print("Calibration:")
        print(f"  physical black-gap length: {calib_gap_mm:.3f} mm")
        print(f"  measured median gap:       {calibration.median_gap_px:.3f} px")
        print(f"  px_per_mm:                  {calibration.px_per_mm:.6f} px/mm")
        print(f"  mm_per_px:                  {calibration.mm_per_px:.6f} mm/px")
        print(f"  number of gaps used:       {calibration.gap_count}")

        if debug_calibration_path is not None:
            print(f"  calibration debug image:   {debug_calibration_path}")

    mm_per_px = calibration.mm_per_px if calibration is not None else None

    writer = None
    if annotated_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, writer_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create the annotated video: {annotated_path}")

    previous_positions: Dict[str, Point] = {}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)

        header = ["frame", "time_s", "dt_s"]

        for cfg in MARKERS:
            header += [
                f"{cfg.name}_x_px",
                f"{cfg.name}_y_px",
                f"{cfg.name}_area_px2",
            ]

            if use_mm:
                header += [
                    f"{cfg.name}_x_mm",
                    f"{cfg.name}_y_mm",
                ]

        if use_mm:
            header += [
                "px_per_mm",
                "mm_per_px",
                "calib_gap_px",
                "calib_gap_mm",
            ]

        out.writerow(header)

        frame_idx = 0

        # The first frame has already been read for calibration,
        # so process it before reading the remaining frames.
        current_frame = first_frame

        while True:
            frame = current_frame

            results = {}

            gx, gy, garea, _ = detect_marker_largest(frame, GREEN)
            bx, by, barea, _ = detect_marker_largest(frame, BLUE)

            results["green"] = (gx, gy, garea)
            results["blue"] = (bx, by, barea)

            anchors: List[Point] = []

            if valid_point(gx, gy):
                anchors.append((gx, gy))
            if valid_point(bx, by):
                anchors.append((bx, by))

            for name in ("red", "yellow"):
                if name in previous_positions:
                    anchors.append(previous_positions[name])

            # Detect the yellow marker first.
            yx, yy, yarea, _ = detect_marker_near_anchors(
                frame,
                YELLOW,
                anchors=anchors,
                max_anchor_distance=max_anchor_distance,
            )

            results["yellow"] = (yx, yy, yarea)

            # Require the red marker to be physically close to the yellow marker.
            # By default, the red-yellow distance must not exceed 50 mm.
            #
            # If calibration is available, convert this distance from millimetres to pixels:
            #     max_distance_px = max_distance_mm / mm_per_px
            #
            # Without millimetre calibration, fall back to the anchor-distance method.
            if valid_point(yx, yy) and mm_per_px is not None and mm_per_px > 0:
                max_red_yellow_distance_px = red_yellow_max_distance_mm / mm_per_px

                rx, ry, rarea, _ = detect_marker_near_point(
                    frame,
                    RED,
                    target=(yx, yy),
                    max_distance_px=max_red_yellow_distance_px,
                )

                # If the strict yellow-proximity search finds nothing,
                # do not automatically jump to a distant noise component.
                # Leave the red-marker coordinates empty instead.
                #
                # To enable a fallback search, restore a call such as:
                # rx, ry, rarea, _ = detect_marker_near_anchors(...)
            else:
                rx, ry, rarea, _ = detect_marker_near_anchors(
                    frame,
                    RED,
                    anchors=anchors,
                    max_anchor_distance=max_anchor_distance,
                )

            results["red"] = (rx, ry, rarea)

            for name in ("green", "blue", "red", "yellow"):
                cx, cy, _ = results[name]
                if valid_point(cx, cy):
                    previous_positions[name] = (cx, cy)

            if frame_timestamps is not None and frame_idx < len(frame_timestamps):
                t = frame_timestamps[frame_idx]
                if frame_idx == 0:
                    dt = np.nan
                else:
                    dt = frame_timestamps[frame_idx] - frame_timestamps[frame_idx - 1]
            else:
                # Fall back to nominal-FPS timing when no timestamp list is available.
                t = frame_idx / fps if fps > 0 else np.nan
                dt = 1.0 / fps if fps > 0 and frame_idx > 0 else np.nan

            row = [
                frame_idx,
                "" if np.isnan(t) else f"{t:.6f}",
                "" if np.isnan(dt) else f"{dt:.6f}",
            ]

            for cfg in MARKERS:
                cx, cy, area = results[cfg.name]

                row += [
                    "" if cx is None else f"{cx:.3f}",
                    "" if cy is None else f"{cy:.3f}",
                    f"{area:.3f}",
                ]

                if use_mm:
                    if cx is None or cy is None or mm_per_px is None:
                        row += ["", ""]
                    else:
                        row += [
                            f"{cx * mm_per_px:.3f}",
                            f"{cy * mm_per_px:.3f}",
                        ]

            if use_mm and calibration is not None:
                row += [
                    f"{calibration.px_per_mm:.8f}",
                    f"{calibration.mm_per_px:.8f}",
                    f"{calibration.median_gap_px:.3f}",
                    f"{calib_gap_mm:.3f}",
                ]

            out.writerow(row)

            if writer is not None:
                ann = frame.copy()

                if calibration is not None:
                    cv2.putText(
                        ann,
                        f"scale: {calibration.median_gap_px:.2f} px = {calib_gap_mm:.1f} mm",
                        (20, height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    if valid_point(yx, yy) and mm_per_px is not None and mm_per_px > 0:
                        max_red_yellow_distance_px = red_yellow_max_distance_mm / mm_per_px
                        cv2.circle(
                            ann,
                            (int(round(yx)), int(round(yy))),
                            int(round(max_red_yellow_distance_px)),
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            ann,
                            f"red must be within {red_yellow_max_distance_mm:.1f} mm of yellow",
                            (20, height - 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                for i, cfg in enumerate(MARKERS):
                    cx, cy, area = results[cfg.name]
                    draw_detection(
                        ann,
                        cfg.name,
                        cx,
                        cy,
                        area,
                        cfg.draw_color_bgr,
                        text_y=40 + 35 * i,
                        mm_per_px=mm_per_px,
                    )

                writer.write(ann)

            frame_idx += 1

            ok, current_frame = cap.read()
            if not ok:
                break

    cap.release()
    if writer is not None:
        writer.release()


def main():
    """Parse command-line arguments and run the marker-tracking pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Track colored markers in a video, export their coordinates to CSV, "
            "and optionally create an annotated MP4."
        )
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--out", default="markers_mm.csv", help="Output CSV file")
    parser.add_argument("--annotated", default=None, help="Optional annotated MP4 file")

    parser.add_argument(
        "--anchor-distance",
        type=float,
        default=260.0,
        help=(
            "Maximum allowed distance, in pixels, between the red/yellow marker candidates and the robot anchor points. "
            "Lower it, for example to 180, if detection still jumps to the background. "
            "Raise it, for example to 320, if the true marker is lost."
        ),
    )

    parser.add_argument(
        "--red-yellow-max-distance-mm",
        type=float,
        default=50.0,
        help=(
            "Maximum allowed distance, in millimetres, between the red and yellow markers. "
            "Default: 50 mm. Raise it, for example to 60, if the constraint is too strict; "
            "lower it, for example to 40, if noise is still detected."
        ),
    )

    parser.add_argument(
        "--calib-gap-mm",
        type=float,
        default=67.0,
        help="Physical length, in millimetres, of the black gap between two adjacent white calibration squares.",
    )

    parser.add_argument(
        "--no-mm",
        action="store_true",
        help="Disable millimetre calibration and write pixel coordinates only.",
    )

    parser.add_argument(
        "--debug-calibration",
        default=None,
        help="Optional debug image showing the detected calibration squares.",
    )

    parser.add_argument(
        "--calib-white-threshold",
        type=int,
        default=150,
        help=(
            "Threshold for white calibration squares on a 0..255 scale. "
            "If they are not detected, try lowering it, for example to 120."
        ),
    )

    parser.add_argument(
        "--no-true-timestamps",
        action="store_true",
        help=(
            "Do not use true frame timestamps read by ffprobe. "
            "Instead, fall back to the legacy frame_idx / FPS timing. "
            "Not recommended for precise measurements."
        ),
    )

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        csv_path=args.out,
        annotated_path=args.annotated,
        max_anchor_distance=args.anchor_distance,
        red_yellow_max_distance_mm=args.red_yellow_max_distance_mm,
        calib_gap_mm=args.calib_gap_mm,
        use_mm=not args.no_mm,
        debug_calibration_path=args.debug_calibration,
        calib_white_threshold=args.calib_white_threshold,
        use_true_timestamps=not args.no_true_timestamps,
    )

    print(f"Created: {args.out}")
    if args.annotated:
        print(f"Created: {args.annotated}")


if __name__ == "__main__":
    main()