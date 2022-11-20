"""
Main competition entry point — MATE ROV 2024.

Launches the selected task pipeline with a live camera feed.
All pipelines share the same preprocessing (white balance, undistortion)
and write to a shared telemetry log.

Usage:
    python scripts/run_competition.py --task transect --camera 0
    python scripts/run_competition.py --task poi --camera 0
    python scripts/run_competition.py --task boxes --camera 0
    python scripts/run_competition.py --task all --camera 0
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.camera.capture import CameraCapture
from src.camera.white_balance import WhiteBalancer
from src.detection.box_detector import BoxDetector
from src.detection.marker_detector import MarkerDetector
from src.analysis.transect_following import TransectFollower
from src.navigation.guidance import GuidanceEngine
from src.navigation.poi_mapping import POIMapper
from src.telemetry.logger import TelemetryLogger
from src.tracking.centroid_tracker import CentroidTracker


def parse_args():
    p = argparse.ArgumentParser(description="Vortex ROV 2024 competition pipeline")
    p.add_argument("--task", choices=["transect", "poi", "boxes", "all"],
                   default="all", help="Task to run")
    p.add_argument("--camera", type=int, default=0, help="Camera device index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--no-display", action="store_true", help="Run headless")
    p.add_argument("--calibration", default=None, help="Path to calibration YAML")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Components
    wb = WhiteBalancer(method="combined")
    follower = TransectFollower()
    box_detector = BoxDetector()
    marker_detector = MarkerDetector()
    guidance = GuidanceEngine(frame_width=args.width, frame_height=args.height)
    poi_mapper = POIMapper(frame_width=args.width, frame_height=args.height)
    box_tracker = CentroidTracker(max_disappeared=20, max_distance=90)

    calibrator = None
    if args.calibration and os.path.exists(args.calibration):
        from src.camera.calibration import CameraCalibrator
        calibrator = CameraCalibrator()
        calibrator.load(args.calibration)
        print(f"[main] Calibration loaded from {args.calibration}")

    log_path = os.path.join(args.output, "telemetry.csv")
    logger = TelemetryLogger(path=log_path)

    print(f"[main] Opening camera {args.camera} at {args.width}x{args.height}@{args.fps}")
    cam = CameraCapture(
        index=args.camera, width=args.width, height=args.height, fps=args.fps
    ).open()

    print(f"[main] Running task: {args.task}  |  Press 'q' to quit, 's' to save snapshot")

    frame_idx = 0
    fps_counter = 0
    fps_display = 0.0
    fps_time = time.monotonic()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            # Preprocessing
            if calibrator is not None:
                frame = calibrator.undistort(frame)
            frame = wb.correct(frame)

            annotated = frame.copy()

            # Task pipelines
            transect_result = None
            if args.task in ("transect", "all"):
                transect_result = follower.process(frame)
                annotated = follower.draw_overlay(annotated, transect_result)

            box_dets = []
            if args.task in ("boxes", "all"):
                box_dets = box_detector.detect(frame)
                tracked = box_tracker.update(box_dets)
                annotated = box_detector.draw(annotated, box_dets)
                annotated = box_tracker.draw(annotated)
                if box_dets:
                    target = box_detector.primary_target(box_dets)
                    if target:
                        g = guidance.compute(target)
                        annotated = guidance.draw_guidance(annotated, g)

            marker_dets = []
            if args.task in ("poi", "all"):
                marker_dets = marker_detector.detect(frame)
                poi_mapper.update(marker_dets)
                annotated = marker_detector.draw(annotated, marker_dets)

            # FPS overlay
            fps_counter += 1
            now = time.monotonic()
            if now - fps_time >= 1.0:
                fps_display = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now
            cv2.putText(
                annotated, f"FPS: {fps_display:.1f}", (10, args.height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 2
            )

            # Telemetry
            logger.log(
                boxes_detected=len(box_dets),
                markers_detected=len(marker_dets),
                transect_offset=transect_result["offset_norm"] if transect_result else 0.0,
                transect_steering=transect_result["steering"] if transect_result else 0.0,
            )

            if not args.no_display:
                cv2.imshow("Vortex ROV 2024", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    from src.utils.image_ops import save_frame
                    path = save_frame(annotated, args.output, prefix="snap")
                    print(f"[main] Snapshot saved: {path}")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[main] Interrupted.")
    finally:
        cam.release()
        logger.close()
        cv2.destroyAllWindows()
        if args.task in ("poi", "all"):
            poi_path = os.path.join(args.output, "poi_map.json")
            poi_mapper.save(poi_path)
            print(f"[main] POI map saved: {poi_path} ({len(poi_mapper.confirmed_pois)} confirmed)")
        print(f"[main] Telemetry: {log_path}  |  Frames processed: {frame_idx}")


if __name__ == "__main__":
    main()
