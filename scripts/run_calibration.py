"""
Camera calibration CLI.

Captures checkerboard images from a live camera or loads from disk,
runs calibration, and saves the YAML calibration file.

Usage:
    python scripts/run_calibration.py --camera 0 --rows 9 --cols 6 --num 25
    python scripts/run_calibration.py --images data/calib/*.jpg --rows 9 --cols 6
"""

import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.camera.calibration import CameraCalibrator
from src.camera.capture import CameraCapture


def parse_args():
    p = argparse.ArgumentParser(description="Camera calibration for ROV pipeline")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--rows", type=int, default=9, help="Inner corners per row")
    p.add_argument("--cols", type=int, default=6, help="Inner corners per column")
    p.add_argument("--square", type=float, default=0.025, help="Square size in metres")
    p.add_argument("--num", type=int, default=25, help="Number of frames to capture")
    p.add_argument("--images", nargs="+", help="Pre-captured calibration images")
    p.add_argument("--output", default="config/calibration.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    calibrator = CameraCalibrator(
        board_rows=args.rows, board_cols=args.cols, square_size_m=args.square
    )

    if args.images:
        paths = []
        for pattern in args.images:
            paths.extend(glob.glob(pattern))
        frames = [cv2.imread(p) for p in sorted(paths) if cv2.imread(p) is not None]
        print(f"Loaded {len(frames)} images.")
    else:
        frames = []
        cam = CameraCapture(index=args.camera, width=1280, height=720, fps=30).open()
        print(f"Capture {args.num} checkerboard frames. SPACE=capture, q=done.")

        while len(frames) < args.num:
            frame = cam.read()
            if frame is None:
                continue

            corners, _ = calibrator.detect_corners(frame)
            display = frame.copy()
            if corners is not None:
                display = calibrator.draw_corners(display, corners)
                cv2.putText(display, f"Found! [{len(frames)}/{args.num}]",
                            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
            else:
                cv2.putText(display, "No board detected",
                            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and corners is not None:
                frames.append(frame.copy())
                print(f"  Captured {len(frames)}/{args.num}")
            elif key == ord("q"):
                break

        cam.release()
        cv2.destroyAllWindows()

    if len(frames) < 6:
        print("Not enough frames. Need at least 6 with detected corners.")
        return

    print(f"\nRunning calibration on {len(frames)} frames...")
    error = calibrator.calibrate(frames)
    print(f"Reprojection error: {error:.4f} px")

    if error > 1.5:
        print("Warning: high reprojection error. Consider recapturing with better coverage.")

    calibrator.save(args.output)
    print(f"Done. Calibration saved to {args.output}")


if __name__ == "__main__":
    main()
