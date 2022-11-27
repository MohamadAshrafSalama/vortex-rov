"""
Photomosaic generation CLI.

Stitches a sequence of images into a composite mosaic using ORB feature
matching or grid fallback. Images can be provided as file paths or
captured live from a camera.

Usage:
    python scripts/run_mosaic.py --images data/frames/*.jpg --output results/
    python scripts/run_mosaic.py --camera 0 --num 8 --output results/
"""

import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.photomosaic import PhotomosaicStitcher
from src.camera.capture import CameraCapture


def parse_args():
    p = argparse.ArgumentParser(description="Photomosaic stitching")
    p.add_argument("--images", nargs="+", help="Input image paths or glob patterns")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--num", type=int, default=8, help="Frames to capture (live mode)")
    p.add_argument("--mode", choices=["feature", "grid"], default="feature")
    p.add_argument("--output", default="results")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def capture_live(camera_idx, num_frames):
    cam = CameraCapture(index=camera_idx, width=1280, height=720).open()
    images = []
    print(f"Live capture: SPACE to grab, q to stop. Target: {num_frames} frames.")
    while len(images) < num_frames:
        frame = cam.read()
        if frame is None:
            continue
        display = frame.copy()
        cv2.putText(
            display, f"Captured {len(images)}/{num_frames}  [SPACE=grab, q=done]",
            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2
        )
        cv2.imshow("Capture", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            images.append(frame.copy())
            print(f"  Frame {len(images)} captured")
        elif key == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
    return images


def main():
    args = parse_args()

    if args.images:
        paths = []
        for pattern in args.images:
            paths.extend(glob.glob(pattern))
        paths = sorted(paths)
        images = []
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                images.append(img)
                print(f"  Loaded: {p}")
        print(f"\n{len(images)} images loaded.")
    else:
        images = capture_live(args.camera, args.num)

    if not images:
        print("No images available.")
        return

    stitcher = PhotomosaicStitcher()
    print(f"\nStitching {len(images)} frames (mode={args.mode})...")

    if args.mode == "feature":
        mosaic = stitcher.stitch_sequence(images, verbose=True)
    else:
        mosaic = stitcher.grid_mosaic(images)

    if mosaic is None:
        print("Stitching failed.")
        return

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "photomosaic.jpg")
    cv2.imwrite(out_path, mosaic, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nMosaic saved to {out_path}  ({mosaic.shape[1]}x{mosaic.shape[0]})")

    if not args.no_display:
        display = mosaic
        if max(display.shape[:2]) > 1400:
            scale = 1400 / max(display.shape[:2])
            display = cv2.resize(display, (int(display.shape[1] * scale), int(display.shape[0] * scale)))
        cv2.imshow("Photomosaic", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
