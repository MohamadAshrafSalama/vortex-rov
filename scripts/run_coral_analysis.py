"""
Coral health analysis CLI.

Compares a before and after image of a simulated coral reef using
the grid-based HSV+histogram+SSIM pipeline and saves annotated output.

Usage:
    python scripts/run_coral_analysis.py --before data/coral_before.jpg \\
           --after data/coral_after.jpg --grid 4 --output results/
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.coral_health import CoralHealthAnalyzer


def parse_args():
    p = argparse.ArgumentParser(description="Coral health assessment")
    p.add_argument("--before", required=True, help="Before image path")
    p.add_argument("--after", required=True, help="After image path")
    p.add_argument("--grid", type=int, default=4, help="Grid size (NxN)")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    before = cv2.imread(args.before)
    after = cv2.imread(args.after)
    if before is None:
        print(f"Cannot read: {args.before}")
        sys.exit(1)
    if after is None:
        print(f"Cannot read: {args.after}")
        sys.exit(1)

    analyzer = CoralHealthAnalyzer(grid_size=args.grid)
    result = analyzer.compare(before, after)

    print(f"\nCoral Health Analysis  ({args.grid}x{args.grid} grid)")
    print("-" * 44)
    for entry in result["health_report"]:
        print(
            f"  Cell [{entry['row']},{entry['col']}]  "
            f"{entry['before_state']:>12} -> {entry['change_label']:<14}  "
            f"severity={entry['severity']:.1f}"
        )

    print("\nSummary:")
    for label, count in result["summary"].items():
        if count > 0:
            print(f"  {label:<18}: {count}")
    print(f"  Mean severity        : {result['mean_severity']:.1f}")

    os.makedirs(args.output, exist_ok=True)
    annotated = analyzer.draw_result(after, result)
    legend = analyzer.build_legend(width=annotated.shape[1])
    output_img = np.vstack([annotated, legend])

    out_path = os.path.join(args.output, "coral_health_result.jpg")
    cv2.imwrite(out_path, output_img)
    print(f"\nResult saved to {out_path}")

    if not args.no_display:
        cv2.imshow("Before", cv2.resize(before, (640, 480)))
        cv2.imshow("Coral Health Result", cv2.resize(output_img, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
