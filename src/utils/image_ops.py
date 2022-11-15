"""
Common image manipulation and annotation utilities.

All functions operate on BGR numpy arrays and return copies unless
documented otherwise. No state is maintained.
"""

import os
import time

import cv2
import numpy as np


def resize_fit(image, target_w, target_h, interpolation=cv2.INTER_LINEAR):
    """Resize image to fit within target dimensions preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def letterbox(image, target_w, target_h, fill=(0, 0, 0)):
    """Resize and pad image to exact dimensions with fill colour."""
    resized = resize_fit(image, target_w, target_h)
    rh, rw = resized.shape[:2]
    canvas = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
    y_off = (target_h - rh) // 2
    x_off = (target_w - rw) // 2
    canvas[y_off:y_off + rh, x_off:x_off + rw] = resized
    return canvas, (x_off, y_off, rw, rh)


def crop_roi(image, x, y, w, h, padding=0):
    """Crop a region of interest with optional padding, clamped to image bounds."""
    ih, iw = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(iw, x + w + padding)
    y2 = min(ih, y + h + padding)
    return image[y1:y2, x1:x2].copy()


def annotate_text(
    image, text, position, font_scale=0.6, color=(0, 255, 0),
    thickness=2, bg_color=None
):
    """Draw text with optional background rectangle for readability."""
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    if bg_color is not None:
        cv2.rectangle(
            out,
            (x - 2, y - th - baseline - 2),
            (x + tw + 2, y + baseline),
            bg_color, -1
        )

    cv2.putText(out, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)
    return out


def draw_fps(image, fps, position=(10, 28), color=(0, 200, 100)):
    """Overlay a formatted FPS counter on the image."""
    return annotate_text(
        image, f"FPS: {fps:.1f}", position,
        font_scale=0.55, color=color, thickness=1,
        bg_color=(0, 0, 0)
    )


def stack_horizontal(images, target_h=None):
    """Horizontally concatenate a list of BGR images, resizing to target_h."""
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h = target_h or min(img.shape[0] for img in images)
    resized = []
    for img in images:
        if img.shape[0] != h:
            scale = h / img.shape[0]
            nw = int(img.shape[1] * scale)
            img = cv2.resize(img, (nw, h))
        resized.append(img)
    return np.hstack(resized)


def stack_vertical(images, target_w=None):
    """Vertically concatenate a list of BGR images, resizing to target_w."""
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    w = target_w or min(img.shape[1] for img in images)
    resized = []
    for img in images:
        if img.shape[1] != w:
            scale = w / img.shape[1]
            nh = int(img.shape[0] * scale)
            img = cv2.resize(img, (w, nh))
        resized.append(img)
    return np.vstack(resized)


def blend_overlay(base, overlay, alpha=0.4):
    """Alpha-blend overlay onto base image."""
    return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)


def save_frame(image, directory, prefix="frame", extension="jpg", quality=92):
    """Save an image to directory with a timestamp-based filename.

    Returns the full save path.
    """
    os.makedirs(directory, exist_ok=True)
    ts = int(time.time() * 1000)
    filename = f"{prefix}_{ts}.{extension}"
    path = os.path.join(directory, filename)
    if extension.lower() in ("jpg", "jpeg"):
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(path, image)
    return path


def mask_to_overlay(frame, mask, color=(0, 255, 180), alpha=0.35):
    """Apply a binary mask as a coloured overlay on frame."""
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def draw_grid(image, rows, cols, color=(80, 80, 80), thickness=1):
    """Draw an NxM grid over the image."""
    h, w = image.shape[:2]
    out = image.copy()
    for r in range(1, rows):
        y = r * h // rows
        cv2.line(out, (0, y), (w, y), color, thickness)
    for c in range(1, cols):
        x = c * w // cols
        cv2.line(out, (x, 0), (x, h), color, thickness)
    return out
