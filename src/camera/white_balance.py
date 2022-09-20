"""
Underwater white balance and color correction.

Water absorbs red and yellow wavelengths preferentially, leaving images
with a blue-green cast and low contrast. Three correction methods are
implemented and can be chained or selected independently.
"""

import cv2
import numpy as np


class WhiteBalancer:
    """Stateless collection of underwater color correction methods."""

    def __init__(self, method="gray_world", clahe_clip=2.0, clahe_grid=(8, 8)):
        """
        Args:
            method: 'gray_world' | 'clahe' | 'histogram_stretch' | 'combined'
            clahe_clip: CLAHE clip limit (higher = more contrast enhancement)
            clahe_grid: tile grid size for CLAHE
        """
        self.method = method
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

    def correct(self, frame):
        """Apply the configured correction method to a BGR frame."""
        if self.method == "gray_world":
            return self.gray_world(frame)
        if self.method == "clahe":
            return self.apply_clahe(frame)
        if self.method == "histogram_stretch":
            return self.histogram_stretch(frame)
        if self.method == "combined":
            return self.combined(frame)
        return frame

    @staticmethod
    def gray_world(frame):
        """Gray-world assumption: scale each channel so means are equal.

        Assumes the scene average is neutral grey. Works well when the
        underwater scene has a variety of colours. Breaks down in
        monochromatic scenes (e.g. all sand).
        """
        frame_f = frame.astype(np.float32)
        mean_b = np.mean(frame_f[:, :, 0])
        mean_g = np.mean(frame_f[:, :, 1])
        mean_r = np.mean(frame_f[:, :, 2])
        mean_all = (mean_b + mean_g + mean_r) / 3.0

        frame_f[:, :, 0] *= mean_all / (mean_b + 1e-6)
        frame_f[:, :, 1] *= mean_all / (mean_g + 1e-6)
        frame_f[:, :, 2] *= mean_all / (mean_r + 1e-6)
        return np.clip(frame_f, 0, 255).astype(np.uint8)

    def apply_clahe(self, frame):
        """CLAHE on the L channel in LAB colorspace.

        Equalises local contrast without blowing out highlights. Leaves
        hue and saturation untouched, only enhancing luminance.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        enhanced = cv2.merge([l_ch, a_ch, b_ch])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def histogram_stretch(frame, low_pct=1, high_pct=99):
        """Per-channel histogram stretching using percentile clipping.

        Clips the lowest and highest percentile pixel values and stretches
        the remaining range to [0, 255]. Robustly handles image outliers
        without being sensitive to a single bright or dark pixel.
        """
        out = np.zeros_like(frame, dtype=np.uint8)
        for ch in range(3):
            channel = frame[:, :, ch].astype(np.float32)
            lo = np.percentile(channel, low_pct)
            hi = np.percentile(channel, high_pct)
            if hi - lo < 1:
                out[:, :, ch] = channel.astype(np.uint8)
                continue
            stretched = (channel - lo) * 255.0 / (hi - lo)
            out[:, :, ch] = np.clip(stretched, 0, 255).astype(np.uint8)
        return out

    def combined(self, frame):
        """Gray-world correction followed by CLAHE — best for murky water.

        The gray-world pass removes the blue-green cast, then CLAHE lifts
        local contrast in dim areas without over-saturating bright regions.
        """
        corrected = self.gray_world(frame)
        return self.apply_clahe(corrected)

    @staticmethod
    def red_channel_compensation(frame, compensation=0.5):
        """Restore attenuated red channel for deeper scenes.

        At depth, red light is absorbed first, leaving images blue-green.
        This heuristic boosts the red channel proportionally to the
        difference between the green and red channel means.
        """
        frame_f = frame.astype(np.float32)
        mean_r = np.mean(frame_f[:, :, 2])
        mean_g = np.mean(frame_f[:, :, 1])
        if mean_g > mean_r:
            boost = compensation * (mean_g - mean_r)
            frame_f[:, :, 2] = np.clip(frame_f[:, :, 2] + boost, 0, 255)
        return frame_f.astype(np.uint8)

    @staticmethod
    def denoise(frame, h=6, template_win=7, search_win=21):
        """Fast NL-means denoising adapted for low-light underwater footage."""
        return cv2.fastNlMeansDenoisingColored(
            frame, None, h, h, template_win, search_win
        )
