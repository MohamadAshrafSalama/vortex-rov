"""
Grid-based coral health analysis for MATE ROV 2024.

Two images taken at the same location at different times are compared
on an NxN grid. Each cell is classified independently into one of four
health states (healthy, bleached, dead, background) using three
complementary signals:
  1. HSV color classification — dominant hue indicates coral condition
  2. Histogram correlation   — spectral similarity between time points
  3. Structural similarity (SSIM) — texture change over time

The three signals are blended into a per-cell severity score and a
change label is assigned from a transition table.
"""

import cv2
import numpy as np


# HSV ranges for coral health categories
HEALTHY_LOWER = np.array([35, 45, 40])
HEALTHY_UPPER = np.array([90, 255, 220])

BLEACHED_LOWER = np.array([0, 0, 170])
BLEACHED_UPPER = np.array([180, 45, 255])

DEAD_LOWER = np.array([8, 50, 25])
DEAD_UPPER = np.array([25, 200, 140])

# Health state codes
STATE_BACKGROUND = 0
STATE_HEALTHY = 1
STATE_BLEACHED = 2
STATE_DEAD = 3

# Change codes
CHANGE_NONE = 0
CHANGE_BLEACHING = 1
CHANGE_RECOVERY = 2
CHANGE_DEATH = 3
CHANGE_COLONISATION = 4
CHANGE_GROWTH = 5

CHANGE_LABELS = {
    CHANGE_NONE: "no_change",
    CHANGE_BLEACHING: "bleaching",
    CHANGE_RECOVERY: "recovery",
    CHANGE_DEATH: "death",
    CHANGE_COLONISATION: "colonisation",
    CHANGE_GROWTH: "growth",
}

CHANGE_COLORS = {
    CHANGE_NONE: (100, 100, 100),
    CHANGE_BLEACHING: (0, 200, 255),
    CHANGE_RECOVERY: (0, 220, 0),
    CHANGE_DEATH: (0, 0, 220),
    CHANGE_COLONISATION: (255, 180, 0),
    CHANGE_GROWTH: (0, 255, 200),
}

TRANSITION_TABLE = {
    (STATE_BACKGROUND, STATE_BACKGROUND): CHANGE_NONE,
    (STATE_BACKGROUND, STATE_HEALTHY):    CHANGE_GROWTH,
    (STATE_BACKGROUND, STATE_BLEACHED):   CHANGE_GROWTH,
    (STATE_BACKGROUND, STATE_DEAD):       CHANGE_NONE,
    (STATE_HEALTHY, STATE_HEALTHY):       CHANGE_NONE,
    (STATE_HEALTHY, STATE_BLEACHED):      CHANGE_BLEACHING,
    (STATE_HEALTHY, STATE_DEAD):          CHANGE_DEATH,
    (STATE_HEALTHY, STATE_BACKGROUND):    CHANGE_DEATH,
    (STATE_BLEACHED, STATE_HEALTHY):      CHANGE_RECOVERY,
    (STATE_BLEACHED, STATE_BLEACHED):     CHANGE_NONE,
    (STATE_BLEACHED, STATE_DEAD):         CHANGE_DEATH,
    (STATE_BLEACHED, STATE_BACKGROUND):   CHANGE_DEATH,
    (STATE_DEAD, STATE_HEALTHY):          CHANGE_COLONISATION,
    (STATE_DEAD, STATE_BLEACHED):         CHANGE_COLONISATION,
    (STATE_DEAD, STATE_DEAD):             CHANGE_NONE,
    (STATE_DEAD, STATE_BACKGROUND):       CHANGE_NONE,
}


def _classify_cell_hsv(cell_bgr, min_pixels=100):
    """Classify a single grid cell by dominant HSV range."""
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)

    healthy_px = cv2.countNonZero(cv2.inRange(hsv, HEALTHY_LOWER, HEALTHY_UPPER))
    bleached_px = cv2.countNonZero(cv2.inRange(hsv, BLEACHED_LOWER, BLEACHED_UPPER))
    dead_px = cv2.countNonZero(cv2.inRange(hsv, DEAD_LOWER, DEAD_UPPER))

    best = max(healthy_px, bleached_px, dead_px)
    if best < min_pixels:
        return STATE_BACKGROUND
    if best == healthy_px:
        return STATE_HEALTHY
    if best == bleached_px:
        return STATE_BLEACHED
    return STATE_DEAD


def _histogram_correlation(cell_a, cell_b):
    """3-channel histogram correlation between two BGR cells. Range: [-1, 1]."""
    hist_a = cv2.calcHist([cell_a], [0, 1, 2], None, [8, 8, 8],
                          [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([cell_b], [0, 1, 2], None, [8, 8, 8],
                          [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def _ssim_score(cell_a, cell_b):
    """Simplified SSIM on grayscale cells. Returns [-1, 1]."""
    ga = cv2.cvtColor(cell_a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(cell_b, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Resize to same shape if needed
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
        gb = gb.astype(np.float64)

    mu_a, mu_b = ga.mean(), gb.mean()
    sig_a, sig_b = ga.std(), gb.std()
    sig_ab = ((ga - mu_a) * (gb - mu_b)).mean()

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a ** 2 + sig_b ** 2 + C2)
    return float(num / (den + 1e-12))


def _health_score_from_state(state):
    """Map a state code to a 0-100 health score."""
    return {
        STATE_HEALTHY: 100,
        STATE_BLEACHED: 40,
        STATE_DEAD: 5,
        STATE_BACKGROUND: 0,
    }.get(state, 0)


class CoralHealthAnalyzer:
    """Grid-based before/after coral health analysis."""

    def __init__(
        self,
        grid_size=4,
        cell_min_pixels=100,
        hist_weight=0.4,
        ssim_weight=0.3,
        hsv_weight=0.3,
        target_size=(800, 800),
    ):
        self.grid_size = grid_size
        self.cell_min_pixels = cell_min_pixels
        self.hist_weight = hist_weight
        self.ssim_weight = ssim_weight
        self.hsv_weight = hsv_weight
        self.target_size = target_size

    def _split_grid(self, image):
        h, w = image.shape[:2]
        ch = h // self.grid_size
        cw = w // self.grid_size
        cells = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                row.append(image[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw])
            cells.append(row)
        return cells

    def classify_grid(self, image):
        """Return NxN array of state codes for image."""
        img = cv2.resize(image, self.target_size)
        cells = self._split_grid(img)
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                grid[r, c] = _classify_cell_hsv(cells[r][c], self.cell_min_pixels)
        return grid

    def compare(self, before_img, after_img):
        """Full analysis pipeline. Returns a result dict.

        Keys:
            before_grid, after_grid: NxN state arrays
            change_grid: NxN change code array
            cell_scores: NxN severity score [0-100] (100 = severe change)
            health_report: list of per-cell dicts
            summary: aggregated counts
        """
        before = cv2.resize(before_img, self.target_size)
        after = cv2.resize(after_img, self.target_size)

        before_grid = self.classify_grid(before)
        after_grid = self.classify_grid(after)

        before_cells = self._split_grid(before)
        after_cells = self._split_grid(after)

        change_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        cell_scores = np.zeros((self.grid_size, self.grid_size), dtype=float)
        health_report = []

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                b_state = int(before_grid[r, c])
                a_state = int(after_grid[r, c])
                change = TRANSITION_TABLE.get((b_state, a_state), CHANGE_NONE)
                change_grid[r, c] = change

                hist_corr = _histogram_correlation(before_cells[r][c], after_cells[r][c])
                ssim = _ssim_score(before_cells[r][c], after_cells[r][c])

                # Similarity signals — convert to change severity (0=no change, 100=severe)
                hist_change = (1.0 - max(hist_corr, 0.0)) * 100.0
                ssim_change = (1.0 - max(ssim, 0.0)) * 100.0

                # HSV-based health delta
                b_health = _health_score_from_state(b_state)
                a_health = _health_score_from_state(a_state)
                hsv_change = abs(b_health - a_health)

                severity = (
                    self.hist_weight * hist_change
                    + self.ssim_weight * ssim_change
                    + self.hsv_weight * hsv_change
                )
                cell_scores[r, c] = round(severity, 1)

                health_report.append({
                    "row": r,
                    "col": c,
                    "before_state": b_state,
                    "after_state": a_state,
                    "change": change,
                    "change_label": CHANGE_LABELS[change],
                    "hist_correlation": round(hist_corr, 3),
                    "ssim": round(ssim, 3),
                    "severity": cell_scores[r, c],
                })

        counts = {label: 0 for label in CHANGE_LABELS.values()}
        for entry in health_report:
            counts[entry["change_label"]] += 1

        return {
            "before_grid": before_grid,
            "after_grid": after_grid,
            "change_grid": change_grid,
            "cell_scores": cell_scores,
            "health_report": health_report,
            "summary": counts,
            "mean_severity": round(float(cell_scores.mean()), 1),
        }

    def draw_result(self, after_img, result, line_color=(255, 255, 255)):
        """Draw grid overlay and change labels on the after image."""
        out = cv2.resize(after_img.copy(), self.target_size)
        h, w = out.shape[:2]
        ch = h // self.grid_size
        cw = w // self.grid_size

        for entry in result["health_report"]:
            r, c = entry["row"], entry["col"]
            change = entry["change"]
            color = CHANGE_COLORS[change]
            pt1 = (c * cw, r * ch)
            pt2 = ((c + 1) * cw, (r + 1) * ch)

            if change != CHANGE_NONE:
                cv2.rectangle(out, pt1, pt2, color, 3)
                label = f"{entry['change_label']}"
                cv2.putText(
                    out, label, (pt1[0] + 4, pt1[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
                )
            sev_label = f"sev:{entry['severity']:.0f}"
            cv2.putText(
                out, sev_label, (pt1[0] + 4, pt2[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1
            )

        # Grid lines
        for i in range(1, self.grid_size):
            cv2.line(out, (0, i * ch), (w, i * ch), line_color, 1)
            cv2.line(out, (i * cw, 0), (i * cw, h), line_color, 1)

        return out

    def build_legend(self, width=800, height=50):
        """Build a colour-coded legend bar."""
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        items = [
            (k, v) for k, v in CHANGE_LABELS.items() if k != CHANGE_NONE
        ]
        col_w = width // len(items)
        for i, (code, label) in enumerate(items):
            color = CHANGE_COLORS[code]
            x0 = i * col_w
            cv2.rectangle(legend, (x0, 0), (x0 + col_w - 2, height), color, -1)
            cv2.putText(
                legend, label, (x0 + 4, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1
            )
        return legend

