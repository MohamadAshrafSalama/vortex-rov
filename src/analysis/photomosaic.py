"""
Photomosaic construction from sequential underwater frames.

Primary method: ORB keypoint detection -> BFMatcher ratio test ->
RANSAC homography -> warpPerspective + feathered blending.

Fallback method: fixed-grid tiling when texture is insufficient for
feature matching (e.g. uniform sand with no distinguishing features).

Feathered blending uses a distance-transform-based weight map to
produce smooth transitions at stitch seams.
"""

import cv2
import numpy as np


class PhotomosaicStitcher:
    """Sequential photomosaic with ORB feature matching and blending."""

    def __init__(
        self,
        orb_features=2000,
        match_ratio=0.72,
        min_matches=12,
        ransac_threshold=5.0,
        blend_feather=40,
        max_canvas_px=6000,
    ):
        self.orb_features = orb_features
        self.match_ratio = match_ratio
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.blend_feather = blend_feather
        self.max_canvas_px = max_canvas_px

        self._orb = cv2.ORB_create(nfeatures=orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _detect_and_describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        return kp, des

    def _ratio_test(self, raw_matches):
        """Lowe ratio test to filter ambiguous matches."""
        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.match_ratio * n.distance:
                good.append(m)
        return good

    def compute_homography(self, src_img, dst_img):
        """Compute homography mapping src into dst coordinate frame.

        Returns:
            H (3x3 float64) on success, or None if insufficient matches.
        """
        kp1, des1 = self._detect_and_describe(src_img)
        kp2, des2 = self._detect_and_describe(dst_img)

        if des1 is None or des2 is None:
            return None, 0

        raw = self._matcher.knnMatch(des1, des2, k=2)
        good = self._ratio_test(raw)

        if len(good) < self.min_matches:
            return None, len(good)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, inlier_mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold
        )
        inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
        return H, inlier_count

    @staticmethod
    def _feather_mask(h, w, feather):
        """Build a float weight map with smooth feathering at edges."""
        mask = np.ones((h, w), dtype=np.float32)
        if feather < 1:
            return mask
        # Use distance transform from a binary border mask
        border = np.zeros((h, w), dtype=np.uint8)
        border[feather:-feather, feather:-feather] = 255
        dist = cv2.distanceTransform(border, cv2.DIST_L2, 5)
        dist = dist / (float(feather) + 1e-6)
        return np.clip(dist, 0, 1).astype(np.float32)

    def _warp_and_blend(self, canvas, src_img, H, offset):
        """Warp src_img into canvas using homography H with offset correction."""
        out_h, out_w = canvas.shape[:2]
        H_offset = np.array([
            [1, 0, offset[0]],
            [0, 1, offset[1]],
            [0, 0, 1],
        ], dtype=np.float64) @ H

        warped = cv2.warpPerspective(src_img, H_offset, (out_w, out_h))
        warp_mask = cv2.warpPerspective(
            np.ones(src_img.shape[:2], np.float32), H_offset, (out_w, out_h)
        )

        sh, sw = src_img.shape[:2]
        src_weight = self._feather_mask(sh, sw, self.blend_feather)
        src_weight_warped = cv2.warpPerspective(src_weight, H_offset, (out_w, out_h))

        # Build canvas weight (1 where canvas has content)
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        canvas_weight = (canvas_gray > 0).astype(np.float32)

        total_w = canvas_weight + src_weight_warped + 1e-6
        for ch in range(3):
            blended = (
                canvas[:, :, ch].astype(np.float32) * canvas_weight
                + warped[:, :, ch].astype(np.float32) * src_weight_warped
            ) / total_w
            canvas[:, :, ch] = np.clip(blended, 0, 255).astype(np.uint8)

        return canvas

    def stitch_pair(self, base_img, new_img):
        """Stitch new_img onto base_img.

        Returns:
            stitched image or None if matching failed.
        """
        H, n_inliers = self.compute_homography(new_img, base_img)
        if H is None:
            return None, n_inliers

        h1, w1 = base_img.shape[:2]
        h2, w2 = new_img.shape[:2]

        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners2_t = cv2.perspectiveTransform(corners2, H)
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)

        all_corners = np.concatenate([corners1, corners2_t], axis=0)
        x_min = int(np.floor(all_corners[:, 0, 0].min()))
        y_min = int(np.floor(all_corners[:, 0, 1].min()))
        x_max = int(np.ceil(all_corners[:, 0, 0].max()))
        y_max = int(np.ceil(all_corners[:, 0, 1].max()))

        out_w = x_max - x_min
        out_h = y_max - y_min

        if out_w > self.max_canvas_px or out_h > self.max_canvas_px:
            return None, n_inliers

        offset = (-x_min, -y_min)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # Place base image at offset
        ox, oy = offset
        canvas[oy:oy + h1, ox:ox + w1] = base_img

        canvas = self._warp_and_blend(canvas, new_img, H, offset)
        return canvas, n_inliers

    def stitch_sequence(self, images, verbose=True):
        """Stitch a list of images sequentially.

        Falls back to grid_mosaic if feature matching fails at any step.
        """
        if not images:
            return None
        if len(images) == 1:
            return images[0]

        result = images[0].copy()
        for i, img in enumerate(images[1:], start=1):
            if verbose:
                print(f"  Stitching frame {i + 1}/{len(images)}...")
            stitched, n_inliers = self.stitch_pair(result, img)
            if stitched is None:
                if verbose:
                    print(f"  Insufficient matches ({n_inliers}), falling back to grid.")
                return self.grid_mosaic(images)
            if verbose:
                print(f"  OK ({n_inliers} inliers), canvas: {stitched.shape[1]}x{stitched.shape[0]}")
            result = stitched

        return result

    def grid_mosaic(self, images, cols=None):
        """Fixed-layout grid tiling fallback."""
        n = len(images)
        if n == 0:
            return None
        cols = cols or max(1, int(np.ceil(np.sqrt(n))))
        rows = int(np.ceil(n / cols))

        target_h = min(img.shape[0] for img in images)
        resized = []
        for img in images:
            scale = target_h / img.shape[0]
            nw = int(img.shape[1] * scale)
            resized.append(cv2.resize(img, (nw, target_h)))

        max_w = max(img.shape[1] for img in resized)

        grid_rows = []
        for r in range(rows):
            row_tiles = []
            for c in range(cols):
                idx = r * cols + c
                if idx < n:
                    tile = resized[idx]
                    if tile.shape[1] < max_w:
                        pad = np.zeros((target_h, max_w - tile.shape[1], 3), np.uint8)
                        tile = np.hstack([tile, pad])
                else:
                    tile = np.zeros((target_h, max_w, 3), np.uint8)
                row_tiles.append(tile)
            grid_rows.append(np.hstack(row_tiles))

        return np.vstack(grid_rows)

    def draw_keypoints(self, img1, img2):
        """Visualise matched keypoints between two images for debugging."""
        kp1, des1 = self._detect_and_describe(img1)
        kp2, des2 = self._detect_and_describe(img2)
        if des1 is None or des2 is None:
            return np.hstack([img1, img2])
        raw = self._matcher.knnMatch(des1, des2, k=2)
        good = self._ratio_test(raw)
        vis = cv2.drawMatches(
            img1, kp1, img2, kp2, good[:40], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return vis

