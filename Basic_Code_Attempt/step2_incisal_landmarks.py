"""
Step 2: Incisal-margin landmark detection + stereo correspondence + triangulation.

Pipeline (mirrors Suenaga et al. 2015):
  1. Manual ROI selection on the LEFT rectified image (stand-in for the paper's
     "manually selected template" step).
  2. Canny edge detection inside the ROI -> restrict corner search to those
     edge pixels only, so we only pick landmarks that sit on the incisal
     margin (tooth-to-cavity high-contrast boundary), not on random texture.
  3. Shi-Tomasi corner detection + cv2.cornerSubPix -> sub-pixel 2D landmarks
     in the LEFT image.
  4. For each left landmark, search for its match in the RIGHT image.
     Because the images are already stereo-RECTIFIED (Step 1 output), the
     epipolar line is simply the same image row -> we only search along x,
     within a disparity range, using normalized cross-correlation
     (cv2.matchTemplate, TM_CCOEFF_NORMED) on an 11x11 patch. This replaces
     the paper's full epipolar-line search with the row-only rectified
     equivalent.
  5. Phase-correlation refinement (cv2.phaseCorrelate) on the winning patch
     pair to push the match to sub-pixel accuracy, same as the paper's PC
     refinement step.
  6. Triangulate each matched (x_left, x_right, y) pair to 3D using the Q
     matrix saved by Step 1 (cv2.stereoRectify output).
  7. Save landmarks_3D.csv and a match-visualization image.

Usage:
    python step2_incisal_landmarks.py

Requires (from Step 1, in the same folder):
    left_rectified.png, right_rectified.png, Q_disparity_to_depth.npy
"""

import os
import sys
import cv2
import numpy as np
import csv

# =========================================================
# TUNABLE PARAMETERS - adjust these first if results look bad
# =========================================================

CANNY_LOW = 50
CANNY_HIGH = 150

MAX_CORNERS = 300          # upper cap on candidate landmarks per ROI
QUALITY_LEVEL = 0.02       # lower = more (weaker) corners kept
MIN_DISTANCE = 6           # min pixel spacing between corners

SUBPIX_WIN = (5, 5)        # cornerSubPix window (half-size)
SUBPIX_ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

PATCH_HALF = 5             # 11x11 patch, matches paper's 11x11 NCC window
DISPARITY_MIN = 0          # px, tune to your baseline/working distance
DISPARITY_MAX = 400        # px, tune to your baseline/working distance
NCC_MIN_SCORE = 0.5        # reject matches below this correlation score

PC_WIN = 15                # phase-correlation refinement window (odd, >= patch)


# =========================================================
# Step A: manual ROI selection
# =========================================================

# def select_roi(left_img):
#     print("Drag a box around the incisal margin region, then press ENTER/SPACE.")
#     print("Press 'c' to cancel selection.")
#     roi = cv2.selectROI("Select incisal-margin ROI (left image)", left_img,
#                          showCrosshair=True, fromCenter=False)
#     cv2.destroyAllWindows()
#     x, y, w, h = roi
#     if w == 0 or h == 0:
#         print("ERROR: empty ROI selected.")
#         sys.exit(1)
#     return x, y, w, h

# For full view:
def select_roi(left_img, max_display_height=800):
    h, w = left_img.shape[:2]
    scale = min(1.0, max_display_height / h)
    display_img = cv2.resize(left_img, (int(w * scale), int(h * scale)))

    print("Drag a box around the incisal margin region, then press ENTER/SPACE.")
    print("Press 'c' to cancel selection.")

    cv2.namedWindow("Select incisal-margin ROI (left image)", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select incisal-margin ROI (left image)", display_img,
                         showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w_roi, h_roi = roi
    if w_roi == 0 or h_roi == 0:
        print("ERROR: empty ROI selected.")
        sys.exit(1)

    # Convert ROI back to original full-resolution coordinates
    x = int(x / scale)
    y = int(y / scale)
    w_roi = int(w_roi / scale)
    h_roi = int(h_roi / scale)

    return x, y, w_roi, h_roi


# =========================================================
# Step B: edge-restricted sub-pixel landmark detection
# =========================================================

def detect_landmarks(gray_roi):
    edges = cv2.Canny(gray_roi, CANNY_LOW, CANNY_HIGH)

    corners = cv2.goodFeaturesToTrack(
        gray_roi, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE, mask=edges
    )
    if corners is None:
        print("WARNING: no landmarks found in ROI. Try lowering QUALITY_LEVEL "
              "or widening CANNY thresholds.")
        return np.empty((0, 2), dtype=np.float32)

    corners = cv2.cornerSubPix(
        gray_roi, corners, SUBPIX_WIN, SUBPIX_ZERO_ZONE, SUBPIX_CRITERIA
    )
    return corners.reshape(-1, 2)


# =========================================================
# Step C: rectified stereo correspondence (row-only search)
# =========================================================

def match_landmark(left_gray, right_gray, x, y):
    h, w = left_gray.shape
    xi, yi = int(round(x)), int(round(y))

    y0, y1 = yi - PATCH_HALF, yi + PATCH_HALF + 1
    x0, x1 = xi - PATCH_HALF, xi + PATCH_HALF + 1
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return None

    template = left_gray[y0:y1, x0:x1]

    search_x0 = max(0, xi - DISPARITY_MAX - PATCH_HALF)
    search_x1 = min(w, xi - DISPARITY_MIN + PATCH_HALF + 1)
    search_y0, search_y1 = y0, y1
    if search_x1 - search_x0 < template.shape[1]:
        return None

    search_band = right_gray[search_y0:search_y1, search_x0:search_x1]

    result = cv2.matchTemplate(search_band, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < NCC_MIN_SCORE:
        return None

    match_x_int = search_x0 + max_loc[0] + PATCH_HALF

    # --- Phase-correlation sub-pixel refinement ---
    hw = PC_WIN // 2
    ry0, ry1 = yi - hw, yi + hw + 1
    lx0, lx1 = xi - hw, xi + hw + 1
    rx0, rx1 = match_x_int - hw, match_x_int + hw + 1
    if ry0 < 0 or ry1 > h or lx0 < 0 or lx1 > w or rx0 < 0 or rx1 > w:
        return float(xi), float(match_x_int), float(max_val)

    left_win = left_gray[ry0:ry1, lx0:lx1].astype(np.float32)
    right_win = right_gray[ry0:ry1, rx0:rx1].astype(np.float32)
    if left_win.shape != right_win.shape:
        return float(xi), float(match_x_int), float(max_val)

    (dx, dy), _ = cv2.phaseCorrelate(left_win, right_win)
    match_x_subpix = match_x_int + dx

    return float(xi), match_x_subpix, float(max_val)


# =========================================================
# Step D: triangulation via Q matrix
# =========================================================

def triangulate(x_left, y_left, disparity, Q):
    vec = np.array([x_left, y_left, disparity, 1.0], dtype=np.float64)
    homog = Q @ vec
    if abs(homog[3]) < 1e-9:
        return None
    X = homog[0] / homog[3]
    Y = homog[1] / homog[3]
    Z = homog[2] / homog[3]
    return X, Y, Z


# =========================================================
# MAIN
# =========================================================

def main():
    for fname in ["left_rectified.png", "right_rectified.png", "Q_disparity_to_depth.npy"]:
        if not os.path.isfile(fname):
            print(f"ERROR: required file not found: {fname}. Run Step 1 first.")
            sys.exit(1)

    left_bgr = cv2.imread("left_rectified.png", cv2.IMREAD_COLOR)
    right_bgr = cv2.imread("right_rectified.png", cv2.IMREAD_COLOR)
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    Q = np.load("Q_disparity_to_depth.npy")

    x, y, w, h = select_roi(left_bgr)
    roi_gray = left_gray[y:y + h, x:x + w]

    landmarks_roi = detect_landmarks(roi_gray)
    print(f"Detected {len(landmarks_roi)} candidate landmarks in ROI.")

    landmarks_full = landmarks_roi + np.array([x, y], dtype=np.float32)

    results = []
    for (xl, yl) in landmarks_full:
        m = match_landmark(left_gray, right_gray, xl, yl)
        if m is None:
            continue
        x_left, x_right_subpix, score = m
        disparity = x_left - x_right_subpix
        if disparity <= 0:
            continue
        pt3d = triangulate(x_left, yl, disparity, Q)
        if pt3d is None:
            continue
        X, Y, Z = pt3d
        results.append({
            "x_left": x_left, "y_left": yl,
            "x_right": x_right_subpix, "y_right": yl,
            "disparity_px": disparity, "ncc_score": score,
            "X_mm": X, "Y_mm": Y, "Z_mm": Z
        })

    print(f"Successfully matched + triangulated {len(results)} landmarks.")

    with open("landmarks_3D.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "x_left", "y_left", "x_right", "y_right",
            "disparity_px", "ncc_score", "X_mm", "Y_mm", "Z_mm"
        ])
        writer.writeheader()
        writer.writerows(results)
    print("Saved: landmarks_3D.csv")

    vis = np.hstack([left_bgr.copy(), right_bgr.copy()])
    offset = left_bgr.shape[1]
    for r in results:
        p_left = (int(round(r["x_left"])), int(round(r["y_left"])))
        p_right = (int(round(r["x_right"])) + offset, int(round(r["y_right"])))
        cv2.circle(vis, p_left, 3, (0, 255, 0), -1)
        cv2.circle(vis, p_right, 3, (0, 255, 0), -1)
        cv2.line(vis, p_left, p_right, (0, 200, 255), 1)
    cv2.imwrite("landmarks_match_visualization.png", vis)
    print("Saved: landmarks_match_visualization.png")


if __name__ == "__main__":
    main()
