"""
Step 1: Split raw DenaCam stereo frame -> undistort + rectify.

Segment A (already working): split raw frame into left/right images.
Segment B (new):             undistort + stereo-rectify the pair.

Usage:
    python step1_split_stereo.py <raw_frame.jpg>

Output:
    left.png, right.png                  (Segment A - raw split)
    left_rectified.png, right_rectified.png  (Segment B - new)
    rectification_check.png              (Segment B - visual sanity check)
"""

import sys
import os
import cv2
import numpy as np


# =========================================================
# SEGMENT A: Split raw frame into left/right (unchanged)
# =========================================================

def split_stereo_frame(frame):
    h, w = frame.shape[:2]
    half = w // 2
    left = frame[:, :half]
    right = frame[:, half:]
    return left, right


# =========================================================
# SEGMENT B: Undistort + rectify (new)
# =========================================================

# Camera intrinsics derived from DenaCam calibration data:
# focal length f = 5.404 mm, pixel size ps = 0.0022 mm -> f_px ~ 2456 px
f_px = 2456.0

# Stereo baseline (mm), from calibration data
baseline_mm = 49.885

# Distortion coefficients [k1, k2, p1, p2, k3] (Brown-Conrady model)
# PLACEHOLDER: replace with your lab's real calibration values before
# trusting any accuracy numbers from this pipeline.
dist_left = np.zeros(5, dtype=np.float64)
dist_right = np.zeros(5, dtype=np.float64)

# Stereo extrinsics: rotation left->right (assume aligned for now),
# translation = baseline along X axis.
R_stereo = np.eye(3, dtype=np.float64)
T_stereo = np.array([[baseline_mm], [0], [0]], dtype=np.float64)


def build_intrinsics(image_width, image_height, f_px):
    cx = image_width / 2.0
    cy = image_height / 2.0
    K = np.array([
        [f_px, 0,    cx],
        [0,    f_px, cy],
        [0,    0,    1]
    ], dtype=np.float64)
    return K


def rectify_pair(left, right, K_left, dist_left, K_right, dist_right, R, T):
    image_size = (left.shape[1], left.shape[0])  # (width, height)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, dist_left, K_right, dist_right,
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K_left, dist_left, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        K_right, dist_right, R2, P2, image_size, cv2.CV_32FC1)

    left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    return left_rect, right_rect, Q


def save_rectification_check(left_rect, right_rect, out_path):
    """Stack rectified images with horizontal guide lines.
    If calibration is correct, the same real-world feature should
    line up on the same horizontal line in both images."""
    stacked = np.vstack([left_rect, right_rect])
    for y in range(0, stacked.shape[0], 40):
        cv2.line(stacked, (0, y), (stacked.shape[1], y), (0, 255, 0), 1)
    cv2.imwrite(out_path, stacked)


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python step1_split_stereo.py <raw_frame.jpg>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.isfile(input_path):
        print(f"ERROR: File not found: {input_path}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)

    print(f"Reading: {input_path}")
    frame = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if frame is None:
        print(f"ERROR: OpenCV could not decode the image: {input_path}")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Loaded image: width={w}, height={h}")

    # --- Segment A: split ---
    left, right = split_stereo_frame(frame)
    print(f"Left image shape:  {left.shape}")
    print(f"Right image shape: {right.shape}")
    cv2.imwrite("left.png", left)
    cv2.imwrite("right.png", right)
    print("Saved: left.png, right.png")

    # --- Segment B: undistort + rectify ---
    img_h, img_w = left.shape[:2]
    K_left = build_intrinsics(img_w, img_h, f_px)
    K_right = K_left.copy()

    print("Running stereo rectification...")
    left_rect, right_rect, Q = rectify_pair(
        left, right, K_left, dist_left, K_right, dist_right,
        R_stereo, T_stereo
    )

    cv2.imwrite("left_rectified.png", left_rect)
    cv2.imwrite("right_rectified.png", right_rect)
    np.save("Q_disparity_to_depth.npy", Q)
    save_rectification_check(left_rect, right_rect, "rectification_check.png")

    print("Saved: left_rectified.png, right_rectified.png")
    print("Saved: rectification_check.png (visual sanity check)")
    print("Saved: Q_disparity_to_depth.npy (for later triangulation/depth step)")


if __name__ == "__main__":
    main()


#  To run: python step1_stereo_preprocessing.py Trial_Jugad.png