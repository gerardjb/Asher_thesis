import cv2
import numpy as np
"""
Utility functions for assessing image quality based on Laplacian variance and motion metrics.
These functions are designed to be used in conjunction with MediaPipe's PoseLandmarker and HandLandmarker.:
    draw_landmarks_on_image: Draws landmarks on an image with customizable styles.
    draw_fingertap: Highlights index/thumb tips for right hand in an image.
    blur_face_pose: Blackouts a square region around the face detected by PoseLandmarker.
    extract_landmarks_for_frame: Extracts landmark data for a single frame into a list of dicts.
    extract_points_for_fingertap: Extracts normalized Y coordinates for right index/thumb tips for fingertap analysis.
Private methods include:
    _resolve_landmark_indices: Resolves landmark specifiers into indices.
"""


def calculate_local_laplacian_variance(laplacian_abs_normalized, center_xy, patch_size):
    """
    Calculates the variance within a patch of a pre-calculated,
    absolute, normalized Laplacian image. Higher variance generally indicates more edges/texture (sharper).

    Args:
        laplacian_abs_normalized (np.ndarray): The absolute Laplacian image, normalized to 0-255 (uint8).
        center_xy (tuple): (x, y) pixel coordinates for the center of the patch.
        patch_size (int): The side length of the square patch (should be odd).

    Returns:
        float (variance): The variance of pixel values within the patch. Returns np.nan if
               the patch cannot be extracted (e.g., center is outside image).
    """
    h, w = laplacian_abs_normalized.shape[:2]
    cx, cy = center_xy
    half_patch = patch_size // 2

    # Calculate patch boundaries
    y_start = cy - half_patch
    y_end = cy + half_patch + 1 # Slicing excludes the end index
    x_start = cx - half_patch
    x_end = cx + half_patch + 1

    # --- Boundary Checks ---
    # Check if center is within image bounds at all
    if not (0 <= cx < w and 0 <= cy < h):
        return np.nan

    # Adjust boundaries if patch goes out of bounds
    y_start_clipped = max(0, y_start)
    y_end_clipped = min(h, y_end)
    x_start_clipped = max(0, x_start)
    x_end_clipped = min(w, x_end)

    # Check if the clipped patch has valid dimensions
    if y_start_clipped >= y_end_clipped or x_start_clipped >= x_end_clipped:
        return np.nan # Patch is entirely outside or has zero area

    # Extract the patch
    patch = laplacian_abs_normalized[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]

    # Calculate variance within the patch
    if patch.size == 0:
        return np.nan
    variance = np.var(patch)
    return float(variance)


def calculate_local_motion(diff_image_normalized, center_xy, patch_size):
    """
    Calculates the mean pixel intensity within a patch of a pre-calculated,
    normalized frame difference image. Higher mean indicates more motion.

    Args:
        diff_image_normalized (np.ndarray): The absolute difference image between
                                            two frames, normalized to 0-255 (uint8).
        center_xy (tuple): (x, y) pixel coordinates for the center of the patch.
        patch_size (int): The side length of the square patch (should be odd).

    Returns:
        float: The mean pixel value within the patch. Returns np.nan if the patch
               cannot be extracted.
    """
    h, w = diff_image_normalized.shape[:2]
    cx, cy = center_xy
    half_patch = patch_size // 2

    # Calculate patch boundaries
    y_start = cy - half_patch
    y_end = cy + half_patch + 1
    x_start = cx - half_patch
    x_end = cx + half_patch + 1

    # --- Boundary Checks ---
    if not (0 <= cx < w and 0 <= cy < h):
        return np.nan

    y_start_clipped = max(0, y_start)
    y_end_clipped = min(h, y_end)
    x_start_clipped = max(0, x_start)
    x_end_clipped = min(w, x_end)

    if y_start_clipped >= y_end_clipped or x_start_clipped >= x_end_clipped:
        return np.nan

    # Extract the patch
    patch = diff_image_normalized[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]

    # Calculate mean within the patch
    if patch.size == 0:
        return np.nan
    mean_motion = np.mean(patch)
    return float(mean_motion)


def calculate_confidence_score(laplacian_var, motion_mean, method='inverse_motion_sharpness'):
    """
    Calculates a basic confidence score based on blur and motion metrics.
    Requires normalized inputs or careful thresholding.

    Args:
        laplacian_var (float): Blur metric (e.g., variance of Laplacian). Higher = sharper.
        motion_mean (float): Motion metric (e.g., mean diff). Higher = more motion.
        method (str): The scoring method to use.
                      'inverse_motion_sharpness': Simple product (assumes higher var/lower motion is better).
                      'threshold_based': Binary 0 or 1 based on thresholds (example only).

    Returns:
        float: A confidence score (range depends on method), or np.nan if inputs are invalid.
    """
    if np.isnan(laplacian_var) or np.isnan(motion_mean):
        return np.nan

    if method == 'inverse_motion_sharpness':
        # Simple example: Assumes inputs are somewhat normalized or scaled appropriately.
        # Higher variance is good, higher motion is bad.
        # Adding epsilon to avoid division by zero if motion_mean can be 0.
        epsilon = 1e-6
        # This score increases with sharpness and decreases with motion.
        # The range is highly dependent on the input scales. Needs normalization for [0,1].
        score = laplacian_var / (motion_mean + epsilon)
        # Clamping or further normalization might be needed depending on expected ranges.
        return max(0.0, score) # Ensure non-negative

    elif method == 'threshold_based':
        # --- Example Thresholds (MUST BE TUNED BASED ON YOUR DATA) ---
        LAPLACIAN_VAR_THRESHOLD = 100 # Example: Need variance > 100 to be considered "sharp"
        MOTION_MEAN_THRESHOLD = 20    # Example: Need mean diff < 20 to be considered "stable"

        is_sharp = laplacian_var > LAPLACIAN_VAR_THRESHOLD
        is_stable = motion_mean < MOTION_MEAN_THRESHOLD

        return 1.0 if (is_sharp and is_stable) else 0.0

    else:
        print(f"Warning: Unknown confidence score method '{method}'. Returning NaN.")
        return np.nan

# --- Example Usage (for testing this module standalone) ---
if __name__ == '__main__':
    # Create dummy images
    sharp_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    cv2.putText(sharp_img, "Sharp", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    blurred_img = cv2.GaussianBlur(sharp_img, (15, 15), 0)

    # Dummy difference image
    diff_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(diff_img, (40, 40), (60, 60), (100), -1) # Area of motion

    # Calculate full Laplacian (absolute, normalized)
    lap_sharp = cv2.Laplacian(sharp_img, cv2.CV_64F)
    lap_sharp_abs = np.absolute(lap_sharp)
    lap_sharp_norm = cv2.normalize(lap_sharp_abs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    lap_blurred = cv2.Laplacian(blurred_img, cv2.CV_64F)
    lap_blurred_abs = np.absolute(lap_blurred)
    lap_blurred_norm = cv2.normalize(lap_blurred_abs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    patch_size = 21
    center_sharp = (50, 50) # Center of the sharp text area
    center_motion = (50, 50) # Center of the motion area
    center_still = (10, 10) # Still area

    # Test Laplacian Variance
    var_sharp = calculate_local_laplacian_variance(lap_sharp_norm, center_sharp, patch_size)
    var_blurred = calculate_local_laplacian_variance(lap_blurred_norm, center_sharp, patch_size)
    print(f"Local Laplacian Variance (Sharp Area): {var_sharp:.2f}")
    print(f"Local Laplacian Variance (Blurred Area): {var_blurred:.2f}")
    assert var_sharp > var_blurred, "Sharp area should have higher Laplacian variance"

    # Test Motion Mean
    motion_high = calculate_local_motion(diff_img, center_motion, patch_size)
    motion_low = calculate_local_motion(diff_img, center_still, patch_size)
    print(f"Local Mean Motion (Motion Area): {motion_high:.2f}")
    print(f"Local Mean Motion (Still Area): {motion_low:.2f}")
    assert motion_high > motion_low, "Motion area should have higher mean difference"

    # Test Confidence Score (example)
    score_good = calculate_confidence_score(var_sharp, motion_low, method='threshold_based')
    score_blur = calculate_confidence_score(var_blurred, motion_low, method='threshold_based')
    score_motion = calculate_confidence_score(var_sharp, motion_high, method='threshold_based')
    print(f"Confidence (Sharp, Still): {score_good}")
    print(f"Confidence (Blurred, Still): {score_blur}")
    print(f"Confidence (Sharp, Motion): {score_motion}")
    assert score_good == 1.0
    assert score_blur == 0.0
    assert score_motion == 0.0

    print("\nImage Quality Utils basic tests passed.")
