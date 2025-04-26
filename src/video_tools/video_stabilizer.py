import cv2
import numpy as np
import argparse # For the main test block
import os     # For the main test block

# Try importing MediaPipe types, but make it optional for standalone use
try:
    from mediapipe.tasks.python.vision import PoseLandmarkerResult
    import mediapipe as mp
except ImportError:
    PoseLandmarkerResult = type(None)
    print("Warning: MediaPipe PoseLandmarkerResult not found. Pose masking will be disabled.")


# --- Helper function to create mask from pose ---
# Kept outside the class for now, I may end up making this a static method
def create_mask_from_pose(image_shape, pose_result: PoseLandmarkerResult, padding=20):
    """
    Creates a binary mask to exclude the area covered by detected pose landmarks.

    Args:
        image_shape: Tuple (height, width) of the image.
        pose_result: The result object from MediaPipe PoseLandmarker for the current frame.
            If None or mediapipe is not installed, a full mask is returned.
        padding: Extra pixels to add around the bounding box of the pose.

    Returns:
        mask: A binary mask (uint8 NumPy array) where the background is 255 (white)
        and the pose region to be excluded is 0 (black).
    """
    mask = np.ones((image_shape[0], image_shape[1]), dtype=np.uint8) * 255

    # Check if PoseLandmarkerResult is available and result is valid
    if PoseLandmarkerResult is None or not pose_result or not pose_result.pose_landmarks:
        return mask # Return full mask if no pose detected or mediapipe types unavailable

    # Consider only the first detected pose
    landmarks = pose_result.pose_landmarks[0]
    h, w = image_shape[:2]

    # Find bounding box of all landmarks in pixel coordinates
    min_x, min_y = w, h
    max_x, max_y = 0, 0
    valid_landmarks = False
    for landmark in landmarks:
        # Convert normalized coordinates to pixel coordinates
        # Ensure landmark coordinates are within [0, 1] before multiplying
        lx = max(0.0, min(1.0, landmark.x))
        ly = max(0.0, min(1.0, landmark.y))
        px = int(lx * w)
        py = int(ly * h)

        min_x = min(min_x, px)
        min_y = min(min_y, py)
        max_x = max(max_x, px)
        max_y = max(max_y, py)
        valid_landmarks = True

    if not valid_landmarks:
        return mask # Return full mask if landmarks were invalid

    # Apply padding and clip to image boundaries
    xmin = max(0, min_x - padding)
    ymin = max(0, min_y - padding)
    xmax = min(w, max_x + padding)
    ymax = min(h, max_y + padding)

    # Set the bounding box area in the mask to 0 (black)
    # Ensure coordinates are valid before drawing rectangle
    if xmin < xmax and ymin < ymax:
        mask[ymin:ymax, xmin:xmax] = 0

    return mask

# --- Video Stabilizer Class ---
class VideoStabilizer:
    """
    A class to stabilize video frames based on background features using ORB.
    """
    def __init__(self, first_frame_rgb, orb_nfeatures=2000, bf_norm_type=cv2.NORM_HAMMING, match_ratio_test=0.75):
        """
        Initializes the stabilizer with the first frame and settings.

        Args:
            first_frame_rgb: The first frame of the video in RGB format (NumPy array).
            orb_nfeatures: Max number of features to detect with ORB.
            bf_norm_type: Norm type for the BruteForce Matcher (use cv2.NORM_HAMMING for ORB).
            match_ratio_test: Ratio test threshold for filtering good matches.
        """
        print("Initializing VideoStabilizer...")
        if first_frame_rgb is None:
            raise ValueError("Cannot initialize stabilizer with a None reference frame.")

        # --- Configuration ---
        self.match_ratio_test = match_ratio_test
        self.min_good_matches = 10 # Minimum required matches for reliable transform
        self.ransac_reproj_threshold = 9.0 # RANSAC reprojection threshold

        # --- Feature Detection & Matching Setup ---
        # Initialize these once per instance.
        self.orb = cv2.ORB_create(nfeatures=orb_nfeatures)
        # Use crossCheck=False to enable ratio test with knnMatch
        self.bf = cv2.BFMatcher(bf_norm_type, crossCheck=False)

        # --- Reference Frame Data ---
        self.ref_frame_shape = first_frame_rgb.shape[:2] # (height, width)
        self.ref_gray = self._preprocess_frame(first_frame_rgb)
        # Detect features in the reference frame (no mask applied here)
        self.ref_kps, self.ref_des = self.orb.detectAndCompute(self.ref_gray, None)

        self.initialized = False
        # Check if descriptors were actually found
        if self.ref_des is not None and len(self.ref_kps) >= self.min_good_matches:
            self.initialized = True
            print(f"Stabilizer initialized successfully with {len(self.ref_kps)} reference features.")
        else:
            num_ref_features = len(self.ref_kps) if self.ref_kps is not None else 0
            print(f"Warning: Insufficient features ({num_ref_features}) detected in the reference frame. Need at least {self.min_good_matches}. Stabilization may fail.")


    def _preprocess_frame(self, frame_rgb):
        """Converts frame to grayscale - not strictly necessary, but
        gives better performance for ORB."""
        # Convert RGB to grayscale
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    def stabilize_frame(self, current_frame_rgb, pose_result: PoseLandmarkerResult = None):
        """
        Stabilizes the current frame by aligning it to the reference frame.

        Args:
            current_frame_rgb: The current frame in RGB format (NumPy array).
            pose_result: Optional PoseLandmarkerResult for the current frame to enable
                masking of the foreground subject during feature detection.

        Returns:
            A tuple containing:
            - stabilized_frame_rgb: The warped frame aligned to the reference (RGB).
                Returns the original frame if stabilization fails.
            - transformation_matrix: The 2x3 affine transformation matrix found, or None.
        """
        # Check if initialization was successful and reference data exists
        if not self.initialized or self.ref_gray is None or self.ref_kps is None or self.ref_des is None:
            return current_frame_rgb, None

        # Check if current frame shape matches reference frame shape
        if current_frame_rgb.shape[:2] != self.ref_frame_shape:
             print(f"Warning: Current frame shape {current_frame_rgb.shape[:2]} differs from reference {self.ref_frame_shape}. Skipping stabilization.")
             return current_frame_rgb, None

        # Prepare current frame
        current_gray = self._preprocess_frame(current_frame_rgb)
        h, w = current_gray.shape # Get dimensions for warping later

        # Create mask for current frame based on pose (if provided)
        # Pass pose_result directly, the helper function handles None
        mask = create_mask_from_pose((h, w), pose_result)

        # Detect features in the masked current frame
        current_kps, current_des = self.orb.detectAndCompute(current_gray, mask)

        # Check if enough features were detected in the current frame's background
        if current_des is None or len(current_kps) < self.min_good_matches:
            print(f"Frame Warn: Not enough features ({len(current_kps) if current_kps else 0}) detected in current background.")
            return current_frame_rgb, None # Return original frame if too few features

        # Match features between reference and current background
        try:
            # Use knnMatch with k=2 for the ratio test
            matches = self.bf.knnMatch(self.ref_des, current_des, k=2)
        except cv2.error as e:
            # Catch potential errors during matching (e.g., if descriptors are invalid)
            print(f"Error during BFMatcher knnMatch: {e}. Skipping stabilization for this frame.")
            return current_frame_rgb, None

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            # Ensure the pair actually contains two matches before unpacking
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio_test * n.distance:
                    good_matches.append(m)
            # else: Handle cases where less than k=2 matches were found for a descriptor (optional)
            #    pass

        # Check if enough good matches remain after filtering
        if len(good_matches) < self.min_good_matches:
            # print(f"Frame Warn: Only {len(good_matches)} good matches found after ratio test. Need {self.min_good_matches}.")
            return current_frame_rgb, None

        # Estimate Transformation
        # Extract coordinates of good matches in both reference and current frames
        ref_pts = np.float32([self.ref_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        current_pts = np.float32([current_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate the affine transformation matrix using RANSAC for robustness
        # estimateAffinePartial2D estimates rotation + translation + uniform scale
        M, ransac_mask = cv2.estimateAffinePartial2D(current_pts, ref_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold)

        # Check if RANSAC successfully estimated a matrix
        if M is None:
            print("Frame Warn: Could not estimate transformation matrix (RANSAC failed).")
            return current_frame_rgb, None

        # Warp current frame to align with reference frame
        # Apply the transformation M to the original *color* frame
        # Use BORDER_CONSTANT with black padding for areas outside the warped image
        stabilized_frame_rgb = cv2.warpAffine(current_frame_rgb, M, (w, h), # (w, h) is the output size
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)) # Black border for RGB

        return stabilized_frame_rgb, M

# --- Main execution block for standalone testing ---
"""
This block allows the class to be run as a standalone script for testing purposes.
It can be used to process a video file and display the original vs stabilized frames side-by-side.
When called from the command line, optional args can be flagged in as:
-f, -n, -d
-f: Path to the input video file.
-n: Number of frames to process for testing (default is 100).
-d: Display original vs stabilized video side-by-side (including flag makes True).
-r: Rotate the video 90 degrees clockwise before processing.
-s: Scale the video by a factor before processing (default is 1.0).
"""
if __name__ == "__main__":
    print("Running VideoStabilizer in standalone test mode...")

    parser = argparse.ArgumentParser(description="Test Video Stabilizer Class")
    parser.add_argument("-f", "--filename", type=str,default=os.path.join("..","..","20250408_fingerTap_decrement.mp4"), help="Path to the input video file.")
    parser.add_argument("-n", "--num_frames", type=int, default=100, help="Number of frames to process for testing.")
    parser.add_argument("-d", "--display", action='store_true', help="Display original vs stabilized video side-by-side.")
    parser.add_argument("-r","--rotate", action='store_true', help="Rotate the video 90 degrees clockwise before processing.")
    parser.add_argument("-s","--scale", type=float, default=1.0, help="Scale the video by a factor before processing.")
    args = parser.parse_args()

    video_path = args.filename
    max_frames = args.num_frames
    display_video = args.display
    rotate_video = args.rotate
    scale_factor = args.scale

    # --- Set up model ---
    import os
    POSE_MODEL_PATH = os.path.join("..","..","pose_landmarker.task")
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
    )

    # --- Input Validation ---
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        exit(1) 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not rotate_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    print(f"Input video: {width}x{height} @ {fps:.2f} FPS")

    # Read the first frame for initialization
    ret, first_frame_bgr = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        exit(1)
    # rotate and scale if requested
    if rotate_video:
        first_frame_bgr = cv2.rotate(first_frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if scale_factor != 1.0:
        new_size = (int(width * scale_factor), int(height * scale_factor))
        first_frame_bgr = cv2.resize(first_frame_bgr, new_size)
    

    # Convert first frame to RGB for the stabilizer
    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)

    # --- Instantiate the Stabilizer ---
    try:
        # Pass the first frame (RGB) to the constructor
        stabilizer = VideoStabilizer(first_frame_rgb)
    except (ValueError, RuntimeError) as e:
        # Catch potential errors during initialization (e.g., no features)
        print(f"Error initializing stabilizer: {e}")
        cap.release()
        exit(1)
    # Check if initialization was actually successful internally
    if not stabilizer.initialized:
        print("Stabilizer failed to initialize properly (likely due to lack of features). Exiting.")
        cap.release()
        exit(1)


    frame_count = 0
    print(f"Processing up to {max_frames} frames for stabilization test...")

    # --- Process Video Frames ---
    with PoseLandmarker.create_from_options(pose_options) as posemarker:
        
        frame_index = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                # Reached end of video or error reading frame
                if frame_count < max_frames:
                    print("Reached end of video.")
                break
            if rotate_video:
                frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            if scale_factor != 1.0:
                new_size = (int(width * scale_factor), int(height * scale_factor))
                frame_bgr = cv2.resize(frame_bgr, new_size)

            # Convert current frame to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # --- Stabilize the frame ---
            # In this standalone test, we compute a pose result, so can pass that in
            # however, this is not strictly necessary - the stabilization can be done with pose data
            # get timestamps and model outputs
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_index * (1000 / fps))
            result = posemarker.detect_for_video(mp_frame, timestamp_ms)
            stabilized_rgb, matrix = stabilizer.stabilize_frame(frame_rgb, pose_result=None)

            frame_index += 1

            # The stabilize_frame method returns the original frame if stabilization fails,
            # so stabilized_rgb will always be a valid image.
            # We can check if 'matrix' is None to know if stabilization succeeded for the frame.
            if matrix is None:
                # print(f"Frame {frame_count + 1}: Stabilization failed.")
                pass

            # --- Display (Optional) ---
            if display_video:
                # Convert the potentially stabilized frame back to BGR for display
                stabilized_bgr = cv2.cvtColor(stabilized_rgb, cv2.COLOR_RGB2BGR)

                # --- Calculate Differences ---
                # Ensure reference gray is available (it should be if stabilizer initialized)
                if stabilizer.ref_gray is not None:
                    # Convert current frames to grayscale for diff calculation
                    current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    stabilized_gray = cv2.cvtColor(stabilized_bgr, cv2.COLOR_BGR2GRAY)

                    # Calculate absolute difference
                    diff_orig_gray = cv2.absdiff(stabilizer.ref_gray, current_gray)
                    diff_stab_gray = cv2.absdiff(stabilizer.ref_gray, stabilized_gray)

                    # Convert diffs to BGR for stacking with color images
                    diff_orig_bgr = cv2.cvtColor(diff_orig_gray, cv2.COLOR_GRAY2BGR)
                    diff_stab_bgr = cv2.cvtColor(diff_stab_gray, cv2.COLOR_GRAY2BGR)
                else:
                    # Fallback if ref_gray isn't available (shouldn't happen if initialized)
                    print("Warning: Reference frame for diff not available.")
                    # Create black images as placeholders for diffs
                    h, w = frame_bgr.shape[:2]
                    diff_orig_bgr = np.zeros((h, w, 3), dtype=np.uint8)
                    diff_stab_bgr = np.zeros((h, w, 3), dtype=np.uint8)


                # --- Prepare Frames for Combined Display ---
                frames_to_stack = [frame_bgr, stabilized_bgr, diff_orig_bgr, diff_stab_bgr]
                labels = ['Original', 'Stabilized', 'Diff Orig', 'Diff Stab']

                # Ensure all frames have the same height for hstack
                target_height = frames_to_stack[0].shape[0]
                consistent_height = all(f.shape[0] == target_height for f in frames_to_stack)

                if not consistent_height:
                    print("Frame height mismatch - resizing for display.")
                    # Find minimum height to avoid upscaling artifacts if possible
                    min_h = min(f.shape[0] for f in frames_to_stack)
                    resized_frames = []
                    for f in frames_to_stack:
                         # Calculate new width maintaining aspect ratio
                         aspect_ratio = f.shape[1] / f.shape[0]
                         new_w = int(min_h * aspect_ratio)
                         resized_f = cv2.resize(f, (new_w, min_h), interpolation=cv2.INTER_AREA)
                         resized_frames.append(resized_f)
                    frames_to_stack = resized_frames

                # Add text labels to each frame
                labeled_frames = []
                for i, f in enumerate(frames_to_stack):
                     labeled_f = f.copy() # Avoid modifying original frames in list
                     cv2.putText(labeled_f, labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                     labeled_frames.append(labeled_f)

                # Stack horizontally
                # Ensure all frames have the same number of channels (should be 3 after conversions)
                try:
                    combined_display = np.hstack(labeled_frames)
                except ValueError as e:
                    print(f"Error stacking frames: {e}")
                    # Print shapes for debugging
                    for i, f in enumerate(labeled_frames):
                        print(f"Frame {i} ('{labels[i]}') shape: {f.shape}")
                    # Fallback: Display only original and stabilized if stacking fails
                    combined_display = np.hstack((labeled_frames[0], labeled_frames[1]))


                # Display the combined image
                cv2.imshow("Original | Stabilized | Diff Orig | Diff Stab", combined_display)

                # Wait for a key press (e.g., 10ms delay). Press 'q' to quit.
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Display quit by user.")
                    break

            frame_count += 1
            # Print progress periodically
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames...")

        # --- Cleanup ---
        cap.release() # Release the video capture object
        if display_video:
            cv2.destroyAllWindows() # Close all OpenCV windows
        print(f"Standalone test finished processing {frame_count} frames.")