# --- At the top of scripts/process_video.py ---
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import argparse
# Import the necessary functions from your new module
from src.analysis_tools.landmark_utils import (
    draw_landmarks_on_image,
    draw_fingertap,
    blur_face_pose,
    extract_landmarks_for_frame,
    extract_points_for_fingertap
)
# Import the stabilizer class
from src.analysis_tools.video_stabilizer import VideoStabilizer

# --- Constants specific to this script's execution ---
OUTPUT_DIR = "output" # Base output directory
VIDEO_DIR = os.path.join(OUTPUT_DIR, "OutputVideos")
CSV_DIR = os.path.join(OUTPUT_DIR, "OutputCSVs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "OutputPlots")
# Create directories if they don't exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# MediaPipe Model Paths (relative to script or absolute)
# Best Practice: Consider making these command-line arguments or config file entries
HAND_MODEL_PATH = 'hand_landmarker.task'
POSE_MODEL_PATH = 'pose_landmarker.task'

# --- Setup Argument Parser (as before) ---
parser = argparse.ArgumentParser(...)
parser.add_argument("-f", "--filename", ...)
# Add arguments for model paths if desired
args = parser.parse_args()
video_path = args.filename
# ... get video_name_tag ...

# --- Setup MediaPipe Options (as before) ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# ... configure hand_options and pose_options ...

# --- Video Capture and Writer Setup (as before) ---
cap = cv2.VideoCapture(video_path)
# ... get fps, width, height ...
# ... setup output_path, fourcc, out (VideoWriter) ...

# --- Initialize Stabilizer ---
stabilizer = None # Initialize later with first frame
# --- Initialize Data Lists ---
all_landmarks_data = []
timestamps_ms = []
frame_indexes = []
right_index_tip_y_coords = []
right_thumb_tip_y_coords = []

# --- Main Loop ---
frame_index = 0
with HandLandmarker.create_from_options(hand_options) as handmarker, \
     PoseLandmarker.create_from_options(pose_options) as posemarker:

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret: break

        # --- Handle Rotation (if needed) ---
        # rotated_frame = ...
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Assuming rotation handled
        timestamp_ms = int(frame_index * (1000 / fps))

        # --- Initialize or Use Stabilizer ---
        if frame_index == 0:
            try:
                stabilizer = VideoStabilizer(frame_rgb.copy())
                if not stabilizer.initialized: stabilizer = None # Disable if init failed
            except Exception as e:
                print(f"Failed to initialize stabilizer: {e}")
                stabilizer = None
            stabilized_rgb = frame_rgb # First frame is reference
            pose_result_for_masking = posemarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms) # Need pose for first frame too
        elif stabilizer:
            # Get pose on original frame for masking
            pose_result_for_masking = posemarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms)
            stabilized_rgb, _ = stabilizer.stabilize_frame(frame_rgb, pose_result=pose_result_for_masking)
        else:
            stabilized_rgb = frame_rgb # No stabilization if failed/disabled
            pose_result_for_masking = posemarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms) # Still need pose

        # --- Run Detection on Stabilized Frame ---
        mp_image_stabilized = mp.Image(image_format=mp.ImageFormat.SRGB, data=stabilized_rgb)
        hand_result = handmarker.detect_for_video(mp_image_stabilized, timestamp_ms)
        pose_result_stabilized = posemarker.detect_for_video(mp_image_stabilized, timestamp_ms) # Use stabilized pose for analysis/drawing

        # --- Process Stabilized Frame using Utility Functions ---
        # 1. De-identify (using stabilized pose result)
        processed_frame = blur_face_pose(stabilized_rgb, pose_result_stabilized)

        # 2. Draw Fingertap landmarks (using stabilized hand result)
        # Can override defaults: draw_fingertap(processed_frame, hand_result, index_color=(0,255,0))
        processed_frame = draw_fingertap(processed_frame, hand_result)

        # 3. Optionally draw other landmarks (e.g., non-hand pose)
        # processed_frame = draw_landmarks_on_image(processed_frame, pose_result_stabilized, exclude_landmarks=['ALL_HANDS_POSE'])


        # --- Extract Data using Utility Functions ---
        timestamps_ms.append(timestamp_ms)
        frame_indexes.append(frame_index)

        # 4. Extract all landmarks (from stabilized results)
        frame_landmarks = extract_landmarks_for_frame(frame_index, timestamp_ms, hand_result, pose_result_stabilized)
        all_landmarks_data.extend(frame_landmarks)

        # 5. Extract fingertap points (from stabilized hand result)
        index_y, thumb_y = extract_points_for_fingertap(hand_result)
        right_index_tip_y_coords.append(index_y)
        right_thumb_tip_y_coords.append(thumb_y)

        # --- Write Output Video Frame ---
        out_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        out.write(out_frame_bgr)

        frame_index += 1
        # Print progress...

# --- Cleanup and Save Data (as before) ---
cap.release()
out.release()
cv2.destroyAllWindows()

# Save CSV
if all_landmarks_data:
    landmarks_df = pd.DataFrame(all_landmarks_data)
    output_csv_path = os.path.join(CSV_DIR, f'landmarks_{video_name_tag}.csv')
    landmarks_df.to_csv(output_csv_path, index=False)
    print(f"Landmark data saved to: {output_csv_path}")

# Plotting (can be moved to a separate script later)
# ... use frame_indexes, right_index_tip_y_coords, right_thumb_tip_y_coords ...
# ... save plot to PLOT_DIR ...

print("Script finished.")