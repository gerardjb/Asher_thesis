# -*- coding: utf-8 -*-
import mediapipe as mp
from mediapipe import solutions
import cv2
import numpy as np
import pandas as pd
import os
import argparse
# ... other imports ...
from src.video_tools.image_quality_utils import (
    calculate_local_laplacian_variance,
    calculate_local_motion,
    calculate_confidence_score
)
# Assuming landmark_utils and VideoStabilizer are also imported
from src.analysis_tools.landmark_utils import extract_landmarks_for_frame # Modified version needed later

# --- Constants ---
OUTPUT_DIR = "output" # Base output directory
VIDEO_DIR = os.path.join(OUTPUT_DIR, "OutputVideos")
CSV_DIR = os.path.join(OUTPUT_DIR, "OutputCSVs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "OutputPlots")
# Create directories if they don't exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# MediaPipe Model Paths (relative to script or absolute)
# TODO: make these command-line arguments or config file entries
HAND_MODEL_PATH = 'hand_landmarker.task'
POSE_MODEL_PATH = 'pose_landmarker.task'

# Size of patch around landmark for quality checks (odd number)
PATCH_SIZE = 21 
# ... other constants ...

# --- Setup MediaPipe Options/Models ---
# Basic options and video running mode
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# hyperparameters for hand detection
num_hands_to_detect = 2

# Set options for hand detection
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO, # Essential for detect_for_video
    num_hands=num_hands_to_detect, 
)

# --- Setup Video Readers/Writers ---
# Use argparse approach for getting filename, name tag
parser = argparse.ArgumentParser(description="Passing in arguments for processing by mediapipe.")
parser.add_argument('-f', '--filename',required=False,type=str,
    help="Path to the input image or video file.")
parser.add_argument("-r","--rotate", action='store_true', help="Rotate the video 90 degrees clockwise before processing.")
parsed_args = parser.parse_args()
rotation_needed = parsed_args.rotate

# Video file.
if parsed_args.filename:
    video_path = parsed_args.filename
else:
    video_path = '20250408_fingerTap_decrement.mp4'
    print(f"Using default video file provided,{video_path}")

# Break the video file path into paths and tags
video_dir = os.path.dirname(video_path)
video_name_tag, video_ext = os.path.splitext(os.path.basename(video_path))

# Open the video file 
cap = cv2.VideoCapture(video_path)

# Get original dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate dimensions *after* rotation - TODO: maybe do an auto-orientation check
if rotation_needed:
    output_width = original_height
    output_height = original_width
else: # No rotation or 180 degrees
    output_width = original_width
    output_height = original_height

# Setup writer for the quality visualization composite video
new_fps = 5 # Reduced FPS for quality video
output_quality_video_path = os.path.join(VIDEO_DIR, f'quality_vis_{video_name_tag}.avi')
fourcc_quality = cv2.VideoWriter_fourcc(*'MJPG')
quality_video_writer = cv2.VideoWriter(output_quality_video_path, fourcc_quality, new_fps, (output_width, output_height))

# --- Initialize Variables ---
timestamps_ms = []
frame_indexes = []
right_index_tip_y_coords = []
right_thumb_tip_y_coords = []
all_landmarks_data = [] # List to hold all landmark data for CSV
previous_gray_frame = None # To store the previous frame for diff calculation

# --- Main Loop ---
frame_index = 0
with HandLandmarker.create_from_options(hand_options) as handmarker:#, \
     #PoseLandmarker.create_from_options(pose_options) as posemarker: # Keep pose for potential future use/masking
    print("Hand Landmarker created successfully.")
    while cap.isOpened():
        ret, frame = cap.read()
        # Check if the end of the video was reached or bad video read
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                print("End of video reached.")
            else:
                print("Failed to read frame due to an issue with the video file.")
            break

        # --- Frame preprocessing ---
        # Rotate the frame if needed
        if rotation_needed:
            rotated_frame = cv2.rotate(frame, ROTATION_CODE)
        else:
            rotated_frame = frame

        # Convert the frame to RGB as MediaPipe expects RGB input, Mediapipe format
        frame_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_index * (1000 / fps)) # TODO: pull from exif if available
        timestamps_ms.append(timestamp_ms)
        frame_indexes.append(frame_index)

        # --- Run Hand Detection on Frame ---
        hand_result = handmarker.detect_for_video(mp_frame, timestamp_ms)

        # --- Calculate Quality Metrics ---
        current_gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        diff_image = None
        laplacian_img_abs_norm = None
        landmark_quality_metrics = {} # Store metrics for this frame's landmarks

        if previous_gray_frame is not None:
            # Possible TODO: export these steps to the corresponding methods
            # Calculate Frame Difference (Motion)
            diff_image = cv2.absdiff(current_gray_frame, previous_gray_frame)
            # Normalize diff image for consistent visualization/patch analysis
            diff_image_norm = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Calculate Laplacian (Blur)
            laplacian_img = cv2.Laplacian(current_gray_frame, cv2.CV_64F)
            laplacian_img_abs = np.absolute(laplacian_img)
            laplacian_img_abs_norm = cv2.normalize(laplacian_img_abs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # --- Calculate Metrics Per Hand Landmark ---
            if hand_result.hand_landmarks:
                 h, w = current_gray_frame.shape
                 for instance_idx, landmarks in enumerate(hand_result.hand_landmarks):
                     for landmark_idx, landmark in enumerate(landmarks):
                         # Get pixel coordinates
                         cx = int(landmark.x * w)
                         cy = int(landmark.y * h)

                         # Calculate local metrics using utility functions
                         blur_score = calculate_local_laplacian_variance(laplacian_img_abs_norm, (cx, cy), PATCH_SIZE)
                         motion_score = calculate_local_motion(diff_image_norm, (cx, cy), PATCH_SIZE)
                         # Calculate confidence (choose a method)
                         conf_score = calculate_confidence_score(blur_score, motion_score, method='inverse_motion_sharpness')

                         # Store metrics keyed by (instance, landmark_id)
                         landmark_quality_metrics[(instance_idx, landmark_idx)] = {
                             'laplacian_variance': blur_score,
                             'mean_motion_diff': motion_score,
                             'quality_score': conf_score
                         }

        # --- Extract Basic Landmark Data ---
        # Using the original function for now, we'll add scores later
        frame_landmarks_basic = extract_landmarks_for_frame(frame_index, timestamp_ms, hand_result, None) # Pass None for pose_result

        # --- Add Quality Metrics to Landmark Data ---
        # Possible TODO: export this to the corresponding method
        for landmark_dict in frame_landmarks_basic:
            # Only add scores for hand landmarks calculated in this frame
            if landmark_dict['source'] == 'hand' and previous_gray_frame is not None:
                key = (landmark_dict['instance_id'], landmark_dict['landmark_id'])
                metrics = landmark_quality_metrics.get(key)
                if metrics:
                    landmark_dict['laplacian_variance'] = metrics['laplacian_variance']
                    landmark_dict['mean_motion_diff'] = metrics['mean_motion_diff']
                    landmark_dict['quality_score'] = metrics['quality_score']
                else: # Should not happen if calculated correctly, but good practice
                    landmark_dict['laplacian_variance'] = np.nan
                    landmark_dict['mean_motion_diff'] = np.nan
                    landmark_dict['quality_score'] = np.nan
            else: # First frame or non-hand landmarks
                landmark_dict['laplacian_variance'] = np.nan
                landmark_dict['mean_motion_diff'] = np.nan
                landmark_dict['quality_score'] = np.nan

        # Append the enriched landmark data
        all_landmarks_data.extend(frame_landmarks_basic)


        # --- Create Composite Visualization Frame ---
        vis_list = []

        # Grayscale with white hand landmarks
        vis_gray = cv2.cvtColor(current_gray_frame, cv2.COLOR_GRAY2BGR)
        if hand_result.hand_landmarks:
            # Draw simple white circles for landmarks
            h_vis, w_vis = vis_gray.shape[:2]
            for landmarks in hand_result.hand_landmarks:
                for landmark in landmarks:
                     cx_vis = int(landmark.x * w_vis)
                     cy_vis = int(landmark.y * h_vis)
                     cv2.circle(vis_gray, (cx_vis, cy_vis), 3, (255, 255, 255), -1)
        vis_list.append(vis_gray)

        # Laplacian (Red Tint)
        if laplacian_img_abs_norm is not None:
            vis_laplacian = cv2.cvtColor(laplacian_img_abs_norm, cv2.COLOR_GRAY2BGR)
            vis_laplacian[:, :, 0] = 0 # Zero out Blue channel
            vis_laplacian[:, :, 1] = 0 # Zero out Green channel
            vis_list.append(vis_laplacian)
        else: # First frame
            vis_list.append(np.zeros_like(vis_gray)) # Black placeholder

        # Motion Diff (Green Tint)
        if diff_image_norm is not None:
            vis_motion = cv2.cvtColor(diff_image_norm, cv2.COLOR_GRAY2BGR)
            vis_motion[:, :, 0] = 0 # Zero out Blue channel
            vis_motion[:, :, 2] = 0 # Zero out Red channel
            vis_list.append(vis_motion)
        else: # First frame
            vis_list.append(np.zeros_like(vis_gray)) # Black placeholder

        # Stack horizontally and write to quality video
        if len(vis_list) == 3: # Ensure all components are present
             composite_frame = np.hstack(vis_list)
             quality_video_writer.write(composite_frame)

        # --- Update previous frame ---
        previous_gray_frame = current_gray_frame.copy()

        # ... rest of the loop (write main video, increment frame_index) ...

# --- Cleanup ---
cap.release()
quality_video_writer.release() # Release the new writer

# After the loop, output df to csv
if all_landmarks_data:
    print(f"Collected data for {len(all_landmarks_data)} landmarks across all frames.")
    # Create DataFrame
    landmarks_df = pd.DataFrame(all_landmarks_data)

    # Define output CSV file path
    output_csv_path = os.path.join(CSV_DIR, f'landmarks_{video_name_tag}.csv')

    # Save to CSV
    try:
        landmarks_df.to_csv(output_csv_path, index=False)
        print(f"Landmark data saved successfully to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving landmark data to CSV: {e}")
else:
    print("No landmark data was collected to save.")