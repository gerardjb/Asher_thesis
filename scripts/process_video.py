# -*- coding: utf-8 -*-
import mediapipe as mp
from mediapipe import solutions
import cv2
import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
# project imports
from video_tools.image_quality_utils import (
    calculate_local_laplacian_variance,
    calculate_local_motion,
    calculate_confidence_score
)
# Assuming landmark_utils and VideoStabilizer are also imported
from analysis_tools.landmark_utils import (
    extract_landmarks_for_frame,
    INDEX_FINGER_TIP_INDEX, # <-- Import this
    THUMB_TIP_INDEX 
    )

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
diff_image_norm = None # To store the normalized diff image

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
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
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
        # TODO: export this to the corresponding method
        # Start with the grayscale frame converted to BGR
        # This sets the base intensity for all channels
        composite_frame = cv2.cvtColor(current_gray_frame, cv2.COLOR_GRAY2BGR)
        composite_frame[:,:,0] = current_gray_frame
        composite_frame[:,:,1] = laplacian_img_abs_norm.astype(np.uint8) if laplacian_img_abs_norm is not None else current_gray_frame
        composite_frame[:,:,2] = diff_image_norm.astype(np.uint8) if diff_image_norm is not None else current_gray_frame
        # Define scaling factors for the added color channels (adjust these to taste)
        # Values > 1 will make the effect stronger but saturate faster.
        # Values < 1 will make the effect subtler.
        # laplacian_weight = 1  # How much Laplacian contributes to Red
        # motion_weight = 0.6     # How much Motion contributes to Green
        """
        # Add Laplacian contribution to the Red channel (Channel 2 in BGR)
        if laplacian_img_abs_norm is not None:
            # Convert channels to float32 for addition to avoid premature uint8 clipping
            base_r_float = composite_frame[:, :, 2].astype(np.float32)
            lap_float = laplacian_img_abs_norm.astype(np.float32)

            # Add scaled laplacian value
            new_r_float = base_r_float + lap_float * laplacian_weight

            # Clip the result to 0-255 and convert back to uint8
            composite_frame[:, :, 2] = np.clip(new_r_float, 0, 255).astype(np.uint8)


        # Add Motion contribution to the Green channel (Channel 1 in BGR)
        if diff_image_norm is not None:
            # Convert channels to float32
            base_g_float = composite_frame[:, :, 1].astype(np.float32)
            mot_float = diff_image_norm.astype(np.float32)

            # Add scaled motion value
            new_g_float = base_g_float + mot_float * motion_weight

            # Clip the result to 0-255 and convert back to uint8
            composite_frame[:, :, 1] = np.clip(new_g_float, 0, 255).astype(np.uint8)
            # Alternatively, use cv2.add:
            # mot_scaled_u8 = cv2.convertScaleAbs(diff_image_norm, alpha=motion_weight)
            # composite_frame[:, :, 1] = cv2.add(composite_frame[:, :, 1], mot_scaled_u8)
        """

        # Draw ONLY right index and thumb tips as white circles
        if hand_result.hand_landmarks and hand_result.handedness:
            h_vis, w_vis = composite_frame.shape[:2]
            # Ensure handedness list matches landmarks list length for safety
            if len(hand_result.handedness) == len(hand_result.hand_landmarks):
                # Iterate through detected hands and their handedness
                for handedness_list, landmarks in zip(hand_result.handedness, hand_result.hand_landmarks):
                    if not handedness_list: continue # Skip if no handedness info

                    hand_label = handedness_list[0].category_name

                    # Check if it's the right hand
                    if hand_label.lower() == 'right':
                        # Iterate through landmarks of the right hand
                        for landmark_idx, landmark in enumerate(landmarks):
                            # Check if it's the index or thumb tip
                            if landmark_idx == INDEX_FINGER_TIP_INDEX or landmark_idx == THUMB_TIP_INDEX:
                                # Calculate pixel coordinates
                                cx_vis = int(landmark.x * w_vis)
                                cy_vis = int(landmark.y * h_vis)
                                # Ensure coordinates are within bounds before drawing
                                if 0 <= cx_vis < w_vis and 0 <= cy_vis < h_vis:
                                    cv2.circle(composite_frame, (cx_vis, cy_vis), 3, (255, 255, 255), -1) #

        # Write the single composite frame to the quality video writer
        quality_video_writer.write(composite_frame)

        # --- Update previous frame ---
        previous_gray_frame = current_gray_frame.copy()

        frame_index += 1
        if frame_index % 50 == 0:
            print(f"Processed {frame_index} frames.")

# --- Cleanup ---
cap.release()
quality_video_writer.release() # Release the new writer

# --- After the loop, output df to csv ---
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

# --- Plotting ---
print("Video processing finished. Generating plots...")

# Check if DataFrame exists and has data
if 'landmarks_df' in locals() and not landmarks_df.empty:

    # --- Plot 1: Fingertip Y-Position vs. Time ---

    # Filter data for right hand index and thumb tips
    right_hand_df = landmarks_df[
        (landmarks_df['source'] == 'hand') &
        (landmarks_df['handedness'] == 'Right')
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Check if specific landmark indices are defined and valid
    if INDEX_FINGER_TIP_INDEX is not None and INDEX_FINGER_TIP_INDEX >= 0:
        index_tip_df = right_hand_df[right_hand_df['landmark_id'] == INDEX_FINGER_TIP_INDEX]
    else:
        index_tip_df = pd.DataFrame() # Empty DataFrame if index not defined
        print("Warning: INDEX_FINGER_TIP_INDEX not defined or invalid.")

    if THUMB_TIP_INDEX is not None and THUMB_TIP_INDEX >= 0:
        thumb_tip_df = right_hand_df[right_hand_df['landmark_id'] == THUMB_TIP_INDEX]
    else:
        thumb_tip_df = pd.DataFrame() # Empty DataFrame if index not defined
        print("Warning: THUMB_TIP_INDEX not defined or invalid.")

    # Proceed only if we have data for at least one landmark
    if not index_tip_df.empty or not thumb_tip_df.empty:
        fig1, ax1 = plt.subplots(figsize=(12, 5)) # Adjust figure size as needed

        # Plot Index Tip Y-position if data exists
        if not index_tip_df.empty:
            ax1.plot(index_tip_df['frame'], index_tip_df['y'],
                     label='Right Index Tip Y', color='red', marker='.',
                     linestyle='-', markersize=4)

        # Plot Thumb Tip Y-position if data exists
        if not thumb_tip_df.empty:
            ax1.plot(thumb_tip_df['frame'], thumb_tip_df['y'],
                     label='Right Thumb Tip Y', color='blue', marker='.',
                     linestyle='-', markersize=4)

        # Customize the plot
        ax1.set_xlabel("Frame Number", fontsize=12)
        ax1.set_ylabel("Normalized Y Position", fontsize=12)
        ax1.set_title(f"Right Fingertip Y Position Over Time ({video_name_tag})", fontsize=14)
        ax1.legend(fontsize=10) # Show the legend
        ax1.grid(True, linestyle='--', alpha=0.6) # Add grid lines
        ax1.invert_yaxis() # Invert Y-axis so 0 is at the top

        # Save the plot
        output_plot_pos_filename = os.path.join(PLOT_DIR, f'fingertip_position_{video_name_tag}.png')
        try:
            plt.savefig(output_plot_pos_filename, format='png', dpi=150, bbox_inches='tight')
            print(f"Position plot saved successfully as: {output_plot_pos_filename}")
        except Exception as e:
            print(f"Error saving position plot: {e}")
        # plt.show() # Uncomment to display interactively
        plt.close(fig1) # Close the figure to free memory

    else:
        print("Skipping position plot: No data found for right index or thumb tip.")


    # --- Plot 2: Quality Metrics vs. Time ---

    # Check if quality columns exist and we have data for landmarks
    quality_cols = ['laplacian_variance', 'mean_motion_diff', 'quality_score']
    if all(col in landmarks_df.columns for col in quality_cols) and \
       (not index_tip_df.empty or not thumb_tip_df.empty):

        # Create a figure with 3 subplots, sharing the x-axis
        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True) # 3 rows, 1 column

        # --- Subplot 1: Laplacian Variance ---
        if not index_tip_df.empty:
            axes2[0].plot(index_tip_df['frame'], index_tip_df['laplacian_variance'],
                          label='Index Tip', color='red', marker='.', linestyle='-', markersize=3, alpha=0.7)
        if not thumb_tip_df.empty:
            axes2[0].plot(thumb_tip_df['frame'], thumb_tip_df['laplacian_variance'],
                          label='Thumb Tip', color='blue', marker='.', linestyle='-', markersize=3, alpha=0.7)
        axes2[0].set_ylabel("Laplacian Var\n(Sharpness)", fontsize=10)
        axes2[0].set_title(f"Landmark Quality Metrics Over Time ({video_name_tag})", fontsize=14)
        axes2[0].legend(fontsize=9)
        axes2[0].grid(True, linestyle='--', alpha=0.6)

        # --- Subplot 2: Mean Motion Difference ---
        if not index_tip_df.empty:
            axes2[1].plot(index_tip_df['frame'], index_tip_df['mean_motion_diff'],
                          label='Index Tip', color='red', marker='.', linestyle='-', markersize=3, alpha=0.7)
        if not thumb_tip_df.empty:
            axes2[1].plot(thumb_tip_df['frame'], thumb_tip_df['mean_motion_diff'],
                          label='Thumb Tip', color='blue', marker='.', linestyle='-', markersize=3, alpha=0.7)
        axes2[1].set_ylabel("Mean Motion Diff\n(Motion)", fontsize=10)
        # axes2[1].legend(fontsize=9) # Legend might be redundant if colors are consistent
        axes2[1].grid(True, linestyle='--', alpha=0.6)

        # --- Subplot 3: Quality Score ---
        if not index_tip_df.empty:
            axes2[2].plot(index_tip_df['frame'], index_tip_df['quality_score'],
                          label='Index Tip', color='red', marker='.', linestyle='-', markersize=3, alpha=0.7)
        if not thumb_tip_df.empty:
            axes2[2].plot(thumb_tip_df['frame'], thumb_tip_df['quality_score'],
                          label='Thumb Tip', color='blue', marker='.', linestyle='-', markersize=3, alpha=0.7)
        axes2[2].set_ylabel("Quality Score", fontsize=10)
        axes2[2].set_xlabel("Frame Number", fontsize=12)
        # axes2[2].legend(fontsize=9) # Legend might be redundant
        axes2[2].grid(True, linestyle='--', alpha=0.6)
        print(f"Mean quality score for index tip: {index_tip_df['quality_score'].mean()}")
        print(f"Mean quality score for thumb tip: {thumb_tip_df['quality_score'].mean()}")

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect to prevent title overlap if needed

        # Save the plot
        output_plot_quality_filename = os.path.join(PLOT_DIR, f'fingertip_quality_{video_name_tag}.png')
        try:
            plt.savefig(output_plot_quality_filename, format='png', dpi=150, bbox_inches='tight')
            print(f"Quality plot saved successfully as: {output_plot_quality_filename}")
        except Exception as e:
            print(f"Error saving quality plot: {e}")
        # plt.show() # Uncomment to display interactively
        plt.close(fig2) # Close the figure

    else:
        print("Skipping quality plot: Quality columns not found in DataFrame or no landmark data.")

else:
    print("Skipping plotting: Landmark DataFrame not found or is empty.")

print("Script finished.") 