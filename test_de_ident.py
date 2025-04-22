# Welcome to mediapipe's maiden voyage on the good ship della
"""
Required CLI arguments:
- Video file path for processing, passed in with -f or --filename
    - Defaulf video file is '20250414_161004.mp4' if no argument is provided
- Note that the video file should be in the same directory as this script or provide the full path
- The script will process the video and save the output in the 'OutputVideos' directory
Optional CLI arguments:
- 
"""

## First, we'll go through standard library imports

import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python # This was part of the recommended header, but I don't actually see it use in any of the demos
from mediapipe.tasks.python import vision # ditto
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import HandLandmarkerResult, PoseLandmarkerResult
import cv2
import numpy as np
import math
from os import makedirs
import os
import argparse
import matplotlib.pyplot as plt

# Constants
## Define drawing constants (it looks like the PRESENCE and VISIBILITY fields don't actually do anything)
HAND_PRESENCE_THRESHOLD = 0.5 # Min visibility/presence score to draw a landmark
HAND_VISIBILITY_THRESHOLD = 0.5 # Min visibility to draw connection lines

# Define colors (BGR format for OpenCV)
RIGHT_INDEX_TIP_COLOR = (0, 0, 255)   # Red
RIGHT_THUMB_TIP_COLOR = (255, 0, 0)   # Blue
DEFAULT_LANDMARK_COLOR = (255, 255, 255) # White
HAND_CONNECTION_COLOR = (200, 200, 200) # Light Gray for connections
# Define landmark indices of interest
HandLandmark = mp.solutions.hands.HandLandmark
INDEX_FINGER_TIP_INDEX = HandLandmark.INDEX_FINGER_TIP.value # Typically 8
THUMB_TIP_INDEX = HandLandmark.THUMB_TIP.value             # Typically 4

HAND_DEFAULT_RADIUS = 8
HAND_HIGH_VIS_COLOR = (0, 255, 0) # Green
HAND_LOW_VIS_COLOR = (0, 0, 255) # Red
HAND_CONNECTION_COLOR = (255, 255, 255) # White
HAND_THICKNESS_LANDMARK = -1 # Fill circle
HAND_THICKNESS_CONNECTION = 1

## Define constants for facial landmarks from pose landmarker to use for de-identification
# Define PoseLandmark constants (ensure mp.solutions.pose is available)
PoseLandmark = mp.solutions.pose.PoseLandmark

# Define the indices of the face landmarks from PoseLandmarker we'll use for the centroid
FACE_LANDMARK_INDICES = [
    PoseLandmark.NOSE.value,
    PoseLandmark.LEFT_EYE_INNER.value,
    PoseLandmark.LEFT_EYE.value,
    PoseLandmark.LEFT_EYE_OUTER.value,
    PoseLandmark.RIGHT_EYE_INNER.value,
    PoseLandmark.RIGHT_EYE.value,
    PoseLandmark.RIGHT_EYE_OUTER.value,
    PoseLandmark.MOUTH_LEFT.value,
    PoseLandmark.MOUTH_RIGHT.value
]

# Define the indices for the outer eye landmarks - used for calculating the blur radius
LEFT_EYE_OUTER_INDEX = PoseLandmark.LEFT_EYE_OUTER.value if PoseLandmark else -1
RIGHT_EYE_OUTER_INDEX = PoseLandmark.RIGHT_EYE_OUTER.value if PoseLandmark else -1

## Define some constants for video processing
### E.g., if the video content needs to be rotated 90 degrees clockwise to be upright
rotation_needed = False # Set to True if the video needs to be rotated
ROTATION_CODE = cv2.ROTATE_90_CLOCKWISE # options as expected from this - CLOCKWISE, COUNTERCLOCKWISE, 180

# Updated version that accepts multiple landmark types and allows inclusion/exclusion of specific landmarks
def draw_landmarks_on_image(
    rgb_image,
    detection_result,
    include_landmarks=None,
    exclude_landmarks=None,
    circle_radius=3,
    connection_thickness=2
    ):
    """
    Draws landmarks from HandLandmarkerResult or PoseLandmarkerResult on an image,
    allowing filtering and handling specific drawing styles.
    """
    annotated_image = np.copy(rgb_image)
    landmarks_list = []
    connections = None
    default_landmark_style = None # Renamed for clarity
    # No longer assume connection style is fetched for both
    base_connection_color = (255, 255, 255) # Default fallback color (white)
    landmark_enum = None
    num_total_landmarks = 0
    is_pose_result = False # Flag to remember type

    # 1. Determine result type and set appropriate variables
    if isinstance(detection_result, HandLandmarkerResult) and detection_result.hand_landmarks:
        landmarks_list = detection_result.hand_landmarks
        connections = solutions.hands.HAND_CONNECTIONS # This is typically a frozenset
        default_landmark_style = solutions.drawing_styles.get_default_hand_landmarks_style()
        default_conn_style_dict = solutions.drawing_styles.get_default_hand_connections_style() # This is a dict

        # Try to get a representative color from the style dictionary
        if default_conn_style_dict and connections: # Check if dict and connections set exist
            # *** CHANGE START ***
            # Convert the frozenset of connections to a list to allow indexing
            connections_list = list(connections)
            if connections_list: # Check if the conversion resulted in a non-empty list
                 # Access the first element (connection tuple) from the list
                 first_connection_tuple = connections_list[0]
                 # Look up the style spec for this specific connection in the dictionary
                 spec_for_first_conn = default_conn_style_dict.get(first_connection_tuple)

                 if spec_for_first_conn and hasattr(spec_for_first_conn, 'color'):
                     # Use the color from the found spec
                     base_connection_color = spec_for_first_conn.color
                 # else: keep the fallback color defined earlier
            # else: Could not get a connection tuple, keep fallback color
            # *** CHANGE END ***

        landmark_enum = solutions.hands.HandLandmark
        num_total_landmarks = len(landmark_enum)
        is_pose_result = False

    elif isinstance(detection_result, PoseLandmarkerResult) and detection_result.pose_landmarks:
        landmarks_list = detection_result.pose_landmarks
        connections = solutions.pose.POSE_CONNECTIONS
        default_landmark_style = solutions.drawing_styles.get_default_pose_landmarks_style()
        # *** CHANGE: Do NOT fetch pose connection style ***
        # Use a hardcoded default color for pose connections
        base_connection_color = (230, 230, 230) # Light Gray - common for pose
        landmark_enum = solutions.pose.PoseLandmark
        num_total_landmarks = len(landmark_enum)
        is_pose_result = True

    else:
        return annotated_image # No landmarks or unknown type

    if not landmarks_list or default_landmark_style is None:
        return annotated_image

    # 2. Resolve included and excluded landmark indices (same as before)
    indices_to_include = _resolve_landmark_indices(include_landmarks, landmark_enum)
    indices_to_exclude = _resolve_landmark_indices(exclude_landmarks, landmark_enum)

    # 3. Determine the final set of indices to draw (same as before)
    if include_landmarks is not None:
        final_indices_to_draw = indices_to_include
    else:
        final_indices_to_draw = set(range(num_total_landmarks))
    final_indices_to_draw.difference_update(indices_to_exclude)

    # 4. Create the custom landmark drawing specification (same logic as before)
    custom_landmark_style = {}
    invisible_spec = solutions.drawing_utils.DrawingSpec(color=(0,0,0,0), thickness=0, circle_radius=0)
    for idx in range(num_total_landmarks):
        if idx in final_indices_to_draw:
            spec = default_landmark_style.get(idx, solutions.drawing_utils.DrawingSpec())
            custom_landmark_style[idx] = solutions.drawing_utils.DrawingSpec(
                color=spec.color,
                thickness=spec.thickness if circle_radius > 0 else 0,
                circle_radius=circle_radius
            )
        else:
            custom_landmark_style[idx] = invisible_spec

    # 5. *** CHANGE: Create the custom connection style ***
    # Use the determined base color (fetched for hands, hardcoded for pose)
    # Use the thickness provided as a parameter to the function
    custom_connection_style = solutions.drawing_utils.DrawingSpec(
        color=base_connection_color,
        thickness=connection_thickness
        )

    # 6. Loop through detected instances and draw (same as before)
    for landmarks in landmarks_list:
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
        ])

        # Draw using the customized styles
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=landmarks_proto,
            connections=connections,
            landmark_drawing_spec=custom_landmark_style,
            connection_drawing_spec=custom_connection_style # Now uses consistently created style
        )

    return annotated_image

# Helper function to resolve landmark specifiers (names/indices/keywords)
def _resolve_landmark_indices(specifier_list, landmark_enum):
    """
    Resolves a list of landmark specifiers (names, indices, keywords)
    into a set of integer indices based on the provided landmark enum.

    Args:
        specifier_list: A list or set containing integers (indices),
                        strings (landmark names like 'WRIST', 'LEFT_EYE'),
                        or special keywords (e.g., 'ALL_HANDS_POSE').
        landmark_enum: The MediaPipe landmark enum (e.g., mp.solutions.hands.HandLandmark
                       or mp.solutions.pose.PoseLandmark).

    Returns:
        A set of integer landmark indices.
    """
    resolved_indices = set()
    if specifier_list is None:
        return resolved_indices # Return empty set if no specifiers

    # Create a name-to-index mapping from the enum
    name_to_index = {lm.name: lm.value for lm in landmark_enum}
    # print(f"Enum {landmark_enum.__name__} Name Map: {name_to_index}") # Debug

    for specifier in specifier_list:
        if isinstance(specifier, int):
            # Check if index is valid for the enum
            if 0 <= specifier < len(landmark_enum):
                 resolved_indices.add(specifier)
            # else: print(f"Warning: Index {specifier} out of range for {landmark_enum.__name__}")
        elif isinstance(specifier, str):
            upper_specifier = specifier.upper()
            if upper_specifier in name_to_index:
                resolved_indices.add(name_to_index[upper_specifier])
            # --- Handle Keywords ---
            elif upper_specifier == 'ALL_HANDS_POSE' and landmark_enum == solutions.pose.PoseLandmark:
                # Add all pose landmarks indices containing 'HAND', 'WRIST', 'PINKY', 'INDEX', 'THUMB'
                for lm in landmark_enum:
                    if any(sub in lm.name for sub in ['HAND', 'WRIST', 'PINKY', 'INDEX', 'THUMB']):
                        resolved_indices.add(lm.value)
            # Add other keywords here if needed (e.g., 'ALL_FACE', 'ALL_LEGS')
            # else: print(f"Warning: Landmark name or keyword '{specifier}' not recognized for {landmark_enum.__name__}")
        # else: print(f"Warning: Invalid specifier type: {type(specifier)}")

    # print(f"Resolved indices for {specifier_list}: {resolved_indices}") # Debug
    return resolved_indices

# This is a customized method for visualizing hand landmarks on an image. Might need to deprecate, but need to check if has anything useful compared to updated draw_landmarks_on_image
def draw_custom_hand_landmarks_cv2(rgb_image, detection_result):
    """Draws hand landmarks using OpenCV as alternative to MediaPipe's builtin solutions"""
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    # Check if necessary results exist
    if not detection_result.hand_landmarks or not detection_result.handedness:
        print("Eith hand landmarks or handedness not detected.")
        return annotated_image # Return original image if no hands/handedness

    # Ensure handedness list matches landmarks list length
    if len(detection_result.handedness) != len(detection_result.hand_landmarks):
        print("Warning: Mismatch between handedness and landmark lists.")
        return annotated_image # Avoid potential errors

    # Loop through the detected hands and their handedness
    # zip pairs corresponding handedness with landmarks
    for handedness_list, hand_landmarks in zip(detection_result.handedness, detection_result.hand_landmarks):

        # Get handedness label (usually the first classification is most confident)
        if not handedness_list:
            print("Warning: Empty handedness list for a detected hand.")
            continue # Skip this hand if handedness info is missing
        hand_label = handedness_list[0].category_name

        landmarks_for_drawing = [] # Store coords for connections

        # Draw Landmarks for the current hand
        for idx, landmark in enumerate(hand_landmarks):
            # --- Calculate Pixel Coordinates ---
            cx = int(landmark.x * width)
            cy = int(landmark.y * height)

            # --- Determine Landmark Color ---
            color = DEFAULT_LANDMARK_COLOR # Default to white
            marker_size = HAND_DEFAULT_RADIUS
            if hand_label.lower() == 'right':
                if idx == INDEX_FINGER_TIP_INDEX:
                    color = RIGHT_INDEX_TIP_COLOR
                    marker_size = HAND_DEFAULT_RADIUS * 2
                elif idx == THUMB_TIP_INDEX:
                    color = RIGHT_THUMB_TIP_COLOR
                    marker_size = HAND_DEFAULT_RADIUS * 2
                # else: color remains DEFAULT_LANDMARK_COLOR for other right hand landmarks

            # --- Optional: At some point may want to play around more with visibility/presence. Useful if it would work, otherwise need alternative solution
            # is_present = getattr(landmark, 'visibility', 1.0) > HAND_PRESENCE_THRESHOLD or \
            #              getattr(landmark, 'presence', 1.0) > HAND_PRESENCE_THRESHOLD
            # if not is_present:
            #     landmarks_for_drawing.append(None) # Keep list length consistent
            #     continue

            # Draw the Circle
            cv2.circle(annotated_image, (cx, cy), marker_size, color, HAND_THICKNESS_LANDMARK)
            landmarks_for_drawing.append((cx, cy)) # Store coords for connections

        # Draw Connections for the current hand
        # Use the connections defined in mp.solutions.hands
        if hasattr(mp.solutions.hands, 'HAND_CONNECTIONS'):
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]

                if start_idx < len(landmarks_for_drawing) and end_idx < len(landmarks_for_drawing):
                    start_lm_coords = landmarks_for_drawing[start_idx]
                    end_lm_coords = landmarks_for_drawing[end_idx]

                    # Draw line if both landmarks were drawn (not skipped)
                    if start_lm_coords is not None and end_lm_coords is not None:
                        cv2.line(annotated_image, start_lm_coords, end_lm_coords, HAND_CONNECTION_COLOR, HAND_THICKNESS_CONNECTION)

    return annotated_image

# Now for doing some automatic de-indentification of the video file
def blur_face(rgb_image,detection_result):
    """"
    Blurs detected faces in the image to de-identify.
    Notes: - The facedetector does not work well when the subject is more than ~2 m away
            - To counter this, I implemented a method based on the face keypoints in poselandmarker
    """
    annotated_image = np.copy(rgb_image)
    image_height, image_width, _ = annotated_image.shape

    if not detection_result.detections:
        print("No face detections found.")
        return annotated_image

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        # MediaPipe provides normalized coordinates relative to the image size.
        # We need to convert them to absolute pixel coordinates.
        origin_x = bbox.origin_x
        origin_y = bbox.origin_y
        width = bbox.width
        height = bbox.height

        # Ensure coordinates are within image bounds
        xmin = max(0, int(origin_x-0.1*width))
        ymin = max(0, int(origin_y-0.1*height))
        xmax = min(image_width, int(origin_x + 1.1*width))
        ymax = min(image_height, int(origin_y + 1.1*height))

        # Extract the face region
        face_region = annotated_image[ymin:ymax, xmin:xmax]
        
        # Apply Gaussian blur to the face region
        blurred_face = cv2.GaussianBlur(face_region, (131, 131), sigmaX=131, sigmaY=131)
        
        # Replace the original face region with the blurred one
        annotated_image[ymin:ymax, xmin:xmax] = blurred_face

    return annotated_image

# --- New Blur Function ---

def blur_face_pose(rgb_image, pose_result: PoseLandmarkerResult):
    """
    Blacks out a square region around the face detected by PoseLandmarker.

    The square's side length is calculated as 1.5 times the distance between
    the outer eye landmarks.
    The square is centered on the centroid (center of mass) of key face landmarks.
    NOTE: This version does NOT check landmark visibility scores.

    Args:
        rgb_image: The input image in RGB format (NumPy array).
        pose_result: The result object from MediaPipe PoseLandmarker.

    Returns:
        The image with the face area blacked out (NumPy array), or the original image
        if no valid face pose is detected or necessary landmarks are out of bounds.
    """
    # Ensure PoseLandmark was loaded correctly
    if not PoseLandmark:
        print("Skipping face blackout: PoseLandmark definitions not available.")
        return rgb_image

    annotated_image = np.copy(rgb_image)
    image_height, image_width, _ = annotated_image.shape

    # Check if any pose landmarks were detected
    if not pose_result.pose_landmarks:
        # No pose detected in this frame, return original image
        return annotated_image

    # --- Process the first detected pose ---
    # Assuming the primary person's pose is the first one in the list
    landmarks = pose_result.pose_landmarks[0]
    num_landmarks = len(landmarks)

    # --- 1. Calculate Centroid of Face Landmarks ---
    face_landmarks_coords_norm = [] # Store normalized coordinates (x, y)
    valid_centroid_landmarks = True

    # Check if all required face landmark indices are within the bounds
    max_face_index = max(FACE_LANDMARK_INDICES) if FACE_LANDMARK_INDICES else -1
    if max_face_index >= num_landmarks:
        # print(f"Warning: Required face landmark index {max_face_index} out of bounds (num_landmarks={num_landmarks}). Cannot calculate centroid.")
        valid_centroid_landmarks = False

    if valid_centroid_landmarks:
        for idx in FACE_LANDMARK_INDICES:
            landmark = landmarks[idx]
            # --- VISIBILITY CHECK REMOVED ---
            # if landmark.visibility < visibility_threshold: # Removed check
            #     valid_centroid_landmarks = False
            #     break
            face_landmarks_coords_norm.append((landmark.x, landmark.y))
    else:
        # Return original if indices were out of bounds initially
        return annotated_image

    # Proceed only if landmarks could be accessed (indices were valid)
    if not face_landmarks_coords_norm: # Should only happen if FACE_LANDMARK_INDICES is empty
        return annotated_image # Return original image

    # Calculate normalized centroid
    center_x_norm = np.mean([coord[0] for coord in face_landmarks_coords_norm])
    center_y_norm = np.mean([coord[1] for coord in face_landmarks_coords_norm])

    # Convert centroid to pixel coordinates
    center_x_px = int(center_x_norm * image_width)
    center_y_px = int(center_y_norm * image_height)

    # --- 2. Calculate Distance Between Outer Eyes ---
    eye_distance_px = 0
    valid_eye_landmarks = True

    # Check if eye landmark indices are valid
    if LEFT_EYE_OUTER_INDEX >= num_landmarks or RIGHT_EYE_OUTER_INDEX >= num_landmarks:
        # print(f"Warning: Eye landmark index out of bounds. Cannot calculate radius.")
        valid_eye_landmarks = False

    if valid_eye_landmarks:
        left_eye_outer = landmarks[LEFT_EYE_OUTER_INDEX]
        right_eye_outer = landmarks[RIGHT_EYE_OUTER_INDEX]

        # --- VISIBILITY CHECK REMOVED ---
        # if left_eye_outer.visibility < visibility_threshold or right_eye_outer.visibility < visibility_threshold: # Removed check
        #      valid_eye_landmarks = False

        # Calculate pixel coordinates for eyes
        left_eye_outer_px = (left_eye_outer.x * image_width, left_eye_outer.y * image_height)
        right_eye_outer_px = (right_eye_outer.x * image_width, right_eye_outer.y * image_height)

        # Calculate Euclidean distance in pixels
        eye_distance_px = math.sqrt(
            (left_eye_outer_px[0] - right_eye_outer_px[0])**2 +
            (left_eye_outer_px[1] - right_eye_outer_px[1])**2
        )
    else:
        # Cannot calculate distance if indices were invalid, return original
        return annotated_image

    # --- 3. Calculate Square Size ---
    # Side length is 1.5 times the eye distance. Ensure minimum size.
    # If eye_distance_px is 0 (e.g., landmarks perfectly overlap or calculation failed silently),
    # side_length will default to 10.
    side_length = max(10, int(1.5 * eye_distance_px))

    # --- 4. Apply Black Square ---
    if side_length > 0:
        # Calculate top-left corner coordinates
        half_side = side_length // 2
        xmin = center_x_px - half_side
        ymin = center_y_px - half_side
        # Calculate bottom-right corner coordinates
        xmax = xmin + side_length
        ymax = ymin + side_length

        # Clip coordinates to image boundaries to avoid errors
        xmin_clipped = max(0, xmin)
        ymin_clipped = max(0, ymin)
        xmax_clipped = min(image_width, xmax)
        ymax_clipped = min(image_height, ymax)

        # Draw a filled black rectangle directly onto the image
        # Ensure coordinates are valid (xmin < xmax, ymin < ymax) after clipping
        if xmin_clipped < xmax_clipped and ymin_clipped < ymax_clipped:
            cv2.rectangle(
                annotated_image,
                (xmin_clipped, ymin_clipped), # Top-left corner
                (xmax_clipped, ymax_clipped), # Bottom-right corner
                (0, 0, 0),    # Color (black in BGR for OpenCV) - Use (0, 0, 0) for RGB if needed elsewhere
                -1            # Thickness = -1 fills the rectangle
            )

    return annotated_image


## And next up, setting up our model and using it for the given dataset

# We need to specify the model path for the pose landmarker.
# Other options include the full and lite models
HAND_MODEL_PATH = 'hand_landmarker.task'
POSE_MODEL_PATH = 'pose_landmarker.task'
FACE_MODEL_PATH = 'detector.tflite'

# And then set up to pass it some video data
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceDetector = mp.tasks.vision.FaceDetector # Note that there is also a "FaceMesh" option - haven't played yet
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Setting the hyperparameters for model detection ---
# --------------------------------------------------
# Initial Detection: Try lowering min_pose_detection_confidence
detection_confidence = 0.5 # Example: Lowered from default 0.5
face_dection_confidence = 0.5 # Setting this lower for de-identification purposes

# For Problem 1 (Tracking/Occlusion): Try adjusting min_tracking_confidence
tracking_confidence = 0.5 # Example: Slightly raised from default 0.5

# Usually keep this at 1 for focusing on the main person
num_hands_to_detect = 2

# Puts the segmentation mask into the landmarker results. Helps troubleshoot
pose_presence_confidence = 0.5

# Set up the PoseLandmarkerOptions with the specified parameters
## Hand
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO, # Essential for detect_for_video
    num_hands=num_hands_to_detect, # Not in hand one, I guess
    min_hand_detection_confidence=detection_confidence,
    # min_tracking_confidence=tracking_confidence,
    # min_pose_presence_confidence=pose_presence_confidence,
    #output_segmentation_masks=True # Default is False, set True only if needed
)
## Pose
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    min_tracking_confidence=tracking_confidence,
    min_pose_presence_confidence=pose_presence_confidence,
)
## Face
face_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=face_dection_confidence
)
# ------------------------------------

# Use argparse approach for getting filename
parser = argparse.ArgumentParser(
      description="Passing in arguments for processing by mediapipe.",
  )
parser.add_argument(
      '-f', '--filename',
      required=False, 
      type=str,
      help="Path to the input image or video file."
  )

parsed_args = parser.parse_args()

# Process this video file.
if parsed_args.filename:
    video_path = parsed_args.filename
else:
    video_path = '20250414_161004.mp4'
    print("Using default video file provided.")

# Break the video file path into components
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

# Prepare the output video file path - note there are a limited number of backend codecs
output_path = os.path.join('OutputVideos','Processed_'+video_name_tag+'.avi')
makedirs('OutputVideos', exist_ok=True)

# Set up the VideoWriter for saving the processed video
new_fps = 5 # Adjust playback for target audiecnce. 5 does ok for like 30 Hz
fourcc = cv2.VideoWriter_fourcc(*'MJPG') # TODO: find out if there's a better supported codec on della
out = cv2.VideoWriter(output_path, fourcc, new_fps, (output_width, output_height))

# Initialize variables for holding plot features
timestamps_ms = []
frame_indexes = []
hand_present_list = []

# Create  landmarker instances as needed for processing the video 
with HandLandmarker.create_from_options(hand_options) as handmarker,\
    PoseLandmarker.create_from_options(pose_options) as posemarker,\
    FaceDetector.create_from_options(face_options) as facedetector:
    # Temp to make sure we have both instances open
    print("Hand Landmarker and  Face detector created successfully.")
    # read the video frame by frame for processing
    frame_index = 0
    while cap.isOpened() and frame_index < 300:  # Limit to 1000 frames for testing
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame or end of video reached.")
            break
        
        # Rotate the frame if needed
        if rotation_needed:
            rotated_frame = cv2.rotate(frame, ROTATION_CODE)
        else:
            rotated_frame = frame

        # Convert the frame to RGB as MediaPipe expects RGB input
        frame_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
        print(frame_index)

        # Process the frame with the Pose Landmarker
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # get timestamps and model outputs
        timestamp_ms = int(frame_index * (1000 / fps))
        timestamps_ms.append(timestamp_ms)
        frame_indexes.append(frame_index)
        result = handmarker.detect_for_video(mp_frame, timestamp_ms)
        pose_result = posemarker.detect_for_video(mp_frame, timestamp_ms)
        face_detection_result = facedetector.detect_for_video(mp_frame, timestamp_ms)


        # --- Processing video frame ---
        # Process the result (e.g., draw landmarks or save results)
        de_id_frame = blur_face_pose(frame_rgb, pose_result)#blur_face(frame_rgb, face_detection_result)# for face blurring
        out_frame = draw_landmarks_on_image(de_id_frame, result)#draw_landmarks_on_image(frame_rgb, result)
        out_frame = draw_landmarks_on_image(
            out_frame, pose_result,
            exclude_landmarks=['ALL_HANDS_POSE'], # Use the special keyword for exclusion
            circle_radius=2
        )
        out_frame = draw_landmarks_on_image(out_frame, pose_result)

        # Write the frame to the output video
        #print(out_frame.dtype)  # Should output: uint8
        #print(out_frame.shape)  # Typically (height, width, 3) for a color image
        bgr_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

        # --- Determine if hand is present ---
        hand_present = len(result.hand_landmarks) > 0
        hand_present_list.append(hand_present)

        frame_index += 1

# Release the video capture
cap.release()
out.release()
cv2.destroyAllWindows()

# --- Plotting ---
print("Video processing finished. Generating plot...")

fig, ax = plt.subplots(figsize=(6, 3)) # Adjust figure size as needed
#fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True) 

# Plot the data
ax.plot(timestamps_ms, hand_present_list, label='Hand presence', color='black', marker='.', linestyle='-', markersize=4)
#ax.plot(timestamps_ms, right_thumb_tip_y_coords, label='Right Thumb Tip', color='red', marker='.', linestyle='-', markersize=4)
#ax[0].plot(frame_indexes, right_index_tip_y_coords, label='Right Index Tip Y', color='red', marker='.', linestyle='-', markersize=4)
#ax[0].plot(frame_indexes, right_thumb_tip_y_coords, label='Right Thumb Tip Y', color='blue', marker='.', linestyle='-', markersize=4)

# Customize the plot
ax.set_xlabel("Timestamp (milliseconds)",fontsize=12) # Or "Frame Number" if using frame_count
ax.set_ylabel("Hand Detected (y=1, n=0)",fontsize=12)
#ax.set_title("Simulated Fingertap Bradykinesia")
#ax.legend(fontsize=12) # Show the legend
ax.grid(True) # Add grid lines
#ax.invert_yaxis() # Invert Y-axis so 0 is at the top, matching image coordinates


# Save the plot - deactivated for now, but keep hand detect viz to allow output assessment
"""output_plot_filename = os.path.join('FigPlots','hand_presence'+video_name_tag+'.jpg')
makedirs('FigPlots', exist_ok=True)
try:
    plt.savefig(output_plot_filename, format='jpg', dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as: {output_plot_filename}")
except Exception as e:
    print(f"Error saving plot: {e}")
"""

# Optionally display the plot interactively
plt.show()
plt.close(fig) # Close the figure after saving/showing if needed

print("Script finished.")