import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
"""
Utility functions for MediaPipe landmark processing and visualization.
Public methods include:
    draw_landmarks_on_image: Draws landmarks on an image with customizable styles.
    draw_fingertap: Highlights index/thumb tips for right hand in an image.
    blur_face_pose: Blackouts a square region around the face detected by PoseLandmarker.
    extract_landmarks_for_frame: Extracts landmark data for a single frame into a list of dicts.
    extract_points_for_fingertap: Extracts normalized Y coordinates for right index/thumb tips for fingertap analysis.
Private methods include:
    _resolve_landmark_indices: Resolves landmark specifiers into indices.
"""

# Try importing specific result types, handle if mediapipe structure changes
try:
    from mediapipe.tasks.python.vision import HandLandmarkerResult, PoseLandmarkerResult
except ImportError:
    print("Warning: Could not import specific LandmarkerResult types. Type hints may be affected.")
    # Define dummy types if needed, or rely on general type hints like 'object'
    HandLandmarkerResult = object
    PoseLandmarkerResult = object

# === MediaPipe Landmark Definitions (Self-contained within module) ===
# Check if solutions are available before accessing landmarks
if hasattr(solutions, 'hands') and hasattr(solutions.hands, 'HandLandmark'):
    HandLandmark = solutions.hands.HandLandmark
    HAND_CONNECTIONS = solutions.hands.HAND_CONNECTIONS
    HAND_LANDMARK_NAMES = {lm.value: lm.name for lm in HandLandmark}
    # Specific Hand Landmarks used
    INDEX_FINGER_TIP_INDEX = HandLandmark.INDEX_FINGER_TIP.value
    THUMB_TIP_INDEX = HandLandmark.THUMB_TIP.value
else:
    print("Warning: Hand landmark definitions not found in mediapipe.solutions.")
    HandLandmark = None
    HAND_CONNECTIONS = None
    HAND_LANDMARK_NAMES = {}
    INDEX_FINGER_TIP_INDEX = -1 # Placeholder
    THUMB_TIP_INDEX = -1      # Placeholder


if hasattr(solutions, 'pose') and hasattr(solutions.pose, 'PoseLandmark'):
    PoseLandmark = solutions.pose.PoseLandmark
    POSE_CONNECTIONS = solutions.pose.POSE_CONNECTIONS
    POSE_LANDMARK_NAMES = {lm.value: lm.name for lm in PoseLandmark}
    # Specific Pose Landmarks used
    FACE_LANDMARK_INDICES = [
        PoseLandmark.NOSE.value, PoseLandmark.LEFT_EYE_INNER.value, PoseLandmark.LEFT_EYE.value,
        PoseLandmark.LEFT_EYE_OUTER.value, PoseLandmark.RIGHT_EYE_INNER.value, PoseLandmark.RIGHT_EYE.value,
        PoseLandmark.RIGHT_EYE_OUTER.value, PoseLandmark.MOUTH_LEFT.value, PoseLandmark.MOUTH_RIGHT.value
    ]
    LEFT_EYE_OUTER_INDEX = PoseLandmark.LEFT_EYE_OUTER.value
    RIGHT_EYE_OUTER_INDEX = PoseLandmark.RIGHT_EYE_OUTER.value
else:
    print("Warning: Pose landmark definitions not found in mediapipe.solutions.")
    PoseLandmark = None
    POSE_CONNECTIONS = None
    POSE_LANDMARK_NAMES = {}
    FACE_LANDMARK_INDICES = []
    LEFT_EYE_OUTER_INDEX = -1 # Placeholder
    RIGHT_EYE_OUTER_INDEX = -1 # Placeholder

# === Drawing Styles (Defaults can be overridden via function args) ===
DEFAULT_DRAWING_SPEC = solutions.drawing_utils.DrawingSpec
INVISIBLE_SPEC = DEFAULT_DRAWING_SPEC(color=(0,0,0,0), thickness=0, circle_radius=0)

# Default colors (BGR for OpenCV)
DEFAULT_INDEX_COLOR = (0, 0, 255)   # Red
DEFAULT_THUMB_COLOR = (255, 0, 0)   # Blue
DEFAULT_LM_COLOR = (255, 255, 255) # White
DEFAULT_HAND_CONN_COLOR = (200, 200, 200) # Light Gray
DEFAULT_POSE_CONN_COLOR = (230, 230, 230) # Lighter Gray

# Default sizes
DEFAULT_LM_RADIUS = 3
DEFAULT_CONN_THICKNESS = 1
DEFAULT_HIGHLIGHT_RADIUS_MULTIPLIER = 2


# === Helper Function (Private to this module) ===
def _resolve_landmark_indices(specifier_list, landmark_enum):
    """
    Resolves a list of landmark specifiers (names, indices, keywords)
    into a set of integer indices based on the provided landmark enum.
    (Internal helper function)
    """
    resolved_indices = set()
    if specifier_list is None or landmark_enum is None:
        return resolved_indices # Return empty set if no specifiers or enum missing

    # Create a name-to-index mapping from the enum
    name_to_index = {lm.name: lm.value for lm in landmark_enum}

    for specifier in specifier_list:
        if isinstance(specifier, int):
            # Check if index is valid for the enum
            if 0 <= specifier < len(landmark_enum):
                 resolved_indices.add(specifier)
        elif isinstance(specifier, str):
            upper_specifier = specifier.upper()
            if upper_specifier in name_to_index:
                resolved_indices.add(name_to_index[upper_specifier])
            # Handle Keywords
            elif upper_specifier == 'ALL_HANDS_POSE' and landmark_enum == PoseLandmark:
                for lm in landmark_enum:
                    if any(sub in lm.name for sub in ['HAND', 'WRIST', 'PINKY', 'INDEX', 'THUMB']):
                        resolved_indices.add(lm.value)
            # Add other keywords here if needed

    return resolved_indices

# === Public Functions ===
def draw_landmarks_on_image(
    rgb_image,
    detection_result, # Should be HandLandmarkerResult or PoseLandmarkerResult
    include_landmarks=None,
    exclude_landmarks=None,
    landmark_color=DEFAULT_LM_COLOR, # Allow overriding default color
    connection_color=None, # If None, will use type-specific default
    circle_radius=DEFAULT_LM_RADIUS,
    connection_thickness=DEFAULT_CONN_THICKNESS
    ):
    """
    Draws landmarks from HandLandmarkerResult or PoseLandmarkerResult on an image,
    allowing filtering and customized drawing styles via arguments.
    """
    annotated_image = np.copy(rgb_image)
    landmarks_list = []
    connections = None
    landmark_enum = None
    num_total_landmarks = 0
    base_connection_color = connection_color # Use provided color if available

    # 1. Determine result type and set appropriate variables
    if HandLandmark and isinstance(detection_result, HandLandmarkerResult) and detection_result.hand_landmarks:
        landmarks_list = detection_result.hand_landmarks
        connections = HAND_CONNECTIONS
        landmark_enum = HandLandmark
        num_total_landmarks = len(landmark_enum)
        if base_connection_color is None: # Use default if not overridden
             base_connection_color = DEFAULT_HAND_CONN_COLOR

    elif PoseLandmark and isinstance(detection_result, PoseLandmarkerResult) and detection_result.pose_landmarks:
        landmarks_list = detection_result.pose_landmarks
        connections = POSE_CONNECTIONS
        landmark_enum = PoseLandmark
        num_total_landmarks = len(landmark_enum)
        if base_connection_color is None: # Use default if not overridden
            base_connection_color = DEFAULT_POSE_CONN_COLOR

    else:
        # print("No landmarks found or result type not recognized/supported.")
        return annotated_image # No landmarks or unknown/unsupported type

    if not landmarks_list or landmark_enum is None:
        return annotated_image

    # 2. Resolve included and excluded landmark indices
    indices_to_include = _resolve_landmark_indices(include_landmarks, landmark_enum)
    indices_to_exclude = _resolve_landmark_indices(exclude_landmarks, landmark_enum)

    # 3. Determine the final set of indices to draw
    if include_landmarks is not None:
        final_indices_to_draw = indices_to_include
    else:
        final_indices_to_draw = set(range(num_total_landmarks))
    final_indices_to_draw.difference_update(indices_to_exclude)

    # 4. Create the custom landmark drawing specification
    custom_landmark_style = {}
    default_spec = DEFAULT_DRAWING_SPEC(color=landmark_color, thickness=-1, circle_radius=circle_radius) # Base spec using args
    for idx in range(num_total_landmarks):
        if idx in final_indices_to_draw:
            # Use the default_spec based on function arguments
            custom_landmark_style[idx] = default_spec
        else:
            custom_landmark_style[idx] = INVISIBLE_SPEC # Make non-drawn landmarks invisible

    # 5. Create the custom connection style
    custom_connection_style = DEFAULT_DRAWING_SPEC(
        color=base_connection_color,
        thickness=connection_thickness
        )

    # 6. Loop through detected instances and draw
    for landmarks in landmarks_list:
        # Convert landmarks to the protobuf format expected by drawing_utils
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
            connection_drawing_spec=custom_connection_style
        )

    return annotated_image


def draw_fingertap(
    rgb_image,
    detection_result: HandLandmarkerResult,
    index_color=DEFAULT_INDEX_COLOR,
    thumb_color=DEFAULT_THUMB_COLOR,
    default_color=DEFAULT_LM_COLOR,
    connection_color=DEFAULT_HAND_CONN_COLOR,
    default_radius=DEFAULT_LM_RADIUS,
    highlight_radius_multiplier=DEFAULT_HIGHLIGHT_RADIUS_MULTIPLIER,
    connection_thickness=DEFAULT_CONN_THICKNESS
    ):
    """
    Draws hand landmarks, highlighting the index and thumb tips of the right hand.

    Args:
        rgb_image: The input image in RGB format (NumPy array).
        detection_result: The result object from MediaPipe HandLandmarker.
        index_color: Color for the index finger tip (BGR).
        thumb_color: Color for the thumb tip (BGR).
        default_color: Color for other landmarks (BGR).
        connection_color: Color for connections (BGR).
        default_radius: Radius for default landmarks.
        highlight_radius_multiplier: Multiplier for index/thumb tip radius.
        connection_thickness: Thickness for connection lines.

    Returns:
        annotated_image: The image with landmarks drawn (NumPy array).
    """
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    # Check if necessary results exist and landmarks are defined
    if not HandLandmark or not HAND_CONNECTIONS or \
       not detection_result.hand_landmarks or not detection_result.handedness:
        # print("Hand landmarks, handedness, or definitions not available.")
        return annotated_image # Return original image

    # Ensure handedness list matches landmarks list length
    if len(detection_result.handedness) != len(detection_result.hand_landmarks):
        print("Warning: Mismatch between handedness and landmark lists.")
        return annotated_image

    highlight_radius = default_radius * highlight_radius_multiplier

    # Loop through the detected hands
    for handedness_list, hand_landmarks in zip(detection_result.handedness, detection_result.hand_landmarks):
        if not handedness_list: continue # Skip if no handedness info
        hand_label = handedness_list[0].category_name

        landmarks_for_drawing = [] # Store coords for connections

        # Draw Landmarks
        for idx, landmark in enumerate(hand_landmarks):
            cx = int(landmark.x * width)
            cy = int(landmark.y * height)
            color = default_color
            radius = default_radius

            # Highlight right hand index and thumb tips
            if hand_label.lower() == 'right':
                if idx == INDEX_FINGER_TIP_INDEX:
                    color = index_color
                    radius = highlight_radius
                elif idx == THUMB_TIP_INDEX:
                    color = thumb_color
                    radius = highlight_radius

            cv2.circle(annotated_image, (cx, cy), radius, color, -1) # Filled circle
            landmarks_for_drawing.append((cx, cy))

        # Draw Connections
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            if start_idx < len(landmarks_for_drawing) and end_idx < len(landmarks_for_drawing):
                start_lm_coords = landmarks_for_drawing[start_idx]
                end_lm_coords = landmarks_for_drawing[end_idx]
                if start_lm_coords is not None and end_lm_coords is not None:
                    cv2.line(annotated_image, start_lm_coords, end_lm_coords,
                             connection_color, connection_thickness)

    return annotated_image

def blur_face_pose(rgb_image, pose_result: PoseLandmarkerResult, scale_factor=2):
    """
    Blacks out a square region around the face detected by PoseLandmarker.

    Args:
        rgb_image: The input image in RGB format (NumPy array).
        pose_result: The result object from MediaPipe PoseLandmarker.
        scale_factor: Multiplier for eye distance to determine blackout size.

    Returns:
        The image with the face area blacked out (NumPy array).
    """
    # Ensure PoseLandmark definitions were loaded correctly
    if not PoseLandmark or LEFT_EYE_OUTER_INDEX == -1 or RIGHT_EYE_OUTER_INDEX == -1 or not FACE_LANDMARK_INDICES:
        print("Skipping face blackout: PoseLandmark definitions not available.")
        return rgb_image

    annotated_image = np.copy(rgb_image)
    image_height, image_width, _ = annotated_image.shape

    if not pose_result or not pose_result.pose_landmarks:
        return annotated_image # No pose detected

    landmarks = pose_result.pose_landmarks[0]
    num_landmarks = len(landmarks)

    # --- Calculate Centroid ---
    face_landmarks_coords_norm = []
    max_face_index = max(FACE_LANDMARK_INDICES)
    if max_face_index >= num_landmarks: return annotated_image # Index out of bounds

    for idx in FACE_LANDMARK_INDICES:
        landmark = landmarks[idx]
        face_landmarks_coords_norm.append((landmark.x, landmark.y))

    if not face_landmarks_coords_norm: return annotated_image

    center_x_norm = np.mean([coord[0] for coord in face_landmarks_coords_norm])
    center_y_norm = np.mean([coord[1] for coord in face_landmarks_coords_norm])
    center_x_px = int(center_x_norm * image_width)
    center_y_px = int(center_y_norm * image_height)

    # --- Calculate Eye Distance ---
    if LEFT_EYE_OUTER_INDEX >= num_landmarks or RIGHT_EYE_OUTER_INDEX >= num_landmarks:
        return annotated_image # Index out of bounds

    left_eye_outer = landmarks[LEFT_EYE_OUTER_INDEX]
    right_eye_outer = landmarks[RIGHT_EYE_OUTER_INDEX]

    left_eye_outer_px = (left_eye_outer.x * image_width, left_eye_outer.y * image_height)
    right_eye_outer_px = (right_eye_outer.x * image_width, right_eye_outer.y * image_height)

    eye_distance_px = math.dist(left_eye_outer_px, right_eye_outer_px) # Use math.dist

    # --- Calculate Square Size & Apply Blackout ---
    side_length = max(10, int(scale_factor * eye_distance_px)) # Ensure min size
    half_side = side_length // 2
    xmin = max(0, center_x_px - half_side)
    ymin = max(0, center_y_px - half_side)
    xmax = min(image_width, center_x_px + half_side + (side_length % 2)) # Adjust for odd lengths
    ymax = min(image_height, center_y_px + half_side + (side_length % 2)) # Adjust for odd lengths

    if xmin < xmax and ymin < ymax:
        cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 0, 0), -1) # Blackout

    return annotated_image

def extract_landmarks_for_frame(
    frame_index,
    timestamp_ms,
    hand_result: HandLandmarkerResult,
    pose_result: PoseLandmarkerResult
    ):
    """
    Extracts landmark data from MediaPipe results for a single frame into a list of dicts.

    Args:
        frame_index: The index of the current video frame.
        timestamp_ms: The timestamp of the current video frame in milliseconds.
        hand_result: The HandLandmarkerResult object for the frame.
        pose_result: The PoseLandmarkerResult object for the frame.

    Returns:
        A list of dictionaries, where each dictionary represents a single landmark's data.
        Returns an empty list if no landmarks are detected in the frame.
    """
    frame_data = []

    # Process Hand Landmarks
    if HandLandmark and hand_result and hand_result.hand_landmarks:
        if len(hand_result.handedness) == len(hand_result.hand_landmarks):
            for instance_idx, landmarks in enumerate(hand_result.hand_landmarks):
                handedness_label = "Unknown"
                if hand_result.handedness[instance_idx]:
                     handedness_label = hand_result.handedness[instance_idx][0].category_name

                for landmark_idx, landmark in enumerate(landmarks):
                    landmark_name = HAND_LANDMARK_NAMES.get(landmark_idx, f"UNKNOWN_{landmark_idx}")
                    frame_data.append({
                        'frame': frame_index, 'timestamp_ms': timestamp_ms, 'source': 'hand',
                        'instance_id': instance_idx, 'handedness': handedness_label,
                        'landmark_id': landmark_idx, 'landmark_name': landmark_name,
                        'x': landmark.x, 'y': landmark.y, 'z': landmark.z,
                    })

    # Process Pose Landmarks
    if PoseLandmark and pose_result and pose_result.pose_landmarks:
        for instance_idx, landmarks in enumerate(pose_result.pose_landmarks):
            for landmark_idx, landmark in enumerate(landmarks):
                landmark_name = POSE_LANDMARK_NAMES.get(landmark_idx, f"UNKNOWN_{landmark_idx}")
                frame_data.append({
                    'frame': frame_index, 'timestamp_ms': timestamp_ms, 'source': 'pose',
                    'instance_id': instance_idx, 'handedness': 'N/A',
                    'landmark_id': landmark_idx, 'landmark_name': landmark_name,
                    'x': landmark.x, 'y': landmark.y, 'z': landmark.z,
                })

    return frame_data

def extract_points_for_fingertap(result: HandLandmarkerResult):
    """
    Extracts normalized Y coordinates for right index/thumb tips for fingertap analysis.
    Args:
        result: The HandLandmarkerResult object for the frame.
    Returns:
        current_index_y, current_thumb_y: tuples containing the normalized Y coordinates of the index finger tip and thumb tip.
    """
    current_index_y = np.nan
    current_thumb_y = np.nan

    if not HandLandmark or not result or not result.hand_landmarks or not result.handedness:
        return current_index_y, current_thumb_y

    if len(result.handedness) == len(result.hand_landmarks):
        for handedness_list, hand_landmarks in zip(result.handedness, result.hand_landmarks):
            if not handedness_list: continue
            hand_label = handedness_list[0].category_name

            if hand_label.lower() == 'right':
                num_hand_landmarks = len(hand_landmarks)
                if INDEX_FINGER_TIP_INDEX != -1 and INDEX_FINGER_TIP_INDEX < num_hand_landmarks:
                    current_index_y = hand_landmarks[INDEX_FINGER_TIP_INDEX].y
                if THUMB_TIP_INDEX != -1 and THUMB_TIP_INDEX < num_hand_landmarks:
                    current_thumb_y = hand_landmarks[THUMB_TIP_INDEX].y
                # Found right hand, no need to check others if only one right hand expected
                break

    return current_index_y, current_thumb_y