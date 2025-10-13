import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def calculate_head_pose(landmarks, image_size):
    """
    Calculate head pose (pitch, yaw, roll) from facial landmarks.
    
    Args:
        landmarks: List of 5 or 68 facial landmark points from InsightFace
                  Each point is [x, y] in the image
        image_size: Tuple of (height, width) of the image
    
    Returns:
        dict: {'pitch': float, 'yaw': float, 'roll': float} in degrees
              or None if calculation fails
    """
    if not landmarks or len(landmarks) < 5:
        return None
        
    # Convert landmarks to numpy array if it's not already
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # 3D model points (approximate locations of facial features)
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye left corner
        (225.0, 170.0, -135.0),    # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)    # Right mouth corner
    ], dtype=np.float32)
    
    # 2D image points from landmarks
    # For 5-point landmarks, we need to map them to the 6 points we need
    if len(landmarks) == 5:  # 5-point landmarks
        # Assuming order is: left_eye, right_eye, nose, left_mouth, right_mouth
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        
        # Calculate chin as a point below the nose
        chin = nose + (nose - (left_eye + right_eye)/2) * 0.5
        
        image_points = np.array([
            nose,                   # Nose tip
            chin,                   # Chin (approximated)
            left_eye,               # Left eye left corner
            right_eye,              # Right eye right corner
            left_mouth,             # Left mouth corner
            right_mouth             # Right mouth corner
        ], dtype=np.float32)
    else:  # 68-point landmarks
        # Use standard 68-point landmark indices
        image_points = np.array([
            landmarks[30],    # Nose tip (index 30 in 68-point model)
            landmarks[8],     # Chin (index 8)
            landmarks[36],    # Left eye left corner (index 36)
            landmarks[45],    # Right eye right corner (index 45)
            landmarks[48],    # Left mouth corner (index 48)
            landmarks[54]     # Right mouth corner (index 54)
        ], dtype=np.float32)
    
    # Camera internals
    height, width = image_size
    focal_length = width  # Approximate focal length
    center = (width/2, height/2)
    
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float32
    )
    
    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    try:
        # Solve PnP
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        # Convert rotation vector to rotation matrix
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        
        # Convert to Euler angles
        rotation = Rotation.from_matrix(rotation_matrix)
        pitch, yaw, roll = rotation.as_euler('xyz', degrees=True)
        
        return {
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll)
        }
        
    except Exception as e:
        print(f"Error in pose estimation: {str(e)}")
        return None

def is_pose_matched(current_pose, target_pose, threshold=5.0):
    """
    Check if current pose matches target pose within threshold.
    
    Args:
        current_pose: Dict with 'pitch', 'yaw', 'roll'
        target_pose: Dict with 'pitch', 'yaw', 'roll'
        threshold: Allowed difference in degrees
    
    Returns:
        bool: True if poses match within threshold
    """
    if not current_pose or not target_pose:
        return False
        
    for angle in ['pitch', 'yaw', 'roll']:
        if angle not in current_pose or angle not in target_pose:
            return False
        if abs(current_pose[angle] - target_pose[angle]) > threshold:
            return False
    return True

def get_required_poses():
    """Define the required poses for multi-angle enrollment"""
    return [
        {'name': 'center', 'pitch': 0, 'yaw': 0, 'roll': 0},      # Front
        {'name': 'left', 'pitch': 0, 'yaw': -25, 'roll': 0},      # Left
        {'name': 'right', 'pitch': 0, 'yaw': 25, 'roll': 0},      # Right
        {'name': 'up', 'pitch': 15, 'yaw': 0, 'roll': 0},         # Up
        {'name': 'down', 'pitch': -15, 'yaw': 0, 'roll': 0},      # Down
    ]
