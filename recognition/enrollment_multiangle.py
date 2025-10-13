import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from .detection import multi_scale_detect, crop_detected_faces
from .normalization import normalize_face
from .facial_extraction import extract_features
from .pose_estimation import calculate_head_pose, is_pose_matched
from .quality_checker import check_face_quality
from .video_capture import VideoCapture
from .enrollment_state import EnrollmentState

# Update the FaceEnrollment class to accept name parameter
class FaceEnrollment:
    def __init__(self, user_id, name, output_dir='media/enrollments'):
        """
        Initialize the face enrollment system.
        
        Args:
            user_id: Unique identifier for the user
            name: User's name
            output_dir: Directory to store enrollment data
        """
        self.user_id = user_id
        self.name = name  # Store the name
        self.output_dir = output_dir
        self.enrollment = None
        self.camera = None
        self.running = False
        
    def start(self):
        """Start a new enrollment session"""
        if self.running:
            return False
            
        # Initialize enrollment state
        self.enrollment = EnrollmentState(self.user_id, self.name, self.output_dir)
        
        # Initialize camera
        try:
            self.camera = VideoCapture(src=0, width=1280, height=720, fps=30)
            self.camera.start()
            self.running = True
            return True
        except Exception as e:
            print(f"Failed to start camera: {str(e)}")
            self.running = False
            return False
    
    def process_frame(self):
        """
        Process a single frame for enrollment.
        
        Returns:
            dict: Result containing status, message, and additional data
        """
        if not self.running or not self.camera or not self.enrollment:
            return {
                'status': 'error',
                'message': 'Enrollment not started or camera not available'
            }
        
        # Get current target pose
        target_pose = self.enrollment.get_current_pose()
        if not target_pose:
            return {
                'status': 'complete' if self.enrollment.is_complete else 'error',
                'message': 'No more poses needed' if self.enrollment.is_complete else 'Invalid pose index',
                'progress': 1.0
            }
        
        # Read frame from camera
        frame, success = self.camera.read()
        if not success or frame is None:
            return {
                'status': 'error',
                'message': 'Failed to capture frame',
                'progress': self.enrollment.get_progress()
            }
        
        # Convert to RGB (detection expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = multi_scale_detect(frame_rgb)
        if not faces:
            return {
                'status': 'adjust',
                'message': 'No face detected',
                'progress': self.enrollment.get_progress()
            }
        
        # For simplicity, take the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Check face quality
        quality = check_face_quality(frame, face.bbox)
        if not quality['is_valid']:
            return {
                'status': 'adjust',
                'message': ' | '.join(quality['messages'][:2]),  # Show first 2 messages
                'progress': self.enrollment.get_progress(),
                'quality': quality
            }
        
        # Calculate head pose
        landmarks = getattr(face, 'kps', None)
        if landmarks is None:
            return {
                'status': 'error',
                'message': 'No facial landmarks detected',
                'progress': self.enrollment.get_progress()
            }
        
        head_pose = calculate_head_pose(landmarks, frame.shape[:2])
        if not head_pose:
            return {
                'status': 'adjust',
                'message': 'Could not estimate head pose',
                'progress': self.enrollment.get_progress()
            }
        
        # Check if pose matches target
        if not is_pose_matched(head_pose, target_pose):
            return {
                'status': 'adjust',
                'message': f'Turn head {target_pose["name"]}',
                'current_pose': head_pose,
                'target_pose': target_pose,
                'progress': self.enrollment.get_progress()
            }
        
        # Pose matches, capture this frame
        # Crop and normalize face
        cropped_faces, _ = crop_detected_faces([face], frame_rgb)
        if not cropped_faces:
            return {
                'status': 'error',
                'message': 'Could not crop face',
                'progress': self.enrollment.get_progress()
            }
        
        # Normalize face
        normalized_face = normalize_face(cropped_faces[0])
        if normalized_face is None:
            return {
                'status': 'error',
                'message': 'Could not normalize face',
                'progress': self.enrollment.get_progress()
            }
        
        # Extract features
        embedding = extract_features(normalized_face)
        if embedding is None:
            return {
                'status': 'error',
                'message': 'Could not extract features',
                'progress': self.enrollment.get_progress()
            }
        
        # Capture the pose
        is_complete = self.enrollment.capture_pose(
            frame=frame_rgb,
            face_bbox=face.bbox.tolist() if hasattr(face.bbox, 'tolist') else face.bbox,
            landmarks=landmarks,
            embedding=embedding,
            quality_metrics=quality
        )
        
        # Prepare response
        response = {
            'status': 'captured' if not is_complete else 'complete',
            'pose': target_pose['name'],
            'progress': self.enrollment.get_progress(),
            'message': f'Captured {target_pose["name"]} pose',
            'quality': quality
        }
        
        if is_complete:
            response['message'] = 'Enrollment complete!'
            self.stop()
        
        return response
    
    def stop(self):
        """Stop the enrollment session"""
        if self.camera:
            self.camera.stop()
            self.camera = None
        
        self.running = False
        return True

    def get_progress(self):
        """Get current enrollment progress"""
        if not self.enrollment:
            return 0.0
        return self.enrollment.get_progress()

    def get_instruction(self):
        """Get current instruction for the user"""
        if not self.enrollment:
            return "Please start the enrollment"
        return self.enrollment.get_instruction()

    def is_complete(self):
        """Check if enrollment is complete"""
        return self.enrollment.is_complete if self.enrollment else False
