import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path


class EnrollmentState:
    """
    Manages the state of face enrollment process across multiple poses.
    """
    
    # Define required poses for enrollment
    # OPTIMIZED: 4 poses including down view for real-world scenarios
    REQUIRED_POSES = [
        {'name': 'front', 'yaw': 0, 'pitch': 0, 'roll': 0, 'tolerance': 15},
        {'name': 'left', 'yaw': -25, 'pitch': 0, 'roll': 0, 'tolerance': 15},
        {'name': 'right', 'yaw': 25, 'pitch': 0, 'roll': 0, 'tolerance': 15},
        {'name': 'down', 'yaw': 0, 'pitch': 18, 'roll': 0, 'tolerance': 15}
    ]
    
    # Original 5-pose system (commented for reference)
    # REQUIRED_POSES = [
    #     {'name': 'front', 'yaw': 0, 'pitch': 0, 'roll': 0, 'tolerance': 15},
    #     {'name': 'left', 'yaw': -30, 'pitch': 0, 'roll': 0, 'tolerance': 15},
    #     {'name': 'right', 'yaw': 30, 'pitch': 0, 'roll': 0, 'tolerance': 15},
    #     {'name': 'up', 'yaw': 0, 'pitch': -20, 'roll': 0, 'tolerance': 15},
    #     {'name': 'down', 'yaw': 0, 'pitch': 20, 'roll': 0, 'tolerance': 15}
    # ]
    
    def __init__(self, user_id, name, output_dir='media/enrollments'):
        """
        Initialize enrollment state.
        
        Args:
            user_id: Unique identifier for the user
            name: User's name
            output_dir: Directory to store enrollment data (for database-only mode, this is ignored)
        """
        self.user_id = user_id
        self.name = name
        
        # Initialize state
        self.current_pose_index = 0
        self.captured_poses = {}
        self.embeddings = []
        self.is_complete = False
        
        # Pose hold timer to prevent false captures
        self.pose_hold_start_time = None
        self.pose_hold_duration = 0.8  # Require 0.8 seconds of stable pose (very user-friendly)
        self.last_valid_pose = None
        
        # Metadata
        self.start_time = datetime.now()
        self.metadata = {
            'user_id': user_id,
            'name': name,
            'start_time': self.start_time.isoformat(),
            'poses_required': len(self.REQUIRED_POSES),
            'poses_captured': 0
        }
    
    def get_current_pose(self):
        """
        Get the current pose that needs to be captured.
        
        Returns:
            dict: Current pose requirements or None if complete
        """
        if self.current_pose_index >= len(self.REQUIRED_POSES):
            return None
        return self.REQUIRED_POSES[self.current_pose_index]
    
    def check_pose_hold_timer(self, pose_name):
        """
        Check if the current pose has been held long enough for capture.
        
        Args:
            pose_name: Name of the currently detected pose
            
        Returns:
            tuple: (is_ready_for_capture, remaining_time)
        """
        import time
        current_time = time.time()
        
        # If this is a new pose or different from last valid pose
        if pose_name != self.last_valid_pose:
            self.pose_hold_start_time = current_time
            self.last_valid_pose = pose_name
            return False, self.pose_hold_duration
        
        # If we don't have a start time, set it now
        if self.pose_hold_start_time is None:
            self.pose_hold_start_time = current_time
            return False, self.pose_hold_duration
        
        # Calculate how long the pose has been held
        hold_time = current_time - self.pose_hold_start_time
        remaining_time = max(0, self.pose_hold_duration - hold_time)
        
        # Check if pose has been held long enough
        is_ready = hold_time >= self.pose_hold_duration
        
        return is_ready, remaining_time
    
    def reset_pose_timer(self):
        """Reset the pose hold timer (called when pose becomes invalid)"""
        self.pose_hold_start_time = None
        self.last_valid_pose = None
    
    def capture_pose(self, frame, face_bbox, landmarks, embedding, quality_metrics):
        """
        Capture the current pose with associated data.
        
        Args:
            frame: The captured frame (numpy array)
            face_bbox: Bounding box of the detected face
            landmarks: Facial landmarks
            embedding: Face embedding vector
            quality_metrics: Quality assessment results
            
        Returns:
            bool: True if enrollment is complete, False otherwise
        """
        if self.current_pose_index >= len(self.REQUIRED_POSES):
            return True
        
        current_pose = self.REQUIRED_POSES[self.current_pose_index]
        pose_name = current_pose['name']
        
        # Store pose data - KEEP EMBEDDING AS NUMPY ARRAY
        pose_data = {
            'pose_name': pose_name,
            'face_bbox': face_bbox,
            'landmarks': landmarks.tolist() if hasattr(landmarks, 'tolist') else landmarks,
            'embedding': embedding,  # Keep as numpy array - DO NOT convert to list
            'quality_metrics': quality_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"🔍 CAPTURE_POSE DEBUG: Storing {pose_name} embedding shape: {embedding.shape}")
        
        # Store in captured poses
        self.captured_poses[pose_name] = pose_data
        self.embeddings.append(embedding)
        
        # Update metadata
        self.metadata['poses_captured'] = len(self.captured_poses)
        
        # Move to next pose
        self.current_pose_index += 1
        
        # Check if enrollment is complete
        if self.current_pose_index >= len(self.REQUIRED_POSES):
            self.is_complete = True
            self._save_enrollment_data()
            return True
        
        return False
    
    def get_progress(self):
        """
        Get enrollment progress as a float between 0 and 1.
        
        Returns:
            float: Progress percentage (0.0 to 1.0)
        """
        return len(self.captured_poses) / len(self.REQUIRED_POSES)
    
    def get_instruction(self):
        """
        Get current instruction for the user.
        
        Returns:
            str: Instruction message
        """
        if self.is_complete:
            return "Enrollment complete!"
        
        current_pose = self.get_current_pose()
        if current_pose:
            return f"Please turn your head {current_pose['name']}"
        
        return "Enrollment in progress..."
    
    def _save_enrollment_data(self):
        """
        Save enrollment data to database only.
        """
        # Update metadata
        self.metadata['completion_time'] = datetime.now().isoformat()
        self.metadata['total_duration'] = (datetime.now() - self.start_time).total_seconds()
        
        # SAVE TO DATABASE
        try:
            from .models import Person, FaceEmbedding
            
            # Create or get person
            person, created = Person.objects.get_or_create(name=self.name)
            print(f"Person {'created' if created else 'found'}: {person.name}")
            
            # Delete any existing embeddings for this person to avoid duplicates
            if not created:
                old_count = FaceEmbedding.objects.filter(person=person).count()
                FaceEmbedding.objects.filter(person=person).delete()
                print(f"Deleted {old_count} old embeddings for {person.name}")
            
            # Save each pose embedding to database
            for pose_name, pose_data in self.captured_poses.items():
                embedding_array = np.array(pose_data['embedding'])
                
                # Ensure embedding is properly formatted
                if len(embedding_array.shape) > 1:
                    embedding_array = embedding_array.reshape(-1)
                
                # Create FaceEmbedding record
                FaceEmbedding.objects.create(
                    person=person,
                    pose=pose_name,
                    embedding=embedding_array.tobytes(),
                    image_path='',  # No file storage for Render deployment
                    quality_metrics=pose_data.get('quality_metrics', {})
                )
            
            print(f"✅ All {len(self.captured_poses)} embeddings saved to database for {self.name}")
            return True
                    
        except Exception as e:
            print(f"❌ Error saving enrollment: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_captured_poses(self):
        """
        Get list of captured pose names.
        
        Returns:
            list: Names of captured poses
        """
        return list(self.captured_poses.keys())
    
    def get_remaining_poses(self):
        """
        Get list of remaining pose names.
        
        Returns:
            list: Names of poses still needed
        """
        captured = set(self.captured_poses.keys())
        required = set(pose['name'] for pose in self.REQUIRED_POSES)
        return list(required - captured)