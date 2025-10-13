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
            output_dir: Directory to store enrollment data
        """
        self.user_id = user_id
        self.name = name
        self.output_dir = Path(output_dir)
        self.user_dir = self.output_dir / str(user_id)
        
        # Create directories if they don't exist
        self.user_dir.mkdir(parents=True, exist_ok=True)
        
        # Enrollment state
        self.current_pose_index = 0
        self.captured_poses = {}
        self.embeddings = []
        self.is_complete = False
        
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
        
        # Save frame image
        frame_filename = f"{pose_name}_frame.jpg"
        frame_path = self.user_dir / frame_filename
        
        # Convert RGB to BGR for OpenCV saving
        import cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)
        
        pose_data['frame_path'] = str(frame_path)
        
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
        Save enrollment data to files AND database.
        """
        # Save metadata to files
        metadata_path = self.user_dir / 'enrollment_metadata.json'
        self.metadata['completion_time'] = datetime.now().isoformat()
        self.metadata['total_duration'] = (datetime.now() - self.start_time).total_seconds()
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save pose data to files
        poses_path = self.user_dir / 'poses_data.json'
        with open(poses_path, 'w') as f:
            json.dump(self.captured_poses, f, indent=2)
        
        # Save embeddings as numpy array to files
        if self.embeddings:
            embeddings_array = np.array(self.embeddings)
            embeddings_path = self.user_dir / 'embeddings.npy'
            np.save(embeddings_path, embeddings_array)
        
        # SAVE TO DATABASE - THIS WAS MISSING!
        try:
            from .models import Person, FaceEmbedding
            
            # Create or get person (using name as unique identifier)
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
                
                print(f"🔍 DEBUG {pose_name}: Original embedding shape: {embedding_array.shape}")
                print(f"🔍 DEBUG {pose_name}: Original embedding dtype: {embedding_array.dtype}")
                print(f"🔍 DEBUG {pose_name}: First 5 values: {embedding_array[:5]}")
                
                # Ensure embedding is properly formatted
                if len(embedding_array.shape) > 1:
                    embedding_array = embedding_array.reshape(-1)
                    print(f"🔍 DEBUG {pose_name}: Reshaped to: {embedding_array.shape}")
                
                # Convert to bytes and check size
                embedding_bytes = embedding_array.tobytes()
                print(f"🔍 DEBUG {pose_name}: Bytes length: {len(embedding_bytes)} (should be {embedding_array.shape[0] * 4})")
                
                # Create FaceEmbedding record
                face_embedding = FaceEmbedding.objects.create(
                    person=person,
                    pose=pose_name,
                    embedding=embedding_bytes,
                    image_path=pose_data.get('frame_path', ''),
                    quality_metrics=pose_data.get('quality_metrics', {})
                )
                
                # Verify what was actually stored
                stored_embedding = face_embedding.get_embedding()
                print(f"✅ Saved {pose_name} embedding: ID={face_embedding.id}")
                print(f"✅ Stored shape: {stored_embedding.shape} (should match original)")
                
                if stored_embedding.shape[0] != embedding_array.shape[0]:
                    print(f"❌ CORRUPTION DETECTED! Original: {embedding_array.shape[0]}D, Stored: {stored_embedding.shape[0]}D")
            
            print(f"🎉 All {len(self.captured_poses)} embeddings saved to database for {self.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving to database: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"Enrollment data saved for user {self.user_id} ({self.name})")
    
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