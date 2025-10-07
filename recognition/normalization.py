import numpy as np
import cv2
from detection import load_and_prepare_image, crop_detected_faces
import math


def normalize_face(cropped_face,landmarks):
    if cropped_face is not None and landmarks is not None:
        left_eye = landmarks[0]   # SCRFD: first point is left eye
        right_eye = landmarks[1]  # SCRFD: second point is right eye
        
        left_eye_x=left_eye[0]
        left_eye_y=left_eye[1]
        
        right_eye_x=right_eye[0]
        right_eye_y=right_eye[1]
        
        
        
        
        
        
        
        
        dy=right_eye_y-left_eye_y
        dx=right_eye_x-left_eye_x
        
        tan_ratio=dy/dx
        
        angle_radians=math.atan(tan_ratio)
        angle_degrees=math.degrees(angle_radians)
        
        rotation_matrix=cv2.getRotationMatrix2D((cropped_face.shape[1]//2,cropped_face.shape[0]//2),angle_degrees,1)
        normalized_face=cv2.warpAffine(cropped_face,rotation_matrix,(cropped_face.shape[1],cropped_face.shape[0]),flags=cv2.INTER_CUBIC)
        
        resized_face=cv2.resize(normalized_face,(128,128))
        
        return resized_face
        
        
        
    else:
        print("No face to normalize")
        return None   
    
    
def normalize_entire_list(face_list,landmarks_list):
    
   
    processed_face = []
    for i, (face, landmarks) in enumerate(zip(face_list, landmarks_list)):
        result = normalize_face(face, landmarks)
        if result is not None:
            processed_face.append(result)
        else:
            print(f"WARNING: Face {i+1} failed normalization")
    return processed_face
 
 
 
 
if __name__ == "__main__":
    from detection import multi_scale_detect  # Add this import
    
    image_path = '/home/muthoni-gathiithi/Downloads/kuwait.jpeg'
    rgb_image = load_and_prepare_image(image_path)
    
    if rgb_image is not None:
    # First detect faces
     face_details = multi_scale_detect(rgb_image)
     print(f"DEBUG: multi_scale_detect found {len(face_details)} faces")
    
    # Then crop them
    face_list, landmarks_list = crop_detected_faces(face_details, rgb_image)
    print(f"DEBUG: crop_detected_faces returned {len(face_list)} faces")
    
    if face_list is not None:
        normalized_faces = normalize_entire_list(face_list, landmarks_list)
        print(f"Normalized {len(normalized_faces)} faces to 128x128!")