from insightface.app import FaceAnalysis
import cv2
import numpy as np
from normalization import normalize_entire_list

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_features(normalized_face):
    """Extract features from a normalized face using ArcFace"""
    if normalized_face is not None:
        
       recognition_model=app.models['recognition']
       
       face_bgr=cv2.cvtColor(normalized_face,cv2.COLOR_RGB2BGR)
        
       embedding = recognition_model.get_feat(face_bgr)
        
       return embedding
             
        
    else:
        print("No normalized face for feature extraction")
        return None    
       
       
       
def extract_features_from_entire_list(normalized_face_list):
    """Extract features from a list of normalized faces"""
    features_list = []
    for i, face in enumerate(normalized_face_list):
        features = extract_features(face)
        if features is not None:
            features_list.append(features)
        else:
            print(f"WARNING: Feature extraction failed for face {i+1}")
    return features_list       





if __name__ == "__main__":
    from detection import load_and_prepare_image, multi_scale_detect, crop_detected_faces
    
    image_path = '/home/muthoni-gathiithi/Downloads/kuwait.jpeg'
    rgb_image = load_and_prepare_image(image_path)
    
    if rgb_image is not None:
        # Detect faces
        face_details = multi_scale_detect(rgb_image)
        face_list, landmarks_list = crop_detected_faces(face_details, rgb_image)
        
        # Normalize faces
        normalized_faces = normalize_entire_list(face_list, landmarks_list)
        
        # Extract features
        embeddings = extract_features_from_entire_list(normalized_faces)
        print(f"Extracted features from {len(embeddings)} faces!")
        print(f"Each embedding has shape: {embeddings[0].shape}")
       
       