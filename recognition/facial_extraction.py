from insightface.app import FaceAnalysis
import cv2
import numpy as np
from .normalization import normalize_entire_list

# Lazy loading - initialize model only when needed
app = None

def get_face_analysis_app():
    """Lazy load the FaceAnalysis model - loads only on first use"""
    global app
    if app is None:
        print("🔄 Loading InsightFace model for feature extraction (first time only)...")
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace model loaded for feature extraction")
    return app

def extract_features(normalized_face):
    """Extract features from a normalized face using ArcFace"""
    if normalized_face is not None:
        face_app = get_face_analysis_app()
        recognition_model = face_app.models['recognition']
        face_bgr = cv2.cvtColor(normalized_face, cv2.COLOR_RGB2BGR)
        embedding = recognition_model.get_feat(face_bgr)
        # Reshape to (1, 512) for consistency
        return embedding.reshape(1, -1)  # This ensures shape is (1, 512)
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





