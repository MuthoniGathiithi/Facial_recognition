# enrollment.py
import os
import cv2
import numpy as np
import base64
from django.conf import settings
from .detection import load_and_prepare_image, multi_scale_detect, crop_detected_faces
from .normalization import normalize_entire_list
from .facial_extraction import extract_features_from_entire_list
from .matching import load_enrolled_embeddings

def enroll_face(name, uploaded_photo_base64):
    """
    Enroll a face:
    - name: string
    - uploaded_photo_base64: base64 string from HTML
    """
    if not name or not uploaded_photo_base64:
        raise ValueError("Name and photo are required")
    
    # Remove base64 header if present
    if "base64," in uploaded_photo_base64:
        uploaded_photo_base64 = uploaded_photo_base64.split("base64,")[1]
    
    # Decode base64 to bytes
    image_data = base64.b64decode(uploaded_photo_base64)
    
    # Create temp directory
    temp_dir = os.path.join(settings.BASE_DIR, 'temp_uploads')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{name}_temp.png")
    
    # Save temp image
    with open(temp_path, 'wb') as f:
        f.write(image_data)
    
    print(f"📷 Temp image saved to: {temp_path}")
    
    # Load & preprocess image
    image_rgb = load_and_prepare_image(temp_path)
    if image_rgb is None:
        raise ValueError("Failed to load image")
    
    print("🔍 Detecting faces...")
    # Detect faces
    faces_detected = multi_scale_detect(image_rgb)

    # If no faces detected, but the uploaded image is small (likely already
    # a cropped/normalized face), accept the image directly as a normalized face.
    if len(faces_detected) == 0:
        h, w = image_rgb.shape[:2]
        if min(h, w) <= 150:
            print("⚠️ No detections but image is small — treating as pre-cropped face")
            normalized_faces = [image_rgb]
        else:
            raise ValueError("No face detected in the photo")
    else:
        print(f"✅ Found {len(faces_detected)} face(s)")
        # Crop & normalize faces
        face_list, landmarks_list = crop_detected_faces(faces_detected, image_rgb)
        normalized_faces = normalize_entire_list(face_list, landmarks_list)
    
    if not normalized_faces:
        raise ValueError("Face normalization failed")
    
    print(f"✅ Normalized {len(normalized_faces)} face(s)")
    
    # Create faces directory structure
    faces_dir = os.path.join(settings.MEDIA_ROOT, 'faces')
    os.makedirs(faces_dir, exist_ok=True)
    
    # Save faces
    save_dir = os.path.join(faces_dir, name)
    # Before creating the directory: check if this face (or very similar)
    # is already enrolled under any name. We extract embeddings from the
    # normalized faces and compare them to enrolled embeddings. If a
    # very similar embedding already exists (>= 0.95) we abort enrollment.
    new_features = extract_features_from_entire_list(normalized_faces)
    if new_features:
        enrolled = load_enrolled_embeddings()
        for nf in new_features:
            nf_arr = np.asarray(nf).ravel()
            for person, emb_list in enrolled.items():
                for emb in emb_list:
                    emb_arr = np.asarray(emb).ravel()
                    sim = np.dot(emb_arr, nf_arr) / (np.linalg.norm(emb_arr) * np.linalg.norm(nf_arr))
                    if sim >= 0.95:
                        raise ValueError(f"Face already enrolled as '{person}' (similarity: {sim:.2%})")

    if os.path.exists(save_dir):
        raise ValueError(f"User '{name}' is already enrolled")
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Created directory: {save_dir}")
    
    saved_paths = []
    for i, face in enumerate(normalized_faces):
        save_path = os.path.join(save_dir, f"{name}_{i+1}.jpg")
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, face_bgr)
        saved_paths.append(save_path)
        print(f"💾 Saved: {save_path}")
    
    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
        print(f"🗑️ Cleaned up temp file")
    
    print(f"✅ Enrollment complete! Saved {len(saved_paths)} face(s)")
    return saved_paths