# enrollment.py
import os
import cv2
import numpy as np
import base64
from django.conf import settings
from .models import Person, FaceEmbedding, get_or_create_person, get_all_embeddings
from .detection import load_and_prepare_image, multi_scale_detect, crop_detected_faces
from .normalization import normalize_entire_list
from .facial_extraction import extract_features_from_entire_list
import json

def enroll_face(name, camera_base64=None, upload_files=None, append=False):
    """Enroll up to three images for a person.

    - name: person name
    - camera_base64: optional base64 string from camera capture
    - upload_files: optional list of file paths (temporary uploaded files)

    This function will:
    - accept up to 3 images (camera + up to 2 uploads)
    - run detection/normalization on each, accept pre-cropped small images
    - extract embeddings from normalized faces
    - check for duplicates against existing enrollments (>=0.95)
    - save normalized images into media/faces/<name>/
    - persist embeddings as embeddings.npy in the same folder
    """
    if not name:
        raise ValueError("Name is required")

    images_rgb = []

    temp_dir = os.path.join(settings.BASE_DIR, 'temp_uploads')
    os.makedirs(temp_dir, exist_ok=True)

    # Helper to process a base64 string
    def _add_base64(b64, tag):
        if not b64:
            return
        if 'base64,' in b64:
            b64 = b64.split('base64,')[1]
        data = base64.b64decode(b64)
        temp_path = os.path.join(temp_dir, f"{name}_{tag}_temp.png")
        with open(temp_path, 'wb') as f:
            f.write(data)
        img = load_and_prepare_image(temp_path)
        if img is not None:
            images_rgb.append((img, temp_path))
        else:
            # keep temp for inspection
            images_rgb.append((None, temp_path))

    # Accept camera image
    _add_base64(camera_base64, 'camera')

    # Accept uploaded files paths (list of temp paths created by view)
    if upload_files:
        for idx, up in enumerate(upload_files[:2], start=1):
            if not up:
                continue
            # file path expected to be a temp path already saved by the view
            img = load_and_prepare_image(up)
            if img is not None:
                images_rgb.append((img, up))
            else:
                images_rgb.append((None, up))

    if not images_rgb:
        raise ValueError('No images provided for enrollment')

    normalized_faces_all = []
    temp_paths_to_cleanup = []

    for img_tuple in images_rgb:
        image_rgb, src_path = img_tuple
        if image_rgb is None:
            print(f"⚠️ Failed to load provided image: {src_path}")
            continue

        # Detect
        faces_detected = multi_scale_detect(image_rgb)
        if len(faces_detected) == 0:
            h, w = image_rgb.shape[:2]
            if min(h, w) <= 150:
                print("⚠️ Small/pre-cropped image — accepting as normalized face")
                normalized_faces = [image_rgb]
            else:
                print("⚠️ No face detected in this image — skipping")
                continue
        else:
            face_list, landmarks_list = crop_detected_faces(faces_detected, image_rgb)
            normalized_faces = normalize_entire_list(face_list, landmarks_list)

        if normalized_faces:
            for nf in normalized_faces:
                normalized_faces_all.append(nf)
        temp_paths_to_cleanup.append(src_path)

    if not normalized_faces_all:
        raise ValueError('No valid normalized faces extracted from provided images')

    # Ensure we only keep up to 3 face crops for an enrollment
    if len(normalized_faces_all) > 3:
        print(f"\u26a0\ufe0f More than 3 faces detected/cropped; keeping first 3 (was {len(normalized_faces_all)})")
        normalized_faces_all = normalized_faces_all[:3]

    # Extract features
    features_list = extract_features_from_entire_list(normalized_faces_all)
    if not features_list:
        # Clean up temp files
        for path in temp_paths_to_cleanup:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Warning: Could not remove temp file {path}: {e}")
        raise ValueError('No features could be extracted from the provided images')

    # Get existing embeddings from the database
    existing_embeddings = get_all_embeddings()
    
    # Check for duplicates
    for existing_name, emb_list in existing_embeddings.items():
        for existing_emb in emb_list:
            for new_emb in features_list:
                similarity = np.dot(existing_emb, new_emb) / (
                    np.linalg.norm(existing_emb) * np.linalg.norm(new_emb)
                )
                if similarity >= 0.95:  # High threshold for duplicate detection
                    # Clean up temp files
                    for path in temp_paths_to_cleanup:
                        try:
                            if os.path.exists(path):
                                os.remove(path)
                        except Exception as e:
                            print(f"Warning: Could not remove temp file {path}: {e}")
                    raise ValueError(f'This face is very similar to an existing enrollment for {existing_name} (similarity: {similarity:.4f})')

    # Get or create the person
    person = get_or_create_person(name)
    
    # Save each embedding to the database
    for embedding in features_list:
        FaceEmbedding.create_from_embedding(person, embedding)
    
    print(f"✅ Saved {len(features_list)} embeddings to database for {name}")

    # Clean up temp files
    for path in temp_paths_to_cleanup:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"🗑️ Cleaned up temp file: {path}")
        except Exception as e:
            print(f"⚠️ Could not remove temp file {path}: {e}")

    print(f"✅ Enrollment complete! Saved {len(features_list)} face embedding(s)")
    return {
        'name': name,
        'num_embeddings': len(features_list),
        'status': 'success'
    }