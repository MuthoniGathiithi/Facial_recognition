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
import json

def enroll_face(name, camera_base64=None, upload_files=None):
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

    # Extract features
    new_features = extract_features_from_entire_list(normalized_faces_all)
    if not new_features:
        raise ValueError('Failed to extract features from normalized faces')

    # Check for duplicates
    enrolled = load_enrolled_embeddings()
    for nf in new_features:
        nf_arr = np.asarray(nf).ravel()
        for person, emb_list in enrolled.items():
            for emb in emb_list:
                emb_arr = np.asarray(emb).ravel()
                sim = np.dot(emb_arr, nf_arr) / (np.linalg.norm(emb_arr) * np.linalg.norm(nf_arr))
                if sim >= 0.95:
                    raise ValueError(f"Face already enrolled as '{person}' (similarity: {sim:.2%})")

    # Save normalized faces and embeddings
    faces_dir = os.path.join(settings.MEDIA_ROOT, 'faces')
    os.makedirs(faces_dir, exist_ok=True)
    save_dir = os.path.join(faces_dir, name)
    if os.path.exists(save_dir):
        raise ValueError(f"User '{name}' is already enrolled")
    os.makedirs(save_dir, exist_ok=True)

    saved_paths = []
    for i, face in enumerate(normalized_faces_all, start=1):
        save_path = os.path.join(save_dir, f"{name}_{i}.jpg")
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, face_bgr)
        saved_paths.append(save_path)
        print(f"💾 Saved normalized face: {save_path}")

    # Persist embeddings.npy
    emb_array = np.stack([np.asarray(f).ravel() for f in new_features], axis=0)
    emb_path = os.path.join(save_dir, 'embeddings.npy')
    try:
        np.save(emb_path, emb_array)
        print(f"🧾 Saved embeddings: {emb_path} ({emb_array.shape})")
    except Exception as e:
        print(f"⚠️ Failed to save embeddings.npy: {e}")

    # Cleanup temp files created from base64 camera capture (but keep uploaded temp files removed by view)
    for p in temp_paths_to_cleanup:
        try:
            if p and p.startswith(temp_dir) and os.path.exists(p):
                os.remove(p)
                print(f"🗑️ Cleaned up temp file: {p}")
        except Exception:
            pass

    print(f"✅ Enrollment complete! Saved {len(saved_paths)} face(s)")
    return saved_paths