# matching.py
import numpy as np
import cv2
import os
import base64
from django.conf import settings
from .models import get_all_embeddings
from .detection import load_and_prepare_image, load_and_prepare_image_from_bytes, crop_detected_faces, multi_scale_detect
from .normalization import normalize_entire_list

# Use shared detection app instead of separate app for matching
# This avoids duplicate model loading and potential conflicts


# Debug print on module load
print(f"\n{'='*60}")
print(f"🔧 MATCHING MODULE LOADED")
print(f"{'='*60}")
print(f"BASE_DIR: {settings.BASE_DIR}")
print(f"MEDIA_ROOT: {settings.MEDIA_ROOT}")
print(f"{'='*60}\n")


def get_face_embeddings(image_rgb):
    """Extract embeddings for faces in an RGB image using same method as enrollment"""
    try:
        # Use the same detection app as enrollment for consistency
        from .detection import get_face_analysis_app
        print("🔄 Loading InsightFace model for matching...")
        detection_app = get_face_analysis_app()
        
        # Convert RGB to BGR for InsightFace
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Use the same app.get() method as enrollment
        faces = detection_app.get(image_bgr)
        
        if not faces:
            print("❌ No faces detected")
            return []
        
        print(f"✅ Detected {len(faces)} face(s).")
        
        embeddings = []
        for face in faces:
            embedding = face.embedding.astype(np.float32)
            embeddings.append(embedding)
            print(f"✅ MATCHING extracted embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Error in get_face_embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return []



def load_enrolled_embeddings():
    """Load all enrolled face embeddings from the database"""
    print("\n" + "="*60)
    print("📂 LOADING ENROLLED FACES FROM DATABASE")
    print("="*60)
    
    # This now uses the get_all_embeddings function from models.py
    enrolled = get_all_embeddings()
    
    print(f"✅ Loaded {len(enrolled)} person(s) from database")
    for name, embeddings in enrolled.items():
        print(f"   👤 {name}: {len(embeddings)} embedding(s)")
        
    print(f"\n📊 TOTAL ENROLLED: {len(enrolled)} person(s)")
    print("="*60 + "\n")
    
    return enrolled


def match_face(uploaded_photo_input, threshold=0.5):
    """
    Compare uploaded face (base64, file path, or bytes) with enrolled embeddings.
    
    Args:
        uploaded_photo_input: Can be:
            - Base64 string (with "base64," prefix)
            - File path string
            - Bytes object (image data)
    
    Returns:
        tuple: (result_message, known_count, recognized_names_with_confidence)
            - result_message: String summary of the matching results
            - known_count: Number of recognized faces
            - recognized_names_with_confidence: List of dicts with 'name' and 'confidence' keys
    """
    print("\n" + "="*60)
    print("🔍 MATCHING STARTED")
    print("="*60)
    
    try:
        image_rgb = None
        
        # Handle different input types
        if isinstance(uploaded_photo_input, bytes):
            print("📷 Input type: Bytes")
            image_rgb = load_and_prepare_image_from_bytes(uploaded_photo_input)
            
        elif isinstance(uploaded_photo_input, str) and "base64," in uploaded_photo_input:
            print("📷 Input type: Base64")
            base64_data = uploaded_photo_input.split("base64,")[1]
            image_data = base64.b64decode(base64_data)
            image_rgb = load_and_prepare_image_from_bytes(image_data)
            
        elif isinstance(uploaded_photo_input, str):
            print(f"📷 Input type: File ({uploaded_photo_input})")
            image_rgb = load_and_prepare_image(uploaded_photo_input)
            
        else:
            print("❌ Unknown input type")
            return "Unknown input type for image data", 0, 0, [], []
        
        if image_rgb is None:
            print("❌ Failed to load image")
            return "Failed to load uploaded image", 0, 0, [], []
        
        print(f"✅ Image loaded: {image_rgb.shape}")

        # Detect faces (or if the uploaded image is already a small cropped
        # normalized face, extract embedding directly without running the
        # detector which often fails on very small/cropped inputs).
        print("🔍 Detecting faces...")
        h, w = image_rgb.shape[:2]
        if min(h, w) <= 150:
            print(f"⚠️ Small image detected ({w}x{h}) - using same extraction as enrollment")
            uploaded_embeddings = get_face_embeddings(image_rgb)
        else:
            uploaded_embeddings = get_face_embeddings(image_rgb)
        
        if not uploaded_embeddings:
            print("❌ No face detected")
            return "No face detected in uploaded image", 0, 0, [], []
        
        print(f"✅ Found {len(uploaded_embeddings)} face(s)")

        # Save normalized crops to temp_uploads for debugging inspection (if filesystem is writable)
        try:
            temp_dir = os.path.join(settings.BASE_DIR, 'temp_uploads')
            # Check if we can write to filesystem for debugging
            can_write = True
            try:
                os.makedirs(temp_dir, exist_ok=True)
                # Test write access
                test_file = os.path.join(temp_dir, 'test_write.tmp')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except (OSError, PermissionError):
                can_write = False
                print("⚠️ Filesystem is read-only, skipping debug file saving")
            
            if can_write:
                # We need to regenerate normalized faces from the image so we save the
                # actual cropped/normalized images used for embedding extraction.
                # Use the detection + normalization pipeline to get the face crops.
                faces = multi_scale_detect(image_rgb)
                if faces:
                    face_list, landmarks_list = crop_detected_faces(faces, image_rgb)
                    normalized_faces = normalize_entire_list(face_list, landmarks_list)
                    for idx, face in enumerate(normalized_faces, start=1):
                        save_path = os.path.join(temp_dir, f"debug_match_face_{idx}.png")
                        # face is RGB, convert to BGR for cv2
                        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, face_bgr)
                        print(f"💾 Saved normalized crop: {save_path}")
        except Exception as e:
            print(f"⚠️ Could not save debug crops: {e}")
        
        # Load enrolled faces
        print("📂 Loading enrolled faces...")
        enrolled_embeddings = load_enrolled_embeddings()
        
        if not enrolled_embeddings:
            print("❌ No enrolled faces")
            return "No enrolled faces found. Please enroll faces first."
        
        print(f"🔍 Comparing with {len(enrolled_embeddings)} person(s)...")
        
        # Match faces: compute best person per uploaded face
        known_count = 0
        recognized_names_with_confidence = []

        for i, up_emb in enumerate(uploaded_embeddings, start=1):
            print(f"\n🔎 Uploaded face #{i} - matching against enrolled people...")
            best_person = None
            best_person_sim = -1.0

            for name, embeddings_list in enrolled_embeddings.items():
                # compute max similarity for this person across their embeddings
                sims = []
                for emb in embeddings_list:
                    sim = np.dot(emb, up_emb) / (np.linalg.norm(emb) * np.linalg.norm(up_emb))
                    sims.append(sim)
                person_max = float(np.max(sims)) if sims else -1.0
                print(f"   {name}: best similarity = {person_max:.4f}")

                if person_max > best_person_sim:
                    best_person_sim = person_max
                    best_person = name

            if best_person_sim >= threshold:
                known_count += 1
                print(f"\n✅ Face #{i} recognized as {best_person} (confidence: {best_person_sim:.2%})")
                recognized_names_with_confidence.append({
                    'name': best_person,
                    'confidence': round(best_person_sim * 100, 1)  # Convert to percentage
                })
            else:
                print(f"\n⚠️ Face #{i} NOT confidently recognized. Closest: {best_person} ({best_person_sim:.2%})")

        # Summary
        print("\n" + "="*40)
        print(f"Summary: {known_count} face(s) recognized")
        for match_result in recognized_names_with_confidence:
            print(f" - {match_result['name']}: {match_result['confidence']}% confidence")
        print("="*40 + "\n")
        
        # Build result message
        total_faces = len(uploaded_embeddings)
        if known_count == 0:
            result_msg = f"Detected {total_faces} face(s). No faces recognized."
        else:
            result_msg = f"Detected {total_faces} face(s). Recognized {known_count} face(s)."

        print(result_msg)
        print("="*60 + "\n")
        return result_msg, known_count, recognized_names_with_confidence

    except Exception as e:
        # On error, return a tuple with error message and zeroed counts
        print(f"\n🔥 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return f"Error: {str(e)}", 0, []