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
        tuple: (result_message, known_count, unknown_count, recognized_names, unknown_faces)
            - result_message: String summary of the matching results
            - known_count: Number of recognized faces
            - unknown_count: Number of unknown faces
            - recognized_names: List of names of recognized people
            - unknown_faces: List of tuples (face_image_data, face_id) for unknown faces
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
        
        # Match faces: compute best person per uploaded face and collect stats
        known_count = 0
        unknown_count = 0
        per_face_results = []
        unknown_faces = []
        
        # Get normalized faces for unknown face handling
        faces = multi_scale_detect(image_rgb)
        face_list, landmarks_list = crop_detected_faces(faces, image_rgb)
        normalized_faces = normalize_entire_list(face_list, landmarks_list)
        
        # Ensure we have the same number of faces as embeddings
        if len(normalized_faces) != len(uploaded_embeddings):
            print(f"⚠️ Mismatch between detected faces ({len(normalized_faces)}) and embeddings ({len(uploaded_embeddings)}). Using first {min(len(normalized_faces), len(uploaded_embeddings))} faces.")
            min_faces = min(len(normalized_faces), len(uploaded_embeddings))
            normalized_faces = normalized_faces[:min_faces]
            uploaded_embeddings = uploaded_embeddings[:min_faces]

        for i, (up_emb, face_img) in enumerate(zip(uploaded_embeddings, normalized_faces), start=1):
            print(f"\n🔎 Uploaded face #{i} - matching against enrolled people...")
            best_person = None
            best_person_sim = -1.0
            per_person_sims = []

            for name, embeddings_list in enrolled_embeddings.items():
                # compute max similarity for this person across their embeddings
                sims = []
                for emb in embeddings_list:
                    sim = np.dot(emb, up_emb) / (np.linalg.norm(emb) * np.linalg.norm(up_emb))
                    sims.append(sim)
                person_max = float(np.max(sims)) if sims else -1.0
                per_person_sims.append((name, person_max))
                print(f"   {name}: best similarity = {person_max:.4f}")

                if person_max > best_person_sim:
                    best_person_sim = person_max
                    best_person = name

            matched = False
            if best_person_sim >= threshold:
                matched = True
                known_count += 1
                print(f"\n✅ Face #{i} recognized as {best_person} (confidence: {best_person_sim:.2%})")
                per_face_results.append({'index': i, 'matched': True, 'name': best_person, 'similarity': best_person_sim, 'per_person': per_person_sims})
            else:
                unknown_count += 1
                print(f"\n⚠️ Face #{i} NOT confidently recognized. Closest: {best_person} ({best_person_sim:.2%})")
                per_face_results.append({'index': i, 'matched': False, 'name': best_person, 'similarity': best_person_sim, 'per_person': per_person_sims})
                
                # Save the unknown face
                try:
                    from .models import UnknownFace
                    unknown_face = UnknownFace.create_from_face(up_emb, face_img)
                    # Convert face image to base64 for the frontend
                    _, buffer = cv2.imencode('.png', cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    unknown_faces.append((img_str, str(unknown_face.id)))
                    print(f"💾 Saved unknown face with ID: {unknown_face.id}")
                except Exception as e:
                    print(f"⚠️ Failed to save unknown face: {e}")

        # Summary
        print("\n" + "="*40)
        print(f"Summary: known_faces={known_count}, unknown_faces={unknown_count}")
        for r in per_face_results:
            if r['matched']:
                print(f" - Face #{r['index']}: KNOWN -> {r['name']} ({r['similarity']:.2%})")
            else:
                print(f" - Face #{r['index']}: UNKNOWN -> Closest {r['name']} ({r['similarity']:.2%})")
        print("="*40 + "\n")

        # Build per-face summary: total faces detected, matched faces (>= threshold),
        # and unknown faces = total_faces - known_faces. Only include names for
        # matched faces so low-confidence matches don't leak identities.
        total_faces = len(per_face_results)
        matched_faces = [r for r in per_face_results if r.get('matched')]
        known_count = len(matched_faces)
        unknown_count = total_faces - known_count

        # recognized_names: one entry per matched face (preserve face order)
        recognized_names = []
        for r in per_face_results:
            if r.get('matched'):
                sim_pct = float(r.get('similarity', 0.0)) * 100.0
                sim_str = f"{sim_pct:.2f}%"
                recognized_names.append(f"{r.get('name')} ({sim_str})")

        # Result headline: always show total detected faces. If there are no
        # confident matches, append a short notice (without revealing names).
        result = f"Detected {total_faces} face(s)."
        if known_count == 0 and unknown_count == 0:
            result = result + " No faces found"
        elif known_count == 0:
            result = result + f" Found {unknown_count} unknown face(s). Please provide names for them."
        elif unknown_count > 0:
            result = result + f" Recognized {known_count} face(s) and found {unknown_count} unknown face(s)."
        else:
            result = result + f" Recognized all {known_count} face(s)."

        print(result)
        print("="*60 + "\n")
        return result, known_count, unknown_count, recognized_names, unknown_faces

    except Exception as e:
        # On error, return a tuple with error message and zeroed counts
        print(f"\n🔥 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return f"Error: {str(e)}", 0, 0, [], []  # Ensure all 5 return values are included