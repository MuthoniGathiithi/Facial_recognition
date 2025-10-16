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
print(f"{'='*60}\n")


def get_face_embeddings(image_rgb):
    """Extract embeddings for faces in an RGB image with speed optimization"""
    try:
        # Resize image for MUCH faster processing if it's too large
        h, w = image_rgb.shape[:2]
        max_size = 512  # Smaller maximum dimension for MUCH faster processing
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"⚡ Aggressively resized image from {w}x{h} to {new_w}x{new_h} for MUCH faster processing")
        
        # Use the same detection app as enrollment for consistency
        from .detection import get_face_analysis_app
        app = get_face_analysis_app()
        
        if app is None:
            print("❌ Failed to get face analysis app")
            return []
        
        # Convert RGB to BGR for InsightFace
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Get faces with embeddings
        faces = app.get(image_bgr)
        
        if not faces:
            print("❌ No faces detected by InsightFace")
            return []
        
        print(f"✅ InsightFace detected {len(faces)} face(s)")
        
        # Limit to first 3 faces for speed (most images have 1-2 faces anyway)
        if len(faces) > 3:
            faces = faces[:3]
            print(f"⚡ Limited to first 3 faces for faster processing")
        
        # Extract embeddings
        embeddings = []
        for i, face in enumerate(faces):
            if hasattr(face, 'embedding') and face.embedding is not None:
                embedding = face.embedding
                # Normalize the embedding
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
                print(f"  Face {i+1}: embedding shape {embedding.shape}")
            else:
                print(f"  Face {i+1}: no embedding available")
        
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

        # Skip debug file operations for faster matching
        print("⚡ Optimized for speed - skipping debug operations")

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