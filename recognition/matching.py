# matching.py
import os
import cv2
import numpy as np
import base64
from django.conf import settings
from insightface.app import FaceAnalysis
from .detection import load_and_prepare_image, crop_detected_faces, multi_scale_detect
from .normalization import normalize_entire_list

# Initialize once at module level
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# FIXED: Use absolute path construction
MEDIA_DIR = os.path.join(settings.BASE_DIR, 'media', 'faces')

# Debug print on module load
print(f"\n{'='*60}")
print(f"🔧 MATCHING MODULE LOADED")
print(f"{'='*60}")
print(f"BASE_DIR: {settings.BASE_DIR}")
print(f"MEDIA_ROOT: {settings.MEDIA_ROOT}")
print(f"MEDIA_DIR (faces): {MEDIA_DIR}")
print(f"Directory exists: {os.path.exists(MEDIA_DIR)}")
if os.path.exists(MEDIA_DIR):
    items = os.listdir(MEDIA_DIR)
    print(f"Items in directory: {items}")
else:
    print("⚠️ Directory does NOT exist!")
print(f"{'='*60}\n")


def get_face_embeddings(image_rgb):
    """Extract embeddings for faces in an RGB image"""
    faces = multi_scale_detect(image_rgb)
    
    if not faces:
        print("⚠️ No faces detected by InsightFace.")
        return []
    
    print(f"✅ Detected {len(faces)} face(s).")
    
    face_list, landmarks_list = crop_detected_faces(faces, image_rgb)
    normalized_faces = normalize_entire_list(face_list, landmarks_list)
    
    embeddings = []
    for face in normalized_faces:
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        emb_data = app.models['recognition'].get_feat(face_bgr)
        if emb_data is not None:
            embeddings.append(np.asarray(emb_data).ravel())
    
    return embeddings


def load_enrolled_embeddings():
    """Load all enrolled face embeddings from the media/faces directory"""
    print("\n" + "="*60)
    print("📂 LOADING ENROLLED FACES")
    print("="*60)
    
    enrolled = {}
    
    # Ensure directory exists
    if not os.path.exists(MEDIA_DIR):
        print(f"❌ MEDIA_DIR does not exist: {MEDIA_DIR}")
        print("💡 Creating directory...")
        os.makedirs(MEDIA_DIR, exist_ok=True)
        return enrolled
    
    print(f"✅ MEDIA_DIR exists: {MEDIA_DIR}")
    
    # List all items in the directory
    try:
        all_items = os.listdir(MEDIA_DIR)
        print(f"📋 Found {len(all_items)} item(s): {all_items}")
    except Exception as e:
        print(f"❌ Error reading directory: {e}")
        return enrolled
    
    # Get person folders
    person_folders = [f for f in all_items if os.path.isdir(os.path.join(MEDIA_DIR, f))]
    
    if not person_folders:
        print(f"⚠️ No person folders found")
        return enrolled
    
    print(f"👥 Found {len(person_folders)} person(s): {person_folders}")
    
    # Load embeddings for each person
    for person_name in person_folders:
        person_dir = os.path.join(MEDIA_DIR, person_name)
        print(f"\n📂 Loading: {person_name}")
        print(f"   Path: {person_dir}")
        
        embeddings_list = []
        
        try:
            # Prefer loading persisted embeddings if present for speed
            emb_path = os.path.join(person_dir, 'embeddings.npy')
            if os.path.exists(emb_path):
                try:
                    arr = np.load(emb_path)
                    # ensure 2D array
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    for row in arr:
                        embeddings_list.append(np.asarray(row).ravel())
                    print(f"   🧾 Loaded persisted embeddings: {emb_path} ({len(embeddings_list)} items)")
                except Exception as e:
                    print(f"   ⚠️ Failed to load embeddings.npy ({e}), falling back to images")
            # Fallback to reading image files if embeddings not loaded
            if not embeddings_list:
                image_files = [f for f in os.listdir(person_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"   🖼️ Image files: {image_files}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

        # Only run image-based extraction if we didn't already load embeddings.npy
        if not embeddings_list:
            for img_file in image_files:
                img_path = os.path.join(person_dir, img_file)
            print(f"   📷 Processing: {img_file}")
            # Try fast path: images saved during enrollment are already
            # normalized cropped face images. Read them directly and
            # extract embedding without re-running detection which can
            # fail on small/cropped images.
            try:
                # Read with OpenCV (BGR)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f"      \u274c Failed to read file with cv2: {img_path}")
                    # Fallback to load_and_prepare_image + detection
                    image_rgb = load_and_prepare_image(img_path)
                    if image_rgb is None:
                        print(f"      \u274c Fallback also failed to load")
                        continue
                    emb = get_face_embeddings(image_rgb)
                    if emb:
                        embeddings_list.append(np.asarray(emb[0]).ravel())
                        print(f"      \u2705 Embedding (fallback): {np.asarray(emb[0]).ravel().shape}")
                    else:
                        print(f"      \u26a0\ufe0f No face detected in fallback")
                else:
                    # If the image was saved by enrollment it should already
                    # be a normalized face (128x128). The recognition model
                    # expects BGR input for get_feat, so pass directly.
                    try:
                        recognition_model = app.models['recognition']
                        emb_vec = recognition_model.get_feat(img_bgr)
                        if emb_vec is not None:
                            embeddings_list.append(np.asarray(emb_vec).ravel())
                            print(f"      \u2705 Embedding: {np.asarray(emb_vec).ravel().shape}")
                        else:
                            print(f"      \u26a0\ufe0f Recognition returned no embedding")
                    except Exception as e:
                        print(f"      \u274c Error extracting embedding: {e}")
                        # As a last resort, try the detection path
                        image_rgb = load_and_prepare_image(img_path)
                        if image_rgb is None:
                            continue
                        emb = get_face_embeddings(image_rgb)
                        if emb:
                            embeddings_list.append(np.asarray(emb[0]).ravel())
                            print(f"      \u2705 Embedding (fallback2): {np.asarray(emb[0]).ravel().shape}")
                        else:
                            print(f"      \u26a0\ufe0f No face detected in fallback2")
            except Exception as e:
                print(f"      \u274c Unexpected error processing {img_file}: {e}")
        
        if embeddings_list:
            enrolled[person_name] = embeddings_list
            print(f"   ✅ Total: {len(embeddings_list)} embedding(s)")
        else:
            print(f"   ⚠️ No embeddings loaded")
    
    print(f"\n📊 TOTAL ENROLLED: {len(enrolled)} person(s)")
    print("="*60 + "\n")
    
    return enrolled


def match_face(uploaded_photo_base64, threshold=0.55):
    """Compare uploaded face (base64 or image) with enrolled embeddings"""
    print("\n" + "="*60)
    print("🔍 MATCHING STARTED")
    print("="*60)
    
    try:
        # Handle base64 or file path
        if isinstance(uploaded_photo_base64, str) and "base64," in uploaded_photo_base64:
            print("📷 Input type: Base64")
            uploaded_photo_base64 = uploaded_photo_base64.split("base64,")[1]
            image_data = base64.b64decode(uploaded_photo_base64)
            
            temp_dir = os.path.join(settings.BASE_DIR, 'temp_uploads')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "temp_match.png")
            
            with open(temp_path, 'wb') as f:
                f.write(image_data)
            print(f"💾 Temp file: {temp_path}")
        else:
            temp_path = uploaded_photo_base64
            print(f"📷 Input type: File ({temp_path})")
        
        # Load image
        print("📂 Loading image...")
        image_rgb = load_and_prepare_image(temp_path)
        
        if image_rgb is None:
            print("❌ Failed to load image")
            return "Failed to load uploaded image"
        
        print(f"✅ Image loaded: {image_rgb.shape}")

        # Detect faces (or if the uploaded image is already a small cropped
        # normalized face, extract embedding directly without running the
        # detector which often fails on very small/cropped inputs).
        print("🔍 Detecting faces...")
        h, w = image_rgb.shape[:2]
        if min(h, w) <= 150:
            print(f"⚠️ Small image detected ({w}x{h}) - using direct embedding extraction")
            try:
                face_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                recognition_model = app.models['recognition']
                emb = recognition_model.get_feat(face_bgr)
                if emb is not None:
                    uploaded_embeddings = [np.asarray(emb).ravel()]
                    print(f"      ✅ Extracted embedding directly: {np.asarray(emb).ravel().shape}")
                else:
                    print("      ⚠️ Direct extraction returned no embedding")
                    uploaded_embeddings = []
            except Exception as e:
                print(f"      ❌ Direct extraction error: {e}")
                uploaded_embeddings = get_face_embeddings(image_rgb)
        else:
            uploaded_embeddings = get_face_embeddings(image_rgb)
        
        if not uploaded_embeddings:
            print("❌ No face detected")
            return "No face detected in uploaded image"
        
        print(f"✅ Found {len(uploaded_embeddings)} face(s)")

        # Save normalized crops to temp_uploads for debugging inspection
        try:
            temp_dir = os.path.join(settings.BASE_DIR, 'temp_uploads')
            os.makedirs(temp_dir, exist_ok=True)
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

        for i, up_emb in enumerate(uploaded_embeddings, start=1):
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

        # Summary
        print("\n" + "="*40)
        print(f"Summary: known_faces={known_count}, unknown_faces={unknown_count}")
        for r in per_face_results:
            if r['matched']:
                print(f" - Face #{r['index']}: KNOWN -> {r['name']} ({r['similarity']:.2%})")
            else:
                print(f" - Face #{r['index']}: UNKNOWN -> Closest {r['name']} ({r['similarity']:.2%})")
        print("="*40 + "\n")

        # Build a concise return message
        if known_count > 0 and unknown_count == 0:
            # All faces matched
            matched_list = [f"{r['name']} ({r['similarity']:.2%})" for r in per_face_results if r['matched']]
            result = "; ".join([f"Face recognized: {m}" for m in matched_list])
        elif known_count > 0:
            matched_list = [f"{r['name']} ({r['similarity']:.2%})" for r in per_face_results if r['matched']]
            result = f"Mixed results - known: {', '.join(matched_list)}; unknown_count: {unknown_count}"
        else:
            # Do NOT reveal enrolled names when there are no confident matches.
            # Report that no confident matches were found and include the best
            # similarity (without the person's name) so user sees why it failed.
            if per_face_results:
                # find the best similarity across all faces
                best_sim = max((r['similarity'] for r in per_face_results), default=0.0)
                result = f"No confident matches. Best similarity: {best_sim:.2%}"
            else:
                result = "No match found"
        print(result)
        print("="*60 + "\n")
        # Also return structured summary for UI
        recognized_names = [f"{r['name']} ({r['similarity']:.2%})" for r in per_face_results if r['matched']]
        return result, known_count, unknown_count, recognized_names

    except Exception as e:
        # On error, return a tuple with error message and zeroed counts
        print(f"\n🔥 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return f"Error: {str(e)}", 0, 0, []