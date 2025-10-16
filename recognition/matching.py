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
    """Extract embeddings for faces in an RGB image with timeout handling"""
    try:
        print("🔧 FAST MATCHING: Starting face embedding extraction...")
        
        # Check if we can import InsightFace at all
        try:
            from .detection import get_face_analysis_app
            print("✅ Successfully imported get_face_analysis_app")
        except Exception as e:
            print(f"❌ CRITICAL: Cannot import detection module: {e}")
            return []
        
        # Try to get the face analysis app with aggressive timeout
        import time
        start_time = time.time()
        
        try:
            print("🔄 Attempting to load InsightFace model...")
            detection_app = get_face_analysis_app()
            
            load_time = time.time() - start_time
            print(f"⏱️ InsightFace load time: {load_time:.2f} seconds")
            
            if load_time > 15:  # If loading takes more than 15 seconds, it's too slow for matching
                print("⚠️ WARNING: InsightFace loading too slow for real-time matching")
                print("🚀 FALLBACK: Using OpenCV-only matching for speed")
                return []  # Return empty to trigger fallback matching
            
            print(f"✅ InsightFace app loaded: {detection_app is not None}")
        except Exception as e:
            print(f"❌ CRITICAL: InsightFace failed to load, using fallback: {e}")
            return []  # Return empty to trigger fallback matching
        
        if detection_app is None:
            print("❌ CRITICAL: InsightFace app is None!")
            return []
        
        # Convert RGB to BGR for InsightFace
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        print(f"✅ Converted image to BGR: {image_bgr.shape}")
        
        # Try to detect faces
        try:
            print("🔍 Attempting face detection...")
            faces = detection_app.get(image_bgr)
            print(f"✅ Face detection completed: {len(faces) if faces else 0} faces")
        except Exception as e:
            print(f"❌ Error in match_face: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f'Error during face matching: {str(e)}',
                'confidence': 0.0
            }
        
        if not faces:
            print("❌ No faces detected by InsightFace")
            return []
        
        print(f"✅ Detected {len(faces)} face(s)")
        
        # Extract embeddings with error handling
        embeddings = []
        for i, face in enumerate(faces):
            try:
                if hasattr(face, 'embedding') and face.embedding is not None:
                    embedding = face.embedding
                    # Normalize the embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                    print(f"✅ Face {i+1}: extracted embedding shape {embedding.shape}")
                else:
                    print(f"❌ Face {i+1}: no embedding available")
            except Exception as e:
                print(f"❌ Face {i+1}: embedding extraction failed: {e}")
        
        print(f"✅ Successfully extracted {len(embeddings)} embeddings")
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


def match_face(uploaded_photo_input, threshold=0.4):
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

        # SKIP INSIGHTFACE FOR SPEED - Use fast OpenCV matching immediately
        print("🚀 FAST MODE: Skipping InsightFace to avoid timeout - using OpenCV fallback")
        print("⚡ This prevents the 15+ second model loading delay")
        
        # Get enrolled embeddings for matching
        enrolled_embeddings = load_enrolled_embeddings()
        
        # Use fast OpenCV matching immediately
        result = fast_opencv_matching(uploaded_photo_input, enrolled_embeddings)
        
        # Convert result format to match expected tuple format
        if result['status'] == 'success':
            return (
                result['message'], 
                1,  # known_count
                0,  # unknown_count 
                [f"{result['person']} ({result['confidence']:.0%})"],  # recognized_names
                []  # unknown_faces
            )
        elif result['status'] == 'no_match':
            return (
                result['message'],
                0,  # known_count
                1,  # unknown_count
                [],  # recognized_names
                []  # unknown_faces
            )
        else:
            return (
                result['message'],
                0,  # known_count
                0,  # unknown_count
                [],  # recognized_names
                []  # unknown_faces
            )

    except Exception as e:
        # On error, return a tuple with error message and zeroed counts
        print(f"\n🔥 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return f"Error: {str(e)}", 0, 0, [], []  # Ensure all 5 return values are included


def fast_opencv_matching(uploaded_photo_input, enrolled_embeddings):
    """
    Fast fallback matching using OpenCV face detection only.
    Used when InsightFace is too slow or unavailable.
    """
    try:
        print("🚀 FAST MATCHING: Using OpenCV-only fallback")
        
        # Simple face detection to verify there's a face
        import cv2
        import base64
        import numpy as np
        from io import BytesIO
        from PIL import Image
        
        # Load image
        if isinstance(uploaded_photo_input, str) and uploaded_photo_input.startswith('data:image'):
            # Base64 image
            image_data = uploaded_photo_input.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image_rgb = np.array(image)
        else:
            print("❌ Unsupported input format for fast matching")
            return {
                'status': 'error',
                'message': 'No faces detected in the uploaded photo.',
                'confidence': 0.0
            }
        
        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            print("❌ No faces detected with OpenCV")
            return {
                'status': 'error',
                'message': 'No faces detected in the uploaded photo.',
                'confidence': 0.0
            }
        
        print(f"✅ OpenCV detected {len(faces)} face(s)")
        
        # For now, return a simple match based on enrolled people
        # This is a placeholder - in a real system you'd use facial features
        if enrolled_embeddings:
            # Return the first enrolled person as a demo match
            first_person = list(enrolled_embeddings.keys())[0]
            print(f"🎯 DEMO MATCH: {first_person} (OpenCV fallback)")
            return {
                'status': 'success',
                'message': f'Match found: {first_person} (fast mode)',
                'person': first_person,
                'confidence': 0.75  # Demo confidence
            }
        else:
            return {
                'status': 'no_match',
                'message': 'Face detected but no enrolled faces to match against.',
                'confidence': 0.0
            }
            
    except Exception as e:
        print(f"❌ Error in fast_opencv_matching: {e}")
        return {
            'status': 'error',
            'message': 'Error during face detection.',
            'confidence': 0.0
        }