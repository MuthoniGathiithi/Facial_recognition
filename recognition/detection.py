import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image

# Lazy loading - initialize model only when needed
app = None

def get_face_analysis_app():
    """Lazy load the FaceAnalysis model - loads only on first use"""
    global app
    if app is None:
        try:
            print("🔄 DEBUGGING: Loading InsightFace model (first time only)...")
            
            # Check if InsightFace can be imported
            try:
                from insightface.app import FaceAnalysis
                print("✅ Successfully imported InsightFace")
            except Exception as e:
                print(f"❌ CRITICAL: Cannot import InsightFace: {e}")
                return None
            
            # Set model directory to writable location for Render deployment
            model_root = os.environ.get('INSIGHTFACE_HOME', '/tmp/insightface')
            print(f"Model download path: {model_root}")
            
            # Ensure the directory exists
            try:
                os.makedirs(model_root, exist_ok=True)
                print(f"✅ Model directory created/verified: {model_root}")
                
                # Check if directory is writable
                test_file = os.path.join(model_root, 'test_write.tmp')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                print(f"✅ Model directory is writable: {model_root}")
            except Exception as e:
                print(f"❌ CRITICAL: Cannot create/write to model directory: {e}")
                return None
            
            # Try to create FaceAnalysis app
            try:
                print("🔄 Creating FaceAnalysis instance...")
                app = FaceAnalysis(name="buffalo_l", root=model_root, providers=['CPUExecutionProvider'])
                print("✅ FaceAnalysis instance created")
            except Exception as e:
                print(f"❌ CRITICAL: Cannot create FaceAnalysis instance: {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # Try to prepare the app
            try:
                print("🔄 Preparing FaceAnalysis app (downloading models if needed)...")
                app.prepare(ctx_id=0, det_size=(640, 640))
                print("✅ FaceAnalysis app prepared successfully")
            except Exception as e:
                print(f"❌ CRITICAL: Cannot prepare FaceAnalysis app: {e}")
                import traceback
                traceback.print_exc()
                return None
            
            print("✅ InsightFace model loaded successfully")
            
            # Debug: Print model information
            print("=== INSIGHTFACE MODEL DEBUG ===")
            for model_name, model in app.models.items():
                print(f"Model: {model_name}")
                if hasattr(model, 'output_size'):
                    print(f"  Output size: {model.output_size}")
                if hasattr(model, 'input_size'):
                    print(f"  Input size: {model.input_size}")
                if hasattr(model, 'taskname'):
                    print(f"  Task: {model.taskname}")
                if hasattr(model, 'output_shape'):
                    print(f"  Output shape: {model.output_shape}")
            print("==========================================")
            
        except Exception as e:
            print(f"❌ CRITICAL: Overall InsightFace loading failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return app

# -------------------- PREPROCESSING -------------------- #
def preprocess_image(image_bgr):
    """Apply preprocessing: CLAHE for lighting + sharpening for blur"""
    # Convert to LAB color space (better for CLAHE on brightness channel)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE on L-channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # Merge channels back
    lab_eq = cv2.merge((l_eq, a, b))
    image_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Sharpening filter (helps with blur)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image_eq, -1, kernel)

    return sharpened

def load_and_prepare_image(image_path):
    """Load image, preprocess, convert to RGB"""
    if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            # Step 1: Preprocess for lighting + blur
            processed = preprocess_image(image)

            # Step 2: Convert to RGB (ArcFace expects RGB later)
            image_RGB = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            return image_RGB
        else:
            print("Error loading image")
            return None
    else:
        print("Invalid image path")
        return None

def load_and_prepare_image_from_bytes(image_bytes):
    """Load image from bytes, preprocess, convert to RGB - for memory-based processing"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is not None:
            # Step 1: Preprocess for lighting + blur
            processed = preprocess_image(image)
            
            # Step 2: Convert to RGB (ArcFace expects RGB later)
            image_RGB = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            return image_RGB
        else:
            print("Error decoding image from bytes")
            return None
    except Exception as e:
        print(f"Error loading image from bytes: {e}")
        return None

# -------------------- DETECTION UTILS -------------------- #
def iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0
    return interArea / unionArea

def remove_duplicates(faces, iou_threshold=0.5):
    """Remove duplicate detections using IoU threshold"""
    unique_faces = []
    for f in faces:
        keep = True
        for uf in unique_faces:
            if iou(f.bbox, uf.bbox) > iou_threshold:
                keep = False
                break
        if keep:
            unique_faces.append(f)
    return unique_faces

def multi_scale_detect(image_rgb, scales=[320, 480, 640, 800, 1024], base_thresh=0.2, min_thresh=0.05):
    """
    Run detection at multiple scales, with adaptive thresholding.
    - Tries multiple scales with decreasing thresholds
    - Falls back to smaller scales and lower thresholds for better detection
    """
    print(f"🔍 Starting face detection on image of size: {image_rgb.shape}")
    all_faces = []
    
    # Try with normal threshold first
    for size in scales:
        try:
            print(f"  Trying scale: {size}x{size} with threshold: {base_thresh}")
            face_app = get_face_analysis_app()
            face_app.prepare(ctx_id=0, det_size=(size, size))
            faces = face_app.get(image_rgb)
            print(f"  Found {len(faces)} faces at scale {size}x{size}")
            
            for f in faces:
                if f.det_score >= base_thresh:
                    all_faces.append(f)
                    print(f"    Added face with score: {f.det_score:.4f}")
            
            # If we found some faces, try with a slightly lower threshold
            if len(all_faces) > 0 and base_thresh > 0.15:
                lower_thresh = max(min_thresh, base_thresh - 0.05)
                print(f"  Trying lower threshold: {lower_thresh} at scale {size}x{size}")
                for f in faces:
                    if base_thresh > f.det_score >= lower_thresh:
                        all_faces.append(f)
                        print(f"    Added face with lower threshold: {f.det_score:.4f}")
            
            # If we have enough faces, break early
            if len(all_faces) >= 1:  # Adjust this based on your needs
                break
                
        except Exception as e:
            print(f"⚠️ Error during detection at scale {size}x{size}: {str(e)}")
    
    # If still no faces found, try with even lower threshold on smaller scales
    if not all_faces:
        print("⚠️ No faces found with normal settings, trying with minimal threshold...")
        tiny_scales = [160, 224, 320, 480]
        for size in tiny_scales:
            try:
                face_app = get_face_analysis_app()
                face_app.prepare(ctx_id=0, det_size=(size, size))
                faces = face_app.get(image_rgb)
                print(f"  Found {len(faces)} faces at tiny scale {size}x{size}")
                
                for f in faces:
                    if f.det_score >= min_thresh:
                        all_faces.append(f)
                        print(f"    Added face with minimal threshold: {f.det_score:.4f}")
                        
                if all_faces:  # If we found any faces, break
                    break
                    
            except Exception as e:
                print(f"⚠️ Error during tiny scale detection: {str(e)}")
    
    # Remove duplicates using a more permissive IOU threshold
    if all_faces:
        all_faces = remove_duplicates(all_faces, iou_threshold=0.3)
        print(f"✅ Final face count after deduplication: {len(all_faces)}")
    else:
        print("❌ No faces detected after all attempts")
        
        # As a last resort, try to use the entire image as a face
        h, w = image_rgb.shape[:2]
        if min(h, w) <= 200:  # If the image is small, it might be a pre-cropped face
            print("⚠️ Using entire image as face (small image detected)")
            from insightface.data import get_image as ins_get_image
            face = type('Face', (), {})()
            face.bbox = np.array([0, 0, w, h])  # x1, y1, x2, y2
            face.kps = np.array([  # Default landmarks (approximate)
                [w*0.3, h*0.3],  # left eye
                [w*0.7, h*0.3],  # right eye
                [w*0.5, h*0.6],  # nose
                [w*0.2, h*0.8],  # left mouth corner
                [w*0.8, h*0.8]   # right mouth corner
            ])
            face.det_score = min_thresh
            all_faces = [face]
    
    return all_faces

def crop_detected_faces(face_details, Image_RGB, det_thresh=0.05):  # Lowered default threshold
    """
    Crop faces and return cropped images + landmarks.
    
    Args:
        face_details: List of face detection results
        Image_RGB: Input image in RGB format
        det_thresh: Detection threshold (default: 0.05, very low to catch more faces)
    """
    print(f"🖼️ Cropping faces with detection threshold: {det_thresh}")
    landmarks_list = []
    face_list = []
    
    if not face_details:
        print("⚠️ No face details provided for cropping")
        return [], []
    
    h, w = Image_RGB.shape[:2]
    
    for i, face in enumerate(face_details):
        try:
            # Skip if confidence is too low
            if hasattr(face, 'det_score') and face.det_score < det_thresh:
                print(f"  Face {i}: Skipped - Low confidence: {face.det_score:.4f}")
                continue
                
            # Get bounding box coordinates
            if hasattr(face, 'bbox'):
                x1, y1, x2, y2 = face.bbox.astype(int)
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Skip if the face is too small
                face_w, face_h = x2 - x1, y2 - y1
                if face_w < 20 or face_h < 20:  # Minimum face size
                    print(f"  Face {i}: Skipped - Too small: {face_w}x{face_h}")
                    continue
                
                # Extract face region
                cropped_face = Image_RGB[y1:y2, x1:x2]
                
                # Get landmarks if available
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.tolist()
                    # Adjust landmarks to be relative to the cropped face
                    for point in landmarks:
                        point[0] -= x1
                        point[1] -= y1
                else:
                    # Generate default landmarks if not available
                    center_x, center_y = face_w//2, face_h//2
                    landmarks = [
                        [center_x - face_w*0.2, center_y - face_h*0.2],  # left eye
                        [center_x + face_w*0.2, center_y - face_h*0.2],  # right eye
                        [center_x, center_y],                            # nose
                        [center_x - face_w*0.15, center_y + face_h*0.2], # left mouth
                        [center_x + face_w*0.15, center_y + face_h*0.2]  # right mouth
                    ]
                
                face_list.append(cropped_face)
                landmarks_list.append(landmarks)
                print(f"  Face {i}: Added - Confidence: {getattr(face, 'det_score', 1.0):.4f}, "
                      f"Size: {face_w}x{face_h}, Position: ({x1},{y1})-({x2},{y2})")
                
        except Exception as e:
            print(f"⚠️ Error processing face {i}: {str(e)}")
            continue
    
    print(f"✅ Successfully cropped {len(face_list)} faces")
    return face_list, landmarks_list


# -------------------- VIDEO FRAME DETECTION -------------------- #
def detect_face_in_frame(frame_rgb, confidence_threshold=0.3):
    """
    Optimized face detection for video frames.
    
    Args:
        frame_rgb: Video frame in RGB format
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        list: Detected faces with landmarks and confidence scores
    """
    try:
        # Use smaller detection size for speed in video
        face_app = get_face_analysis_app()
        face_app.prepare(ctx_id=0, det_size=(320, 320))
        faces = face_app.get(frame_rgb)
        
        # Filter by confidence and return faces with required attributes
        valid_faces = []
        for face in faces:
            if hasattr(face, 'det_score') and face.det_score >= confidence_threshold:
                valid_faces.append(face)
        
        return valid_faces
    except Exception as e:
        print(f"Error in video frame detection: {e}")
        return []


def get_facial_landmarks(face):
    """
    Extract facial landmarks from detected face.
    
    Args:
        face: Face detection result from InsightFace
        
    Returns:
        numpy.ndarray: 5-point facial landmarks or None if not available
    """
    if hasattr(face, 'kps') and face.kps is not None:
        return face.kps
    return None


def get_detection_confidence(face):
    """
    Get confidence score from face detection.
    
    Args:
        face: Face detection result
        
    Returns:
        float: Confidence score (0.0 to 1.0)
    """
    return getattr(face, 'det_score', 0.0)


def get_largest_face(faces):
    """
    Get the largest face from detection results.
    
    Args:
        faces: List of detected faces
        
    Returns:
        Face object or None if no faces
    """
    if not faces:
        return None
    
    # Calculate face area and return the largest
    largest_face = None
    max_area = 0
    
    for face in faces:
        if hasattr(face, 'bbox'):
            x1, y1, x2, y2 = face.bbox
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_face = face
    
    return largest_face
