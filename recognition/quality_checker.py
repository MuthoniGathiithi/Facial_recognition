import cv2
import numpy as np

def check_brightness(image, min_brightness=30, max_brightness=220):
    """
    Check if the image has sufficient brightness.
    
    Args:
        image: Input image (BGR or RGB)
        min_brightness: Minimum average brightness (0-255)
        max_brightness: Maximum average brightness (0-255)
    
    Returns:
        tuple: (is_valid, message)
    """
    if image is None or image.size == 0:
        return False, "Empty image"
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    avg_brightness = np.mean(gray)
    
    if avg_brightness < min_brightness:
        return False, f"Image too dark: {avg_brightness:.1f} < {min_brightness}"
    elif avg_brightness > max_brightness:
        return False, f"Image too bright: {avg_brightness:.1f} > {max_brightness}"
    
    return True, f"Brightness OK: {avg_brightness:.1f}"

def check_sharpness(image, min_sharpness=30):
    """
    Check if the image is sharp enough using Laplacian variance.
    
    Args:
        image: Input image (BGR or RGB)
        min_sharpness: Minimum sharpness threshold
    
    Returns:
        tuple: (is_valid, message)
    """
    if image is None or image.size == 0:
        return False, "Empty image"
        
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    if sharpness < min_sharpness:
        return False, f"Image too blurry: {sharpness:.1f} < {min_sharpness}"
    
    return True, f"Sharpness OK: {sharpness:.1f}"

def check_face_size(face_bbox, image_size, min_face_ratio=0.15):
    """
    Check if the face is large enough in the frame.
    
    Args:
        face_bbox: Tuple of (x1, y1, x2, y2)
        image_size: Tuple of (height, width)
        min_face_ratio: Minimum ratio of face size to image size
    
    Returns:
        tuple: (is_valid, message)
    """
    if not face_bbox or len(face_bbox) != 4:
        return False, "Invalid face bbox"
        
    height, width = image_size
    x1, y1, x2, y2 = face_bbox
    
    face_w = x2 - x1
    face_h = y2 - y1
    
    # Check if face is too small
    if face_w < 40 or face_h < 40:  # Absolute minimum size
        return False, f"Face too small: {face_w}x{face_h}px"
    
    # Check face size relative to image
    face_ratio = max(face_w/width, face_h/height)
    if face_ratio < min_face_ratio:
        return False, f"Face too small in frame: {face_ratio*100:.1f}% < {min_face_ratio*100}%"
    
    return True, f"Face size OK: {face_w}x{face_h}px"

def check_face_position(face_bbox, image_size, margin_ratio=0.15):
    """
    Check if the face is properly centered in the frame.
    
    Args:
        face_bbox: Tuple of (x1, y1, x2, y2)
        image_size: Tuple of (height, width)
        margin_ratio: Allowed margin as ratio of image dimensions
    
    Returns:
        tuple: (is_valid, message)
    """
    if not face_bbox or len(face_bbox) != 4:
        return False, "Invalid face bbox"
        
    height, width = image_size
    x1, y1, x2, y2 = face_bbox
    
    face_cx = (x1 + x2) / 2
    face_cy = (y1 + y2) / 2
    
    center_x = width / 2
    center_y = height / 2
    
    # Calculate allowed margins
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    
    # Check if face is centered
    if (abs(face_cx - center_x) > margin_x or 
        abs(face_cy - center_y) > margin_y):
        return False, "Face not centered in frame"
    
    return True, "Face centered in frame"

def check_occlusion(image, face_bbox, landmarks=None):
    """
    Check for face occlusions including masks, sunglasses, hands, shadows, and objects.
    
    Args:
        image: Input image (BGR or RGB)
        face_bbox: Tuple of (x1, y1, x2, y2)
        landmarks: Optional facial landmarks (68 points)
    
    Returns:
        tuple: (is_valid, message, occlusion_details)
    """
    if image is None or image.size == 0:
        return False, "Empty image", {}
        
    if not face_bbox or len(face_bbox) != 4:
        return False, "Invalid face bbox", {}
    
    x1, y1, x2, y2 = face_bbox
    face_region = image[y1:y2, x1:x2]
    
    if face_region.size == 0:
        return False, "Invalid face region", {}
    
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    
    occlusion_issues = []
    occlusion_details = {
        'mask_detected': False,
        'sunglasses_detected': False,
        'hand_detected': False,
        'shadow_detected': False,
        'hair_occlusion': False,
        'object_detected': False,
        'extreme_pose': False
    }
    
    # 1. Check for masks (lower face coverage)
    lower_face = gray[int(gray.shape[0] * 0.6):, :]
    if lower_face.size > 0:
        # Look for uniform regions that might indicate masks
        lower_std = np.std(lower_face)
        lower_mean = np.mean(lower_face)
        
        # Masks often create uniform regions with low variance
        if lower_std < 15 and (lower_mean < 80 or lower_mean > 180):
            occlusion_issues.append("Please remove anything covering your face")
            occlusion_details['mask_detected'] = True
    
    # 2. Check for sunglasses (upper face coverage)
    upper_face = gray[:int(gray.shape[0] * 0.5), :]
    if upper_face.size > 0:
        # Look for dark regions that might be sunglasses
        dark_pixels = np.sum(upper_face < 50)
        total_pixels = upper_face.size
        dark_ratio = dark_pixels / total_pixels
        
        if dark_ratio > 0.3:  # More than 30% dark pixels in upper face
            # Additional check for horizontal dark bands (typical of sunglasses)
            horizontal_profile = np.mean(upper_face, axis=1)
            if np.min(horizontal_profile) < 40:
                occlusion_issues.append("Please remove anything covering your face")
                occlusion_details['sunglasses_detected'] = True
    
    # 3. Check for shadows and poor lighting
    # Analyze brightness distribution
    brightness_std = np.std(gray)
    brightness_mean = np.mean(gray)
    
    # Check for high contrast regions that might indicate shadows
    if brightness_std > 60:  # High variance indicates uneven lighting
        # Check for very dark regions
        dark_regions = np.sum(gray < 30) / gray.size
        if dark_regions > 0.25:  # More than 25% very dark pixels
            occlusion_issues.append("Please improve lighting - face is too dark")
            occlusion_details['shadow_detected'] = True
    
    # 4. Check for extreme head poses using face aspect ratio
    face_width = x2 - x1
    face_height = y2 - y1
    aspect_ratio = face_width / face_height
    
    # Normal face aspect ratio is around 0.75-0.85
    if aspect_ratio < 0.6 or aspect_ratio > 1.0:
        occlusion_issues.append("Please position your face straight in the camera")
        occlusion_details['extreme_pose'] = True
    
    # 5. Check for hand-like skin tones near face
    # Convert to HSV for better skin detection
    h, s, v = cv2.split(hsv)
    
    # Skin tone detection in HSV
    skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    skin_pixels = np.sum(skin_mask > 0)
    total_pixels = gray.size
    skin_ratio = skin_pixels / total_pixels
    
    # If too much skin detected, might indicate hands on face
    if skin_ratio > 0.8:  # More than 80% skin-like pixels
        # Check for irregular shapes that might be hands
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 3:  # Multiple skin regions might indicate hands
            occlusion_issues.append("Please remove your hands from your face")
            occlusion_details['hand_detected'] = True
    
    # 6. Check for hair covering eyes (upper region analysis)
    eye_region = gray[:int(gray.shape[0] * 0.4), :]
    if eye_region.size > 0:
        # Look for texture patterns that might indicate hair
        # Hair typically has more texture/edges than skin
        edges = cv2.Canny(eye_region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.15:  # High edge density might indicate hair
            # Check if it's covering critical eye areas
            eye_center_region = eye_region[int(eye_region.shape[0] * 0.3):int(eye_region.shape[0] * 0.8), :]
            if np.mean(eye_center_region) < 60:  # Dark region where eyes should be
                occlusion_issues.append("Please move hair away from your eyes")
                occlusion_details['hair_occlusion'] = True
    
    # 7. Check for foreign objects using edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Very high edge density might indicate objects like phones, microphones
    if edge_density > 0.25:
        # Look for geometric shapes that don't belong to faces
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > gray.size * 0.1:  # Large objects
                # Check if it's a geometric shape (not organic like face)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 8:  # Geometric shapes have fewer vertices
                    occlusion_issues.append("Please remove any objects from your face area")
                    occlusion_details['object_detected'] = True
                    break
    
    # Determine overall result
    if occlusion_issues:
        is_valid = False
        message = "Some parts of the face are not detected: " + "; ".join(occlusion_issues)
    else:
        is_valid = True
        message = "No significant occlusions detected"
    
    return is_valid, message, occlusion_details

def check_face_quality(image, face_bbox, min_confidence=0.7):
    """
    Comprehensive face quality check.
    
    Args:
        image: Input image (BGR or RGB)
        face_bbox: Tuple of (x1, y1, x2, y2)
        min_confidence: Minimum detection confidence (if available)
    
    Returns:
        dict: {
            'is_valid': bool,
            'messages': list of strings,
            'brightness': float,
            'sharpness': float,
            'occlusion_details': dict
        }
    """
    results = {
        'is_valid': True,
        'messages': [],
        'brightness': 0,
        'sharpness': 0,
        'occlusion_details': {}
    }
    
    # Check image validity
    if image is None or image.size == 0:
        results['is_valid'] = False
        results['messages'].append("Empty image")
        return results
    
    # Check brightness
    bright_ok, bright_msg = check_brightness(image)
    results['brightness'] = float(bright_msg.split(':')[-1].strip())
    if not bright_ok:
        results['is_valid'] = False
        results['messages'].append(bright_msg)
    
    # Check sharpness
    sharp_ok, sharp_msg = check_sharpness(image)
    results['sharpness'] = float(sharp_msg.split(':')[-1].strip().split()[0])
    if not sharp_ok:
        results['is_valid'] = False
        results['messages'].append(sharp_msg)
    
    # Check face size and position if bbox is provided
    if face_bbox and len(face_bbox) == 4:
        size_ok, size_msg = check_face_size(face_bbox, image.shape[:2])
        if not size_ok:
            results['is_valid'] = False
            results['messages'].append(size_msg)
        
        pos_ok, pos_msg = check_face_position(face_bbox, image.shape[:2])
        if not pos_ok:
            results['messages'].append(pos_msg)  # Warning but not invalid
        
        # Check for occlusions
        occlusion_ok, occlusion_msg, occlusion_details = check_occlusion(image, face_bbox)
        results['occlusion_details'] = occlusion_details
        if not occlusion_ok:
            results['is_valid'] = False
            results['messages'].append(occlusion_msg)
    
    return results