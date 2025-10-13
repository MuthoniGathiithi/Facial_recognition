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
            'sharpness': float
        }
    """
    results = {
        'is_valid': True,
        'messages': [],
        'brightness': 0,
        'sharpness': 0
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
    
    return results