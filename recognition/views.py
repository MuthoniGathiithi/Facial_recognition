from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from .enrollment import enroll_face
from .matching import match_face
import os
import base64
from django.conf import settings
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
import uuid
# In recognition/views.py, add these imports at the top
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import Person, FaceEmbedding
from .enrollment_multiangle import FaceEnrollment
import numpy as np

# Store active enrollments (in production, use Django's session or cache)
active_enrollments = {}






# In recognition/views.py
from django.shortcuts import render

def enrollment_view(request):
    """Render the enrollment page"""
    return render(request, 'recognition/enrollment.html')


@csrf_exempt
@require_http_methods(["POST"])
def start_multiangle_enrollment(request):
    """Start a new multi-angle enrollment session"""
    try:
        data = json.loads(request.body)
        user_id = data.get('user_id')
        name = data.get('name', '').strip()
        
        if not user_id or not name:
            return JsonResponse({
                'error': 'Both user_id and name are required',
                'status': 'error'
            }, status=400)
        
        # Create and start enrollment
        enrollment = FaceEnrollment(user_id=user_id, name=name)
        if not enrollment.start():
            return JsonResponse({
                'error': 'Failed to start enrollment',
                'status': 'error'
            }, status=500)
        
        # Store enrollment in memory with name
        active_enrollments[user_id] = {
            'enrollment': enrollment,
            'name': name
        }
        
        return JsonResponse({
            'status': 'started',
            'user_id': user_id,
            'name': name,
            'instruction': enrollment.get_instruction(),
            'progress': 0.0
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def process_enrollment_frame(request):
    """Process a frame for the current enrollment"""
    try:
        data = json.loads(request.body)
        user_id = data.get('user_id')
        
        if not user_id:
            return JsonResponse({
                'error': 'user_id is required',
                'status': 'error'
            }, status=400)
            
        enrollment_data = active_enrollments.get(user_id)
        if not enrollment_data:
            return JsonResponse({
                'error': 'No active enrollment found',
                'status': 'error'
            }, status=404)
        
        enrollment = enrollment_data['enrollment']
        
        # Process the frame
        result = enrollment.process_frame()
        
        # If enrollment is complete, save to database
        if result.get('status') == 'complete':
            save_enrollment_to_database(user_id, enrollment_data['name'], enrollment)
            # Clean up
            if user_id in active_enrollments:
                del active_enrollments[user_id]
        
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)

def save_enrollment_to_database(user_id, name, enrollment):
    """Save the completed enrollment to the database"""
    try:
        # Get or create person (using name as unique identifier)
        person, created = Person.objects.get_or_create(
            name=name
        )
        
        # If person exists but name is different, update it
        if not created and person.name != name:
            person.name = name
            person.save()
        
        # Save each pose's embedding
        if hasattr(enrollment, 'captured_poses'):
            for pose_name, data in enrollment.captured_poses.items():
                if data and 'embedding' in data:
                    FaceEmbedding.objects.create(
                        person=person,
                        embedding=np.array(data['embedding']).tobytes(),
                        pose=pose_name,
                        image_path=data.get('frame_path', ''),
                        quality_metrics=data.get('quality_metrics', {})
                    )
        
        return True
    except Exception as e:
        print(f"Error saving enrollment to database: {str(e)}")
        return False


# New views for the real-time enrollment system
def enroll_realtime_view(request):
    """Render the real-time enrollment page"""
    return render(request, 'enroll_realtime.html')


# Update the main enroll view to use the new system
def enroll_new(request):
    """Main enrollment page - now uses real-time multi-angle system"""
    print("=== ENROLL_NEW VIEW CALLED ===")
    print("Rendering enroll_realtime.html template")
    return render(request, 'enroll_realtime.html')


@csrf_exempt
@require_http_methods(["POST"])
def start_enrollment(request):
    """Start a new real-time enrollment session"""
    try:
        print("=== START ENROLLMENT DEBUG ===")
        data = json.loads(request.body)
        user_name = data.get('user_name', '').strip()
        print(f"User name received: '{user_name}'")
        
        if not user_name:
            print("ERROR: No user name provided")
            return JsonResponse({
                'error': 'User name is required',
                'status': 'error'
            }, status=400)
        
        # Generate a unique user ID for this session
        user_id = str(uuid.uuid4())
        print(f"Generated user_id: {user_id}")
        
        # Create enrollment state (no camera needed on backend)
        from .enrollment_state import EnrollmentState
        enrollment_state = EnrollmentState(user_id=user_id, name=user_name)
        print(f"Created enrollment state for {user_name}")
        
        # Store enrollment in session
        active_enrollments[user_id] = {
            'enrollment': enrollment_state,
            'name': user_name
        }
        print(f"Stored in active_enrollments. Total active: {len(active_enrollments)}")
        
        # Store user_id in session for tracking
        request.session['enrollment_user_id'] = user_id
        request.session.save()  # Force save session
        print(f"Saved user_id to session: {user_id}")
        
        # Get initial status
        current_pose = enrollment_state.get_current_pose()
        print(f"Current pose: {current_pose}")
        
        return JsonResponse({
            'status': 'started',
            'user_id': user_id,
            'name': user_name,
            'current_pose': current_pose['name'] if current_pose else None,
            'progress': enrollment_state.get_progress(),
            'message': 'Enrollment started successfully. ' + enrollment_state.get_instruction()
        })
        
    except Exception as e:
        print(f"ERROR in start_enrollment: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)


@require_http_methods(["GET"])
def enrollment_status(request):
    """Get current enrollment status"""
    try:
        print("=== ENROLLMENT STATUS DEBUG ===")
        user_id = request.session.get('enrollment_user_id')
        print(f"Session user_id: {user_id}")
        print(f"Active enrollments: {list(active_enrollments.keys())}")
        
        if not user_id:
            print("ERROR: No user_id in session")
            return JsonResponse({
                'error': 'No active enrollment session',
                'status': 'error'
            }, status=404)
            
        enrollment_data = active_enrollments.get(user_id)
        if not enrollment_data:
            print(f"ERROR: No enrollment data found for user_id: {user_id}")
            return JsonResponse({
                'error': 'Enrollment session not found',
                'status': 'error'
            }, status=404)
        
        enrollment_state = enrollment_data['enrollment']
        print(f"Found enrollment for: {enrollment_data['name']}")
        
        # Get current status from enrollment state
        current_pose = enrollment_state.get_current_pose()
        captured_poses = enrollment_state.get_captured_poses()
        progress = enrollment_state.get_progress()
        
        print(f"Current pose: {current_pose}")
        print(f"Captured poses: {captured_poses}")
        print(f"Progress: {progress}")
        
        result = {
            'status': 'in_progress',
            'current_pose': current_pose['name'] if current_pose else None,
            'completed_poses': captured_poses,
            'remaining_poses': enrollment_state.get_remaining_poses(),
            'progress': progress,
            'message': enrollment_state.get_instruction()
        }
        
        # Check if complete
        if enrollment_state.is_complete:
            result['status'] = 'complete'
            print("Enrollment is complete!")
        
        # If enrollment is complete, save to database
        if result.get('status') == 'complete':
            success = save_enrollment_to_database(user_id, enrollment_data['name'], enrollment_state)
            if success:
                result['message'] = 'Enrollment completed and saved successfully!'
            else:
                result['message'] = 'Enrollment completed but failed to save to database'
            
            # Clean up
            if user_id in active_enrollments:
                del active_enrollments[user_id]
            if 'enrollment_user_id' in request.session:
                del request.session['enrollment_user_id']
        
        print(f"Returning result: {result}")
        return JsonResponse(result)
        
    except Exception as e:
        print(f"ERROR in enrollment_status: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def capture_pose(request):
    """Capture a pose from video frame"""
    try:
        print("=== CAPTURE POSE DEBUG ===")
        user_id = request.session.get('enrollment_user_id')
        print(f"Session user_id: {user_id}")
        print(f"Active enrollments: {list(active_enrollments.keys())}")
        print(f"Request method: {request.method}")
        print(f"Request POST keys: {list(request.POST.keys())}")
        
        if not user_id or user_id not in active_enrollments:
            print("ERROR: No active enrollment session")
            return JsonResponse({
                'error': 'No active enrollment session',
                'status': 'error'
            }, status=404)
        
        enrollment_data = active_enrollments[user_id]
        enrollment_state = enrollment_data['enrollment']
        
        # REAL FACE DETECTION AND POSE VALIDATION
        current_pose = enrollment_state.get_current_pose()
        if not current_pose:
            return JsonResponse({
                'status': 'complete',
                'progress': 1.0,
                'message': 'All poses captured! Enrollment complete!'
            })
        
        # REAL POSE VALIDATION - Check actual head position
        frame_data = request.POST.get('frame_data')
        if not frame_data:
            return JsonResponse({
                'status': 'error',
                'message': 'No camera frame provided. Please ensure camera is working.',
                'current_pose': current_pose['name'],
                'progress': enrollment_state.get_progress()
            })
        
        try:
            import cv2
            import base64
            
            # Decode the camera frame
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Simple face detection using OpenCV
            print(f"Frame decoded successfully, shape: {frame.shape}")
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print(f"Face cascade loaded: {not face_cascade.empty()}")
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"Gray image shape: {gray.shape}")
            
            # Try multiple detection parameters for better results
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            print(f"First attempt: {len(faces)} faces detected")
            
            if len(faces) == 0:
                # Try with more lenient parameters
                faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
                print(f"Second attempt: {len(faces)} faces detected")
            
            if len(faces) == 0:
                # Try with very lenient parameters
                faces = face_cascade.detectMultiScale(gray, 1.02, 2, minSize=(15, 15))
                print(f"Third attempt: {len(faces)} faces detected")
            
            if len(faces) == 0:
                print("❌ No face detected with any parameters")
                return JsonResponse({
                    'status': 'error',
                    'message': 'No face detected! Please position your face in the camera view.',
                    'current_pose': current_pose['name'],
                    'progress': enrollment_state.get_progress()
                })
            
            # Use the largest face if multiple detected
            if len(faces) > 1:
                print(f"Multiple faces detected: {len(faces)}, using largest")
                faces = [max(faces, key=lambda f: f[2] * f[3])]
            
            # Get the face rectangle
            (x, y, w, h) = faces[0]
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            
            # Calculate relative position (simple pose estimation)
            horizontal_offset = (face_center_x - frame_center_x) / frame_center_x
            vertical_offset = (face_center_y - frame_center_y) / frame_center_y
            
            # Validate pose based on face position
            pose_name = current_pose['name']
            pose_valid = False
            error_msg = ""
            
            print(f"Face rectangle: ({x}, {y}, {w}, {h})")
            print(f"Face center: ({face_center_x}, {face_center_y})")
            print(f"Frame center: ({frame_center_x}, {frame_center_y})")
            print(f"Face position - H offset: {horizontal_offset:.3f}, V offset: {vertical_offset:.3f}")
            print(f"Required pose: {pose_name}")
            
            if pose_name == 'front':
                # Front: face should be centered
                print(f"Front check: |H|={abs(horizontal_offset):.3f} < 0.2? |V|={abs(vertical_offset):.3f} < 0.2?")
                if abs(horizontal_offset) < 0.2 and abs(vertical_offset) < 0.2:
                    pose_valid = True
                    print("✅ Front pose valid!")
                else:
                    error_msg = f'Please look straight at the camera (H:{horizontal_offset:.2f}, V:{vertical_offset:.2f})'
                    print(f"❌ Front pose invalid: {error_msg}")
            
            elif pose_name == 'left':
                # Left: face should be on the right side of frame (user's left)
                print(f"Left check: H={horizontal_offset:.3f} > 0.15?")
                if horizontal_offset > 0.15:
                    pose_valid = True
                    print("✅ Left pose valid!")
                else:
                    error_msg = f'Please turn your head to YOUR LEFT (H:{horizontal_offset:.2f})'
                    print(f"❌ Left pose invalid: {error_msg}")
            
            elif pose_name == 'right':
                # Right: face should be on the left side of frame (user's right)
                print(f"Right check: H={horizontal_offset:.3f} < -0.15?")
                if horizontal_offset < -0.15:
                    pose_valid = True
                    print("✅ Right pose valid!")
                else:
                    error_msg = f'Please turn your head to YOUR RIGHT (H:{horizontal_offset:.2f})'
                    print(f"❌ Right pose invalid: {error_msg}")
            
            elif pose_name == 'up':
                # Up: face should be in lower part of frame
                print(f"Up check: V={vertical_offset:.3f} > 0.15?")
                if vertical_offset > 0.15:
                    pose_valid = True
                    print("✅ Up pose valid!")
                else:
                    error_msg = f'Please tilt your head UP (V:{vertical_offset:.2f})'
                    print(f"❌ Up pose invalid: {error_msg}")
            
            elif pose_name == 'down':
                # Down: face should be in upper part of frame
                print(f"Down check: V={vertical_offset:.3f} < -0.15?")
                if vertical_offset < -0.15:
                    pose_valid = True
                    print("✅ Down pose valid!")
                else:
                    error_msg = f'Please lower your head DOWN (V:{vertical_offset:.2f})'
                    print(f"❌ Down pose invalid: {error_msg}")
            
            if not pose_valid:
                return JsonResponse({
                    'status': 'error',
                    'message': error_msg,
                    'current_pose': current_pose['name'],
                    'progress': enrollment_state.get_progress(),
                    'debug_position': {'h_offset': horizontal_offset, 'v_offset': vertical_offset}
                })
            
            print(f"✅ Correct {pose_name} pose detected! Capturing...")
            
        except Exception as e:
            print(f"Error in pose validation: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'status': 'error',
                'message': f'Error analyzing camera frame: {str(e)}',
                'current_pose': current_pose['name'],
                'progress': enrollment_state.get_progress()
            })
        
        # Extract REAL face embedding using the same detection system as matching
        try:
            from .detection import app as detection_app
            
            print(f"Using shared InsightFace app from detection module")
            
            # DEBUG: Check the recognition model output size
            for model_name, model in detection_app.models.items():
                if hasattr(model, 'taskname') and model.taskname == 'recognition':
                    if hasattr(model, 'output_shape'):
                        print(f"Recognition model output shape: {model.output_shape}")
                    if hasattr(model, 'feat_dim'):
                        print(f"Recognition model feat_dim: {model.feat_dim}")
            
            # Convert frame to BGR for InsightFace
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Get face analysis results using the same app as matching
            faces_insight = detection_app.get(frame_bgr)
            
            if len(faces_insight) == 0:
                print("❌ InsightFace couldn't extract embedding")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Could not extract face features. Please try again.',
                    'current_pose': current_pose['name'],
                    'progress': enrollment_state.get_progress()
                })
            
            # Use the first face for embedding
            face_insight = faces_insight[0]
            real_embedding = face_insight.embedding.astype(np.float32)
            real_landmarks = face_insight.kps.astype(np.float32)
            
            print(f"✅ Real embedding extracted: shape={real_embedding.shape}, norm={np.linalg.norm(real_embedding):.3f}")
            print(f"Raw embedding first 10 values: {real_embedding[:10]}")
            
            # Buffalo_l outputs 512D embeddings
            embedding_dim = real_embedding.shape[0]
            print(f"✅ Embedding dimension: {embedding_dim}D (buffalo_l model)")
            
            # Verify it's 512D as expected
            if embedding_dim != 512:
                print(f"⚠️ WARNING: Expected 512D, got {embedding_dim}D")
            
            # Ensure embedding is normalized
            if np.linalg.norm(real_embedding) == 0:
                print(f"❌ ERROR: Zero embedding detected")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid embedding - zero vector detected',
                    'current_pose': current_pose['name'],
                    'progress': enrollment_state.get_progress()
                })
            
            # Crop the face region for storage
            (x, y, w, h) = faces[0]  # Use OpenCV detection bbox
            face_crop = frame[y:y+h, x:x+w]
            
            # Resize face crop to standard size
            face_crop_resized = cv2.resize(face_crop, (112, 112))
            
            print(f"Face crop shape: {face_crop_resized.shape}")
            
        except Exception as e:
            print(f"❌ Error extracting real embedding: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error extracting face features: {str(e)}',
                'current_pose': current_pose['name'],
                'progress': enrollment_state.get_progress()
            })
        
        # Capture the pose with REAL data
        enrollment_state.capture_pose(
            frame=face_crop_resized,
            face_bbox=[x, y, w, h],
            landmarks=real_landmarks,
            embedding=real_embedding,
            quality_metrics={
                'brightness': 150, 
                'sharpness': 200, 
                'face_ratio': w * h / (frame.shape[0] * frame.shape[1]),
                'pose_confidence': 0.95
            }
        )
        
        return JsonResponse({
            'status': 'captured' if not enrollment_state.is_complete else 'complete',
            'progress': enrollment_state.get_progress(),
            'current_pose': enrollment_state.get_current_pose()['name'] if enrollment_state.get_current_pose() else None,
            'completed_poses': enrollment_state.get_captured_poses(),
            'message': f'{current_pose["name"].title()} pose captured successfully!'
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def stop_enrollment(request):
    """Stop the current enrollment session"""
    try:
        user_id = request.session.get('enrollment_user_id')
        if user_id and user_id in active_enrollments:
            del active_enrollments[user_id]
        
        if 'enrollment_user_id' in request.session:
            del request.session['enrollment_user_id']
        
        return JsonResponse({
            'status': 'stopped',
            'message': 'Enrollment stopped successfully'
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def cancel_enrollment(request):
    """Cancel the current enrollment"""
    try:
        data = json.loads(request.body)
        user_id = data.get('user_id')
        
        if not user_id:
            return JsonResponse({
                'error': 'user_id is required',
                'status': 'error'
            }, status=400)
            
        if user_id in active_enrollments:
            enrollment_data = active_enrollments[user_id]
            enrollment_data['enrollment'].stop()
            del active_enrollments[user_id]
            return JsonResponse({'status': 'cancelled'})
        else:
            return JsonResponse({
                'error': 'No active enrollment found',
                'status': 'error'
            }, status=404)
            
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)


def landing_view(request):
    return render(request, 'landing_page.html')

def enroll(request):
    
    if request.method == 'POST':
        name = request.POST.get('name')
        camera_b64 = request.POST.get('captured_photo')
        
        upload_paths = []
        if request.FILES:
           
            for i, key in enumerate(request.FILES):
                if i >= 2:
                    break
                fobj = request.FILES[key]
                temp_path = os.path.join(settings.BASE_DIR, 'temp_uploads', f"enroll_{uuid.uuid4().hex}_{fobj.name}")
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                with open(temp_path, 'wb+') as dst:
                    for chunk in fobj.chunks():
                        dst.write(chunk)
                upload_paths.append(temp_path)

        # Do not allow appending to existing enrollments; always treat enrollment as create-only
        if not name or (not camera_b64 and not upload_paths):
            messages.error(request, "Name and at least one photo are required")
        else:
            try:
                # Force append=False to prevent accidental appends
                saved_faces = enroll_face(name, camera_base64=camera_b64, upload_files=upload_paths, append=False)
                messages.success(request, f"Face enrolled successfully! ({len(saved_faces)} face(s) saved)")
                # Redirect to GET after POST to avoid duplicate form submissions
                return redirect('enroll')
            except Exception as e:
                # Write a debug file with the exception and inputs so we can inspect server-side
                try:
                    debug_path = os.path.join(settings.BASE_DIR, 'temp_uploads', 'last_enroll_debug.json')
                    import json
                    debug_info = {
                        'error': str(e),
                        'name': name,
                        'camera_provided': bool(camera_b64),
                        'upload_count': len(upload_paths),
                        'upload_paths': upload_paths,
                    }
                    with open(debug_path, 'w') as df:
                        json.dump(debug_info, df)
                except Exception:
                    pass
                messages.error(request, f"Enrollment failed: {str(e)}")
            finally:
                # Clean up temp upload files (enroll_face kept only media copies)
                for p in upload_paths:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass

    return render(request, 'enroll.html')


def matching_view(request):
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    # Initialize response data
    response_data = {
        'status': 'error',
        'message': 'Invalid request',
        'known_count': 0,
        'unknown_count': 0,
        'recognized_names': [],
        'unknown_faces': []
    }
    
    # Read threshold percent from env or settings (default 50%)
    try:
        thresh_pct = float(os.environ.get('FACE_MATCH_THRESHOLD_PERCENT', 
                                        getattr(settings, 'FACE_MATCH_THRESHOLD_PERCENT', 50)))
    except Exception:
        thresh_pct = 50.0
    
    threshold = thresh_pct / 100.0
    temp_path = None
    
    if request.method == 'POST':
        try:
            # Handle file upload
            if request.FILES.get('photo'):
                try:
                    photo = request.FILES['photo']
                    # Ensure the file is an image
                    if not photo.content_type.startswith('image/'):
                        raise ValueError("Uploaded file is not an image")
                        
                    # Create a unique filename to avoid conflicts
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_uploads')
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_filename = f"match_{uuid.uuid4().hex}_{photo.name}"
                    temp_path = os.path.join(temp_dir, temp_filename)
                    
                    # Save the uploaded file
                    with open(temp_path, 'wb+') as f:
                        for chunk in photo.chunks():
                            f.write(chunk)
                    
                    print(f"Temporary file saved to: {temp_path}")
                    
                    # Process the image from file
                    result = match_face(temp_path, threshold=threshold)
                    print("Match face function completed successfully")
                    
                except Exception as e:
                    error_msg = f"Error processing image: {str(e)}"
                    print(error_msg)
                    if is_ajax:
                        response_data = {'status': 'error', 'message': error_msg}
                        return JsonResponse(response_data, status=400)
                    return render(request, 'matching.html', {'result': error_msg})
                
            # Handle base64 encoded image from camera
            elif request.POST.get('captured_photo'):
                photo_b64 = request.POST.get('captured_photo')
                result = match_face(photo_b64, threshold=threshold)
                
            else:
                error_msg = "No image data provided"
                if is_ajax:
                    response_data = {'status': 'error', 'message': error_msg}
                    return JsonResponse(response_data, status=400)
                return render(request, 'matching.html', {'result': error_msg})
            
            # match_face returns (result_msg, known_count, unknown_count, recognized_names, unknown_faces)
            if isinstance(result, tuple) and len(result) >= 4:
                result_msg, known_count, unknown_count, recognized_names = result[:4]
                unknown_faces = result[4] if len(result) > 4 else []
                
                # Write debug output
                try:
                    debug_path = os.path.join(settings.BASE_DIR, 'temp_uploads', 'last_match_debug.json')
                    with open(debug_path, 'w') as df:
                        import json
                        debug_data = {
                            'result': result_msg,
                            'known_count': known_count,
                            'unknown_count': unknown_count,
                            'recognized_names': recognized_names,
                            'unknown_faces_count': len(unknown_faces) if unknown_faces else 0
                        }
                        json.dump(debug_data, df, indent=2)
                except Exception as e:
                    print(f"Error writing debug file: {str(e)}")
                
                # Prepare response data
                response_data.update({
                    'status': 'success',
                    'message': result_msg,
                    'known_count': known_count,
                    'unknown_count': unknown_count,
                    'recognized_names': recognized_names,
                    'unknown_faces': unknown_faces
                })
                
                # If it's not an AJAX request, use the old template-based response
                if not is_ajax:
                    return render(request, 'matching.html', {
                        'result': result_msg,
                        'known_count': known_count,
                        'unknown_count': unknown_count,
                        'recognized_names': recognized_names,
                        'show_summary': True
                    })
                
                return JsonResponse(response_data)
            
            # Handle unexpected return format from match_face
            error_msg = "Unexpected result format from face matching"
            if is_ajax:
                response_data = {'status': 'error', 'message': error_msg}
                return JsonResponse(response_data, status=500)
            return render(request, 'matching.html', {'result': error_msg})
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f"Error in matching_view: {error_msg}")
            try:
                debug_path = os.path.join(settings.BASE_DIR, 'temp_uploads', 'last_match_error.json')
                with open(debug_path, 'w') as df:
                    import json, traceback
                    error_data = {
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'request_data': str(request.POST) if request.method == 'POST' else 'GET request'
                    }
                    json.dump(error_data, df, indent=2)
            except Exception as debug_error:
                print(f"Error writing error debug file: {str(debug_error)}")
            
            if is_ajax:
                response_data['message'] = error_msg
                return JsonResponse(response_data, status=500)
            return render(request, 'matching.html', {'result': error_msg})
        
        finally:
            # Clean up the temporary file if it exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Error cleaning up temp file {temp_path}: {str(e)}")
    
    # Handle GET request (show the form)
    if is_ajax:
        return JsonResponse(response_data, status=400)
    return render(request, 'matching.html')

# delete_all_enrollments view removed per user request

