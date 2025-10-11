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

