from django.shortcuts import render
from .enrollment import enroll_face
from .matching import match_face
import os
import base64
from django.conf import settings
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
import uuid

# Landing Page
def landing_view(request):
    return render(request, 'landing_page.html')

# Enroll View
def enroll(request):
    message = ""
    if request.method == 'POST':
        name = request.POST.get('name')
        photo_data = request.POST.get('captured_photo')
        
        if not name or not photo_data:
            message = "Name and photo are required"
        else:
            try:
                # Pass the base64 string directly to enroll_face
                saved_faces = enroll_face(name, photo_data)
                message = f"Face enrolled successfully! ({len(saved_faces)} face(s) saved)"
            except Exception as e:
                message = f"Enrollment failed: {str(e)}"
    
    return render(request, 'enroll.html', {'message': message})

# Matching View
def matching_view(request):
    result = ""
    known_count = 0
    unknown_count = 0
    recognized_names = []
    show_summary = False
    if request.method == 'POST' and request.FILES.get('photo'):
        photo = request.FILES['photo']
        temp_path = os.path.join(settings.BASE_DIR, 'temp_uploads', photo.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        with open(temp_path, 'wb+') as f:
            for chunk in photo.chunks():
                f.write(chunk)
        
        res = match_face(temp_path)
        # match_face now returns (result_msg, known_count, unknown_count, recognized_names)
        if isinstance(res, tuple) and len(res) == 4:
            result, known_count, unknown_count, recognized_names = res
            show_summary = True
        else:
            result = res
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # Write debug output so we can inspect server-side result even if UI misbehaves
        try:
            debug_path = os.path.join(settings.BASE_DIR, 'temp_uploads', 'last_match_debug.json')
            import json
            with open(debug_path, 'w') as df:
                json.dump({'result': result, 'known_count': known_count, 'unknown_count': unknown_count, 'recognized_names': recognized_names, 'show_summary': show_summary}, df)
        except Exception:
            pass
        
    # Support camera-captured base64 POSTs (field name 'captured_photo')
    elif request.method == 'POST' and request.POST.get('captured_photo'):
        photo_b64 = request.POST.get('captured_photo')
        res = match_face(photo_b64)
        if isinstance(res, tuple) and len(res) == 4:
            result, known_count, unknown_count, recognized_names = res
            show_summary = True
        else:
            result = res
        # Write debug output for base64 path too
        try:
            debug_path = os.path.join(settings.BASE_DIR, 'temp_uploads', 'last_match_debug.json')
            import json
            with open(debug_path, 'w') as df:
                json.dump({'result': result, 'known_count': known_count, 'unknown_count': unknown_count, 'recognized_names': recognized_names, 'show_summary': show_summary}, df)
        except Exception:
            pass
    
    return render(request, 'matching.html', {'result': result, 'known_count': known_count, 'unknown_count': unknown_count, 'recognized_names': recognized_names, 'show_summary': show_summary})

