from django.shortcuts import render
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
    message = ""
    if request.method == 'POST':
        name = request.POST.get('name')
        camera_b64 = request.POST.get('captured_photo')
        # Accept up to 2 uploaded files
        upload_paths = []
        if request.FILES:
            # Save up to 2 files to temp_uploads and pass their paths
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
            message = "Name and at least one photo are required"
        else:
            try:
                # Force append=False to prevent accidental appends
                saved_faces = enroll_face(name, camera_base64=camera_b64, upload_files=upload_paths, append=False)
                message = f"Face enrolled successfully! ({len(saved_faces)} face(s) saved)"
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
                message = f"Enrollment failed: {str(e)}"
            finally:
                # Clean up temp upload files (enroll_face kept only media copies)
                for p in upload_paths:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
    
    return render(request, 'enroll.html', {'message': message})


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


def delete_all_enrollments(request):
    """Delete all enrolled people (remove folders under MEDIA_ROOT/faces).

    This view only accepts POST to avoid accidental deletions via GET.
    """
    from django.http import HttpResponseRedirect
    from django.urls import reverse
    import shutil
    from django.conf import settings

    message = ''
    if request.method != 'POST':
        message = 'Invalid request method. Use POST to delete enrollments.'
        return render(request, 'enroll.html', {'message': message})

    faces_root = os.path.join(settings.MEDIA_ROOT, 'faces')
    deleted = 0
    try:
        if os.path.exists(faces_root):
            for name in os.listdir(faces_root):
                path = os.path.join(faces_root, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    deleted += 1
        message = f'Deleted {deleted} enrolled person(s)'.strip()
    except Exception as e:
        message = f'Failed to delete enrollments: {e}'

    return render(request, 'enroll.html', {'message': message})

