"""
Views for handling unknown faces in the face recognition system.
"""
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import ObjectDoesNotExist
import json

from .models import UnknownFace

@csrf_exempt
@require_http_methods(["POST"])
def enroll_unknown_face(request):
    """
    API endpoint to enroll an unknown face with a provided name.
    
    Expected POST data:
    {
        "face_id": "uuid-of-unknown-face",
        "name": "Person's Name"
    }
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        face_id = data.get('face_id')
        name = data.get('name')
        
        if not face_id or not name:
            return JsonResponse(
                {'status': 'error', 'message': 'Both face_id and name are required'}, 
                status=400
            )
        
        # Get the unknown face
        try:
            unknown_face = UnknownFace.objects.get(id=face_id)
        except ObjectDoesNotExist:
            return JsonResponse(
                {'status': 'error', 'message': 'Unknown face ID not found'}, 
                status=404
            )
        
        # Enroll the face as a new person
        person = unknown_face.enroll_as_person(name)
        
        return JsonResponse({
            'status': 'success',
            'message': f'Successfully enrolled as {name}',
            'person_id': person.id,
            'person_name': person.name
        })
        
    except json.JSONDecodeError:
        return JsonResponse(
            {'status': 'error', 'message': 'Invalid JSON data'}, 
            status=400
        )
    except Exception as e:
        return JsonResponse(
            {'status': 'error', 'message': str(e)}, 
            status=500
        )

@require_http_methods(["GET"])
def list_unknown_faces(request):
    """
    API endpoint to list all unknown faces that need to be enrolled.
    """
    try:
        unknown_faces = []
        for face in UnknownFace.objects.all().order_by('-created_at'):
            unknown_faces.append({
                'id': str(face.id),
                'image_url': face.face_image.url if face.face_image else None,
                'created_at': face.created_at.isoformat(),
            })
            
        return JsonResponse({
            'status': 'success',
            'count': len(unknown_faces),
            'faces': unknown_faces
        })
        
    except Exception as e:
        return JsonResponse(
            {'status': 'error', 'message': str(e)}, 
            status=500
        )
