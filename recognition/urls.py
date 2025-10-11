from django.urls import path
from . import views
from . import views_unknown

urlpatterns = [
    path('', views.landing_view, name='landing'),
    path('enroll/', views.enroll, name='enroll'),
    path('match/', views.matching_view, name='match'),
    
    # Unknown face handling endpoints
    path('api/unknown-faces/', views_unknown.list_unknown_faces, name='list_unknown_faces'),
    path('api/enroll-unknown-face/', views_unknown.enroll_unknown_face, name='enroll_unknown_face'),
]

