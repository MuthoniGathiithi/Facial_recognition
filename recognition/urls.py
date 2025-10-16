from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_view, name='landing'),
    path('enroll/', views.enroll_new, name='enroll'),  # Now uses new real-time system
    path('enroll-old/', views.enroll, name='enroll_old'),  # Keep old system as backup
    path('match/', views.matching_view, name='match'),
    
    # Real-time multi-angle enrollment endpoints
    path('enroll-realtime/', views.enroll_realtime_view, name='enroll_realtime'),
    path('start_enrollment/', views.start_enrollment, name='start_enrollment'),
    path('enrollment_status/', views.enrollment_status, name='enrollment_status'),
    path('capture_pose/', views.capture_pose, name='capture_pose'),
    path('stop_enrollment/', views.stop_enrollment, name='stop_enrollment'),
    
    # Health check endpoint
    path('health/', views.health_check, name='health_check'),
]

