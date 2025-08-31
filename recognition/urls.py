from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_view, name='landing'),
    path('enroll/', views.enroll, name='enroll'),
    path('enroll/delete_all/', views.delete_all_enrollments, name='delete_all_enrollments'),
    path('match/', views.matching_view, name='match'),
]

