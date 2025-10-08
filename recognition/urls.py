from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_view, name='landing'),
    path('enroll/', views.enroll, name='enroll'),
    path('match/', views.matching_view, name='match'),
]

